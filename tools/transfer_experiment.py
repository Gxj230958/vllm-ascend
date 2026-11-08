"""Measure production HCCL packing with a preallocated copy loader."""

import argparse
import contextlib
import gc
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path


class MeasuredGroup:
    def __init__(self, group):
        self.group = group
        self.sizes = []
        self.streams = {}

    def broadcast(self, tensor, **kwargs):
        self.sizes.append(tensor.numel() * tensor.element_size())
        return self.group.broadcast(tensor, **kwargs)


def transfer(rank, mode, weights, metadata, group, loader, capacity):
    from vllm_ascend.distributed.weight_transfer.bounded_tensor import (
        bounded_broadcast_consumer, bounded_broadcast_producer,
    )
    from vllm_ascend.distributed.weight_transfer.packed_tensor import (
        packed_broadcast_consumer, packed_broadcast_producer,
    )

    slots = 1 if mode == "B" else 2
    options = dict(group=group, src=0, buffer_size_bytes=capacity, num_buffers=slots)
    if mode != "A":
        import torch

        if mode not in group.streams:
            group.streams[mode] = tuple(torch.npu.Stream() for _ in range(slots + int(rank != 0)))
        streams = group.streams[mode]
        options["streams"] = streams[:slots]
        if rank:
            options["copy_stream"] = streams[-1]
    if rank == 0:
        if mode == "A":
            packed_broadcast_producer(iter(weights.items()), post_iter_func=lambda item: item[1], **options)
        else:
            bounded_broadcast_producer(iter(weights.items()), iter(metadata), **options)
    elif mode == "A":
        packed_broadcast_consumer(iter(metadata), post_unpack_func=loader, **options)
    else:
        bounded_broadcast_consumer(iter(metadata), post_unpack_func=loader, **options)


def cpu_pattern(shape, dtype, step):
    import math
    import torch

    # Reinterpret special floating-point bit patterns without arithmetic on NaNs.
    bits = torch.tensor([0, -2147483648, 2139095040, -8388608, 2143289345 + step, step + 1, -1], dtype=torch.int32)
    pattern = bits.view(torch.uint8).roll(step)
    size = math.prod(shape) * dtype.itemsize
    return pattern.repeat((size + pattern.numel() - 1) // pattern.numel())[:size].clone().view(dtype).reshape(shape)


def byte_correctness(rank, args, group, dist, device):
    import torch

    records = []
    capacity = args.bucket_mib * 2**20
    for ratio in (0.5, 1, 4, 16):
        metadata = [
            ("prefix", ((7,), torch.uint8)),
            ("large", ((int(capacity * ratio) // 4,), torch.float32)),
            ("scalar", ((), torch.bfloat16)),
            ("empty", ((0, 3), torch.float16)),
            ("tail", ((19,), torch.float64)),
        ]
        for step in range(3):
            reference = {name: cpu_pattern(shape, dtype, step) for name, (shape, dtype) in metadata}
            weights = {name: tensor.to(device) for name, tensor in reference.items()} if rank == 0 else {}
            for mode in args.modes:
                received = {}

                def loader(items):
                    for name, tensor in items:
                        # Retaining the input verifies ownership independently of slot reuse.
                        received[name] = tensor

                torch.npu.synchronize()
                dist.barrier()
                group.sizes.clear()
                transfer(rank, mode, weights, metadata, group, loader, capacity)
                torch.npu.synchronize()
                if rank:
                    assert received.keys() == reference.keys()
                    for name, expected in reference.items():
                        actual = received[name].cpu()
                        assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), (ratio, step, mode, name)
                    # Mutating one owning input must not change another parameter.
                    received["prefix"].zero_()
                    assert torch.equal(received["tail"].cpu().view(torch.uint8), reference["tail"].view(torch.uint8))
                assert mode == "A" or max(group.sizes, default=0) <= capacity
                records.append(dict(mode=mode, ratio=ratio, step=step, collective_bytes=list(group.sizes), correct=True))
                dist.barrier()
                del received
            del weights, reference
    return records


def memory_performance(rank, args, group, dist, device):
    import psutil
    import torch

    capacity = args.bucket_mib * 2**20
    inventory = int(args.inventory_gib * 2**30)
    largest = int(args.largest_mib * 2**20)
    assert 0 < largest <= inventory
    sizes = [largest]
    remaining = inventory - largest
    while remaining:
        size = min(capacity // 2, remaining)
        sizes.append(size)
        remaining -= size
    metadata = [(f"weight_{index}", ((size,), torch.uint8)) for index, size in enumerate(sizes)]
    weights = {name: torch.empty(shape, dtype=dtype, device=device) for name, (shape, dtype) in metadata}
    if args.budget_mib and rank:
        torch.npu.synchronize()
        torch.npu.empty_cache()
        total = torch.npu.get_device_properties(device).total_memory
        allocator_limit = torch.npu.memory_allocated() + args.budget_mib * 2**20
        torch.npu.set_per_process_memory_fraction(allocator_limit / total, device)
    records = []
    start_stability = time.monotonic()
    sample = 0
    while sample < args.warmup + args.samples or time.monotonic() - start_stability < args.min_seconds:
        value = (sample % 251) + 1
        for tensor in weights.values():
            tensor.fill_(value if rank == 0 else 0)
        # Balanced order shares source contents across all variants of a paired sample.
        order = args.modes[sample % len(args.modes):] + args.modes[:sample % len(args.modes)]
        for mode in order:
            if rank:
                for tensor in weights.values():
                    tensor.zero_()
            torch.npu.synchronize()
            gc.collect()
            torch.npu.empty_cache()
            dist.barrier()
            group.sizes.clear()
            before = torch.npu.memory_allocated()
            before_reserved = torch.npu.memory_reserved()
            free_before, total = torch.npu.mem_get_info()
            torch.npu.reset_peak_memory_stats()

            def loader(items):
                for name, tensor in items:
                    weights[name].copy_(tensor)

            dist.barrier()
            started = time.perf_counter()
            try:
                profiler = contextlib.nullcontext()
                if args.profile and sample >= args.warmup:
                    from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler

                    profiler = profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                        record_shapes=True, profile_memory=True,
                        on_trace_ready=tensorboard_trace_handler(str(args.output / f"profiler-rank{rank}-{mode}")),
                    )
                with profiler:
                    transfer(rank, mode, weights, metadata, group, loader, capacity)
            except torch.OutOfMemoryError as error:
                failure = dict(rank=rank, mode=mode, sample=sample, receiver_headroom_mib=args.budget_mib, allocated_before=before, allocated_peak=torch.npu.max_memory_allocated(), error=str(error))
                (args.output / f"allocation-failure-rank{rank}.json").write_text(json.dumps(failure, indent=2) + "\n")
                raise
            torch.npu.synchronize()
            # End-to-end collective/loader completion on every rank.
            dist.barrier()
            elapsed = time.perf_counter() - started
            result = dict(
                sample=sample - args.warmup, mode=mode, seconds=elapsed,
                inventory_bytes=inventory, largest_parameter_bytes=largest,
                bucket_bytes=capacity, slots=1 if mode == "B" else 2,
                allocated_before=before, allocated_after=torch.npu.memory_allocated(),
                allocated_peak=torch.npu.max_memory_allocated(),
                transport_allocated_delta=torch.npu.max_memory_allocated() - before,
                reserved_before=before_reserved, reserved_peak=torch.npu.max_memory_reserved(),
                device_free_before=free_before, device_free_after=torch.npu.mem_get_info()[0],
                device_total=total, cpu_rss=psutil.Process().memory_info().rss,
                cpu_peak_rss_kib=__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss,
                loader_retained_bytes=0, collective_bytes=list(group.sizes),
                receiver_allocator_headroom_mib=args.budget_mib if rank else None,
                transport_storage_bound=(1 if mode == "B" else 2) * capacity + (largest if rank else 0) if mode != "A" else None,
            )
            assert mode == "A" or max(group.sizes, default=0) <= capacity
            if rank:
                # Full byte comparison outside the timed/peak interval, in bounded blocks.
                for tensor in weights.values():
                    for offset in range(0, tensor.numel(), 64 * 2**20):
                        assert bool((tensor[offset:offset + 64 * 2**20] == value).all().item()), (sample, mode, offset)
            result["all_bytes_correct"] = True
            records.append(result)
            with (args.output / f"samples-rank{rank}.jsonl").open("a") as output:
                output.write(json.dumps(result) + "\n")
            print(json.dumps({key: result[key] for key in ("sample", "mode", "seconds", "transport_allocated_delta", "all_bytes_correct")}) + f" rank={rank}", flush=True)
        sample += 1
    return records


def worker(rank, args, port):
    import ctypes
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

    device = args.devices[rank]
    torch.npu.set_device(device)
    physical = ctypes.c_int32()
    acl = ctypes.CDLL("libascendcl.so")
    assert acl.aclrtGetPhyDevIdByLogicDevId(device, ctypes.byref(physical)) == 0
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=len(args.devices), timeout=timedelta(seconds=180))
    native = PyHcclCommunicator(dist.group.WORLD, device=device)
    group = MeasuredGroup(native)
    try:
        if args.task == "bytes":
            records = byte_correctness(rank, args, group, dist, f"npu:{device}")
        else:
            records = memory_performance(rank, args, group, dist, f"npu:{device}")
        (args.output / f"rank{rank}.json").write_text(json.dumps(dict(rank=rank, devices=args.devices, physical_device=physical.value, roce=os.environ.get("HCCL_INTRA_ROCE_ENABLE"), task=args.task, records=records), indent=2) + "\n")
    finally:
        native.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("bytes", "memory"), required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--modes", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    parser.add_argument("--bucket-mib", type=int, default=2)
    parser.add_argument("--inventory-gib", type=float, default=4)
    parser.add_argument("--largest-mib", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--min-seconds", type=int, default=0)
    parser.add_argument("--budget-mib", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "120")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(args, port), nprocs=len(args.devices), join=True)


if __name__ == "__main__":
    main()
