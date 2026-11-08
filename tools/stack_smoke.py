"""Check NPU arithmetic, events and two-rank HCCL on the installed stack."""

import argparse
import importlib.metadata
import json
import os
import socket
from datetime import timedelta
from pathlib import Path


def worker(rank, port, directory):
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    torch.npu.set_device(rank)
    stream = torch.npu.Stream()
    event = torch.npu.Event()
    with torch.npu.stream(stream):
        lhs = torch.arange(4096, device=f"npu:{rank}", dtype=torch.float32)
        output = lhs * 2 + 3
        event.record(stream)
    event.synchronize()
    assert torch.equal(output.cpu(), torch.arange(4096, dtype=torch.float32) * 2 + 3)
    dist.init_process_group(
        "hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
        world_size=2, timeout=timedelta(seconds=120),
    )
    try:
        # Include signed zero, infinities and NaN payloads as byte data.
        bits = torch.tensor([0, -2147483648, 2139095040, -8388608, 2143289345, 1, -1], dtype=torch.int32)
        source = bits.repeat(4096).view(torch.uint8)
        payload = source.to(f"npu:{rank}") if rank == 0 else torch.empty_like(source, device=f"npu:{rank}")
        dist.broadcast(payload, src=0)
        torch.npu.synchronize()
        assert torch.equal(payload.cpu(), source)
        result = {
            "rank": rank, "tensor_and_event": "passed", "hccl_bytes": payload.numel(),
            "hccl_byte_equality": "passed", "allocated": torch.npu.memory_allocated(),
            "reserved": torch.npu.memory_reserved(),
        }
        (directory / f"stack-rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torch_npu", "numpy",
    )}
    (args.output / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "120")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "120")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(port, args.output), nprocs=2, join=True)


if __name__ == "__main__":
    main()
