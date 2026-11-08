"""Exercise standard Qwen2 workers, native updates and independent cold loading."""

import argparse
import json
import os
import time
from pathlib import Path


def prepare(directory, large=False):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=262144 if large else 512,
        hidden_size=4096 if large else 128, intermediate_size=8192 if large else 512,
        num_hidden_layers=2, num_attention_heads=32 if large else 4,
        num_key_value_heads=8 if large else 2,
        max_position_embeddings=256, tie_word_embeddings=False,
        bos_token_id=1, eos_token_id=2,
    )
    for step, name in enumerate(("initial", "updated")):
        torch.manual_seed(1701 + step)
        model = Qwen2ForCausalLM(config)
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                parameter.normal_(1.0 if "norm.weight" in key else 0.0, 0.02)
        model.to(torch.bfloat16).save_pretrained(directory / name)


def snapshot(worker, directory, label):
    import torch
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    model = worker.get_model()
    torch.npu.synchronize()
    rank = get_tensor_model_parallel_rank()
    parameters = dict(model.named_parameters())
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in parameters.items()},
        Path(directory) / f"{label}-rank-{rank}.pt",
    )
    return {name: tensor.data_ptr() for name, tensor in parameters.items()}


def memory_sample(worker, reset=False):
    import resource
    import torch

    torch.npu.synchronize()
    if reset:
        torch.npu.reset_peak_memory_stats()
    return {
        "allocated": torch.npu.memory_allocated(),
        "reserved": torch.npu.memory_reserved(),
        "peak_allocated": torch.npu.max_memory_allocated(),
        "peak_reserved": torch.npu.max_memory_reserved(),
        "device_free": torch.npu.mem_get_info()[0],
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_parameter_bytes": sum(p.numel() * p.element_size() for p in worker.get_model().parameters()),
    }


def check_first_weight(worker, expected):
    import torch
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() == 0:
        actual = worker.get_model().model.embed_tokens.weight[0, 0].item()
        assert actual == expected, (actual, expected)
    torch.npu.synchronize()


def set_worker_headroom(worker, headroom_mib):
    import torch

    torch.npu.synchronize()
    torch.npu.empty_cache()
    allocated = torch.npu.memory_allocated()
    total = torch.npu.get_device_properties(torch.npu.current_device()).total_memory
    limit = allocated + headroom_mib * 2**20
    torch.npu.set_per_process_memory_fraction(limit / total)
    return {"allocated_before": allocated, "allocator_limit": limit, "headroom_mib": headroom_mib}


class WorkerClient:
    """Connect the trainer protocol to the LLM's native worker RPCs."""

    def __init__(self, llm):
        self.llm = llm

    def init_weight_transfer_engine(self, init_info):
        self.llm.collective_rpc("init_weight_transfer_engine", timeout=180, args=(init_info,))

    def start_weight_update(self):
        self.llm.collective_rpc("start_weight_update", timeout=180)

    def update_weights(self, update_info):
        self.llm.collective_rpc("update_weights", timeout=180, args=(update_info,))

    def finish_weight_update(self, weight_version=None):
        self.llm.collective_rpc("finish_weight_update", timeout=180)


def infer(llm):
    from vllm import SamplingParams

    prompts = [{"prompt_token_ids": ids} for ids in ([1, 17, 23], [1, 111, 37, 51])]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=5),
        use_tqdm=False,
    )
    return [
        {
            "tokens": list(result.outputs[0].token_ids),
            "logprobs": [
                {str(token): value.logprob for token, value in distribution.items()}
                for distribution in result.outputs[0].logprobs
            ],
        }
        for result in outputs
    ]


def run(args):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["HCCL_CONNECT_TIMEOUT"] = "120"
    os.environ["HCCL_EXEC_TIMEOUT"] = "120"
    from vllm import LLM

    model_directory = args.model_directory or args.directory
    checkpoint = model_directory / ("updated" if args.task == "cold" else "initial")
    graph_options = {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]} if args.graph else {}
    llm = LLM(
        model=str(checkpoint), skip_tokenizer_init=True,
        tensor_parallel_size=args.tp, distributed_executor_backend="mp",
        dtype="bfloat16", enforce_eager=not args.graph, max_model_len=128,
        compilation_config=graph_options,
        max_num_batched_tokens=256, max_num_seqs=4,
        kv_cache_memory_bytes=256 * 2**20, enable_prefix_caching=False,
        weight_transfer_config={"backend": "hccl"},
        additional_config={"weight_nz_mode": 0}, seed=0,
    )
    trainer = None
    update_timings = []
    memory_samples = []
    stability_seconds = 0.0
    try:
        before = llm.collective_rpc(snapshot, args=(str(args.directory), args.task + "-before"))
        if args.headroom_mib:
            limits = llm.collective_rpc(set_worker_headroom, args=(args.headroom_mib,))
            (args.directory / "allocator-limits.json").write_text(json.dumps(limits, indent=2) + "\n")
        if args.task == "update":
            import torch
            import torch_npu  # noqa: F401
            from transformers import AutoModelForCausalLM
            from vllm.distributed.weight_transfer.base import ModuleSource
            from vllm.utils.network_utils import get_open_port
            from vllm_ascend.distributed.weight_transfer.hccl_engine import (
                HCCLTrainerInitInfo,
                HCCLTrainerWeightTransferEngine,
            )

            # Standard platform and Worker initialization has completed above.
            torch.npu.set_device(args.tp)
            source = AutoModelForCausalLM.from_pretrained(
                model_directory / "updated", dtype=torch.bfloat16,
            ).to(f"npu:{args.tp}")
            trainer = HCCLTrainerWeightTransferEngine.trainer_init(
                HCCLTrainerInitInfo(
                    rank=0, master_address="127.0.0.1", master_port=get_open_port(),
                    world_size=args.tp + 1, packed=True,
                    packed_buffer_size_bytes=args.bucket_bytes, packed_num_buffers=args.slots,
                    packed_transfer_mode=args.mode,
                ),
                client=WorkerClient(llm), source=ModuleSource(source),
            )
            first = next(source.parameters()).detach().reshape(-1)[:1]
            original = first.clone()
            if args.attribution:
                from native_memory_probe import start_probe

                llm.collective_rpc(start_probe, args=(str(args.directory),))
            stability_started = time.monotonic()
            for step in range(args.updates):
                if args.min_seconds:
                    target = stability_started + args.min_seconds * step / max(1, args.updates - 1)
                    time.sleep(max(0, target - time.monotonic()))
                first.copy_(original)
                if step != args.updates - 1:
                    first.add_((step % 2 + 1) * 0.125)
                torch.npu.synchronize()
                memory_before = llm.collective_rpc(memory_sample, args=(True,))
                started = time.perf_counter()
                trainer.send_weights()
                update_timings.append(time.perf_counter() - started)
                memory_samples.append({"before": memory_before, "after": llm.collective_rpc(memory_sample)})
                if args.min_seconds:
                    llm.collective_rpc(check_first_weight, args=(first.item(),))
                    if step % 25 == 0 or step == args.updates - 1:
                        progress = {"updates": step + 1, "elapsed_seconds": time.monotonic() - stability_started,
                                    "changed_weight_correct": True, "last_memory": memory_samples[-1]}
                        (args.directory / "stability-progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                        print(json.dumps(progress), flush=True)
                    if step % 100 == 0:
                        infer(llm)
            stability_seconds = time.monotonic() - stability_started
            if args.attribution:
                from native_memory_probe import stop_probe

                llm.collective_rpc(stop_probe, args=(str(args.directory),))
        result = infer(llm)
        after = llm.collective_rpc(snapshot, args=(str(args.directory), args.task))
        record = {
            "task": args.task, "tp": args.tp, "inference": result,
            "mode": args.mode, "slots": args.slots, "bucket_bytes": args.bucket_bytes, "graph": args.graph,
            "complete_update_seconds": update_timings,
            "worker_memory": memory_samples,
            "stability_elapsed_seconds": stability_seconds,
            "parameter_addresses_stable": before == after,
            "parameter_counts": [len(rank) for rank in after],
        }
        (args.directory / f"{args.task}-tp{args.tp}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)
    finally:
        if trainer is not None:
            trainer.shutdown()
        llm.llm_engine.engine_core.shutdown()


def compare(args):
    import torch

    for rank in range(args.tp):
        updated = torch.load(args.directory / f"update-rank-{rank}.pt", weights_only=True)
        cold = torch.load(args.directory / f"cold-rank-{rank}.pt", weights_only=True)
        initial = torch.load(args.directory / f"update-before-rank-{rank}.pt", weights_only=True)
        assert updated.keys() == cold.keys(), "Parameter sets differ"
        assert any(not torch.equal(initial[name], updated[name]) for name in updated), "Weights were not updated"
        for name in updated:
            lhs, rhs = updated[name], cold[name]
            assert lhs.shape == rhs.shape and lhs.dtype == rhs.dtype, name
            assert torch.equal(lhs.reshape(-1).view(torch.uint8), rhs.reshape(-1).view(torch.uint8)), name
    records = [json.loads((args.directory / f"{task}-tp{args.tp}.json").read_text()) for task in ("update", "cold")]
    assert records[0]["inference"] == records[1]["inference"], "Updated and cold-loaded inference differ"
    assert records[0]["parameter_addresses_stable"], "Update changed parameter addresses"
    result = {"tp": args.tp, "all_parameter_bytes": "passed", "cold_inference": "passed"}
    (args.directory / f"comparison-tp{args.tp}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("prepare", "cold", "update", "compare"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--mode", choices=("legacy", "bounded"), default="legacy")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--bucket-bytes", type=int, default=128 * 2**20)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--min-seconds", type=int, default=0)
    parser.add_argument("--attribution", action="store_true")
    parser.add_argument("--headroom-mib", type=int, default=0)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    if args.model_directory is not None:
        args.model_directory = args.model_directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.task == "prepare":
        prepare(args.directory, args.large)
    elif args.task == "compare":
        compare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
