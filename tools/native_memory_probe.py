"""Attribute native layer reload storage separately from transport buffers."""


def start_probe(worker, directory):
    import torch
    from vllm.model_executor.model_loader.reload import layerwise

    model = worker.get_model()
    state = {"materializations": [], "loader_callbacks": []}
    materialize = layerwise.materialize_layer
    load = model.load_weights

    def retained_inputs():
        storages = {}
        for layer in model.modules():
            info = layerwise.get_layerwise_info(layer)
            for _, arguments in info.loaded_weights:
                for value in arguments.arguments.values():
                    if isinstance(value, torch.Tensor) and value.device.type == "npu":
                        storage = value.untyped_storage()
                        storages[storage.data_ptr()] = storage.nbytes()
        return sum(storages.values())

    def measured_materialize(layer, info):
        before = torch.npu.memory_allocated()
        result = materialize(layer, info)
        state["materializations"].append({
            "layer": type(layer).__name__,
            "allocated_delta": torch.npu.memory_allocated() - before,
            "layer_tensor_bytes": sum(t.nbytes for t in layerwise.get_layer_tensors(layer).values()),
            "loader_buffered_input_bytes": retained_inputs(),
        })
        return result

    def measured_load(items):
        result = load(items)
        state["loader_callbacks"].append({"buffered_input_bytes_after_callback": retained_inputs()})
        return result

    layerwise.materialize_layer = measured_materialize
    model.load_weights = measured_load
    worker._memory_probe = (state, materialize, load, retained_inputs)


def stop_probe(worker, directory):
    import json
    from pathlib import Path
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.model_executor.model_loader.reload import layerwise

    state, materialize, load, retained_inputs = worker._memory_probe
    state["loader_buffered_input_bytes_after_finish"] = retained_inputs()
    layerwise.materialize_layer = materialize
    worker.get_model().load_weights = load
    rank = get_tensor_model_parallel_rank()
    (Path(directory) / f"attribution-rank{rank}.json").write_text(json.dumps(state, indent=2) + "\n")
    del worker._memory_probe
    return state
