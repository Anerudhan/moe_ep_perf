"""Compare stock routing with negative IDs for non-local picks."""

import os

import bench_moe


variant = os.environ.get("MOE_ROUTE_VARIANT", "stock")
if variant not in ("stock", "skip_nonlocal"):
    raise ValueError(variant)

original_check = bench_moe.check_output


def check(shape, tensors, weights, output, rank, world):
    if tensors.hidden_states.shape[0] == 1:
        import json

        print(json.dumps({
            "source_rank": rank,
            "ids": tensors.topk_ids.tolist(),
            "weights": tensors.topk_weights.tolist(),
            "routes_to_first_local_expert": (
                tensors.topk_ids % (shape.experts // world) == 0
            ).sum().item(),
            "variant": variant,
        }), flush=True)
    return original_check(shape, tensors, weights, output, rank, world)


bench_moe.check_output = check
original_benchmark = bench_moe.benchmark


def benchmark(args, shape):
    if variant == "skip_nonlocal":
        import torch
        from flashinfer.moe_ep.backends.split.kernel.fused_moe import backend

        original_bridge = backend.build_activation_pack_rank_major

        def bridge(recv, ids, weights, **kwargs):
            pack = original_bridge(recv, ids, weights, **kwargs)
            local = (ids >= 0) & (ids < kwargs["num_local_experts"])
            pack.topk_ids = torch.where(local, pack.topk_ids, -1)
            return pack

        backend.build_activation_pack_rank_major = bridge
    result = original_benchmark(args, shape)
    result["routing_variant"] = variant
    return result


bench_moe.benchmark = benchmark
bench_moe.main()
