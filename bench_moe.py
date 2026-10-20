"""BF16 routed-MoE microbenchmark. Run one case with torchrun."""

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import statistics


FLASHINFER_COMMIT = "35a10560da05517ceb5472ba90c88d53f9129a40"


@dataclass(frozen=True)
class Shape:
    hidden: int
    intermediate: int
    experts: int
    top_k: int
    model_id: str
    revision: str


SHAPES = {
    "qwen3": Shape(4096, 1536, 128, 8, "Qwen/Qwen3-235B-A22B",
                   "8efa61729e24bd65b1d152b5ab5409052aa80e65"),
    "qwen35": Shape(4096, 1024, 512, 10, "Qwen/Qwen3.5-397B-A17B",
                    "8472618112abcbd45acbcdc58436aff4233c23f7"),
    "dsv41": Shape(5120, 2304, 384, 6, "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                   "1bae0e1088993b9b82de08f59de5be5a34e42170"),
}
BACKENDS = ("mega", "nvlink", "nccl_ll", "nccl_ht")
TOKENS = (1, 8, 32, 128, 512, 1024, 2048, 4096, 8192)
# The BF16 finalizer sums routes without applying weights.
MEGA_KNOBS = {"apply_topk_in_fc1": True}


def skip_reason(shape, backend):
    if backend == "nccl_ll":
        if shape.top_k > 8:
            return "NCCL-EP LL requires top_k <= 8; model top-k is preserved"
        if shape.hidden not in (2048, 2560, 4096, 5120, 6144, 7168, 8192):
            return "Hidden width is outside the NCCL-EP LL BF16 whitelist"
    return None


def latency_summary(rank_samples, global_tokens):
    # Each sample is bounded by its slowest rank.
    slowest = [max(sample) for sample in zip(*rank_samples, strict=True)]
    p50 = statistics.median(slowest)
    p95 = sorted(slowest)[math.ceil(0.95 * len(slowest)) - 1]
    return dict(p50_us=p50, p95_us=p95, tokens_per_second=global_tokens * 1e6 / p50)


def build_layer(args, shape, weights, rank, world):
    import torch
    import flashinfer.moe_ep as ep
    from flashinfer.fused_moe.api import (
        BackendOptions, ExecutionConfig, ExpertConfig, MoEConfig, QuantConfig,
        QuantFormat, RoutingConfig, TrtllmBf16Config,
    )

    local_experts = shape.experts // world
    fleet = ep.FleetParams(
        num_experts=shape.experts, max_tokens_per_rank=args.tokens,
        token_hidden_size=shape.hidden, dtype_bytes=2,
        algorithm=(ep.EpAlgorithm.HIGH_THROUGHPUT if args.backend == "nccl_ht"
                   else ep.EpAlgorithm.LOW_LATENCY),
        layout=(ep.EpLayout.EXPERT_MAJOR if args.backend == "nccl_ht"
                else ep.EpLayout.RANK_MAJOR),
    )
    if args.backend == "mega":
        backend = ep.MegaConfig(
            megakernel=ep.Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=shape.intermediate, top_k=shape.top_k,
                knobs=MEGA_KNOBS.copy(),
            ),
            quantize_input=False,
        )
        pack = ep.MoEWeightPack(w13=weights[0], w2=weights[1])
    else:
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=shape.experts, top_k=shape.top_k),
            quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
            experts=ExpertConfig(
                intermediate_size=shape.intermediate,
                local_expert_offset=rank * local_experts,
                local_num_experts=local_experts,
            ),
            backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
            execution=ExecutionConfig(tune_max_num_tokens=args.tokens * world),
        )
        comm = (ep.NVLinkOneSidedConfig(use_low_precision_combine=False)
                if args.backend == "nvlink" else ep.NcclEpConfig())
        backend = ep.SplitConfig(comm=comm, kernel=ep.FusedMoeKernelConfig(moe_config=cfg))
        # Mega accepts gate|up; the TRT-LLM split adapter accepts up|gate.
        gate, up = weights[0].chunk(2, dim=1)
        pack = ep.MoEWeightPack(w13=torch.cat((up, gate), dim=1), w2=weights[1])
    return ep.MoEEpLayer(
        ep.BootstrapConfig(world_size=world, rank=rank,
                           stream=torch.cuda.current_stream().cuda_stream),
        fleet, weights=pack, backend=backend,
    )


def check_output(shape, tensors, weights, output, rank, world):
    """Check sampled source tokens against FP32 local experts plus all-reduce."""
    import torch
    import torch.distributed as dist

    indices = torch.linspace(0, output.shape[0] - 1, min(8, output.shape[0]),
                             device="cuda").long()

    def gather(tensor):
        rows = tensor[indices].contiguous()
        parts = [torch.empty_like(rows) for _ in range(world)]
        dist.all_gather(parts, rows)
        return torch.cat(parts)

    x = gather(tensors.hidden_states).float()
    ids = gather(tensors.topk_ids)
    scales = gather(tensors.topk_weights)
    expected = torch.zeros_like(x)
    local_experts = shape.experts // world
    for expert in range(local_experts):
        rows, slots = (ids == rank * local_experts + expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        gate, up = (x[rows] @ weights[0][expert].float().T).chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate) * up).bfloat16().float()
        value = activated @ weights[1][expert].float().T
        expected.index_add_(0, rows, value * scales[rows, slots, None])
    dist.all_reduce(expected)
    expected = expected.chunk(world)[rank]
    actual = output[indices].float()
    rel_l2 = ((actual - expected).norm() / expected.norm().clamp_min(1e-8))
    valid = torch.isfinite(output).all() & (rel_l2 < 0.03)
    valid &= torch.all((actual - expected).abs() <= 0.03 + 0.03 * expected.abs())
    passed = valid.to(torch.int32)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    dist.all_reduce(rel_l2, op=dist.ReduceOp.MAX)
    if not passed.item():
        raise RuntimeError(f"BF16 reference check failed: max rank relative L2={rel_l2.item()}")
    return rel_l2.item()


def benchmark(args, shape):
    import ctypes
    import torch
    import torch.distributed as dist
    import flashinfer
    from flashinfer import autotune
    from flashinfer.moe_ep import MoEEpTensors

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 8:
        raise RuntimeError("This benchmark requires torchrun with 8 ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This comparison requires SM100 (B200 or GB200)")
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    nccl_version = ctypes.c_int()
    status = ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(nccl_version))
    if status != 0 or (args.backend.startswith("nccl_") and nccl_version.value < 23007):
        raise RuntimeError(f"NCCL-EP requires loaded NCCL >= 23007; got {nccl_version.value}")
    rank = dist.get_rank()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    local_experts = shape.experts // world
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    w13 = torch.randn(local_experts, 2 * shape.intermediate, shape.hidden,
                      device="cuda", dtype=torch.bfloat16, generator=g) / math.sqrt(shape.hidden)
    w2 = torch.randn(local_experts, shape.hidden, shape.intermediate,
                     device="cuda", dtype=torch.bfloat16, generator=g) / math.sqrt(shape.intermediate)
    x = torch.randn(args.tokens, shape.hidden, device="cuda", dtype=torch.bfloat16, generator=g)
    scores = torch.randn(args.tokens, shape.experts, device="cuda", generator=g)
    values, ids = scores.topk(shape.top_k, dim=-1)
    scales = values.softmax(dim=-1)
    tensors = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=scales)
    layer = build_layer(args, shape, (w13, w2), rank, world)
    state = None
    if args.backend != "mega" and args.mode == "graph":
        state = layer.create_graph_state(tensors)

    def forward():
        if args.backend == "mega":
            return layer.forward(tensors, return_workspace_view=True)
        return layer.forward(tensors, graph_state=state)

    with autotune(args.backend != "mega"):
        output = forward()
    for _ in range(args.warmup):
        output = forward()
    torch.cuda.synchronize()
    error = check_output(shape, tensors, (w13, w2), output, rank, world)
    graph = None
    if args.mode == "graph":
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()
        launch = graph.replay
        for _ in range(args.warmup):
            launch()
        torch.cuda.synchronize()
        error = max(error, check_output(shape, tensors, (w13, w2), output, rank, world))
    else:
        launch = forward

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
    for start, end in zip(starts, ends):
        dist.barrier()
        torch.cuda.synchronize()
        start.record()
        for _ in range(args.inner):
            launch()
        end.record()
        end.synchronize()
    samples = torch.tensor([a.elapsed_time(b) * 1000 / args.inner
                            for a, b in zip(starts, ends)], device="cuda")
    gathered = [torch.empty_like(samples) for _ in range(world)]
    dist.all_gather(gathered, samples)
    result = latency_summary([s.tolist() for s in gathered], args.tokens * world)
    result.update(status="ok", relative_l2=error, gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  nccl_runtime_version=nccl_version.value,
                  torch_nccl_build_version=torch.cuda.nccl.version(),
                  flashinfer_version=flashinfer.__version__,
                  timing="max-rank CUDA events per batch of repeated forwards",
                  warmup=args.warmup, repeat=args.repeat, inner=args.inner)
    if args.backend == "mega":
        result["mega_knob_overrides"] = MEGA_KNOBS.copy()
    torch.cuda.synchronize()
    if graph is not None:
        graph.reset()
    if state is not None:
        state.destroy()
    layer.destroy()
    dist.destroy_process_group()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SHAPES, default="qwen3")
    parser.add_argument("--backend", choices=BACKENDS, default="mega")
    parser.add_argument("--tokens", type=int, default=8, help="source tokens per rank")
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.tokens, args.warmup, args.repeat, args.inner) <= 0:
        parser.error("tokens, warmup, repeat, and inner must be positive")
    shape = SHAPES[args.model]
    result = dict(model=args.model, shape=asdict(shape), backend=args.backend,
                  tokens_per_rank=args.tokens, global_tokens=args.tokens * 8,
                  ep_size=8, mode=args.mode, dtype="bf16", activation="unclamped_swiglu",
                  routing="synthetic_uniform_precomputed", shared_experts=False,
                  target_flashinfer_commit=FLASHINFER_COMMIT)
    reason = skip_reason(shape, args.backend)
    if reason:
        result.update(status="skipped", reason=reason)
    elif args.dry_run:
        result.update(status="planned")
    else:
        result.update(benchmark(args, shape))
    if int(os.environ.get("RANK", "0")) == 0:
        line = json.dumps(result)
        print(line, flush=True)
        if not args.dry_run:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a") as handle:
                handle.write(line + "\n")


if __name__ == "__main__":
    main()
