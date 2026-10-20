"""Profile one warmed eager compute call; exclude these timings from the sweep."""

import json
import os
from pathlib import Path

import bench_moe

main = bench_moe.main
bench_moe.main = lambda: None
import bench_cute_bf16
bench_moe.main = main

import torch
from flashinfer.moe_ep.core.kernel import registry

register = registry.register_split_kernel


def profile_registration(name):
    def decorate(cls):
        compute = cls.compute
        calls = 0

        def measured(self, ctx):
            nonlocal calls
            calls += 1
            if calls != 4:
                return compute(self, ctx)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]) as prof:
                output = compute(self, ctx)
                torch.cuda.synchronize()
            events = sorted(prof.key_averages(),
                            key=lambda e: e.self_device_time_total, reverse=True)
            data = dict(rank=int(os.environ['RANK']),
                        receive_shape=list(ctx.expert_tensors.shape),
                        routing_shape=list(ctx.recv_topk_idx.shape),
                        events=[dict(name=e.key, calls=e.count,
                                     self_cuda_us=e.self_device_time_total,
                                     cuda_us=e.device_time_total)
                                for e in events if e.self_device_time_total > 0])
            path = Path(os.environ['PROFILE_DIR']) / f"rank-{data['rank']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + '\n')
            return output

        cls.compute = measured
        return register(name)(cls)
    return decorate


registry.register_split_kernel = profile_registration
benchmark = bench_moe.benchmark


def profile_benchmark(args, shape):
    if (args.mode, args.warmup, args.repeat, args.inner) != ("eager", 2, 1, 1):
        raise ValueError("Use --mode eager --warmup 2 --repeat 1 --inner 1")
    result = benchmark(args, shape)
    for key in ("p50_us", "p95_us", "tokens_per_second"):
        result.pop(key)
    result.update(status="diagnostic", timing="One warmed compute call under profiler")
    return result


bench_moe.benchmark = profile_benchmark
main()
