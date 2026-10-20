"""Experimental split EP adapter for the SM100 CuTe DSL BF16 FC1+FC2 kernel."""

from dataclasses import dataclass
from pathlib import Path
import sys

import bench_moe


@dataclass
class CuteBf16KernelConfig:
    moe_config: object
    kernel_name: str = "study_cute_bf16"


def install():
    import torch
    import flashinfer
    import flashinfer.moe_ep as ep
    from flashinfer.moe_ep.core.kernel.base import SplitKernelBackend
    from flashinfer.moe_ep.core.kernel.registry import register_split_kernel
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    source = (Path(flashinfer.__file__).parent / "moe_ep/kernel_src/sm100/"
              "cutedsl_megamoe/src")
    sys.path.insert(0, str(source.parent))
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "moe_nvfp4_swapab"))
    from moe_bf16_glu.runner_fc12 import SwigluBf16Fc12Tester
    from moe_bf16_glu.runner_common import TrainingImplDesc
    from moe_nvfp4_swapab.runner_fc12_common import ProblemDesc, MiscDesc

    @register_split_kernel("study_cute_bf16")
    class CuteBf16Kernel(SplitKernelBackend):
        @classmethod
        def kernel_name(cls):
            return "study_cute_bf16"

        def preprocess_weights(self, weights, fleet_params):
            cfg = self._config.moe_config
            self.experts = cfg.experts.local_num_experts
            self.hidden = fleet_params.token_hidden_size
            self.intermediate = cfg.experts.intermediate_size
            # The shared split harness supplies up|gate; this kernel uses gate|up.
            up, gate = weights.w13.chunk(2, dim=1)
            pack = ep.MoEWeightPack(w13=torch.cat((gate, up), dim=1), w2=weights.w2)
            self.weights = preprocess_mega_weights(
                pack, intermediate_size=self.intermediate, hidden_size=self.hidden,
            )
            self.runner = None
            return self.weights

        def _prepare(self, rows):
            problem = ProblemDesc(
                tokens_after_topk=rows, experts=self.experts, hidden=self.hidden,
                intermediate=2 * self.intermediate, kind="bf16",
            )
            impl = TrainingImplDesc(generate_c=False, enable_static_expert_shape=True)
            misc = MiscDesc(run_target_kernel_only=True, ref_compute_graph="deepgemm")
            runner = SwigluBf16Fc12Tester(problem, impl, misc)
            self.tile = runner._epilogue_token_tile
            capacity = ((rows + self.experts * (self.tile - 1) + self.tile - 1)
                        // self.tile * self.tile)
            runner.activation = torch.empty(
                (capacity, self.hidden), device="cuda", dtype=torch.bfloat16)
            runner.fc1_weight = self.weights[0][0]
            runner.fc2_weight = self.weights[1][0]
            runner.fc2_output = torch.empty_like(runner.activation)
            runner.topk_scores = torch.empty(capacity, device="cuda", dtype=torch.float32)
            runner.offs = torch.empty(self.experts, device="cuda", dtype=torch.int32)
            runner.data_physical_offsets = [0, capacity]
            self.positions = torch.arange(rows, device="cuda")
            self.ones = torch.ones(rows, device="cuda", dtype=torch.int32)
            self.runner = runner
            self.rows = rows
            self.compiled = False

        def compute(self, ctx):
            import cuda.bindings.driver as cuda

            if ctx.recv_topk_idx is None or ctx.recv_topk_weights is None:
                raise ValueError("This adapter requires received rank-major routing.")
            x = ctx.expert_tensors.reshape(-1, self.hidden)
            ids = ctx.recv_topk_idx.reshape(x.shape[0], -1)
            scales = ctx.recv_topk_weights.reshape_as(ids)
            routes = ids.numel()
            if self.runner is None:
                self._prepare(routes)
            if routes != self.rows:
                raise ValueError("Use a fresh adapter for each receive capacity.")
            runner = self.runner

            # Pack live local routes into padded expert groups on the GPU.
            local = (ids >= 0) & (ids < self.experts)
            keys = torch.where(local, ids, self.experts).flatten().long()
            sorted_experts, order = keys.sort(stable=True)
            counts = torch.zeros(self.experts + 1, device="cuda", dtype=torch.int32)
            counts.scatter_add_(0, keys, self.ones)
            ends = counts.cumsum(0)
            starts = torch.cat((ends.new_zeros(1), ends[:-1]))
            padded = ((counts + self.tile - 1) // self.tile * self.tile).cumsum(0)
            physical = torch.cat((padded.new_zeros(1), padded[:-1]))
            destinations = self.positions - starts[sorted_experts] + physical[sorted_experts]
            sources = order // ids.shape[1]
            runner.offs.copy_(ends[:self.experts])
            runner.activation.index_copy_(0, destinations, x.index_select(0, sources))
            runner.topk_scores.index_copy_(0, destinations, scales.flatten()[order].float())

            if not self.compiled:
                runner.run_kernel()
                self.counters = runner._partition_workspace()[1]
                self.compiled = True
            else:
                self.counters.zero_()
                kwargs = runner._launch_runtime_kwargs
                kwargs["stream"] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                runner._compiled_kernel(**kwargs)

            values = runner.fc2_output.index_select(0, destinations)
            values = torch.where((sorted_experts < self.experts)[:, None], values, 0)
            out = torch.zeros_like(x, dtype=torch.float32)
            out.index_add_(0, sources, values.float())
            return out.bfloat16().reshape_as(ctx.expert_tensors)

    ep.FusedMoeKernelConfig = CuteBf16KernelConfig


original_benchmark = bench_moe.benchmark


def benchmark(args, shape):
    if args.backend == "mega":
        raise ValueError("Use bench_moe.py for the native MegaMoE baseline.")
    install()
    result = original_benchmark(args, shape)
    result.update(compute="cute_dsl_bf16_fc12_experimental",
                  routing_variant="local_routes_packed",
                  compute_tile=[256, 256, 64], compute_cluster=[2, 1, 1],
                  route_weight_stage="SwiGLU output before FC2",
                  adapter="GPU sort, padded gather, fused BF16 FC1+FC2, FP32 scatter sum")
    return result


bench_moe.benchmark = benchmark
bench_moe.main()
