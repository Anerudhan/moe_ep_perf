# BF16 MoE on eight Blackwell GPUs

Repository: [Anerudhan/moe_ep_perf](https://github.com/Anerudhan/moe_ep_perf).
See [the results](results/gb200/results.md). Measured hardware: **8× NVIDIA GB200**.
B200 support is checked in source; B200 performance has not been measured.

The harness compares one routed MoE layer at EP8 / TP1:

| Backend | Communication | Compute | Decode / prefill |
|---|---|---|---|
| `mega` | Fused NVSHMEM path | CuTe DSL BF16 MegaMoE | Both |
| `nvlink` | NVLink one-sided | TRT-LLM BF16 | Both |
| `nccl_ll` | NCCL-EP low latency, rank-major | TRT-LLM BF16 | Both, subject to LL limits |
| `nccl_ht` | NCCL-EP high throughput, flat | TRT-LLM BF16 | Both |

Use LL as the decode candidate and HT as the prefill candidate. Measure both
across the token sweep; their names do not establish a performance winner.
`nvlink` is FlashInfer's `NVLinkOneSidedConfig`, the transport corresponding to
vLLM's `flashinfer_nvlink_one_sided` selection. It has no NCCL LL/HT mode switch.

Support was checked against FlashInfer commit
`35a10560da05517ceb5472ba90c88d53f9129a40` on 2026-10-19.
The corrected TRT-LLM and experimental BF16 CuTe DSL comparisons each have
99 passes and nine expected skips, including the same 27 MegaMoE baseline rows.
B200 performance has not been measured. See
[GB200 results](results/gb200/results.md). CPU tests and the dry run pass.

The TRT-LLM results use the [routing fix](#routing-fix) below.

## Shapes

| Preset | Model | Hidden | Expert intermediate | Experts | Top-k | Experts/rank |
|---|---|---:|---:|---:|---:|---:|
| `qwen3` | Qwen3-235B-A22B | 4096 | 1536 | 128 | 8 | 16 |
| `qwen35` | Qwen3.5-397B-A17B | 4096 | 1024 | 512 | 10 | 64 |
| `dsv41` | DeepSeek-V4.1-Flash-1017 | 5120 | 2304 | 384 | 6 | 48 |

Dimensions come from the pinned model configs:
[Qwen3](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json),
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/8472618112abcbd45acbcdc58436aff4233c23f7/config.json),
[DS-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json).

These are **shape benchmarks** with random BF16 weights, unclamped SwiGLU,
and synthetic precomputed routing. They exclude shared experts, router compute,
attention, checkpoint loading, and model-specific routing scales. DS-V4.1's
checkpoint uses quantized experts and a SwiGLU limit of 10; this benchmark
deliberately measures unquantized BF16 at its dimensions. It is not a model
accuracy or serving benchmark.

NCCL-EP LL accepts BF16 hidden widths {2048, 2560, 4096, 5120, 6144, 7168, 8192}
and top-k ≤ 8 in the checked runtime. Thus `qwen35/nccl_ll` is recorded as
`skipped`; top-k stays 10. Qwen3 and DS-V4.1 meet these constraints.
See the [LL limits](https://github.com/flashinfer-ai/flashinfer/blob/35a10560da05517ceb5472ba90c88d53f9129a40/docs/design_docs/moe_ep_runbook.md#nccl-ep-low-latency-device-kernel-limits).

## Routing fix

[diagnose_routing.py](diagnose_routing.py) wraps FlashInfer's rank-major
bridge when `MOE_ROUTE_VARIANT=skip_nonlocal` is set:

```python
pack = original_bridge(recv, ids, weights, **kwargs)
local = (ids >= 0) & (ids < kwargs["num_local_experts"])
pack.topk_ids = torch.where(local, pack.topk_ids, -1)
```

The received IDs use local expert indices. The stock bridge maps non-local
picks to the first local expert with zero weight. Duplicate IDs can overwrite
a real route in the small-token kernel, and zero-weight routes still cost
compute. Setting those IDs to `-1` makes TRT-LLM skip them.

The wrapper runs inside the measured forward. It preserves inputs, weights,
and correctness thresholds. This is a harness fix on pinned FlashInfer
`35a1056`, separate from [PR #6840](https://github.com/flashinfer-ai/flashinfer/pull/6840),
which uses a valid remote expert ID. MegaMoE does not use this wrapper; the
experimental CuTe adapter packs local routes itself.

## Run

Unpack the shared archive and enter `moe_ep_perf`. With repository access,
you can also clone it:

```bash
git clone https://github.com/Anerudhan/moe_ep_perf.git
cd moe_ep_perf
```

Use eight NVLink-connected GB200 GPUs, Python ≥3.10, CUDA-enabled PyTorch, and
the checked FlashInfer source with its EP dependencies. B200 NCCL-EP requires
the **loaded** NCCL library to be ≥2.30.7. MegaMoE needs NVSHMEM and CUTLASS DSL.
The upstream [EP environment builder](https://github.com/flashinfer-ai/flashinfer/blob/35a10560da05517ceb5472ba90c88d53f9129a40/docker/install/build_flashinfer_ep_pytorch.sh)
pins a compatible stack inside an NVIDIA PyTorch image. Follow that recipe;
a plain FlashInfer wheel alone does not establish this environment.

Pin the FlashInfer source before following its environment builder:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git flashinfer
git -C flashinfer checkout 35a10560da05517ceb5472ba90c88d53f9129a40
git -C flashinfer submodule update --init --recursive
```

Run the builder inside a dedicated NVIDIA PyTorch container, with
`FI_SRC` set to the absolute path of that checkout. The upstream recipe uses
`nvcr.io/nvidia/pytorch:26.05-py3`. It changes the container's Python packages.

```bash
export FI_SRC="$PWD/flashinfer"
FI_EP_PREWARM=1 bash "$FI_SRC/docker/install/build_flashinfer_ep_pytorch.sh"
```

Prewarm builds the TRT-LLM modules before the distributed run. Use the same
built container for every backend. This is the portable upstream setup recipe;
the measured runtime versions below identify the environment used for the data.

The measured runtime versions are in [environment.json](results/gb200/environment.json).
For a direct source import, run `python prepare_flashinfer.py` and add the
FlashInfer checkout to `PYTHONPATH` inside the prepared environment.

The commands below assume all eight GPUs are visible to one launcher. If the
GPUs span hosts, use `--launcher-prefix` with a launcher that starts one
`torchrun` agent per host, with a common rendezvous endpoint and eight total
workers. Run the matrix driver once; use the same source and environment on
every worker. Site allocation steps are outside this repro.

From this directory, run the three sweeps used in the report:

```bash
python -m unittest -v test_bench_moe.py

# MegaMoE baseline.
python run_matrix.py --backends mega --output-dir results/mega

# TRT-LLM with the routing fix above.
MOE_ROUTE_VARIANT=skip_nonlocal python run_matrix.py \
  --benchmark-script diagnose_routing.py --backends nvlink nccl_ll nccl_ht \
  --output-dir results/trtllm

# Experimental BF16 CuTe compute with the same three transports.
python run_matrix.py --benchmark-script bench_cute_bf16.py \
  --backends nvlink nccl_ll nccl_ht --output-dir results/cute-bf16

# Reuse the same MegaMoE baseline in both comparisons.
cat results/mega/results.jsonl results/trtllm/results.jsonl > results/trtllm-comparison.jsonl
cat results/mega/results.jsonl results/cute-bf16/results.jsonl > results/cute-comparison.jsonl
```

Each sweep uses 1, 8, 32, 128, 512, 1024, 2048, 4096, and 8192 tokens/GPU.
Add `--dry-run` to inspect a sweep without launching workers. For a smaller
check, add `--tokens 8 4096`. Each case runs in a fresh process group.

Use a new output directory for each run. Results append to `results.jsonl`;
logs and per-case JSONL retain failed attempts. A passing aggregate row is
accepted only after the launcher exits successfully. Failures make the matrix
exit nonzero. Use `--retry-failed path/to/results.jsonl` to rerun failures into
a fresh directory after fixing the cause.

The CuTe adapter reuses FlashInfer's SM100 fused BF16 FC1+FC2 kernel. GPU route
sorting, padded gather, counter reset, and FP32 output reduction are timed.
Weights and activations stay BF16. The tile is fixed at 256×256×64 with a
two-CTA cluster; this is an experimental split backend.

## Result files and generated views

The published raw comparisons are [corrected.jsonl](results/gb200/corrected.jsonl)
(TRT-LLM) and [cute-bf16.jsonl](results/gb200/cute-bf16.jsonl). Both include the
same MegaMoE baseline. These files are reference outputs, not benchmark inputs.
The [main report](results/gb200/results.md) contains summary tables and plots;
[environment.json](results/gb200/environment.json) records the runtime versions.
The remaining JSON files support the profile and timing check cited in the report.
Debug runs and duplicate generated tables are omitted.

Generate a full table or plot from either raw comparison:

```bash
python summarize_results.py results/gb200/corrected.jsonl --output results/trtllm-table.md
python summarize_results.py results/gb200/cute-bf16.jsonl --output results/cute-table.md

# Plotting requires matplotlib; the published plots use version 3.10.9.
python plot_results.py results/gb200/corrected.jsonl \
  --output results/trtllm-latency.png --corrected-routing
python plot_results.py results/gb200/cute-bf16.jsonl \
  --output results/cute-latency.png --corrected-routing --split-compute-label 'CuTe BF16'
```

The plot script writes PNG and SVG. Only the PNGs embedded in the main report
are stored in the repo.

## Measurement

- Same seeded weights, activations, routes, and probabilities across backends.
- Source tokens per rank are explicit. Global tokens = 8 × tokens/rank.
  Small counts model decode batches; larger counts model prefill chunks.
- Time the full forward: staging, dispatch, local compute, and combine.
  Initial allocation, weight conversion, JIT, autotuning, and correctness
  checks are outside timing. Eager timing includes per-forward allocations.
  Dispatch includes any per-forward layout work.
- Graph replay is the default. NCCL split paths use persistent graph handles.
  Eager mode includes host launch gaps visible to CUDA events.
- TRT-LLM tunes during warmup. Mega uses its fixed BF16 kernel geometry, with
  routing weights applied to SwiGLU before FC2. Split TRT-LLM weights after
  FC2, so BF16 rounding can differ. No exhaustive Mega tuning is claimed.
- The adapters use different gate/up orders. Conversion occurs before timing.
  An independent FP32 reference checks up to eight spread-out tokens per rank,
  with a BF16 GEMM handoff, before timing and after graph warmup. All output
  elements must be finite. Failed checks prevent timing results.
- Each sample averages ten launches. Report p50 and p95 of the slowest rank's
  sample, plus global tokens/s. Barriers and rank reductions are outside timing.
  The p95 is for batches of launches, not request tail latency.
- Routing stays fixed across replay. This measures steady state with synthetic
  balanced-in-expectation routing; it does not characterize hot-expert skew.
## CuTe DSL compute

BF16 CuTe DSL **MegaMoE is available on B200** through
[`Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig`](https://github.com/flashinfer-ai/flashinfer/blob/35a10560da05517ceb5472ba90c88d53f9129a40/flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_bf16_bf16_cutedsl/backend.py).
Its shape checks require hidden % 32 = 0 and intermediate % 64 = 0; all three
presets meet them.

The standalone [`cute_dsl_fused_moe_bf16`](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.cute_dsl_fused_moe_bf16.html)
is SM90-only. B200's split `CuteDslConfig` supports W4A4, W4A8, and W4A16;
W4A16 still has 4-bit weights. It does not provide a BF16×BF16 replacement for
TRT-LLM in this split comparison. Such a replacement needs a separate compute
adapter/kernel integration.
This repo supplies an experimental integration in
[bench_cute_bf16.py](bench_cute_bf16.py), using the lower-level
[`Sm100SwigluBf16Fc12Kernel`](https://github.com/flashinfer-ai/flashinfer/blob/35a10560da05517ceb5472ba90c88d53f9129a40/flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src/moe_bf16_glu/kernel_bf16_glu_fc12.py).
All 72 supported NVLink, NCCL LL, and NCCL HT cases passed correctness and
graph replay on eight GB200 GPUs. See the [CuTe results and adapter profile](results/gb200/results.md#bf16-cute-dsl-with-separate-communication).

The [split architecture](https://github.com/flashinfer-ai/flashinfer/blob/35a10560da05517ceb5472ba90c88d53f9129a40/docs/design_docs/moe_ep_architecture.md)
and [HT BF16 test](https://github.com/flashinfer-ai/flashinfer/blob/35a10560da05517ceb5472ba90c88d53f9129a40/tests/moe_ep/test_moe_ep_ht_correctness.py)
confirm TRT-LLM BF16 composition with the EP transports. Hardware support does
not establish which backend wins for these exact shapes.
