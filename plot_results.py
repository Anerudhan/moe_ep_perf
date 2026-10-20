"""Plot measured EP8 latency. Requires matplotlib."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_moe import SHAPES


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("--output", type=Path, default=Path("results/latency.png"))
parser.add_argument("--title", default="EP8 BF16 routed MoE")
parser.add_argument("--corrected-routing", action="store_true")
parser.add_argument("--split-compute-label", default="TRT-LLM")
args = parser.parse_args()
rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
series = {
    "mega": ("MegaMoE", "#0072B2"),
    "nvlink": (f"NVLink + {args.split_compute_label}", "#D55E00"),
    "nccl_ll": (f"NCCL LL + {args.split_compute_label}", "#009E73"),
    "nccl_ht": (f"NCCL HT + {args.split_compute_label}", "#CC79A7"),
}
titles = {"qwen3": "Qwen3-235B", "qwen35": "Qwen3.5-397B", "dsv41": "DS-V4.1-Flash"}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for ax, model in zip(axes, SHAPES):
    for backend, (label, color) in series.items():
        points = sorted(
            [r for r in rows if r["model"] == model and r["backend"] == backend
             and r["status"] == "ok" and r["mode"] == "graph"],
            key=lambda r: r["tokens_per_rank"],
        )
        if points:
            ax.plot([r["tokens_per_rank"] for r in points],
                    [r["p50_us"] for r in points], "o-", label=label,
                    color=color, linewidth=1.8, markersize=4)
    ax.set(xscale="log", yscale="log", title=titles[model],
           xlabel="Tokens per GPU", ylabel="p50 latency (µs)")
    ax.set_xticks([1, 8, 32, 128, 512, 2048, 8192],
                  labels=["1", "8", "32", "128", "512", "2048", "8192"], rotation=45)
    ax.grid(True, which="major", alpha=0.25)
    if model == "qwen35":
        ax.text(0.03, 0.96, "NCCL LL unsupported: top-k 10", transform=ax.transAxes,
                va="top", fontsize=8)
    failed = [r for r in rows if r["model"] == model and r["status"] == "failed"]
    if failed:
        note = "No timing: " + ", ".join(
            f"{r['backend']} T={r['tokens_per_rank']}" for r in failed)
        ax.text(0.03, 0.88 if model == "qwen35" else 0.96, note,
                transform=ax.transAxes, va="top", fontsize=8)
fig.suptitle(args.title, fontsize=14)
legend = {}
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    legend.update(zip(labels, handles))
labels = [label for label, _ in series.values() if label in legend]
fig.legend([legend[label] for label in labels], labels, loc="lower center",
           bbox_to_anchor=(0.5, 0.09), ncol=4)
note = ("Split: non-local routes excluded; adapter cost included. "
        "MegaMoE reuses the measured baseline." if args.corrected_routing else
        "Stock split adapter computes masked routes. These are full layer times, "
        "not optimized TRT-LLM compute times.")
fig.text(0.5, 0.035, note, ha="center", fontsize=9)
fig.subplots_adjust(top=0.84, bottom=0.30, wspace=0.30)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)
fig.savefig(args.output.with_suffix(".svg"))
plt.close(fig)
