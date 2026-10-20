"""Render measured JSONL rows as a Markdown report; retain skips and failures."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def render(rows, label, source):
    keys = [(r["model"], r["backend"], r["tokens_per_rank"], r["mode"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate cases: use a fresh output directory for each run")
    baseline = {(r["model"], r["tokens_per_rank"], r["mode"]): r["p50_us"]
                for r in rows if r["status"] == "ok" and r["backend"] == "mega"}
    variants = {r.get("routing_variant", "stock") for r in rows
                if r["status"] == "ok" and r["backend"] != "mega"}
    if variants == {"local_routes_packed"}:
        routing_note = ("Experimental CuTe DSL BF16 split adapter. GPU route sorting, "
                        "padded gather, counter reset, and output reduction are timed. "
                        "MegaMoE rows reuse the measured baseline.")
    elif variants == {"skip_nonlocal"}:
        routing_note = ("Split rows exclude non-local picks; timing includes the wrapper. "
                        "MegaMoE rows reuse the measured baseline.")
    elif len(variants) > 1:
        routing_note = "Mixed routing variants: check each raw row before comparing timings."
    else:
        routing_note = ("The stock split adapter computes zero-weight non-local routes. "
                        "These are full layer timings, not optimized TRT-LLM expert compute.")
    lines = [f"# {label} results", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"Raw input: `{source}`", "",
             "EP8 / TP1, synthetic BF16 weights, unclamped SwiGLU, precomputed routing.",
             "Latency is per forward, using the slowest rank. Speedup = Mega p50 / backend p50.",
             "Missing baselines have no speedup. No timing is assigned to skipped or failed cases.", "",
             routing_note, "",
             "| Model | Backend | Tokens/rank | Mode | Status | p50 µs | p95 µs | Tokens/s | Speedup |",
             "|---|---|---:|---|---|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda r: (r["model"], r["tokens_per_rank"], r["mode"], r["backend"])):
        prefix = f"| {row['model']} | {row['backend']} | {row['tokens_per_rank']} | {row['mode']} | {row['status']} |"
        if row["status"] != "ok":
            lines.append(prefix + " — | — | — | — |")
            continue
        base = baseline.get((row["model"], row["tokens_per_rank"], row["mode"]))
        speedup = f"{base / row['p50_us']:.3f}×" if base is not None else "—"
        lines.append(prefix + f" {row['p50_us']:.2f} | {row['p95_us']:.2f} | {row['tokens_per_second']:.0f} | {speedup} |")
    lines += ["", "## Skips and failures", ""]
    for row in rows:
        if row["status"] != "ok":
            detail = row.get("reason", row.get("failure_record",
                                             f"See {row.get('log', 'case log')}"))
            lines.append(f"- {row['model']}/{row['backend']}/{row['tokens_per_rank']}: {detail}")
    lines += ["", "Hardware, library versions, and source commits accompany the raw data.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/results.md"))
    parser.add_argument("--label", "--cluster", dest="label", default="8× NVIDIA GB200")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows or any(r["status"] not in ("ok", "failed", "skipped") for r in rows):
        parser.error("Input must contain execution results, not dry-run plans")
    report = render(rows, args.label, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)


if __name__ == "__main__":
    main()
