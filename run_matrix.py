"""Run each EP8 case in a fresh process group; keep failures and logs."""

import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys

from bench_moe import BACKENDS, SHAPES, TOKENS, skip_reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--tokens", nargs="+", type=int, default=list(TOKENS))
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launcher-prefix", help="Command that launches all eight ranks")
    parser.add_argument("--benchmark-script", type=Path,
                        default=Path(__file__).with_name("bench_moe.py"))
    parser.add_argument("--retry-failed", type=Path, help="Run only failed cases from this JSONL")
    args = parser.parse_args()
    if min(args.tokens) <= 0:
        parser.error("tokens must be positive")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    retry_cases = None
    if args.retry_failed:
        rows = [json.loads(line) for line in args.retry_failed.read_text().splitlines()]
        retry_cases = {(r["model"], r["backend"], r["tokens_per_rank"], r["mode"])
                       for r in rows if r["status"] == "failed"}
    failed = 0
    for model, backend, tokens in itertools.product(args.models, args.backends, args.tokens):
        if retry_cases is not None and (model, backend, tokens, args.mode) not in retry_cases:
            continue
        case = f"{model}_{backend}_{tokens}_{args.mode}"
        case_result = args.output_dir / f"{case}.jsonl"
        cmd = [sys.executable]
        if not args.dry_run and not skip_reason(SHAPES[model], backend):
            cmd = (shlex.split(args.launcher_prefix) if args.launcher_prefix else
                   cmd + ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8"])
        cmd += [str(args.benchmark_script), "--model", model,
                "--backend", backend, "--tokens", str(tokens), "--mode", args.mode,
                "--output", str(case_result)]
        if args.dry_run:
            subprocess.run(cmd + ["--dry-run"], check=True)
            continue
        print(case, flush=True)
        if case_result.exists():
            raise FileExistsError(f"Use a fresh output directory: {case_result}")
        log = args.output_dir / f"{case}.log"
        with log.open("w") as handle:
            run = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
        if run.returncode:
            failed += 1
            with (args.output_dir / "results.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(status="failed", model=model, backend=backend,
                                             tokens_per_rank=tokens, mode=args.mode,
                                             returncode=run.returncode, log=str(log))) + "\n")
            print(f"FAILED: {log}", flush=True)
        else:
            # Publish only after every launcher process exits successfully.
            with (args.output_dir / "results.jsonl").open("a") as handle:
                handle.write(case_result.read_text())
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
