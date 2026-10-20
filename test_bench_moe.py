import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from bench_moe import SHAPES, latency_summary, skip_reason
from summarize_results import render


class BenchmarkChecks(unittest.TestCase):
    def test_corrected_matrix_skip_without_gpu_dependencies(self):
        for script in ("diagnose_routing.py", "bench_cute_bf16.py"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                subprocess.run([
                    sys.executable, "run_matrix.py", "--benchmark-script", script,
                    "--models", "qwen35", "--backends", "nccl_ll", "--tokens", "1",
                    "--output-dir", directory,
                ], env={**os.environ, "MOE_ROUTE_VARIANT": "skip_nonlocal"},
                    check=True, capture_output=True, text=True)
                row = json.loads((Path(directory) / "results.jsonl").read_text())
                self.assertEqual(row["status"], "skipped")

    def test_external_launcher_receives_case(self):
        with tempfile.TemporaryDirectory() as directory:
            captured = Path(directory) / "args.json"
            failed = Path(directory) / "previous.jsonl"
            failed.write_text("\n".join(json.dumps(dict(
                model="qwen3", backend="nvlink", tokens_per_rank=tokens,
                mode="graph", status=status,
            )) for tokens, status in ((1, "ok"), (8, "failed"))) + "\n")
            prefix = shlex.join([
                sys.executable, "-c",
                "import json, pathlib, sys; "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:])); "
                "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}\\n')",
                str(captured),
            ])
            run = subprocess.run([
                sys.executable, "run_matrix.py", "--models", "qwen3",
                "--backends", "nvlink", "--tokens", "1", "8", "--output-dir", directory,
                "--launcher-prefix", prefix, "--retry-failed", str(failed),
            ], check=True, capture_output=True, text=True)
            self.assertEqual(run.stdout.strip(), "qwen3_nvlink_8_graph")
            args = json.loads(captured.read_text())
            self.assertEqual(Path(args[0]).name, "bench_moe.py")
            self.assertEqual(args[1:7], ["--model", "qwen3", "--backend", "nvlink",
                                        "--tokens", "8"])

    def test_launcher_failure_excludes_partial_success(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = shlex.join([
                sys.executable, "-c",
                "import pathlib, sys; "
                "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text("
                "'{\"status\":\"ok\"}\\n'); sys.exit(2)",
            ])
            run = subprocess.run([
                sys.executable, "run_matrix.py", "--models", "qwen3",
                "--backends", "nvlink", "--tokens", "8", "--output-dir", directory,
                "--launcher-prefix", prefix,
            ], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            rows = [json.loads(s) for s in
                    (Path(directory) / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(rows[0]["returncode"], 2)

    def test_qwen_top10_is_not_silently_changed(self):
        shape = SHAPES["qwen35"]
        self.assertEqual(shape.top_k, 10)
        self.assertIsNotNone(skip_reason(shape, "nccl_ll"))
        for backend in ("mega", "nvlink", "nccl_ht"):
            self.assertIsNone(skip_reason(shape, backend))

    def test_slowest_rank_per_sample(self):
        stats = latency_summary([[1, 9, 2], [8, 2, 7]], 80)
        self.assertEqual(stats["p50_us"], 8)
        self.assertEqual(stats["p95_us"], 9)
        self.assertEqual(stats["tokens_per_second"], 10_000_000)

    def test_matrix_without_gpu_dependencies(self):
        run = subprocess.run([sys.executable, "run_matrix.py", "--dry-run",
                              "--tokens", "1", "4096"], check=True,
                             capture_output=True, text=True)
        rows = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(row["status"] == "skipped" for row in rows), 2)
        self.assertTrue(all(row["global_tokens"] == row["tokens_per_rank"] * 8
                            for row in rows))

    def test_report_preserves_skips(self):
        row = dict(model="qwen35", backend="nccl_ll", tokens_per_rank=8,
                   mode="graph", status="skipped", reason="top-k exceeds 8")
        report = render([row], "test", "test.jsonl")
        self.assertIn("skipped | — | — | — | — |", report)
        self.assertIn("top-k exceeds 8", report)

    def test_report_rejects_mixed_repeated_runs(self):
        row = dict(model="qwen3", backend="nvlink", tokens_per_rank=8,
                   mode="graph", status="failed")
        with self.assertRaisesRegex(ValueError, "Duplicate cases"):
            render([row, row], "test", "test.jsonl")


if __name__ == "__main__":
    unittest.main()
