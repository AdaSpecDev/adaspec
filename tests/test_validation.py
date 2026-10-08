"""Exercise the pilot's fail-closed contract without requiring a GPU."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "analysis", Path(__file__).parents[1] / "scripts/analyze.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class ValidationTests(unittest.TestCase):
    def test_scheduler_worker_disagreement_rejects_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "traces").mkdir()
            cfg = {"batch_size": 2, "draft_depth": 3, "mixtures": {"uniform": [16, 16]}}
            (root / "manifest.json").write_text(json.dumps({"config": cfg}))
            trial = {
                "mixture": "uniform",
                "repeat": 0,
                "warmup": False,
                "request_ids": ["a", "b"],
            }
            (root / "trials.json").write_text(json.dumps([trial]))
            rows = [
                {
                    "request_id": rid,
                    "cached_tokens": 16,
                    "worker_cached_tokens": 16,
                    "prompt_tokens": 16,
                    "draft_tokens": 3,
                    "query_tokens": 4,
                    "allocated_blocks_by_group": [2],
                    "has_prefill": False,
                }
                for rid in ["a", "b"]
            ]
            event = {
                "requests": rows,
                "preempted_request_ids": [],
                "target_forward_gpu_ms": 1.0,
                "step": 1,
            }
            trace = root / "traces/target-1.jsonl"
            trace.write_text((json.dumps(event) + "\n") * 5)
            analysis.analyze(root)
            self.assertTrue(
                json.loads((root / "validation.json").read_text())["passed"]
            )
            rows[0]["worker_cached_tokens"] += 1
            trace.write_text((json.dumps(event) + "\n") * 5)
            with self.assertRaises(SystemExit):
                analysis.analyze(root)
            self.assertFalse(
                json.loads((root / "validation.json").read_text())["passed"]
            )


if __name__ == "__main__":
    unittest.main()
