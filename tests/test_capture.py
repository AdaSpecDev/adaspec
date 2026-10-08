"""Validate live-state snapshots without importing the GPU runtime."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "trace", Path(__file__).parents[2] / "vllm/vllm/v1/adaspec_trace.py"
)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class CaptureTests(unittest.TestCase):
    def test_capture_records_actual_blocks_and_preserves_prepass_lengths(self):
        request = SimpleNamespace(num_computed_tokens=4096, num_prompt_tokens=4096)
        scheduler = SimpleNamespace(
            requests={"r": request},
            kv_cache_manager=SimpleNamespace(get_block_ids=lambda _: ([4, 8, 12],)),
        )
        output = SimpleNamespace(
            adaspec_metadata=None,
            num_scheduled_tokens={"r": 4},
            scheduled_spec_decode_tokens={"r": [1, 2, 3]},
            preempted_req_ids=set(),
        )
        with patch.dict(os.environ, {"ADASPEC_TRACE_DIR": "/unused-test-path"}):
            trace.capture_schedule(scheduler, output)
        request.num_computed_tokens += 4
        captured = output.adaspec_metadata["requests"][0]
        self.assertEqual(captured["cached_tokens"], 4096)
        self.assertEqual(captured["allocated_blocks_by_group"], [3])
        self.assertEqual(captured["draft_tokens"], 3)
        self.assertFalse(captured["has_prefill"])

    def test_disabled_capture_does_not_touch_runtime_state(self):
        with patch.dict(os.environ, {}, clear=True):
            trace.capture_schedule(None, None)


if __name__ == "__main__":
    unittest.main()
