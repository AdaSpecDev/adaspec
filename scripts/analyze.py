"""Fail closed on invalid batches; export live per-pass features for inspection."""

import csv
import json
from pathlib import Path
import statistics
import sys


def analyze(root):
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = manifest["config"]
    trials = json.loads((root / "trials.json").read_text())
    lookup = {
        request_id: trial for trial in trials for request_id in trial["request_ids"]
    }
    features = []
    counts = {}
    rejected = {}
    integrity_failures = []
    initial_passes = {}
    for path in sorted((root / "traces").glob("target-*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            rows = event["requests"]
            if not rows:
                continue
            trial = lookup.get(rows[0]["request_id"])
            if trial is None:
                raise ValueError("Trace contains an unknown request")
            reason = None
            if any(row["has_prefill"] for row in rows):
                reason = "prefill"
            elif event["preempted_request_ids"]:
                reason = "preemption"
            elif len(rows) != cfg["batch_size"] or {
                r["request_id"] for r in rows
            } != set(trial["request_ids"]):
                reason = "partial_or_mixed_trial_batch"
            elif any(r["cached_tokens"] != r["worker_cached_tokens"] for r in rows):
                reason = "scheduler_worker_length_mismatch"
            elif any(
                r["draft_tokens"] != cfg["draft_depth"]
                or r["query_tokens"] != cfg["draft_depth"] + 1
                for r in rows
            ):
                reason = "incomplete_speculation"
            elif event["target_forward_gpu_ms"] <= 0:
                reason = "invalid_timing"
            if reason:
                if reason in {
                    "scheduler_worker_length_mismatch",
                    "invalid_timing",
                    "preemption",
                }:
                    integrity_failures.append(f"step {event['step']}: {reason}")
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            expected_lengths = sorted(cfg["mixtures"][trial["mixture"]])
            if sorted(r["prompt_tokens"] for r in rows) != expected_lengths:
                raise ValueError(
                    "Live prompt distribution differs from configured workload"
                )
            lengths = [r["cached_tokens"] for r in rows]
            feature = {
                "mixture": trial["mixture"],
                "repeat": trial["repeat"],
                "warmup": trial["warmup"],
                "step": event["step"],
                "batch_size": len(rows),
                "draft_depth": cfg["draft_depth"],
                "at_prompt_boundary": all(
                    r["cached_tokens"] == r["prompt_tokens"] for r in rows
                ),
                "mean_cached_tokens": statistics.mean(lengths),
                "max_cached_tokens": max(lengths),
                "std_cached_tokens": statistics.pstdev(lengths),
                "short_count": sum(x < 2048 for x in lengths),
                "long_count": sum(x >= 6144 for x in lengths),
                "allocated_blocks": sum(
                    sum(r["allocated_blocks_by_group"]) for r in rows
                ),
                "query_tokens": sum(r["query_tokens"] for r in rows),
                "target_forward_gpu_ms": event["target_forward_gpu_ms"],
            }
            features.append(feature)
            key = f"{trial['mixture']}/{trial['repeat']}"
            counts[key] = counts.get(key, 0) + 1
            if feature["at_prompt_boundary"]:
                initial_passes.setdefault(key, feature)
    failures = list(integrity_failures)
    for trial in trials:
        key = f"{trial['mixture']}/{trial['repeat']}"
        if key not in initial_passes:
            failures.append(
                f"{key}: no full verification at exact initial context lengths"
            )
        if counts.get(key, 0) < 5:
            failures.append(f"{key}: fewer than five validated verification passes")
    if features:
        with (root / "features.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(features[0]))
            writer.writeheader()
            writer.writerows(features)
    (root / "matched_initial_passes.json").write_text(
        json.dumps(list(initial_passes.values()), indent=2) + "\n"
    )
    report = {
        "passed": not failures,
        "failures": failures,
        "valid_passes_by_trial": counts,
        "excluded_passes": rejected,
        "scope": "Pilot validates live batches and target-forward timing. No hypothesis conclusion: live mean lengths may differ and instrumentation synchronizes each target forward.",
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
