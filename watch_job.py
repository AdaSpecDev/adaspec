#!/usr/bin/env python3
"""Emit one durable allocation event, or a terminal event if allocation was missed."""

import argparse
import datetime
import json
import pathlib
import subprocess
import time


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def inspect(job_id):
    live = query(["squeue", "-h", "-j", job_id, "-o", "%T|%N"])
    if live:
        state, nodes = live.splitlines()[0].split("|", 1)
        return state, nodes
    history = query(
        ["sacct", "-n", "-X", "-j", job_id, "--format=State,NodeList", "-P"]
    )
    if history:
        state, nodes, *_ = history.splitlines()[0].split("|")
        return state.strip(), nodes.strip()
    return "UNKNOWN", ""


def classify(state, nodes):
    if nodes and nodes.lower() not in {"none", "(null)", "unknown", "n/a"}:
        return "job_allocated"
    if state.split()[0].rstrip("+") in {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
    }:
        return "job_terminated_without_observed_allocation"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-hours", type=float, default=48)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_hours * 3600
    previous = None
    while time.monotonic() < deadline:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            state, nodes = inspect(args.job_id)
            if (state, nodes) != previous:
                print(
                    f"{now} job={args.job_id} state={state} nodes={nodes}", flush=True
                )
                previous = (state, nodes)
            event_type = classify(state, nodes)
            if event_type:
                event = {
                    "event": event_type,
                    "job_id": args.job_id,
                    "observed_at": now,
                    "state": state,
                    "nodes": nodes,
                    "source": "slurm_poll",
                    "poll_interval_seconds": args.interval,
                }
                # Exclusive creation prevents duplicate events from multiple watchers.
                try:
                    with args.output.open("x") as stream:
                        stream.write(json.dumps(event) + "\n")
                except FileExistsError:
                    pass
                print(json.dumps(event), flush=True)
                return
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"{now} query error: {error}", flush=True)
        time.sleep(args.interval)
    print("Watcher expired; no allocation observed within its time limit.", flush=True)


if __name__ == "__main__":
    main()
