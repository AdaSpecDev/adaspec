"""Offline exact-token workloads; trace validation decides which passes qualify."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    models = json.loads(args.models.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    manifest = {
        "config": cfg,
        "models": models,
        "vllm_path": vllm.__file__,
        "vllm_version": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "timing_scope": "target transformer forward; excludes logits and sampling",
        "instrumentation": "synchronous CUDA events; pilot only",
        "git_commit": subprocess.check_output(
            [
                "git",
                "-C",
                os.environ["VLLM_CHECKOUT"],
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    checkout = Path(os.environ["VLLM_CHECKOUT"])
    assert Path(vllm.__file__).resolve().is_relative_to(checkout.resolve()), (
        "vLLM must import from the instrumented checkout"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["target"], revision=models["target_revision"]
    )
    base = tokenizer.encode(
        "def solve(items):\n    # Process each item and return the result.\n",
        add_special_tokens=False,
    )
    workload = {}
    for name, lengths in cfg["mixtures"].items():
        workload[name] = []
        for index, length in enumerate(lengths):
            lead = tokenizer.encode(f"# Request {index}\n", add_special_tokens=False)
            ids = (lead + base * (length // len(base) + 2))[:length]
            assert len(ids) == length
            workload[name].append({"prompt_token_ids": ids})
    (args.output / "workloads.json").write_text(json.dumps(workload))
    manifest["workload_sha256"] = hashlib.sha256(
        (args.output / "workloads.json").read_bytes()
    ).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    llm = LLM(
        model=cfg["target"],
        revision=models["target_revision"],
        tokenizer_revision=models["target_revision"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=cfg["max_model_len"],
        max_num_seqs=cfg["batch_size"],
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        attention_backend="FLASH_ATTN",
        seed=cfg["seed"],
        speculative_config={
            "model": cfg["draft"],
            "revision": models["draft_revision"],
            "method": "draft_model",
            "num_speculative_tokens": cfg["draft_depth"],
        },
    )
    params = SamplingParams(
        temperature=0, max_tokens=cfg["max_output_tokens"], ignore_eos=True
    )
    trials = []
    for repeat in range(cfg["warmup_repeats"] + cfg["measured_repeats"]):
        order = ["uniform", "mixed"] if repeat % 2 == 0 else ["mixed", "uniform"]
        for mixture in order:
            trial = {
                "mixture": mixture,
                "repeat": repeat,
                "warmup": repeat < cfg["warmup_repeats"],
            }
            started = time.perf_counter()
            outputs = llm.generate(workload[mixture], params, use_tqdm=False)
            trial["elapsed_seconds"] = time.perf_counter() - started
            trial["request_ids"] = [out.request_id for out in outputs]
            trial["output_token_ids"] = [out.outputs[0].token_ids for out in outputs]
            assert all(
                len(ids) == cfg["max_output_tokens"]
                for ids in trial["output_token_ids"]
            )
            trials.append(trial)
            (args.output / "trials.json").write_text(json.dumps(trials, indent=2))
    print(f"Pilot completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
