# AdaSpec: controlled live-batch pilot

First milestone: one H100 run demonstrating that recorded target passes contain
our intended full speculative batches. This is a measurement validation pilot,
not evidence for the heterogeneity hypothesis yet.

## Pinned choices

- vLLM v0.22.0, commit `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`.
- Python 3.12; released CUDA 12.9 binary, matching PyTorch dependency.
- Qwen/Qwen3-8B target and Qwen/Qwen3-0.6B draft, TP=1, BF16.
  Model commits are recorded in `config/models.lock.json` before inference.
- FlashAttention, eager execution, synchronous scheduling, prefix caching off.
- B=8, K=3; 4096 x 8 versus 1024 x 4 + 7168 x 4 prompt tokens.
- Greedy decoding, ignore EOS, 96 output tokens; one warmup and three repeats.
  Mixture order alternates. Prompts are saved as exact token IDs.

Eager/synchronous execution simplifies live-state auditing. CUDA synchronization
around every transformer forward perturbs execution; production throughput and
controller benefit must be measured later with lighter instrumentation.

## Setup and run

`requirements.txt` lists the direct dependencies. The resolved dependency lock
is saved under `config/requirements.lock.txt` after installation. The setup script
also installs our editable source patch and downloads the model weights.

All caches, weights, environments and large run artifacts live under the user's
personal PSC project directory. The experiment harness stays in the home workspace. The vLLM checkout also
lives in project storage because editable installation extracts large native
libraries there; a workspace symlink preserves the VS Code path.

```bash
bash scripts/setup.sh
sbatch slurm/pilot.sbatch
```

`setup.sh` creates a project-storage vLLM checkout if needed, verifies its base commit,
applies the version-specific patch, installs matching released native binaries
with editable Python, and prepares pinned model snapshots. Reinstallation is
required after changing the native base; Python edits require restarting the job.
No GitHub vLLM fork is required to reproduce the pilot: the base SHA and patch
are both recorded here. A separate fork can be added when integration expands.

Run artifacts are at `/ocean/projects/cis260267p/vdonde/adaspec/runs/pilot-JOBID`:

- `manifest.json`, `environment.txt`, `vllm.patch`, hardware details.
- `workloads.json`: exact prompt token IDs; `trials.json`: request/output mapping.
- `traces/target-*.jsonl`: per-target-pass records including prefill.
- `validation.json`: fail-closed validation; `features.csv`: qualifying passes.

Inspect the trace yourself:

```bash
source scripts/env.sh
python scripts/analyze.py "$ADASPEC_STORAGE/runs/pilot-JOBID"
head -n 2 "$ADASPEC_STORAGE/runs/pilot-JOBID"/traces/target-*.jsonl
```

## What is measured

`cached_tokens` is the scheduler's already computed token count before the pass;
`worker_cached_tokens` independently records the worker's view.
`query_tokens` includes the new target input and proposed tokens;
`attention_sequence_tokens` is worker cached length plus this query.
`allocated_blocks_by_group` counts actual allocated blocks, including lookahead;
it is not an estimate from sequence length and does not equal occupied KV tokens.

CUDA events bracket **target transformer forward only**, excluding logits,
rejection sampling, drafting and scheduling. Separate feedback traces record
accepted draft counts and scheduler-capture-to-feedback wall time; this wall
interval excludes scheduler construction and is not full decode-step timing. This pilot does not yet isolate
attention kernels or measure a complete verification/step boundary. Trial elapsed
time is an instrumented workload duration, not production serving latency.

Validation requires at least five full, pure decode passes per trial, exact
prompt distributions, no preemption, matching scheduler/worker lengths, and K
actual draft tokens with K+1 query tokens per request. Startup and tail passes
with fewer drafts or requests are excluded and counted. Live means drift as
acceptance commits different lengths: CSV rows expose this, and unadjusted
mixture timing summaries must not be treated as equal-mean comparisons.

## Source map

- `vllm/v1/core/sched/scheduler.py`: scheduling and actual KV block allocation.
- `vllm/v1/core/sched/output.py`: metadata carried to the worker.
- `vllm/v1/worker/gpu_model_runner.py`: target forward boundary.
- `vllm/v1/adaspec_trace.py`: opt-in capture (`ADASPEC_TRACE_DIR`).

Instrumentation is disabled when the environment variable is absent. The pilot
runs only on one GPU, so the schema makes no claims about distributed timing.

## Remaining gates before scientific sweeps

1. Successful environment import, model startup and live-batch validation on GPU.
2. Greedy target-only versus speculative output comparison; validate acceptance feedback.
3. Full verification boundary, attention profiler decomposition and overhead check.
4. Equal-live-mean matching/control, repeated trials and held-out mixture analysis.
5. SWE-bench content with documented prompt construction and tokenizer lengths.

AI assistance was used to prepare this research code. No upstream PR is created.
