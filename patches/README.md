# vLLM pilot instrumentation

Apply `vllm-v0.22.0.patch` only to upstream commit
`0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`.
The setup script verifies the base and recognizes an already-applied patch.

Changes by source file:

1. `vllm/v1/core/sched/output.py`: add optional metadata carried with the
   scheduler output to the worker.
2. `vllm/v1/core/sched/scheduler.py`: capture selected requests and actual block
   allocations before scheduled token counters advance; capture returned token
   counts and accepted draft counts during output processing.
3. `vllm/v1/worker/gpu_model_runner.py`: bracket the target transformer forward
   with CUDA events; synchronize and export the worker's live input lengths.
4. `vllm/v1/adaspec_trace.py`: implement the opt-in metadata capture, timing,
   JSONL records, and per-request acceptance feedback.

Only set `ADASPEC_TRACE_DIR` in diagnostic/pilot runs. With it absent, capture
and timing return immediately. The implementation targets synchronous TP=1
execution. It does not establish production-performance overhead, distributed
measurement validity, attention-kernel timing, or full verification latency.

The recorded acceptance count is the number of returned tokens minus the target
replacement/bonus token, following this vLLM version's scheduler logic. Terminal
output clipping can make final-step counts unsuitable for acceptance estimation.

After a Python edit, restart the Slurm process. After a base/native revision
change, rebuild the environment against matching native artifacts.
