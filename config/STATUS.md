# Pilot readiness

- Prior Slurm hardware probe 47319885 completed successfully on w005 (H100 80GB).
- vLLM source pinned and version-specific opt-in patch prepared.
- Exact-token harness and strict trace validator implemented.
- CPU synthetic validation passed, including rejection of inconsistent KV lengths.
- Dependency installation, editable import and model downloads are in progress.
- GPU inference and live trace validation have not yet been executed.

Do not interpret any timing as a validated result until the GPU pilot's
`validation.json` reports `passed: true`.
