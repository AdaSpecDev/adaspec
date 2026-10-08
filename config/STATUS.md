# Pilot readiness

- Prior Slurm hardware probe 47319885 completed on w005 (H100 80GB).
- Source pinned to vLLM v0.22.0, immutable base commit recorded in pilot.json.
- Isolated environment installed; editable source and instrumentation imports passed.
- Native dependency check passed; target and draft snapshots downloaded and pinned.
- Three CPU tests passed: valid/invalid batches, prepass snapshot semantics,
  and disabled instrumentation behavior. Ruff and shell syntax checks passed.
- Single-H100, 30-minute pilot submitted as Slurm job 48687832.
- GPU model startup and live trace validation are not yet verified.

The editable package's generated version can display 0.22.1.dev0; the source
base SHA and released 0.22.0 native wheel are pinned explicitly.

Do not interpret any timing as a validated result until the GPU pilot's
`validation.json` reports `passed: true`. Target-only output equivalence,
attention-kernel decomposition and instrumentation overhead remain later gates.
