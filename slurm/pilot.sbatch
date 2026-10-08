#!/bin/bash
#SBATCH --account=cis260267p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=adaspec-pilot
#SBATCH --output=/jet/home/vdonde/projects/adaspec/results/pilot-%j.log
set -euo pipefail
source /jet/home/vdonde/projects/adaspec/scripts/env.sh
run_dir="$ADASPEC_STORAGE/runs/pilot-$SLURM_JOB_ID"
mkdir -p "$run_dir"
export ADASPEC_TRACE_DIR="$run_dir/traces"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi > "$run_dir/nvidia-smi.txt"
uv pip freeze --python "$ADASPEC_STORAGE/.venv/bin/python" > "$run_dir/environment.txt"
cp "$ADASPEC_ROOT/patches/vllm-v0.22.0.patch" "$run_dir/vllm.patch"
git -C "$ADASPEC_ROOT" rev-parse HEAD > "$run_dir/experiment-commit.txt"
git -C "$ADASPEC_ROOT" diff HEAD > "$run_dir/experiment-dirty.patch"
"$ADASPEC_STORAGE/.venv/bin/python" "$ADASPEC_ROOT/scripts/pilot.py" --config "$ADASPEC_ROOT/config/pilot.json" --models "$ADASPEC_ROOT/config/models.lock.json" --output "$run_dir"
"$ADASPEC_STORAGE/.venv/bin/python" "$ADASPEC_ROOT/scripts/analyze.py" "$run_dir"
