#!/bin/bash
#SBATCH --account=cis260267p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=adaspec-probe
#SBATCH --output=/jet/home/vdonde/projects/adaspec/results/probe-%j.log
set -euo pipefail
hostname
nvidia-smi
python_bin=/ocean/projects/cis260267p/vdonde/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3
"$python_bin" --version
ldd --version | head -n 1
