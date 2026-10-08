#!/bin/bash
export ADASPEC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ADASPEC_STORAGE="${ADASPEC_STORAGE:-/ocean/projects/cis260267p/vdonde/adaspec}"
export UV_CACHE_DIR="$ADASPEC_STORAGE/cache/uv"
export HF_HOME="$ADASPEC_STORAGE/cache/huggingface"
export XDG_CACHE_HOME="$ADASPEC_STORAGE/cache"
export TMPDIR="$ADASPEC_STORAGE/tmp"
export VLLM_CACHE_ROOT="$ADASPEC_STORAGE/cache/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$ADASPEC_STORAGE/runs"
export PATH="$ADASPEC_STORAGE/.venv/bin:$PATH"
export CARGO_HOME="$ADASPEC_STORAGE/cache/cargo"
export RUSTUP_HOME="$ADASPEC_STORAGE/cache/rustup"
export CARGO_BUILD_JOBS=2

export VLLM_CHECKOUT="${VLLM_CHECKOUT:-$ADASPEC_STORAGE/vllm}"
# Ensure spawned workers resolve the checkout even from its parent directory.
export PYTHONPATH="$VLLM_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
