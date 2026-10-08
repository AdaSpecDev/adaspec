#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
checkout="$VLLM_CHECKOUT"
base_commit=0b3ba88f165976e77ca5e6a7a3f5bba4562b80af
if [[ ! -d "$checkout/.git" ]]; then
  git clone https://github.com/vllm-project/vllm.git "$checkout"
  git -C "$checkout" switch -c adaspec-pilot "$base_commit"
fi
if [[ ! -e "$ADASPEC_ROOT/../vllm" ]]; then
  ln -s "$checkout" "$ADASPEC_ROOT/../vllm"
fi
[[ "$(git -C "$checkout" rev-parse HEAD)" = "$base_commit" ]] || { echo 'Unexpected vLLM base revision'; exit 1; }
if [[ ! -d "$ADASPEC_STORAGE/.venv" ]]; then
  uv venv --python 3.12 "$ADASPEC_STORAGE/.venv"
fi
if [[ -f "$ADASPEC_ROOT/config/requirements.lock.txt" ]]; then
  uv pip sync --python "$ADASPEC_STORAGE/.venv/bin/python" "$ADASPEC_ROOT/config/requirements.lock.txt" --torch-backend=cu129
else
  uv pip install --python "$ADASPEC_STORAGE/.venv/bin/python" -r "$ADASPEC_ROOT/requirements.txt" --torch-backend=cu129
fi
uv pip install --python "$ADASPEC_STORAGE/.venv/bin/python" 'setuptools>=77,<81' setuptools-scm setuptools-rust wheel packaging jinja2 cmake ninja pre-commit
"$ADASPEC_STORAGE/.venv/bin/python" "$ADASPEC_ROOT/scripts/wheel_metadata.py" "$ADASPEC_ROOT/config/wheel.lock.json" > "$ADASPEC_STORAGE/wheel-url.txt"
if git -C "$checkout" apply --check "$ADASPEC_ROOT/patches/vllm-v0.22.0.patch"; then
  git -C "$checkout" apply "$ADASPEC_ROOT/patches/vllm-v0.22.0.patch"
else
  git -C "$checkout" apply --reverse --check "$ADASPEC_ROOT/patches/vllm-v0.22.0.patch"
fi
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$(cat "$ADASPEC_STORAGE/wheel-url.txt")"
uv pip install --python "$ADASPEC_STORAGE/.venv/bin/python" --no-build-isolation --no-deps --editable "$checkout"
(cd "$checkout" && "$ADASPEC_STORAGE/.venv/bin/pre-commit" install)
"$ADASPEC_STORAGE/.venv/bin/python" "$ADASPEC_ROOT/scripts/prepare_models.py" "$ADASPEC_ROOT"
