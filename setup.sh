#!/usr/bin/env bash
# Bootstrap Python; setup.py contains the installation steps.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ADASPEC_STORAGE="${ADASPEC_STORAGE:-/ocean/projects/cis260267p/$USER/adaspec}"
export UV_CACHE_DIR="$ADASPEC_STORAGE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ADASPEC_STORAGE/python"
exec uv run --no-project --python 3.12 "$repo_dir/setup.py" "$@"
