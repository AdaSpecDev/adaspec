# AdaSpec

We study how the context lengths within a speculative decoding batch affect
verification cost. The research proposal is in [AdaSpec.pdf](AdaSpec.pdf).

**This PR sets up the environment only.** Batch inspection, an example runner,
and GPU measurements will be added after review.

## Get started on PSC

You need Git, `uv`, and access to your personal Bridges-2 project directory.

```bash
git clone https://github.com/AdaSpecDev/adaspec.git
cd adaspec
bash setup.sh
source /ocean/projects/cis260267p/$USER/adaspec/.venv/bin/activate
```

Setup downloads Python 3.12 if needed, installs the locked dependencies, checks
out the pinned [team vLLM fork](https://github.com/AdaSpecDev/vllm), and installs
it in editable mode with matching native libraries. It then checks imports and
dependency compatibility. You do not need a GPU to perform this setup.

Setup does not submit jobs or download models. The first install is several GB
and may take time: PSC's glibc requires building `llguidance` from source.

For another personal storage location:

```bash
export ADASPEC_STORAGE=/path/to/your/project/adaspec
bash setup.sh
source "$ADASPEC_STORAGE/.venv/bin/activate"
```

## Where things live

| Location | Contents |
| --- | --- |
| Your AdaSpec clone | Configuration, setup code, documentation, paper |
| `$ADASPEC_STORAGE/.venv` | Python environment |
| `$ADASPEC_STORAGE/vllm` | Editable checkout of the team fork |
| `$ADASPEC_STORAGE/cache` | Downloaded packages, native wheel, compiler caches |
| `$ADASPEC_STORAGE/python` | Python interpreter managed by `uv` |

Without an override, storage is `/ocean/projects/cis260267p/$USER/adaspec`.
Everything is in your personal space. Generated files stay outside the repo.

If setup finds an existing vLLM checkout with edits or a different `origin`, it
stops and explains the conflict. It does not discard your work. You can rerun
setup to repair or recreate the environment; it synchronizes packages to the lock.

## Files to read

| File | What it controls |
| --- | --- |
| `experiment.json` | Fork URL, exact source commit, matching native wheel, Python/PyTorch pins, and model revisions reserved for the next step |
| `setup.sh` | Small launcher that bootstraps Python and selects personal storage |
| `setup.py` | Checkout, installation, and verification steps |
| `requirements.txt` | Runtime/build requirements used to generate the lock |
| `requirements.lock.txt` | Generated exact versions; most entries are vLLM dependencies |
| `.gitignore` | Keeps local environments and generated files out of Git |

For now, the team fork is pinned to unmodified upstream v0.22.0. The inspector
PR will add ordinary commits to that fork; a follow-up here will update the pin.
There are no patch files or automatic patch application.

## Check your installation

After activating the environment:

```bash
python -c 'import vllm; print(vllm.__file__)'
```

The path should end in `adaspec/vllm/vllm/__init__.py` under your chosen storage.
Python edits there are picked up when a new process starts. Do not update the
source pin without checking native-wheel compatibility.

## Updating dependencies

`requirements.txt` describes the upstream runtime used for resolution and the
build tools needed for editable installation. Setup installs the fork itself,
not the upstream Python wheel. To regenerate the lock with the current tools:

```bash
uv pip compile requirements.txt --python-version 3.12 --torch-backend cu129 \
  --no-emit-package vllm --no-annotate --no-header \
  --output-file requirements.lock.txt
```

The command omits vLLM itself because setup installs the editable fork. Commit
dependency changes with the corresponding config changes and rerun setup. A change to the pinned base/native wheel needs a fresh
compatibility check.

## Next step

After this setup PR is reviewed, we will add a metadata-only batch inspector and
one offline inference example in the vLLM fork. Timing and controlled experiments
will follow separately. The old implementation remains in Git history at
`c447508`; its pilot was cancelled before it ran.
