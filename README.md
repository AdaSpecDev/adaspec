# AdaSpec

Requires Git, `uv`, and access to your personal PSC project storage.

```bash
git clone https://github.com/AdaSpecDev/adaspec.git
cd adaspec
bash setup.sh
source /ocean/projects/cis260267p/$USER/adaspec/.venv/bin/activate
```

`setup.sh` installs Python, locked dependencies, and an editable checkout of the
[team vLLM fork](https://github.com/AdaSpecDev/vllm). Versions are pinned in
`experiment.json` and `requirements.lock.txt`. The first install can take time.

To use another storage location:

```bash
export ADASPEC_STORAGE=/path/to/your/project/adaspec
bash setup.sh
source "$ADASPEC_STORAGE/.venv/bin/activate"
```

Verify the installation:

```bash
python -c 'import vllm; print(vllm.__file__)'
```

The path should point to `$ADASPEC_STORAGE/vllm/vllm/__init__.py` (the default
storage is `/ocean/projects/cis260267p/$USER/adaspec`). Edit Python code there,
then restart your Python process to use the changes.
