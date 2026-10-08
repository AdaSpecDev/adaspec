"""Install the pinned team fork. No models are downloaded or GPU jobs submitted."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parent


def run(*command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def git(checkout, *args):
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True
    ).strip()


def prepare_checkout(checkout, config):
    """Reuse only a clean checkout of the configured repository."""
    if checkout.exists():
        if not (checkout / ".git").is_dir():
            raise RuntimeError(f"{checkout} exists but is not a Git checkout")
        if git(checkout, "status", "--porcelain"):
            raise RuntimeError(f"{checkout} has local edits; commit or move them first")
        actual = (
            git(checkout, "remote", "get-url", "origin")
            .rstrip("/")
            .removesuffix(".git")
        )
        expected = config["repository"].rstrip("/").removesuffix(".git")
        if actual != expected:
            raise RuntimeError(f"{checkout} belongs to {actual}; expected {expected}")
        run("git", "-C", checkout, "fetch", "origin")
    else:
        run("git", "clone", config["repository"], checkout)
    run("git", "-C", checkout, "checkout", "--detach", config["commit"])
    run(
        "git",
        "-C",
        checkout,
        "merge-base",
        "--is-ancestor",
        config["base_commit"],
        config["commit"],
    )


def download_wheel(cache, config):
    """Cache and verify the matching native artifact before installing it."""
    wheel = cache / config["url"].rsplit("/", 1)[-1]
    if not wheel.exists():
        partial = wheel.with_suffix(".partial")
        print(f"Downloading native wheel: {wheel.name}", flush=True)
        with urllib.request.urlopen(config["url"], timeout=120) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        partial.rename(wheel)
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != config["sha256"]:
        raise RuntimeError(
            f"Native wheel checksum mismatch: {wheel}; remove it and retry"
        )
    return wheel


def main():
    config = json.loads((REPO / "experiment.json").read_text())["software"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != config["python"]:
        raise RuntimeError(f"Expected Python {config['python']}, got {actual_python}")
    storage = Path(os.environ["ADASPEC_STORAGE"]).expanduser().resolve()
    cache = storage / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name, directory in {
        "UV_CACHE_DIR": cache / "uv",
        "TMPDIR": storage / "tmp",
        "CARGO_HOME": cache / "cargo",
        "RUSTUP_HOME": cache / "rustup",
    }.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ["CARGO_BUILD_JOBS"] = "2"
    checkout = storage / "vllm"
    prepare_checkout(checkout, config["vllm"])
    wheel = download_wheel(cache, config["vllm"]["native_wheel"])
    environment = storage / ".venv"
    if not environment.exists():
        run("uv", "venv", "--python", sys.executable, environment)
    python = environment / "bin/python"
    environment_version = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()
    if environment_version != config["python"]:
        raise RuntimeError(
            f"{environment} uses Python {environment_version}; expected {config['python']}"
        )
    run(
        "uv",
        "pip",
        "sync",
        "--python",
        python,
        REPO / "requirements.lock.txt",
        "--torch-backend",
        config["torch_backend"],
    )
    os.environ["VLLM_USE_PRECOMPILED"] = "1"
    os.environ["VLLM_PRECOMPILED_WHEEL_LOCATION"] = str(wheel)
    run(
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "--no-build-isolation",
        "--no-deps",
        "--editable",
        checkout,
        "--config-settings",
        "editable_mode=compat",
    )
    run("uv", "pip", "check", "--python", python)
    run(
        python,
        "-c",
        "import pathlib, sys, torch, vllm; "
        "assert pathlib.Path(vllm.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1])); "
        "assert torch.__version__ == sys.argv[2]; "
        'print("vLLM source:", vllm.__file__); print("PyTorch:", torch.__version__)',
        checkout,
        config["torch_version"],
        cwd=storage,
    )
    print(f"\nSetup complete. Activate with:\nsource {environment}/bin/activate")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Setup failed: {error}") from error
