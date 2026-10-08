import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, snapshot_download

root = Path(sys.argv[1])
cfg = json.loads((root / "config/pilot.json").read_text())
lock = root / "config/models.lock.json"
models = json.loads(lock.read_text()) if lock.exists() else {}
for role in ("target", "draft"):
    repo = cfg[role]
    revision = models.get(role + "_revision") or HfApi().model_info(repo).sha
    models[role + "_revision"] = revision
    models[role + "_repo"] = repo
    lock.write_text(json.dumps(models, indent=2) + "\n")
    snapshot_download(
        repo,
        revision=revision,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.tiktoken",
            "*.jinja",
        ],
    )
print(lock)
