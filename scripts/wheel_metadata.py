"""Resolve the exact released binary artifact rather than a moving nightly."""

import json
from pathlib import Path
import sys
import urllib.request

with urllib.request.urlopen(
    "https://pypi.org/pypi/vllm/0.22.0/json", timeout=60
) as response:
    release = json.load(response)
files = [
    entry
    for entry in release["urls"]
    if entry["filename"].endswith("manylinux_2_28_x86_64.whl")
]
assert len(files) == 1, [entry["filename"] for entry in files]
entry = files[0]
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "url": entry["url"],
            "sha256": entry["digests"]["sha256"],
            "filename": entry["filename"],
        },
        indent=2,
    )
    + "\n"
)
print(entry["url"])
