#!/usr/bin/env python3
"""Download exact distribution sources identified by Linux desktop staging."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def fetch(requirements: Path, output: Path):
    """Use configured deb-src indexes; never install packages or select newer sources."""
    packages = json.loads(requirements.read_text())
    for package in packages:
        if not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", package["package"]) or not re.fullmatch(r"[0-9][a-zA-Z0-9.+:~\-]*", package["version"]):
            raise ValueError("invalid source package identity")
    if not packages:
        raise ValueError("source package list is empty")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sources-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for package in packages:
            subprocess.run(["apt-get", "source", "--download-only", "--only-source",
                            f"{package['package']}={package['version']}"], cwd=stage, check=True, timeout=300)
        hashes = {}
        for path in sorted(stage.iterdir()):
            if path.is_file():
                with path.open("rb") as source:
                    hashes[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
        if not hashes:
            raise ValueError("no source artifacts were downloaded")
        (stage / "sources.json").write_text(json.dumps({"packages": packages, "sha256": hashes}, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError(f"output already exists: {output}")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        fetch(args.requirements, args.output)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Source download failed: {error}\n")
