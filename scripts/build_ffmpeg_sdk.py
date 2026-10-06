#!/usr/bin/env python3
"""Build the pinned Linux FFmpeg SDK without changing system installations."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

MANIFEST = Path(__file__).with_name("ffmpeg-sdk.json")


def build(work: Path, prefix: Path, jobs: int, archive: Path | None):
    """Use fresh output directories; failed builds are left available for diagnosis."""
    if sys.version_info < (3, 12):
        raise ValueError("Python 3.12 or newer is required")
    if sys.platform != "linux":
        raise ValueError("this SDK recipe requires Linux")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if work.exists() or prefix.exists() or work.is_symlink() or prefix.is_symlink():
        raise ValueError("work and prefix must be new directories")
    work, prefix = work.resolve(), prefix.resolve()
    if work in prefix.parents or prefix in work.parents or work == prefix:
        raise ValueError("work and prefix must be separate directories")
    manifest = json.loads(MANIFEST.read_text())
    work.mkdir(parents=True)
    if archive is None:
        archive = work / "source.tar.xz"
        with urllib.request.urlopen(manifest["source"], timeout=60) as response, archive.open("xb") as target:
            shutil.copyfileobj(response, target)
    with archive.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != manifest["sha256"]:
        raise ValueError("FFmpeg source checksum mismatch")
    for package, expected in manifest["linux_headers"].items():
        actual = subprocess.check_output(["pkg-config", "--modversion", package], text=True).strip()
        if actual != expected:
            raise ValueError(f"{package} headers: expected {expected}, found {actual}")
    with tarfile.open(archive) as source:
        source.extractall(work, filter="data")
    source = work / f"ffmpeg-{manifest['version']}"
    subprocess.run(
        [str(source / "configure"), f"--prefix={prefix}", *manifest["linux_configure"]],
        cwd=source, check=True,
    )
    subprocess.run(["make", f"-j{jobs}"], cwd=source, check=True)
    subprocess.run(["make", "install"], cwd=source, check=True)
    provenance = prefix / "share" / "reco-ffmpeg"
    provenance.mkdir(parents=True)
    (provenance / MANIFEST.name).write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(source / "ffbuild" / "config.mak", provenance)
    shutil.copy2(source / "COPYING.GPLv3", provenance)
    shutil.copy2(source / "COPYING.GPLv2", provenance)
    shutil.copy2(source / "LICENSE.md", provenance)
    shutil.copy2(archive, provenance / f"ffmpeg-{manifest['version']}.tar.xz")
    shutil.copy2(Path(__file__), provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--archive", type=Path, help="reuse the checksum-verified source archive")
    args = parser.parse_args()
    try:
        build(args.work_dir, args.prefix, args.jobs, args.archive)
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as error:
        parser.exit(1, f"SDK build failed: {error}\n")


if __name__ == "__main__":
    main()
