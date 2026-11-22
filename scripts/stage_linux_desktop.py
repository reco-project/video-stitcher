#!/usr/bin/env python3
"""Stage Linux application dependencies and notices for release validation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from bundle_linux import bundle
from linux_notices import collect

MANIFEST = Path(__file__).with_name("linux-desktop.json")


def stage(binary: Path, sdk: Path, ort_archive: Path, directories: list[Path], output: Path):
    """Publish staging only after every copied native dependency has an identity."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    sdk, binary = sdk.resolve(strict=True), binary.resolve(strict=True)
    manifest = json.loads(MANIFEST.read_text())
    ffmpeg = json.loads(Path(__file__).with_name("ffmpeg-sdk.json").read_text())
    provenance = sdk / "share/reco-ffmpeg"
    if json.loads((provenance / "ffmpeg-sdk.json").read_text()) != ffmpeg:
        raise ValueError("SDK recipe differs from the selected FFmpeg manifest")
    for archive, checksum in ((ort_archive, manifest["onnxruntime"]["sha256"]),
                              (provenance / f"ffmpeg-{ffmpeg['version']}.tar.xz", ffmpeg["sha256"])):
        with archive.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != checksum:
                raise ValueError(f"archive checksum mismatch: {archive.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".desktop-", dir=output.parent) as scratch:
        work = Path(scratch)
        version = manifest["onnxruntime"]["version"]
        ort_library = work / "libonnxruntime.so"
        with zipfile.ZipFile(ort_archive) as wheel:
            # The CPU wheel carries the standalone C runtime. Only the library and its notices are needed.
            with wheel.open(f"onnxruntime/capi/libonnxruntime.so.{version}") as library, ort_library.open("xb") as target:
                shutil.copyfileobj(library, target)
            ort_notices = work / "ort-notices"
            ort_notices.mkdir()
            for name in ("LICENSE", "ThirdPartyNotices.txt"):
                (ort_notices / name).write_bytes(wheel.read(f"onnxruntime/{name}"))
        roots = [binary, ort_library, sdk / "bin/ffmpeg", sdk / "bin/ffprobe"]
        package = work / "package"
        inputs = bundle(roots, [sdk / "lib", *directories], package, manifest["glibc_ceiling"])
        supplied = {binary: "reco", ort_library.resolve(): "onnxruntime"}
        supplied.update({path: "ffmpeg" for path in inputs.values() if path.is_relative_to(sdk)})
        notices = package / "notices"
        receipt = collect(inputs, supplied, notices)
        shutil.copytree(sdk / "share/reco-ffmpeg", notices / "ffmpeg")
        ort_notices.rename(notices / "onnxruntime")
        shutil.copy2(MANIFEST, notices)
        shutil.copy2(Path(__file__).resolve().parents[1] / "LICENSE", package / "LICENSE")
        shutil.copy2(Path(__file__).resolve().parents[1] / "THIRD_PARTY_NOTICES.md", package / "THIRD_PARTY_NOTICES.md")
        # Exact distribution sources must be acquired before publishing this staging.
        sources = sorted({(p["source_package"], p["source_version"]) for p in receipt["packages"].values()})
        (notices / "required-sources.json").write_text(json.dumps([
            {"package": name, "version": version} for name, version in sources
        ], indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError(f"output already exists: {output}")
        package.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--ort-archive", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    try:
        stage(args.binary, args.sdk, args.ort_archive, args.library_dir, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(1, f"Desktop staging failed: {error}\n")
