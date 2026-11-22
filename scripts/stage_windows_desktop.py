#!/usr/bin/env python3
"""Stage a Windows desktop application with pinned native runtimes and notices."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from fetch_windows_sources import digest, recipe, relative_path

MANIFEST = Path(__file__).with_name("windows-desktop.json")
LIBRARIES = ("avcodec", "avdevice", "avfilter", "avformat", "avutil", "swresample", "swscale")


def verify_sdk(sdk):
    """Reject a different recipe, incomplete SDK, or changed receipt-named bytes."""
    provenance = sdk / "share/reco-ffmpeg"
    sources = json.loads((provenance / "sources.json").read_text())
    expected = recipe()
    components = [{key: value for key, value in item.items() if key != "notices"}
                  for item in sources["sources"]]
    if {**sources, "sources": components} != expected:
        raise ValueError("SDK sources differ from the selected recipe")
    receipt = json.loads((provenance / "build.json").read_text())
    if receipt["toolchain_image"] != expected["toolchain_image"]:
        raise ValueError("SDK toolchain differs from the selected recipe")
    files = receipt["files"]
    for library in LIBRARIES:
        if (f"lib/{library}.lib" not in files
                or len([name for name in files if name.startswith(f"bin/{library}-") and name.endswith(".dll")]) != 1):
            raise ValueError(f"SDK missing import library or runtime: {library}")
    for name in ("bin/ffmpeg.exe", "bin/ffprobe.exe", "include/libavcodec/avcodec.h"):
        if name not in files:
            raise ValueError(f"SDK missing required file: {name}")
    for name, checksum in files.items():
        path = sdk / relative_path(name)
        if path.is_symlink() or digest(path) != checksum:
            raise ValueError(f"SDK checksum mismatch: {name}")
    for component in sources["sources"]:
        notices = component["notices"]
        if not set(component["required_notices"]) <= {item["path"] for item in notices}:
            raise ValueError(f"SDK missing required notices: {component['name']}")
        for notice in notices:
            path = provenance / "notices" / component["name"] / relative_path(notice["path"])
            if path.is_symlink() or digest(path) != notice["sha256"]:
                raise ValueError(f"SDK notice checksum mismatch: {path}")
    return receipt


def runtime_archive(cache, name, component, offline):
    """Download only missing archives; never trust unverified cached bytes."""
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name}-{component['version']}.nupkg"
    if target.is_symlink():
        raise ValueError(f"runtime cache entry is a symlink: {target}")
    if not target.exists():
        if offline:
            raise ValueError(f"runtime unavailable offline: {name}")
        with tempfile.TemporaryDirectory(prefix=".runtime-", dir=cache) as scratch:
            download = Path(scratch) / "download"
            with urllib.request.urlopen(component["url"], timeout=60) as source, download.open("wb") as dest:
                if not source.url.startswith("https://"):
                    raise ValueError("runtime download redirected away from HTTPS")
                shutil.copyfileobj(source, dest)
            if digest(download) != component["sha256"]:
                raise ValueError(f"runtime checksum mismatch: {name}")
            with download.open("rb") as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest)
    if digest(target) != component["sha256"]:
        raise ValueError(f"runtime checksum mismatch: {name}")
    return target


def stage(binary, sdk, cache, output, offline=False):
    """Publish a fresh package only after required runtime and notice files pass."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    receipt = verify_sdk(sdk)
    manifest = json.loads(MANIFEST.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".windows-desktop-", dir=output.parent) as scratch:
        package = Path(scratch) / "package"
        package.mkdir()
        shutil.copy2(binary, package / binary.name)
        for name in receipt["files"]:
            if name.startswith("bin/") and name.endswith((".dll", ".exe")):
                shutil.copy2(sdk / name, package / Path(name).name)
        shutil.copytree(sdk / "share/reco-ffmpeg", package / "notices/ffmpeg")
        for name, component in manifest.items():
            with zipfile.ZipFile(runtime_archive(cache, name, component, offline)) as archive:
                for member in component["libraries"] + component["notices"]:
                    info = archive.getinfo(member)
                    if info.is_dir() or not 0 < info.file_size <= 128 * 1024 * 1024:
                        raise ValueError(f"invalid runtime member: {member}")
                    target = (package / Path(member).name if member in component["libraries"]
                              else package / "notices" / name / relative_path(member))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("xb") as dest:
                        shutil.copyfileobj(source, dest)
        shutil.copy2(MANIFEST, package / "notices")
        shutil.copy2(Path(__file__).resolve().parents[1] / "LICENSE", package / "LICENSE")
        shutil.copy2(Path(__file__).resolve().parents[1] / "THIRD_PARTY_NOTICES.md", package / "THIRD_PARTY_NOTICES.md")
        (package / "files.json").write_text(json.dumps({
            path.relative_to(package).as_posix(): digest(path)
            for path in sorted(package.rglob("*")) if path.is_file()
        }, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError(f"output already exists: {output}")
        package.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--runtime-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("binary", type=Path, nargs="?")
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_sdk(args.sdk)
        else:
            if not all((args.binary, args.output, args.runtime_cache)):
                parser.error("staging requires binary, --output and --runtime-cache")
            stage(args.binary, args.sdk, args.runtime_cache, args.output, args.offline)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Windows desktop staging failed: {error}\n")


if __name__ == "__main__":
    main()
