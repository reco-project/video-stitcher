#!/usr/bin/env python3
"""Collect checksum-pinned Windows FFmpeg sources and their native notices."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.request

MANIFEST = Path(__file__).with_name("windows-sdk-sources.json")
FFMPEG = Path(__file__).with_name("ffmpeg-sdk.json")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    """Reject absolute paths, traversal and alternate Windows separators."""
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or "\\" in value or ":" in value or str(path) != value):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def recipe():
    """Use the shared FFmpeg pin alongside the Windows dependency pins."""
    manifest = json.loads(MANIFEST.read_text())
    ffmpeg = json.loads(FFMPEG.read_text())
    manifest["sources"].insert(0, {
        "name": "ffmpeg", "revision": ffmpeg["version"],
        "url": ffmpeg["source"], "archive": "ffmpeg.tar.xz",
        "sha256": ffmpeg["sha256"], "prefix": f"ffmpeg-{ffmpeg['version']}",
        "required_notices": ["COPYING.GPLv2", "COPYING.GPLv3", "LICENSE.md"],
    })
    return manifest


def validate(manifest):
    names, archives = set(), set()
    if not manifest.get("sources"):
        raise ValueError("source list is empty")
    for component in manifest["sources"]:
        name, archive = component["name"], component["archive"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or name in names:
            raise ValueError(f"invalid or duplicate component: {name}")
        if len(relative_path(archive).parts) != 1 or archive.casefold() in archives:
            raise ValueError(f"invalid or duplicate archive: {archive}")
        if not re.fullmatch(r"[0-9a-f]{64}", component["sha256"]):
            raise ValueError(f"missing checksum: {name}")
        if not component["url"].startswith("https://"):
            raise ValueError(f"source URL must use HTTPS: {name}")
        if component["prefix"]:
            relative_path(component["prefix"])
        if not component["required_notices"]:
            raise ValueError(f"required notices missing: {name}")
        for notice in component["required_notices"]:
            relative_path(notice)
        names.add(name)
        archives.add(archive.casefold())


def collect_notices(archive, component, output):
    """Read regular notice members only; never unpack source links or devices."""
    required = set(component["required_notices"])
    prefix = component["prefix"] + "/" if component["prefix"] else ""
    found, receipt = set(), []
    with tarfile.open(archive, "r|*") as source:
        for member in source:
            if member.isdir():
                continue
            if not member.name.startswith(prefix):
                raise ValueError(f"unexpected source root: {member.name}")
            name = member.name[len(prefix):]
            path = relative_path(name)
            if (name not in required
                    and not path.name.upper().startswith(("COPYING", "LICENSE", "NOTICE", "PATENTS", "DISCLAIMER"))):
                continue
            if (member.issym() or member.islnk()) and name not in required:
                continue
            if not member.isfile() or not 0 < member.size <= 16 * 1024 * 1024:
                raise ValueError(f"invalid notice member: {member.name}")
            if name.casefold() in found:
                raise ValueError(f"duplicate notice member: {member.name}")
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open("xb") as dest:
                shutil.copyfileobj(stream, dest)
            receipt.append({"path": name, "sha256": digest(target)})
            found.add(name.casefold())
    missing = {name for name in required if name.casefold() not in found}
    if missing:
        raise ValueError(f"missing notices for {component['name']}: {sorted(missing)}")
    return receipt


def collect(cache, output, manifest=None, offline=False):
    """Publish a fresh source directory only when every pin and notice passes."""
    manifest = recipe() if manifest is None else manifest
    validate(manifest)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    cache, output = cache.resolve(), output.resolve()
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError("cache and output must be separate directories")
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".windows-sources-", dir=output.parent) as scratch:
        stage = Path(scratch) / "sources"
        (stage / "archives").mkdir(parents=True)
        components = []
        for component in manifest["sources"]:
            archive = cache / component["archive"]
            if archive.is_symlink():
                raise ValueError(f"source cache entry is a symlink: {archive}")
            if not archive.exists():
                if offline:
                    raise ValueError(f"source unavailable offline: {archive.name}")
                download = Path(scratch) / "download"
                with urllib.request.urlopen(component["url"], timeout=60) as response, download.open("wb") as dest:
                    if not response.url.startswith("https://"):
                        raise ValueError("source download redirected away from HTTPS")
                    shutil.copyfileobj(response, dest)
                if digest(download) != component["sha256"]:
                    raise ValueError(f"source checksum mismatch: {component['name']}")
                # An exclusive copy does not replace a cache entry created by another job.
                with download.open("rb") as source, archive.open("xb") as dest:
                    shutil.copyfileobj(source, dest)
            if digest(archive) != component["sha256"]:
                raise ValueError(f"source checksum mismatch: {component['name']}")
            staged = stage / "archives" / archive.name
            shutil.copyfile(archive, staged)
            if digest(staged) != component["sha256"]:
                raise ValueError(f"staged source checksum mismatch: {component['name']}")
            notices = collect_notices(staged, component, stage / "notices" / component["name"])
            components.append({**component, "notices": notices})
        (stage / "sources.json").write_text(json.dumps({**manifest, "sources": components}, indent=2) + "\n")
        shutil.copy2(MANIFEST, stage)
        shutil.copy2(FFMPEG, stage)
        shutil.copy2(Path(__file__), stage)
        shutil.copy2(Path(__file__).with_name("build_windows_ffmpeg_sdk.py"), stage)
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        collect(args.cache, args.output, offline=args.offline)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Windows source collection failed: {error}\n")


if __name__ == "__main__":
    main()
