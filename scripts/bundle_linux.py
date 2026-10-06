#!/usr/bin/env python3
"""Stage trusted ELF executables and their non-system dependencies for relocation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

# Keep the operating-system ABI and graphics driver stack on the target machine.
# All other DT_NEEDED libraries must resolve from explicitly supplied directories.
SYSTEM_LIBRARIES = frozenset({
    "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1",
    "libresolv.so.2", "libutil.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
    "libva.so.2", "libva-drm.so.2", "libva-x11.so.2", "libdrm.so.2",
    "libvulkan.so.1", "libcuda.so.1", "libnvidia-encode.so.1", "libnvcuvid.so.1",
    "libEGL.so.1", "libGL.so.1", "libGLX.so.0", "libGLdispatch.so.0",
})


def elf_identity(path: Path) -> bytes:
    """Compare class, byte order and machine without executing the inspected file."""
    with path.open("rb") as binary:
        header = binary.read(20)
    if len(header) != 20 or header[:4] != b"\x7fELF":
        raise ValueError(f"not an ELF file: {path}")
    return header[4:6] + header[18:20]


def inspect(option: str, path: Path) -> list[str]:
    output = subprocess.check_output(["patchelf", option, str(path)], text=True)
    return output.splitlines()


def basename(name: str) -> str:
    if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError(f"dependency must be a plain SONAME: {name!r}")
    return name


def resolve(name: str, directories: list[Path], identity: bytes) -> Path:
    """Do not consult LD_LIBRARY_PATH, the working directory or the loader cache."""
    for directory in directories:
        candidate = directory / basename(name)
        if candidate.is_file():
            if elf_identity(candidate) != identity:
                raise ValueError(f"wrong ELF architecture: {candidate}")
            soname = inspect("--print-soname", candidate)
            if soname != [name]:
                raise ValueError(f"SONAME mismatch for {candidate}: {soname}")
            return candidate.resolve()
    raise ValueError(f"unresolved dependency: {name}")


def glibc_references(path: Path) -> list[str]:
    """Inspect required version nodes, not versions exported by a shared library."""
    output = subprocess.check_output(
        ["readelf", "--version-info", "--wide", str(path)], text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    needs = output.partition("Version needs section")[2]
    return sorted(set(re.findall(r"Name: (GLIBC_\S+)", needs)))


def check_glibc(references: list[str], maximum: str, name: str):
    """Conservatively reject newer or private glibc version requirements."""
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", maximum):
        raise ValueError(f"invalid glibc ceiling: {maximum}")
    ceiling = tuple(map(int, maximum.split(".")))
    for reference in references:
        version = reference.removeprefix("GLIBC_")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version) or tuple(map(int, version.split("."))) > ceiling:
            raise ValueError(f"{name} references {reference}, beyond glibc ceiling {maximum}")


def bundle(executables: list[Path], directories: list[Path], output: Path, max_glibc: str | None = None):
    """Build in a sibling staging directory; never alter input files or overwrite output."""
    if sys.version_info < (3, 12):
        raise ValueError("Python 3.12 or newer is required")
    if sys.platform != "linux":
        raise ValueError("ELF bundling requires Linux")
    if not executables:
        raise ValueError("at least one executable is required")
    if max_glibc is not None:
        check_glibc([], max_glibc, "bundle")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    directories = [directory.resolve(strict=True) for directory in directories]
    identity = elf_identity(executables[0])
    inputs: dict[str, Path] = {}
    for executable in executables:
        name = basename(executable.name)
        if name == "bundle.json" or name in SYSTEM_LIBRARIES or name in inputs:
            raise ValueError(f"duplicate or reserved executable name: {name}")
        if elf_identity(executable) != identity:
            raise ValueError(f"wrong ELF architecture: {executable}")
        inputs[name] = executable.resolve(strict=True)
    pending = list(inputs.values())
    root_names = set(inputs)
    external = set()
    while pending:
        for name in inspect("--print-needed", pending.pop()):
            basename(name)
            if name == "bundle.json" or name in root_names:
                raise ValueError(f"dependency collides with a reserved output name: {name}")
            if name in SYSTEM_LIBRARIES:
                external.add(name)
            elif name not in inputs:
                inputs[name] = resolve(name, directories, identity)
                pending.append(inputs[name])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=output.parent) as staging:
        stage = Path(staging)
        hashes = {}
        versions = {}
        for name, source in sorted(inputs.items()):
            versions[name] = glibc_references(source)
            if max_glibc is not None:
                check_glibc(versions[name], max_glibc, name)
            target = stage / name
            shutil.copy2(source, target)
            subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(target)], check=True)
            with target.open("rb") as binary:
                hashes[name] = hashlib.file_digest(binary, "sha256").hexdigest()
        (stage / "bundle.json").write_text(json.dumps({
            "sha256": hashes, "system_dependencies": sorted(external),
            "glibc_references": versions, "glibc_ceiling": max_glibc,
        }, indent=2) + "\n")
        # rename preserves the original inputs and prevents partially published bundles.
        if output.exists() or output.is_symlink():
            raise ValueError(f"output already exists: {output}")
        stage.rename(output)
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-glibc", help="reject glibc references above this version (e.g. 2.39)")
    parser.add_argument("executables", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        bundle(args.executables, args.library_dir, args.output, args.max_glibc)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Bundle failed: {error}\n")


if __name__ == "__main__":
    main()
