#!/usr/bin/env python3
"""Relocate trusted Mach-O dependencies into a signed, self-contained directory."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(*args):
    return subprocess.check_output(list(map(str, args)), text=True, stderr=subprocess.STDOUT, timeout=60)


def version(value):
    parts = value.split(".")
    if not 1 <= len(parts) <= 3 or any(not part.isdecimal() for part in parts):
        raise ValueError(f"invalid macOS version: {value}")
    return tuple(map(int, parts)) + (0,) * (3 - len(parts))


def system_library(name):
    return name.startswith(("/usr/lib/", "/System/Library/")) and ".." not in Path(name).parts


def verify_loaded_libraries(log, directory):
    """Require dyld evidence and reject any loaded non-system file outside the package."""
    paths = re.findall(r"^dyld\[\d+\]: <[^>]+> (/.+)$", log, flags=re.MULTILINE)
    if not paths or not any(Path(path).parent == directory for path in paths):
        raise ValueError("missing packaged dyld loader evidence")
    for path in paths:
        if not system_library(path) and Path(path).parent != directory:
            raise ValueError(f"loaded library escaped the package: {path}")


def load_commands(text):
    """Read runtime search paths and the declared macOS deployment target."""
    rpaths, minimum = [], None
    blocks = text.split("Load command ")[1:]
    for block in blocks:
        lines = [line.strip() for line in block.splitlines()]
        if "cmd LC_RPATH" in lines:
            rpaths.append(next(line[5:].split(" (offset", 1)[0] for line in lines if line.startswith("path ")))
        if "cmd LC_BUILD_VERSION" in lines:
            if "platform 1" not in lines:
                raise ValueError("Mach-O platform is not macOS")
            minimum = next(line.split()[1] for line in lines if line.startswith("minos "))
        elif "cmd LC_VERSION_MIN_MACOSX" in lines:
            minimum = next(line.split()[1] for line in lines if line.startswith("version "))
    if minimum is None:
        raise ValueError("missing macOS deployment target")
    return rpaths, minimum


def inspect(path, arch, max_macos):
    if run("lipo", "-archs", path).split() != [arch]:
        raise ValueError(f"expected a single {arch} Mach-O: {path}")
    identity = run("otool", "-D", path).splitlines()[1:]
    identity = identity[0].strip() if identity else None
    dependencies = [line.strip().split(" (compatibility version", 1)[0]
                    for line in run("otool", "-L", path).splitlines()[1:]]
    rpaths, minimum = load_commands(run("otool", "-l", path))
    if version(minimum) > version(max_macos):
        raise ValueError(f"{path} requires macOS {minimum}, exceeds {max_macos}")
    return {"identity": identity, "dependencies": [name for name in dependencies if name != identity],
            "rpaths": rpaths, "minimum_macos": minimum}


def resolve_dependency(name, owner, directories):
    """Only explicit library roots may supply non-system dependencies."""
    if name.startswith("@loader_path/"):
        candidates = [owner.parent / name.removeprefix("@loader_path/")]
    elif name.startswith("@rpath/"):
        relative = name.removeprefix("@rpath/")
        if not relative or ".." in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError(f"unsafe runtime dependency: {name}")
        candidates = [directory / relative for directory in directories]
    elif name.startswith("/"):
        candidates = [Path(name)]
    else:
        raise ValueError(f"unsupported runtime dependency: {name}")
    found = {path.resolve() for path in candidates if path.is_file()}
    if len(found) != 1:
        raise ValueError(f"missing or ambiguous runtime dependency: {name}")
    path = found.pop()
    if not any(path.is_relative_to(directory) for directory in directories):
        raise ValueError(f"runtime dependency outside trusted roots: {path}")
    return path


def plan(inputs, directories, arch, max_macos):
    """Check the entire dependency closure before copying or modifying anything."""
    directories = [directory.resolve(strict=True) for directory in directories]
    pending = list(inputs)
    records, names = {}, {}
    while pending:
        path = pending.pop().resolve(strict=True)
        name = path.name
        previous = names.get(name.casefold())
        if previous:
            if previous != path:
                raise ValueError(f"runtime filename collision: {name}")
            continue
        if name.casefold() == "mach-o.json":
            raise ValueError("reserved runtime filename: mach-o.json")
        names[name.casefold()] = path
        record = inspect(path, arch, max_macos)
        replacements = {}
        for dependency in record["dependencies"]:
            if not system_library(dependency):
                target = resolve_dependency(dependency, path, directories)
                pending.append(target)
                replacements[dependency] = target.name
        records[name] = {**record, "source": str(path), "replacements": replacements}
    return records


def bundle(inputs, directories, output, arch, max_macos):
    """Publish only after path rewriting, ad-hoc signing and verification succeed."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    if not inputs:
        raise ValueError("no package inputs")
    records = plan(inputs, directories, arch, max_macos)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".macos-bundle-", dir=output.parent) as scratch:
        package = Path(scratch) / "package"
        package.mkdir()
        for name, record in records.items():
            source, target = Path(record["source"]), package / name
            shutil.copy2(source, target)
            target.chmod(target.stat().st_mode | 0o200)
            args = []
            if record["identity"]:
                args += ["-id", f"@loader_path/{name}"]
            for old, new in record["replacements"].items():
                args += ["-change", old, f"@loader_path/{new}"]
            for rpath in record["rpaths"]:
                if rpath != "@loader_path":
                    args += ["-delete_rpath", rpath]
            if "@loader_path" not in record["rpaths"]:
                args += ["-add_rpath", "@loader_path"]
            # Preserve the existing signature layout during rewriting; removing
            # it first can make Apple's linker reject an otherwise valid binary.
            if args:
                run("install_name_tool", *args, target)
            run("codesign", "--force", "--sign", "-", target)
            run("codesign", "--verify", "--strict", target)
            actual = inspect(target, arch, max_macos)
            expected = [f"@loader_path/{record['replacements'][dep]}" if dep in record["replacements"] else dep
                        for dep in record["dependencies"]]
            if (actual["dependencies"] != expected or actual["rpaths"] != ["@loader_path"]
                    or actual["identity"] != (f"@loader_path/{name}" if record["identity"] else None)):
                raise ValueError(f"runtime relocation verification failed: {name}")
            with source.open("rb") as stream:
                record["input_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            with target.open("rb") as stream:
                record["output_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        (package / "mach-o.json").write_text(json.dumps(records, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError(f"output already exists: {output}")
        package.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--library-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", choices=("arm64", "x86_64"), required=True)
    parser.add_argument("--max-macos", required=True)
    args = parser.parse_args()
    try:
        bundle(args.inputs, args.library_dir, args.output, args.arch, args.max_macos)
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
        parser.exit(1, f"macOS bundling failed: {error}\n")


if __name__ == "__main__":
    main()
