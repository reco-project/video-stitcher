#!/usr/bin/env python3
"""Build the pinned Intel macOS ONNX Runtime with upstream's CMake recipe."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from bundle_macos import inspect
from fetch_windows_sources import collect, digest

MANIFEST = Path(__file__).with_name("macos-desktop.json")
MINIMUM_MACOS = "14.0"


def build(cache, work, output, jobs=2):
    """Publish only a complete x86-64 runtime; keep logs and source provenance."""
    if sys.platform != "darwin" or sys.version_info < (3, 12) or jobs < 1:
        raise ValueError("requires macOS, Python 3.12+ and positive jobs")
    for path in (work, output):
        if path.exists() or path.is_symlink():
            raise ValueError("work and output must be new directories")
    cache, work, output = cache.resolve(), work.resolve(), output.resolve()
    for left, right in ((cache, work), (cache, output), (work, output)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("cache, work and output must be separate directories")
    runtime = json.loads(MANIFEST.read_text())["onnxruntime"]
    pinned = runtime["source"]
    collect(cache, work / "sources", {"sources": [pinned]}, support_files=(
        MANIFEST, Path(__file__), Path(__file__).with_name("fetch_windows_sources.py"),
        Path(__file__).with_name("bundle_macos.py")))
    with tarfile.open(work / "sources/archives" / pinned["archive"]) as archive:
        archive.extractall(work / "source", filter="data")
    source = work / "source" / pinned["prefix"]
    if (source / "VERSION_NUMBER").read_text().strip() != runtime["version"]:
        raise ValueError("ONNX source version differs from manifest")
    env = {key: value for key, value in os.environ.items() if key not in (
        "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LIBRARY_PATH", "CPATH",
        "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "SDKROOT", "CMAKE_PREFIX_PATH")
        and not key.startswith(("CMAKE_", "DYLD_", "LD_", "PKG_CONFIG"))}
    env.update(MACOSX_DEPLOYMENT_TARGET=MINIMUM_MACOS, CC="/usr/bin/clang", CXX="/usr/bin/clang++")
    command = [sys.executable, str(source / "tools/ci_build/build.py"),
        "--update", "--build", "--build_dir", str(work / "build"), "--config", "Release",
        "--build_shared_lib", "--parallel", str(jobs), "--skip_tests", "--skip_submodule_sync",
        "--skip_pip_install", "--cmake_generator", "Ninja", "--osx_arch", "x86_64",
        "--compile_no_warning_as_error", "--cmake_extra_defines",
        f"CMAKE_OSX_DEPLOYMENT_TARGET={MINIMUM_MACOS}", "onnxruntime_BUILD_UNIT_TESTS=OFF",
        "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER", "CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"]
    (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (work / "build.log").open("xb") as log:
        subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    library = work / "build/Release" / f"libonnxruntime.{runtime['version']}.dylib"
    info = inspect(library, "x86_64", MINIMUM_MACOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ort-", dir=output.parent) as scratch:
        stage = Path(scratch) / "runtime"
        (stage / "lib").mkdir(parents=True)
        # ort loads this unversioned filename dynamically.
        target = stage / "lib/libonnxruntime.dylib"
        shutil.copy2(library, target)
        for name in ("LICENSE", "ThirdPartyNotices.txt", "VERSION_NUMBER"):
            shutil.copy2(source / name, stage / name)
        (stage / "GIT_COMMIT_ID").write_text(pinned["revision"] + "\n")
        shutil.copytree(work / "sources", stage / "sources")
        receipt = {"architecture": "x86_64", "minimum_macos": MINIMUM_MACOS,
                   "command": command, "mach_o": info, "sha256": digest(target),
                   "dependencies_sha256": digest(source / "cmake/deps.txt"),
                   "compiler": subprocess.check_output(["/usr/bin/clang", "--version"], text=True)}
        (stage / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError("output already exists")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    try:
        build(args.source_cache, args.work_dir, args.output, args.jobs)
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Intel ONNX Runtime build failed: {error}\n")
