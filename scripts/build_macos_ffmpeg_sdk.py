#!/usr/bin/env python3
"""Build a pinned macOS FFmpeg SDK and retain its matching codec sources."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile

from fetch_windows_sources import MANIFEST, FFMPEG, collect, digest, recipe
from fetch_windows_sources import relative_path, verify_sources

MINIMUM_MACOS = "14.0"
CODECS = ("x264", "x265", "aom", "svt-av1", "dav1d", "ffmpeg")
CONFIGURE = [
    "--cc=/usr/bin/clang", "--cxx=/usr/bin/clang++",
    "--disable-autodetect", "--disable-debug", "--disable-doc", "--disable-ffplay",
    "--enable-shared", "--disable-static", "--enable-gpl", "--enable-version3", "--enable-pthreads",
    "--enable-securetransport", "--enable-zlib", "--enable-bzlib", "--enable-iconv",
    "--enable-videotoolbox", "--enable-audiotoolbox", "--enable-avfoundation",
    "--enable-libx264", "--enable-libx265", "--enable-libaom", "--enable-libdav1d", "--enable-libsvtav1",
    # With autodetection disabled, FFmpeg probes iconv only in libc, whereas
    # macOS supplies it in a separate system library.
    "--extra-libs=-liconv",
]


def native_recipe():
    """Reuse the existing verified codec pins and common FFmpeg source release."""
    sources = {item["name"]: item for item in recipe()["sources"]}
    return {"minimum_macos": MINIMUM_MACOS, "sources": [sources[name] for name in CODECS]}


def verify_sdk(sdk, arch):
    """Check source/notices, build identity and every receipt-named SDK output."""
    provenance = sdk / "share/reco-ffmpeg"
    verify_sources(provenance, native_recipe())
    receipt = json.loads((provenance / "build.json").read_text())
    configurations = [item["args"] for item in receipt["commands"] if item["label"] == "ffmpeg-configure"]
    if len(configurations) != 1 or len(configurations[0]) < 3:
        raise ValueError("SDK is missing its configure command")
    command = configurations[0]
    prefix = command[1].removeprefix("--prefix=")
    expected = [command[0], f"--prefix={prefix}", *CONFIGURE, f"--extra-ldflags=-Wl,-rpath,{prefix}/lib"]
    if (receipt["architecture"] != arch or receipt["minimum_macos"] != MINIMUM_MACOS
            or not Path(prefix).is_absolute() or command != expected):
        raise ValueError("SDK build configuration differs from the selected recipe")
    required = {"bin/ffmpeg", "bin/ffprobe", "include/libavcodec/avcodec.h"}
    required.update(f"lib/lib{name}.dylib" for name in (
        "avcodec", "avdevice", "avfilter", "avformat", "avutil", "swresample", "swscale",
        "x264", "x265", "aom", "SvtAv1Enc", "dav1d"))
    if not required <= receipt["files"].keys():
        raise ValueError("SDK is missing required native outputs")
    for name, checksum in receipt["files"].items():
        path = sdk / relative_path(name)
        if not path.resolve().is_relative_to(sdk.resolve()) or digest(path) != checksum:
            raise ValueError(f"SDK checksum mismatch: {name}")
    return receipt


def build(cache, work, prefix, jobs=2, offline=False):
    if sys.platform != "darwin" or sys.version_info < (3, 12):
        raise ValueError("the native build requires macOS and Python 3.12 or newer")
    if platform.machine() not in ("arm64", "x86_64") or jobs < 1:
        raise ValueError("unsupported architecture or nonpositive jobs")
    for path in (work, prefix):
        if path.exists() or path.is_symlink():
            raise ValueError("work and prefix must be new directories")
    cache, work, prefix = cache.resolve(), work.resolve(), prefix.resolve()
    for left, right in ((work, prefix), (cache, work), (cache, prefix)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("cache, work and prefix must be separate directories")
    env = {key: value for key, value in os.environ.items() if key not in (
        "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LIBRARY_PATH", "CPATH",
        "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "SDKROOT", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR")
        and not key.startswith(("CMAKE_", "DYLD_", "LD_"))}
    env.update(MACOSX_DEPLOYMENT_TARGET=MINIMUM_MACOS, CC="/usr/bin/clang", CXX="/usr/bin/clang++",
               PKG_CONFIG_LIBDIR=str(prefix / "lib/pkgconfig"), PKG_CONFIG_PATH="")
    support = (MANIFEST, FFMPEG, Path(__file__).with_name("fetch_windows_sources.py"), Path(__file__))
    collect(cache, work / "sources", native_recipe(), offline, support_files=support)
    pinned = json.loads((work / "sources/sources.json").read_text())
    commands = []

    def run(label, args, cwd=work):
        command = list(map(str, args))
        commands.append({"label": label, "args": command, "cwd": str(cwd)})
        print(label, flush=True)
        with (work / f"{label}.log").open("xb") as log:
            subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def cmake(name, source, options, install=True):
        directory = work / f"build-{name}"
        run(f"{name}-configure", ["cmake", "-S", source, "-B", directory, "-G", "Ninja",
            f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={MINIMUM_MACOS}", "-DCMAKE_INSTALL_LIBDIR=lib", *options])
        run(f"{name}-build", ["cmake", "--build", directory, "--parallel", jobs])
        if install:
            run(f"{name}-install", ["cmake", "--install", directory])
        return directory

    sources = {}
    for component in pinned["sources"]:
        directory = work / component["name"]
        directory.mkdir()
        with tarfile.open(work / "sources/archives" / component["archive"]) as archive:
            archive.extractall(directory, filter="data")
        sources[component["name"]] = directory / component["prefix"] if component["prefix"] else directory
    try:
        run("x264-configure", [sources["x264"] / "configure", f"--prefix={prefix}",
            "--enable-shared", "--disable-cli", "--disable-opencl"], sources["x264"])
        run("x264-build", ["make", f"-j{jobs}"], sources["x264"])
        run("x264-install", ["make", "install"], sources["x264"])
        high = []
        for depth in (10, 12):
            directory = cmake(f"x265-{depth}", sources["x265"] / "source",
                ["-DHIGH_BIT_DEPTH=ON", "-DEXPORT_C_API=OFF", "-DENABLE_SHARED=OFF", "-DENABLE_CLI=OFF",
                 "-DENABLE_PIC=ON", "-DNATIVE_BUILD=OFF", f"-DMAIN12={'ON' if depth == 12 else 'OFF'}"], install=False)
            high.append(str(directory / "libx265.a"))
        cmake("x265-8", sources["x265"] / "source", ["-DENABLE_SHARED=ON", "-DENABLE_CLI=OFF",
            "-DNATIVE_BUILD=OFF", "-DLINKED_10BIT=ON", "-DLINKED_12BIT=ON", "-DEXTRA_LIB=" + ";".join(high)])
        cmake("aom", sources["aom"], ["-DBUILD_SHARED_LIBS=ON", "-DENABLE_TESTS=OFF",
            "-DENABLE_TOOLS=OFF", "-DENABLE_EXAMPLES=OFF", "-DENABLE_DOCS=OFF"])
        cmake("svt-av1", sources["svt-av1"], ["-DBUILD_SHARED_LIBS=ON", "-DBUILD_TESTING=OFF",
            "-DBUILD_APPS=OFF", "-DNATIVE=OFF", "-DSVT_AV1_LTO=OFF"])
        davbuild = work / "build-dav1d"
        run("dav1d-configure", ["meson", "setup", davbuild, sources["dav1d"], f"--prefix={prefix}",
            "--libdir=lib", "--buildtype=release", "--default-library=shared", "-Denable_tools=false",
            "-Denable_tests=false", "-Denable_examples=false", "--wrap-mode=nodownload"])
        run("dav1d-build", ["meson", "compile", "-C", davbuild, "-j", jobs])
        run("dav1d-install", ["meson", "install", "-C", davbuild])
        ffbuild = work / "build-ffmpeg"
        ffbuild.mkdir()
        run("ffmpeg-configure", [sources["ffmpeg"] / "configure", f"--prefix={prefix}", *CONFIGURE,
            f"--extra-ldflags=-Wl,-rpath,{prefix}/lib"], ffbuild)
        run("ffmpeg-build", ["make", f"-j{jobs}"], ffbuild)
        run("ffmpeg-install", ["make", "install"], ffbuild)
        provenance = prefix / "share/reco-ffmpeg"
        shutil.copytree(work / "sources", provenance)
        shutil.copy2(ffbuild / "ffbuild/config.mak", provenance)
        receipt = {
            "architecture": platform.machine(), "minimum_macos": MINIMUM_MACOS, "commands": commands,
            "compiler": subprocess.check_output(["/usr/bin/clang", "--version"], text=True),
            "sdk": subprocess.check_output(["xcrun", "--show-sdk-version"], text=True).strip(),
            "tools": {tool: subprocess.check_output([tool, "--version"], env=env, text=True).strip()
                      for tool in ("cmake", "meson", "ninja", "pkg-config")},
            "files": {path.relative_to(prefix).as_posix(): digest(path)
                      for directory in (prefix / "bin", prefix / "lib", prefix / "include")
                      for path in sorted(directory.rglob("*")) if path.is_file()},
        }
        (provenance / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        (work / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        build(args.source_cache, args.work_dir, args.prefix, args.jobs, args.offline)
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as error:
        parser.exit(1, f"macOS SDK build failed: {error}\n")


if __name__ == "__main__":
    main()
