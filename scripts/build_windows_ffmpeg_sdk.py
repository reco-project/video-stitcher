#!/usr/bin/env python3
"""Build the Windows SDK inside the digest-pinned Linux cross-toolchain image."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from fetch_windows_sources import MANIFEST, FFMPEG, collect_notices, digest, recipe, relative_path

CONFIGURE = [
    "--pkg-config=pkg-config", "--pkg-config-flags=--static",
    "--cross-prefix=x86_64-w64-mingw32-", "--arch=x86_64", "--target-os=mingw32",
    "--cc=x86_64-w64-mingw32-gcc", "--cxx=x86_64-w64-mingw32-g++",
    "--ar=x86_64-w64-mingw32-gcc-ar", "--ranlib=x86_64-w64-mingw32-gcc-ranlib",
    "--nm=x86_64-w64-mingw32-gcc-nm", "--disable-autodetect", "--disable-doc",
    "--disable-debug", "--disable-ffplay", "--enable-shared", "--disable-static",
    "--enable-gpl", "--enable-version3", "--disable-w32threads", "--enable-pthreads",
    "--enable-zlib", "--disable-bzlib", "--enable-lzma", "--enable-schannel",
    "--enable-libx264", "--enable-libx265", "--enable-libaom", "--enable-libsvtav1",
    "--enable-libdav1d", "--enable-libvpl", "--enable-ffnvcodec", "--enable-nvenc",
    "--enable-nvdec", "--enable-cuvid", "--enable-amf", "--enable-d3d11va", "--enable-dxva2",
]


def verify_sources(sources):
    """Require the complete collected receipt and recheck the actual archive bytes."""
    expected = recipe()
    actual = json.loads((sources / "sources.json").read_text())
    recorded = [{key: value for key, value in component.items() if key != "notices"}
                for component in actual["sources"]]
    if {**actual, "sources": recorded} != expected:
        raise ValueError("source receipt differs from the pinned recipe")
    for component in expected["sources"]:
        if digest(sources / "archives" / component["archive"]) != component["sha256"]:
            raise ValueError(f"source checksum mismatch: {component['name']}")
    # Regenerate the expected notice list from pinned bytes, not a mutable cache receipt.
    with tempfile.TemporaryDirectory(prefix="windows-sdk-notices-") as scratch:
        for component in actual["sources"]:
            notices = collect_notices(sources / "archives" / component["archive"],
                                      component, Path(scratch) / component["name"])
            if notices != component["notices"]:
                raise ValueError(f"notice list differs from source: {component['name']}")
            for notice in notices:
                path = relative_path(notice["path"])
                if digest(sources / "notices" / component["name"] / path) != notice["sha256"]:
                    raise ValueError(f"notice checksum mismatch: {component['name']}/{path}")
    return actual


def build(sources, work, prefix, jobs):
    if sys.version_info < (3, 12) or sys.platform != "linux":
        raise ValueError("the cross-build requires Linux and Python 3.12 or newer")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    for directory in (work, prefix):
        if directory.exists() or directory.is_symlink():
            raise ValueError("work and prefix must be new directories")
    sources, work, prefix = sources.resolve(), work.resolve(), prefix.resolve()
    for left, right in ((work, prefix), (sources, work), (sources, prefix)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("sources, work and prefix must be separate directories")
    pinned = verify_sources(sources)
    work.mkdir(parents=True)
    env = {
        "PATH": "/opt/ct-ng/bin:/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8",
        "PKG_CONFIG_LIBDIR": "/opt/ffbuild/lib/pkgconfig:/opt/ffbuild/share/pkgconfig",
    }
    headers = next(item for item in pinned["sources"] if item["name"] == "nv-codec-headers")
    ffmpeg = pinned["sources"][0]
    for component in (headers, ffmpeg):
        with tarfile.open(sources / "archives" / component["archive"]) as archive:
            archive.extractall(work, filter="data")
    header_prefix = work / "headers"
    subprocess.run(["make", "install", f"PREFIX={header_prefix}"],
                   cwd=work / headers["prefix"], env=env, check=True)
    env["PKG_CONFIG_PATH"] = str(header_prefix / "lib/pkgconfig")
    version = subprocess.check_output(["pkg-config", "--modversion", "ffnvcodec"], env=env, text=True).strip()
    if version != headers["revision"]:
        raise ValueError(f"incorrect nv-codec headers: {version}")
    build_dir = work / "build"
    build_dir.mkdir()
    command = [str(work / ffmpeg["prefix"] / "configure"), f"--prefix={prefix}", *CONFIGURE,
               f"--extra-cflags=-I{header_prefix}/include -I/opt/ffbuild/include -O2 -pipe -D_FORTIFY_SOURCE=2 -fstack-protector-strong",
               "--extra-ldflags=-L/opt/ffbuild/lib -static-libgcc -static-libstdc++ -pthread"]
    subprocess.run(command, cwd=build_dir, env=env, check=True)
    subprocess.run(["make", f"-j{jobs}"], cwd=build_dir, env=env, check=True)
    subprocess.run(["make", "install"], cwd=build_dir, env=env, check=True)
    # ffmpeg-sys-next searches FFMPEG_DIR/lib, while FFmpeg installs MSVC import files in bin.
    for name in ("avcodec", "avdevice", "avfilter", "avformat", "avutil", "swresample", "swscale"):
        (prefix / "bin" / f"{name}.lib").rename(prefix / "lib" / f"{name}.lib")
    provenance = prefix / "share/reco-ffmpeg"
    provenance.mkdir(parents=True)
    for component in pinned["sources"]:
        for notice in component["notices"]:
            relative = Path(component["name"]) / relative_path(notice["path"])
            target = provenance / "notices" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources / "notices" / relative, target)
    for path in (MANIFEST, FFMPEG, Path(__file__), sources / "sources.json", build_dir / "ffbuild/config.mak"):
        shutil.copy2(path, provenance)
    receipt = {
        "toolchain_image": pinned["toolchain_image"],
        "configure": command,
        "compiler": subprocess.check_output(["x86_64-w64-mingw32-gcc", "--version"], env=env, text=True),
        "files": {str(path.relative_to(prefix)): digest(path)
                  for directory in (prefix / "bin", prefix / "lib", prefix / "include")
                  for path in sorted(directory.rglob("*")) if path.is_file()},
    }
    (provenance / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    try:
        build(args.sources, args.work_dir, args.prefix, args.jobs)
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Windows SDK build failed: {error}\n")


if __name__ == "__main__":
    main()
