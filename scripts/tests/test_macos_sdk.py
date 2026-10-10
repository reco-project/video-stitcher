"""Native SDK builds must retain verified sources and reject unsafe destinations."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_macos_ffmpeg_sdk import CODECS, CONFIGURE, build, native_recipe, verify_sdk
from fetch_windows_sources import collect, digest, recipe, validate


class MacosSdkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.work, self.sdk = self.root / "work", self.root / "sdk"
        self.enterContext(patch("build_macos_ffmpeg_sdk.sys.platform", "darwin"))
        self.enterContext(patch("build_macos_ffmpeg_sdk.platform.machine", return_value="arm64"))

    def fixture(self):
        components = []
        for name in CODECS:
            path = self.cache / f"{name}.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                info = tarfile.TarInfo("source/LICENSE")
                info.size = 7
                archive.addfile(info, io.BytesIO(b"license"))
            components.append({"name": name, "revision": "test", "archive": path.name, "prefix": "source",
                               "url": "https://example.org/source.tar.gz", "required_notices": ["LICENSE"],
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return {"minimum_macos": "14.0", "sources": components}

    def test_source_pins_are_shared_without_windows_toolchain_inputs(self):
        native = native_recipe()
        validate(native)
        shared = {item["name"]: item for item in recipe()["sources"]}
        self.assertEqual(tuple(item["name"] for item in native["sources"]), CODECS)
        for item in native["sources"]:
            self.assertEqual(item, shared[item["name"]])
        self.assertNotIn("toolchain_image", native)

    def test_existing_and_overlapping_directories_rejected_before_commands(self):
        with patch("build_macos_ffmpeg_sdk.subprocess.run") as command:
            for work, prefix in ((self.cache, self.sdk), (self.work, self.work / "nested"),
                                 (self.cache / "nested", self.sdk)):
                with self.subTest(work=work, prefix=prefix), self.assertRaises(ValueError):
                    build(self.cache, work, prefix)
            command.assert_not_called()

    def test_corrupt_archive_prevents_build(self):
        fixture = self.fixture()
        (self.cache / "x264.tar.gz").write_bytes(b"changed")
        with patch("build_macos_ffmpeg_sdk.native_recipe", return_value=fixture), \
             patch("build_macos_ffmpeg_sdk.subprocess.run") as command:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build(self.cache, self.work, self.sdk, offline=True)
            command.assert_not_called()
        self.assertFalse(self.sdk.exists())

    def sdk_fixture(self):
        fixture = self.fixture()
        collect(self.cache, self.sdk / "share/reco-ffmpeg", fixture, offline=True)
        names = ["bin/ffmpeg", "bin/ffprobe", "include/libavcodec/avcodec.h"]
        names += [f"lib/lib{name}.dylib" for name in (
            "avcodec", "avdevice", "avfilter", "avformat", "avutil", "swresample", "swscale",
            "x264", "x265", "aom", "SvtAv1Enc", "dav1d")]
        for name in names:
            path = self.sdk / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        receipt = {"architecture": "arm64", "minimum_macos": "14.0",
                   "commands": [{"label": "ffmpeg-configure", "args": ["/source/configure", "--prefix=/sdk",
                       *CONFIGURE, "--extra-ldflags=-Wl,-rpath,/sdk/lib"]}],
                   "files": {name: digest(self.sdk / name) for name in names}}
        path = self.sdk / "share/reco-ffmpeg/build.json"
        path.write_text(json.dumps(receipt))
        self.enterContext(patch("build_macos_ffmpeg_sdk.native_recipe", return_value=fixture))
        return path, receipt

    def test_sdk_verifies_architecture_configuration_and_required_outputs(self):
        path, receipt = self.sdk_fixture()
        self.assertEqual(verify_sdk(self.sdk, "arm64"), receipt)
        with self.assertRaisesRegex(ValueError, "configuration"):
            verify_sdk(self.sdk, "x86_64")
        original = receipt["commands"][0]["args"][:]
        receipt["commands"][0]["args"].remove("--enable-videotoolbox")
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "configuration"):
            verify_sdk(self.sdk, "arm64")
        receipt["commands"][0]["args"] = original
        del receipt["files"]["bin/ffprobe"]
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "required native outputs"):
            verify_sdk(self.sdk, "arm64")

    def test_sdk_rejects_modified_output_and_escaping_symlink(self):
        self.sdk_fixture()
        path = self.sdk / "bin/ffmpeg"
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_sdk(self.sdk, "arm64")
        path.unlink()
        outside = self.root / "external"
        outside.write_bytes(original)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_sdk(self.sdk, "arm64")

    def test_sdk_rejects_modified_sources_and_notices(self):
        self.sdk_fixture()
        archive = self.sdk / "share/reco-ffmpeg/archives/x264.tar.gz"
        original = archive.read_bytes()
        archive.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
            verify_sdk(self.sdk, "arm64")
        archive.write_bytes(original)
        (self.sdk / "share/reco-ffmpeg/notices/x264/LICENSE").write_text("changed")
        with self.assertRaisesRegex(ValueError, "notice checksum mismatch"):
            verify_sdk(self.sdk, "arm64")

    def test_build_retains_sources_notices_commands_and_hashes(self):
        fixture = self.fixture()
        calls = []

        def command(args, **kwargs):
            calls.append((args, kwargs))
            if args == ["make", "install"] and kwargs["cwd"].name == "build-ffmpeg":
                config = kwargs["cwd"] / "ffbuild/config.mak"
                config.parent.mkdir()
                config.write_text("config")
                (self.sdk / "bin").mkdir(parents=True)
                (self.sdk / "bin/ffmpeg").write_bytes(b"ffmpeg")

        with patch("build_macos_ffmpeg_sdk.native_recipe", return_value=fixture), \
             patch("build_macos_ffmpeg_sdk.subprocess.run", side_effect=command), \
             patch("build_macos_ffmpeg_sdk.subprocess.check_output", return_value="tool version"), \
             patch.dict("os.environ", {"CFLAGS": "-march=native", "PKG_CONFIG_PATH": "/foreign",
                                       "SDKROOT": "/foreign", "CMAKE_TOOLCHAIN_FILE": "/foreign.cmake"}):
            build(self.cache, self.work, self.sdk, offline=True)
        provenance = self.sdk / "share/reco-ffmpeg"
        self.assertEqual(len(list((provenance / "archives").iterdir())), len(CODECS))
        self.assertEqual((provenance / "notices/x265/LICENSE").read_bytes(), b"license")
        self.assertTrue((provenance / "build_macos_ffmpeg_sdk.py").is_file())
        self.assertFalse((provenance / "build_windows_ffmpeg_sdk.py").exists())
        receipt = json.loads((provenance / "build.json").read_text())
        self.assertEqual(receipt["files"]["bin/ffmpeg"], hashlib.sha256(b"ffmpeg").hexdigest())
        self.assertEqual(receipt["architecture"], "arm64")
        for _, kwargs in calls:
            self.assertNotIn("CFLAGS", kwargs["env"])
            self.assertNotIn("SDKROOT", kwargs["env"])
            self.assertNotIn("CMAKE_TOOLCHAIN_FILE", kwargs["env"])
            self.assertEqual(kwargs["env"]["PKG_CONFIG_PATH"], "")
            self.assertEqual(kwargs["env"]["MACOSX_DEPLOYMENT_TARGET"], "14.0")
        arguments = [argument for args, _ in calls for argument in args]
        self.assertIn("-DLINKED_10BIT=ON", arguments)
        self.assertIn("-DLINKED_12BIT=ON", arguments)
        self.assertIn("--enable-securetransport", arguments)
        self.assertIn("--enable-videotoolbox", arguments)
        self.assertIn("--extra-libs=-liconv", arguments)


if __name__ == "__main__":
    unittest.main()
