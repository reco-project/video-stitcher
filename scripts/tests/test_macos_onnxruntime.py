"""Intel runtime collection must fail closed before publishing an incomplete build."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_macos_onnxruntime import MINIMUM_MACOS, build
from build_macos_ffmpeg_sdk import MINIMUM_MACOS as SDK_MINIMUM
from fetch_windows_sources import digest


class MacosOnnxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.work, self.output = self.root / "work", self.root / "runtime"
        archive = self.cache / "onnxruntime.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name, value in (("LICENSE", b"license"), ("ThirdPartyNotices.txt", b"notices"),
                                ("VERSION_NUMBER", b"1.24.4"), ("cmake/deps.txt", b"pinned dependencies")):
                info = tarfile.TarInfo("source/" + name)
                info.size = len(value)
                stream.addfile(info, io.BytesIO(value))
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"onnxruntime": {"version": "1.24.4", "source": {
            "name": "onnxruntime", "revision": "fixed", "archive": archive.name, "prefix": "source",
            "sha256": digest(archive), "url": "https://example.org/source.tar.gz",
            "required_notices": ["LICENSE", "ThirdPartyNotices.txt"]}}}))
        self.manifest = manifest
        self.enterContext(patch("build_macos_onnxruntime.MANIFEST", manifest))
        self.enterContext(patch("build_macos_onnxruntime.sys.platform", "darwin"))

    def test_os_baseline_matches_sdk(self):
        self.assertEqual(MINIMUM_MACOS, SDK_MINIMUM)

    def test_existing_or_overlapping_output_rejected(self):
        with patch("build_macos_onnxruntime.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "new directories"):
                build(self.cache, self.work, self.cache)
            with self.assertRaisesRegex(ValueError, "separate"):
                build(self.cache, self.work, self.work / "runtime")
            run.assert_not_called()

    def test_failed_build_does_not_publish_runtime(self):
        with patch("build_macos_onnxruntime.subprocess.run", side_effect=subprocess.CalledProcessError(1, "build")):
            with self.assertRaises(subprocess.CalledProcessError):
                build(self.cache, self.work, self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue((self.work / "command.json").is_file())

    def test_source_version_mismatch_fails_before_build(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["onnxruntime"]["version"] = "1.23.0"
        self.manifest.write_text(json.dumps(manifest))
        with patch("build_macos_onnxruntime.subprocess.run") as command:
            with self.assertRaisesRegex(ValueError, "source version differs"):
                build(self.cache, self.work, self.output)
            command.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_corrupt_source_archive_fails_before_build(self):
        (self.cache / "onnxruntime.tar.gz").write_bytes(b"changed")
        with patch("build_macos_onnxruntime.subprocess.run") as command:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build(self.cache, self.work, self.output)
            command.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_success_preserves_sources_notices_and_intel_identity(self):
        calls = []

        def command(args, **kwargs):
            calls.append((args, kwargs))
            library = self.work / "build/Release/libonnxruntime.1.24.4.dylib"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"runtime")

        with patch("build_macos_onnxruntime.subprocess.run", side_effect=command), \
             patch("build_macos_onnxruntime.subprocess.check_output", return_value="compiler"), \
             patch("build_macos_onnxruntime.inspect", return_value={"minimum_macos": "14.0"}) as inspect, \
             patch.dict("os.environ", {"CFLAGS": "-march=native", "DYLD_LIBRARY_PATH": "/foreign",
                                       "CMAKE_TOOLCHAIN_FILE": "/foreign.cmake"}):
            build(self.cache, self.work, self.output)
        args, kwargs = calls[0]
        self.assertEqual(args[args.index("--osx_arch") + 1], "x86_64")
        self.assertIn("FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER", args)
        self.assertIn("--skip_pip_install", args)
        self.assertNotIn("CFLAGS", kwargs["env"])
        self.assertNotIn("DYLD_LIBRARY_PATH", kwargs["env"])
        self.assertNotIn("CMAKE_TOOLCHAIN_FILE", kwargs["env"])
        inspect.assert_called_once_with(self.work / "build/Release/libonnxruntime.1.24.4.dylib", "x86_64", "14.0")
        self.assertEqual((self.output / "lib/libonnxruntime.dylib").read_bytes(), b"runtime")
        self.assertEqual((self.output / "ThirdPartyNotices.txt").read_bytes(), b"notices")
        self.assertEqual((self.output / "GIT_COMMIT_ID").read_text(), "fixed\n")
        self.assertTrue((self.output / "sources/archives/onnxruntime.tar.gz").is_file())
        receipt = json.loads((self.output / "build.json").read_text())
        self.assertEqual(receipt["sha256"], digest(self.output / "lib/libonnxruntime.dylib"))


if __name__ == "__main__":
    unittest.main()
