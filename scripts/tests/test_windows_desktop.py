"""Windows packaging fails closed for missing or changed native dependencies."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fetch_windows_sources import digest
from stage_windows_desktop import LIBRARIES, stage, verify_sdk, runtime_archive


class WindowsDesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sdk = self.root / "sdk"
        self.provenance = self.sdk / "share/reco-ffmpeg"
        self.provenance.mkdir(parents=True)
        self.recipe = {"toolchain_image": "pinned", "sources": [{
            "name": "example", "required_notices": ["LICENSE"],
        }]}
        notice = self.provenance / "notices/example/LICENSE"
        notice.parent.mkdir(parents=True)
        notice.write_text("license")
        sources = {**self.recipe, "sources": [{**self.recipe["sources"][0], "notices": [
            {"path": "LICENSE", "sha256": digest(notice)},
        ]}]}
        (self.provenance / "sources.json").write_text(json.dumps(sources))
        files = ["bin/ffmpeg.exe", "bin/ffprobe.exe", "include/libavcodec/avcodec.h"]
        files += [f"lib/{name}.lib" for name in LIBRARIES]
        files += [f"bin/{name}-1.dll" for name in LIBRARIES]
        self.receipt = {"toolchain_image": "pinned", "files": {}}
        for name in files:
            path = self.sdk / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"sdk")
            self.receipt["files"][name] = digest(path)
        self.save_receipt()
        self.recipe_patch = patch("stage_windows_desktop.recipe", return_value=self.recipe)
        self.recipe_patch.start()
        self.addCleanup(self.recipe_patch.stop)
        self.binary = self.root / "reco.exe"
        self.binary.write_bytes(b"binary")
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.manifest = self.root / "runtime.json"
        self.output = self.root / "package"
        self.runtime()
        self.manifest_patch = patch("stage_windows_desktop.MANIFEST", self.manifest)
        self.manifest_patch.start()
        self.addCleanup(self.manifest_patch.stop)

    def save_receipt(self):
        (self.provenance / "build.json").write_text(json.dumps(self.receipt))

    def runtime(self, missing=None):
        archive = self.cache / "onnxruntime-1.nupkg"
        with zipfile.ZipFile(archive, "w") as stream:
            for name in ("bin/onnxruntime.dll", "LICENSE"):
                if name != missing:
                    stream.writestr(name, b"runtime")
        self.component = {"version": "1", "url": "https://example.org/runtime.zip",
                          "sha256": digest(archive), "libraries": ["bin/onnxruntime.dll"],
                          "notices": ["LICENSE"]}
        self.manifest.write_text(json.dumps({"onnxruntime": self.component}))

    def test_complete_package_and_receipt(self):
        stage(self.binary, self.sdk, self.cache, self.output, offline=True)
        self.assertTrue((self.output / "onnxruntime.dll").is_file())
        self.assertTrue((self.output / "notices/onnxruntime/LICENSE").is_file())
        self.assertFalse((self.output / "avcodec.lib").exists())
        files = json.loads((self.output / "files.json").read_text())
        for name, checksum in files.items():
            self.assertEqual(digest(self.output / name), checksum)

    def test_changed_sdk_recipe_and_notice_rejected(self):
        self.assertEqual(verify_sdk(self.sdk), self.receipt)
        self.receipt["toolchain_image"] = "different"
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "toolchain"):
            verify_sdk(self.sdk)
        self.receipt["toolchain_image"] = "pinned"
        self.save_receipt()
        (self.provenance / "notices/example/LICENSE").write_text("changed")
        with self.assertRaisesRegex(ValueError, "notice checksum"):
            verify_sdk(self.sdk)

    def test_sdk_bytes_and_missing_import_library_rejected(self):
        (self.sdk / "bin/avcodec-1.dll").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SDK checksum"):
            verify_sdk(self.sdk)
        del self.receipt["files"]["lib/avcodec.lib"]
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "missing import library"):
            verify_sdk(self.sdk)

    def test_missing_dll_or_notice_never_publishes(self):
        for missing in ("bin/onnxruntime.dll", "LICENSE"):
            with self.subTest(missing=missing):
                self.runtime(missing)
                with self.assertRaises(KeyError):
                    stage(self.binary, self.sdk, self.cache, self.output, offline=True)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob(".windows-desktop-*")), [])

    def test_bad_archive_never_publishes(self):
        (self.cache / "onnxruntime-1.nupkg").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "runtime checksum"):
            stage(self.binary, self.sdk, self.cache, self.output, offline=True)
        self.assertFalse(self.output.exists())

    def test_existing_output_preserved(self):
        self.output.mkdir()
        marker = self.output / "user.txt"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            stage(self.binary, self.sdk, self.cache, self.output)
        self.assertEqual(marker.read_text(), "keep")

    def test_missing_offline_runtime_does_not_download(self):
        (self.cache / "onnxruntime-1.nupkg").unlink()
        with patch("stage_windows_desktop.urllib.request.urlopen") as download:
            with self.assertRaisesRegex(ValueError, "unavailable offline"):
                runtime_archive(self.cache, "onnxruntime", self.component, True)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
