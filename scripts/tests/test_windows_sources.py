"""Source archives must match their pins before any package is published."""

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
from fetch_windows_sources import collect, recipe, validate
from build_windows_ffmpeg_sdk import build, verify_sources


class WindowsSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.output = self.root / "output"

    def fixture(self, members=None):
        archive = self.cache / "example.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name, kind, data in members or [("source/COPYING", tarfile.REGTYPE, b"license")]:
                item = tarfile.TarInfo(name)
                item.type = kind
                if kind == tarfile.REGTYPE:
                    item.size = len(data)
                    stream.addfile(item, io.BytesIO(data))
                else:
                    item.linkname = "../../outside"
                    stream.addfile(item)
        return {"sources": [{
            "name": "example", "archive": archive.name,
            "url": "https://example.org/source.tar.gz",
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "prefix": "source", "required_notices": ["COPYING"], "revision": "1",
        }]}

    def test_complete_offline_receipt_and_notices(self):
        manifest = self.fixture()
        collect(self.cache, self.output, manifest, offline=True)
        receipt = json.loads((self.output / "sources.json").read_text())
        self.assertEqual(receipt["sources"][0]["sha256"], manifest["sources"][0]["sha256"])
        self.assertEqual((self.output / "notices/example/COPYING").read_bytes(), b"license")
        self.assertEqual((self.cache / "example.tar.gz").read_bytes(),
                         (self.output / "archives/example.tar.gz").read_bytes())

    def test_wrong_checksum_leaves_no_output_or_replacement(self):
        manifest = self.fixture()
        archive = self.cache / "example.tar.gz"
        original = archive.read_bytes()
        manifest["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            collect(self.cache, self.output, manifest)
        self.assertFalse(self.output.exists())
        self.assertEqual(archive.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".windows-sources-*")), [])

    def test_unsafe_and_duplicate_notices_are_rejected(self):
        for members in (
            [("source/../../COPYING", tarfile.REGTYPE, b"unsafe")],
            [("source/COPYING", tarfile.SYMTYPE, b"")],
            [("source/COPYING", tarfile.LNKTYPE, b"")],
            [("source/COPYING", tarfile.REGTYPE, b"" )],
            [("source/COPYING", tarfile.REGTYPE, b"one"), ("source/copying", tarfile.REGTYPE, b"two")],
        ):
            with self.subTest(members=members):
                with self.assertRaises(ValueError):
                    collect(self.cache, self.output, self.fixture(members), offline=True)
                self.assertFalse(self.output.exists())

    def test_optional_notice_link_is_not_followed(self):
        manifest = self.fixture([("source/COPYING", tarfile.REGTYPE, b"license"),
                                 ("source/LICENSE", tarfile.SYMTYPE, b"")])
        collect(self.cache, self.output, manifest, offline=True)
        self.assertFalse((self.output / "notices/example/LICENSE").exists())

    def test_missing_required_notice(self):
        with self.assertRaisesRegex(ValueError, "missing notices"):
            collect(self.cache, self.output,
                    self.fixture([("source/code.c", tarfile.REGTYPE, b"code")]), offline=True)
        self.assertFalse(self.output.exists())

    def test_offline_missing_archive_never_downloads(self):
        manifest = self.fixture()
        (self.cache / "example.tar.gz").unlink()
        with patch("fetch_windows_sources.urllib.request.urlopen") as download:
            with self.assertRaisesRegex(ValueError, "unavailable offline"):
                collect(self.cache, self.output, manifest, offline=True)
            download.assert_not_called()

    def test_bad_download_not_cached_or_published(self):
        manifest = self.fixture()
        (self.cache / "example.tar.gz").unlink()
        response = io.BytesIO(b"bad archive")
        response.url = "https://example.org/source.tar.gz"
        with patch("fetch_windows_sources.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                collect(self.cache, self.output, manifest)
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertFalse(self.output.exists())

    def test_existing_output_and_cache_links_rejected(self):
        manifest = self.fixture()
        self.output.symlink_to(self.root / "absent")
        with self.assertRaisesRegex(ValueError, "already exists"):
            collect(self.cache, self.output, manifest)
        self.output.unlink()
        archive = self.cache / "example.tar.gz"
        archive.rename(self.root / "saved.tar.gz")
        archive.symlink_to(self.root / "saved.tar.gz")
        with self.assertRaisesRegex(ValueError, "symlink"):
            collect(self.cache, self.output, manifest)

    def test_manifest_validation_and_common_ffmpeg_pin(self):
        pinned = recipe()
        validate(pinned)
        self.assertEqual(pinned["sources"][0]["name"], "ffmpeg")
        self.assertEqual(len(pinned["sources"]), 18)
        for field, value in (("name", "../escape"), ("archive", "../escape.tar.gz"),
                             ("sha256", "missing"), ("url", "http://example.org")):
            manifest = self.fixture()
            manifest["sources"][0][field] = value
            with self.assertRaises(ValueError):
                validate(manifest)

    def test_sdk_rejects_mutated_notice_and_omitted_notice_receipt(self):
        manifest = self.fixture()
        collect(self.cache, self.output, manifest, offline=True)
        with patch("build_windows_ffmpeg_sdk.recipe", return_value=manifest):
            self.assertEqual(verify_sources(self.output)["sources"][0]["name"], "example")
            receipt_path = self.output / "sources.json"
            receipt = json.loads(receipt_path.read_text())
            original = receipt_path.read_bytes()
            receipt["sources"][0]["notices"] = []
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "notice list differs"):
                verify_sources(self.output)
            receipt_path.write_bytes(original)
            (self.output / "notices/example/COPYING").write_text("changed")
            with self.assertRaisesRegex(ValueError, "notice checksum mismatch"):
                verify_sources(self.output)

    def test_sdk_rejects_changed_archive_or_recipe(self):
        manifest = self.fixture()
        collect(self.cache, self.output, manifest, offline=True)
        with patch("build_windows_ffmpeg_sdk.recipe", return_value=manifest):
            receipt_path = self.output / "sources.json"
            receipt = json.loads(receipt_path.read_text())
            original = receipt_path.read_bytes()
            receipt["sources"][0]["revision"] = "different"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "recipe"):
                verify_sources(self.output)
            receipt_path.write_bytes(original)
            (self.output / "archives/example.tar.gz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                verify_sources(self.output)

    def test_build_rejects_existing_directories_before_starting_commands(self):
        with patch("build_windows_ffmpeg_sdk.subprocess.run") as command:
            with self.assertRaisesRegex(ValueError, "new directories"):
                build(self.output, self.cache, self.root / "sdk", 2)
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
