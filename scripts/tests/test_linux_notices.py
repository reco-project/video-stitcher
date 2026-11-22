"""Fail closed on unattributed or locally modified distribution libraries."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linux_notices import debian_owner
from fetch_linux_sources import fetch
from stage_linux_desktop import stage


class NoticeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.library = Path(self.temp.name) / "libexample.so.1"
        self.library.write_bytes(b"library")
        self.md5 = hashlib.md5(b"library").hexdigest()

    def lookup(self, search=None, md5=None, status="installed"):
        return [search or f"libexample1:amd64: {self.library}\n",
                f"libexample1:amd64\t1.2-3\texample\t1.2-3\t{status}",
                f"{md5 or self.md5}  {str(self.library).lstrip('/')}\n"]

    def test_exact_identity_and_checksum(self):
        with patch("linux_notices.query", side_effect=self.lookup()):
            self.assertEqual(debian_owner(self.library), {
                "package": "libexample1:amd64", "version": "1.2-3",
                "source_package": "example", "source_version": "1.2-3",
            })

    def test_modified_library_is_rejected(self):
        with patch("linux_notices.query", side_effect=self.lookup(md5="0" * 32)):
            with self.assertRaisesRegex(ValueError, "differs"):
                debian_owner(self.library)

    def test_multiple_owners_or_diversions_are_rejected(self):
        for search in (f"first, second: {self.library}\n", f"diversion by local from: {self.library}\n"):
            with patch("linux_notices.query", side_effect=self.lookup(search=search)):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    debian_owner(self.library)

    def test_missing_owner_is_rejected(self):
        with patch("linux_notices.query", side_effect=subprocess.CalledProcessError(1, "dpkg-query")):
            with self.assertRaisesRegex(ValueError, "no installed package"):
                debian_owner(self.library)

    def test_removed_package_is_rejected(self):
        with patch("linux_notices.query", side_effect=self.lookup(status="config-files")):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                debian_owner(self.library)

    def test_source_failure_does_not_publish_or_select_newer_version(self):
        root = Path(self.temp.name)
        required = root / "required.json"
        required.write_text(json.dumps([{"package": "example", "version": "1:1.2-3"}]))
        with patch("fetch_linux_sources.subprocess.run", side_effect=subprocess.CalledProcessError(100, "apt-get")) as command:
            with self.assertRaises(subprocess.CalledProcessError):
                fetch(required, root / "output")
            self.assertEqual(command.call_args.args[0][-1], "example=1:1.2-3")
        self.assertFalse((root / "output").exists())
        self.assertEqual(list(root.glob(".sources-*")), [])

    def test_invalid_source_identity_does_not_run_apt(self):
        root = Path(self.temp.name)
        required = root / "required.json"
        required.write_text(json.dumps([{"package": "--bad", "version": "1"}]))
        with patch("fetch_linux_sources.subprocess.run") as command:
            with self.assertRaises(ValueError):
                fetch(required, root / "output")
            command.assert_not_called()

    def test_staging_does_not_overwrite_existing_or_dangling_output(self):
        root = Path(self.temp.name)
        existing = root / "existing"
        existing.mkdir()
        dangling = root / "dangling"
        dangling.symlink_to(root / "absent")
        for output in (existing, dangling):
            with self.assertRaisesRegex(ValueError, "already exists"):
                stage(self.library, root / "missing-sdk", root / "missing-ort", [], output)
        self.assertTrue(existing.is_dir())
        self.assertTrue(dangling.is_symlink())

    def test_staging_rejects_mismatched_sdk_recipe(self):
        root = Path(self.temp.name)
        provenance = root / "sdk/share/reco-ffmpeg"
        provenance.mkdir(parents=True)
        (provenance / "ffmpeg-sdk.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "SDK recipe differs"):
            stage(self.library, root / "sdk", root / "missing-ort", [], root / "output")
        self.assertFalse((root / "output").exists())

    def test_staging_rejects_wrong_ort_archive_before_extraction(self):
        root = Path(self.temp.name)
        provenance = root / "sdk/share/reco-ffmpeg"
        provenance.mkdir(parents=True)
        manifest = Path(__file__).resolve().parents[1] / "ffmpeg-sdk.json"
        (provenance / "ffmpeg-sdk.json").write_bytes(manifest.read_bytes())
        archive = root / "ort.whl"
        archive.write_bytes(b"wrong runtime")
        with self.assertRaisesRegex(ValueError, "archive checksum mismatch"):
            stage(self.library, root / "sdk", archive, [], root / "output")
        self.assertFalse((root / "output").exists())
        self.assertEqual(list(root.glob(".desktop-*")), [])
