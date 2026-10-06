"""Exercise ELF dependency closure and relocation with compiled shared libraries."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundle_linux import bundle, check_glibc, glibc_references, inspect
from build_ffmpeg_sdk import build


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "build"
        self.source.mkdir()
        self.output = self.root / "package"
        self.compile("leaf.c", "int leaf(void) { return 41; }", [
            "-shared", "-fPIC", "-Wl,-soname,libleaf.so.1", "-o", "libleaf.so.1",
        ])
        self.compile("branch.c", "int leaf(void); int branch(void) { return leaf()+1; }", [
            "-shared", "-fPIC", "-Wl,-soname,libbranch.so.1", "-L.",
            "-l:libleaf.so.1", "-o", "libbranch.so.1",
        ])
        self.compile("main.c", '#include <stdio.h>\nint branch(void); '
                     'int main(void) { printf("%d\\n", branch()); return 0; }', [
            "-L.", "-l:libbranch.so.1", "-Wl,-rpath-link,.", "-o", "app",
        ])
        self.app = self.source / "app"

    def compile(self, name, code, options):
        (self.source / name).write_text(code)
        subprocess.run(["cc", name, *options], cwd=self.source, check=True)

    def test_relocated_transitive_closure_preserves_inputs(self):
        original = {p.name: p.read_bytes() for p in self.source.iterdir()}
        bundle([self.app], [self.source], self.output)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.source.rename(self.root / "hidden-build")
        relocated = self.root / "relocated with spaces"
        self.output.rename(relocated)
        result = subprocess.check_output([str(relocated / "app")], cwd="/", env={"PATH": "/usr/bin:/bin"})
        self.assertEqual(result, b"42\n")
        manifest = json.loads((relocated / "bundle.json").read_text())
        self.assertEqual(set(manifest["sha256"]), {"app", "libbranch.so.1", "libleaf.so.1"})
        self.assertIn("libc.so.6", manifest["system_dependencies"])
        for name, digest in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((relocated / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(inspect("--print-rpath", relocated / name), ["$ORIGIN"])

    def test_missing_dependency_does_not_publish_partial_output(self):
        (self.source / "libleaf.so.1").unlink()
        with self.assertRaisesRegex(ValueError, "unresolved dependency: libleaf"):
            bundle([self.app], [self.source], self.output)
        self.assertFalse(self.output.exists())

    def test_environment_is_not_a_dependency_search_path(self):
        other = self.root / "other"
        other.mkdir()
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": str(self.source)}):
            with self.assertRaisesRegex(ValueError, "unresolved dependency"):
                bundle([self.app], [other], self.output)

    def test_existing_output_is_untouched(self):
        self.output.mkdir()
        marker = self.output / "user-file"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            bundle([self.app], [self.source], self.output)
        self.assertEqual(marker.read_text(), "keep")

    def test_dangling_output_symlink_is_untouched(self):
        self.output.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            bundle([self.app], [self.source], self.output)
        self.assertTrue(self.output.is_symlink())

    def test_wrong_architecture_is_rejected(self):
        library = self.source / "libleaf.so.1"
        contents = bytearray(library.read_bytes())
        contents[18] ^= 1
        library.write_bytes(contents)
        with self.assertRaisesRegex(ValueError, "wrong ELF architecture"):
            bundle([self.app], [self.source], self.output)

    def test_wrong_soname_is_rejected(self):
        subprocess.run(["patchelf", "--set-soname", "libother.so.1", str(self.source / "libleaf.so.1")], check=True)
        with self.assertRaisesRegex(ValueError, "SONAME mismatch"):
            bundle([self.app], [self.source], self.output)

    def test_absolute_dependency_is_rejected(self):
        subprocess.run(["patchelf", "--replace-needed", "libbranch.so.1", str(self.source / "libbranch.so.1"), str(self.app)], check=True)
        with self.assertRaisesRegex(ValueError, "plain SONAME"):
            bundle([self.app], [self.source], self.output)

    def test_driver_libraries_stay_external(self):
        subprocess.run(["patchelf", "--add-needed", "libva.so.2", str(self.app)], check=True)
        # Even an available same-named file must not cause a driver-stack copy.
        shutil.copy2(self.source / "libleaf.so.1", self.source / "libva.so.2")
        bundle([self.app], [self.source], self.output)
        manifest = json.loads((self.output / "bundle.json").read_text())
        self.assertIn("libva.so.2", manifest["system_dependencies"])
        self.assertFalse((self.output / "libva.so.2").exists())

    def test_duplicate_roots_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bundle([self.app, self.app], [self.source], self.output)

    def test_dependency_cannot_overwrite_bundle_metadata(self):
        subprocess.run(["patchelf", "--add-needed", "bundle.json", str(self.app)], check=True)
        with self.assertRaisesRegex(ValueError, "reserved output name"):
            bundle([self.app], [self.source], self.output)

    def test_patch_failure_leaves_no_partial_bundle(self):
        with patch("bundle_linux.subprocess.run", side_effect=subprocess.CalledProcessError(1, "patchelf")):
            with self.assertRaises(subprocess.CalledProcessError):
                bundle([self.app], [self.source], self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".bundle-*")), [])

    def test_glibc_ceiling_checks_transitive_dependencies(self):
        def versions(path):
            return ["GLIBC_2.40"] if path.name == "libleaf.so.1" else ["GLIBC_2.34"]

        with patch("bundle_linux.glibc_references", side_effect=versions):
            with self.assertRaisesRegex(ValueError, "libleaf.so.1 references GLIBC_2.40"):
                bundle([self.app], [self.source], self.output, "2.39")
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".bundle-*")), [])

    def test_glibc_receipt_records_every_staged_elf(self):
        bundle([self.app], [self.source], self.output, "2.39")
        manifest = json.loads((self.output / "bundle.json").read_text())
        self.assertEqual(manifest["glibc_ceiling"], "2.39")
        self.assertEqual(set(manifest["glibc_references"]), set(manifest["sha256"]))
        self.assertIn("GLIBC_2.2.5", manifest["glibc_references"]["app"])


class GlibcTests(unittest.TestCase):
    def test_version_comparison_is_numeric_and_inclusive(self):
        check_glibc(["GLIBC_2.9", "GLIBC_2.39", "GLIBC_2.2.5"], "2.39", "app")
        with self.assertRaisesRegex(ValueError, "beyond glibc ceiling"):
            check_glibc(["GLIBC_2.10"], "2.9", "app")

    def test_private_or_unknown_version_is_not_silently_ignored(self):
        for name in ("GLIBC_PRIVATE", "GLIBC_ABI_DT_RELR"):
            with self.assertRaises(ValueError):
                check_glibc([name], "2.39", "app")
        with self.assertRaisesRegex(ValueError, "invalid glibc ceiling"):
            check_glibc([], "2.invalid", "app")

    def test_exports_do_not_raise_the_requirement(self):
        listing = """Version definition section '.gnu.version_d' contains 2 entries:
  Name: GLIBC_2.99
Version needs section '.gnu.version_r' contains 1 entry:
  Name: GLIBC_2.34  Flags: none  Version: 2
"""
        with patch("bundle_linux.subprocess.check_output", return_value=listing):
            self.assertEqual(glibc_references(Path("app")), ["GLIBC_2.34"])
        with patch("bundle_linux.subprocess.check_output", return_value=listing.split("Version needs section")[0]):
            self.assertEqual(glibc_references(Path("app")), [])


class SdkTests(unittest.TestCase):
    def test_sdk_retains_verified_source_and_build_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ffmpeg-test"
            (source / "ffbuild").mkdir(parents=True)
            for name in ("ffbuild/config.mak", "COPYING.GPLv3", "COPYING.GPLv2", "LICENSE.md"):
                (source / name).write_text(name)
            archive = root / "source.tar.xz"
            with tarfile.open(archive, "w:xz") as tar:
                tar.add(source, arcname=source.name)
            manifest = root / "ffmpeg-sdk.json"
            manifest.write_text(json.dumps({
                "version": "test", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "linux_headers": {}, "linux_configure": [],
            }))
            # Exercise provenance without repeating a native build in a unit test.
            with patch("build_ffmpeg_sdk.MANIFEST", manifest), patch("build_ffmpeg_sdk.subprocess.run"):
                build(root / "work", root / "sdk", 1, archive)
            provenance = root / "sdk/share/reco-ffmpeg"
            self.assertEqual((provenance / "ffmpeg-test.tar.xz").read_bytes(), archive.read_bytes())
            self.assertEqual(json.loads((provenance / "ffmpeg-sdk.json").read_text()), json.loads(manifest.read_text()))
            self.assertEqual((provenance / "config.mak").read_text(), "ffbuild/config.mak")
            self.assertEqual((provenance / "build_ffmpeg_sdk.py").read_bytes(), Path(build.__code__.co_filename).read_bytes())

    def test_checksum_failure_precedes_extraction_or_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "wrong.tar.xz"
            archive.write_bytes(b"incorrect source")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build(root / "work", root / "sdk", 1, archive)
            self.assertFalse((root / "sdk").exists())

    def test_overlapping_or_existing_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for work, prefix in [(root, root / "sdk"), (root / "work", root / "work" / "sdk")]:
                with self.assertRaises(ValueError):
                    build(work, prefix, 1, None)

    def test_dangling_prefix_symlink_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "sdk"
            prefix.symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "new directories"):
                build(root / "work", prefix, 1, None)
            self.assertTrue(prefix.is_symlink())
            self.assertFalse((root / "work").exists())

    def test_header_mismatch_precedes_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.tar.xz"
            archive.write_bytes(b"source")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "sha256": hashlib.sha256(b"source").hexdigest(),
                "linux_headers": {"ffnvcodec": "12.1.14.0"},
            }))
            with patch("build_ffmpeg_sdk.MANIFEST", manifest), patch("build_ffmpeg_sdk.subprocess.check_output", return_value="13.1\n"):
                with self.assertRaisesRegex(ValueError, "headers: expected"):
                    build(root / "work", root / "sdk", 1, archive)
            self.assertFalse((root / "sdk").exists())


if __name__ == "__main__":
    unittest.main()
