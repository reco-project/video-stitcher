"""Mach-O packaging rejects foreign, ambiguous and incomplete dependency closures."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundle_macos import bundle, inspect, load_commands, plan, resolve_dependency, version, verify_loaded_libraries


class MacosBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.libs = self.root / "sdk/lib"
        self.libs.mkdir(parents=True)
        self.binary = self.root / "reco"
        self.binary.write_bytes(b"application")
        self.library = self.libs / "libexample.dylib"
        self.library.write_bytes(b"library")
        self.output = self.root / "output"
        self.records = {
            "reco": {"identity": None, "dependencies": [str(self.library)], "rpaths": ["@loader_path"], "minimum_macos": "11.0"},
            self.library.name: {"identity": str(self.library), "dependencies": ["/usr/lib/libSystem.B.dylib"],
                                "rpaths": [], "minimum_macos": "14.0"},
        }

    def test_load_command_formats_and_platform(self):
        text = "Load command 0\ncmd LC_RPATH\npath @loader_path (offset 12)\nLoad command 1\ncmd LC_BUILD_VERSION\nplatform 1\nminos 14.0\n"
        self.assertEqual(load_commands(text), (["@loader_path"], "14.0"))
        self.assertEqual(load_commands("Load command 0\ncmd LC_VERSION_MIN_MACOSX\nversion 11.0\n"), ([], "11.0"))
        with self.assertRaisesRegex(ValueError, "not macOS"):
            load_commands(text.replace("platform 1", "platform 2"))
        with self.assertRaisesRegex(ValueError, "missing"):
            load_commands("")
        self.assertEqual(version("14"), version("14.0.0"))
        for value in ("", "14.x", "14.0.0.0", "-1"):
            with self.assertRaises(ValueError):
                version(value)

    def test_foreign_architecture_and_newer_os_rejected(self):
        with patch("bundle_macos.run", return_value="x86_64 arm64"):
            with self.assertRaisesRegex(ValueError, "single arm64"):
                inspect(self.binary, "arm64", "14.0")
        with patch("bundle_macos.run", side_effect=["arm64", "binary:\n", "binary:\n",
                "Load command 0\ncmd LC_BUILD_VERSION\nplatform 1\nminos 15.0\n"]):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                inspect(self.binary, "arm64", "14.0")

    def test_loaded_libraries_require_package_evidence_and_reject_foreign_paths(self):
        log = f"dyld[123]: <ABC> {self.output}/reco\ndyld[123]: <DEF> /usr/lib/libSystem.B.dylib\n"
        verify_loaded_libraries(log, self.output)
        for bad in ("", "dyld[123]: <DEF> /usr/lib/libSystem.B.dylib\n",
                    log + "dyld[123]: <DEF> /opt/homebrew/lib/example.dylib\n",
                    log + "dyld[123]: <DEF> /usr/lib/../../tmp/example.dylib\n"):
            with self.assertRaises(ValueError):
                verify_loaded_libraries(bad, self.output)

    def test_resolution_is_confined_to_explicit_roots(self):
        for name in (str(self.library), "@rpath/libexample.dylib", "@loader_path/libexample.dylib"):
            self.assertEqual(resolve_dependency(name, self.library, [self.libs]), self.library)
        for name in (str(self.binary), "@rpath/../reco", "@executable_path/reco", "relative.dylib", "@rpath/missing"):
            with self.assertRaises(ValueError):
                resolve_dependency(name, self.library, [self.libs])
        (self.libs / "escape.dylib").symlink_to(self.binary)
        with self.assertRaisesRegex(ValueError, "outside trusted"):
            resolve_dependency("@rpath/escape.dylib", self.library, [self.libs])

    def test_ambiguous_dependency_is_rejected(self):
        other = self.root / "other"
        other.mkdir()
        (other / self.library.name).write_bytes(b"different library")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_dependency("@rpath/libexample.dylib", self.binary, [self.libs, other])

    def test_dependency_closure_and_cycles(self):
        self.records[self.library.name]["dependencies"].append(str(self.library))
        with patch("bundle_macos.inspect", side_effect=lambda path, *_: copy.deepcopy(self.records[path.name])):
            records = plan([self.binary], [self.libs], "arm64", "14.0")
        self.assertEqual(set(records), {"reco", "libexample.dylib"})
        self.assertEqual(records["reco"]["replacements"], {str(self.library): self.library.name})

    def test_case_insensitive_filename_collisions(self):
        other = self.root / "Reco"
        other.write_bytes(b"different application")
        with patch("bundle_macos.inspect", return_value=copy.deepcopy(self.records["reco"])):
            with self.assertRaisesRegex(ValueError, "collision"):
                plan([self.binary, other], [self.libs], "arm64", "14.0")

    def test_existing_output_is_never_modified(self):
        self.output.symlink_to(self.root / "absent")
        with patch("bundle_macos.run") as command:
            with self.assertRaisesRegex(ValueError, "already exists"):
                bundle([self.binary], [self.libs], self.output, "arm64", "14.0")
            command.assert_not_called()

    def test_sign_failure_leaves_no_package_and_preserves_inputs(self):
        with patch("bundle_macos.inspect", side_effect=lambda path, *_: copy.deepcopy(self.records[path.name])), \
             patch("bundle_macos.run", side_effect=subprocess.CalledProcessError(1, "codesign")):
            with self.assertRaises(subprocess.CalledProcessError):
                bundle([self.binary], [self.libs], self.output, "arm64", "14.0")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.binary.read_bytes(), b"application")
        self.assertEqual(self.library.read_bytes(), b"library")
        self.assertEqual(list(self.root.glob(".macos-bundle-*")), [])

    def test_success_requires_verified_rewrites_and_signatures(self):
        rewritten = {}
        calls = []

        def command(*args):
            calls.append(args)
            if args[0] == "install_name_tool":
                path = Path(args[-1])
                record = copy.deepcopy(self.records[path.name])
                record["identity"] = f"@loader_path/{path.name}" if record["identity"] else None
                record["dependencies"] = [f"@loader_path/{self.library.name}" if dep == str(self.library) else dep
                                          for dep in record["dependencies"]]
                record["rpaths"] = ["@loader_path"]
                rewritten[path] = record
            return ""

        with patch("bundle_macos.inspect", side_effect=lambda path, *_: copy.deepcopy(rewritten.get(path, self.records[path.name]))), \
             patch("bundle_macos.run", side_effect=command):
            bundle([self.binary], [self.libs], self.output, "arm64", "14.0")
        self.assertTrue((self.output / "mach-o.json").is_file())
        self.assertEqual(sum(call[:3] == ("codesign", "--verify", "--strict") for call in calls), 2)
        self.assertFalse(any("--remove-signature" in call for call in calls))


if __name__ == "__main__":
    unittest.main()
