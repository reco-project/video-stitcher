"""Identify copied Debian libraries and retain notices and exact source identities."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def query(*args: str) -> str:
    return subprocess.check_output(["dpkg-query", *args], text=True, stderr=subprocess.PIPE,
                                   env={**os.environ, "LC_ALL": "C.UTF-8"})


def debian_owner(path: Path) -> dict:
    """Require one exact owner, allowing only an equivalent usr-merge path."""
    path = path.resolve(strict=True)
    candidates = [path]
    if path.is_relative_to("/usr/lib"):
        alias = Path("/") / path.relative_to("/usr")
        if alias.exists() and alias.samefile(path):
            candidates.append(alias)
    for candidate in candidates:
        try:
            lines = query("-S", str(candidate)).splitlines()
        except subprocess.CalledProcessError as error:
            if error.returncode == 1:
                continue
            raise
        owners = [line.removesuffix(f": {candidate}") for line in lines if line.endswith(f": {candidate}")]
        if len(lines) != 1 or len(owners) != 1 or not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*(?::[a-z0-9]+)?", owners[0]):
            raise ValueError(f"ambiguous or diverted library owner: {path}")
        fields = query("-W", "-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\t${db:Status-Status}", owners[0]).split("\t")
        if len(fields) != 5 or not all(fields) or fields[4] != "installed":
            raise ValueError(f"incomplete installed package identity: {path}")
        # Do not attribute a locally replaced library to the distribution's sources.
        checksums = query("--control-show", owners[0], "md5sums")
        expected = [line.split(maxsplit=1)[0] for line in checksums.splitlines()
                    if len(line.split(maxsplit=1)) == 2 and line.split(maxsplit=1)[1] == str(candidate).lstrip("/")]
        with path.open("rb") as library:
            actual = hashlib.file_digest(library, "md5").hexdigest()
        if expected != [actual]:
            raise ValueError(f"library differs from installed package checksums: {path}")
        return dict(zip(("package", "version", "source_package", "source_version"), fields[:4]))
    raise ValueError(f"no installed package owns library: {path}")


def collect(inputs: dict[str, Path], supplied: dict[Path, str], destination: Path):
    """Copy notices for a bundle closure; supplied components have separate provenance."""
    destination.mkdir()
    packages = {}
    libraries = {}
    for name, path in sorted(inputs.items()):
        path = path.resolve(strict=True)
        if path in supplied:
            libraries[name] = {"component": supplied[path]}
            continue
        owner = debian_owner(path)
        package = owner["package"]
        libraries[name] = {"package": package}
        if package not in packages:
            notice = Path("/usr/share/doc") / package.split(":")[0] / "copyright"
            target = destination / package
            target.mkdir()
            shutil.copyfile(notice, target / "copyright")
            packages[package] = owner
    shutil.copytree("/usr/share/common-licenses", destination / "common-licenses")
    receipt = {"libraries": libraries, "packages": packages}
    (destination / "dependencies.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
