"""Rehearse the release build so a broken release fails on the pull request.

On a push to main, .github/workflows/release.yml lets python-semantic-release
stamp the next version into the files listed in ``[tool.semantic_release]`` and
then run its ``build_command``, which pins the intra-workspace dependencies and
re-resolves uv.lock. Pull requests only check the committed lock with
``uv sync --locked``, which doesn't re-resolve it, so a lock that no longer
resolves (a new interpreter in uv's universe, a newer uv, a yanked pin) used to
surface only after merge, as a failed release.

This script replays that step on the current tree. It stamps a version the way
semantic-release does, runs ``build_command`` through bash with the same
restricted environment, then builds every package the way the deploy job does
and checks that each wheel pins its siblings to the stamped version. It edits
the working tree, so run it in CI or a throwaway worktree. Pure standard
library; ``build_command`` installs its own uv.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# What semantic-release passes to build_command besides NEW_VERSION and
# PACKAGE_NAME (semantic_release.cli.commands.version.build_distributions).
PASSED_THROUGH = ("PATH", "HOME", "VIRTUAL_ENV", "CI", "GITHUB_ACTIONS")


def _next_version(current: str) -> str:
    # Any other version forces the re-resolution a real release performs; bump
    # the minor, as a pre-1.0 release does.
    major, minor, *_ = current.split(".")
    return f"{major}.{int(minor) + 1}.0"


def _stamp_toml(path: Path, key: str, version: str) -> None:
    if key != "project.version":
        raise SystemExit(f"error: {path}: can only stamp project.version, not {key}")
    text = path.read_text()
    table = re.search(r"(?ms)^\[project\]\s*$(.*?)(?=^\[|\Z)", text)
    if table is None:
        raise SystemExit(f"error: {path}: no [project] table")
    body, count = re.subn(
        r'(?m)^(version\s*=\s*)"[^"]*"', rf'\g<1>"{version}"', table.group(1), count=1
    )
    if count != 1:
        raise SystemExit(f"error: {path}: no project.version to stamp")
    path.write_text(text[: table.start(1)] + body + text[table.end(1) :])


def _stamp_variable(path: Path, name: str, version: str) -> None:
    text, count = re.subn(
        rf"""(?m)^({re.escape(name)}\s*=\s*)["'][^"']*["']""",
        rf'\g<1>"{version}"',
        path.read_text(),
        count=1,
    )
    if count != 1:
        raise SystemExit(f"error: {path}: no {name} to stamp")
    path.write_text(text)


def _unpinned_siblings(wheel: Path, siblings: set[str], version: str) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        lines = archive.read(metadata).decode().splitlines()
    requirements = [
        line.split(":", 1)[1].strip() for line in lines if line.startswith("Requires-Dist:")
    ]
    return [
        requirement
        for requirement in requirements
        if (name := re.split(r"[\s;=<>!~\[(]", requirement, maxsplit=1)[0]) in siblings
        and not requirement.startswith(f"{name}=={version}")
    ]


def main() -> int:
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    config = root["tool"]["semantic_release"]
    version = sys.argv[1] if len(sys.argv) > 1 else _next_version(root["project"]["version"])

    print(f"stamping {version}")
    for entry in config["version_toml"]:
        path, key = entry.split(":")[:2]
        _stamp_toml(REPO_ROOT / path, key, version)
    for entry in config["version_variables"]:
        path, name = entry.split(":")[:2]
        _stamp_variable(REPO_ROOT / path, name, version)

    command = config["build_command"]
    print(f"running build_command: {command}", flush=True)
    env = {name: value for name in PASSED_THROUGH if (value := os.environ.get(name)) is not None}
    env |= {"NEW_VERSION": version, "PACKAGE_NAME": root["project"]["name"]}
    if subprocess.run(["bash", "-c", command], cwd=REPO_ROOT, env=env).returncode:
        print("error: build_command failed, so the release would fail too", file=sys.stderr)
        return 1
    subprocess.run(["git", "--no-pager", "diff", "--stat"], cwd=REPO_ROOT)

    packages = sorted(
        tomllib.loads(p.read_text())["project"]["name"]
        for p in REPO_ROOT.glob("packages/*/pyproject.toml")
    )
    problems = []
    with tempfile.TemporaryDirectory() as out:
        for package in packages:
            print(f"building {package}", flush=True)
            build = ["uv", "build", "--package", package, "--out-dir", out]
            if subprocess.run(build, cwd=REPO_ROOT).returncode:
                problems.append(f"{package}: build failed")
                continue
            wheels = list(Path(out).glob(f"{package.replace('-', '_')}-{version}-*.whl"))
            if len(wheels) != 1:
                problems.append(f"{package}: expected one {version} wheel, found {len(wheels)}")
                continue
            for requirement in _unpinned_siblings(wheels[0], set(packages), version):
                problems.append(f"{package}: {requirement!r} isn't pinned to {version}")

    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if not problems:
        print(f"release {version} would stamp, lock, and build {len(packages)} packages")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
