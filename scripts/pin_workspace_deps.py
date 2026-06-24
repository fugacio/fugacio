"""Pin Fugacio's inter-package dependencies to the release version (lockstep).

python-semantic-release stamps each package's ``project.version`` during a
release, but it does not touch dependency specifiers, so the published wheels
would otherwise depend on *unpinned* sibling packages (e.g. ``fugacio-sim``
requiring any ``fugacio-thermo``). This script rewrites every intra-workspace
dependency to ``==<version>`` so the three packages are pinned to one another at
the shared release version, the same way the Hydrateless (npm ``^version``) and
WeaveFFI (cargo ``version``) monorepos pin their internal dependencies.

It runs from ``[tool.semantic_release]`` ``build_command`` with ``NEW_VERSION``
in the environment, and also accepts the version as the first CLI argument for
local testing. Pure standard library so it runs under the semantic-release
container's interpreter without installing anything.

Switch the ``==`` below to ``~=`` if you'd prefer the looser "compatible
release" range (the PEP 440 equivalent of the caret ranges the other two repos
use) instead of an exact pin.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"
OPERATOR = "=="


def _project_name(pyproject_text: str) -> str | None:
    match = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"', pyproject_text)
    return match.group(1) if match else None


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NEW_VERSION", "")
    version = raw.strip().lstrip("v")
    if not version:
        print("error: no version (pass as argv[1] or set NEW_VERSION)", file=sys.stderr)
        return 1

    texts = {p: p.read_text() for p in sorted(PACKAGES_DIR.glob("*/pyproject.toml"))}
    internal = {name for t in texts.values() if (name := _project_name(t))}

    for path, text in texts.items():
        own = _project_name(text)
        updated = text
        # Pin every sibling (never the package's own name, which also appears in
        # a quoted `name = "..."` field). A dependency entry is a quoted string
        # that begins with the distribution name, optionally already carrying a
        # version specifier; `[tool.uv.sources]` keys are bare, so they're left
        # untouched.
        for name in sorted(internal - {own}):
            pattern = re.compile(r'"' + re.escape(name) + r'(?:[=<>!~][^"]*)?"')
            updated = pattern.sub(f'"{name}{OPERATOR}{version}"', updated)
        if updated != text:
            path.write_text(updated)
            print(f"pinned internal deps in {path.relative_to(REPO_ROOT)} -> {OPERATOR}{version}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
