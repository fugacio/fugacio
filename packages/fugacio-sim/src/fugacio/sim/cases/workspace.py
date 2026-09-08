"""Atomic, content-addressed case revisions, runs, and studies in a local directory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fugacio.sim.cases.jsonio import read_json, write_json
from fugacio.sim.cases.results import CaseRun, verify_artifact
from fugacio.sim.cases.schema import ProcessCase

_ID = re.compile(r"[0-9a-f]{64}\Z")


class CaseWorkspace:
    """An explicit artifact directory; identifiers never act as arbitrary file paths.

    Cases are immutable revisions identified by their content, so updates don't
    overwrite earlier runs or require a mutable global 'current case' pointer.
    Existing objects are verified before an idempotent save succeeds.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, category: str, identity: str) -> Path:
        if not isinstance(identity, str) or not _ID.fullmatch(identity):
            raise ValueError("artifact IDs must be 64 lowercase hexadecimal characters")
        path = self.root / category / (identity + ".json")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("artifact path escapes workspace")
        return path

    def save_case(self, case: ProcessCase) -> str:
        """Save an immutable case revision and return its ID."""
        path = self._path("cases", case.case_id)
        if path.exists():
            if self.load_case(case.case_id).to_dict() != case.to_dict():
                raise ValueError("case identity collision")
        else:
            write_json(path, case.to_dict())
        return case.case_id

    def load_case(self, case_id: str) -> ProcessCase:
        """Load a revision and verify its filename against its content."""
        case = ProcessCase.load(self._path("cases", case_id))
        if case.case_id != case_id:
            raise ValueError("stored case content hash mismatch")
        return case

    def save_run(self, run: CaseRun) -> str:
        """Persist the case revision and its complete audited result."""
        self.save_case(ProcessCase.from_dict(run.to_dict()["case"]))
        self.save_artifact(run.to_dict())
        return run.run_id

    def load_run(self, run_id: str) -> CaseRun:
        """Load and verify a stored run."""
        return CaseRun.from_dict(self.load_artifact(run_id))

    def save_artifact(self, artifact: dict[str, Any]) -> str:
        """Save a verified run, study, or comparison envelope atomically."""
        d = verify_artifact(artifact)
        identity = d["artifact_id"]
        path = self._path("artifacts", identity)
        if path.exists():
            if self.load_artifact(identity) != d:
                raise ValueError("artifact identity collision")
        else:
            write_json(path, d)
        return identity

    def load_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Read an artifact, checking both envelope and filename identities."""
        d = verify_artifact(read_json(self._path("artifacts", artifact_id)))
        if d["artifact_id"] != artifact_id:
            raise ValueError("stored artifact identity mismatch")
        return d

    def list_cases(self) -> list[dict[str, str]]:
        """List verified case revisions in deterministic identity order."""
        result = []
        for path in sorted((self.root / "cases").glob("*.json")):
            case = self.load_case(path.stem)
            result.append({"case_id": case.case_id, "name": case.name})
        return result

    def replay(self, run_id: str) -> CaseRun:
        """Recompute a stored run with its case, requested values, and recorded policy."""
        from fugacio.sim.cases.quantities import display_value
        from fugacio.sim.cases.runtime import CaseRunner, SolverOptions
        from fugacio.thermo.acceptance import AcceptancePolicy

        d = self.load_run(run_id).to_dict()
        case = ProcessCase.from_dict(d["case"])
        overrides = {
            k: {"value": display_value(v, case.parameters[k].unit), "unit": case.parameters[k].unit}
            for k, v in d["requested_parameters_si"].items()
        }
        run = CaseRunner(
            case, options=SolverOptions(**d["solver"]), policy=AcceptancePolicy(**d["policy"])
        ).run(overrides)
        self.save_run(run)
        return run
