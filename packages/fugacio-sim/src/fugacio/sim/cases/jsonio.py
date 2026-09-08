"""Strict, bounded JSON and atomic, content-addressed artifact storage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fugacio.sim.cases.quantities import CaseValidationError, number

MAX_JSON_BYTES = 16 * 1024 * 1024


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaseValidationError("json", f"duplicate key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise CaseValidationError("json", f"nonfinite constant {value!r}")


def canonical_json(value: Any) -> str:
    """Canonical UTF-8-compatible JSON used for artifact identities."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    )


def digest(value: Any) -> str:
    """SHA-256 identity over canonical JSON, independent of mapping insertion order."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def loads(raw: str) -> Any:
    """Decode strict JSON, rejecting duplicate keys, nonfinite values, and oversized inputs."""
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise CaseValidationError("json", "document exceeds 16 MiB")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=lambda value: number(float(value), "json"),
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CaseValidationError("json", str(exc)) from exc


def read_json(path: str | Path) -> Any:
    """Read a bounded UTF-8 document."""
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise CaseValidationError("json", "document exceeds 16 MiB")
    return loads(raw.decode("utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON file after validating its complete serialized contents."""
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raw = canonical_json(value) + "\n"
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise CaseValidationError("json", "document exceeds 16 MiB")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".fugacio-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
