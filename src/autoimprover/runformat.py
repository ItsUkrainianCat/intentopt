"""The JSON formats of the run folder files and the rules for reading them (ADR-007): pure
functions over decoded JSON, no file access; `runstore.py`, their only importer, does all reading
and writing (SPEC R22).

Every field is checked against its exact JSON type and every object must have exactly its fields,
so a damaged or foreign file is never half-read. A parser raises `ValueError`, `KeyError` or
`TypeError` for a damaged document; `runstore.py` turns that into its own error or, for a cache
entry, a miss. The cache key is the SHA-256 of all `Call` fields (SPEC R17, R24; ADR-004).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from autoimprover.types import Call, Check, Contract, Models, Plan, Reply, Scenario

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheEntry:
    """A stored call: a reply (`outcome` "ok") or a tombstone of a call that failed inside the
    search (`outcome` "failed", `reply` None), with the duration of the original call."""

    outcome: Literal["ok", "failed"]
    reply: Reply | None
    error: str
    duration_s: float


def cache_key(call: Call) -> str:
    """SHA-256 of the canonical JSON of every `Call` field plus the schema version (ADR-007)."""
    return key_hash(dataclasses.asdict(call))


def key_hash(fields: object) -> str:
    """The cache key of a cache entry's stored key fields."""
    text = json.dumps(
        {"schema_version": SCHEMA_VERSION, "call": fields}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode()).hexdigest()


def newer_version(doc: object) -> int | None:
    """The document's schema version when a newer tool wrote it, else None."""
    version = doc.get("schema_version") if type(doc) is dict else None
    return version if type(version) is int and version > SCHEMA_VERSION else None


# Field specs: a JSON type, or one of the named kinds that `_value` checks.
_VERSION, _COUNT, _SECONDS, _TEXTS, _OPTIONAL = "version", "count", "seconds", "texts", "optional"
_MANIFEST_SPEC = {"schema_version": _VERSION, "prompt": str, "plan": dict, "opts": dict}
_MANIFEST_SPEC |= {"created": str, "elapsed_s": _SECONDS, "calls_used": _COUNT}
_PLAN_SPEC = {"models": dict, "strictness": str, "budget": int, "wall_clock_s": int}
_PLAN_SPEC |= {"allow_growth": bool, "merge": bool, "seed": int}
_MODELS_SPEC: dict[str, object] = dict.fromkeys(("task", "judge", "reflect", "target"), str)
_CONTRACT_SPEC = {"goal": str, "kind": str, "keep": _TEXTS, "constraints": _TEXTS, "checks": list}
_CONTRACT_SPEC |= dict.fromkeys(("output_format", "language", "tone"), str)
_CHECK_SPEC = {"id": str, "group": str, "text": str, "rule": _OPTIONAL, "arg": _OPTIONAL}
_SCENARIO_SPEC = {"id": str, "input": str, "expected": _OPTIONAL, "criteria": _TEXTS}
_REPLY_SPEC = {"text": str, "tokens_in": _COUNT, "tokens_out": _COUNT}
_ENTRY_SPEC = {"schema_version": _VERSION, "key_fields": dict, "outcome": str}
_ENTRY_SPEC |= {"duration_s": _SECONDS}


def parse_manifest(doc: Any) -> tuple[dict[str, Any], Plan]:
    """The manifest (with `elapsed_s` as a float) and the `Plan` it holds."""
    manifest = _typed(doc, _MANIFEST_SPEC)
    plan = _typed(manifest["plan"], _PLAN_SPEC)
    models = Models(**_typed(plan["models"], _MODELS_SPEC))
    return manifest, Plan(**{**plan, "models": models})


def parse_checkpoint(doc: Any) -> tuple[int, float]:
    """`search_start`: the call count and clock reading when the search first began."""
    start = _typed(doc, {"schema_version": _VERSION, "search_start": dict})["search_start"]
    start = _typed(start, {"used": _COUNT, "elapsed": _SECONDS})
    return start["used"], start["elapsed"]


def parse_contract(doc: Any) -> Contract:
    body = _typed(doc, {"schema_version": _VERSION, "contract": dict})["contract"]
    body = _typed(body, _CONTRACT_SPEC)
    checks = tuple(Check(**_typed(check, _CHECK_SPEC)) for check in body["checks"])
    return Contract(**{**body, "checks": checks})


def parse_scenarios(doc: Any) -> list[Scenario]:
    items = _typed(doc, {"schema_version": _VERSION, "scenarios": list})["scenarios"]
    return [Scenario(**_typed(item, _SCENARIO_SPEC)) for item in items]


def parse_cache_entry(doc: Any, key: str) -> CacheEntry | None:
    """The entry stored under `key`, or None when it is damaged, foreign or stored under the
    wrong name (its key fields do not hash to `key`)."""
    try:
        if key_hash(doc["key_fields"]) != key:
            return None
        if doc["outcome"] == "ok":
            entry = _typed(doc, {**_ENTRY_SPEC, "reply": dict})
            fields = _typed(entry["reply"], _REPLY_SPEC)
            reply = Reply(**fields, cached=True, duration_s=entry["duration_s"])
            return CacheEntry("ok", reply, "", reply.duration_s)
        if doc["outcome"] == "failed":
            entry = _typed(doc, {**_ENTRY_SPEC, "error": str})
            return CacheEntry("failed", None, entry["error"], entry["duration_s"])
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        return None  # damaged or foreign: a miss (ADR-007)
    return None


def _typed(doc: Any, spec: Mapping[str, object]) -> dict[str, Any]:
    """`doc` as an object with exactly the fields of `spec`, each of the kind it names."""
    if type(doc) is not dict or set(doc) != set(spec):
        raise ValueError(f"expected an object with the fields {', '.join(spec)}")
    return {key: _value(key, doc[key], kind) for key, kind in spec.items()}


def _value(key: str, value: Any, kind: object) -> Any:
    """`value` if it is of `kind`: a JSON type (a bool is not an int here) or a named kind."""
    of = type(value)  # exact JSON types: a bool is not an int
    if kind == _TEXTS and of is list and all(type(v) is str for v in value):
        return tuple(value)
    if kind == _SECONDS and of in (int, float) and math.isfinite(value) and value >= 0:
        return float(value)
    if (
        of is kind
        or (kind == _VERSION and of is int and value == SCHEMA_VERSION)
        or (kind == _COUNT and of is int and value >= 0)
        or (kind == _OPTIONAL and of in (str, type(None)))
    ):
        return value
    raise ValueError(f"{key} is not a valid {getattr(kind, '__name__', kind)}")
