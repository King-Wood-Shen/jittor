"""Durable primitives for a file-backed coordinator feedback inbox.

This module deliberately does not collect machine-specific status, launch workers,
or choose a model. Callers provide events and dispatch callbacks explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterable


def _identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def normalize_inbox(value: Any) -> dict[str, Any]:
    """Upgrade the envelope while treating absent acknowledgements as pending.

    Raises without modifying the caller's file when the envelope is malformed.
    Unknown fields and event payloads are retained verbatim.
    """
    if not isinstance(value, dict) or not isinstance(value.get("events"), dict):
        raise ValueError("invalid inbox envelope; original file must be preserved")
    normalized = dict(value)
    events = dict(value["events"])
    received = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for event_id, original in events.items():
        if not isinstance(original, dict):
            original = {"kind": "schema_error", "content": original}
        row = dict(original)
        row.setdefault("event_id", event_id)
        row.setdefault("received_utc", received)
        row["acknowledged"] = row.get("acknowledged") is True
        events[event_id] = row
    normalized["schema"] = 2
    normalized["events"] = events
    return normalized


def merge_pending(path: Path, incoming: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Durably merge newly collected events and return all unacknowledged rows."""
    path = Path(path)
    with _file_lock(path.with_suffix(path.suffix + ".lock")):
        if path.exists():
            current = normalize_inbox(json.loads(path.read_text(encoding="utf-8")))
        else:
            current = {"schema": 2, "events": {}}
        received = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for event in incoming:
            if not isinstance(event, dict):
                raise ValueError("incoming event must be an object")
            row = dict(event)
            event_id = row.get("event_id") or _identity(row)
            row["event_id"] = event_id
            row.setdefault("received_utc", received)
            row["acknowledged"] = False
            current["events"].setdefault(event_id, row)
        _atomic_json(path, current)
        return [row for row in current["events"].values() if row["acknowledged"] is not True]


def acknowledge(path: Path, event_ids: Iterable[str], decision_path: Path) -> None:
    """Acknowledge known events only after binding them to a saved decision."""
    path, decision_path = Path(path), Path(decision_path)
    decision_hash = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    with _file_lock(path.with_suffix(path.suffix + ".lock")):
        current = normalize_inbox(json.loads(path.read_text(encoding="utf-8")))
        for event_id in event_ids:
            if event_id not in current["events"]:
                raise KeyError(event_id)
            current["events"][event_id].update(
                acknowledged=True,
                decision_sha256=decision_hash,
                decision_path=str(decision_path),
            )
        _atomic_json(path, current)


def retry_call(
    operation: Callable[..., Any],
    *args: Any,
    on_error: Callable[[Exception], None],
    sleep: Callable[[float], None] = time.sleep,
    delay: float = 10,
    retryable: tuple[type[Exception], ...] = (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ),
) -> Any:
    """Retry declared infrastructure/schema faults and report every attempt."""
    while True:
        try:
            return operation(*args)
        except retryable as error:
            on_error(error)
            sleep(delay)


def apply_once(
    ledger_path: Path,
    decision: dict[str, Any],
    apply: Callable[[], Any],
) -> Any:
    """Apply a decision once; an interrupted intent is never silently replayed."""
    ledger_path = Path(ledger_path)
    key = _identity(decision)
    with _file_lock(ledger_path.with_suffix(ledger_path.suffix + ".lock")):
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        else:
            ledger = {}
        existing = ledger.get(key)
        if existing is not None:
            if existing.get("state") != "applied":
                raise RuntimeError("interrupted dispatch intent requires reconciliation")
            return existing.get("result")
        ledger[key] = {"state": "intent"}
        _atomic_json(ledger_path, ledger)
        result = apply()
        ledger[key] = {"state": "applied", "result": result}
        _atomic_json(ledger_path, ledger)
        return result


def dispatch_if_idle(
    sid_is_live: Callable[[], bool],
    sid_lock_is_free: Callable[[], bool],
    dispatch: Callable[[], Any],
) -> tuple[bool, Any]:
    """Fail closed when this SID is live or its worker lock is held."""
    if sid_is_live() or not sid_lock_is_free():
        return False, None
    return True, dispatch()
