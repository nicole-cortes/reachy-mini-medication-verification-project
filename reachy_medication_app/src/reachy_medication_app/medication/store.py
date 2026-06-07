"""In-memory medication store for the demo.

For Friday's demo we don't need a database — just a dict keyed by drug name
that the tools can read/write. This intentionally lives at module level so
all tool calls within a single app run see the same data.

Future work: replace with a sqlite-backed store + per-user keying.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, asdict, field
import json
import os
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


@dataclass
class Medication:
    """A medication the user has added to Reachy's memory."""

    drug_name: str                  # e.g. "Lisinopril"
    strength: str                   # e.g. "10mg"
    instructions: str               # e.g. "Take 1 tablet by mouth once daily with food"
    appearance: str                 # e.g. "Small white round tablet"
    dosage_count: int               # how many pills per dose, e.g. 1
    frequency: str                  # e.g. "once daily", "twice daily"
    # Optional fields
    color: str = ""                 # e.g. "white" — split out for quick verification matches
    shape: str = ""                 # e.g. "round" / "oval" / "capsule"
    doctor_override: Optional[str] = None  # If user said "doctor told me different from label", stored here
    label_original_instructions: str = ""  # The original label text if doctor_override is set
    added_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Module-level singleton ----------------------------------------------------

_lock = threading.Lock()
_medications: dict[str, Medication] = {}
_dose_log: list[dict[str, Any]] = []  # appended whenever the user takes a dose
_loaded = False


def _store_path() -> Path:
    env_path = os.getenv("REACHY_MEDICATION_STORE_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path(__file__).resolve().parents[3] / ".reachy_medications.json"


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    path = _store_path()
    if not path.exists():
        _loaded = True
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not load medication store from %s: %s", path, e)
        _loaded = True
        return

    meds = payload.get("medications", [])
    dose_log = payload.get("dose_log", [])
    with _lock:
        _medications.clear()
        for raw in meds:
            if not isinstance(raw, dict):
                continue
            try:
                med = Medication(**raw)
                _medications[med.drug_name.strip().lower()] = med
            except Exception:
                continue
        _dose_log.clear()
        if isinstance(dose_log, list):
            _dose_log.extend(entry for entry in dose_log if isinstance(entry, dict))
    _loaded = True
    logger.info("Loaded %d medications from %s", len(_medications), path)


def _persist_locked() -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "medications": [med.to_dict() for med in _medications.values()],
            "dose_log": list(_dose_log),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not persist medication store to %s: %s", path, e)


def add_or_update(med: Medication) -> Medication:
    """Add a medication or update if a med with the same name already exists."""
    _ensure_loaded()
    key = med.drug_name.strip().lower()
    if not key:
        raise ValueError("drug_name cannot be empty")
    with _lock:
        _medications[key] = med
        logger.info("medication stored: %s (%s, %s)", med.drug_name, med.strength, med.frequency)
        _persist_locked()
    return med


def get(drug_name: str) -> Optional[Medication]:
    """Return the medication with the closest matching name, or None."""
    _ensure_loaded()
    key = drug_name.strip().lower()
    with _lock:
        # exact match
        if key in _medications:
            return _medications[key]
        # prefix / substring fallback — handles "lisinopril" vs "lisinopril 10mg"
        for stored_key, med in _medications.items():
            if stored_key.startswith(key) or key in stored_key:
                return med
    return None


def list_all() -> list[Medication]:
    _ensure_loaded()
    with _lock:
        return list(_medications.values())


def remove(drug_name: str) -> bool:
    _ensure_loaded()
    key = drug_name.strip().lower()
    with _lock:
        if key in _medications:
            del _medications[key]
            _persist_locked()
            return True
    return False


def log_dose_taken(drug_name: str, notes: str = "") -> dict[str, Any]:
    """Record that a dose was taken (or skipped)."""
    _ensure_loaded()
    entry = {
        "drug_name": drug_name,
        "taken_at": time.time(),
        "taken_at_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
    }
    with _lock:
        _dose_log.append(entry)
        _persist_locked()
    logger.info("dose logged: %s", entry)
    return entry


def get_dose_log() -> list[dict[str, Any]]:
    _ensure_loaded()
    with _lock:
        return list(_dose_log)


def doses_today(drug_name: str, *, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Return dose-log entries for `drug_name` that were logged today (local time).

    Matches drug names the same loose way `get()` does (case-insensitive,
    substring either direction) so "lisinopril" matches a stored "Lisinopril 10mg".
    Results are sorted oldest-first by `taken_at`.
    """
    _ensure_loaded()
    key = drug_name.strip().lower()
    if not key:
        return []
    today = time.strftime("%Y-%m-%d", time.localtime(now if now is not None else time.time()))

    matches: list[dict[str, Any]] = []
    with _lock:
        key_first = key.split()[0] if key.split() else key
        for entry in _dose_log:
            name = str(entry.get("drug_name", "")).strip().lower()
            name_first = name.split()[0] if name.split() else name
            if not (
                name == key
                or name.startswith(key)
                or key in name
                or (name_first and name_first == key_first)
            ):
                continue
            taken_at = entry.get("taken_at")
            if isinstance(taken_at, (int, float)):
                entry_day = time.strftime("%Y-%m-%d", time.localtime(taken_at))
            else:
                # Fallback to the stored "%Y-%m-%d %H:%M:%S" string prefix.
                entry_day = str(entry.get("taken_at_str", ""))[:10]
            if entry_day == today:
                matches.append(entry)

    matches.sort(key=lambda e: e.get("taken_at", 0))
    return matches


def clear_all() -> None:
    """For testing only."""
    with _lock:
        _medications.clear()
        _dose_log.clear()
        _persist_locked()
