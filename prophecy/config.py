"""Prophecy runtime config + canonical paths.

Single source of truth for: which instruments are enabled, which model + horizons
to use, the soft USD budget alert threshold, and the on-disk locations of every
artifact. Physical dir is ``data/prophecy_runs/`` (NOT ``data/prophecy/`` — see
package docstring / premortem #1).
"""

from __future__ import annotations

import os
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

from prophecy import strategies
from prophecy.schema import HORIZONS

# repo_root / prophecy / config.py  ->  repo_root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROPHECY_DIR = DATA_DIR / "prophecy_runs"

# NOTE: named prophecy_config.json (NOT config.json) — the repo .gitignore has a
# global `config.json` rule guarding the API-key secret symlink; a file literally
# named config.json anywhere would be silently untracked.
CONFIG_FILE = PROPHECY_DIR / "prophecy_config.json"
# Freeze sentinel: if present, the daily .bat exits 0 before any Claude call
# (premortem #3 — this path bypasses claude_gate, so the sentinel is its guard).
FROZEN_SENTINEL = PROPHECY_DIR / "SYSTEM_DISABLED"

DEFAULT_MODEL = "claude-opus-4-8"


def _default_config() -> dict:
    return {
        "model": DEFAULT_MODEL,
        "horizons": list(HORIZONS),
        "budget_usd_soft_cap": None,  # null => unhinged (alert only, never blocks)
        "instruments": {
            inst: {"enabled": True, "strategy": pb.strategy_id, "asset_class": pb.asset_class}
            for inst, pb in strategies.PLAYBOOKS.items()
        },
    }


def context_file(date: str) -> Path:
    return PROPHECY_DIR / f"context_{date}.json"


def raw_file(date: str) -> Path:
    return PROPHECY_DIR / f"raw_{date}.json"


def run_file(date: str) -> Path:
    return PROPHECY_DIR / f"run_{date}.json"


def run_log_file(date: str) -> Path:
    return PROPHECY_DIR / f"run_{date}.log"


JOURNAL_FILE = PROPHECY_DIR / "prediction_journal.jsonl"
LATEST_FILE = PROPHECY_DIR / "latest.json"
ACCURACY_JSONL = PROPHECY_DIR / "accuracy.jsonl"
ACCURACY_FILE = PROPHECY_DIR / "accuracy.json"
COST_LOG = PROPHECY_DIR / "cost_log.jsonl"
QUARANTINE_DIR = PROPHECY_DIR / "quarantine"


def ensure_dirs() -> None:
    PROPHECY_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load config, writing the default file on first run. File overrides defaults.

    Per-instrument blocks merge so a config that predates a new instrument still
    picks the new one up (defaults to enabled).
    """
    ensure_dirs()
    default = _default_config()
    on_disk = load_json(CONFIG_FILE, default=None)
    if not isinstance(on_disk, dict):
        atomic_write_json(CONFIG_FILE, default)
        return default

    merged = {**default, **on_disk}
    inst = dict(default["instruments"])
    for name, block in (on_disk.get("instruments") or {}).items():
        if isinstance(block, dict):
            inst[name] = {**inst.get(name, {}), **block}
    merged["instruments"] = inst
    return merged


def enabled_instruments(config: dict | None = None) -> list[str]:
    """Enabled instruments, in playbook order, skipping unknown/disabled ones."""
    config = config or load_config()
    blocks = config.get("instruments", {})
    return [
        inst for inst in strategies.all_instruments()
        if blocks.get(inst, {}).get("enabled", False) and strategies.playbook_for(inst)
    ]


def is_system_frozen() -> bool:
    """True if the freeze sentinel is present (system-level kill switch)."""
    return FROZEN_SENTINEL.exists()


def model(config: dict | None = None) -> str:
    return (config or load_config()).get("model", DEFAULT_MODEL)


def budget_soft_cap(config: dict | None = None):
    return (config or load_config()).get("budget_usd_soft_cap")
