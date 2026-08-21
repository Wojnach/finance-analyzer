"""This repo is PUBLIC. No tracked file may carry the real Swedbank book.

`data/swedbank_*.json` is gitignored for exactly this reason and that guard has
held. What was never guarded is everything *else*: on 2026-08-21 a session wrote
a book analysis to `docs/REPORT_2026-08-21-swedbank-and-crypto-surge.md` — real
share counts, per-account P&L, account labels — into a directory that is fully
tracked. It was never staged, but it sat one `git add -A` from publication, and
the ignore rule protecting `data/` says nothing about `docs/`.

Ignore rules only cover paths somebody predicted. This test checks content
instead: it greps what git actually tracks for the account labels, so a leak
fails at commit time rather than at disclosure.

Two design notes, both learned by getting it wrong first:

* **Instrument names are not the secret.** A first version grepped for
  "CoinShares XBT Provider", "Astera Labs" and friends and failed on 7 of 7 —
  `portfolio/swedbank/instruments.py` pins those names deliberately, and the
  docs discuss them. The instrument universe is public; the *positions* are
  private. A guard keyed on the public half is a permanent false alarm, which
  is worse than no guard because the next session learns to ignore it.
* **The needles are read from the gitignored snapshot, never hardcoded.**
  Writing the account labels into this file would itself put private metadata
  into the public repo. The test derives them at runtime and skips when the
  book is absent (CI, fresh clone, herc2 without a synced book).

Matching is word-boundary. Substring matching on a short label is useless —
one of the three labels appears inside 36 tracked files as a fragment of
ordinary words like "rotation" and "instantiation".
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SNAPSHOT = _REPO_ROOT / "data" / "swedbank_snapshot.json"

# This file necessarily talks about the mechanism; never flag it.
_ALLOWLIST = {"tests/test_no_real_positions_committed.py"}


def _account_labels():
    """Account labels from the gitignored snapshot, or None if unavailable."""
    try:
        data = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    labels = [
        k for k in (data.get("accounts") or {}) if isinstance(k, str) and k.strip()
    ]
    return labels or None


def _tracked_files_matching_word(word: str):
    """Tracked files containing *word* as a whole word. `git grep` only ever
    searches tracked content, which is exactly the blast radius we care about."""
    r = subprocess.run(
        ["git", "grep", "-I", "-w", "-l", "--", word],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # exit 1 means no match — the passing case.
    return [p for p in r.stdout.split("\n") if p and p not in _ALLOWLIST]


def test_no_tracked_file_names_a_real_account():
    labels = _account_labels()
    if labels is None:
        pytest.skip("no local swedbank snapshot — nothing to derive needles from")
    offenders = {lbl: _tracked_files_matching_word(lbl) for lbl in labels}
    leaked = {lbl: files for lbl, files in offenders.items() if files}
    assert not leaked, (
        "tracked files name real Swedbank accounts: "
        + "; ".join(f"{lbl} in {files}" for lbl, files in leaked.items())
        + ". This repo is public — move the file outside the repo "
        "(e.g. ~/finance-reports/) or add it to .gitignore."
    )


def test_the_data_gitignore_guard_is_still_in_place():
    """Regression guard on the rule that has been holding all along."""
    r = subprocess.run(
        ["git", "check-ignore", "data/swedbank_snapshot.json"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, "data/swedbank_*.json is no longer gitignored"


def test_no_swedbank_data_file_is_tracked():
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "data/swedbank_*"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked = [p for p in out.split("\0") if p and p.endswith(".json")]
    assert not tracked, f"real-position files are tracked: {tracked}"
