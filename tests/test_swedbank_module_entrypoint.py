"""The documented Swedbank entry point has to actually run.

CLAUDE.md documents `python -m portfolio.swedbank {show,quotes}` and
`portfolio/swedbank/cli.py` sets `prog="python -m portfolio.swedbank"`, but the
package had no `__main__.py`, so the documented command died with

    No module named portfolio.swedbank.__main__; 'portfolio.swedbank' is a
    package and cannot be directly executed

Second trap: `.gitignore` carries a `_*.py` scratch rule that matches every
leading-underscore file, dunders included. `__init__.py` was rescued with an
explicit negation; `__main__.py` walks into the same rule, so a working entry
point would be committed as nothing at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "portfolio.swedbank", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_module_is_executable_with_dash_m():
    """The exact invocation CLAUDE.md documents."""
    r = _run("--help")
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "cannot be directly executed" not in r.stderr


def test_help_lists_the_documented_subcommands():
    out = _run("--help").stdout
    for cmd in ("show", "quotes", "sync"):
        assert cmd in out, f"{cmd!r} missing from --help: {out}"


def test_bad_subcommand_exits_nonzero_rather_than_traceback():
    r = _run("no-such-subcommand")
    assert r.returncode != 0
    assert "Traceback" not in r.stderr


def _git_would_track(relpath: str) -> bool:
    """True if git offers the path for staging.

    NOT `git check-ignore` — that exits 0 whenever any pattern matches, a
    negation included, so it reports a re-included file as "ignored".
    `ls-files --other --exclude-standard` answers the question we mean:
    untracked *and* not excluded.
    """
    if (_REPO_ROOT / relpath).exists() is False:
        return False
    tracked = (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relpath],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    if tracked:
        return True
    listed = subprocess.run(
        ["git", "ls-files", "--other", "--exclude-standard", relpath],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.split()
    return relpath in listed


def test_entrypoint_is_not_swallowed_by_the_gitignore_scratch_rule():
    """`_*.py` in .gitignore matches __main__.py — negation required."""
    assert _git_would_track("portfolio/swedbank/__main__.py")


def test_package_init_is_also_still_not_ignored():
    """Regression guard on the negation that was already needed once."""
    assert _git_would_track("portfolio/swedbank/__init__.py")
