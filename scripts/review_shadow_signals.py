"""CLI: shadow-signal review + auto-promotion/retirement.

Usage:
    python scripts/review_shadow_signals.py [--stale-days N] [--seed]
                                            [--promote] [--retire] [--dry-run]

Options:
    --stale-days N   Threshold for "stale" classification (default 30).
    --seed           Run seed_defaults() before reporting. Safe to re-run —
                     never overwrites existing entries.
    --promote        Flip shadows that meet promotion_criteria to status=promoted.
                     Reads accuracy from data/llm_probability_log.jsonl joined
                     with data/llm_probability_outcomes.jsonl.
    --retire         Flip currently-promoted shadows back to retired when their
                     rolling 30d accuracy drops > 0.05 below min_accuracy.
    --dry-run        Print what would happen but do not mutate the registry.

Exit code:
    0  — no stale shadows (or only resolved ones) AND no flips needed.
    1  — one or more shadows are stale or would have been flipped under
         --dry-run. Useful in CI/scheduled-task checks.

Auto-promotion semantics (--promote):
    A shadow becomes "promoted" when ALL of:
      * n_with_outcome >= promotion_criteria.min_samples
      * accuracy >= promotion_criteria.min_accuracy
      * (if specified) join_rate >= 1 - max_missing_outcome_rate
      * brier_mean <= promotion_criteria.max_brier (default 0.66 ≈ uniform;
        2026-05-29) — only enforced once Brier is computed on enough matched
        samples, so a thin sample never blocks on a noisy Brier. A shadow
        whose logged probability distribution scores worse than a uniform
        guess on its directional outcomes must never auto-promote.
    Uses n_with_outcome (matched samples), not raw log count — a HOLD-only
    scaffold that nobody validates against outcomes must not auto-promote.

Auto-retirement semantics (--retire):
    Only fires on signals currently status="promoted". Looks at the last
    30 days of joined rows and retires if accuracy drops by more than 0.05
    below the original min_accuracy threshold. Conservative retire bar
    prevents a single-day blip from undoing a real promotion.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

# Allow running as `python scripts/review_shadow_signals.py` from the repo root
# without requiring an install. Matches the pattern used by other scripts/ files.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.file_utils import load_jsonl  # noqa: E402
from portfolio.llm_probability_log import is_directional_prediction  # noqa: E402
from portfolio.shadow_registry import (  # noqa: E402
    _invalidate_promoted_cache,
    days_in_shadow,
    load_registry,
    resolve_shadow,
    seed_defaults,
    stale_shadows,
)

# 2026-05-29 (brier-ceiling): default Brier ceiling for promotion. 0.66 ≈ a
# uniform 1/3-each guess (multi-class Brier of uniform = 2/3). A shadow whose
# logged distribution scores worse than a coin-flip-ish guess on its directional
# outcomes must never auto-promote. Backward-compat: registry entries lacking
# `promotion_criteria.max_brier` fall back to this value.
_DEFAULT_MAX_BRIER = 0.66

# Action classes for the multi-class Brier accumulation in
# _compute_signal_stats — kept local so the script has no extra import.
_BRIER_ACTIONS = ("BUY", "HOLD", "SELL")


def _compute_signal_stats(window_days: float | None = None) -> dict:
    """Return per-signal {n, n_matched, correct, n_directional, brier_sum}
    from log+outcomes.

    Only DIRECTIONAL predictions (``confidence > 0`` AND
    ``chosen in {BUY, SELL}``) count toward ``n_matched`` and ``correct``.
    Abstain rows (``confidence=0`` from canonical ``_abstain()`` helpers
    in ``portfolio/signals/*``) and HOLD votes (non-information for trade
    direction) are excluded — see
    ``portfolio.llm_probability_log.is_directional_prediction`` for the
    rationale and the 2026-05-18 plan for the failure mode this guards
    against.

    ``n`` still counts every log row including abstains so the missing-
    outcome rate is computed on the full sample, but the promotion gate
    (accuracy = correct / n_matched) only fires when the model actually
    emitted directional predictions.

    2026-05-29 (brier-ceiling): ``brier_sum`` accumulates the multi-class
    Brier score (Σ_c (P(c)-I[c==outcome])^2) over the SAME directional+joined
    rows that feed ``correct``/``n_matched``, so ``brier_mean =
    brier_sum / n_matched`` is the calibration metric the promotion gate
    checks against ``promotion_criteria.max_brier``. This mirrors the
    canonical leaderboard computation in
    ``dashboard/app.py:_compute_llm_leaderboard`` (which must match this
    function) — both score over directional, outcome-joined rows only.

    When ``window_days`` is provided, only log rows within that lookback
    are counted — used by ``--retire`` to focus on recent performance
    instead of all-time accuracy.
    """
    data_dir = _REPO_ROOT / "data"
    log_rows = load_jsonl(str(data_dir / "llm_probability_log.jsonl")) or []
    out_rows = load_jsonl(str(data_dir / "llm_probability_outcomes.jsonl")) or []

    cutoff = None
    if window_days is not None and window_days > 0:
        cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=window_days)

    outcomes = {}
    for row in out_rows:
        if not isinstance(row, dict):
            continue
        key = (row.get("ts"), row.get("signal"), row.get("ticker"), row.get("horizon"))
        outcomes[key] = row.get("outcome")

    per_sig: dict = {}
    for row in log_rows:
        if not isinstance(row, dict):
            continue
        sig = row.get("signal")
        if not sig:
            continue
        if cutoff is not None:
            ts_raw = row.get("ts")
            if not ts_raw:
                continue
            try:
                ts_dt = _dt.datetime.fromisoformat(ts_raw)
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=_dt.UTC)
                if ts_dt < cutoff:
                    continue
            except (TypeError, ValueError):
                continue
        d = per_sig.setdefault(
            sig,
            {"n": 0, "n_matched": 0, "correct": 0, "n_directional": 0,
             "brier_sum": 0.0},
        )
        d["n"] += 1
        if not is_directional_prediction(row):
            continue
        d["n_directional"] += 1
        key = (row.get("ts"), sig, row.get("ticker"), row.get("horizon"))
        actual = outcomes.get(key)
        if actual is None:
            continue
        d["n_matched"] += 1
        if row.get("chosen") == actual:
            d["correct"] += 1
        # 2026-05-29 (brier-ceiling): accumulate multi-class Brier over the
        # logged distribution vs. the realized outcome, mirroring
        # dashboard/app.py:_compute_llm_leaderboard so the gate and the
        # dashboard agree on brier_mean.
        probs = row.get("probs")
        if isinstance(probs, dict) and actual in _BRIER_ACTIONS:
            d["brier_sum"] += sum(
                (float(probs.get(c, 0.0)) - (1.0 if c == actual else 0.0)) ** 2
                for c in _BRIER_ACTIONS
            )
    return per_sig


def _eligible_for_promotion(entry: dict, stats: dict) -> tuple[bool, str]:
    """Return (eligible, reason). reason explains gate failure for printing.

    2026-05-29 (brier-ceiling): in addition to the sample-count / accuracy /
    missing-rate gates, a shadow is rejected when its calibration Brier
    exceeds ``promotion_criteria.max_brier`` (default 0.66 ≈ uniform). The
    Brier is the mean of ``stats['brier_sum'] / matched`` over the same
    directional, outcome-joined rows that feed accuracy. Crucially the Brier
    ceiling only blocks when it is COMPUTED on a real sample (matched >=
    min_samples, or >= 30 when no min_samples is set): a shadow that simply
    hasn't accumulated enough joined outcomes yet must not be blocked on a
    noisy/uncomputable Brier — only on a Brier that is computed and bad.
    """
    if entry.get("status") != "shadow":
        return False, f"status={entry.get('status')}"
    criteria = entry.get("promotion_criteria") or {}
    min_samples = criteria.get("min_samples")
    min_acc = criteria.get("min_accuracy")
    max_missing = criteria.get("max_missing_outcome_rate")
    # Backward-compatible: entries without max_brier fall back to the default.
    max_brier = criteria.get("max_brier", _DEFAULT_MAX_BRIER)
    n = stats.get("n", 0)
    matched = stats.get("n_matched", 0)
    correct = stats.get("correct", 0)
    if min_samples and matched < min_samples:
        return False, f"matched={matched} < min_samples={min_samples}"
    if matched == 0:
        return False, "no outcomes joined yet"
    accuracy = correct / matched
    if min_acc is not None and accuracy < min_acc:
        return False, f"accuracy={accuracy:.3f} < min={min_acc}"
    if max_missing is not None and n > 0:
        missing_rate = 1.0 - (matched / n)
        if missing_rate > max_missing:
            return False, f"missing={missing_rate:.2f} > max={max_missing}"
    # 2026-05-29 (brier-ceiling): reject worse-than-uniform calibration. Only
    # enforce when the Brier is computed on enough matched samples — gate it
    # on the same bar as min_samples (or 30 when min_samples is unset) so a
    # too-thin sample never blocks on a noisy Brier; the accuracy/min_samples
    # checks above already guard thin samples, but `brier_sum` may be absent
    # on stats dicts built before this change, so default it to None.
    if max_brier is not None:
        brier_sum = stats.get("brier_sum")
        brier_min = min_samples if min_samples else 30
        if brier_sum is not None and matched >= brier_min:
            brier_mean = brier_sum / matched
            if brier_mean > max_brier:
                return False, f"brier {brier_mean:.3f} > max {max_brier}"
    return True, f"matched={matched} accuracy={accuracy:.3f} — promote"


def _should_retire(entry: dict, stats: dict) -> tuple[bool, str]:
    """Return (retire, reason). Only fires on already-promoted signals."""
    if entry.get("status") != "promoted":
        return False, ""
    criteria = entry.get("promotion_criteria") or {}
    min_acc = criteria.get("min_accuracy")
    if min_acc is None:
        return False, "no min_accuracy threshold"
    matched = stats.get("n_matched", 0)
    correct = stats.get("correct", 0)
    if matched < 30:
        return False, f"only {matched} recent samples — not enough to retire"
    accuracy = correct / matched
    retire_threshold = min_acc - 0.05
    if accuracy < retire_threshold:
        return True, f"30d accuracy={accuracy:.3f} < {retire_threshold:.3f} — retire"
    return False, f"30d accuracy={accuracy:.3f} ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stale-days", type=int, default=30)
    parser.add_argument(
        "--seed", action="store_true",
        help="Run seed_defaults() before reporting (idempotent).",
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="Flip eligible shadows to status=promoted.",
    )
    parser.add_argument(
        "--retire", action="store_true",
        help="Flip degraded promoted shadows back to status=retired.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without mutating the registry.",
    )
    args = parser.parse_args(argv)

    if args.seed:
        seed_defaults()

    reg = load_registry()
    if not reg["shadows"]:
        print("(registry empty — run with --seed to populate defaults)")
        return 0

    actions_taken = 0

    if args.promote or args.retire:
        all_stats = _compute_signal_stats()
        retire_stats = _compute_signal_stats(window_days=30) if args.retire else {}

        _default_stats = {
            "n": 0, "n_matched": 0, "correct": 0, "n_directional": 0,
            "brier_sum": 0.0,
        }
        for sig, entry in sorted(reg["shadows"].items()):
            stats = all_stats.get(sig, dict(_default_stats))
            if args.promote:
                ok, reason = _eligible_for_promotion(entry, stats)
                if ok:
                    if args.dry_run:
                        print(f"[DRY] would promote {sig}: {reason}")
                    else:
                        resolve_shadow(
                            sig, "promoted",
                            notes=(entry.get("notes", "") +
                                   f"\n[auto-promoted {_dt.datetime.now(_dt.UTC).date()}: {reason}]"),
                        )
                        print(f"[OK] promoted {sig}: {reason}")
                    actions_taken += 1
            if args.retire:
                r_stats = retire_stats.get(sig, dict(_default_stats))
                retire, reason = _should_retire(entry, r_stats)
                if retire:
                    if args.dry_run:
                        print(f"[DRY] would retire {sig}: {reason}")
                    else:
                        resolve_shadow(
                            sig, "retired",
                            notes=(entry.get("notes", "") +
                                   f"\n[auto-retired {_dt.datetime.now(_dt.UTC).date()}: {reason}]"),
                        )
                        print(f"[OK] retired {sig}: {reason}")
                    actions_taken += 1

        # Force signal_engine to pick up flips on its next 60s cache refresh.
        _invalidate_promoted_cache()
        if actions_taken == 0:
            print("(no promotion/retire actions needed)")

        # Reload registry so the stale-shadow report below reflects flips.
        reg = load_registry()

    stale = stale_shadows(stale_days=args.stale_days)
    total = len(reg["shadows"])
    resolved = sum(
        1 for e in reg["shadows"].values() if e.get("status") != "shadow"
    )

    print(f"Shadow registry: {total} total, {resolved} resolved, "
          f"{len(stale)} stale (>{args.stale_days}d in shadow).")
    print()

    for sig, entry in sorted(reg["shadows"].items()):
        status = entry.get("status", "?")
        notes = (entry.get("notes") or "").splitlines()[0]
        if status == "shadow":
            age = days_in_shadow(sig)
            if age is not None:
                marker = "STALE " if age >= args.stale_days else "active"
                print(f"  [{status:>9}] {sig:<28} {marker}  "
                      f"{age:>5.1f}d   {notes[:60]}")
            else:
                print(f"  [{status:>9}] {sig:<28}   ?      "
                      f"?d   {notes[:60]}")
        else:
            print(f"  [{status:>9}] {sig:<28}                    "
                  f"{notes[:60]}")

    if args.dry_run and actions_taken > 0:
        return 1
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
