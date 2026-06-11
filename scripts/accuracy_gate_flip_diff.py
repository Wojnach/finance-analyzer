#!/usr/bin/env python3
"""Compare two accuracy_cache.json snapshots and report signals whose
accuracy-gate force-HOLD status flips between them.

Premortem hook 10 (B6 audit, 2026-06-11): the B6 signal-core changes can
shift which signals pass the accuracy gate. After the loop restarts and the
cache repopulates, run this against the pre-B6 snapshot to see exactly which
signals crossed the force-HOLD boundary (and by how many pp), so a gate flip
attributable to the change can be told apart from organic accuracy drift.

Gate logic mirrors portfolio/signal_engine._weighted_consensus (the live
path), kept in sync with these constants:
    ACCURACY_GATE_THRESHOLD            = 0.47   # standard tier floor
    ACCURACY_GATE_MIN_SAMPLES          = 30     # need this many before gating
    _ACCURACY_GATE_HIGH_SAMPLE_MIN     = 7000   # established-signal tier
    _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD = 0.50 # tighter floor for that tier
A signal is GATED (force-HOLD) when total >= 30 and accuracy < effective_gate,
where effective_gate is 0.50 for total >= 7000 else 0.47. Directional rescue,
circuit-breaker relaxation and per-ticker gates are NOT modelled here — this
diff reports the base overall-accuracy gate only (the dominant force-HOLD
mechanism), which is what the snapshot cache carries.

Usage:
    python scripts/accuracy_gate_flip_diff.py OLD.json NEW.json [--horizon 1d]

Pure stdlib, read-only. Exits 0 always (it is a report, not a check).
"""
import argparse
import json
import sys

GATE_STANDARD = 0.47
GATE_MIN_SAMPLES = 30
GATE_HIGH_SAMPLE_MIN = 7000
GATE_HIGH_SAMPLE_THRESHOLD = 0.50


def _effective_gate(total):
    """Return the accuracy floor that applies for a signal with `total` samples."""
    if total >= GATE_HIGH_SAMPLE_MIN:
        return GATE_HIGH_SAMPLE_THRESHOLD
    return GATE_STANDARD


def gate_status(stats):
    """(gated: bool, accuracy: float|None, total: int) for one signal's stats.

    gated is True when the signal would be force-HOLD by the overall accuracy
    gate. Returns gated=False when there is too little data to gate (total < 30)
    or accuracy is missing — matching the live path, which only gates with
    >= MIN_SAMPLES samples.
    """
    if not isinstance(stats, dict):
        return False, None, 0
    total = stats.get("total", stats.get("samples", 0)) or 0
    acc = stats.get("accuracy")
    if not isinstance(acc, (int, float)):
        return False, None, total
    if total < GATE_MIN_SAMPLES:
        return False, acc, total
    return acc < _effective_gate(total), acc, total


def _horizon_block(cache, horizon):
    block = cache.get(horizon)
    if not isinstance(block, dict):
        sys.exit(f"horizon {horizon!r} not found or not a dict in snapshot "
                 f"(available: {[k for k in cache if isinstance(cache.get(k), dict)]})")
    return block


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", help="older accuracy_cache.json snapshot (e.g. pre_b6)")
    ap.add_argument("new", help="newer accuracy_cache.json snapshot")
    ap.add_argument("--horizon", default="1d", help="horizon block to compare (default: 1d)")
    args = ap.parse_args(argv)

    with open(args.old, encoding="utf-8") as f:
        old = _horizon_block(json.load(f), args.horizon)
    with open(args.new, encoding="utf-8") as f:
        new = _horizon_block(json.load(f), args.horizon)

    flips = []
    for sig in sorted(set(old) | set(new)):
        og, oa, ot = gate_status(old.get(sig))
        ng, na, nt = gate_status(new.get(sig))
        if og != ng:
            flips.append((sig, og, ng, oa, na, ot, nt))

    print(f"accuracy-gate flip diff  horizon={args.horizon}")
    print(f"  old: {args.old}")
    print(f"  new: {args.new}")
    print(f"  gate: <{GATE_STANDARD:.0%} (>= {GATE_MIN_SAMPLES} sam), "
          f"<{GATE_HIGH_SAMPLE_THRESHOLD:.0%} for >= {GATE_HIGH_SAMPLE_MIN} sam")
    if not flips:
        print("  no gate-status flips.")
        return 0
    print(f"  {len(flips)} signal(s) flipped:")
    for sig, og, ng, oa, na, ot, nt in flips:
        direction = "GATED -> OPEN" if og and not ng else "OPEN -> GATED"
        oa_s = f"{oa*100:.1f}%" if oa is not None else "n/a"
        na_s = f"{na*100:.1f}%" if na is not None else "n/a"
        pp = (f"{(na - oa)*100:+.1f}pp" if oa is not None and na is not None else "n/a")
        print(f"    {sig:<28} {direction:<13} "
              f"acc {oa_s} ({ot} sam) -> {na_s} ({nt} sam)  delta {pp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
