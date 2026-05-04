"""Regression tests for the 2026-04-18 entry-gate hardening.

Post-mortem for MINI L SILVER AVA 336 (BUY 2026-04-17T16:02 @ 15.58,
SELL 19:50 @ 14.79, -5.07%). The standard entry gates passed at the
edge (conf 0.66 just above 0.60 threshold; RSI 68 at RSI_ENTRY_HIGH;
MACD +0.015 barely positive after decaying from +0.22). Three new
gates address the equation bugs that allowed the entry:

- Gate A: confidence persistence (2 cycles ≥ threshold)
- Gate B: MACD decay ratio vs recent peak
- Gate C: RSI slope OR recent-dip check

See docs/plans/2026-04-18-entry-gates.md for the full debate including
steelman counter-arguments.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_swing_config as cfg
import metals_swing_trader as mst


def _make_trader():
    trader = mst.SwingTrader.__new__(mst.SwingTrader)
    trader.state = mst._default_state()
    trader.regime_history = {"XAG-USD": [("BUY", "trending-up")] * 3}
    trader.state["macd_history"] = {"XAG-USD": [0.1, 0.11, 0.12, 0.13]}
    trader.state["confidence_history"] = {"XAG-USD": [
        {"action": "BUY", "confidence": 0.7},
        {"action": "BUY", "confidence": 0.7},
        {"action": "BUY", "confidence": 0.7},
    ]}
    trader.state["rsi_history"] = {"XAG-USD": [45, 50, 52, 53, 54, 55]}
    trader.check_count = 10
    return trader


def _signal(
    *,
    confidence=0.70,
    buy_count=5,
    sell_count=1,
    rsi=55.0,
    macd=0.12,
    macd_hist=0.05,
    regime="trending-up",
    action="BUY",
):
    return {
        "action": action,
        "confidence": confidence,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "rsi": rsi,
        "macd": macd,
        "macd_hist": macd_hist,
        "regime": regime,
        "timeframes": {
            "Now": "BUY", "12h": "BUY", "2d": "BUY",
            "7d": "HOLD", "1mo": "HOLD", "3mo": "HOLD", "6mo": "BUY",
        },  # 4/7 BUY → passes MIN_BUY_TF_RATIO=0.43
    }


# ---------------------------------------------------------------------------
# Baseline: a well-configured standard entry should still pass
# ---------------------------------------------------------------------------

def test_baseline_standard_entry_passes(monkeypatch):
    """Sanity: the hardening must not block a clean standard entry."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None  # no momentum override
    ok, reason = trader._evaluate_entry(_signal(), "XAG-USD")
    assert ok, f"baseline entry should pass; got: {reason}"


# ---------------------------------------------------------------------------
# Gate A — signal persistence
# ---------------------------------------------------------------------------

def test_gate_A_blocks_single_cycle_phantom_spike(monkeypatch):
    """Phantom spike: conf was 0.40 last cycle, jumped to 0.70 this cycle.
    Gate A rejects because prior cycle(s) were below threshold."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "BUY", "confidence": 0.40},
    ]
    ok, reason = trader._evaluate_entry(_signal(confidence=0.70), "XAG-USD")
    assert not ok
    assert "signal persistence" in reason


def test_gate_A_passes_with_two_consecutive_strong_cycles(monkeypatch):
    """Persistent signal: prior cycle also above threshold → passes."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "BUY", "confidence": 0.65},
        {"action": "BUY", "confidence": 0.70},
    ]
    ok, reason = trader._evaluate_entry(_signal(confidence=0.70), "XAG-USD")
    assert ok, f"persistent signal should pass; got: {reason}"


def test_gate_A_cold_start_passes(monkeypatch):
    """First cycle after restart: no history. Can't pay the 2-cycle cost
    on cold start → allow."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    trader.state["confidence_history"]["XAG-USD"] = []
    ok, reason = trader._evaluate_entry(_signal(confidence=0.70), "XAG-USD")
    assert ok, f"cold-start entry should pass; got: {reason}"


def test_gate_A_bypassed_by_momentum_override(monkeypatch):
    """Momentum override path skips Gate A — fast moves can't wait."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    # Fresh momentum candidate for XAG-USD (LONG)
    trader._check_momentum_candidate = lambda t: {"direction": "LONG", "ts": 0}
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "BUY", "confidence": 0.40},
    ]
    # Momentum threshold is 0.50 (relaxed); confidence 0.55 passes
    ok, reason = trader._evaluate_entry(_signal(confidence=0.55), "XAG-USD")
    assert ok, f"momentum override should bypass Gate A; got: {reason}"


# ---------------------------------------------------------------------------
# Gate B — MACD decay vs recent peak
# ---------------------------------------------------------------------------

def test_gate_B_blocks_on_macd_decay(monkeypatch):
    """MACD peaked at 0.22 earlier, now at 0.015 = 7% of peak. Reject."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    # MACD history with a prior peak at 0.22 and current = 0.015
    # Need 2 improving checks for the existing MACD-improving gate to pass
    trader.state["macd_history"]["XAG-USD"] = [
        0.22, 0.21, 0.18, 0.15, 0.10, 0.05, 0.01, 0.001, 0.008, 0.015
    ]
    ok, reason = trader._evaluate_entry(_signal(macd=0.015), "XAG-USD")
    assert not ok
    assert "MACD decay" in reason


def test_gate_B_passes_when_macd_near_peak(monkeypatch):
    """MACD at 0.18 with recent peak 0.22 = 82% of peak. Above 30% floor."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    trader.state["macd_history"]["XAG-USD"] = [
        0.15, 0.16, 0.18, 0.20, 0.21, 0.22, 0.20, 0.19, 0.17, 0.18
    ]
    # Codex 2026-04-18: gate reads current sig's macd_hist, not prior-cycle
    # state history. Set macd_hist=0.18 to match the "near peak" intent.
    ok, reason = trader._evaluate_entry(_signal(macd=0.18, macd_hist=0.18), "XAG-USD")
    assert ok, f"MACD near peak should pass; got: {reason}"


def test_gate_B_cold_start_passes(monkeypatch):
    """Fewer than 2 MACD samples → skip decay check."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    trader.state["macd_history"]["XAG-USD"] = [0.015]  # only 1 sample
    ok, reason = trader._evaluate_entry(_signal(macd=0.015), "XAG-USD")
    assert ok, f"cold-start should pass Gate B; got: {reason}"


# ---------------------------------------------------------------------------
# Gate C — RSI slope / recent-dip
# ---------------------------------------------------------------------------

def test_gate_C_blocks_when_rsi_falling_from_high(monkeypatch):
    """2026-04-17 pattern: RSI 79 → 77 → 75 → 72 → 68 (declining, never
    dipped below 55). Both slope and dip check fail."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    # Seed RSI history with 79→72 over 5 cycles; current at 68 (still falling)
    trader.state["rsi_history"]["XAG-USD"] = [79, 77, 75, 73, 72, 70, 68]
    ok, reason = trader._evaluate_entry(_signal(rsi=68.0), "XAG-USD")
    assert not ok
    assert "RSI no slope/dip" in reason


def test_gate_C_passes_when_rsi_rising_through(monkeypatch):
    """Healthy breakout: RSI 50 → 55 → 62 → 68 (rising through)."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    trader.state["rsi_history"]["XAG-USD"] = [50, 52, 55, 60, 62, 65]
    # Current RSI 68, 5 cycles ago was 52 → rising, slope_ok
    ok, reason = trader._evaluate_entry(_signal(rsi=68.0), "XAG-USD")
    assert ok, f"rising RSI should pass Gate C; got: {reason}"


def test_gate_C_passes_on_recent_dip(monkeypatch):
    """Pullback-and-recover: RSI touched 45 recently, now at 65 but slope is
    down from 70. The dip_ok branch saves it."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    # RSI had a dip to 45 about 10 cycles ago, now at 65 but 5-cycle ago was 70
    trader.state["rsi_history"]["XAG-USD"] = [70, 65, 55, 45, 50, 60, 68, 70, 68, 65]
    ok, reason = trader._evaluate_entry(_signal(rsi=65.0), "XAG-USD")
    assert ok, f"recent-dip pullback should pass Gate C; got: {reason}"


# ---------------------------------------------------------------------------
# Counterfactual — reconstruct the 2026-04-17 failure and verify it is blocked
# ---------------------------------------------------------------------------

def test_counterfactual_2026_04_17_silver_blocked(monkeypatch):
    """Reconstruct the MINI L SILVER AVA 336 entry (2026-04-17T16:02):
    conf 0.66, RSI 68, MACD 0.015, regime trending-up. MACD history had
    peaked at 0.22 ~60 min earlier. Expected: at least Gate B (decay)
    blocks the entry."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 18.0)  # 16:02 UTC = 18:02 CET
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    # Real MACD trace for XAG on 2026-04-17 15:00-16:02
    trader.state["macd_history"]["XAG-USD"] = [
        0.22, 0.221, 0.216, 0.215, 0.215, 0.20, 0.15, 0.08, 0.03, 0.008, 0.001, 0.015
    ]
    # Real confidence trace (0.40→0.80→0.66 oscillating), all BUY action
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "BUY", "confidence": c}
        for c in [0.40, 0.37, 0.32, 0.37, 0.74, 0.66, 0.66, 0.80, 0.80, 0.66]
    ]
    # Real RSI trace: 79 → 68 (declining through the hour)
    trader.state["rsi_history"]["XAG-USD"] = [
        79, 79, 77, 75, 75, 72, 68, 68, 67, 67, 66, 68
    ]
    ok, reason = trader._evaluate_entry(
        _signal(confidence=0.66, rsi=68.0, macd=0.015, macd_hist=0.015),
        "XAG-USD",
    )
    assert not ok, "yesterday's trade should have been blocked by the new gates"
    # Accept rejection for any of A/B/C (order-dependent on which fires first).
    assert any(tag in reason for tag in (
        "MACD decay", "signal persistence", "RSI no slope/dip"
    )), f"expected A/B/C rejection; got: {reason}"


# ---------------------------------------------------------------------------
# History buffers — update methods append + cap correctly
# ---------------------------------------------------------------------------

def test_update_confidence_history_caps_at_max(monkeypatch):
    trader = _make_trader()
    trader.state["confidence_history"] = {}
    monkeypatch.setattr(mst, "_save_state", lambda *a, **k: None)
    # Feed more than the cap
    for i in range(cfg.CONFIDENCE_HISTORY_MAX + 5):
        sig_data = {"XAG-USD": _signal(confidence=0.60 + i * 0.001)}
        trader._update_confidence_history(sig_data)
    hist = trader.state["confidence_history"]["XAG-USD"]
    assert len(hist) == cfg.CONFIDENCE_HISTORY_MAX
    # Entries are dicts with action + confidence (Codex 2026-04-18 follow-up).
    assert isinstance(hist[-1], dict)
    assert hist[-1]["action"] == "BUY"
    last_conf = hist[-1]["confidence"]
    assert abs(last_conf - (0.60 + (cfg.CONFIDENCE_HISTORY_MAX + 4) * 0.001)) < 1e-6


def test_gate_A_blocks_action_flip_even_with_high_confidence(monkeypatch):
    """Codex 2026-04-18: a SELL@0.70 followed by BUY@0.70 is a flip, not
    persistence. Prior action != current action → reject. Tests the per-
    cycle action tracking added on top of the numeric-confidence gate."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "SELL", "confidence": 0.70},  # prior was SELL with high conf
    ]
    ok, reason = trader._evaluate_entry(_signal(confidence=0.70, action="BUY"), "XAG-USD")
    assert not ok
    assert "signal persistence" in reason


def test_gate_A_tolerates_legacy_float_entries(monkeypatch):
    """Migration window: a state file from the prior build has legacy
    float entries in confidence_history. We must not crash; legacy
    entries are treated as same-action (benefit of the doubt for one
    cycle until the ring gets overwritten)."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    trader.state["confidence_history"]["XAG-USD"] = [0.70]  # legacy float
    ok, reason = trader._evaluate_entry(_signal(confidence=0.70), "XAG-USD")
    assert ok, f"legacy entries must not crash; got: {reason}"


def test_update_rsi_history_caps_at_max(monkeypatch):
    trader = _make_trader()
    trader.state["rsi_history"] = {}
    monkeypatch.setattr(mst, "_save_state", lambda *a, **k: None)
    for i in range(cfg.RSI_HISTORY_MAX + 5):
        sig_data = {"XAG-USD": _signal(rsi=40.0 + i)}
        trader._update_rsi_history(sig_data)
    hist = trader.state["rsi_history"]["XAG-USD"]
    assert len(hist) == cfg.RSI_HISTORY_MAX
    assert hist[-1] == 40.0 + (cfg.RSI_HISTORY_MAX + 4)


# ---------------------------------------------------------------------------
# Config-constant floors (prevent silent tune-down regressions)
# ---------------------------------------------------------------------------

def test_config_constants_pinned():
    """Pin the hardening thresholds so future tune-downs don't quietly
    re-open the incident territory."""
    assert cfg.SIGNAL_PERSISTENCE_CHECKS >= 2, "persistence must be ≥ 2 cycles"
    assert 0.1 <= cfg.MACD_DECAY_MIN_RATIO <= 0.5, (
        "MACD decay ratio in [0.1, 0.5]; 0.3 is the original setpoint"
    )
    assert cfg.MACD_DECAY_PEAK_LOOKBACK >= 10, "need at least 10 cycles of MACD history"
    assert 1 <= cfg.RSI_SLOPE_LOOKBACK_CHECKS <= 10
    assert cfg.RSI_DIP_BELOW_LEVEL <= 60, (
        "dip level must be below the usual RSI_ENTRY_HIGH cap"
    )


# ---------------------------------------------------------------------------
# MIN_BUY_CONFIDENCE — 2026-05-04 threshold reanchor (0.60 -> 0.56)
# ---------------------------------------------------------------------------
# Context: signal_engine Stage 7 calibration compression (added 2026-04-18)
# caps post-everything confidence at 0.685 even on a perfect 7/7 raw
# consensus. The previous 0.60 gate was structurally unclearable for any
# ticker carrying a Stage 6 PTC accuracy penalty. See
# docs/plans/2026-05-04-conf-threshold-fix.md.
#
# These tests pin the new value AND its lower bound so a future tune-down
# either has to update the floor here (forcing a conscious decision) or
# breaks. Stop the gate from drifting back into structurally-unclearable
# territory.

def test_min_buy_confidence_post_calibration_anchor():
    """Pin the post-calibration anchor. Floor protects against accidentally
    raising it back above the calibration ceiling (~0.685); ceiling
    protects against tune-down to 'noise level' that the user explicitly
    rejected (memory: 'don't recommend signal trades below 60% calibrated
    confidence, 55% is not good enough')."""
    # Floor: must not exceed the empirical post-Stage-7 ceiling minus a
    # small headroom for Stage 6 PTC penalties on average tickers
    assert cfg.MIN_BUY_CONFIDENCE <= 0.60, (
        "MIN_BUY_CONFIDENCE > 0.60 is structurally unclearable post-calibration "
        "compression (Stage 7 caps at 0.685 even on perfect raw consensus). "
        "If raising, first verify scripts/perf/backtest_conf_threshold.py shows "
        "trades still firing."
    )
    # Ceiling: must stay above the 'noise floor' where 50% raw consensus
    # leaks through. The user's rule baseline.
    assert cfg.MIN_BUY_CONFIDENCE >= 0.55, (
        "MIN_BUY_CONFIDENCE < 0.55 lets 50%-raw-consensus signals through; "
        "this is below the level the user rejected as noise."
    )


def test_confidence_at_threshold_passes(monkeypatch):
    """conf == MIN_BUY_CONFIDENCE clears the gate (>= comparison, not >)."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    # History at the same level so Gate A persistence passes
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "BUY", "confidence": cfg.MIN_BUY_CONFIDENCE},
        {"action": "BUY", "confidence": cfg.MIN_BUY_CONFIDENCE},
    ]
    ok, reason = trader._evaluate_entry(
        _signal(confidence=cfg.MIN_BUY_CONFIDENCE), "XAG-USD",
    )
    assert ok, f"conf at threshold should pass; got: {reason}"


def test_confidence_just_below_threshold_fails(monkeypatch):
    """conf one tick below the threshold rejects with the confidence reason."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    just_below = cfg.MIN_BUY_CONFIDENCE - 0.01
    ok, reason = trader._evaluate_entry(
        _signal(confidence=just_below), "XAG-USD",
    )
    assert not ok
    assert "confidence" in reason
    assert f"{just_below:.2f}" in reason
    assert f"{cfg.MIN_BUY_CONFIDENCE}" in reason


def test_realistic_ranging_regime_consensus_now_clears(monkeypatch):
    """The prototypical case the 0.56 threshold is meant to unblock:
    6/7 BUY voters in ranging regime with mild Stage-6 PTC penalty leaves
    final conf at 0.58 — above 0.56 but below 0.60. This is what was
    previously SKIP_BUY-ing every cycle and producing total_trades=0.

    Codex 2026-05-04 P3: explicitly set buy_count=6, sell_count=1 so the
    docstring "6/7" matches the actual signal under test (the _signal()
    default is 5/1).
    """
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader.regime_history = {"XAG-USD": [("BUY", "ranging")] * 3}  # confirmed
    trader._check_momentum_candidate = lambda t: None
    # 0.58 = realistic post-calibration value for 6/7 ranging with mild PTC
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "BUY", "confidence": 0.58},
        {"action": "BUY", "confidence": 0.58},
    ]
    ok, reason = trader._evaluate_entry(
        _signal(confidence=0.58, buy_count=6, sell_count=1, regime="ranging"),
        "XAG-USD",
    )
    assert ok, (
        f"6/7 ranging consensus with conf 0.58 must pass at the 0.56 "
        f"threshold (this is the case the threshold change unblocks); got: {reason}"
    )


@pytest.mark.parametrize("conf,should_pass", [
    (0.559, False),   # one tick below
    (0.560, True),    # exactly at floor (>= comparison)
    (0.561, True),    # one tick above
])
def test_min_buy_confidence_boundary(monkeypatch, conf, should_pass):
    """Codex 2026-05-04 P3: boundary parametrize at 0.559/0.560/0.561 to
    catch off-by-one drift at the exact live threshold."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    # Pre-populate confidence_history at the same level so Gate A persistence
    # passes (otherwise 0.559 lacks a precedent and Gate A blocks for a
    # different reason than the confidence floor).
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "BUY", "confidence": conf},
        {"action": "BUY", "confidence": conf},
    ]
    ok, reason = trader._evaluate_entry(_signal(confidence=conf), "XAG-USD")
    if should_pass:
        assert ok, f"conf {conf} should pass {cfg.MIN_BUY_CONFIDENCE} floor; got: {reason}"
    else:
        assert not ok
        # Reason format is "confidence X.XX < <floor>" (2 decimal rounding,
        # so 0.559 displays as 0.56). Just confirm the reason cites the
        # confidence gate, not some other gate.
        assert "confidence" in reason and str(cfg.MIN_BUY_CONFIDENCE) in reason


def test_confidence_just_above_threshold_passes(monkeypatch):
    """Codex 2026-05-04 P3: explicit just_above_threshold test as promised
    in plan section 'What changes'. conf one tick above the floor passes."""
    monkeypatch.setattr(mst, "_cet_hour", lambda: 14.0)
    trader = _make_trader()
    trader._check_momentum_candidate = lambda t: None
    just_above = round(cfg.MIN_BUY_CONFIDENCE + 0.01, 4)
    trader.state["confidence_history"]["XAG-USD"] = [
        {"action": "BUY", "confidence": just_above},
        {"action": "BUY", "confidence": just_above},
    ]
    ok, reason = trader._evaluate_entry(_signal(confidence=just_above), "XAG-USD")
    assert ok, f"conf just above threshold should pass; got: {reason}"
