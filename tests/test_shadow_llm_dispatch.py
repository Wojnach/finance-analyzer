"""Tests for the throttled expensive-LLM-shadow dispatch gate (2026-06-01).

Covers `signal_engine._shadow_llm_runs_now` and `shadow_registry.get_cycle_modulo`
— the mechanism that lets phi4_mini (and future expensive LLM shadows) compute
for accuracy tracking at most ONCE per throttle-tick on a single rotating ticker,
instead of every-cycle-every-ticker (the 5×22s cycle-blowout the FGL premortem
flagged).

xdist-safe: registry access is monkeypatched; no real files, no GPU.
"""
from __future__ import annotations

import pytest

import portfolio.shadow_registry as sr
import portfolio.signal_engine as se


@pytest.fixture
def fake_registry(monkeypatch):
    """phi4_mini in shadow status, cycle_modulo=10, phase=1 (matches prod)."""
    monkeypatch.setattr(sr, "get_status", lambda s, **k: "shadow" if s == "phi4_mini" else None)
    monkeypatch.setattr(sr, "get_cycle_modulo", lambda s, **k: 10 if s == "phi4_mini" else 1)
    monkeypatch.setattr(sr, "should_run_this_cycle", lambda s, c, **k: c % 10 == 1)
    # signal_engine imports these names lazily inside the gate, so patching the
    # source module is sufficient.
    return None


def test_fires_one_ticker_per_throttle_tick(fake_registry):
    rot = se._SHADOW_LLM_ROTATION
    fires = {}
    for cyc in range(0, 60):
        hit = [t for t in rot if se._shadow_llm_runs_now("phi4_mini", t, cyc)]
        if hit:
            fires[cyc] = hit
    # Only phase-1 cycles fire, and exactly one ticker each.
    assert set(fires) == {1, 11, 21, 31, 41, 51}
    assert all(len(v) == 1 for v in fires.values())


def test_rotates_through_all_tickers(fake_registry):
    rot = se._SHADOW_LLM_ROTATION
    ordered = [t for cyc in (1, 11, 21, 31, 41) for t in rot if se._shadow_llm_runs_now("phi4_mini", t, cyc)]
    # BTC, ETH, XAU, XAG, MSTR — every instrument sampled once per full rotation.
    assert ordered == list(rot)
    assert set(ordered) == set(rot)


def test_no_fire_on_off_phase_cycles(fake_registry):
    rot = se._SHADOW_LLM_ROTATION
    for cyc in (0, 2, 5, 9, 10, 12, 20):  # none are ≡1 mod 10
        assert not any(se._shadow_llm_runs_now("phi4_mini", t, cyc) for t in rot)


def test_fail_closed_on_none_ticker(fake_registry):
    assert se._shadow_llm_runs_now("phi4_mini", None, 1) is False


def test_fail_closed_when_not_shadow(monkeypatch):
    # status != "shadow" (e.g. promoted/retired) -> never run via this path.
    monkeypatch.setattr(sr, "get_status", lambda s, **k: "promoted")
    monkeypatch.setattr(sr, "should_run_this_cycle", lambda s, c, **k: True)
    monkeypatch.setattr(sr, "get_cycle_modulo", lambda s, **k: 10)
    assert se._shadow_llm_runs_now("phi4_mini", "BTC-USD", 1) is False


def test_fail_closed_on_registry_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("registry unreadable")
    monkeypatch.setattr(sr, "get_status", _boom)
    # Any exception in the gate -> False (skip the expensive call), NOT True.
    assert se._shadow_llm_runs_now("phi4_mini", "BTC-USD", 1) is False


def test_get_cycle_modulo_default_for_unknown():
    assert sr.get_cycle_modulo("definitely_not_a_signal_xyz") == 1


def test_phi4_in_shadow_llm_set_not_shadow_safe():
    # phi4 must be in the throttled LLM set, NOT the every-cycle math set.
    assert "phi4_mini" in se._SHADOW_LLM_SIGNALS
    assert "phi4_mini" not in se._SHADOW_SAFE_SIGNALS


# ---------------------------------------------------------------------------
# (e) Multi-signal enrollment (2026-06-01): finance_llama + cryptotrader_lm
#     joined phi4_mini in _SHADOW_LLM_SIGNALS. meta_trader stays out (scaffold).
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_registry(monkeypatch):
    """phi4(10,1), finance_llama(3,2), cryptotrader_lm(3,0) — all shadow."""
    mods = {"phi4_mini": 10, "finance_llama": 3, "cryptotrader_lm": 3}
    phases = {"phi4_mini": 1, "finance_llama": 2, "cryptotrader_lm": 0}
    monkeypatch.setattr(sr, "get_status", lambda s, **k: "shadow" if s in mods else None)
    monkeypatch.setattr(sr, "get_cycle_modulo", lambda s, **k: mods.get(s, 1))
    monkeypatch.setattr(sr, "should_run_this_cycle",
                        lambda s, c, **k: s in mods and c % mods[s] == phases[s])
    return None


def test_meta_trader_excluded_scaffold():
    # _FEATURE_AVAILABLE=False scaffold — must NOT be enrolled.
    assert "meta_trader" not in se._SHADOW_LLM_SIGNALS
    assert {"phi4_mini", "finance_llama", "cryptotrader_lm"} <= se._SHADOW_LLM_SIGNALS


def test_cryptotrader_scoped_to_crypto(multi_registry):
    # cryptotrader_lm must never fire on metals/MSTR (its LoRA is BTC/ETH-only).
    sampled = {t for cyc in range(120) for t in se._SHADOW_LLM_ROTATION
               if se._shadow_llm_runs_now("cryptotrader_lm", t, cyc)}
    assert sampled <= {"BTC-USD", "ETH-USD"}
    assert sampled == {"BTC-USD", "ETH-USD"}  # both crypto tickers do get sampled


def test_finance_llama_rotates_all_tickers(multi_registry):
    sampled = {t for cyc in range(120) for t in se._SHADOW_LLM_ROTATION
               if se._shadow_llm_runs_now("finance_llama", t, cyc)}
    assert sampled == set(se._SHADOW_LLM_ROTATION)


def test_at_most_two_shadows_per_cycle(multi_registry):
    """Co-fire budget: never >2 distinct heavy LLM shadows in one cycle, and
    the two 4s signals (finance_llama/cryptotrader_lm) never co-fire."""
    sigs = ["phi4_mini", "finance_llama", "cryptotrader_lm"]
    for cyc in range(300):
        firing = {s for s in sigs for t in se._SHADOW_LLM_ROTATION
                  if se._shadow_llm_runs_now(s, t, cyc)}
        assert len(firing) <= 2, f"cyc {cyc}: {firing}"
        assert firing != {"finance_llama", "cryptotrader_lm"}, f"cyc {cyc}: both 4s signals"


# ---------------------------------------------------------------------------
# (f) Rich-context dispatch (2026-06-11): LLM shadows must receive the SAME
#     rich context the ministral/qwen3 voters get (price_usd, rsi, ...), with
#     "ticker" in full "BTC-USD" form. Passing the minimal context_data dict
#     made cryptotrader_lm KeyError on 'price_usd' on every run since its
#     2026-05-17 wiring (zero llm_probability_log rows in 24 days), while
#     finance_llama / phi4_mini silently rendered "RSI=?" placeholder prompts.
#     Also covers the off-rotation `_throttled` flag so the log_vote hook
#     stays silent for ticks that never computed.
# ---------------------------------------------------------------------------

import unittest.mock as mock

from conftest import make_candles, make_indicators


def _null_cached(key, ttl, func, *args):
    """Block external accuracy/activation lookups (same as test_consensus)."""
    if key and ("accuracy" in key or "activation_rates" in key):
        return {}
    return None


@pytest.fixture
def dispatch_registry(monkeypatch):
    """cryptotrader_lm enrolled + always phase-aligned; no registry file I/O."""
    monkeypatch.setattr(
        sr, "get_status",
        lambda s, **k: "shadow" if s in se._SHADOW_LLM_SIGNALS else None,
    )
    monkeypatch.setattr(sr, "get_cycle_modulo", lambda s, **k: 3)
    monkeypatch.setattr(
        sr, "should_run_this_cycle", lambda s, c, **k: s == "cryptotrader_lm",
    )
    # cyc=0 -> rotation index 0 -> BTC-USD is the chosen crypto ticker.
    monkeypatch.setattr(sr, "cycle_count_now", lambda: 0)
    monkeypatch.setattr(sr, "is_promoted", lambda s, **k: False)
    # Restrict the enhanced-dispatch loop to cryptotrader_lm only: with a real
    # df present every enhanced signal would compute (news/econ do network I/O).
    entry = dict(se.get_enhanced_signals()["cryptotrader_lm"])
    monkeypatch.setattr(se, "get_enhanced_signals", lambda: {"cryptotrader_lm": entry})


def _capture_dispatch(monkeypatch, captured):
    """Route cryptotrader_lm's compute through a capture fn; others load normally."""
    orig = se.load_signal_func

    def fake_load(entry):
        if entry.get("module_path") == "portfolio.signals.cryptotrader_lm":
            def compute(df, context=None):
                captured.append(context)
                return {
                    "action": "BUY",
                    "confidence": 0.7,
                    "sub_signals": {"cryptotrader_lm": "BUY"},
                    "indicators": {},
                }
            return compute
        return orig(entry)

    monkeypatch.setattr(se, "load_signal_func", fake_load)


@mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
def test_shadow_llm_receives_rich_context(_mock, dispatch_registry, monkeypatch):
    captured = []
    _capture_dispatch(monkeypatch, captured)
    ind = make_indicators(close=69_000.0, rsi=41.0)
    # Enhanced dispatch only runs with a real df of >=26 candles.
    df = make_candles([69_000.0 + i for i in range(30)])
    _action, _conf, extra = se.generate_signal(ind, ticker="BTC-USD", df=df)
    assert len(captured) == 1
    ctx = captured[0]
    # Rich keys the prompt builders need — 'price_usd' is the one whose
    # absence KeyError'd cryptotrader_lm for 24 days.
    assert ctx["price_usd"] == 69_000.0
    assert ctx["rsi"] == 41.0
    assert "macd_hist" in ctx
    assert "bb_position" in ctx
    assert "ema_bullish" in ctx
    # Full ticker form: cryptotrader_lm's crypto-only guard matches "BTC-USD";
    # _build_llm_context's stripped display form ("BTC") must not leak through.
    assert ctx["ticker"] == "BTC-USD"
    # Minimal-context keys survive the merge (config/regime for gating helpers).
    assert "config" in ctx
    assert "regime" in ctx
    # Shadow result recorded for accuracy tracking, run not flagged throttled.
    assert extra["cryptotrader_lm_action"] == "BUY"
    assert extra["cryptotrader_lm_confidence"] > 0
    assert "cryptotrader_lm_throttled" not in extra


@mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
def test_off_rotation_tick_marked_throttled(_mock, dispatch_registry, monkeypatch):
    captured = []
    _capture_dispatch(monkeypatch, captured)
    ind = make_indicators()
    df = make_candles([4_000.0 + i for i in range(30)])
    # XAU is outside cryptotrader_lm's BTC/ETH rotation scope -> never computes;
    # the tick must be flagged throttled so the log_vote hook skips it silently
    # instead of emitting a [log_vote_skipped] abstain_conf_zero line per cycle.
    _action, _conf, extra = se.generate_signal(ind, ticker="XAU-USD", df=df)
    assert captured == []
    assert extra.get("cryptotrader_lm_throttled") is True
    assert "cryptotrader_lm_action" not in extra
