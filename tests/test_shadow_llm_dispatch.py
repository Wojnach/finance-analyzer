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
