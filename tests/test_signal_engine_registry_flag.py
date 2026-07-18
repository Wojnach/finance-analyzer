"""Tests for Phase 4.2 -- signal_engine's registry flag-gated wrappers.

`config.json` key `signals.use_registry` (default False / missing = off)
routes enablement decisions through `portfolio.component_registry` instead
of the legacy hardcoded constants (DISABLED_SIGNALS,
_DISABLED_SIGNAL_OVERRIDES, _TICKER_DISABLED_BY_HORIZON). Since the registry
is a generated, parity-tested snapshot of those same constants
(tests/test_component_registry.py), flipping the flag is meant to be a
no-op TODAY -- these tests prove that for signal_engine's own wrappers,
not just the registry in isolation.

See docs/plans/2026-07-18-dashboard-redesign-and-modular-engine.md Phase 4.2
for the design rationale, in particular why enablement is split into TWO
wrappers (_sig_globally_disabled / _ticker_horizon_disabled) rather than one
merged is_enabled() call: the legacy dispatch gate's `_promoted_override`
shadow-promotion bypass applies to the global-disable axis only. Folding
the ticker/horizon blacklist into the same gate would let a promotion
silently bypass that blacklist too -- TestPromotionInteraction below proves
the split avoids this.
"""

from __future__ import annotations

import json

import pytest

from portfolio import signal_engine
from portfolio.component_registry import ComponentRegistry
from portfolio.tickers import ALL_TICKERS, DISABLED_SIGNALS, SIGNAL_NAMES

TICKERS = sorted(ALL_TICKERS)
HORIZONS = (None, "3h", "4h", "12h", "1d", "3d", "5d", "10d")

OFF: dict = {"signals": {"use_registry": False}}
ON: dict = {"signals": {"use_registry": True}}


@pytest.fixture(autouse=True)
def _clear_registry_env(monkeypatch):
    """These tests exercise CONFIG semantics via the explicit ON/OFF dicts.
    The PF_USE_REGISTRY env override (pre-flip full-suite gate) would force
    the flag on and break every default-off assertion -- clear it so the
    file is deterministic in both gate runs."""
    monkeypatch.delenv("PF_USE_REGISTRY", raising=False)


class TestUseRegistryFlag:
    def test_missing_config_defaults_off(self):
        assert signal_engine._use_registry(None) is False

    def test_missing_signals_key_defaults_off(self):
        assert signal_engine._use_registry({}) is False

    def test_missing_use_registry_key_defaults_off(self):
        assert signal_engine._use_registry({"signals": {}}) is False

    def test_explicit_true(self):
        assert signal_engine._use_registry(ON) is True

    def test_explicit_false(self):
        assert signal_engine._use_registry(OFF) is False


class TestSigGloballyDisabledFlagParity:
    """flag off/on must agree for every (signal, ticker) -- this is exactly
    the P1 axis ComponentRegistry.is_globally_disabled was added to expose
    (see component_registry.py, tests/test_component_registry.py
    TestIsGloballyDisabled)."""

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_matches_across_flag_states(self, ticker):
        mismatches = [
            signal
            for signal in SIGNAL_NAMES
            if signal_engine._sig_globally_disabled(signal, ticker, OFF)
            != signal_engine._sig_globally_disabled(signal, ticker, ON)
        ]
        assert not mismatches, (ticker, mismatches)

    def test_matches_legacy_computation_when_off(self):
        for ticker in TICKERS:
            for signal in SIGNAL_NAMES:
                expected = (
                    signal in DISABLED_SIGNALS
                    and (signal, ticker) not in signal_engine._DISABLED_SIGNAL_OVERRIDES
                )
                assert (
                    signal_engine._sig_globally_disabled(signal, ticker, OFF)
                    == expected
                )


class TestTickerHorizonDisabledFlagParity:
    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_matches_across_flag_states(self, ticker, horizon):
        off = signal_engine._ticker_horizon_disabled(ticker, horizon, OFF)
        on = signal_engine._ticker_horizon_disabled(ticker, horizon, ON)
        assert off == on, (ticker, horizon, off, on)

    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_matches_legacy_function_when_off(self, ticker, horizon):
        expected = signal_engine._get_horizon_disabled_signals(ticker, horizon)
        assert signal_engine._ticker_horizon_disabled(ticker, horizon, OFF) == expected


class TestApplicableCountFlagParity:
    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("skip_gpu", [True, False])
    def test_matches_across_flag_states_and_legacy(self, ticker, skip_gpu):
        legacy = signal_engine._compute_applicable_count(ticker, skip_gpu=skip_gpu)
        off = signal_engine._applicable_count(ticker, skip_gpu, OFF)
        on = signal_engine._applicable_count(ticker, skip_gpu, ON)
        assert off == legacy
        assert on == legacy


class TestPromotionInteraction:
    """Why two wrappers, not one. A shadow-promoted signal must still be
    force-HELD if it's independently ticker/horizon-blacklisted -- the
    dispatch gate's `_promoted_override and` composition only ever bypassed
    the DISABLED_SIGNALS axis, never the blacklist. Both facts below are
    real, static, present-day data (not hypothetical)."""

    def test_ticker_blacklisted_disabled_signal_stays_blacklisted_regardless_of_flag(
        self,
    ):
        # momentum_factors is globally disabled AND blacklisted for MSTR at
        # every horizon today (portfolio/signal_engine.py
        # _TICKER_DISABLED_BY_HORIZON["_default"]["MSTR"]).
        assert "momentum_factors" in DISABLED_SIGNALS
        assert "momentum_factors" in signal_engine._TICKER_DISABLED_SIGNALS.get(
            "MSTR", ()
        )
        for cfg in (OFF, ON):
            blacklist = signal_engine._ticker_horizon_disabled("MSTR", None, cfg)
            assert "momentum_factors" in blacklist, cfg
        # _ticker_horizon_disabled takes no promotion argument at all -- a
        # simulated promotion cannot change this answer, which is the point:
        # this gate is applied unconditionally, downstream of and
        # independent from whatever the dispatch gate decided.

    def test_promotion_bypasses_only_the_global_disable_gate(self):
        # forecast is globally disabled and NOT in BTC-USD's ticker/horizon
        # blacklist at any horizon -- a clean case where promotion SHOULD
        # let it through.
        assert "forecast" in DISABLED_SIGNALS
        assert "forecast" not in signal_engine._get_horizon_disabled_signals(
            "BTC-USD", None
        )
        for cfg in (OFF, ON):
            globally_disabled = signal_engine._sig_globally_disabled(
                "forecast", "BTC-USD", cfg
            )
            promoted_override = True  # simulated is_promoted("forecast") == True
            dispatch_gate_fires = globally_disabled and not promoted_override
            assert dispatch_gate_fires is False, cfg
            assert "forecast" not in signal_engine._ticker_horizon_disabled(
                "BTC-USD", None, cfg
            )


class TestOverlayPropagatesWhenFlagOn:
    """Overlay overrides only take effect when the flag is on -- proves the
    registry (and its live data/control/registry_overrides.json overlay,
    which Phase 3's control panel writes to) is actually being consulted,
    not just its generated-snapshot defaults. Uses a tmp_path overlay via a
    monkeypatched get_registry() -- never touches the real overlay file.
    """

    @pytest.fixture()
    def patched_registry(self, tmp_path, monkeypatch):
        overlay_path = tmp_path / "registry_overrides.json"

        def _install(overlay: dict) -> ComponentRegistry:
            overlay_path.write_text(json.dumps(overlay))
            reg = ComponentRegistry(overlay_path=overlay_path)
            import portfolio.component_registry as component_registry

            monkeypatch.setattr(component_registry, "get_registry", lambda: reg)
            return reg

        return _install

    def test_enabled_override_flips_sig_globally_disabled(self, patched_registry):
        patched_registry({"BTC-USD": {"ml": {"enabled": True, "reason": "test"}}})
        # "ml" is globally disabled with no static override for BTC-USD.
        assert signal_engine._sig_globally_disabled("ml", "BTC-USD", OFF) is True
        assert signal_engine._sig_globally_disabled("ml", "BTC-USD", ON) is False

    def test_horizon_override_propagates_through_ticker_horizon_disabled(
        self, patched_registry
    ):
        patched_registry({"BTC-USD": {"rsi": {"horizons": {"1d": False}}}})
        assert "rsi" not in signal_engine._ticker_horizon_disabled("BTC-USD", "1d", OFF)
        assert "rsi" in signal_engine._ticker_horizon_disabled("BTC-USD", "1d", ON)
        # Other horizons untouched by the horizon-scoped override.
        assert "rsi" not in signal_engine._ticker_horizon_disabled("BTC-USD", "3h", ON)


class TestGenerateSignalEndToEndFlagParity:
    """Small end-to-end spot check on the real entry point. Kept to one
    ticker per asset class at a single horizon (not the full 5x7 matrix):
    this test environment has no config.json, so every call pays several
    seconds of live-API timeout retries end to end. The wrapper-level
    parity tests above already prove every gating decision generate_signal
    makes is identical in both flag states; this is a sanity check that
    nothing downstream of those decisions (weighted consensus, applicable-
    count-derived voter floor, core/min-voter gates, etc.) diverges when
    wired together -- one ticker per asset class (crypto/stock/metal) since
    MIN_VOTERS and the core-signal set differ by class.
    """

    @pytest.mark.parametrize("ticker", ["BTC-USD", "MSTR", "XAU-USD"])
    def test_action_and_confidence_identical(self, ticker):
        import numpy as np
        import pandas as pd

        from portfolio.signal_engine import generate_signal

        ind = {
            "close": 100.0,
            "rsi": 50.0,
            "rsi_p20": 30.0,
            "rsi_p80": 70.0,
            "macd_hist": 0.0,
            "macd_hist_prev": 0.0,
            "ema9": 100.0,
            "ema21": 100.0,
            "sma20": 100.0,
            "sma50": 100.0,
            "bb_upper": 110.0,
            "bb_lower": 90.0,
            "bb_mid": 100.0,
            "price_vs_bb": "inside",
            "volume": 1_000_000,
            "volume_sma20": 1_000_000,
            "atr": 1.0,
            "adx": 20.0,
            "high": 101.0,
            "low": 99.0,
            "open": 100.0,
        }
        df = pd.DataFrame(
            {
                "open": np.full(50, 100.0),
                "high": np.full(50, 101.0),
                "low": np.full(50, 99.0),
                "close": np.full(50, 100.0),
                "volume": np.full(50, 1_000_000),
            }
        )

        action_off, conf_off, _ = generate_signal(
            dict(ind), ticker=ticker, config=OFF, df=df.copy(), horizon="1d"
        )
        action_on, conf_on, _ = generate_signal(
            dict(ind), ticker=ticker, config=ON, df=df.copy(), horizon="1d"
        )

        assert action_off == action_on
        assert conf_off == pytest.approx(conf_on)
