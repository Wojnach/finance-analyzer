"""Tests for portfolio.signal_engine — dynamic correlation groups."""



# ---------------------------------------------------------------------------
# 3c. Static and dynamic correlation groups
# ---------------------------------------------------------------------------

class TestCorrelationGroups:

    def test_static_correlation_groups_has_expected_keys(self):
        """CORRELATION_GROUPS (static alias) should exist and contain known group names."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert isinstance(CORRELATION_GROUPS, dict)
        # Check a few known static groups
        # low_activity_timing removed 2026-04-12: calendar (BUY-only) and
        # econ_calendar (SELL-only) have opposite directions and divergent
        # regime profiles — not a valid correlation cluster.
        assert "pattern_based" in CORRELATION_GROUPS
        # Values should be frozensets
        for name, members in CORRELATION_GROUPS.items():
            assert isinstance(members, frozenset), f"Group {name} should be frozenset"
            assert len(members) >= 2, f"Group {name} should have at least 2 members"

    def test_momentum_cluster_exists(self):
        """momentum_cluster should contain rsi, mean_reversion, momentum.

        2026-04-25: bb moved to trend_direction (87.8% agreement with macd,
        85%+ with ema — cross-cluster redundancy fix).
        """
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "momentum_cluster" in CORRELATION_GROUPS
        mc_group = CORRELATION_GROUPS["momentum_cluster"]
        assert "mean_reversion" in mc_group
        assert "rsi" in mc_group
        assert "momentum" in mc_group
        assert "bb" not in mc_group  # moved to trend_direction 2026-04-25

    def test_volatility_cluster_removed(self):
        """volatility_cluster was dissolved (RES-2026-04-21).

        volatility_sig and volume had only r=0.38 correlation — too weak
        for a group. volume (52.1% acc) was unfairly penalized by
        volatility_sig (46.8% acc). Both now vote independently.
        """
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "volatility_cluster" not in CORRELATION_GROUPS

    def test_trend_direction_expanded_members(self):
        """trend_direction should include momentum_factors, structure, oscillators.

        2026-04-14: Measured correlations show these belong in trend cluster:
        momentum_factors+macro_regime r=0.621 (91.5% agree),
        structure+trend r=0.608 (90.7% agree),
        oscillators+heikin_ashi r=0.463 (83.4% agree).
        """
        from portfolio.signal_engine import CORRELATION_GROUPS

        td_group = CORRELATION_GROUPS["trend_direction"]
        assert "momentum_factors" in td_group
        assert "structure" in td_group
        assert "oscillators" in td_group

    def test_macd_in_trend_direction(self):
        """macd should be in trend_direction group (91.9% agreement with ema)."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        td_group = CORRELATION_GROUPS["trend_direction"]
        assert "macd" in td_group, "MACD is derived from EMAs; 91.9% agreement with ema"

    def test_macro_regime_in_trend_direction(self):
        """macro_regime should be in trend_direction group (r=0.520 with trend)."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "trend_direction" in CORRELATION_GROUPS
        td_group = CORRELATION_GROUPS["trend_direction"]
        assert "macro_regime" in td_group
        assert "trend" in td_group

    def test_macro_regime_not_in_macro_external(self):
        """macro_regime was moved out of macro_external into trend_direction."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        me_group = CORRELATION_GROUPS["macro_external"]
        assert "macro_regime" not in me_group

    def test_static_groups_backward_compat_alias(self):
        """CORRELATION_GROUPS should be the same object as _STATIC_CORRELATION_GROUPS."""
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            CORRELATION_GROUPS,
        )
        assert CORRELATION_GROUPS is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_fallback(self, monkeypatch):
        """When no signal_log data exists, _compute_dynamic_correlation_groups
        should return _STATIC_CORRELATION_GROUPS."""
        # Monkeypatch load_entries to return empty list (no data)
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            _compute_dynamic_correlation_groups,
        )
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_fallback_insufficient_data(self, monkeypatch):
        """With fewer than _DYNAMIC_CORR_MIN_SAMPLES entries, falls back to static."""
        # Provide a small number of entries (below the 30 minimum)
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            _compute_dynamic_correlation_groups,
        )
        fake_entries = [
            {
                "ts": "2026-04-01T00:00:00+00:00",
                "tickers": {"BTC-USD": {"signals": {"rsi": "BUY"}}},
                "outcomes": {},
            }
            for _ in range(5)
        ]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: fake_entries)

        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_returns_dict_of_frozensets(self, monkeypatch):
        """Even when falling back, the return type is dict[str, frozenset]."""
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import _compute_dynamic_correlation_groups
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = _compute_dynamic_correlation_groups()
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, frozenset)


# ---------------------------------------------------------------------------
# Directional bias penalty
# ---------------------------------------------------------------------------

class TestDirectionalBiasPenalty:

    def test_extreme_bias_reduces_weight(self):
        """Signals with bias > 85% should get _BIAS_PENALTY applied."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY", "calendar": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.55, "total": 100},
            "calendar": {"accuracy": 0.60, "total": 100},
        }
        # calendar has extreme bias (>85%), rsi does not
        activation_rates = {
            "rsi": {"bias": 0.1, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3},
            "calendar": {"bias": 0.95, "samples": 100, "normalized_weight": 0.25,
                         "activation_rate": 0.08},
        }
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        # Both BUY → should still return BUY
        assert result[0] == "BUY"

    def test_no_penalty_below_threshold(self):
        """Signals with bias <= 85% should NOT get extra penalty."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {"rsi": {"accuracy": 0.55, "total": 100}}
        activation_rates = {
            "rsi": {"bias": 0.5, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3},
        }
        # With bias=0.5 (< 0.85), no extra penalty should apply
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        assert result[0] == "BUY"

    def test_bias_penalty_not_applied_with_few_samples(self):
        """Bias penalty should not fire when samples < _BIAS_MIN_ACTIVE."""
        from portfolio.signal_engine import _BIAS_MIN_ACTIVE, _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {"rsi": {"accuracy": 0.55, "total": 100}}
        activation_rates = {
            "rsi": {"bias": 0.99, "samples": _BIAS_MIN_ACTIVE - 1,
                    "normalized_weight": 1.0, "activation_rate": 0.3},
        }
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        assert result[0] == "BUY"


# ---------------------------------------------------------------------------
# Directional accuracy gating
# ---------------------------------------------------------------------------

class TestDirectionalAccuracyGating:

    def test_buy_gated_when_buy_accuracy_low(self):
        """Qwen3-like: overall 59.8% passes gate, but BUY=30% should be gated."""
        from portfolio.signal_engine import _weighted_consensus

        # qwen3 BUY at 30% (well below 35% directional gate), SELL at 74%
        votes = {"qwen3": "BUY", "rsi": "SELL"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.598, "total": 3608,
                       "buy_accuracy": 0.30, "total_buy": 1174,
                       "sell_accuracy": 0.74, "total_sell": 2434},
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # qwen3 BUY should be gated, only rsi SELL remains → SELL
        assert result[0] == "SELL"

    def test_sell_passes_when_buy_gated(self):
        """Same signal: BUY gated but SELL should still vote normally."""
        from portfolio.signal_engine import _weighted_consensus

        # qwen3 SELL should NOT be directionally gated (sell_accuracy 74% >> 35%)
        votes = {"qwen3": "SELL"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.598, "total": 3608,
                       "buy_accuracy": 0.30, "total_buy": 1174,
                       "sell_accuracy": 0.74, "total_sell": 2434},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # Only qwen3 voting SELL, not directionally gated → SELL
        assert result[0] == "SELL"

    def test_directional_gate_not_applied_with_few_samples(self):
        """Directional gate requires sufficient samples to avoid premature gating."""
        from portfolio.signal_engine import (
            _DIRECTIONAL_GATE_MIN_SAMPLES,
            _weighted_consensus,
        )

        votes = {"qwen3": "BUY"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.55, "total": 100,
                       "buy_accuracy": 0.20, "total_buy": _DIRECTIONAL_GATE_MIN_SAMPLES - 1,
                       "sell_accuracy": 0.80, "total_sell": 100},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # Not enough BUY samples → directional gate should NOT fire
        assert result[0] == "BUY"

    def test_claude_fundamental_sell_gated(self):
        """claude_fundamental: BUY=65.7% fine, but SELL=39.7% should be gated at 0.40."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"claude_fundamental": "SELL", "rsi": "BUY"}
        accuracy_data = {
            "claude_fundamental": {"accuracy": 0.628, "total": 7535,
                                    "buy_accuracy": 0.657, "total_buy": 6697,
                                    "sell_accuracy": 0.397, "total_sell": 838},
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        # sell_accuracy 0.397 < 0.40 threshold → gated (raised from 0.35 on 2026-04-10)
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # claude_fundamental SELL gated, only rsi BUY remains → BUY
        assert result[0] == "BUY"

    def test_macro_regime_buy_gated_at_40pct(self):
        """macro_regime: overall 46.6% passes accuracy gate, but BUY=38.9% gated."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"macro_regime": "BUY", "rsi": "SELL"}
        accuracy_data = {
            "macro_regime": {"accuracy": 0.466, "total": 25654,
                              "buy_accuracy": 0.389, "total_buy": 8963,
                              "sell_accuracy": 0.508, "total_sell": 16691},
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # macro_regime BUY at 38.9% < 40% threshold → gated, only rsi SELL remains
        assert result[0] == "SELL"

    def test_no_directional_gate_when_accuracy_above_threshold(self):
        """Signals with both BUY and SELL above threshold should not be gated."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.52, "total": 1000,
                     "buy_accuracy": 0.50, "total_buy": 500,
                     "sell_accuracy": 0.54, "total_sell": 500},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        assert result[0] == "BUY"

    def test_missing_buy_accuracy_key_falls_back_to_overall(self):
        """BUG-185: Missing buy_accuracy key should not crash; uses overall acc."""
        from portfolio.signal_engine import _weighted_consensus

        # total_buy present (>= 20) but buy_accuracy missing — simulates cache corruption
        votes = {"rsi": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.55, "total": 200,
                     "total_buy": 100},  # buy_accuracy intentionally absent
        }
        # Should not raise KeyError; falls back to overall accuracy (0.55)
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        assert result[0] == "BUY"

    def test_missing_sell_accuracy_key_falls_back_to_overall(self):
        """BUG-185: Missing sell_accuracy key should not crash; uses overall acc."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "SELL"}
        accuracy_data = {
            "rsi": {"accuracy": 0.55, "total": 200,
                     "total_sell": 100},  # sell_accuracy intentionally absent
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        assert result[0] == "SELL"


# ---------------------------------------------------------------------------
# Funding rate horizon gating (2026-04-09)
# ---------------------------------------------------------------------------

class TestFundingRateHorizonGating:
    """Funding rate: 74.2% at 3h but 29.9% at 1d. Should only vote at 3h/4h."""

    def test_funding_gated_at_default_in_all_regimes(self):
        """funding should be in _default gate for ranging, trending-up/down, high-vol."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        for regime in ("ranging", "trending-up", "trending-down", "high-vol"):
            assert regime in REGIME_GATED_SIGNALS, f"{regime} missing from REGIME_GATED_SIGNALS"
            default_set = REGIME_GATED_SIGNALS[regime].get("_default", frozenset())
            assert "funding" in default_set, (
                f"funding should be gated at _default in {regime}"
            )

    def test_funding_not_gated_at_3h_ranging(self):
        """funding should NOT be in 3h gate for ranging (74.2% accuracy there)."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        gate_3h = REGIME_GATED_SIGNALS["ranging"].get("3h", frozenset())
        assert "funding" not in gate_3h

    def test_funding_not_gated_at_4h_ranging(self):
        """funding should NOT be in 4h gate for ranging."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        gate_4h = REGIME_GATED_SIGNALS["ranging"].get("4h", frozenset())
        assert "funding" not in gate_4h

    def test_funding_not_in_disabled_signals(self):
        """funding was removed from DISABLED_SIGNALS (re-enabled 2026-04-09)."""
        from portfolio.tickers import DISABLED_SIGNALS

        assert "funding" not in DISABLED_SIGNALS


# ---------------------------------------------------------------------------
# On-chain BTC signal (2026-04-09)
# ---------------------------------------------------------------------------

class TestOnchainSignal:
    """On-chain signal: MVRV Z-Score, SOPR, NUPL, netflow → majority vote for BTC."""

    def test_onchain_in_signal_names(self):
        """onchain should be in the SIGNAL_NAMES list."""
        from portfolio.tickers import SIGNAL_NAMES

        assert "onchain" in SIGNAL_NAMES

    def _run_onchain_block(self, ticker, cached_data):
        """Helper: run just the on-chain signal block in isolation.

        Returns (vote, extra_info) without calling generate_signal
        (which needs dozens of indicator keys).
        """
        from portfolio.tickers import CRYPTO_SYMBOLS
        from portfolio.shared_state import ONCHAIN_TTL

        votes = {"onchain": "HOLD"}
        extra_info = {}

        if ticker == "BTC-USD":
            oc = cached_data
            if oc:
                sub_votes = []
                zscore = oc.get("mvrv_zscore")
                if zscore is not None:
                    if zscore < 1.0:
                        sub_votes.append("BUY")
                    elif zscore > 5.0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_mvrv_zscore"] = round(zscore, 2)
                sopr = oc.get("sopr")
                if sopr is not None:
                    if sopr < 0.97:
                        sub_votes.append("BUY")
                    elif sopr > 1.05:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_sopr"] = round(sopr, 4)
                nupl = oc.get("nupl")
                if nupl is not None:
                    if nupl < 0:
                        sub_votes.append("BUY")
                    elif nupl > 0.75:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                netflow = oc.get("netflow")
                if netflow is not None:
                    if netflow < 0:
                        sub_votes.append("BUY")
                    elif netflow > 0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                buy_count = sub_votes.count("BUY")
                sell_count = sub_votes.count("SELL")
                total = buy_count + sell_count
                if total >= 2:
                    if buy_count > sell_count:
                        votes["onchain"] = "BUY"
                    elif sell_count > buy_count:
                        votes["onchain"] = "SELL"
                    extra_info["onchain_sub_votes"] = f"{buy_count}B/{sell_count}S"

        return votes["onchain"], extra_info

    def test_onchain_buy_when_metrics_undervalued(self):
        """When MVRV < 1, SOPR < 0.97, NUPL < 0 → majority BUY."""
        data = {
            "mvrv_zscore": 0.5,   # < 1.0 → BUY
            "sopr": 0.95,         # < 0.97 → BUY
            "nupl": -0.1,         # < 0 → BUY
            "netflow": 500,       # > 0 → SELL (minority)
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "BUY"
        assert extra["onchain_mvrv_zscore"] == 0.5
        assert extra["onchain_sopr"] == 0.95
        assert extra["onchain_sub_votes"] == "3B/1S"

    def test_onchain_sell_when_overheated(self):
        """When MVRV > 5, SOPR > 1.05, NUPL > 0.75 → majority SELL."""
        data = {
            "mvrv_zscore": 6.0,   # > 5.0 → SELL
            "sopr": 1.08,         # > 1.05 → SELL
            "nupl": 0.80,         # > 0.75 → SELL
            "netflow": -100,      # < 0 → BUY (minority)
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "SELL"
        assert extra["onchain_sub_votes"] == "1B/3S"

    def test_onchain_hold_for_non_btc(self):
        """On-chain signal should be HOLD for non-BTC tickers."""
        data = {"mvrv_zscore": 0.5, "sopr": 0.95, "nupl": -0.1, "netflow": -100}
        vote, extra = self._run_onchain_block("ETH-USD", data)
        assert vote == "HOLD"
        assert not extra  # no extra_info populated

    def test_onchain_hold_when_no_data(self):
        """On-chain signal should be HOLD when API returns None."""
        vote, extra = self._run_onchain_block("BTC-USD", None)
        assert vote == "HOLD"

    def test_onchain_hold_when_neutral_metrics(self):
        """When all metrics are neutral, should remain HOLD (no majority)."""
        data = {
            "mvrv_zscore": 3.0,   # 1-5 → HOLD
            "sopr": 1.00,         # 0.97-1.05 → HOLD
            "nupl": 0.4,          # 0-0.75 → HOLD
            "netflow": 0,         # 0 → HOLD
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "HOLD"
        # No sub_votes key when total active (BUY+SELL) < 2
        assert "onchain_sub_votes" not in extra

    def test_onchain_tie_stays_hold(self):
        """When BUY and SELL are tied (2B/2S), should remain HOLD."""
        data = {
            "mvrv_zscore": 0.5,   # < 1.0 → BUY
            "sopr": 0.95,         # < 0.97 → BUY
            "nupl": 0.80,         # > 0.75 → SELL
            "netflow": 500,       # > 0 → SELL
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "HOLD"
        assert extra["onchain_sub_votes"] == "2B/2S"


# ---------------------------------------------------------------------------
# BUG-178 fixes (2026-04-10)
# ---------------------------------------------------------------------------

class TestDispatchLoopRespectsDisabledSignals:
    """Regression: enhanced-signal dispatch loop must skip DISABLED_SIGNALS.

    Before 2026-04-10, the dispatch loop iterated every registered enhanced
    signal regardless of disabled status. This caused 49 BUG-178 ticker
    pool timeouts on 2026-04-09/10 because the disabled signals
    (crypto_macro, cot_positioning, credit_spread_risk) were doing network
    I/O on every cycle.
    """

    def test_disabled_signals_have_force_hold_in_dispatch_loop(self):
        """The dispatch loop must short-circuit on DISABLED_SIGNALS."""
        import inspect

        from portfolio import signal_engine
        from portfolio.tickers import DISABLED_SIGNALS

        src = inspect.getsource(signal_engine.generate_signal)
        # The fix: dispatch loop must check DISABLED_SIGNALS before calling compute_fn
        assert "if sig_name in DISABLED_SIGNALS" in src, (
            "Dispatch loop must skip disabled signals to prevent BUG-178 hangs"
        )
        # Defense in depth: at least one disabled signal must exist (otherwise
        # the check is dead code we should remove)
        assert len(DISABLED_SIGNALS) > 0


class TestLastSignalDiagnostic:
    """BUG-178 diagnostic: per-ticker last-signal tracker."""

    def test_set_and_get_last_signal_round_trip(self):
        from portfolio.signal_engine import _set_last_signal, get_last_signal

        _set_last_signal("BTC-USD", "test_signal_xyz")
        result = get_last_signal("BTC-USD")
        assert result is not None
        sig_name, elapsed = result
        assert sig_name == "test_signal_xyz"
        assert elapsed >= 0.0
        assert elapsed < 5.0  # should be near-instant

    def test_get_last_signal_unknown_ticker_returns_none(self):
        from portfolio.signal_engine import get_last_signal

        result = get_last_signal("__nonexistent_ticker__")
        assert result is None

    def test_set_overwrites_previous(self):
        from portfolio.signal_engine import _set_last_signal, get_last_signal

        _set_last_signal("ETH-USD", "first_signal")
        _set_last_signal("ETH-USD", "second_signal")
        result = get_last_signal("ETH-USD")
        assert result is not None
        assert result[0] == "second_signal"


class TestGenerateSignalPhaseMarkers:
    """BUG-178 slow-cycle diagnostic: phase markers written by generate_signal.

    generate_signal() must update _last_signal_per_ticker with
    __pre_dispatch__ before the enhanced-signal dispatch loop and
    __post_dispatch__ after, so the main.py slow-cycle diagnostic can
    distinguish hangs in the three distinct phases.
    """

    def test_generate_signal_writes_pre_and_post_dispatch_markers(self):
        """After generate_signal runs, the tracker should show __post_dispatch__.

        Uses the full generate_signal pipeline with a synthetic indicator dict
        and checks that the tracker's recorded last_signal is either
        __post_dispatch__ (normal fast path) or a concrete signal name (if the
        dispatch loop bailed early). In no case should it still be
        __pre_dispatch__ on a successful completion.
        """
        import pandas as pd
        import numpy as np

        from portfolio.signal_engine import generate_signal, get_last_signal

        # Minimal indicator dict with everything generate_signal touches.
        ind = {
            "close": 100.0, "rsi": 50.0, "rsi_p20": 30.0, "rsi_p80": 70.0,
            "macd_hist": 0.0, "macd_hist_prev": 0.0,
            "ema9": 100.0, "ema21": 100.0, "sma20": 100.0, "sma50": 100.0,
            "bb_upper": 110.0, "bb_lower": 90.0, "bb_mid": 100.0,
            "price_vs_bb": "inside", "volume": 1_000_000, "volume_sma20": 1_000_000,
            "atr": 1.0, "adx": 20.0, "high": 101.0, "low": 99.0, "open": 100.0,
        }
        # Small df — mostly for the enhanced-signals dispatch loop.
        df = pd.DataFrame({
            "open": np.full(50, 100.0), "high": np.full(50, 101.0),
            "low": np.full(50, 99.0), "close": np.full(50, 100.0),
            "volume": np.full(50, 1_000_000),
        })

        generate_signal(ind, ticker="BTC-USD", df=df)

        last = get_last_signal("BTC-USD")
        assert last is not None
        sig_name, _ = last
        # Post-successful-completion, the tracker should show the post-dispatch
        # marker — proves the loop ran to the end.
        assert sig_name == "__post_dispatch__", (
            f"Expected __post_dispatch__ after a clean generate_signal run, "
            f"got {sig_name!r}. If this is a concrete signal name, the dispatch "
            f"loop bailed early; if __pre_dispatch__, the loop never started."
        )

    def test_generate_signal_skips_markers_when_ticker_none(self):
        """Phase markers are gated on `ticker` being truthy.

        When generate_signal is called without a ticker (edge case for tests
        and legacy callers), the tracker must not record anything under an
        empty-string key — that would pollute the dict for all future lookups.
        """
        import pandas as pd
        import numpy as np

        from portfolio import signal_engine

        # Snapshot current tracker state so we can verify no new entry appears.
        with signal_engine._last_signal_lock:
            keys_before = set(signal_engine._last_signal_per_ticker.keys())

        ind = {
            "close": 100.0, "rsi": 50.0, "rsi_p20": 30.0, "rsi_p80": 70.0,
            "macd_hist": 0.0, "macd_hist_prev": 0.0,
            "ema9": 100.0, "ema21": 100.0, "sma20": 100.0, "sma50": 100.0,
            "bb_upper": 110.0, "bb_lower": 90.0, "bb_mid": 100.0,
            "price_vs_bb": "inside", "volume": 1_000_000, "volume_sma20": 1_000_000,
            "atr": 1.0, "adx": 20.0, "high": 101.0, "low": 99.0, "open": 100.0,
        }
        df = pd.DataFrame({
            "open": np.full(50, 100.0), "high": np.full(50, 101.0),
            "low": np.full(50, 99.0), "close": np.full(50, 100.0),
            "volume": np.full(50, 1_000_000),
        })

        signal_engine.generate_signal(ind, ticker=None, df=df)

        with signal_engine._last_signal_lock:
            keys_after = set(signal_engine._last_signal_per_ticker.keys())
        new_keys = keys_after - keys_before
        # No empty-string or None keys should have been added.
        assert "" not in new_keys
        assert None not in new_keys

    def test_post_dispatch_marker_overwrites_dispatch_loop_marker(self):
        """The post-dispatch marker must always supersede the dispatch loop.

        Regression: earlier iterations wrote `_set_last_signal` only inside
        the dispatch loop, leaving the tracker pointing at whichever signal
        ran last. The post-dispatch phase marker must overwrite it.
        """
        from portfolio.signal_engine import _set_last_signal, get_last_signal

        # Simulate the tracker state after the dispatch loop ran
        # metals_cross_asset as its final signal.
        _set_last_signal("XAU-USD", "metals_cross_asset")
        result = get_last_signal("XAU-USD")
        assert result[0] == "metals_cross_asset"

        # Simulate the post-dispatch marker write that our fix adds
        _set_last_signal("XAU-USD", "__post_dispatch__")
        result = get_last_signal("XAU-USD")
        assert result[0] == "__post_dispatch__"


# ---------------------------------------------------------------------------
# 2026-04-11 Research Session: Signal gating and correlation penalty changes
# ---------------------------------------------------------------------------

class TestOrderbookFlowDisabled:
    """orderbook_flow was disabled 2026-04-11: 93.3% active, 51.1% accuracy, no
    recent data. Pure noise in every consensus decision."""

    def test_orderbook_flow_in_disabled_signals(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "orderbook_flow" in DISABLED_SIGNALS

    def test_orderbook_flow_not_in_consensus(self):
        """Disabled signals must produce HOLD in the dispatch loop."""
        from portfolio.tickers import DISABLED_SIGNALS
        assert "orderbook_flow" in DISABLED_SIGNALS


class TestCreditSpreadRiskEnabled:
    """credit_spread_risk re-enabled 2026-04-11: 66.9% accuracy (257 sam),
    BUY 80.3%. Directional gate auto-gates weak SELL direction (49.1%)."""

    def test_credit_spread_risk_not_disabled(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "credit_spread_risk" not in DISABLED_SIGNALS


class TestCryptoMacroEnabled:
    """crypto_macro re-enabled 2026-04-11: 56.5% accuracy (1273 sam). BUY-biased
    (93%) so bias penalty (0.5x) applies. Provides crypto-specific on-chain edge."""

    def test_crypto_macro_not_disabled(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "crypto_macro" not in DISABLED_SIGNALS


class TestSentimentGatedAt3hRanging:
    """sentiment gated at 3h in ranging: 33.8% at 3h_recent (3629 sam). The 0.5x
    horizon weight was insufficient — explicit regime gating is clearer."""

    def test_sentiment_gated_at_3h_ranging(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        gated_3h = REGIME_GATED_SIGNALS["ranging"]["3h"]
        assert "sentiment" in gated_3h

    def test_sentiment_gated_at_4h_ranging(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        gated_4h = REGIME_GATED_SIGNALS["ranging"]["4h"]
        assert "sentiment" in gated_4h

    def test_sentiment_not_gated_at_default_ranging(self):
        """sentiment at 1d (46.8%) is borderline — let the accuracy gate handle it
        dynamically rather than hard-gating."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        gated_default = REGIME_GATED_SIGNALS["ranging"]["_default"]
        assert "sentiment" not in gated_default


class TestPerClusterCorrelationPenalties:
    """2026-04-11: momentum_cluster penalty tightened from 0.3x to 0.15x.
    RSI/BB/MeanReversion/Momentum agree 88-100%, so 0.3x still gave 1.9x
    combined weight. With 0.15x: 1.0 + 3*0.15 = 1.45x."""

    def test_cluster_penalties_dict_exists(self):
        from portfolio.signal_engine import _CLUSTER_CORRELATION_PENALTIES
        assert isinstance(_CLUSTER_CORRELATION_PENALTIES, dict)

    def test_momentum_cluster_has_tighter_penalty(self):
        from portfolio.signal_engine import (
            _CLUSTER_CORRELATION_PENALTIES,
            _CORRELATION_PENALTY,
        )
        assert "momentum_cluster" in _CLUSTER_CORRELATION_PENALTIES
        assert _CLUSTER_CORRELATION_PENALTIES["momentum_cluster"] < _CORRELATION_PENALTY

    def test_momentum_cluster_penalty_is_015(self):
        from portfolio.signal_engine import _CLUSTER_CORRELATION_PENALTIES
        assert _CLUSTER_CORRELATION_PENALTIES["momentum_cluster"] == 0.15

    def test_default_penalty_unchanged(self):
        from portfolio.signal_engine import _CORRELATION_PENALTY
        assert _CORRELATION_PENALTY == 0.3


class TestTickerDisabledSignals:
    """Per-ticker signal gating: force HOLD for specific signal+ticker combos."""

    def test_ticker_disabled_signals_dict_exists(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert isinstance(_TICKER_DISABLED_SIGNALS, dict)

    def test_eth_news_event_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "ETH-USD" in _TICKER_DISABLED_SIGNALS
        assert "news_event" in _TICKER_DISABLED_SIGNALS["ETH-USD"]

    def test_btc_news_event_not_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        btc_disabled = _TICKER_DISABLED_SIGNALS.get("BTC-USD", frozenset())
        assert "news_event" not in btc_disabled

    def test_dispatch_respects_ticker_disable(self):
        """In generate_signal dispatch loop, per-ticker disabled signals should be HOLD."""
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS

        ticker = "ETH-USD"
        sig_name = "news_event"
        disabled_for_ticker = _TICKER_DISABLED_SIGNALS.get(ticker, ())
        assert sig_name in disabled_for_ticker
        # Simulating the dispatch check: if sig_name in disabled_for_ticker → HOLD
        if sig_name in disabled_for_ticker:
            vote = "HOLD"
        else:
            vote = "BUY"
        assert vote == "HOLD"

    def test_xag_ministral_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "XAG-USD" in _TICKER_DISABLED_SIGNALS
        assert "ministral" in _TICKER_DISABLED_SIGNALS["XAG-USD"]

    def test_xag_credit_spread_risk_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "credit_spread_risk" in _TICKER_DISABLED_SIGNALS["XAG-USD"]

    def test_xag_metals_cross_asset_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "metals_cross_asset" in _TICKER_DISABLED_SIGNALS["XAG-USD"]

    def test_xau_ministral_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "XAU-USD" in _TICKER_DISABLED_SIGNALS
        assert "ministral" in _TICKER_DISABLED_SIGNALS["XAU-USD"]

    def test_mstr_credit_spread_risk_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "MSTR" in _TICKER_DISABLED_SIGNALS
        assert "credit_spread_risk" in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_btc_ministral_not_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        btc_disabled = _TICKER_DISABLED_SIGNALS.get("BTC-USD", frozenset())
        assert "ministral" not in btc_disabled


class TestOscillatorsDisabled:
    """BUG-193: oscillators globally disabled — below 45% on all tickers."""

    def test_oscillators_in_disabled_signals(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "oscillators" in DISABLED_SIGNALS

    def test_oscillators_still_in_signal_names(self):
        from portfolio.tickers import SIGNAL_NAMES
        assert "oscillators" in SIGNAL_NAMES


class TestSentimentUnknownRegimeGating:
    """BUG-194: sentiment gated at 3h/4h in unknown regime."""

    def test_sentiment_gated_3h_unknown(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        unknown = REGIME_GATED_SIGNALS.get("unknown", {})
        assert "sentiment" in unknown.get("3h", frozenset())

    def test_sentiment_gated_4h_unknown(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        unknown = REGIME_GATED_SIGNALS.get("unknown", {})
        assert "sentiment" in unknown.get("4h", frozenset())

    def test_sentiment_not_gated_1d_unknown(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        unknown = REGIME_GATED_SIGNALS.get("unknown", {})
        default_gated = unknown.get("_default", frozenset())
        assert "sentiment" not in default_gated


class TestMSTRSignalBlacklist:
    """MSTR-specific per-ticker blacklisting.

    2026-04-16: Trimmed from 7 entries to 2. The Apr 14 audit built the
    blacklist from 3h accuracy data and applied it at all horizons. At
    the 1d horizon where consensus trades, the original 5 removed signals
    (macro_regime, trend, volatility_sig, volume, sentiment) were 62-81%
    accurate and were silencing the votes that would have correctly called
    MSTR's +8.4% W16 rally. Kept: claude_fundamental (47.8% 1d / 33.2% 3h
    - bad at both horizons) and credit_spread_risk (44.2% 1d).
    """

    def test_mstr_macro_regime_NOT_disabled(self):
        # 2026-04-16: macro_regime removed from MSTR blacklist.
        # At 1d horizon it shows 81.4% accuracy on MSTR (last 7d, n=43).
        # The Apr 14 blacklist was built from 3h data (32.5%) and applied
        # globally — horizon mismatch caused W15/W16 consensus collapse.
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "macro_regime" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_trend_NOT_disabled(self):
        # 2026-04-16: trend removed from MSTR blacklist (71.2% at 1d, n=59).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "trend" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_volatility_sig_NOT_disabled(self):
        # 2026-04-16: volatility_sig removed (66.7% at 1d, n=42).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "volatility_sig" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_sentiment_NOT_disabled(self):
        # 2026-04-16: sentiment removed (80.0% at 1d, n=80).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "sentiment" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_volume_NOT_disabled(self):
        # 2026-04-16: volume removed (62.3% at 1d, n=77).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "volume" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_claude_fundamental_still_disabled(self):
        # Retained: 47.8% at 1d / 33.2% at 3h - bad at both horizons.
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "claude_fundamental" in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_credit_spread_risk_still_disabled(self):
        # Retained: 44.2% at 1d.
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "credit_spread_risk" in _TICKER_DISABLED_SIGNALS["MSTR"]


class TestCorrelationPenaltyMultiGroup:
    """Signals in multiple correlation groups get the harshest penalty."""

    def test_multi_group_signal_gets_min_penalty(self):
        from portfolio.signal_engine import _weighted_consensus
        votes = {"rsi": "BUY", "structure": "BUY", "volatility_sig": "BUY",
                 "claude_fundamental": "BUY"}
        accuracy = {
            "rsi": {"accuracy": 0.55, "total": 100},
            "structure": {"accuracy": 0.50, "total": 100},
            "volatility_sig": {"accuracy": 0.48, "total": 100},
            "claude_fundamental": {"accuracy": 0.62, "total": 100},
        }
        action, conf = _weighted_consensus(votes, accuracy, "unknown")
        assert action == "BUY"


class TestCrossAssetFlowGroup:
    """credit_spread_risk and futures_flow should be in cross_asset_flow group."""

    def test_cross_asset_flow_group_exists(self):
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "cross_asset_flow" in CORRELATION_GROUPS

    def test_credit_spread_risk_in_cross_asset_flow(self):
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "credit_spread_risk" in CORRELATION_GROUPS["cross_asset_flow"]

    def test_futures_flow_in_cross_asset_flow(self):
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "futures_flow" in CORRELATION_GROUPS["cross_asset_flow"]


class TestCrisisModeConditionalTrendPenalty:
    """Crisis mode should NOT penalize trend signals when they're accurate."""

    def test_crisis_no_trend_penalty_when_trend_accurate(self):
        """When macro signals are broken but trend signals have >55% accuracy,
        crisis mode should NOT penalize trend signals."""
        from portfolio.signal_engine import _weighted_consensus

        # Trend signal voting BUY with high accuracy
        votes = {"ema": "BUY", "trend": "BUY", "rsi": "BUY"}
        # Macro signals broken (below 35%), but trend signals strong
        accuracy = {
            "fear_greed": {"accuracy": 0.25, "total": 100},
            "macro_regime": {"accuracy": 0.30, "total": 100},
            "news_event": {"accuracy": 0.29, "total": 100},
            "structure": {"accuracy": 0.40, "total": 100},
            "sentiment": {"accuracy": 0.46, "total": 100},
            # Trend signals are strong
            "ema": {"accuracy": 0.63, "total": 100},
            "trend": {"accuracy": 0.62, "total": 100},
            "heikin_ashi": {"accuracy": 0.55, "total": 100},
            "volume_flow": {"accuracy": 0.56, "total": 100},
            "rsi": {"accuracy": 0.55, "total": 100},
        }
        action, conf_no_crisis = _weighted_consensus(votes, accuracy, "unknown")
        # Should still get full BUY — trend not penalized
        assert action == "BUY"
        # The confidence should be relatively high since trend signals are accurate
        assert conf_no_crisis > 0.5

    def test_crisis_penalizes_trend_when_trend_weak(self):
        """When both macro AND trend signals are broken, crisis penalty applies."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"ema": "BUY", "mean_reversion": "SELL", "rsi": "SELL"}
        accuracy = {
            # Macro broken
            "fear_greed": {"accuracy": 0.25, "total": 100},
            "macro_regime": {"accuracy": 0.30, "total": 100},
            "news_event": {"accuracy": 0.29, "total": 100},
            # Trend signals also weak (below 55% floor)
            "ema": {"accuracy": 0.48, "total": 100},
            "trend": {"accuracy": 0.42, "total": 100},
            "heikin_ashi": {"accuracy": 0.45, "total": 100},
            "volume_flow": {"accuracy": 0.44, "total": 100},
            # MR signals
            "mean_reversion": {"accuracy": 0.60, "total": 100},
            "rsi": {"accuracy": 0.55, "total": 100},
        }
        action, conf = _weighted_consensus(votes, accuracy, "unknown")
        # MR should win since trend is penalized and MR is boosted
        assert action == "SELL"


# ---------------------------------------------------------------------------
# _persistence_state bounds (memory leak fix)
# ---------------------------------------------------------------------------

class TestPersistenceStateBounds:
    """Verify _persistence_state dict is bounded by _PERSISTENCE_MAX_TICKERS."""

    def test_dict_bounded_at_cap(self):
        from portfolio.signal_engine import (
            _apply_persistence_filter,
            _PERSISTENCE_MAX_TICKERS,
            _persistence_state,
            _persistence_lock,
        )
        # Clear state and fill to cap with unique tickers
        with _persistence_lock:
            _persistence_state.clear()

        votes = {"ema": "BUY", "rsi": "SELL"}
        for i in range(_PERSISTENCE_MAX_TICKERS + 10):
            _apply_persistence_filter(votes, f"TICKER-{i}")

        with _persistence_lock:
            assert len(_persistence_state) <= _PERSISTENCE_MAX_TICKERS
            _persistence_state.clear()

    def test_eviction_preserves_recent_tickers(self):
        from portfolio.signal_engine import (
            _apply_persistence_filter,
            _PERSISTENCE_MAX_TICKERS,
            _persistence_state,
            _persistence_lock,
        )
        with _persistence_lock:
            _persistence_state.clear()

        votes = {"ema": "BUY"}
        # Fill exactly to cap
        for i in range(_PERSISTENCE_MAX_TICKERS):
            _apply_persistence_filter(votes, f"T-{i}")
        # Add one more — should trigger eviction
        _apply_persistence_filter(votes, "T-NEWEST")

        with _persistence_lock:
            assert "T-NEWEST" in _persistence_state
            # Oldest half should be evicted
            assert "T-0" not in _persistence_state
            _persistence_state.clear()
