"""Tests for portfolio.component_registry — Phase 4.1 parity harness.

The registry is a generated snapshot (portfolio/registry_defaults.py) plus
independent read logic (portfolio/component_registry.py). The parity tests
below compute the "legacy answer" directly from the LIVE constants in
portfolio.tickers / portfolio.signal_engine — the mechanisms the registry
is meant to eventually replace — and assert the registry agrees. A mismatch
here means either registry_defaults.py is stale (regenerate via
scripts/gen_registry_defaults.py) or component_registry.py's logic drifted
from _compute_applicable_count's semantics. This is the Phase 4.2 cutover
gate — see docs/plans/2026-07-18-dashboard-redesign-and-modular-engine.md.
"""

from __future__ import annotations

import json

import pytest

from portfolio import signal_engine
from portfolio.component_registry import ComponentRegistry
from portfolio.tickers import ALL_TICKERS, DISABLED_SIGNALS, SIGNAL_NAMES

TICKERS = sorted(ALL_TICKERS)

# Every horizon key signal_engine._TICKER_DISABLED_BY_HORIZON recognizes,
# plus None (== "_default", the compute-time / horizon-less gate).
HORIZONS = (None, "3h", "4h", "12h", "1d", "3d", "5d", "10d")


def _legacy_is_enabled(signal: str, ticker: str, horizon: str | None) -> bool:
    """Ground truth for is_enabled(), computed straight from the live
    signal_engine/tickers constants (not from the generated snapshot)."""
    if (
        signal in DISABLED_SIGNALS
        and (signal, ticker) not in signal_engine._DISABLED_SIGNAL_OVERRIDES
    ):
        return False
    if (
        signal in signal_engine._CRYPTO_ONLY_SIGNALS
        and ticker not in signal_engine.CRYPTO_SYMBOLS
    ):
        return False
    if (
        signal in signal_engine._METALS_ONLY_SIGNALS
        and ticker not in signal_engine.METALS_SYMBOLS
    ):
        return False
    if (
        signal in signal_engine._NON_STOCK_SIGNALS
        and ticker in signal_engine.STOCK_SYMBOLS
    ):
        return False
    if signal in signal_engine._get_horizon_disabled_signals(ticker, horizon):
        return False
    return True


@pytest.fixture()
def registry(tmp_path) -> ComponentRegistry:
    """A registry with no overlay (nonexistent path) — parity is measured
    against the generated defaults only. A real overlay is meant to
    deliberately diverge from legacy, so it has no place in this harness."""
    return ComponentRegistry(overlay_path=tmp_path / "no_overlay.json")


class TestParityIsEnabled:
    def test_matches_legacy_for_every_signal_ticker_horizon(self, registry):
        mismatches = []
        for signal in SIGNAL_NAMES:
            for ticker in TICKERS:
                for horizon in HORIZONS:
                    expected = _legacy_is_enabled(signal, ticker, horizon)
                    actual = registry.is_enabled(signal, ticker, horizon)
                    if actual != expected:
                        mismatches.append((signal, ticker, horizon, expected, actual))
        assert not mismatches, (
            f"{len(mismatches)} parity mismatches (signal, ticker, horizon, "
            f"expected, actual): {mismatches[:20]}"
        )


class TestParityApplicableCount:
    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("skip_gpu", [True, False])
    def test_matches_compute_applicable_count(self, registry, ticker, skip_gpu):
        expected = signal_engine._compute_applicable_count(ticker, skip_gpu=skip_gpu)
        actual = registry.applicable_count(ticker, skip_gpu=skip_gpu)
        assert actual == expected

    @pytest.mark.parametrize("ticker", TICKERS)
    @pytest.mark.parametrize("skip_gpu", [True, False])
    def test_applicable_signals_matches_applicable_count(
        self, registry, ticker, skip_gpu
    ):
        # applicable_count must be exactly len(applicable_signals) — no
        # separate counting logic to drift out of sync.
        signals = registry.applicable_signals(ticker, skip_gpu=skip_gpu)
        assert len(signals) == registry.applicable_count(ticker, skip_gpu=skip_gpu)
        assert signals <= frozenset(SIGNAL_NAMES)


class TestOverlay:
    def test_overlay_enabled_override_flips_is_enabled(self, tmp_path):
        overlay_path = tmp_path / "registry_overrides.json"
        # "ml" is globally disabled and has no rescue for BTC-USD.
        assert not ComponentRegistry(overlay_path=overlay_path).is_enabled(
            "ml", "BTC-USD"
        )
        overlay_path.write_text(
            json.dumps(
                {"BTC-USD": {"ml": {"enabled": True, "reason": "operator test"}}}
            )
        )
        reg = ComponentRegistry(overlay_path=overlay_path)
        assert reg.is_enabled("ml", "BTC-USD") is True
        assert reg.disabled_reason("ml", "BTC-USD") is None

    def test_overlay_disabled_override_flips_is_enabled(self, tmp_path):
        overlay_path = tmp_path / "registry_overrides.json"
        # "rsi" is a core, globally-enabled signal.
        assert ComponentRegistry(overlay_path=overlay_path).is_enabled("rsi", "BTC-USD")
        overlay_path.write_text(
            json.dumps(
                {
                    "BTC-USD": {
                        "rsi": {"enabled": False, "reason": "operator paused rsi"}
                    }
                }
            )
        )
        reg = ComponentRegistry(overlay_path=overlay_path)
        assert reg.is_enabled("rsi", "BTC-USD") is False
        assert reg.disabled_reason("rsi", "BTC-USD") == "operator paused rsi"

    def test_overlay_horizon_override_is_scoped_to_that_horizon(self, tmp_path):
        overlay_path = tmp_path / "registry_overrides.json"
        overlay_path.write_text(
            json.dumps({"BTC-USD": {"rsi": {"horizons": {"1d": False}}}})
        )
        reg = ComponentRegistry(overlay_path=overlay_path)
        assert reg.is_enabled("rsi", "BTC-USD", "1d") is False
        # Other horizons untouched by the horizon-scoped override.
        assert reg.is_enabled("rsi", "BTC-USD", "3h") is True
        assert reg.is_enabled("rsi", "BTC-USD") is True

    def test_overlay_reloads_when_file_changes(self, tmp_path):
        overlay_path = tmp_path / "registry_overrides.json"
        reg = ComponentRegistry(overlay_path=overlay_path)
        assert reg.is_enabled("ml", "BTC-USD") is False
        overlay_path.write_text(json.dumps({"BTC-USD": {"ml": {"enabled": True}}}))
        assert reg.is_enabled("ml", "BTC-USD") is True

    def test_missing_overlay_file_means_no_overrides(self, tmp_path):
        reg = ComponentRegistry(overlay_path=tmp_path / "does_not_exist.json")
        assert reg.is_enabled("rsi", "BTC-USD") is True
        assert reg.is_enabled("ml", "BTC-USD") is False

    def test_malformed_overlay_is_ignored_not_raised(self, tmp_path, caplog):
        overlay_path = tmp_path / "registry_overrides.json"
        overlay_path.write_text("[not, an, object]")
        reg = ComponentRegistry(overlay_path=overlay_path)
        # Falls back to defaults rather than raising.
        assert reg.is_enabled("rsi", "BTC-USD") is True
        assert reg.is_enabled("ml", "BTC-USD") is False

    def test_overlay_with_non_dict_ticker_entry_is_ignored(self, tmp_path):
        overlay_path = tmp_path / "registry_overrides.json"
        overlay_path.write_text(json.dumps({"BTC-USD": "not a dict"}))
        reg = ComponentRegistry(overlay_path=overlay_path)
        assert reg.is_enabled("rsi", "BTC-USD") is True


class TestUnknownSignalAndTicker:
    def test_unknown_signal_is_not_enabled(self, registry):
        assert registry.is_enabled("totally_made_up_signal", "BTC-USD") is False

    def test_unknown_signal_disabled_reason(self, registry):
        assert registry.disabled_reason("totally_made_up_signal") == "unknown signal"
        assert (
            registry.disabled_reason("totally_made_up_signal", "BTC-USD")
            == "unknown signal"
        )

    def test_unknown_signal_voter_state(self, registry):
        state = registry.voter_state("totally_made_up_signal")
        assert state == {"state": "DISABLED", "reason": "unknown signal"}

    def test_unknown_ticker_falls_back_to_global_only(self, registry):
        # "rsi" is globally enabled and not asset-class-restricted -> an
        # unknown ticker still gets it, since no per-ticker table applies.
        assert registry.is_enabled("rsi", "NOPE-USD") is True
        # "ml" is globally disabled with no override for an unknown ticker.
        assert registry.is_enabled("ml", "NOPE-USD") is False
        # crypto-only signals: an unrecognized ticker isn't in CRYPTO_SYMBOLS,
        # so it's excluded exactly like a known non-crypto ticker would be.
        assert registry.is_enabled("funding", "NOPE-USD") is False


class TestIsGloballyDisabled:
    """Phase 4.2 prep: signal_engine needs a pure global-disable check (no
    ticker/horizon blacklist, no asset-class restriction) to preserve the
    shadow-registry promotion nuance at the dispatch gate — see
    docs/plans/2026-07-18-dashboard-redesign-and-modular-engine.md Phase 4.2.
    """

    def test_matches_legacy_for_every_signal_ticker(self, registry):
        mismatches = []
        for signal in SIGNAL_NAMES:
            for ticker in TICKERS:
                expected = (
                    signal in DISABLED_SIGNALS
                    and (signal, ticker) not in signal_engine._DISABLED_SIGNAL_OVERRIDES
                )
                actual = registry.is_globally_disabled(signal, ticker)
                if actual != expected:
                    mismatches.append((signal, ticker, expected, actual))
        assert not mismatches, mismatches[:20]

    def test_enabled_signal_is_not_globally_disabled(self, registry):
        assert registry.is_globally_disabled("rsi", "BTC-USD") is False

    def test_disabled_signal_with_no_override_is_globally_disabled(self, registry):
        assert registry.is_globally_disabled("macd", "BTC-USD") is True

    def test_disabled_signal_with_ticker_override_is_not_globally_disabled(
        self, registry
    ):
        # ml is disabled globally but rescued for ETH-USD.
        assert registry.is_globally_disabled("ml", "ETH-USD") is False
        assert registry.is_globally_disabled("ml", "BTC-USD") is True

    def test_ignores_ticker_horizon_blacklist(self, registry):
        # news_event is globally enabled but blacklisted for ETH-USD at every
        # horizon — is_globally_disabled must not see that axis at all.
        assert not registry.is_enabled("news_event", "ETH-USD")
        assert registry.is_globally_disabled("news_event", "ETH-USD") is False

    def test_unknown_signal_is_NOT_globally_disabled(self, registry):
        """Parity with the legacy dispatch gate: `sig in DISABLED_SIGNALS` is
        False for unknown names, so the P1 axis must not disable them.
        (is_enabled still answers False for unknowns -- different question.)
        Regression: 2026-07-18 pre-flip gate caught a synthetic test signal
        flipping from voting to force-HOLD under the flag."""
        assert (
            registry.is_globally_disabled("totally_made_up_signal", "BTC-USD") is False
        )
        assert not registry.is_enabled("totally_made_up_signal", "BTC-USD")

    def test_overlay_enabled_override_wins(self, tmp_path):
        overlay_path = tmp_path / "registry_overrides.json"
        overlay_path.write_text(json.dumps({"BTC-USD": {"ml": {"enabled": True}}}))
        reg = ComponentRegistry(overlay_path=overlay_path)
        assert reg.is_globally_disabled("ml", "BTC-USD") is False


class TestIsTickerHorizonBlacklisted:
    """The P2 counterpart to is_globally_disabled — pure ticker/horizon
    blacklist, ignoring global disable/override and asset-class
    restriction. signal_engine composes the two independently so a
    shadow-registry promotion (which only bypasses the global-disable axis)
    can never silently bypass this one too.
    """

    def test_matches_legacy_for_every_signal_ticker_horizon(self, registry):
        mismatches = []
        for signal in SIGNAL_NAMES:
            for ticker in TICKERS:
                for horizon in HORIZONS:
                    expected = signal in signal_engine._get_horizon_disabled_signals(
                        ticker, horizon
                    )
                    actual = registry.is_ticker_horizon_blacklisted(
                        signal, ticker, horizon
                    )
                    if actual != expected:
                        mismatches.append((signal, ticker, horizon, expected, actual))
        assert not mismatches, mismatches[:20]

    def test_ignores_global_disable(self, registry):
        # smart_money is globally disabled AND blacklisted for BTC-USD —
        # this must report True purely from the blacklist axis, not because
        # it's also disabled (that's is_globally_disabled's job).
        assert registry.is_globally_disabled("smart_money", "BTC-USD") is True
        assert registry.is_ticker_horizon_blacklisted("smart_money", "BTC-USD") is True
        # rsi (enabled globally, not blacklisted anywhere) stays False.
        assert registry.is_ticker_horizon_blacklisted("rsi", "BTC-USD") is False

    def test_enabled_but_blacklisted_signal(self, registry):
        # news_event is globally enabled but blacklisted for ETH-USD.
        assert registry.is_globally_disabled("news_event", "ETH-USD") is False
        assert registry.is_ticker_horizon_blacklisted("news_event", "ETH-USD") is True

    def test_overlay_horizon_key_overrides_static_table(self, tmp_path):
        overlay_path = tmp_path / "registry_overrides.json"
        overlay_path.write_text(
            json.dumps({"BTC-USD": {"rsi": {"horizons": {"1d": False}}}})
        )
        reg = ComponentRegistry(overlay_path=overlay_path)
        assert reg.is_ticker_horizon_blacklisted("rsi", "BTC-USD", "1d") is True
        assert reg.is_ticker_horizon_blacklisted("rsi", "BTC-USD", "3h") is False
        assert reg.is_ticker_horizon_blacklisted("rsi", "BTC-USD") is False

    def test_overlay_enabled_key_is_not_consulted(self, tmp_path):
        # is_ticker_horizon_blacklisted must NOT look at the overlay's
        # top-level `enabled` key -- that's is_globally_disabled's exclusive
        # concern (avoids the two wrappers double-answering the same key).
        overlay_path = tmp_path / "registry_overrides.json"
        overlay_path.write_text(json.dumps({"BTC-USD": {"rsi": {"enabled": False}}}))
        reg = ComponentRegistry(overlay_path=overlay_path)
        assert reg.is_ticker_horizon_blacklisted("rsi", "BTC-USD") is False


class TestDisabledReason:
    def test_global_disabled_reason_ignores_per_ticker_rescue(self, registry):
        # "ml" is rescued for ETH-USD but the ticker=None (global) query
        # reports the global DISABLED_SIGNALS reason regardless.
        assert registry.disabled_reason("ml") is not None
        assert registry.is_enabled("ml", "ETH-USD") is True

    def test_enabled_signal_has_no_disabled_reason(self, registry):
        assert registry.disabled_reason("rsi", "BTC-USD") is None
        assert registry.disabled_reason("rsi") is None

    def test_ticker_horizon_blacklisted_signal_has_a_reason(self, registry):
        # "news_event" is globally enabled but blacklisted for ETH-USD at
        # every horizon (_default) — must not silently report None.
        assert not registry.is_enabled("news_event", "ETH-USD")
        assert registry.disabled_reason("news_event", "ETH-USD") is not None


class TestVoterState:
    def test_shadow_llm_signal_is_shadow_even_when_ticker_rescued(self, registry):
        # phi4_mini is rescued per-ticker via _DISABLED_SIGNAL_OVERRIDES but
        # is also in _KNOWN_SHADOW_LLMS — registry-level state stays SHADOW
        # (dashboard layers remote-gate truth on top; see module docstring).
        assert registry.is_enabled("phi4_mini", "BTC-USD") is True
        assert registry.voter_state("phi4_mini", "BTC-USD")["state"] == "SHADOW"

    def test_core_active_signal_is_voting(self, registry):
        assert registry.voter_state("rsi", "BTC-USD") == {
            "state": "VOTING",
            "reason": None,
        }

    def test_disabled_signal_is_disabled(self, registry):
        state = registry.voter_state("macd", "BTC-USD")
        assert state["state"] == "DISABLED"
        assert state["reason"]


class TestSnapshot:
    def test_snapshot_covers_all_tickers_and_signals(self, registry):
        snap = registry.snapshot()
        from portfolio.registry_defaults import TICKERS as REG_TICKERS

        assert set(snap.keys()) == set(REG_TICKERS)
        for ticker, signals in snap.items():
            assert set(signals.keys()) == set(SIGNAL_NAMES)
            for entry in signals.values():
                assert "enabled_default" in entry
                assert "horizons" in entry
                assert "voter_state" in entry
