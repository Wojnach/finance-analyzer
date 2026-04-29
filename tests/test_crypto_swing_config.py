"""Schema + invariant tests for data/crypto_swing_config.py."""
from __future__ import annotations

import importlib

import pytest

from data import crypto_swing_config as cfg


class TestCryptoSwingConfigSchema:
    def test_instruments_are_btc_and_eth(self):
        assert cfg.INSTRUMENTS == ("BTC-USD", "ETH-USD")

    def test_data_sources_have_binance_symbols(self):
        for ticker in cfg.INSTRUMENTS:
            assert ticker in cfg.DATA_SOURCES
            assert "binance_symbol" in cfg.DATA_SOURCES[ticker]
            assert "fapi_symbol" in cfg.DATA_SOURCES[ticker]

    def test_dry_run_is_default_on(self):
        # Critical: must ship in DRY_RUN mode. Live trading requires manual flip.
        assert cfg.DRY_RUN is True

    def test_min_buy_confidence_at_or_above_60_pct(self):
        # User rule: never trade below 60% calibrated confidence
        assert cfg.MIN_BUY_CONFIDENCE >= 0.60

    def test_min_buy_voters_at_or_above_3(self):
        assert cfg.MIN_BUY_VOTERS >= 3

    def test_momentum_thresholds_relaxed_relative_to_standard(self):
        assert cfg.MOMENTUM_MIN_BUY_CONFIDENCE < cfg.MIN_BUY_CONFIDENCE
        assert cfg.MOMENTUM_MIN_BUY_VOTERS < cfg.MIN_BUY_VOTERS

    def test_state_files_paths_are_strings(self):
        for attr in ("STATE_FILE", "DECISIONS_LOG", "TRADES_LOG",
                      "VALUE_HISTORY_LOG", "SIGNAL_LOG", "SIGNAL_OUTCOMES_LOG",
                      "RISK_FILE", "DEEP_CONTEXT_FILE", "WARRANT_CATALOG_FILE"):
            v = getattr(cfg, attr)
            assert isinstance(v, str), attr
            assert v.startswith("data/"), f"{attr} should be under data/"

    def test_loss_escalation_monotonically_non_decreasing(self):
        prev = 0
        for k in sorted(cfg.LOSS_ESCALATION.keys()):
            v = cfg.LOSS_ESCALATION[k]
            assert v >= prev, f"escalation not monotonic at k={k}"
            prev = v

    def test_min_barrier_distance_at_or_above_metals(self):
        # Crypto wicks are wider than metals; should not be tighter.
        assert cfg.MIN_BARRIER_DISTANCE_PCT >= 10

    def test_hard_stop_wider_than_take_profit_threshold_isnt_required(self):
        # Sanity: hard stop should be a positive percent, less than TP.
        assert cfg.HARD_STOP_UNDERLYING_PCT > 0
        assert cfg.TAKE_PROFIT_UNDERLYING_PCT > cfg.HARD_STOP_UNDERLYING_PCT

    def test_max_hold_hours_no_eod_for_24_7(self):
        # 24/7 markets should not be force-closed by EOD logic.
        assert cfg.EOD_EXIT_MINUTES_BEFORE == 0
        assert cfg.MAX_HOLD_HOURS >= 24


class TestWarrantCatalogFallback:
    def test_fallback_has_btc_and_eth_trackers(self):
        assert "XBT_TRACKER_AVA" in cfg.WARRANT_CATALOG_FALLBACK
        assert "ETH_TRACKER_AVA" in cfg.WARRANT_CATALOG_FALLBACK

    def test_each_fallback_warrant_has_required_fields(self):
        for key, w in cfg.WARRANT_CATALOG_FALLBACK.items():
            for field in ("api_type", "underlying", "direction",
                          "leverage", "name"):
                assert field in w, f"{key} missing {field}"
            assert w["underlying"] in ("BTC-USD", "ETH-USD")
            assert w["direction"] in ("LONG", "SHORT")


def test_module_reimport_idempotent():
    """Reload the module — fields must remain stable."""
    importlib.reload(cfg)
    assert cfg.DRY_RUN is True
    assert cfg.INSTRUMENTS == ("BTC-USD", "ETH-USD")
