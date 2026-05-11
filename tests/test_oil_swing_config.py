"""Schema + invariant tests for data/oil_swing_config.py."""
from __future__ import annotations

import importlib

from data import oil_swing_config as cfg


class TestOilSwingConfigSchema:
    def test_instruments_is_oil_usd_only_in_v1(self):
        assert cfg.INSTRUMENTS == ("OIL-USD",)

    def test_data_sources_have_yfinance_symbols(self):
        for ticker in cfg.INSTRUMENTS:
            assert ticker in cfg.DATA_SOURCES
            assert "yfinance_symbol" in cfg.DATA_SOURCES[ticker]
            # WTI is the canonical front-month feed
            assert cfg.DATA_SOURCES[ticker]["yfinance_symbol"] == "CL=F"

    def test_dry_run_is_default_on(self):
        # Critical: must ship in DRY_RUN mode. Live trading requires manual flip.
        assert cfg.DRY_RUN is True

    def test_min_buy_confidence_at_or_above_25_pct(self):
        # 2026-05-11 Stage 2 follow-up: post-penalty floor (was 0.60
        # pre-Stage 1+2). See oil_swing_config.py rationale.
        assert cfg.MIN_BUY_CONFIDENCE >= 0.25

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

    def test_state_files_namespaced_to_oil(self):
        # Avoid collisions with metals/crypto state files.
        for attr in ("STATE_FILE", "DECISIONS_LOG", "TRADES_LOG",
                      "VALUE_HISTORY_LOG", "WARRANT_CATALOG_FILE"):
            v = getattr(cfg, attr)
            assert "oil" in v.lower(), f"{attr}={v!r} not namespaced to oil"

    def test_loss_escalation_monotonically_non_decreasing(self):
        prev = 0
        for k in sorted(cfg.LOSS_ESCALATION.keys()):
            v = cfg.LOSS_ESCALATION[k]
            assert v >= prev, f"escalation not monotonic at k={k}"
            prev = v

    def test_min_barrier_distance_reasonable(self):
        # Oil wicks are between metals (10%) and crypto (15%).
        assert 10 <= cfg.MIN_BARRIER_DISTANCE_PCT <= 15

    def test_take_profit_greater_than_hard_stop(self):
        assert cfg.HARD_STOP_UNDERLYING_PCT > 0
        assert cfg.TAKE_PROFIT_UNDERLYING_PCT > cfg.HARD_STOP_UNDERLYING_PCT

    def test_eod_exit_disabled_for_continuous_market(self):
        # Oil futures trade nearly 24/7 on CME; no EOD anchor.
        assert cfg.EOD_EXIT_MINUTES_BEFORE == 0
        assert cfg.MAX_HOLD_HOURS >= 24

    def test_max_concurrent_is_one(self):
        # v1 conservatism: single oil position (commodities have higher
        # single-event risk — OPEC, inventory).
        assert cfg.MAX_CONCURRENT == 1

    def test_account_id_matches_user_rule(self):
        # Per memory: cash-only account 1625505
        assert cfg.ACCOUNT_ID == "1625505"

    def test_min_trade_sek_meets_courtage_threshold(self):
        # Per memory: min order size 1000 SEK
        assert cfg.MIN_TRADE_SEK >= 1000


class TestWarrantCatalogFallback:
    def test_fallback_has_seed_warrants(self):
        # Seeded from data/avanza_instruments_live.json scrape (2026-04-30)
        assert len(cfg.WARRANT_CATALOG_FALLBACK) >= 4

    def test_fallback_includes_both_directions(self):
        directions = {w["direction"] for w in cfg.WARRANT_CATALOG_FALLBACK.values()}
        assert "LONG" in directions
        assert "SHORT" in directions

    def test_each_fallback_warrant_has_required_fields(self):
        for key, w in cfg.WARRANT_CATALOG_FALLBACK.items():
            for field in ("ob_id", "api_type", "underlying", "direction",
                          "name", "parity"):
                assert field in w, f"{key} missing {field}"
            assert w["underlying"] == "OIL-USD"
            assert w["direction"] in ("LONG", "SHORT")
            assert w["api_type"] in ("warrant", "certificate", "etf", "tracker")

    def test_each_fallback_has_ob_id(self):
        # Real Avanza instruments should have an ob_id from the scrape.
        for key, w in cfg.WARRANT_CATALOG_FALLBACK.items():
            assert w["ob_id"], f"{key} has no ob_id"
            assert w["ob_id"].isdigit() or w["ob_id"].startswith("AVA"), (
                f"{key} ob_id {w['ob_id']!r} not a numeric id"
            )


def test_module_reimport_idempotent():
    """Reload the module — fields must remain stable."""
    importlib.reload(cfg)
    assert cfg.DRY_RUN is True
    assert cfg.INSTRUMENTS == ("OIL-USD",)
