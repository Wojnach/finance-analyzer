"""Regression tests for the 2026-05-28 adversarial bug-hunt fixes.

Each test pins a specific confirmed-bug fix so it can't silently regress.
See the session notes for the full finding list; numbers below match them.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# #5 — avanza_client._place_order CONFIRM-path size guards
# ---------------------------------------------------------------------------

class TestAvanzaClientSizeGuards:
    def test_rejects_below_min_order(self):
        from portfolio import avanza_client
        # 5 * 10 = 50 SEK, well below the 1000 SEK minimum. The guard raises
        # before get_client(), so no live session is needed.
        with pytest.raises(ValueError, match="below minimum"):
            avanza_client._place_order("123", object(), price=10.0, volume=5,
                                       valid_until=None)

    def test_rejects_above_max_order(self):
        from portfolio import avanza_client
        # 100 * 1000 = 100_000 SEK, above the 50_000 SEK cap.
        with pytest.raises(ValueError, match="exceeds maximum"):
            avanza_client._place_order("123", object(), price=1000.0, volume=100,
                                       valid_until=None)


# ---------------------------------------------------------------------------
# #16 — price_source yfinance fallback alias translation
# ---------------------------------------------------------------------------

class TestPriceSourceYfinanceAlias:
    def test_dashed_metals_map_to_yfinance_symbols(self):
        from portfolio import price_source as ps
        assert ps._to_yfinance_symbol("XAG-USD") == "SI=F"
        assert ps._to_yfinance_symbol("XAU-USD") == "GC=F"
        assert ps._to_yfinance_symbol("XAGUSDT") == "SI=F"
        assert ps._to_yfinance_symbol("XAUUSDT") == "GC=F"

    def test_unmapped_symbol_is_identity(self):
        from portfolio import price_source as ps
        assert ps._to_yfinance_symbol("CL=F") == "CL=F"
        assert ps._to_yfinance_symbol("MSTR") == "MSTR"


# ---------------------------------------------------------------------------
# #4 — agent_invocation completion detection is prune-robust
# ---------------------------------------------------------------------------

class TestDetectAppend:
    def test_count_increase_is_append(self):
        from portfolio import agent_invocation as ai
        assert ai._detect_append(11, 10, None, None) is True

    def test_count_unchanged_same_ts_is_not_append(self):
        from portfolio import agent_invocation as ai
        assert ai._detect_append(5000, 5000, "t1", "t1") is False

    def test_pruned_but_ts_changed_is_append(self):
        from portfolio import agent_invocation as ai
        # prune dropped a line (count fell) but a new entry landed -> ts moved.
        assert ai._detect_append(5000, 5001, "t2", "t1") is True

    def test_count_unchanged_ts_changed_is_append(self):
        from portfolio import agent_invocation as ai
        # file at the prune cap: append + prune nets to equal count, but the
        # newest-entry ts moved -> a genuine append happened.
        assert ai._detect_append(5000, 5000, "t2", "t1") is True

    def test_pruned_no_ts_info_is_not_append(self):
        from portfolio import agent_invocation as ai
        assert ai._detect_append(4999, 5001, None, "t1") is False


# ---------------------------------------------------------------------------
# #9 — dashboard auth fails CLOSED on cold-start config read failure
# ---------------------------------------------------------------------------

class TestAuthColdStart:
    def _reset(self, monkeypatch, auth):
        monkeypatch.setattr(auth, "_CFG_VALUE", None)
        monkeypatch.setattr(auth, "_CFG_AT", 0.0)
        monkeypatch.setattr(auth, "_LAST_READ_OK", True)

    def test_cold_start_read_failure_is_unknown(self, monkeypatch):
        from dashboard import auth
        self._reset(monkeypatch, auth)
        # File present but unreadable/corrupt -> (data, ok=False).
        monkeypatch.setattr(auth, "_read_config_uncached", lambda: ({}, False))
        assert auth._get_config() == {}
        # Not cached, and config is NOT known -> require_auth must fail closed.
        assert auth._config_is_known() is False
        assert auth._get_dashboard_token() is None

    def test_genuine_absence_stays_open(self, monkeypatch):
        from dashboard import auth
        self._reset(monkeypatch, auth)
        # Genuinely absent config -> (data, ok=True): backward-compat fail-open.
        monkeypatch.setattr(auth, "_read_config_uncached", lambda: ({}, True))
        auth._get_config()
        assert auth._config_is_known() is True

    def test_warm_cache_survives_later_failure(self, monkeypatch):
        from dashboard import auth
        self._reset(monkeypatch, auth)
        monkeypatch.setattr(auth, "_read_config_uncached",
                            lambda: ({"dashboard_token": "SECRET"}, True))
        assert auth._get_dashboard_token() == "SECRET"
        # Now reads fail, but the warm value must be retained (B11) and known.
        monkeypatch.setattr(auth, "_CFG_AT", 0.0)  # force re-read past TTL
        monkeypatch.setattr(auth, "_read_config_uncached", lambda: ({}, False))
        assert auth._get_dashboard_token() == "SECRET"
        assert auth._config_is_known() is True
