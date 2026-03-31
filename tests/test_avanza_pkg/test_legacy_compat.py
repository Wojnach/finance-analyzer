"""Tests for legacy import compatibility and new package imports.

Verifies that existing import paths (avanza_session, avanza_client)
still work alongside the new portfolio.avanza package.
"""

from __future__ import annotations


class TestLegacyImports:
    def test_avanza_session_exports(self):
        """Legacy avanza_session imports still resolve."""
        from portfolio.avanza_session import (  # noqa: F401
            AvanzaSessionError,
            api_get,
            api_post,
            cancel_order,
            get_buying_power,
            get_instrument_price,
            get_open_orders,
            get_positions,
            get_quote,
            is_session_expiring_soon,
            load_session,
            place_buy_order,
            place_sell_order,
            session_remaining_minutes,
            verify_session,
        )
        assert callable(api_get)

    def test_avanza_client_exports(self):
        """Legacy avanza_client imports still resolve."""
        from portfolio.avanza_client import (  # noqa: F401
            find_instrument,
            get_client,
            get_portfolio_value,
            get_positions,
            get_price,
        )
        assert callable(get_price)

    def test_new_package_importable(self):
        """New portfolio.avanza package exports all public symbols."""
        from portfolio.avanza import (  # noqa: F401
            AvanzaAuth,
            AvanzaClient,
            cancel_order,
            get_buying_power,
            get_market_data,
            get_ohlc,
            get_positions,
            get_quote,
            get_tick_rules,
            modify_order,
            place_order,
            place_trailing_stop,
            round_to_tick,
            search,
        )
        assert callable(get_quote)

    def test_streaming_importable(self):
        """AvanzaStream is importable from the package root."""
        from portfolio.avanza import AvanzaStream
        assert callable(AvanzaStream)

    def test_all_list_complete(self):
        """__all__ contains all expected public symbols."""
        import portfolio.avanza

        expected = {
            "AvanzaAuth", "AuthError", "AvanzaClient", "AvanzaStream",
            "ScannedInstrument", "scan_instruments", "format_scan_results",
            "get_positions", "get_buying_power", "get_transactions",
            "get_quote", "get_market_data", "get_ohlc", "get_instrument_info", "get_news",
            "search", "find_warrants", "find_certificates",
            "get_tick_rules", "round_to_tick", "clear_tick_cache",
            "place_order", "modify_order", "cancel_order",
            "get_orders", "get_deals",
            "place_stop_loss", "place_trailing_stop", "get_stop_losses", "delete_stop_loss",
        }
        actual = set(portfolio.avanza.__all__)
        assert actual == expected, f"Missing: {expected - actual}, Extra: {actual - expected}"
