"""The kill switch must fail closed, and every order path must respect it."""

from __future__ import annotations

import ast
import pathlib

import pytest

from portfolio import trading_gate as tg

REPO = pathlib.Path(__file__).resolve().parent.parent

# This module tests the kill switch itself, so it must not run under the
# conftest bypass that lets other suites exercise order-validation logic.
pytestmark = pytest.mark.real_trading_gate

# Every function that can place, modify or cancel a live Avanza order.
GATED = {
    "portfolio/avanza_session.py": [
        "place_buy_order",
        "place_sell_order",
        "cancel_order",
        "place_stop_loss",
    ],
    "portfolio/avanza_client.py": ["place_buy_order", "place_sell_order"],
    "portfolio/avanza/trading.py": ["place_order", "cancel_order", "place_stop_loss"],
    "portfolio/avanza_control.py": [
        "place_order",
        "place_stop_loss",
        "place_order_no_page",
        "place_stop_loss_no_page",
    ],
}


class TestFailsClosed:
    def test_disabled_when_flag_present(self, monkeypatch):
        monkeypatch.setattr(tg, "flag_present", lambda path=None: True)
        monkeypatch.setattr(tg, "config_allows", lambda: True)
        assert tg.trading_enabled() is False

    def test_disabled_when_config_not_opted_in(self, monkeypatch):
        monkeypatch.setattr(tg, "flag_present", lambda path=None: False)
        monkeypatch.setattr(tg, "config_allows", lambda: False)
        assert tg.trading_enabled() is False

    def test_requires_BOTH_flag_removed_and_config_true(self, monkeypatch):
        # Two independent locks: no single accidental `rm` can arm live trading.
        monkeypatch.setattr(tg, "flag_present", lambda path=None: False)
        monkeypatch.setattr(tg, "config_allows", lambda: True)
        assert tg.trading_enabled() is True

    def test_unreadable_flag_is_treated_as_disabled(self, monkeypatch):
        def boom(path=None):
            raise OSError("filesystem gone")

        monkeypatch.setattr(tg.os.path, "exists", boom)
        assert tg.flag_present() is True
        assert tg.trading_enabled() is False

    def test_unparseable_config_is_treated_as_disabled(self, monkeypatch):
        monkeypatch.setattr(tg, "flag_present", lambda path=None: False)

        def boom(*a, **k):
            raise ValueError("corrupt json")

        import portfolio.file_utils as fu

        monkeypatch.setattr(fu, "load_json", boom)
        assert tg.config_allows() is False
        assert tg.trading_enabled() is False


class TestRequireRaises:
    def test_raises_rather_than_returning_falsy(self, monkeypatch):
        # A caller that ignored a return code would go on to place the order.
        monkeypatch.setattr(tg, "trading_enabled", lambda: False)
        with pytest.raises(tg.TradingDisabledError) as e:
            tg.require_trading_enabled("place_buy_order")
        assert "place_buy_order" in str(e.value)
        assert "by hand" in str(e.value)

    def test_passes_through_when_enabled(self, monkeypatch):
        monkeypatch.setattr(tg, "trading_enabled", lambda: True)
        assert tg.require_trading_enabled("x") is True


class TestEveryOrderPathIsGated:
    """AST check, not a runtime call: a new order function added without the
    guard must fail this test even if nothing imports it yet."""

    @pytest.mark.parametrize("relpath,fnnames", [(k, v) for k, v in GATED.items()])
    def test_functions_call_the_gate(self, relpath, fnnames):
        tree = ast.parse((REPO / relpath).read_text(encoding="utf-8"))
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in fnnames:
                calls = {
                    c.func.id
                    for c in ast.walk(node)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                }
                found[node.name] = "require_trading_enabled" in calls
        for fn in fnnames:
            assert fn in found, f"{relpath}: {fn} not found — was it renamed?"
            assert found[fn], f"{relpath}: {fn} does NOT call require_trading_enabled"


class TestLiveStateIsDisabled:
    def test_repo_ships_with_trading_off(self):
        # The operator places every order by hand. If this ever fails, someone
        # armed live trading — that must be a deliberate, visible change.
        assert (
            tg.trading_enabled() is False
        ), f"live trading is ENABLED ({tg.reason()}) — expected disabled"
