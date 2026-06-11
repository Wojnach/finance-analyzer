"""crypto_loop.fetch_live_prices 0.0-price guard (audit B8 fix 5).

A missing/zero lastPrice in a 200 response previously stored price 0.0,
feeding -100% moves into the swing trader and a ZeroDivisionError into the
fast-tick reference. Mirror the oil_loop guard: non-positive => skip ticker.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests

from data import crypto_loop


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def test_zero_price_is_dropped(monkeypatch):
    # BTC returns a valid price, ETH returns 0.0 -> ETH must be absent.
    def _fake_get(url, params=None, timeout=None):
        sym = params["symbol"]
        if sym == "BTCUSDT":
            return _Resp({"lastPrice": "65000.0"})
        return _Resp({"lastPrice": "0"})

    monkeypatch.setattr(requests, "get", _fake_get)
    out = crypto_loop.fetch_live_prices()
    assert out.get("BTC-USD") == 65000.0
    assert "ETH-USD" not in out  # zero treated as fetch failure


def test_negative_price_is_dropped(monkeypatch):
    def _fake_get(url, params=None, timeout=None):
        return _Resp({"lastPrice": "-1.0"})

    monkeypatch.setattr(requests, "get", _fake_get)
    out = crypto_loop.fetch_live_prices()
    assert out == {}  # nothing positive => empty


def test_missing_lastprice_is_dropped(monkeypatch):
    def _fake_get(url, params=None, timeout=None):
        return _Resp({})  # no lastPrice key -> defaults to 0

    monkeypatch.setattr(requests, "get", _fake_get)
    out = crypto_loop.fetch_live_prices()
    assert out == {}
