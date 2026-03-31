"""Tests for microstructure feature computations."""
from __future__ import annotations

import math

import numpy as np
import pytest


class TestDepthImbalance:
    def test_balanced_book(self):
        from portfolio.microstructure import depth_imbalance
        depth = {"bids": [[100, 10.0]], "asks": [[101, 10.0]]}
        result = depth_imbalance(depth)
        assert result == pytest.approx(0.0)

    def test_bid_heavy(self):
        from portfolio.microstructure import depth_imbalance
        depth = {"bids": [[100, 20.0], [99, 10.0]], "asks": [[101, 5.0]]}
        result = depth_imbalance(depth)
        assert result > 0

    def test_ask_heavy(self):
        from portfolio.microstructure import depth_imbalance
        depth = {"bids": [[100, 5.0]], "asks": [[101, 20.0], [102, 10.0]]}
        result = depth_imbalance(depth)
        assert result < 0

    def test_log_ratio_formula(self):
        from portfolio.microstructure import depth_imbalance
        depth = {"bids": [[100, 10.0]], "asks": [[101, 5.0]]}
        result = depth_imbalance(depth)
        expected = math.log(10.0) - math.log(5.0)
        assert result == pytest.approx(expected)

    def test_multi_level_sums_volumes(self):
        from portfolio.microstructure import depth_imbalance
        depth = {
            "bids": [[100, 5.0], [99, 5.0]],
            "asks": [[101, 3.0], [102, 7.0]],
        }
        result = depth_imbalance(depth)
        assert result == pytest.approx(0.0)


class TestTradeFlowImbalance:
    def test_all_buys(self):
        from portfolio.microstructure import trade_flow_imbalance
        trades = [
            {"price": 100, "qty": 1.0, "sign": 1},
            {"price": 101, "qty": 2.0, "sign": 1},
        ]
        result = trade_flow_imbalance(trades)
        assert result["signed_volume"] == pytest.approx(3.0)
        assert result["buy_volume"] == pytest.approx(3.0)
        assert result["sell_volume"] == pytest.approx(0.0)
        assert result["imbalance_ratio"] == pytest.approx(1.0)

    def test_all_sells(self):
        from portfolio.microstructure import trade_flow_imbalance
        trades = [
            {"price": 100, "qty": 1.0, "sign": -1},
            {"price": 101, "qty": 2.0, "sign": -1},
        ]
        result = trade_flow_imbalance(trades)
        assert result["signed_volume"] == pytest.approx(-3.0)
        assert result["imbalance_ratio"] == pytest.approx(-1.0)

    def test_balanced_trades(self):
        from portfolio.microstructure import trade_flow_imbalance
        trades = [
            {"price": 100, "qty": 5.0, "sign": 1},
            {"price": 100, "qty": 5.0, "sign": -1},
        ]
        result = trade_flow_imbalance(trades)
        assert result["signed_volume"] == pytest.approx(0.0)
        assert result["imbalance_ratio"] == pytest.approx(0.0)

    def test_empty_trades(self):
        from portfolio.microstructure import trade_flow_imbalance
        result = trade_flow_imbalance([])
        assert result is None


class TestVPIN:
    def test_one_sided_flow(self):
        from portfolio.microstructure import compute_vpin
        trades = [{"price": 100, "qty": 1.0, "sign": 1} for _ in range(100)]
        result = compute_vpin(trades, n_buckets=10)
        assert result == pytest.approx(1.0)

    def test_balanced_flow(self):
        from portfolio.microstructure import compute_vpin
        trades = []
        for i in range(100):
            trades.append({"price": 100, "qty": 1.0, "sign": 1 if i % 2 == 0 else -1})
        result = compute_vpin(trades, n_buckets=10)
        assert result == pytest.approx(0.0, abs=0.15)

    def test_insufficient_trades(self):
        from portfolio.microstructure import compute_vpin
        trades = [{"price": 100, "qty": 1.0, "sign": 1}]
        result = compute_vpin(trades, n_buckets=10)
        assert result is None


class TestOFI:
    def test_bid_improving(self):
        from portfolio.microstructure import compute_ofi
        snapshots = [
            {"best_bid": 100.0, "best_ask": 101.0,
             "bids": [[100.0, 10.0]], "asks": [[101.0, 10.0]]},
            {"best_bid": 100.5, "best_ask": 101.0,
             "bids": [[100.5, 8.0]], "asks": [[101.0, 10.0]]},
        ]
        result = compute_ofi(snapshots)
        assert result > 0

    def test_ask_improving(self):
        from portfolio.microstructure import compute_ofi
        snapshots = [
            {"best_bid": 100.0, "best_ask": 101.0,
             "bids": [[100.0, 10.0]], "asks": [[101.0, 10.0]]},
            {"best_bid": 100.0, "best_ask": 100.5,
             "bids": [[100.0, 10.0]], "asks": [[100.5, 8.0]]},
        ]
        result = compute_ofi(snapshots)
        assert result < 0

    def test_single_snapshot(self):
        from portfolio.microstructure import compute_ofi
        snapshots = [
            {"best_bid": 100.0, "best_ask": 101.0,
             "bids": [[100.0, 10.0]], "asks": [[101.0, 10.0]]},
        ]
        result = compute_ofi(snapshots)
        assert result == 0.0


class TestSpreadZScore:
    def test_normal_spread(self):
        from portfolio.microstructure import spread_zscore
        spreads = [0.50] * 20
        result = spread_zscore(spreads)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_wide_spread(self):
        from portfolio.microstructure import spread_zscore
        spreads = [0.50] * 19 + [2.0]
        result = spread_zscore(spreads)
        assert result > 1.5

    def test_insufficient_data(self):
        from portfolio.microstructure import spread_zscore
        result = spread_zscore([0.5])
        assert result is None
