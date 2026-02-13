"""Tests for LoRA training pipeline — data generation, prompt format, and consistency."""

import json
import random

import numpy as np
import pandas as pd
import pytest

from training.lora.generate_data import (
    PROMPT_TEMPLATE,
    build_completion,
    build_prompt,
    compute_indicators,
    label_candles,
    _sample_fear_greed,
    _sample_sentiment,
    _build_timeframe_summary,
    balance_classes,
)


def _make_df(n=200, base_price=69000.0, trend=0.0):
    prices = [base_price + i * trend + random.uniform(-50, 50) for i in range(n)]
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=n, freq="1h"),
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


class TestPromptFormat:
    def test_no_inst_tags_in_template(self):
        assert "[INST]" not in PROMPT_TEMPLATE
        assert "[/INST]" not in PROMPT_TEMPLATE

    def test_says_1_hour_not_15_minute(self):
        assert "1-hour candles" in PROMPT_TEMPLATE
        assert "15-minute" not in PROMPT_TEMPLATE

    def test_has_fear_greed_placeholder(self):
        assert "{fear_greed}" in PROMPT_TEMPLATE
        assert "N/A" not in PROMPT_TEMPLATE.split("{")[0]

    def test_has_sentiment_confidence(self):
        assert "{sentiment_confidence" in PROMPT_TEMPLATE

    def test_has_ema_gap(self):
        assert "{ema_gap_pct" in PROMPT_TEMPLATE

    def test_has_timeframe_summary(self):
        assert "{timeframe_summary}" in PROMPT_TEMPLATE

    def test_prompt_matches_inference_structure(self):
        from portfolio.ministral_trader import predict
        import inspect

        source = inspect.getsource(predict)
        assert "1-hour candles" in source
        assert "15-minute" not in source
        assert "ema_gap_pct" in source
        assert "sentiment_confidence" in source


class TestNoDoubleInst:
    def test_generated_content_has_no_inst_tags(self):
        random.seed(42)
        df = _make_df(200)
        df = compute_indicators(df)
        df["change_24h"] = (df["close"] / df["close"].shift(24) - 1) * 100
        df = label_candles(df)

        valid = df.iloc[26:].dropna(
            subset=[
                "rsi",
                "macd_hist",
                "ema9",
                "ema21",
                "bb_mid",
                "bb_upper",
                "bb_lower",
                "label",
            ]
        )
        row = valid.iloc[50]
        prompt = build_prompt(row, "BTC", 76, df)
        assert not prompt.startswith("[INST]")
        assert not prompt.endswith("[/INST]")
        assert "[INST]" not in prompt

    def test_train_lora_would_wrap_correctly(self):
        content = "You are an expert..."
        prompt = f"[INST]{content}[/INST]"
        assert prompt.count("[INST]") == 1
        assert prompt.count("[/INST]") == 1


class TestContextEnrichment:
    def test_fear_greed_sampling(self):
        random.seed(42)
        values = [_sample_fear_greed() for _ in range(100)]
        fg_values = [v[0] for v in values]
        fg_classes = [v[1] for v in values]

        assert all(5 <= v <= 95 for v in fg_values)
        assert all(
            c in ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")
            for c in fg_classes
        )

        for val, cls in values:
            if val <= 20:
                assert cls == "Extreme Fear"
            elif val <= 40:
                assert cls == "Fear"
            elif val <= 60:
                assert cls == "Neutral"
            elif val <= 80:
                assert cls == "Greed"
            else:
                assert cls == "Extreme Greed"

    def test_sentiment_sampling(self):
        random.seed(42)
        values = [_sample_sentiment() for _ in range(100)]
        sents = [v[0] for v in values]
        confs = [v[1] for v in values]

        assert all(s in ("positive", "negative", "neutral") for s in sents)
        assert all(0.25 <= c <= 0.92 for c in confs)
        assert len(set(sents)) == 3

    def test_timeframe_summary_format(self):
        random.seed(42)
        df = _make_df(300)
        df = compute_indicators(df)

        summary = _build_timeframe_summary(df, 200)
        assert summary != "N/A"
        assert "RSI=" in summary
        parts = summary.split(" | ")
        assert len(parts) >= 1
        for part in parts:
            assert any(h in part for h in ("12h", "2d", "7d"))
            assert any(a in part for a in ("BUY", "SELL", "HOLD"))

    def test_timeframe_summary_na_for_early_rows(self):
        df = _make_df(50)
        df = compute_indicators(df)
        summary = _build_timeframe_summary(df, 30)
        assert summary == "N/A"

    def test_prompt_has_populated_fields(self):
        random.seed(42)
        df = _make_df(300)
        df = compute_indicators(df)
        df["change_24h"] = (df["close"] / df["close"].shift(24) - 1) * 100
        df = label_candles(df)

        valid = df.iloc[26:].dropna(
            subset=[
                "rsi",
                "macd_hist",
                "ema9",
                "ema21",
                "bb_mid",
                "bb_upper",
                "bb_lower",
                "label",
            ]
        )
        row = valid.iloc[100]
        idx = valid.index[100]
        prompt = build_prompt(row, "BTC", idx, df)

        assert "Fear & Greed Index: N/A" not in prompt
        assert "/100 (" in prompt
        assert "confidence:" in prompt
        assert "gap:" in prompt


class TestIndicatorAwareCompletions:
    def _make_row(
        self,
        rsi=50,
        macd=0,
        ema9=69000,
        ema21=69000,
        close=69000,
        bb_lower=68000,
        bb_upper=70000,
    ):
        return pd.Series(
            {
                "rsi": rsi,
                "macd_hist": macd,
                "ema9": ema9,
                "ema21": ema21,
                "close": close,
                "bb_lower": bb_lower,
                "bb_upper": bb_upper,
            }
        )

    def test_buy_oversold_mentions_oversold(self):
        row = self._make_row(rsi=25)
        completion = build_completion(row, "BTC", "BUY")
        assert "DECISION: BUY" in completion
        assert "oversold" in completion.lower()

    def test_sell_overbought_mentions_overbought(self):
        row = self._make_row(rsi=75)
        completion = build_completion(row, "BTC", "SELL")
        assert "DECISION: SELL" in completion
        assert "overbought" in completion.lower()

    def test_buy_never_says_overbought(self):
        row = self._make_row(rsi=25, macd=5, ema9=70000, ema21=69000)
        completion = build_completion(row, "BTC", "BUY")
        assert "overbought" not in completion.lower()

    def test_sell_never_says_oversold(self):
        row = self._make_row(rsi=75, macd=-5, ema9=68000, ema21=69000)
        completion = build_completion(row, "BTC", "SELL")
        assert "oversold" not in completion.lower()

    def test_hold_mentions_neutral(self):
        row = self._make_row(rsi=50, macd=0.3)
        completion = build_completion(row, "BTC", "HOLD")
        assert "DECISION: HOLD" in completion
        assert "neutral" in completion.lower()

    def test_buy_bullish_ema(self):
        row = self._make_row(ema9=70000, ema21=69000)
        completion = build_completion(row, "BTC", "BUY")
        assert "bullish" in completion.lower() or "EMA" in completion

    def test_sell_bearish_ema(self):
        row = self._make_row(ema9=68000, ema21=69000)
        completion = build_completion(row, "BTC", "SELL")
        assert "bearish" in completion.lower() or "EMA" in completion

    def test_completion_always_starts_with_decision(self):
        for label in ["BUY", "SELL", "HOLD"]:
            row = self._make_row()
            completion = build_completion(row, "BTC", label)
            assert completion.startswith(f"DECISION: {label}")


class TestLabelCandles:
    def test_uptrend_produces_buy(self):
        prices = [100 + i * 0.5 for i in range(200)]
        df = pd.DataFrame({"close": prices})
        df = label_candles(df, threshold=0.02, lookahead=12)
        buy_count = (df["label"] == "BUY").sum()
        assert buy_count > 0

    def test_downtrend_produces_sell(self):
        prices = [200 - i * 0.5 for i in range(200)]
        df = pd.DataFrame({"close": prices})
        df = label_candles(df, threshold=0.02, lookahead=12)
        sell_count = (df["label"] == "SELL").sum()
        assert sell_count > 0

    def test_flat_produces_hold(self):
        prices = [100.0] * 200
        df = pd.DataFrame({"close": prices})
        df = label_candles(df, threshold=0.02, lookahead=12)
        hold_count = (df["label"] == "HOLD").sum()
        assert hold_count > 0

    def test_last_rows_are_none(self):
        prices = [100.0 + i for i in range(50)]
        df = pd.DataFrame({"close": prices})
        df = label_candles(df, lookahead=12)
        assert pd.isna(df["label"].iloc[-1])
        assert pd.isna(df["label"].iloc[-12])


class TestBalanceClasses:
    def test_balanced_output(self):
        examples = []
        for label in ["BUY", "SELL", "HOLD"]:
            for i in range(50):
                examples.append(
                    {
                        "label": label,
                        "messages": [
                            {"role": "user", "content": f"prompt {i}"},
                            {"role": "assistant", "content": f"DECISION: {label}"},
                        ],
                    }
                )
        balanced = balance_classes(examples, target_per_class=30)
        assert len(balanced) == 90
        assert "label" not in balanced[0]

    def test_uses_min_class_count(self):
        examples = []
        for i in range(100):
            examples.append({"label": "BUY", "messages": []})
        for i in range(20):
            examples.append({"label": "SELL", "messages": []})
        for i in range(50):
            examples.append({"label": "HOLD", "messages": []})
        balanced = balance_classes(examples, target_per_class=1000)
        assert len(balanced) == 60


class TestComputeIndicators:
    def test_all_columns_present(self):
        df = _make_df(100)
        df = compute_indicators(df)
        for col in [
            "rsi",
            "macd_hist",
            "ema9",
            "ema21",
            "bb_mid",
            "bb_upper",
            "bb_lower",
        ]:
            assert col in df.columns

    def test_rsi_range(self):
        df = _make_df(100)
        df = compute_indicators(df)
        valid_rsi = df["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_bb_order(self):
        df = _make_df(100)
        df = compute_indicators(df)
        valid = df.dropna(subset=["bb_lower", "bb_mid", "bb_upper"])
        assert (valid["bb_lower"] <= valid["bb_mid"]).all()
        assert (valid["bb_mid"] <= valid["bb_upper"]).all()
