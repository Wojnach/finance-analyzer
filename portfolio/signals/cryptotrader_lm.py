"""CryptoTrader-LM signal — scaffold (2026-05-15).

Status: SCAFFOLD. Returns HOLD with feature_unavailable=True until the
PEFT LoRA loader for `Q:/models/cryptotrader-lm/` is implemented and
verified. The model is a LoRA adapter on Ministral-8B-Instruct trained
for BTC/ETH buy/sell/hold decisions. Published card claims 0.94 Sharpe
and 72% directional accuracy on validation.

Crypto-only: this signal MUST refuse to vote on non-crypto tickers. The
contract enforces that at the compute boundary so the dispatch loop
above doesn't need a special case.

Inference contract (once wired):

    1. Build a crypto-specific prompt (recent OHLCV summary, funding,
       on-chain MVRV if BTC).
    2. Load Ministral-8B base + LoRA adapter via portfolio.llm_batch or
       a dedicated PEFT loader.
    3. Single forward pass; extract probabilities from {BUY, HOLD, SELL}
       token logprobs.
    4. Return the canonical result dict.

Until then this scaffold exists so the registry and probability log can
accumulate the housekeeping rows (days-in-shadow, abstention count)
that prove the wiring is live.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("portfolio.signals.cryptotrader_lm")

_MODEL_DIR = r"Q:\models\cryptotrader-lm"
_CRYPTO_TICKERS = frozenset({"BTC-USD", "ETH-USD"})
_FEATURE_AVAILABLE = False  # flip to True once inference is wired


def compute_cryptotrader_lm_signal(df, context=None):
    """Scaffold: HOLD on every call, with feature_unavailable=True.

    Refuses to vote on non-crypto tickers even after inference is wired
    (model was trained only on BTC/ETH; voting on metals or MSTR would
    be out-of-distribution).
    """
    ticker = (context or {}).get("ticker") if isinstance(context, dict) else None

    if ticker and ticker not in _CRYPTO_TICKERS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {"cryptotrader_lm": "HOLD"},
            "indicators": {
                "feature_unavailable": True,
                "reason": "non_crypto_ticker",
                "ticker": ticker,
            },
        }

    if _FEATURE_AVAILABLE:
        raise NotImplementedError(
            "cryptotrader_lm inference not yet implemented — scaffold only. "
            "See module docstring for the wiring contract."
        )

    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {"cryptotrader_lm": "HOLD"},
        "indicators": {
            "feature_unavailable": True,
            "reason": "scaffold",
            "model_dir": _MODEL_DIR,
        },
    }
