"""CryptoTrader-LM signal — real PEFT LoRA inference (2026-05-17, was scaffold 2026-05-15).

Routes inference through the shared `llama_server` rotation under model
name ``ministral8_lora`` — that entry already loads Ministral-8B base
with `--lora cryptotrader-lm-lora.gguf` (PEFT LoRA adapter) via
`extra_args` in `portfolio.llama_server._MODEL_CONFIGS`.

Model lineage
-------------
* Base: Ministral-8B-Instruct-2410 (mistralai/Ministral-8B-Instruct-2410)
* Adapter: agarkovv/CryptoTrader-LM PEFT LoRA, FinNLP @ COLING-2025
* Published claim: 72% directional accuracy, 0.94 Sharpe on BTC/ETH
  validation set
* Promotion bar in `data/shadow_registry.json`: ``min_accuracy: 0.60``
  (higher than the 0.55 default) so we hold the model to the paper claim

Prompt format
-------------
Base is Mistral-instruct so we reuse ``ministral_trader._build_prompt``
verbatim — same ``[INST]…[/INST]`` template, same indicator dump, same
JSON shape request. Unlike `finance_llama` (Llama-3 completion model
requiring few-shot plain-text), the LoRA was fine-tuned on the base
chat template, so deviating from the Mistral format would be off-
distribution.

Crypto-only refusal
-------------------
LoRA was trained only on BTC/ETH; voting on metals or MSTR would be
out-of-distribution. The ticker guard at the top of the compute fn
runs BEFORE the `_FEATURE_AVAILABLE` check so non-crypto tickers
return a no-vote permanently, regardless of inference state.

Failure modes
-------------
Same canonical abstention shape (HOLD/conf=0) with reason strings used
by `finance_llama`: ``missing_context``, ``server_unavailable``,
``inference_error``, ``plex_vram_tight``, ``dependency_unavailable``,
``prompt_build_failed``, ``non_crypto_ticker``, ``scaffold``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("portfolio.signals.cryptotrader_lm")

_MODEL_DIR = r"Q:\models\cryptotrader-lm"
_CRYPTO_TICKERS = frozenset({"BTC-USD", "ETH-USD"})
_LLAMA_SERVER_MODEL = "ministral8_lora"  # _MODEL_CONFIGS key with LoRA extra-arg
_STOP_TOKENS = ["[INST]"]

# 2026-05-17: flipped to True after wiring real inference. Flag stays
# around for parity with other scaffold signals (meta_trader) which
# still need their flips.
_FEATURE_AVAILABLE = True


def _abstain(reason: str, extra: dict | None = None) -> dict:
    """Return the canonical HOLD/conf=0 shape with a reason string.

    Same helper pattern as `finance_llama._abstain`. Centralised so every
    error path produces the same shape — important for the validator at
    `signal_engine._validate_signal_result` and for downstream calibration
    parity.
    """
    indicators = {
        "feature_unavailable": True,
        "reason": reason,
        "model_dir": _MODEL_DIR,
    }
    if extra:
        indicators.update(extra)
    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {"cryptotrader_lm": "HOLD"},
        "indicators": indicators,
    }


def compute_cryptotrader_lm_signal(df, context=None):
    """Run cryptotrader_lm inference against the supplied context.

    df: pandas.DataFrame with OHLCV — unused (the Ministral prompt builder
        summarises indicators from the context dict).
    context: dict from signal_engine. Required keys: ticker, price_usd.
        Optional: rsi, macd_hist, ema_bullish, ema_gap_pct, bb_position,
        volume_ratio, fear_greed, fear_greed_class, news_sentiment,
        sentiment_confidence, timeframe_summary, headlines, change_24h.

    Returns the standard enhanced-signal dict.

    The crypto-only ticker guard runs BEFORE the `_FEATURE_AVAILABLE`
    check so a non-crypto ticker always returns ``non_crypto_ticker``
    regardless of whether real inference is wired.
    """
    ticker = (context or {}).get("ticker") if isinstance(context, dict) else None

    if ticker and ticker not in _CRYPTO_TICKERS:
        return _abstain("non_crypto_ticker", {"ticker": ticker})

    if not _FEATURE_AVAILABLE:
        return _abstain("scaffold")

    if not isinstance(context, dict) or not context.get("ticker"):
        return _abstain("missing_context")

    # Lazy imports — keep module import cheap because signal_registry
    # imports every signal module at startup.
    try:
        from portfolio.llama_server import model_load_safe, query_llama_server
        from portfolio.ministral_trader import _build_prompt, _parse_response
    except ImportError as e:
        logger.warning("cryptotrader_lm dependencies missing: %s", e)
        return _abstain("dependency_unavailable", {"error": str(e)})

    # 2026-05-11 Plex-VRAM-coord pattern (copied from ministral_signal._call_model
    # and finance_llama): when VRAM is tight because Plex is transcoding, skip
    # rather than forcing a cold-swap that evicts Plex's NVENC context.
    try:
        if not model_load_safe():
            logger.warning(
                "cryptotrader_lm: abstaining — Plex transcoding and VRAM "
                "<7168MB; skipping inference",
            )
            return _abstain("plex_vram_tight")
    except Exception:
        # model_load_safe is informational; never let it abort inference.
        logger.debug("model_load_safe check failed", exc_info=True)

    try:
        prompt = _build_prompt(context)
    except (KeyError, ValueError) as e:
        logger.warning("cryptotrader_lm prompt build failed: %s", e)
        return _abstain("prompt_build_failed", {"error": str(e)})

    text = None
    try:
        text = query_llama_server(_LLAMA_SERVER_MODEL, prompt, stop=_STOP_TOKENS)
    except Exception as e:
        logger.warning("cryptotrader_lm query failed: %s", e)
        return _abstain("inference_error", {"error": str(e)})

    if not text:
        # query_llama_server returns None when the server is unreachable
        # or the model swap timed out. Same abstention semantics as
        # ministral_signal and finance_llama.
        return _abstain("server_unavailable")

    decision, reasoning, confidence = _parse_response(text)
    if confidence is None:
        # Default to 0.50 (NOT 0.0) so the BUY/SELL argmax survives
        # derive_probs_from_result downstream. The conf<=0 branch in
        # llm_probability_log forces {action: 0.5, others: 0.25}; 0.50
        # gives the same shape but preserves the parser's intent that
        # an action WAS recovered, just without a quantified certainty.
        # Same fallback pattern as finance_llama.
        confidence = 0.50

    return {
        "action": decision,
        "confidence": float(confidence),
        "sub_signals": {"cryptotrader_lm": decision},
        "indicators": {
            "model": "cryptotrader-lm (Ministral-8B + PEFT LoRA)",
            "model_dir": _MODEL_DIR,
            "raw_confidence": confidence,
            "reasoning": (reasoning or "")[:200],
            "ticker": ticker,
        },
    }
