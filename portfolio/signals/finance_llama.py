"""Finance-Llama-8B signal — real inference (2026-05-16, was scaffold 2026-05-15).

Routes inference through the shared `llama_server` rotation (port 8787)
so it competes for GPU residency with `ministral3`, `qwen3`, and
`ministral8_lora`. Path metadata is already in `llama_server._MODEL_CONFIGS`
under key ``finance-llama-8b`` (registered 2026-04-09 as part of the
FinGPT → llm_batch migration); this signal is the second consumer of
that entry alongside `portfolio.sentiment` fingpt fan-out.

Cycle cost target: 3-4 s on GPU when the model is already resident,
8-12 s including a swap-in. The cycle_modulo=3 entry in
``data/shadow_registry.json`` keeps the loop budget bounded — at most
one run every 3 minutes of UTC wall clock.

Prompt format
-------------
finance-llama-8b is a COMPLETION model (Llama-3.1-based, not Mistral
instruction-tuned). The first revision of this file reused
``ministral_trader._build_prompt`` which emits Mistral ``[INST]…[/INST]``
markers. That format is off-distribution for this model and would
produce echoed prompts / nonsense — the same root-cause as the
2026-04-09 FinGPT incident, see
``Q:/models/fingpt_infer.py:PROMPT_TEMPLATES`` for the historical fix.

We instead emit a few-shot plain-text trading-decision template
(``Situation: … Decision: … Confidence: …``) that gives the model a
clear continuation pattern. Stop tokens are ``["\\n\\n", "Situation:"]``
so generation halts after the first Decision/Confidence pair.

Parser
------
``ministral_trader._parse_response`` is still reused — its regex
fallbacks (``\\b(BUY|SELL|HOLD)\\b`` and ``"?confidence"?\\s*:\\s*…``)
match plain text and quoted JSON alike. So switching prompt format
doesn't require a separate parser.

Failure modes return the canonical HOLD/conf=0 abstention shape so the
voter falls under `_weighted_consensus` HOLD-filtering — never drives a
trade decision when inference is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("portfolio.signals.finance_llama")

# Path is informational only — actual inference goes through llama_server,
# which holds the canonical path in _MODEL_CONFIGS["finance-llama-8b"].
# Keep this constant for the test that asserts the model exists on disk.
_MODEL_PATH = r"Q:\models\finance-llama-8b-gguf\wiroai-finance-llama-8b-q4_k_m.gguf"

# 2026-05-16: flipped to True after wiring real inference. Old comment at
# top of file describes the contract that's now implemented; flag stays
# around for parity with other scaffold signals (cryptotrader_lm,
# meta_trader) which still need their flips.
_FEATURE_AVAILABLE = True


# 2026-05-16: finance-llama-8b few-shot completion template. Mirrors the
# pattern used in Q:/models/fingpt_infer.py:PROMPT_TEMPLATES for the same
# model. Three examples cover BUY/SELL/HOLD so the model has a clear
# label distribution; the stop tokens cut generation after the first
# Decision/Confidence pair so the model can't drift into making up more
# situations (the 2026-04-09 fingpt failure mode).
_PROMPT_TEMPLATE = (
    "Classify a trading decision for each financial situation as BUY, SELL, or HOLD "
    "with a confidence score (0-100).\n\n"
    "Situation: RSI=22, oversold; MACD turning up; volume 1.8x avg; bullish reversal setup\n"
    "decision: BUY\n"
    "confidence: 75\n\n"
    "Situation: RSI=82, overbought; MACD turning down; volume 2.1x avg; exhaustion top\n"
    "decision: SELL\n"
    "confidence: 70\n\n"
    "Situation: RSI=55, neutral; MACD flat; volume 0.8x avg; no clear edge\n"
    "decision: HOLD\n"
    "confidence: 65\n\n"
    "Situation: {situation}\n"
    "decision:"
)
# Labels are lowercased (decision:/confidence:) because the shared
# ministral_trader._parse_response regex matches the literal lowercase
# 'confidence' substring — a capitalized 'Confidence:' label in the
# generated text would slip past the regex and trigger the 0.50
# fallback path, suppressing real per-call certainty.

_STOP_TOKENS = ["\n\n", "Situation:"]


def _build_finance_llama_prompt(context: dict) -> str:
    """Render the few-shot completion prompt for finance-llama-8b.

    `context` keys mirror the ministral context but we only use a subset
    here because the few-shot examples are short — the model has to do
    the heavy lifting from a compact one-line situation, not a long
    indicator dump. Long contexts diluted the few-shot pattern in
    2026-04-09 prompt-engineering experiments.
    """
    ticker = context.get("ticker", "UNKNOWN")
    rsi = context.get("rsi", "?")
    macd = context.get("macd_hist", "?")
    ema_dir = "up" if context.get("ema_bullish") else "down"
    ema_gap = context.get("ema_gap_pct", "?")
    bb = context.get("bb_position", "?")
    vol = context.get("volume_ratio", "?")
    fg = context.get("fear_greed", "?")
    sentiment = context.get("news_sentiment", "?")
    change = context.get("change_24h", "?")
    situation = (
        f"{ticker} 24h {change}; RSI={rsi}; MACD hist={macd}; "
        f"EMA9 vs EMA21 {ema_dir} ({ema_gap}% gap); BB position {bb}; "
        f"volume {vol}x avg; Fear&Greed={fg}; news sentiment {sentiment}"
    )
    return _PROMPT_TEMPLATE.format(situation=situation)


def _abstain(reason: str, extra: dict | None = None) -> dict:
    """Return the canonical HOLD/conf=0 shape with a reason string.

    Centralised so every error path produces the same shape — important
    for the validator at signal_engine._validate_signal_result and for
    test parity with the prior scaffold behaviour.
    """
    indicators = {"feature_unavailable": True, "reason": reason, "model_path": _MODEL_PATH}
    if extra:
        indicators.update(extra)
    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {"finance_llama": "HOLD"},
        "indicators": indicators,
    }


def compute_finance_llama_signal(df, context=None):
    """Run finance-llama-8b inference against the supplied context.

    df: pandas.DataFrame with OHLCV — currently unused (the Ministral
        prompt template already summarises the relevant indicators from
        the context dict; we'd add df-derived features here if a future
        prompt revision needed them).

    context: dict from signal_engine. Required keys: ticker, price_usd.
        Optional keys mirror the ministral context (rsi, macd_hist,
        ema_bullish, ema_gap_pct, bb_position, volume_ratio, fear_greed,
        fear_greed_class, news_sentiment, sentiment_confidence,
        timeframe_summary, headlines, change_24h, asset_type).

    Returns the standard enhanced-signal dict consumed by
    `signal_engine._validate_signal_result`.
    """
    if not _FEATURE_AVAILABLE:
        return _abstain("scaffold")

    if not isinstance(context, dict) or not context.get("ticker"):
        return _abstain("missing_context")

    # Lazy imports — keeping module import cheap matters because
    # signal_registry imports every signal module at startup.
    try:
        from portfolio.llama_server import model_load_safe, query_llama_server
        from portfolio.ministral_trader import _parse_response
    except ImportError as e:
        logger.warning("finance_llama dependencies missing: %s", e)
        return _abstain("dependency_unavailable", {"error": str(e)})

    # 2026-05-11 Plex-VRAM-coord pattern (copied from ministral_signal._call_model):
    # when VRAM is tight because Plex is transcoding, skip rather than
    # forcing a cold-swap that would evict Plex's NVENC context. Treat
    # the abstention as a no-vote rather than a real HOLD prediction.
    try:
        if not model_load_safe():
            logger.warning(
                "finance_llama: abstaining — Plex transcoding and VRAM "
                "<7168MB; skipping inference",
            )
            return _abstain("plex_vram_tight")
    except Exception:
        # model_load_safe is informational; never let it abort inference.
        logger.debug("model_load_safe check failed", exc_info=True)

    try:
        prompt = _build_finance_llama_prompt(context)
    except (KeyError, ValueError) as e:
        # Builder is permissive (defaults to "?" for missing fields) so
        # a crash here would mean something genuinely malformed slipped
        # past the missing_context guard above; surface as abstention.
        logger.warning("finance_llama prompt build failed: %s", e)
        return _abstain("prompt_build_failed", {"error": str(e)})

    text = None
    try:
        text = query_llama_server("finance-llama-8b", prompt, stop=_STOP_TOKENS)
    except Exception as e:
        logger.warning("finance_llama query failed: %s", e)
        return _abstain("inference_error", {"error": str(e)})

    if not text:
        # query_llama_server returns None when the server is unreachable
        # or the model swap timed out. Same abstention semantics as the
        # ministral_signal fallback path.
        return _abstain("server_unavailable")

    decision, reasoning, confidence = _parse_response(text)
    if confidence is None:
        # Parser couldn't recover a numeric confidence even from the
        # regex fallback. Surface as low-confidence vote at 0.50 — the
        # parser already produced a decision via the action regex, so
        # we trust the argmax but cap the confidence at the threshold
        # below which downstream gates would treat it as HOLD anyway.
        # Using 0.50 (not 0.0) preserves the BUY/SELL argmax in
        # derive_probs_from_result — the conf<=0 branch would force a
        # 50/25/25 shape that masks the model's actual verdict.
        confidence = 0.50

    return {
        "action": decision,
        "confidence": float(confidence),
        "sub_signals": {"finance_llama": decision},
        "indicators": {
            "model": "finance-llama-8b",
            "model_path": _MODEL_PATH,
            "raw_confidence": confidence,
            "reasoning": (reasoning or "")[:200],
        },
    }
