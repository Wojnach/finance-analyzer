"""Phi-4-mini-reasoning shadow signal — SHADOW ONLY, force-HOLD in consensus.

Routes inference through the shared ``llama_server`` rotation (port 8787)
under slot ``phi4_mini``. The model is Microsoft Phi-4-mini-reasoning (3.8B),
a reasoning model that emits a ``<think>…</think>`` block before its final
answer. The parser strips the think block and extracts BUY/SELL/HOLD +
confidence from the structured decision line.

Added 2026-06-01 as a shadow challenger to the existing 8B LLM voters
(ministral3, qwen3). Phi-4-mini has a smaller VRAM footprint (~2.5 GB
Q4_K_M) which may allow concurrent residency with other models during
metals-loop cycles. Registered in ``data/shadow_registry.json`` with
``cycle_modulo=3`` to bound loop budget.

Prompt format
-------------
Phi-4-mini-reasoning uses the Phi-4 chat template with special tokens:
  <|system|>\\n{system}<|end|>\\n<|user|>\\n{user}<|end|>\\n<|assistant|>\\n

Unlike Ministral (completion model) and Qwen3 (im_start/im_end), Phi-4
uses ``<|system|>`` / ``<|user|>`` / ``<|assistant|>`` with ``<|end|>``
terminators (verified from Microsoft Phi-4 model card and GGUF metadata
2026-06-01). The model emits a reasoning trace inside ``<think>…</think>``
before its answer — we need max_tokens headroom for the think block.

Parser
------
The prompt instructs the model to end with:
  decision: BUY|SELL|HOLD
  confidence: <0-100>

``_parse_phi4_response`` strips the think block first, then applies
the same regex fallbacks as ``ministral_trader._parse_response`` so
shared infrastructure can recover both action and confidence from the
post-think text even if the model's formatting drifts slightly.

Force-HOLD
----------
``phi4_mini`` is listed in ``portfolio.tickers.DISABLED_SIGNALS`` so it
is NEVER included in the active voting consensus. It runs as a shadow
signal: predictions go into ``_shadow_votes`` for accuracy tracking via
``outcome_tracker``, and each inference is logged to
``data/llm_probability_log.jsonl`` by ``llm_probability_log.log_vote``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("portfolio.signals.phi4_mini_reasoning")

# Windows path — same platform-conditional idiom as other _MODEL_CONFIGS entries.
# Informational only; canonical path is in llama_server._MODEL_CONFIGS["phi4_mini"].
_MODEL_PATH = (
    r"Q:\models\phi4-mini-reasoning-gguf\microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf"
)

# 2026-06-01: set True — model file confirmed on disk at
# /mnt/q/models/phi4-mini-reasoning-gguf/microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf
# (2,491,874,848 bytes). Inference wired via phi4_mini slot in llama_server.
_FEATURE_AVAILABLE = True

# Phi-4 chat template (verified from Microsoft Phi-4 model card 2026-06-01).
# Uses <|system|>/<|user|>/<|assistant|> with <|end|> terminators — distinct
# from Qwen3's <|im_start|>/<|im_end|> and Ministral's [INST]/[/INST].
_SYSTEM_PROMPT = (
    "You are an expert financial analyst. Analyze the trading situation and "
    "make a precise BUY, SELL, or HOLD decision. Think step by step through "
    "the evidence. After your reasoning, end your response with exactly:\n"
    "decision: BUY|SELL|HOLD\n"
    "confidence: <number 0-100>"
)

_PROMPT_TEMPLATE = (
    "<|system|>\n{system}<|end|>\n"
    "<|user|>\n"
    "Asset: {ticker} ({asset_type})\n"
    "Price: ${price:,.2f}\n"
    "RSI(14): {rsi}\n"
    "MACD Histogram: {macd_hist}\n"
    "EMA(9) vs EMA(21): {ema_dir} (gap: {ema_gap}%)\n"
    "Bollinger Bands: Price is {bb_pos}\n"
    "Volume Ratio: {vol_ratio}x avg\n"
    "Fear & Greed: {fg}/100 ({fg_class})\n"
    "News Sentiment: {sentiment}\n"
    "Multi-timeframe: {tf_summary}\n"
    "24h Change: {change_24h}\n\n"
    "Analyze carefully, then provide your decision and confidence score."
    "<|end|>\n"
    "<|assistant|>\n"
)

# 2026-06-01: reasoning models need room for the <think> block.
# 512 tokens: ~400 for think trace + ~50 for the decision line. Cap keeps
# cycle budget bounded — we only need the final decision, not a full essay.
_MAX_TOKENS = 512

# Stop tokens: <|end|> is Phi-4's turn terminator; double-newline cuts
# runaway generation if the model ignores the template terminator.
_STOP_TOKENS = ["<|end|>", "<|endoftext|>"]


def _build_phi4_prompt(context: dict) -> str:
    """Render the Phi-4 chat-format prompt from signal context."""
    ticker = context.get("ticker", "UNKNOWN")
    asset_type = context.get("asset_type", "cryptocurrency")
    price = float(context.get("price_usd", 0) or 0)
    rsi = context.get("rsi", "N/A")
    macd_hist = context.get("macd_hist", "N/A")
    ema_dir = "Bullish (9>21)" if context.get("ema_bullish") else "Bearish (9<21)"
    ema_gap = context.get("ema_gap_pct", "N/A")
    bb_pos = context.get("bb_position", "N/A")
    vol_ratio = context.get("volume_ratio", "N/A")
    fg = context.get("fear_greed", "N/A")
    fg_class = context.get("fear_greed_class", "")
    sentiment = context.get("news_sentiment", "N/A")
    tf_summary = context.get("timeframe_summary", "N/A")
    change_24h = context.get("change_24h", "N/A")
    return _PROMPT_TEMPLATE.format(
        system=_SYSTEM_PROMPT,
        ticker=ticker,
        asset_type=asset_type,
        price=price,
        rsi=rsi,
        macd_hist=macd_hist,
        ema_dir=ema_dir,
        ema_gap=ema_gap,
        bb_pos=bb_pos,
        vol_ratio=vol_ratio,
        fg=fg,
        fg_class=fg_class,
        sentiment=sentiment,
        tf_summary=tf_summary,
        change_24h=change_24h,
    )


def _strip_think_block(text: str) -> str:
    """Remove the <think>…</think> reasoning block emitted by Phi-4-mini.

    Phi-4-mini-reasoning (and other reasoning models) emit a chain-of-thought
    block enclosed in <think> tags before the actual answer. We only need
    the post-think text for BUY/SELL/HOLD + confidence extraction.

    If there is no think block the original text is returned unchanged so
    non-reasoning fallback outputs are parsed identically.
    """
    # 2026-06-01: DOTALL so the think block can span multiple lines.
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return stripped.strip()


def _parse_phi4_response(text: str):
    """Parse post-think Phi-4 output into (action, reasoning, confidence).

    Strategy (mirrors ministral_trader._parse_response):
    1. Try to match the structured ``decision:`` / ``confidence:`` lines
       that the prompt requests.
    2. Fall back to ``\\b(BUY|SELL|HOLD)\\b`` regex for action.
    3. Fall back to ``"?confidence"?\\s*:\\s*<number>`` regex for confidence.
    4. If confidence still None, caller defaults to 0.50 (preserves argmax).

    Returns (action: str, reasoning: str, confidence: float | None).
    """
    post_think = _strip_think_block(text)

    # 1a. Structured decision line: "decision: BUY" (case-insensitive)
    action = None
    dm = re.search(r"decision\s*:\s*(BUY|SELL|HOLD)", post_think, re.IGNORECASE)
    if dm:
        action = dm.group(1).upper()

    # 1b. Fallback: bare BUY/SELL/HOLD word anywhere in post-think text
    if action is None:
        am = re.search(r"\b(BUY|SELL|HOLD)\b", post_think.upper())
        action = am.group(1) if am else "HOLD"

    # 2. Confidence: structured line first, then regex fallback
    confidence = None
    cm = re.search(r"confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)", post_think, re.IGNORECASE)
    if cm:
        try:
            raw = float(cm.group(1))
            # Normalise 0-100 → 0-1; preserve already-normalised 0-1 values.
            if raw > 1.0:
                raw = raw / 100.0
            confidence = max(0.0, min(1.0, raw))
        except (ValueError, TypeError):
            pass

    # Reasoning: everything in post_think up to the decision line (truncated).
    reasoning = post_think[:200]

    return action, reasoning, confidence


def _abstain(reason: str, extra: dict | None = None) -> dict:
    """Canonical HOLD/conf=0 abstention shape.

    Every error/guard path returns this so signal_engine._validate_signal_result
    and llm_probability_log.log_vote both see a consistent shape.
    """
    indicators: dict = {
        "feature_unavailable": True,
        "reason": reason,
        "model_path": _MODEL_PATH,
    }
    if extra:
        indicators.update(extra)
    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {"phi4_mini": "HOLD"},
        "indicators": indicators,
    }


def compute_phi4_mini_signal(df, context=None):
    """Run Phi-4-mini-reasoning inference against the supplied context.

    df: pandas.DataFrame with OHLCV — not used directly; indicators are
        summarised in context by signal_engine before dispatch.

    context: dict from signal_engine. Required: ticker, price_usd.
        Optional: rsi, macd_hist, ema_bullish, ema_gap_pct, bb_position,
        volume_ratio, fear_greed, fear_greed_class, news_sentiment,
        timeframe_summary, change_24h, asset_type.

    Returns the standard enhanced-signal dict consumed by
    signal_engine._validate_signal_result. This signal is registered in
    DISABLED_SIGNALS and shadow_registry.json — it NEVER drives a trade.
    """
    if not _FEATURE_AVAILABLE:
        return _abstain("scaffold")

    if not isinstance(context, dict) or not context.get("ticker"):
        return _abstain("missing_context")

    # Lazy imports — cheap startup matters because signal_registry imports
    # every module at boot time.
    try:
        from portfolio.llama_server import model_load_safe, query_llama_server
    except ImportError as e:
        logger.warning("phi4_mini dependencies missing: %s", e)
        return _abstain("dependency_unavailable", {"error": str(e)})

    # 2026-06-01: Plex-VRAM guard (same pattern as finance_llama / ministral_signal).
    # Skip if Plex is transcoding and VRAM is below 7168 MB — a cold-swap of
    # Phi-4-mini would evict Plex's NVENC context. Treat as no-vote, not HOLD.
    try:
        if not model_load_safe():
            logger.warning(
                "phi4_mini: abstaining — Plex transcoding and VRAM <7168MB; "
                "skipping inference",
            )
            return _abstain("plex_vram_tight")
    except Exception:
        logger.debug("model_load_safe check failed", exc_info=True)

    try:
        prompt = _build_phi4_prompt(context)
    except (KeyError, ValueError) as e:
        logger.warning("phi4_mini prompt build failed: %s", e)
        return _abstain("prompt_build_failed", {"error": str(e)})

    text = None
    try:
        # 2026-06-01: llama_server.query_llama_server uses n_predict (not
        # max_tokens) — matches the llama-server /completion API parameter name.
        # 512 tokens gives the <think> block enough room without blowing cycle time.
        text = query_llama_server(
            "phi4_mini",
            prompt,
            stop=_STOP_TOKENS,
            n_predict=_MAX_TOKENS,
        )
    except Exception as e:
        logger.warning("phi4_mini query failed: %s", e)
        return _abstain("inference_error", {"error": str(e)})

    if not text:
        return _abstain("server_unavailable")

    action, reasoning, confidence = _parse_phi4_response(text)

    if confidence is None:
        # Parser found action but no numeric confidence. Default to 0.50 so
        # derive_probs_from_result preserves the BUY/SELL argmax rather than
        # collapsing to the uniform-ish conf<=0 shape. Same rationale as
        # finance_llama (2026-05-16 fix).
        confidence = 0.50

    return {
        "action": action,
        "confidence": float(confidence),
        "sub_signals": {"phi4_mini": action},
        "indicators": {
            "model": "phi4-mini-reasoning",
            "model_path": _MODEL_PATH,
            "raw_confidence": confidence,
            "reasoning": reasoning[:200],
        },
    }
