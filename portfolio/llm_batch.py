"""Batch queue for LLM signals — eliminates model swap overhead.

During parallel ticker processing, expired LLM cache entries are enqueued
instead of triggering immediate model loads. After all tickers finish,
flush_llm_batch() processes them grouped by model: load Ministral once →
query all tickers → swap to Qwen3 once → query all tickers → swap to fingpt
once → run all sentiment prompts.

Result: max 2 model swaps per cycle instead of N swaps.

Uses query_llama_server_batch() which holds the file lock for the entire
model phase, preventing the metals loop from swapping mid-batch (Codex #4).

2026-04-09 update (feat/fingpt-in-llmbatch): added Phase 3 for fingpt
sentiment. Previously fingpt ran in its own bespoke NDJSON daemon
(scripts/fingpt_daemon.py) on CPU (~60-150s/cycle). Moving it into this
batched rotation lets it use full GPU offload (-ngl 99) in its own
llama-server phase, trading one extra swap (~10-25s) for a ~70-120s
reduction in fingpt inference time. See project_fingpt_llmbatch_session
memory entry for the full design rationale.
"""

import logging
import threading
import time

logger = logging.getLogger("portfolio.llm_batch")

_lock = threading.Lock()
_ministral_queue: list[tuple[str, dict]] = []   # (cache_key, context)
_qwen3_queue: list[tuple[str, dict]] = []       # (cache_key, context)
# _fingpt_queue entries are (ab_key, sub_key, context) — sub_key is
# "headlines" for per-headline inference or "cumul:<N>" for a cumulative
# cluster. The ab_key is shared by all fingpt calls for a single ticker's
# get_sentiment() invocation so the results can be stitched back into one
# sentiment_ab_log.jsonl entry by sentiment.flush_ab_log() post-cycle.
_fingpt_queue: list[tuple[str, str, dict]] = []


# 2026-04-10 (perf/llama-swap-reduction) — ROTATION SCHEDULING
#
# Rotation across the three llama-server LLMs reduces the LLM batch phase
# from running all 3 models every cycle (~85 s: 40 s Ministral + 19 s Qwen3
# + 9 s fingpt + 15-18 s of swaps) to running ONE model per cycle (~25-40 s
# depending on which one). Each LLM still gets a fresh vote every 3rd full-
# LLM batch, and _cached_or_enqueue returns stale data on the off-cycle 2
# of 3 cycles (max staleness is bounded by max_stale_factor=5 passed at the
# call site in signal_engine.py / sentiment.py).
#
# Design decisions (see docs/PLAN.md / plan file for full rationale):
#
# 1. Counter lives in shared_state._full_llm_cycle_count, increments AFTER
#    flush when the batch actually had work. In-memory only — restart resets
#    to 0 and triggers a warmup cycle that runs all three models to establish
#    a baseline before the rotation begins.
#
# 2. Rotation gate sits at the _cached_or_enqueue caller via should_enqueue_fn,
#    NOT inside enqueue_ministral/qwen3 themselves, because the enqueue helpers
#    also need to be callable directly (from sentiment.py for fingpt) without
#    going through _cached_or_enqueue. Gating inside the enqueue functions
#    would also poison _loading_keys when the rotation skips.
#
# 3. Counter advances once per flush-with-work, not once per loop iteration.
#    That way rotation is driven by actual LLM invocations, not by idle cache-
#    hit cycles where nothing needs to run.
#
# 4. Warmup: on the very first flush after process start (counter == 0), ALL
#    LLMs run so we have a full baseline before rotation kicks in. Subsequent
#    flushes rotate.
_LLM_ROTATION: tuple[str, ...] = ("ministral", "qwen3", "fingpt")


def is_llm_on_cycle(llm_name: str) -> bool:
    """Return True if `llm_name` is scheduled to run during the current cycle.

    Called at enqueue time to decide whether to skip the enqueue. The current
    cycle's slot is `(shared_state._full_llm_cycle_count - 1) % 3` because
    the counter advances AFTER the flush — at enqueue time, the counter
    represents "how many flushes have already completed" and the next slot
    is `counter % 3`, but we want to treat "counter == 0" as a warmup in
    which everything runs. So:

        counter == 0  → warmup → every LLM returns True
        counter == 1  → slot 0 → ministral only
        counter == 2  → slot 1 → qwen3 only
        counter == 3  → slot 2 → fingpt only
        counter == 4  → slot 0 → ministral again
        ...

    Unknown llm_name raises ValueError (from tuple.index) — that's a
    programming error we want to catch in tests rather than silently
    return False.
    """
    from portfolio import shared_state as _ss
    count = _ss._full_llm_cycle_count
    if count == 0:
        return True  # warmup — run everything the first time through
    idx = _LLM_ROTATION.index(llm_name)  # raises ValueError for bad names
    slot = (count - 1) % len(_LLM_ROTATION)
    return slot == idx


def enqueue_ministral(cache_key, context):
    """Add a Ministral cache miss to the batch queue."""
    with _lock:
        if not any(k == cache_key for k, _ in _ministral_queue):
            _ministral_queue.append((cache_key, context))


def enqueue_qwen3(cache_key, context):
    """Add a Qwen3 cache miss to the batch queue."""
    with _lock:
        if not any(k == cache_key for k, _ in _qwen3_queue):
            _qwen3_queue.append((cache_key, context))


def enqueue_fingpt(ab_key: str, sub_key: str, context: dict) -> None:
    """Add a fingpt sentiment request to the batch queue.

    Args:
        ab_key: Shared key identifying the parent get_sentiment() call
            (e.g. "BTC:2026-04-09T18:04:00+00:00"). All fingpt calls for
            the same get_sentiment() invocation share this key so their
            results can be merged into one A/B log entry.
        sub_key: "headlines" for per-headline inference, or "cumul:<N>"
            for the N-th cumulative cluster.
        context: {"mode": "headlines"|"cumulative", "texts": [...],
                  "ticker": "BTC"}
    """
    with _lock:
        # Deduplicate on (ab_key, sub_key). Unlike ministral/qwen3 we use a
        # composite key because one ticker may enqueue both headlines and
        # multiple cumulative clusters in the same get_sentiment() call.
        if not any(k == ab_key and s == sub_key for k, s, _ in _fingpt_queue):
            _fingpt_queue.append((ab_key, sub_key, context))


def _flush_via_server(model_name, batch, build_prompt_fn, parse_response_fn, stop_tokens):
    """Flush a batch using query_llama_server_batch (atomic, lock held for entire phase)."""
    try:
        from portfolio.llama_server import query_llama_server_batch
    except ImportError:
        return {}

    prompts_and_params = []
    for _cache_key, ctx in batch:
        prompt = build_prompt_fn(ctx)
        prompts_and_params.append({
            "prompt": prompt,
            "stop": stop_tokens,
        })

    texts = query_llama_server_batch(model_name, prompts_and_params)

    results = {}
    for (cache_key, _ctx), text in zip(batch, texts):
        if text is not None:
            parsed = parse_response_fn(text)
            if parsed:
                results[cache_key] = parsed
    return results


def flush_llm_batch():
    """Process all queued LLM requests, batched by model.

    Called once after ThreadPoolExecutor completes in main.py.
    Returns dict of {cache_key: result} for cache updates — ministral and
    qwen3 results only. Fingpt results are stashed into sentiment._pending_ab_entries
    via _stash_fingpt_result() and emitted later by sentiment.flush_ab_log().

    2026-04-10 (perf/llama-swap-reduction): after processing, advances the
    rotation counter in shared_state._full_llm_cycle_count if at least one
    phase had queued work. This is what makes is_llm_on_cycle() rotate through
    ministral → qwen3 → fingpt across successive flushes.
    """
    with _lock:
        m_batch = list(_ministral_queue)
        q_batch = list(_qwen3_queue)
        f_batch = list(_fingpt_queue)
        _ministral_queue.clear()
        _qwen3_queue.clear()
        _fingpt_queue.clear()

    if not m_batch and not q_batch and not f_batch:
        return {}

    # Log which LLMs actually ran this cycle vs which were rotation-gated
    # out at the call site. Useful for debugging rotation behaviour in logs.
    from portfolio import shared_state as _ss
    rotation_slot = (
        "warmup" if _ss._full_llm_cycle_count == 0
        else _LLM_ROTATION[(_ss._full_llm_cycle_count - 1) % len(_LLM_ROTATION)]
    )
    logger.info(
        "LLM batch start: rotation_slot=%s counter=%d queues M=%d Q=%d F=%d",
        rotation_slot, _ss._full_llm_cycle_count, len(m_batch), len(q_batch), len(f_batch),
    )

    results = {}
    t0 = time.monotonic()

    # Phase 1: All Ministral queries (lock held for entire phase)
    if m_batch:
        logger.info("LLM batch: %d Ministral queries", len(m_batch))
        try:
            from portfolio.ministral_trader import _build_prompt, _parse_response

            def _parse_ministral(text):
                decision, reasoning, confidence = _parse_response(text)
                result = {
                    "original": {"action": decision, "reasoning": reasoning, "model": "Ministral-3-8B"},
                    "custom": None,
                }
                if confidence is not None:
                    result["original"]["confidence"] = confidence
                return result

            phase = _flush_via_server("ministral3", m_batch, _build_prompt, _parse_ministral, ["[INST]"])
            results.update(phase)
        except Exception as e:
            logger.warning("LLM batch Ministral failed: %s", e)

    # Phase 2: All Qwen3 queries (lock held for entire phase)
    if q_batch:
        logger.info("LLM batch: %d Qwen3 queries", len(q_batch))
        try:
            from portfolio.qwen3_trader import _build_prompt as _qwen_build
            from portfolio.qwen3_trader import _parse_response as _qwen_parse_raw

            def _parse_qwen3(text):
                decision, reasoning, confidence = _qwen_parse_raw(text)
                result = {"action": decision, "reasoning": reasoning, "model": "Qwen3-8B"}
                if confidence is not None:
                    result["confidence"] = confidence
                return result

            phase = _flush_via_server(
                "qwen3", q_batch, _qwen_build, _parse_qwen3,
                ["<|endoftext|>", "<|im_end|>"],
            )
            results.update(phase)
        except Exception as e:
            logger.warning("LLM batch Qwen3 failed: %s", e)

    # Phase 3: All fingpt sentiment queries (lock held for entire phase)
    # Added 2026-04-09 (feat/fingpt-in-llmbatch). Replaces the bespoke
    # scripts/fingpt_daemon.py NDJSON daemon with the shared llama_server
    # rotation, trading ~1 extra swap for a ~70-120s reduction in fingpt
    # inference time per cycle. Fingpt is a SHADOW sentiment signal (does
    # not vote) so its failures are log-only — primary sentiment (CryptoBERT
    # / Trading-Hero-LLM) is unaffected if this phase breaks.
    if f_batch:
        logger.info("LLM batch: %d fingpt queries", len(f_batch))
        _flush_fingpt_phase(f_batch)

    elapsed = time.monotonic() - t0
    logger.info("LLM batch: %d results in %.1fs (M:%d Q:%d F:%d)",
                len(results), elapsed, len(m_batch), len(q_batch), len(f_batch))

    # Advance rotation counter — next flush will target the next LLM in rotation.
    # Only bumped when at least one phase had work (we already returned early
    # for an empty flush above). Wrapping behaviour in is_llm_on_cycle handles
    # arbitrary large counters.
    _ss._full_llm_cycle_count += 1

    return results


def _flush_fingpt_phase(f_batch: list[tuple[str, str, dict]]) -> None:
    """Execute Phase 3: load finance-llama-8b once, run all queued sentiment
    prompts, stash results in sentiment._pending_ab_entries.

    Per-item failure (None text from the server) is logged as a tagged
    fingpt:error result — the A/B logger sees it and writes a zero-confidence
    entry instead of silently dropping the sample.

    The whole phase is wrapped in try/except so fingpt errors never leak out
    into the main loop. Shadow signals must not crash anything above them.
    """
    try:
        # fingpt_infer provides the prompt templates, stop tokens, and
        # response parsers that were originally used by the retired daemon.
        # Imported here (lazy) so a missing Q:\models path degrades gracefully
        # — if the import fails, fingpt just doesn't run this cycle.
        import platform
        import sys
        if platform.system() == "Windows":
            _models_dir = r"Q:\models"
        else:
            _models_dir = "/home/deck/models"
        if _models_dir not in sys.path:
            sys.path.insert(0, _models_dir)
        import fingpt_infer  # noqa: E402  (path injection above)

        # Flatten the batch into per-prompt requests and keep a parallel meta
        # list so we can group results back by (ab_key, sub_key) afterward.
        prompts_and_params: list[dict] = []
        meta: list[tuple[str, str, dict, int]] = []  # (ab_key, sub_key, ctx, prompt_idx_within_call)
        for ab_key, sub_key, ctx in f_batch:
            mode = ctx.get("mode", "headlines")
            texts = ctx.get("texts") or []
            if mode == "cumulative":
                headlines_block = "\n".join(f"- {h}" for h in texts[:20])
                prompt = fingpt_infer.CUMULATIVE_PROMPT.format(
                    count=len(texts),
                    headlines_block=headlines_block,
                )
                prompts_and_params.append({
                    "prompt": prompt,
                    "n_predict": 30,
                    "temperature": 0.1,
                    # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
                    # Old stop ["\n", "<|eot_id|>"] was designed for the Llama-3
                    # chat-format prompt that wiroai-finance-llama-8b doesn't
                    # recognize. New CUMULATIVE_PROMPT is a plain-text one-shot
                    # template that ends each section with a blank line, so
                    # "\n\n" is the natural stop. The <|eot_id|> token was never
                    # emitted by this model (it's not chat-tuned) so removing
                    # it is a no-op.
                    "stop": ["\n\n"],
                })
                meta.append((ab_key, sub_key, ctx, 0))
            else:
                # Headlines mode: one prompt per headline. The daemon used
                # PROMPT_TEMPLATES[name] for the loaded model; llama_server
                # loads finance-llama-8b so we index into that entry directly.
                template = fingpt_infer.PROMPT_TEMPLATES.get(
                    "finance-llama-8b",
                    next(iter(fingpt_infer.PROMPT_TEMPLATES.values())),
                )
                for i, headline in enumerate(texts):
                    prompts_and_params.append({
                        "prompt": template.format(headline=headline),
                        "n_predict": 20,
                        "temperature": 0.1,
                        # 2026-04-09 (fix/fingpt-parser-prompt): ["\n\n"] only.
                        # Same reason as the cumulative case above. The old
                        # stop ["\n", "<|eot_id|>", "[INST]"] cut the few-shot
                        # prompt apart at the first newline, which is exactly
                        # where the expected answer word appears — so even a
                        # correctly-answering model would have been silenced.
                        "stop": ["\n\n"],
                    })
                    meta.append((ab_key, sub_key, ctx, i))

        if not prompts_and_params:
            return

        # Single HTTP batch — llama_server holds its own file lock for the
        # duration so no other process can swap the model mid-phase.
        from portfolio.llama_server import query_llama_server_batch
        texts_out = query_llama_server_batch("finance-llama-8b", prompts_and_params)

        # Group results back by (ab_key, sub_key) → list of per-prompt parsed dicts.
        grouped: dict[tuple[str, str], list[tuple[int, dict | None, dict]]] = {}
        for (ab_key, sub_key, ctx, prompt_idx), text in zip(meta, texts_out):
            parsed = _parse_fingpt_completion(text, fingpt_infer)
            grouped.setdefault((ab_key, sub_key), []).append((prompt_idx, parsed, ctx))

        # Stash each (ab_key, sub_key) result into the sentiment buffer. The
        # buffer is consumed by sentiment.flush_ab_log() which runs right
        # after flush_llm_batch() in main.py and writes the final A/B log
        # entries.
        from portfolio.sentiment import _stash_fingpt_result
        for (ab_key, sub_key), items in grouped.items():
            items.sort(key=lambda t: t[0])
            mode = items[0][2].get("mode", "headlines")
            if mode == "cumulative":
                parsed = items[0][1]
                # Cumulative: daemon applied a +0.1 confidence boost when
                # len(headlines) >= 5. Replicate here so the A/B log shows
                # the same numbers it did pre-migration.
                if parsed is not None:
                    texts = items[0][2].get("texts") or []
                    if len(texts) >= 5:
                        parsed = dict(parsed)
                        parsed["confidence"] = min(
                            parsed.get("confidence", 0.0) + 0.1, 0.95,
                        )
                    parsed["headline_count"] = len(texts)
                    parsed["model"] = "fingpt:cumulative"
                _stash_fingpt_result(ab_key, sub_key, parsed)
            else:
                per_headline = [p for (_idx, p, _c) in items]
                _stash_fingpt_result(ab_key, sub_key, per_headline)
    except Exception:
        logger.warning("LLM batch fingpt phase failed", exc_info=True)


def _parse_fingpt_completion(text: str | None, fingpt_infer) -> dict | None:
    """Parse one llama-server completion into the dict shape sentiment.py
    expects. Returns None on hard failure (the None bubbles up to the A/B
    logger which writes a tagged fingpt:error entry).

    2026-04-09 (fix/fingpt-parser-prompt): the original fingpt migration
    left this wrapper untouched because the parser bug was upstream in
    fingpt_infer._parse_sentiment / _estimate_confidence + the Llama-3
    chat template in PROMPT_TEMPLATES["finance-llama-8b"]. That was all
    fixed in the same commit as this comment update — wiroai-finance-llama-8b
    is a completion model and the new few-shot plain-text templates make
    it emit clean sentiment words. See /mnt/q/models/fingpt_infer.py for
    the parser + template changes.
    """
    if text is None:
        return None
    try:
        sentiment = fingpt_infer._parse_sentiment(text)
        confidence = fingpt_infer._estimate_confidence(text, sentiment)
        scores = {"positive": 0.1, "negative": 0.1, "neutral": 0.1}
        scores[sentiment] = confidence
        remaining = 1.0 - confidence
        other_labels = [lb for lb in fingpt_infer.SENTIMENT_LABELS if lb != sentiment]
        if other_labels:
            share = remaining / len(other_labels)
            for ol in other_labels:
                scores[ol] = share
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "model": "fingpt:finance-llama-8b",
        }
    except Exception:
        logger.debug("fingpt completion parse failed for text=%r", text, exc_info=True)
        return None
