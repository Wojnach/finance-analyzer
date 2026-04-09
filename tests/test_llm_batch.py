"""Tests for portfolio/llm_batch.py — focused on the fingpt Phase 3 added
2026-04-09 as part of feat/fingpt-in-llmbatch. Phases 1 (ministral) and 2
(qwen3) are exercised indirectly through test_sentiment tests; this file
covers the new fingpt path end-to-end with mocked dependencies so the suite
does not require /mnt/q/models, llama-cpp-python, or a running llama-server.

The tests drive `portfolio.llm_batch._flush_fingpt_phase` directly (rather
than the full `flush_llm_batch`) to keep the scope tight on the new code.
The monkeypatches replace:
- `sys.modules["fingpt_infer"]` with a stub providing `PROMPT_TEMPLATES`,
  `CUMULATIVE_PROMPT`, `_parse_sentiment`, `_estimate_confidence`,
  `SENTIMENT_LABELS` — exactly the module surface Phase 3 consumes
- `portfolio.llama_server.query_llama_server_batch` with a fake that returns
  predetermined texts
- `portfolio.sentiment._stash_fingpt_result` with a recorder so the test
  can assert on what Phase 3 handed back to sentiment.py
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fake_fingpt_infer(monkeypatch):
    """Install a stub `fingpt_infer` module in sys.modules so Phase 3 can
    import it without /mnt/q/models on sys.path.
    """
    fake = types.ModuleType("fingpt_infer")
    fake.PROMPT_TEMPLATES = {
        "finance-llama-8b": "SYS:{headline}:END",
    }
    fake.CUMULATIVE_PROMPT = "CUM({count}):{headlines_block}:END"
    fake.SENTIMENT_LABELS = {"positive", "negative", "neutral"}

    def _parse_sentiment(text: str) -> str:
        # Trivial deterministic parser for tests: the last token wins.
        t = (text or "").strip().lower()
        for label in ("positive", "negative", "neutral"):
            if label in t:
                return label
        return "neutral"

    def _estimate_confidence(text: str, sentiment: str) -> float:
        # Deterministic: return 0.8 if the exact word is the whole response,
        # 0.5 otherwise. Lets us distinguish clean vs noisy completions.
        return 0.8 if (text or "").strip().lower() == sentiment else 0.5

    fake._parse_sentiment = _parse_sentiment
    fake._estimate_confidence = _estimate_confidence
    monkeypatch.setitem(sys.modules, "fingpt_infer", fake)
    return fake


@pytest.fixture
def fake_llama_server(monkeypatch):
    """Replace query_llama_server_batch with a callable that returns a
    preset list. The test controls the return value via `.next_response`.
    """
    holder = {"response": []}

    def fake_batch(name, prompts_and_params):
        assert name == "finance-llama-8b", f"unexpected model {name}"
        resp = holder["response"]
        # Support either a callable (so the test can compute per-prompt
        # based on the prompt text) or a plain list.
        if callable(resp):
            return [resp(p) for p in prompts_and_params]
        # Pad with None if the test provided fewer responses than prompts.
        out = list(resp)
        while len(out) < len(prompts_and_params):
            out.append(None)
        return out[:len(prompts_and_params)]

    import portfolio.llama_server as llama_server_mod
    monkeypatch.setattr(llama_server_mod, "query_llama_server_batch", fake_batch)
    # The Phase 3 code imports query_llama_server_batch inside the function,
    # so patching on the module is sufficient.
    return holder


@pytest.fixture
def stash_recorder(monkeypatch):
    """Replace sentiment._stash_fingpt_result with a recorder. Returns a
    list of (ab_key, sub_key, result) tuples captured during the phase.
    """
    captured: list[tuple[str, str, object]] = []

    def recorder(ab_key, sub_key, result):
        captured.append((ab_key, sub_key, result))

    import portfolio.sentiment as sentiment_mod
    monkeypatch.setattr(sentiment_mod, "_stash_fingpt_result", recorder)
    return captured


def test_fingpt_phase_empty_queue(fake_fingpt_infer, fake_llama_server, stash_recorder):
    """Empty queue is a clean no-op — no stash calls, no server calls."""
    from portfolio.llm_batch import _flush_fingpt_phase
    _flush_fingpt_phase([])
    assert stash_recorder == []


def test_fingpt_phase_headlines(fake_fingpt_infer, fake_llama_server, stash_recorder):
    """Per-headline mode: 3 headlines → 3 prompts → 3 parsed results stashed
    back under the same ab_key/sub_key as a list."""
    fake_llama_server["response"] = ["positive", "negative", "neutral"]

    from portfolio.llm_batch import _flush_fingpt_phase
    _flush_fingpt_phase([
        (
            "BTC:t0",
            "headlines",
            {
                "mode": "headlines",
                "ticker": "BTC",
                "texts": ["h1", "h2", "h3"],
            },
        ),
    ])

    assert len(stash_recorder) == 1
    ab_key, sub_key, result = stash_recorder[0]
    assert ab_key == "BTC:t0"
    assert sub_key == "headlines"
    assert isinstance(result, list) and len(result) == 3
    assert [r["sentiment"] for r in result] == ["positive", "negative", "neutral"]
    # _estimate_confidence returns 0.8 for clean single-word responses
    assert result[0]["confidence"] == 0.8
    assert result[0]["model"] == "fingpt:finance-llama-8b"
    # Scores dict sums to ~1.0 and puts the bulk on the chosen label
    scores = result[0]["scores"]
    assert scores["positive"] == 0.8
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_fingpt_phase_cumulative(fake_fingpt_infer, fake_llama_server, stash_recorder):
    """Cumulative mode: single prompt → single dict with +0.1 confidence boost
    when headline count >= 5 (matching the daemon-era behavior)."""
    fake_llama_server["response"] = ["positive"]

    from portfolio.llm_batch import _flush_fingpt_phase
    _flush_fingpt_phase([
        (
            "BTC:t0",
            "cumul:0",
            {
                "mode": "cumulative",
                "ticker": "BTC",
                "texts": ["h1", "h2", "h3", "h4", "h5", "h6"],  # >=5 → boost applies
            },
        ),
    ])

    assert len(stash_recorder) == 1
    _, sub_key, result = stash_recorder[0]
    assert sub_key == "cumul:0"
    assert result["sentiment"] == "positive"
    # 0.8 (base) + 0.1 (boost for >=5 headlines) = 0.9
    assert abs(result["confidence"] - 0.9) < 1e-6
    assert result["model"] == "fingpt:cumulative"
    assert result["headline_count"] == 6


def test_fingpt_phase_cumulative_no_boost_under_five(
    fake_fingpt_infer, fake_llama_server, stash_recorder,
):
    """The +0.1 cumulative boost only applies when >=5 headlines; smaller
    clusters get the raw parser confidence."""
    fake_llama_server["response"] = ["negative"]

    from portfolio.llm_batch import _flush_fingpt_phase
    _flush_fingpt_phase([
        (
            "ETH:t0",
            "cumul:1",
            {
                "mode": "cumulative",
                "ticker": "ETH",
                "texts": ["h1", "h2", "h3"],  # <5 → no boost
            },
        ),
    ])

    _, _, result = stash_recorder[0]
    assert result["sentiment"] == "negative"
    assert abs(result["confidence"] - 0.8) < 1e-6  # parser's 0.8, no boost


def test_fingpt_phase_error_returns_none(
    fake_fingpt_infer, fake_llama_server, stash_recorder,
):
    """When query_llama_server_batch returns None for a prompt, the parser
    passes None through and the stashed per-headline list contains None
    for that slot. flush_ab_log filters these downstream."""
    fake_llama_server["response"] = [None, None, None]

    from portfolio.llm_batch import _flush_fingpt_phase
    _flush_fingpt_phase([
        (
            "XAU:t0",
            "headlines",
            {
                "mode": "headlines",
                "ticker": "XAU",
                "texts": ["h1", "h2", "h3"],
            },
        ),
    ])

    assert len(stash_recorder) == 1
    _, _, result = stash_recorder[0]
    assert result == [None, None, None]


def test_fingpt_phase_preserves_order_across_tickers(
    fake_fingpt_infer, fake_llama_server, stash_recorder,
):
    """Interleaved batch of multiple (ab_key, sub_key) pairs still groups
    correctly — results for BTC go to BTC, not ETH."""
    def by_prompt(p):
        # Tag response based on the prompt text so we can verify ordering
        text = p["prompt"]
        if "btc_h" in text:
            return "positive"
        if "eth_h" in text:
            return "negative"
        return "neutral"

    fake_llama_server["response"] = by_prompt

    from portfolio.llm_batch import _flush_fingpt_phase
    _flush_fingpt_phase([
        ("BTC:t0", "headlines", {"mode": "headlines", "ticker": "BTC",
                                  "texts": ["btc_h1", "btc_h2"]}),
        ("ETH:t0", "headlines", {"mode": "headlines", "ticker": "ETH",
                                  "texts": ["eth_h1", "eth_h2", "eth_h3"]}),
    ])

    assert len(stash_recorder) == 2
    by_key = {(ab, sub): r for ab, sub, r in stash_recorder}
    btc_result = by_key[("BTC:t0", "headlines")]
    eth_result = by_key[("ETH:t0", "headlines")]
    assert [r["sentiment"] for r in btc_result] == ["positive", "positive"]
    assert [r["sentiment"] for r in eth_result] == ["negative", "negative", "negative"]


def test_fingpt_phase_swallows_top_level_failure(
    fake_fingpt_infer, stash_recorder, monkeypatch,
):
    """The phase is wrapped in a try/except so a crash inside
    query_llama_server_batch cannot leak out into the main cycle. Fingpt is
    a shadow signal — its failures must never break voting."""
    def boom(name, prompts_and_params):
        raise RuntimeError("server exploded")

    import portfolio.llama_server as llama_server_mod
    monkeypatch.setattr(llama_server_mod, "query_llama_server_batch", boom)

    from portfolio.llm_batch import _flush_fingpt_phase
    # Should not raise.
    _flush_fingpt_phase([
        ("BTC:t0", "headlines", {"mode": "headlines", "ticker": "BTC",
                                  "texts": ["x"]}),
    ])
    # No stash calls because the phase bailed before reaching the stash step.
    assert stash_recorder == []


def test_enqueue_fingpt_dedup():
    """Duplicate (ab_key, sub_key) enqueues are deduped — prevents the same
    ticker from piling up multiple fingpt requests if get_sentiment is
    called twice in the same cycle (cache race)."""
    from portfolio.llm_batch import enqueue_fingpt, _fingpt_queue, _lock

    with _lock:
        _fingpt_queue.clear()

    ctx = {"mode": "headlines", "ticker": "BTC", "texts": ["h"]}
    enqueue_fingpt("BTC:t0", "headlines", ctx)
    enqueue_fingpt("BTC:t0", "headlines", ctx)  # dup
    enqueue_fingpt("BTC:t0", "cumul:0", ctx)    # different sub_key — allowed
    enqueue_fingpt("ETH:t0", "headlines", ctx)  # different ab_key — allowed

    with _lock:
        assert len(_fingpt_queue) == 3
        # The duplicate headlines entry for BTC should NOT have been added.
        keys = {(ab, sub) for ab, sub, _ in _fingpt_queue}
        assert keys == {
            ("BTC:t0", "headlines"),
            ("BTC:t0", "cumul:0"),
            ("ETH:t0", "headlines"),
        }
        _fingpt_queue.clear()


def test_flush_ab_log_merges_fingpt_and_finbert(monkeypatch, tmp_path):
    """End-to-end buffer test: stash an A/B context, stash finbert + fingpt
    results, flush, verify the JSONL row contains both shadows."""
    import portfolio.sentiment as sentiment_mod

    # Redirect AB_LOG_FILE to a tmp path so we don't scribble over the real log
    tmp_log = tmp_path / "sentiment_ab_log.jsonl"
    monkeypatch.setattr(sentiment_mod, "AB_LOG_FILE", tmp_log)

    ab_key = "BTC:flush-test-1"
    sentiment_mod._stash_ab_context(
        ab_key,
        "BTC",
        primary_result={
            "overall_sentiment": "positive",
            "confidence": 0.72,
            "model": "CryptoBERT",
        },
        all_articles=[{"title": "h1"}, {"title": "h2"}, {"title": "h3"}],
        diss_mult=1.0,
    )
    sentiment_mod._stash_finbert_shadow(ab_key, {
        "model": "FinBERT",
        "sentiment": "neutral",
        "confidence": 0.55,
        "avg_scores": {"positive": 0.25, "negative": 0.2, "neutral": 0.55},
    })
    # Stash a 3-headline fingpt result (as Phase 3 would after inference)
    sentiment_mod._stash_fingpt_result(ab_key, "headlines", [
        {"sentiment": "positive", "confidence": 0.8,
         "scores": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
         "model": "fingpt:finance-llama-8b"},
        {"sentiment": "positive", "confidence": 0.8,
         "scores": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
         "model": "fingpt:finance-llama-8b"},
        {"sentiment": "neutral", "confidence": 0.5,
         "scores": {"positive": 0.25, "negative": 0.25, "neutral": 0.5},
         "model": "fingpt:finance-llama-8b"},
    ])

    sentiment_mod.flush_ab_log()

    # File should now contain one row
    assert tmp_log.exists()
    import json
    lines = tmp_log.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["ticker"] == "BTC"
    assert row["primary"]["model"] == "CryptoBERT"
    shadow_models = [s["model"] for s in row["shadow"]]
    # Fingpt headlines shadow + FinBERT shadow
    assert "fingpt:finance-llama-8b" in shadow_models
    assert "FinBERT" in shadow_models

    # Buffer must be empty after flush
    with sentiment_mod._pending_ab_lock:
        assert ab_key not in sentiment_mod._pending_ab_entries


def test_flush_ab_log_empty_buffer_is_noop(tmp_path, monkeypatch):
    """Calling flush_ab_log with nothing pending must not write anything."""
    import portfolio.sentiment as sentiment_mod

    tmp_log = tmp_path / "sentiment_ab_log.jsonl"
    monkeypatch.setattr(sentiment_mod, "AB_LOG_FILE", tmp_log)

    with sentiment_mod._pending_ab_lock:
        sentiment_mod._pending_ab_entries.clear()

    sentiment_mod.flush_ab_log()
    assert not tmp_log.exists()
