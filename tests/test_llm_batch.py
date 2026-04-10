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
import time
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


# ---------------------------------------------------------------------------
# Rotation scheduling (perf/llama-swap-reduction, 2026-04-10)
# ---------------------------------------------------------------------------
#
# Verifies that is_llm_on_cycle() rotates across ministral → qwen3 → fingpt
# as the shared_state counter advances, and that flush_llm_batch() advances
# the counter by exactly 1 when any phase had queued work.


@pytest.fixture
def reset_rotation_counter():
    """Reset the module-level rotation counter before and after each test."""
    from portfolio import shared_state as _ss
    original = _ss._full_llm_cycle_count
    _ss._full_llm_cycle_count = 0
    yield _ss
    _ss._full_llm_cycle_count = original


class TestRotationSchedule:
    """is_llm_on_cycle drives which LLM runs each flush. Warmup (counter==0)
    runs all; from counter==1 onward, rotation: ministral → qwen3 → fingpt → …
    """

    def test_warmup_runs_all_llms(self, reset_rotation_counter):
        from portfolio.llm_batch import is_llm_on_cycle
        reset_rotation_counter._full_llm_cycle_count = 0
        assert is_llm_on_cycle("ministral") is True
        assert is_llm_on_cycle("qwen3") is True
        assert is_llm_on_cycle("fingpt") is True

    def test_counter_1_runs_ministral_only(self, reset_rotation_counter):
        from portfolio.llm_batch import is_llm_on_cycle
        reset_rotation_counter._full_llm_cycle_count = 1
        assert is_llm_on_cycle("ministral") is True
        assert is_llm_on_cycle("qwen3") is False
        assert is_llm_on_cycle("fingpt") is False

    def test_counter_2_runs_qwen3_only(self, reset_rotation_counter):
        from portfolio.llm_batch import is_llm_on_cycle
        reset_rotation_counter._full_llm_cycle_count = 2
        assert is_llm_on_cycle("ministral") is False
        assert is_llm_on_cycle("qwen3") is True
        assert is_llm_on_cycle("fingpt") is False

    def test_counter_3_runs_fingpt_only(self, reset_rotation_counter):
        from portfolio.llm_batch import is_llm_on_cycle
        reset_rotation_counter._full_llm_cycle_count = 3
        assert is_llm_on_cycle("ministral") is False
        assert is_llm_on_cycle("qwen3") is False
        assert is_llm_on_cycle("fingpt") is True

    def test_counter_4_wraps_to_ministral(self, reset_rotation_counter):
        """After a full rotation, counter=4 should land back on ministral."""
        from portfolio.llm_batch import is_llm_on_cycle
        reset_rotation_counter._full_llm_cycle_count = 4
        assert is_llm_on_cycle("ministral") is True
        assert is_llm_on_cycle("qwen3") is False
        assert is_llm_on_cycle("fingpt") is False

    def test_full_six_cycle_rotation(self, reset_rotation_counter):
        """Over cycles 1..6, each LLM runs exactly twice (2 full rotations)."""
        from portfolio.llm_batch import is_llm_on_cycle
        counts = {"ministral": 0, "qwen3": 0, "fingpt": 0}
        for counter in range(1, 7):
            reset_rotation_counter._full_llm_cycle_count = counter
            for name in counts:
                if is_llm_on_cycle(name):
                    counts[name] += 1
        assert counts == {"ministral": 2, "qwen3": 2, "fingpt": 2}

    def test_unknown_llm_name_raises(self, reset_rotation_counter):
        """Programming errors should fail loudly, not silently skip."""
        from portfolio.llm_batch import is_llm_on_cycle
        reset_rotation_counter._full_llm_cycle_count = 1
        with pytest.raises(ValueError):
            is_llm_on_cycle("not_a_real_llm")


class TestFlushAdvancesCounter:
    """flush_llm_batch() must bump _full_llm_cycle_count after processing
    work; an empty flush must not bump the counter (no cycle was spent).
    """

    def test_empty_flush_does_not_advance_counter(self, reset_rotation_counter):
        from portfolio.llm_batch import flush_llm_batch, _ministral_queue, _qwen3_queue, _fingpt_queue, _lock
        with _lock:
            _ministral_queue.clear()
            _qwen3_queue.clear()
            _fingpt_queue.clear()
        reset_rotation_counter._full_llm_cycle_count = 5
        flush_llm_batch()
        assert reset_rotation_counter._full_llm_cycle_count == 5

    def test_nonempty_flush_advances_counter(
        self, reset_rotation_counter, fake_fingpt_infer, fake_llama_server, stash_recorder,
    ):
        """A flush that processes at least one phase must bump counter by 1."""
        from portfolio.llm_batch import (
            enqueue_fingpt, flush_llm_batch,
            _ministral_queue, _qwen3_queue, _fingpt_queue, _lock,
        )
        # Code-review finding N2: clear ALL three queues defensively so any
        # leftover ministral/qwen3 item from another test doesn't trip the
        # fake server's assert on model name.
        with _lock:
            _ministral_queue.clear()
            _qwen3_queue.clear()
            _fingpt_queue.clear()

        fake_llama_server["response"] = ["positive"]
        enqueue_fingpt(
            "BTC:rot-test", "headlines",
            {"mode": "headlines", "ticker": "BTC", "texts": ["h1"]},
        )
        reset_rotation_counter._full_llm_cycle_count = 2
        flush_llm_batch()
        assert reset_rotation_counter._full_llm_cycle_count == 3


class TestCachedOrEnqueueRotationGate:
    """_cached_or_enqueue's should_enqueue_fn parameter lets callers skip
    enqueue when rotation says off-cycle, as long as stale data is available.
    """

    def _setup_cache(self, key, age_seconds, data="cached_vote"):
        """Put a fake cache entry at the given age."""
        from portfolio import shared_state as _ss
        _ss._tool_cache[key] = {
            "data": data,
            "time": time.time() - age_seconds,
            "ttl": 900,
        }
        _ss._loading_keys.discard(key)
        _ss._loading_timestamps.pop(key, None)

    def _clear_cache(self, key):
        from portfolio import shared_state as _ss
        _ss._tool_cache.pop(key, None)
        _ss._loading_keys.discard(key)
        _ss._loading_timestamps.pop(key, None)

    def test_off_cycle_skips_enqueue_and_returns_stale(self):
        """When rotation says off-cycle and stale data is available, enqueue
        should be skipped and the stale data returned."""
        from portfolio.shared_state import _cached_or_enqueue
        key = "test_llm_off"
        self._setup_cache(key, age_seconds=1800)  # 30 min, within 5x stale

        enqueues = []
        def fake_enqueue(k, ctx):
            enqueues.append((k, ctx))

        result = _cached_or_enqueue(
            key, ttl=900, enqueue_fn=fake_enqueue, context={"x": 1},
            should_enqueue_fn=lambda: False,
            max_stale_factor=5,
        )
        assert result == "cached_vote"
        assert enqueues == []  # skipped because off-cycle with stale fallback
        self._clear_cache(key)

    def test_on_cycle_enqueues_and_returns_stale(self):
        """When rotation says on-cycle, enqueue should fire and stale data
        should be returned (caller gets stale now, fresh next flush)."""
        from portfolio.shared_state import _cached_or_enqueue
        key = "test_llm_on"
        self._setup_cache(key, age_seconds=1800)  # stale but within window

        enqueues = []
        def fake_enqueue(k, ctx):
            enqueues.append((k, ctx))

        result = _cached_or_enqueue(
            key, ttl=900, enqueue_fn=fake_enqueue, context={"x": 1},
            should_enqueue_fn=lambda: True,
            max_stale_factor=5,
        )
        assert result == "cached_vote"
        assert len(enqueues) == 1
        self._clear_cache(key)

    def test_off_cycle_force_enqueues_when_no_stale(self):
        """When rotation says off-cycle BUT stale is NOT available, enqueue
        MUST fire anyway — the caller has no fallback value so we cannot
        leave them empty-handed."""
        from portfolio.shared_state import _cached_or_enqueue
        key = "test_llm_cold_off"
        self._clear_cache(key)  # no cache at all

        enqueues = []
        def fake_enqueue(k, ctx):
            enqueues.append((k, ctx))

        result = _cached_or_enqueue(
            key, ttl=900, enqueue_fn=fake_enqueue, context={"x": 1},
            should_enqueue_fn=lambda: False,
            max_stale_factor=5,
        )
        assert result is None  # no data to return, first-ever enqueue
        assert len(enqueues) == 1  # forced enqueue because no stale fallback
        self._clear_cache(key)

    def test_stale_beyond_max_stale_factor_returns_none(self):
        """Stale data older than ttl * max_stale_factor is not returned; the
        call should force-enqueue and return None (cold start semantic)."""
        from portfolio.shared_state import _cached_or_enqueue
        key = "test_llm_too_old"
        # 1 hour old with TTL 900 × max_stale_factor 2 = 30 min max → too old
        self._setup_cache(key, age_seconds=3600)

        enqueues = []
        def fake_enqueue(k, ctx):
            enqueues.append((k, ctx))

        result = _cached_or_enqueue(
            key, ttl=900, enqueue_fn=fake_enqueue, context={"x": 1},
            should_enqueue_fn=lambda: False,
            max_stale_factor=2,  # 1800 s max stale, 3600 s age exceeds it
        )
        assert result is None  # stale too old to return
        assert len(enqueues) == 1  # force-enqueued
        self._clear_cache(key)

    def test_fresh_cache_returns_without_enqueue(self):
        """Fresh cache (age < ttl) bypasses rotation entirely — no enqueue."""
        from portfolio.shared_state import _cached_or_enqueue
        key = "test_llm_fresh"
        self._setup_cache(key, age_seconds=60)  # very fresh

        enqueues = []
        def fake_enqueue(k, ctx):
            enqueues.append((k, ctx))

        result = _cached_or_enqueue(
            key, ttl=900, enqueue_fn=fake_enqueue, context={"x": 1},
            should_enqueue_fn=lambda: False,
            max_stale_factor=5,
        )
        assert result == "cached_vote"
        assert enqueues == []  # fresh cache, no enqueue regardless of rotation
        self._clear_cache(key)

    def test_cached_none_triggers_force_enqueue_off_cycle(self):
        """Code-review finding N1: when main.py wrote _update_cache(key, None)
        as a retry cooldown after a failed LLM flush, the rotation gate must
        treat this as 'stale not available' and force-enqueue instead of
        returning None silently.

        Setup: age=1800s (30 min), TTL=900s → stale within max_stale_factor=5
        (max stale = 4500s = 75 min). Cached data is None. Rotation off-cycle.
        Expected: force enqueue because None is not a valid stale fallback.
        """
        from portfolio.shared_state import _cached_or_enqueue
        key = "test_llm_none_cache"
        self._setup_cache(key, age_seconds=1800, data=None)  # stale path

        enqueues = []
        def fake_enqueue(k, ctx):
            enqueues.append((k, ctx))

        # Rotation says off-cycle, but cached data is None → should force-enqueue
        result = _cached_or_enqueue(
            key, ttl=900, enqueue_fn=fake_enqueue, context={"x": 1},
            should_enqueue_fn=lambda: False,
            max_stale_factor=5,
        )
        assert result is None  # None cached was not treated as stale
        assert len(enqueues) == 1  # force-enqueued for retry
        self._clear_cache(key)

    def test_cached_none_fresh_ttl_still_returns_none(self):
        """Complement to the above: while the cache is still FRESH (age < ttl),
        a None entry is preserved as the short-lived failure cooldown. Do not
        re-enqueue during this window — that was the original intent of
        main.py writing _update_cache(key, None, ttl=60).
        """
        from portfolio.shared_state import _cached_or_enqueue
        key = "test_llm_none_fresh"
        self._setup_cache(key, age_seconds=30, data=None)  # fresh path

        enqueues = []
        def fake_enqueue(k, ctx):
            enqueues.append((k, ctx))

        result = _cached_or_enqueue(
            key, ttl=900, enqueue_fn=fake_enqueue, context={"x": 1},
            should_enqueue_fn=lambda: False,
            max_stale_factor=5,
        )
        # Fresh path: returns the cached None, no enqueue (legacy behavior).
        assert result is None
        assert enqueues == []
        self._clear_cache(key)
