"""Tests for ``portfolio/llm_prewarmer.py``.

The prewarmer is called from ``flush_llm_batch()`` after the rotation
counter bumps. Its job: pre-load the NEXT LLM in the rotation so the
next loop cycle (~60 s later) doesn't pay a cold-swap cost that would
otherwise race Chronos's ``gpu_gate("chronos", timeout=30)``.

These tests cover the public contract:

- ROTATION_SLOTS pinned to (ministral, qwen3, fingpt).
- prewarm_next_model dispatches a 1-token query to the right next-slot
  model when the slot is cold.
- prewarm_next_model is a no-op when the slot is already loaded.
- prewarm_next_model NEVER raises (contract guarantee — a broken
  prewarmer cannot regress the working rotation).
- State JSONL is appended after each attempt.
- Restart idempotency: a prewarm at counter=C for slot S that was
  already recorded as "warmed" in state is skipped.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    """Redirect STATE_FILE / DATA_DIR to a tmp path so the test doesn't
    touch the real ``data/llm_rotation_state.jsonl``. Also sets
    PF_PREWARM_FORCE_RUN=1 so the prewarmer's pytest-auto-skip guard
    is bypassed — these tests are the canonical exerciser of the code.
    """
    from portfolio import llm_prewarmer as mod
    state_file = tmp_path / "llm_rotation_state.jsonl"
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "STATE_FILE", state_file)
    monkeypatch.setenv("PF_PREWARM_FORCE_RUN", "1")
    return state_file


@pytest.fixture
def mock_query(monkeypatch):
    """Replace query_llama_server with a recorder. Test can configure
    .return_value to None (failure) or any string (success).
    """
    calls: list[tuple] = []
    holder = {"return_value": "ok"}

    def fake(name, prompt, n_predict=1024, temperature=0.0, top_p=0.2, stop=None):
        calls.append((name, prompt, n_predict, temperature))
        return holder["return_value"]

    import portfolio.llama_server as llama_server_mod
    monkeypatch.setattr(llama_server_mod, "query_llama_server", fake)
    holder["calls"] = calls
    return holder


@pytest.fixture
def mock_pid_reader(monkeypatch):
    """Replace _read_pid_model so we can simulate "slot already loaded"."""
    holder = {"current_model": None}

    def fake_read():
        return (1234, holder["current_model"])

    import portfolio.llama_server as llama_server_mod
    monkeypatch.setattr(llama_server_mod, "_read_pid_model", fake_read)
    return holder


# -------------------- ROTATION ORDER --------------------

def test_rotation_order_pinned():
    """If this fails, the prewarmer is out of sync with llm_batch's
    rotation. The order must match exactly so we pre-warm the right slot.
    """
    from portfolio.llm_prewarmer import ROTATION_SLOTS
    assert ROTATION_SLOTS == ("ministral", "qwen3", "fingpt")


def test_rotation_slot_to_server_mapping():
    """Sanity: each abstract rotation name maps to a real llama_server slot."""
    from portfolio.llm_prewarmer import ROTATION_SLOTS, ROTATION_SLOT_TO_SERVER
    for slot in ROTATION_SLOTS:
        assert slot in ROTATION_SLOT_TO_SERVER
    assert ROTATION_SLOT_TO_SERVER["ministral"] == "ministral3"
    assert ROTATION_SLOT_TO_SERVER["qwen3"] == "qwen3"
    assert ROTATION_SLOT_TO_SERVER["fingpt"] == "finance-llama-8b"


def test_rotation_alignment_with_llm_batch():
    """The prewarmer's rotation tuple must equal llm_batch._LLM_ROTATION.
    Drift between these two = wrong model pre-warmed = no win.
    """
    from portfolio import llm_batch
    from portfolio.llm_prewarmer import ROTATION_SLOTS
    assert ROTATION_SLOTS == llm_batch._LLM_ROTATION


# -------------------- HAPPY PATH --------------------

def test_prewarm_next_model_queries_next_slot_when_cold(
    isolated_state, mock_query, mock_pid_reader,
):
    """After flush at counter=2 (ministral just ran), is_llm_on_cycle says
    next cycle is slot (2-1)%3 = 1 = qwen3. So prewarm must call
    query_llama_server with server name "qwen3" and n_predict=1.
    """
    mock_pid_reader["current_model"] = "ministral3"  # different from qwen3
    from portfolio.llm_prewarmer import prewarm_next_model
    result = prewarm_next_model(current_counter=2)
    assert result is True
    assert len(mock_query["calls"]) == 1
    name, prompt, n_predict, _temp = mock_query["calls"][0]
    assert name == "qwen3"
    assert n_predict == 1
    assert prompt  # non-empty


def test_prewarm_after_qwen3_targets_fingpt(
    isolated_state, mock_query, mock_pid_reader,
):
    """counter=3 (qwen3 just ran) → next cycle slot=(3-1)%3=2=fingpt →
    server name "finance-llama-8b".
    """
    mock_pid_reader["current_model"] = "qwen3"
    from portfolio.llm_prewarmer import prewarm_next_model
    assert prewarm_next_model(current_counter=3) is True
    name, _, n_predict, _ = mock_query["calls"][0]
    assert name == "finance-llama-8b"
    assert n_predict == 1


def test_prewarm_after_fingpt_wraps_to_ministral(
    isolated_state, mock_query, mock_pid_reader,
):
    """counter=4 (fingpt just ran) → next cycle slot=(4-1)%3=0=ministral →
    server name "ministral3".
    """
    mock_pid_reader["current_model"] = "finance-llama-8b"
    from portfolio.llm_prewarmer import prewarm_next_model
    assert prewarm_next_model(current_counter=4) is True
    name, _, _, _ = mock_query["calls"][0]
    assert name == "ministral3"


# -------------------- NO-OP PATHS --------------------

def test_prewarm_next_model_returns_false_when_slot_already_loaded(
    isolated_state, mock_query, mock_pid_reader,
):
    """If the target server slot is already the active model, skip the
    query entirely. No swap is needed, so a dummy query would just waste
    time.
    """
    # counter=2 → next slot = qwen3. PID file says qwen3 is loaded.
    mock_pid_reader["current_model"] = "qwen3"
    from portfolio.llm_prewarmer import prewarm_next_model
    result = prewarm_next_model(current_counter=2)
    assert result is False
    assert mock_query["calls"] == []
    # State still written so we can audit no-op cycles.
    assert isolated_state.exists()
    lines = isolated_state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["outcome"] == "already_loaded"
    assert entry["prewarmed_slot"] == "qwen3"


def test_prewarm_zero_counter_is_noop(
    isolated_state, mock_query, mock_pid_reader,
):
    """counter=0 means no flush has happened yet (warmup runs all three).
    Pre-warming at counter=0 is meaningless — no rotation has started.
    """
    from portfolio.llm_prewarmer import prewarm_next_model
    assert prewarm_next_model(current_counter=0) is False
    assert mock_query["calls"] == []


def test_prewarm_negative_counter_is_noop(
    isolated_state, mock_query, mock_pid_reader,
):
    """Defensive: a negative counter is nonsense; refuse without crashing."""
    from portfolio.llm_prewarmer import prewarm_next_model
    assert prewarm_next_model(current_counter=-1) is False
    assert mock_query["calls"] == []


# -------------------- EXCEPTION SAFETY (CONTRACT) --------------------

def test_prewarm_next_model_swallows_query_exceptions(
    isolated_state, mock_pid_reader, monkeypatch,
):
    """query_llama_server() raising → prewarmer returns False, no
    exception propagates. Contract guarantee: a broken prewarmer can
    NEVER regress the working rotation path.
    """
    mock_pid_reader["current_model"] = "ministral3"

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic prewarm failure")

    import portfolio.llama_server as llama_server_mod
    monkeypatch.setattr(llama_server_mod, "query_llama_server", boom)

    from portfolio.llm_prewarmer import prewarm_next_model
    # counter=2 → next slot qwen3 (cold). Will trigger boom.
    result = prewarm_next_model(current_counter=2)
    assert result is False


def test_prewarm_next_model_swallows_pid_read_exceptions(
    isolated_state, mock_query, monkeypatch,
):
    """If _read_pid_model raises, the prewarmer falls through to "not
    already loaded" and still attempts the query. No exception leaks.
    """
    def boom():
        raise RuntimeError("pid read busted")

    import portfolio.llama_server as llama_server_mod
    monkeypatch.setattr(llama_server_mod, "_read_pid_model", boom)

    from portfolio.llm_prewarmer import prewarm_next_model
    # Should proceed to call query_llama_server because is_loaded fell back
    # to False on the exception. No raise.
    result = prewarm_next_model(current_counter=2)
    assert result is True
    assert len(mock_query["calls"]) == 1


def test_prewarm_returns_false_when_query_returns_none(
    isolated_state, mock_query, mock_pid_reader,
):
    """query_llama_server returns None on legit server failure. Prewarmer
    treats this as a soft failure: returns False, records state with
    outcome=query_none.
    """
    mock_pid_reader["current_model"] = "ministral3"
    mock_query["return_value"] = None
    from portfolio.llm_prewarmer import prewarm_next_model
    assert prewarm_next_model(current_counter=2) is False
    lines = isolated_state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["outcome"] == "query_none"


# -------------------- STATE PERSISTENCE --------------------

def test_state_jsonl_written_after_prewarm(
    isolated_state, mock_query, mock_pid_reader,
):
    """A successful prewarm must append exactly one line to the state
    JSONL with the right fields.
    """
    mock_pid_reader["current_model"] = "ministral3"
    from portfolio.llm_prewarmer import prewarm_next_model
    prewarm_next_model(current_counter=2)
    assert isolated_state.exists()
    lines = isolated_state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["counter"] == 2
    assert entry["prewarmed_slot"] == "qwen3"
    assert entry["server_slot"] == "qwen3"
    assert entry["outcome"] == "warmed"
    assert "ts" in entry
    assert "duration_s" in entry


def test_state_jsonl_read_on_restart_skips_duplicate(
    isolated_state, mock_query, mock_pid_reader,
):
    """If state file already has a "warmed" record for (counter=5, qwen3),
    a new prewarm at counter=5 must be a no-op — the previous process
    already warmed it.
    """
    # Seed state file with a prior "warmed" record at counter=5.
    seed = {
        "ts": "2026-05-11T12:00:00+00:00",
        "counter": 5,
        "prewarmed_slot": "qwen3",
        "server_slot": "qwen3",
        "outcome": "warmed",
        "duration_s": 12.3,
    }
    isolated_state.write_text(json.dumps(seed) + "\n", encoding="utf-8")

    mock_pid_reader["current_model"] = "ministral3"  # qwen3 NOT loaded
    from portfolio.llm_prewarmer import prewarm_next_model
    # counter=5 → next slot index (5-1)%3 = 1 → qwen3. Should be skipped.
    result = prewarm_next_model(current_counter=5)
    assert result is False
    assert mock_query["calls"] == []
    # No new state line appended.
    lines = isolated_state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_state_jsonl_different_counter_does_not_skip(
    isolated_state, mock_query, mock_pid_reader,
):
    """Idempotency is per-counter. A new counter must trigger a fresh
    prewarm even if the previous record was a "warmed" entry.
    """
    seed = {
        "ts": "2026-05-11T12:00:00+00:00",
        "counter": 5,
        "prewarmed_slot": "qwen3",
        "server_slot": "qwen3",
        "outcome": "warmed",
    }
    isolated_state.write_text(json.dumps(seed) + "\n", encoding="utf-8")

    mock_pid_reader["current_model"] = "qwen3"  # ministral3 (the new target) NOT loaded
    from portfolio.llm_prewarmer import prewarm_next_model
    # counter=7 → next slot index (7-1)%3 = 0 → ministral. Should run.
    result = prewarm_next_model(current_counter=7)
    assert result is True
    assert len(mock_query["calls"]) == 1
    assert mock_query["calls"][0][0] == "ministral3"


def test_state_corruption_does_not_crash(
    isolated_state, mock_query, mock_pid_reader,
):
    """A corrupt state file (invalid JSON) must not crash the prewarmer.
    We fall back to "no prior state" and run a fresh prewarm.
    """
    isolated_state.write_text("this is not valid json\n", encoding="utf-8")
    mock_pid_reader["current_model"] = "ministral3"
    from portfolio.llm_prewarmer import prewarm_next_model
    result = prewarm_next_model(current_counter=2)
    # Falls back to running the prewarm because the read failed.
    assert result is True


# -------------------- INTEGRATION SHAPE --------------------

def test_prewarm_signature_compatible_with_caller(
    isolated_state, mock_query, mock_pid_reader,
):
    """flush_llm_batch passes shared_state._full_llm_cycle_count (int).
    Prewarmer must accept positional int counter and return bool.
    """
    mock_pid_reader["current_model"] = "ministral3"
    from portfolio.llm_prewarmer import prewarm_next_model
    result = prewarm_next_model(2)
    assert isinstance(result, bool)
