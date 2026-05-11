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
    """If state file already has a "warmed" record for (counter=5, qwen3)
    AND the llama_server currently has qwen3 loaded, a new prewarm at
    counter=5 must be a no-op — the previous process already warmed it
    and the slot is still warm.

    Fix A 2026-05-11 (codex review): added the "currently loaded" check
    on top of the state-record check. See
    test_restart_with_stale_state_and_different_loaded_model_forces_prewarm
    below for the converse case.
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

    # qwen3 still loaded — restart sees consistent state.
    mock_pid_reader["current_model"] = "qwen3"
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


# -------------------- FIX A: RESTART STALE-STATE GUARD --------------------

def test_restart_with_stale_state_and_different_loaded_model_forces_prewarm(
    isolated_state, mock_query, mock_pid_reader,
):
    """Fix A 2026-05-11 (codex review): the rotation counter is in-memory
    only — it resets to 0 on process restart. A fresh process will re-hit
    counter=1, counter=2, ... and the previous lifetime's "warmed" record
    for those counters is stale. We must NOT skip the prewarm if the
    currently loaded slot is something other than the expected next slot.

    Scenario: state JSONL says (counter=1, slot=qwen3) was warmed in a
    prior process lifetime. New process boots and ends up at counter=1
    (just ran ministral). llama_server currently has ministral3 loaded
    (NOT qwen3). The prewarm at counter=1 must dispatch — the in-memory
    counter is a fresh sequence and the slot is genuinely cold.
    """
    seed = {
        "ts": "2026-05-11T08:00:00+00:00",
        "counter": 1,
        "prewarmed_slot": "qwen3",
        "server_slot": "qwen3",
        "outcome": "warmed",
        "duration_s": 12.3,
    }
    isolated_state.write_text(json.dumps(seed) + "\n", encoding="utf-8")

    # Restart aftermath: ministral3 is loaded, NOT qwen3. State is stale.
    mock_pid_reader["current_model"] = "ministral3"
    from portfolio.llm_prewarmer import prewarm_next_model
    # counter=1 → next slot (1-1)%3 = 0 → ministral. But ministral is
    # already loaded — pick a counter where expected_next != currently_loaded.
    # Use counter=2 → next slot index (2-1)%3 = 1 → qwen3. State seed must
    # match (counter=2, qwen3) to exercise the stale-state path.
    seed2 = {
        "ts": "2026-05-11T08:00:00+00:00",
        "counter": 2,
        "prewarmed_slot": "qwen3",
        "server_slot": "qwen3",
        "outcome": "warmed",
        "duration_s": 12.3,
    }
    isolated_state.write_text(json.dumps(seed2) + "\n", encoding="utf-8")

    result = prewarm_next_model(current_counter=2)
    assert result is True, (
        "Expected force-prewarm because currently loaded (ministral3) "
        "differs from expected next slot (qwen3)"
    )
    assert len(mock_query["calls"]) == 1
    assert mock_query["calls"][0][0] == "qwen3"
    # Fresh "warmed" record appended on top of the stale seed.
    lines = isolated_state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    fresh = json.loads(lines[-1])
    assert fresh["outcome"] == "warmed"
    assert fresh["prewarmed_slot"] == "qwen3"


def test_restart_with_stale_state_and_matching_loaded_model_skips(
    isolated_state, mock_query, mock_pid_reader,
):
    """Fix A 2026-05-11 (codex review): converse of the above. State
    JSONL says (counter=2, qwen3) was warmed AND llama_server currently
    has qwen3 loaded. This is the legitimate idempotency case: the slot
    really is warm, skip. No query, no new state line.
    """
    seed = {
        "ts": "2026-05-11T08:00:00+00:00",
        "counter": 2,
        "prewarmed_slot": "qwen3",
        "server_slot": "qwen3",
        "outcome": "warmed",
        "duration_s": 12.3,
    }
    isolated_state.write_text(json.dumps(seed) + "\n", encoding="utf-8")

    mock_pid_reader["current_model"] = "qwen3"
    from portfolio.llm_prewarmer import prewarm_next_model
    result = prewarm_next_model(current_counter=2)
    assert result is False
    assert mock_query["calls"] == []
    # No new line appended — clean idempotent skip via state path.
    lines = isolated_state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


# -------------------- FIX C: TAIL-READ STATE FILE --------------------

def test_read_last_state_tail_read_on_large_file(
    isolated_state, monkeypatch,
):
    """Fix C 2026-05-11 (codex review): ``_read_last_state`` must do a
    real tail read, not ``f.readlines()`` over the whole file.

    Strategy: write 100 records, then patch ``open`` to count how many
    bytes are actually read. We assert:
      (a) The last record's counter is returned correctly.
      (b) The read is bounded — we did NOT slurp the entire file.
    """
    from portfolio import llm_prewarmer as mod

    # Write 100 lines. Each line is well over 100 bytes so the file is
    # >> the 8KB tail block — guarantees a multi-block file.
    lines = []
    for i in range(100):
        rec = {
            "ts": f"2026-05-11T12:{i % 60:02d}:00+00:00",
            "counter": i,
            "prewarmed_slot": "qwen3",
            "server_slot": "qwen3",
            "outcome": "warmed",
            "duration_s": 12.3,
            "filler": "x" * 200,  # pad to make the file fat
        }
        lines.append(json.dumps(rec))
    isolated_state.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total_size = isolated_state.stat().st_size
    assert total_size > 8192, (
        f"test setup: file too small ({total_size}B) to exercise tail read"
    )

    # Track how many bytes get read from the state file across all opens.
    bytes_read = {"total": 0}
    real_open = open

    def counting_open(path, *args, **kwargs):
        fh = real_open(path, *args, **kwargs)
        try:
            same = (
                str(path) == str(isolated_state)
                or getattr(fh, "name", None) == str(isolated_state)
            )
        except Exception:
            same = False
        if not same:
            return fh
        # Wrap read methods to accumulate byte counts.
        orig_read = fh.read

        def counting_read(*a, **kw):
            data = orig_read(*a, **kw)
            try:
                bytes_read["total"] += len(data) if isinstance(data, (bytes, str)) else 0
            except Exception:
                pass
            return data

        fh.read = counting_read  # type: ignore[assignment]
        return fh

    monkeypatch.setattr("builtins.open", counting_open)

    result = mod._read_last_state()
    assert result is not None, "tail read returned None on valid file"
    assert result["counter"] == 99, (
        f"expected last record counter=99, got {result.get('counter')}"
    )
    # Bounded read: must not pull the whole file. 16KB headroom covers the
    # 8KB tail block plus any small ancillary reads; the full file is much
    # larger.
    assert bytes_read["total"] < 16384, (
        f"tail read pulled {bytes_read['total']}B of a {total_size}B file "
        "— expected bounded read (<16KB)"
    )
    assert bytes_read["total"] < total_size, (
        "tail read pulled the entire file — Fix C not applied"
    )


def test_read_last_state_empty_file_returns_none(isolated_state):
    """Fix C edge case: zero-byte state file must return None, not raise."""
    from portfolio import llm_prewarmer as mod
    isolated_state.write_text("", encoding="utf-8")
    assert mod._read_last_state() is None


def test_read_last_state_small_file_still_works(isolated_state):
    """Fix C: files smaller than the tail block must still return the
    last record correctly (we read the whole thing in that case).
    """
    from portfolio import llm_prewarmer as mod
    rec = {
        "ts": "2026-05-11T12:00:00+00:00",
        "counter": 3,
        "prewarmed_slot": "fingpt",
        "server_slot": "finance-llama-8b",
        "outcome": "warmed",
    }
    isolated_state.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    result = mod._read_last_state()
    assert result is not None
    assert result["counter"] == 3
    assert result["prewarmed_slot"] == "fingpt"
