"""Property-based invariants — universal truths that hold for ANY valid input.

The user explicitly distrusts both Claude-written production code AND
Claude-written example tests (both can be wrong together). Property-based
tests with `hypothesis` cannot be silently faked: a property is a universal
quantifier over a strategy, and Claude cannot tilt the property without
breaking the property itself.

Scope (Codex 2026-05-10 review):
  These properties test SEQUENTIAL CONTRACTS — not concurrent atomicity.
  P2 verifies that ``atomic_append_jsonl`` round-trips writes in order
  for a single producer. P3 verifies ``atomic_write_json`` cleans up its
  tempfile on success. Neither exercises crash-injection or concurrent
  writers; if you need that coverage, add a separate suite that uses
  ``multiprocessing`` + a fault-injection harness. The properties as
  written still gate against the "Claude rewrote the function and broke
  basic round-trip" failure mode, which is the current threat model.

Each property here asserts something that MUST be true regardless of input:
  1. Portfolio bookkeeping conservation: total_value = cash + Σ qty·price.
  2. JSONL append→read round-trip: same data, same order, no torn lines.
  3. atomic_write_json hygiene: no .tmp residue beside the target on success.
  4. _weighted_consensus determinism: same input → same output.
  5. load_json contract: missing path → default returned (no raise).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    load_json,
)
from portfolio.portfolio_mgr import portfolio_value
from portfolio.signal_engine import _weighted_consensus

# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

TICKERS = ["BTC-USD", "ETH-USD", "XAG-USD", "XAU-USD", "MSTR"]

finite_price = st.floats(
    min_value=0.01, max_value=1_000_000.0,
    allow_nan=False, allow_infinity=False,
)
finite_share = st.floats(
    min_value=0.0, max_value=1_000_000.0,
    allow_nan=False, allow_infinity=False,
)
finite_cash = st.floats(
    min_value=0.0, max_value=10_000_000.0,
    allow_nan=False, allow_infinity=False,
)
finite_fx = st.floats(
    min_value=0.01, max_value=100.0,
    allow_nan=False, allow_infinity=False,
)

# JSON-serialisable scalar values — strings, ints, finite floats, bools, None.
json_scalar = st.one_of(
    st.text(max_size=40),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.booleans(),
    st.none(),
)
# Plain {str: scalar} dict — sufficient for the JSONL round-trip property.
json_dict = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=json_scalar,
    max_size=8,
)


# --------------------------------------------------------------------------
# Property 1 — Portfolio total_value conservation
# --------------------------------------------------------------------------

@st.composite
def portfolio_state(draw):
    """Build a portfolio state dict with the same shape used in production."""
    n = draw(st.integers(min_value=0, max_value=5))
    chosen = draw(st.lists(st.sampled_from(TICKERS), min_size=n, max_size=n, unique=True))
    holdings = {}
    for t in chosen:
        holdings[t] = {"shares": draw(finite_share)}
    return {
        "cash_sek": draw(finite_cash),
        "holdings": holdings,
    }


@given(state=portfolio_state(), fx=finite_fx, data=st.data())
@settings(max_examples=100, deadline=None)
def test_portfolio_total_equals_cash_plus_holdings(state, fx, data):
    """Universal truth: portfolio_value == cash + Σ shares·price·fx.

    For any state and any valid prices, the bookkeeping identity must hold
    within float tolerance. If this ever fails, we have lost money on paper.
    """
    # Prices: every ticker in holdings must have a finite positive price.
    prices = {
        t: data.draw(finite_price, label=f"price_{t}")
        for t in state["holdings"]
    }
    expected = state["cash_sek"] + sum(
        h["shares"] * prices[t] * fx
        for t, h in state["holdings"].items()
    )
    actual = portfolio_value(state, prices, fx)

    # Absolute tolerance scales with magnitude (float64 ~15 digit precision).
    tol = max(1e-6, abs(expected) * 1e-9)
    assert math.isfinite(actual), f"non-finite total: {actual!r}"
    assert abs(actual - expected) <= tol, (
        f"conservation broken: expected={expected} actual={actual} "
        f"diff={actual - expected} state={state} prices={prices} fx={fx}"
    )


# --------------------------------------------------------------------------
# Property 2 — atomic_append_jsonl round-trip
# --------------------------------------------------------------------------

@given(entries=st.lists(json_dict, min_size=0, max_size=20), data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_atomic_write_jsonl_round_trip(tmp_path, entries, data):
    """Universal truth: append N entries, read back N entries in same order.

    No torn lines, no reordering, no silent drops.

    NOTE: pytest's tmp_path fixture is function-scoped, but Hypothesis reuses
    the same fixture across all generated examples in a single test function.
    A naive path = tmp_path / "rt.jsonl" therefore accumulates entries across
    draws and the round-trip looks broken (we found this empirically). Use a
    fresh sub-directory per draw.
    """
    import tempfile
    sub = Path(tempfile.mkdtemp(dir=str(tmp_path), prefix="rt_"))
    path = sub / "rt.jsonl"
    # data fixture reserved here so Hypothesis treats each draw as independent.
    _ = data.draw(st.integers(min_value=0, max_value=10**9))
    for e in entries:
        atomic_append_jsonl(path, e)

    if not entries:
        # Empty input → file may not exist OR be empty.
        if path.exists():
            assert path.read_text(encoding="utf-8") == ""
        return

    text = path.read_text(encoding="utf-8")
    # Each line is exactly one entry; trailing newline guaranteed by writer.
    lines = text.split("\n")
    assert lines[-1] == "", "JSONL must end with a newline (no partial last line)"
    parsed = [json.loads(line) for line in lines[:-1]]

    assert parsed == entries, (
        f"round-trip broken: wrote {len(entries)} read {len(parsed)}"
    )


# --------------------------------------------------------------------------
# Property 3 — atomic_write_json leaves no .tmp residue
# --------------------------------------------------------------------------

@given(payload=json_dict, data=st.data())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_atomic_write_no_tmp_residue(tmp_path, payload, data):
    """Universal truth: a successful atomic_write_json must clean up its tempfile.

    A leftover .tmp file is a leak that grows the data dir over time and
    can confuse recovery code looking for orphaned partial writes.

    Uses a fresh sub-directory per draw — Hypothesis reuses the function-scoped
    tmp_path across examples, so without this isolation prior draws' files
    pollute the .tmp scan.
    """
    import tempfile
    sub = Path(tempfile.mkdtemp(dir=str(tmp_path), prefix="atomic_"))
    _ = data.draw(st.integers(min_value=0, max_value=10**9))
    target = sub / "atomic.json"
    atomic_write_json(target, payload)

    # The mkstemp suffix is ".tmp" — the tempfile names look like
    # "<random>.tmp" and live in `sub` (the parent of `target`).
    leftovers = [
        p for p in sub.iterdir()
        if p != target and (p.suffix == ".tmp" or p.name.endswith(".tmp"))
    ]
    assert leftovers == [], f"tmp residue leaked: {leftovers}"

    # And the actual write must round-trip.
    assert json.loads(target.read_text(encoding="utf-8")) == payload


# --------------------------------------------------------------------------
# Property 4 — Signal voting is deterministic
# --------------------------------------------------------------------------

# Small, well-conditioned signal universe: real signal names so the engine's
# accuracy/regime/horizon-weight lookups operate on canonical keys, not
# strings the engine will silently downweight to ~0.
KNOWN_SIGNALS = ("rsi", "macd", "ema", "bb", "volume_confirmation")
VOTE_VALUES = ("BUY", "SELL", "HOLD")


@st.composite
def voting_input(draw):
    """Build a (votes, accuracy_data, regime) triple.

    Forces enough mature, above-gate accuracies that the gate doesn't
    silently zero-out every voter — this is the determinism property,
    not the accuracy-gate property. Accuracies are bounded floats so
    repeated calls hash to the exact same float bits.
    """
    n = draw(st.integers(min_value=3, max_value=len(KNOWN_SIGNALS)))
    chosen = draw(
        st.lists(st.sampled_from(KNOWN_SIGNALS), min_size=n, max_size=n, unique=True)
    )
    votes = {s: draw(st.sampled_from(VOTE_VALUES)) for s in chosen}
    accuracy = {
        s: {
            "accuracy": draw(st.floats(min_value=0.50, max_value=0.85,
                                       allow_nan=False, allow_infinity=False)),
            "total": draw(st.integers(min_value=50, max_value=2000)),
        }
        for s in chosen
    }
    regime = draw(st.sampled_from(("trending-up", "trending-down", "ranging")))
    return votes, accuracy, regime


@given(triple=voting_input())
@settings(max_examples=100, deadline=None)
def test_signal_voting_deterministic(triple):
    """Universal truth: pure functions are deterministic.

    `_weighted_consensus` does not consume randomness, time, or external
    state — given identical inputs it MUST return identical (decision, conf).
    Two calls back-to-back, no mutation between them.
    """
    votes, accuracy, regime = triple

    out1 = _weighted_consensus(votes, accuracy, regime=regime)
    out2 = _weighted_consensus(votes, accuracy, regime=regime)

    assert out1 == out2, (
        f"non-deterministic vote: {out1} vs {out2} for "
        f"votes={votes} regime={regime}"
    )

    # And the contract: action ∈ {BUY,SELL,HOLD}, confidence finite ∈ [0,1].
    action, conf = out1
    assert action in {"BUY", "SELL", "HOLD"}
    assert isinstance(conf, float) and math.isfinite(conf)
    assert 0.0 <= conf <= 1.0


# --------------------------------------------------------------------------
# Property 5 — load_json on missing path returns the default verbatim
# --------------------------------------------------------------------------

# Defaults must be picklable & json-comparable; reuse json_scalar plus dict/list.
default_value = st.one_of(
    json_scalar,
    st.lists(json_scalar, max_size=5),
    json_dict,
)


@given(
    name=st.text(
        alphabet=st.characters(
            whitelist_categories=("Ll", "Lu", "Nd"),
            whitelist_characters="_-",
        ),
        min_size=1, max_size=30,
    ),
    default=default_value,
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_load_json_default_preserved_on_missing(tmp_path, name, default):
    """Universal truth: load_json(missing_path, default=X) == X.

    Contract: load_json must NEVER raise on a missing file — it returns the
    default. Callers across the codebase rely on this (~150 call sites);
    a regression here cascades into silent crashes everywhere.
    """
    target = tmp_path / f"definitely-missing-{name}.json"
    assert not target.exists(), "hypothesis seed collision; skip"

    result = load_json(str(target), default=default)
    assert result == default, (
        f"missing-file default not preserved: got {result!r}, want {default!r}"
    )
    # And: the missing file must not be created as a side effect.
    assert not target.exists(), "load_json created the file as a side effect"
