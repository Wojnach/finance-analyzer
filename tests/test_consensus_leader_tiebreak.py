"""Correlation-group leader selection must be deterministic.

`_weighted_consensus` picks one "leader" per correlation group — the
highest-accuracy signal that is actively voting — and applies a follower
penalty (0.15x-0.35x depending on the cluster) to everyone else in that group.

The selection was `max(active_in_group, key=_leader_accuracy_key)` where
`active_in_group` is a **set** and the key returned a bare float. On an exact
accuracy tie, `max` returns the first maximal element in *set iteration order*,
which Python randomizes per process via PYTHONHASHSEED. The loser then eats the
follower penalty, and when the tied signals vote in OPPOSITE directions that
flips the consensus.

Measured on 2026-08-20 with bb=BUY and rsi=SELL, both at 0.60 accuracy, both in
`momentum_cluster`:

    PYTHONHASHSEED=0  leader=rsi  -> SELL 0.8415
    PYTHONHASHSEED=1  leader=bb   -> BUY  0.7638
    PYTHONHASHSEED=2  leader=bb   -> BUY  0.7638
    PYTHONHASHSEED=3  leader=bb   -> BUY  0.7638

So the live loop was not reproducible across restarts, and
tests/test_signal_hold_bias_reduction.py::TestSoftConsensusDampening
::test_all_soft_slate_has_lower_weighted_conf_than_one_strong passed or failed
depending on the interpreter's hash seed.

Note the sibling `_topn_accuracy_key` in the same function already returned
`(base, s)` for exactly this reason — the pattern existed and was missed here.
"""

import subprocess
import sys
import textwrap

from portfolio.signal_engine import (
    MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER,
    _group_leader_key,
)

# Reproduces the live tie: bb and rsi share momentum_cluster at equal accuracy.
_PROBE = textwrap.dedent("""
    from portfolio.signal_engine import (
        _weighted_consensus,
        EMA_DEAD_ZONE_SOFT_CONF,
        BB_INSIDE_SOFT_CONF,
        MACD_DEAD_ZONE_SOFT_CONF,
    )

    acc = {
        k: {
            "accuracy": 0.60, "total": 1000,
            "buy_accuracy": 0.60, "total_buy": 500,
            "sell_accuracy": 0.60, "total_sell": 500,
        }
        for k in ("ema", "bb", "macd", "rsi")
    }
    action, conf = _weighted_consensus(
        {"ema": "BUY", "bb": "BUY", "macd": "BUY", "rsi": "SELL"},
        acc,
        regime="ranging",
        soft_confidences={
            "_soft_conf_ema": EMA_DEAD_ZONE_SOFT_CONF,
            "_soft_conf_bb": BB_INSIDE_SOFT_CONF,
            "_soft_conf_macd": MACD_DEAD_ZONE_SOFT_CONF,
        },
    )
    print(f"{action} {conf}")
    """)

_SEEDS = ["0", "1", "2", "3", "4", "5"]


def _run_probe(seed, cwd):
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONPATH": str(cwd)},
        cwd=str(cwd),
        timeout=180,
    )


class TestConsensusIsHashSeedIndependent:
    def test_same_slate_gives_same_verdict_across_hash_seeds(self, pytestconfig):
        """The end-to-end property: no PYTHONHASHSEED may change the verdict."""
        root = pytestconfig.rootpath
        results = {}
        for seed in _SEEDS:
            proc = _run_probe(seed, root)
            assert (
                proc.returncode == 0
            ), f"probe failed at seed {seed}: {proc.stderr[-1500:]}"
            results[seed] = proc.stdout.strip()

        distinct = set(results.values())
        assert len(distinct) == 1, (
            "consensus depends on PYTHONHASHSEED — a correlation-group leader "
            f"tie is being broken by set iteration order: {results}"
        )


class TestGroupLeaderKey:
    """The tie-break itself, unit-tested."""

    def test_orders_primarily_by_accuracy(self):
        acc = {
            "a": {"accuracy": 0.70, "total": 10},
            "b": {"accuracy": 0.55, "total": 10},
        }
        assert _group_leader_key("a", acc, False) > _group_leader_key("b", acc, False)

    def test_breaks_an_accuracy_tie_on_sample_count(self):
        """Equal accuracy: trust the estimate measured on more data."""
        acc = {
            "a": {"accuracy": 0.60, "total": 50},
            "b": {"accuracy": 0.60, "total": 5000},
        }
        assert _group_leader_key("b", acc, False) > _group_leader_key("a", acc, False)

    def test_breaks_a_full_tie_on_name(self):
        """Identical accuracy AND samples must still order deterministically."""
        acc = {
            "bb": {"accuracy": 0.60, "total": 1000},
            "rsi": {"accuracy": 0.60, "total": 1000},
        }
        assert _group_leader_key("rsi", acc, False) > _group_leader_key(
            "bb", acc, False
        )

    def test_max_over_a_set_is_now_stable(self):
        acc = {
            "bb": {"accuracy": 0.60, "total": 1000},
            "rsi": {"accuracy": 0.60, "total": 1000},
        }
        for candidates in ({"bb", "rsi"}, {"rsi", "bb"}):
            winner = max(candidates, key=lambda s: _group_leader_key(s, acc, False))
            assert winner == "rsi"

    def test_applies_the_macro_window_downweight(self):
        from portfolio.signal_engine import MACRO_WINDOW_DOWNWEIGHT_SIGNALS

        sig = next(iter(MACRO_WINDOW_DOWNWEIGHT_SIGNALS))
        acc = {sig: {"accuracy": 0.80, "total": 100}}
        plain = _group_leader_key(sig, acc, False)
        damped = _group_leader_key(sig, acc, True)
        assert damped[0] == plain[0] * MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
        assert damped < plain

    def test_missing_stats_default_to_neutral(self):
        assert _group_leader_key("nope", {}, False) == (0.5, 0.0, "nope")

    def test_tolerates_none_and_junk_stats(self):
        """A half-written accuracy cache must not crash leader selection."""
        acc = {"a": None, "b": {"accuracy": None, "total": None}}
        assert _group_leader_key("a", acc, False)[0] == 0.5
        assert _group_leader_key("b", acc, False) == (0.5, 0.0, "b")
