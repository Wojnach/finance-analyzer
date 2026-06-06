"""Prophecy — daily AI price-prediction system.

A standalone, separate subsystem (NOT to be confused with
``portfolio.prophecy``, which is the unrelated macro-beliefs store). Built
2026-06-06; see ``docs/superpowers/specs/2026-06-06-prophecy-design.md``.

Pipeline (all steps but the Claude run spend ZERO tokens):

    prep.py    -> data/prophecy_runs/context_<date>.json   (gather stored signals + live px)
    claude -p  -> data/prophecy_runs/raw_<date>.json        (deep-research + per-instrument fuse)
    publish.py -> prediction_journal.jsonl + latest.json    (validate + snapshot)
    outcomes.py-> accuracy.jsonl + accuracy.json            (score matured horizons)
    cost.py    -> cost_log.jsonl + latest.json.cost_summary (token spend tracking)

Physical data dir is ``data/prophecy_runs/`` (NOT ``data/prophecy/``) to keep a
hard separation from the load-bearing ``data/prophecy.json`` macro-beliefs file.
"""

__version__ = "0.1.0"
