# Claude critique of codex findings — signals-modules

## Verdicts

- [P2] Avoid host-specific Kronos subprocess paths — `Q:\finance-analyzer\portfolio\signals\forecast.py:107-111`
  Verdict: CONFIRMED
  Reason: Lines 107-108 hard-code `Q:\finance-analyzer\.venv\Scripts\python.exe` and `Q:\models\kronos_infer.py`; lines 110-111 hard-code `/home/deck/models/.venv/bin/python` and `/home/deck/models/kronos_infer.py`. The correct pattern (used in qwen3_signal.py, ministral_signal.py) is dynamic derivation via `repo_root = Path(__file__).resolve().parent.parent` followed by platform-specific path construction. Any machine with a different layout will have Kronos fail silently and trip the circuit breaker indefinitely.

- [P2] Declare SciPy for the realized_skewness signal — `Q:\finance-analyzer\portfolio\signals\realized_skewness.py:25`
  Verdict: CONFIRMED
  Reason: Line 25 imports `from scipy import stats` at module level. Neither `pyproject.toml` (lines 6-13, dependencies list) nor `requirements.txt` (lines 1-21) declare scipy. When load_signal_func() attempts to import this module on a clean install, the import fails, and realized_skewness defaults to HOLD forever, silenced after initial warning. scikit-learn depends on scipy transitively, but scipy is not explicitly declared and should be for reliability and clarity.

## New findings (mine)

- [P3] Orphaned signal in tickers.py — `Q:\finance-analyzer\portfolio\tickers.py:92`
  `realized_skewness` is added to DISABLED_SIGNALS with a note "KILLED 2026-04-29: 33.3% at 1d (90 sam). Below coin flip." However, it is still registered in signal_registry.py:218-219 and appears in the APPLICABLE_SIGNALS dict at line 303. This creates a mismatch: the signal is disabled in one place but registered and callable in another. Either remove the registration entirely or move the signal from DISABLED_SIGNALS to active, not both. Current state allows stale references.

## Summary

- Confirmed: 2
- Partial: 0
- False-positive: 0
- New: 1
