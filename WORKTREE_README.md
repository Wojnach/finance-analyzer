# Worktree README

## Branch

- `local-llm-accuracy-inrepo`

## What changed

- Added the local-LLM accuracy plan in `docs/plans/2026-03-09-local-llm-accuracy-plan.md`.
- Gated `Ministral` votes by per-ticker historical accuracy.
- Gated `Chronos` and `Kronos` sub-signals before composite forecast voting.
- Added `--local-llm-report` plus daily export to `data/local_llm_report_latest.json` and `data/local_llm_report_history.jsonl`.
- Kept the newer `main` implementations for dashboard APIs, native Ministral inference, and post-cycle metals precompute while merging the accuracy/reporting behavior on top.

## How to run

- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests/test_local_llm_accuracy.py`
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests/test_local_llm_report.py`
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests/test_portfolio.py -k "Ministral"`
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests/test_forecast_accuracy_gating.py -k "ComputeForecastWithGating or KronosDisabledDefault or AccuracyWeightedVote or VolatilityGate or RegimeInVote"`
- `Q:\finance-analyzer\.venv\Scripts\python.exe portfolio\main.py --export-local-llm-report 30`

## Notes

- No live trading path was intentionally broadened by this merge.
- Model upgrades such as `chronos-bolt-small` or newer Mistral small variants remain documented rather than enabled by default.
