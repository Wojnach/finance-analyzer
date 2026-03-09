# Worktree README

## Branch

- `local-llm-accuracy`

## What changed

- Added a research-backed plan for improving local LLM accuracy at [`docs/plans/2026-03-09-local-llm-accuracy-plan.md`](Q:/finance-analyzer/.worktrees/local-llm-accuracy/docs/plans/2026-03-09-local-llm-accuracy-plan.md).
- Gated `Ministral` votes by per-ticker historical accuracy.
- Gated `Chronos` / `Kronos` sub-signals individually before composite forecast voting.
- Switched the `Ministral` subprocess wrapper to the repo-managed inference script.
- Tightened `Ministral` output parsing and added targeted tests.
- Added `--local-llm-report` for local-model health, accuracy, and gating summaries.

## How to run

- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests/test_local_llm_accuracy.py`
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests/test_portfolio.py -k "Ministral"`
- `Q:\finance-analyzer\.venv\Scripts\python.exe -m pytest -q tests/test_forecast_accuracy_gating.py -k "ComputeForecastWithGating or KronosDisabledDefault or AccuracyWeightedVote or VolatilityGate or RegimeInVote"`

## Notes

- No live trading process was touched.
- Model upgrades such as `chronos-bolt-small` or newer Mistral small models are documented in the plan but not enabled by default in code.
