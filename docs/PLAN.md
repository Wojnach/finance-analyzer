# PLAN — Enroll all LLMs through shadow registry (2026-05-15)

## Why

`data/accuracy_cache.json` shows only 2 LLM voters today (ministral 58%, qwen3 60%). Three more LLM signals are wired but force-disabled (sentiment 46%, forecast 47%, claude_fundamental 58%). Four sentiment sub-models (CryptoBERT, FinBERT, Trading-Hero, FinGPT) are averaged inside one aggregate — individual accuracy never measured. Three more models on disk have no wrappers at all: `finance-llama-8b-gguf`, `cryptotrader-lm`, `custom-meta-trader`.

The shadow registry infra (`portfolio/shadow_registry.py`, `data/shadow_registry.json`, `scripts/review_shadow_signals.py`) exists but only 5 entries are registered. The probability log (`portfolio/llm_probability_log.py`, central call at `signal_engine.py:3656`) accepts only 6 signal names via `_LLM_SIGNALS`.

Goal: route every LLM-class model on disk through the shadow → measure → promote pipeline this session. Keep cycle near 60s by throttling expensive shadows. Don't change vote weighting until shadow data justifies promotion.

## Pre-flight (2026-05-15 14:00 UTC)

Three pre-existing critical errors noted, all OUTSIDE this session's scope:
- accuracy_degradation on macro_regime/momentum_factors/structure (non-LLM signals)
- accuracy_degradation on forecast::chronos_24h — VALIDATES re-enabling forecast as shadow not voter
- contract_violation on Layer 2 silent fail at 13:01

None touched by this session.

## Scope this session

| Batch | Files | Risk |
|---|---|---|
| 1 | `portfolio/tickers.py`, `data/shadow_registry.json`, `portfolio/signal_engine.py` (small gate) | Low |
| 2 | `portfolio/sentiment.py`, `portfolio/llm_probability_log.py`, `data/shadow_registry.json` | Medium |
| 3 | `portfolio/signals/finance_llama.py` (new), `portfolio/signals/cryptotrader_lm.py` (new), `portfolio/signals/meta_trader.py` (new), `portfolio/signal_registry.py`, `data/shadow_registry.json` | Low |
| 4 | `portfolio/shadow_registry.py`, `portfolio/signal_engine.py` | Low |

## Hard rules

- **No vote-weight changes.** Every new signal enters shadow, not active vote pool. Existing 2-LLM consensus (ministral+qwen3) unchanged.
- **Scaffold > broken inference.** Where a new model lacks a verified loader (finance-llama, cryptotrader-lm, meta_trader), the wrapper returns HOLD with `feature_unavailable=True` indicator and a TODO. Shadow registry entry created so future inference fill-in is incremental.
- **Cycle budget stays near 60s.** Cheap CPU shadows run every cycle. Expensive GGUF shadows gated by `cycle_modulo` (default 3 for 8B models, 5 for Qwen2-36L).
- **Existing tests must stay green.** Worktree lacks `config.json` symlink — run only targeted tests inside worktree. Full suite runs in main after merge.

## What could break

1. **`_LLM_SIGNALS` expansion** without matching `extra_info` keys in `signal_engine.py:3654-3666` → silent log skip. Mitigation: every new signal name MUST set `extra_info[f"{name}_confidence"]` (and optionally `_indicators`) at compute site.
2. **Shadow signal still voting.** `signal_engine` includes a signal in `votes` regardless of shadow status. Mitigation: add explicit drop step that consults `shadow_registry.load_registry()` after vote dict is built, before consensus.
3. **Cycle modulo bug skips signal forever.** Persisted `cycle_phase` could desync. Mitigation: derive cycle counter from monotonic UTC minute, not persisted state.
4. **Probability sum drift on derived-prob path.** Already normalized in `derive_probs_from_result()`. Leave alone.

## Execution order

1. Worktree + plan commit
2. Batch 1: re-enable as shadows. Test. Commit.
3. Batch 2: split sentiment. Test. Commit.
4. Batch 3: scaffold 3 new wrappers. Test. Commit.
5. Batch 4: cycle throttle. Test. Commit.
6. Adversarial review. Address P1/P2.
7. Targeted pytest pass.
8. Merge to main. Push via cmd.exe.
9. Restart loops.
10. 1h trail.

## Verification per batch

| Batch | Pass criteria |
|---|---|
| 1 | shadow_registry.json has 3 new entries documenting prior disable reason. signal_engine drops shadow-status signals from vote pool but keeps in raw_votes/log path. |
| 2 | `_LLM_SIGNALS` includes 4 new names. Sentiment call emits 5 rows (legacy aggregate + 4 sub-voters) to llm_probability_log per invocation. |
| 3 | 3 new modules import clean. Each returns valid result dict with `action="HOLD"`, `confidence=0.0`, indicator `feature_unavailable=True`. Registered in signal_registry. Shadow entries present. |
| 4 | `cycle_modulo` skip path covered by unit test. Cheap shadows modulo=1, GGUF modulo=3, meta-trader modulo=5. |

## Post-merge 1h trail

1. `tail -F data/health_state.json` — abort if mean cycle_ms > 120000 over 5 min window.
2. `tail -F data/critical_errors.jsonl` — any new signal-name entry → set status="retired" via `shadow_registry.resolve_shadow()`.
3. After ~30 min: count rows by signal in `data/llm_probability_log.jsonl`. Any new signal with 0 rows = silent failure → retire.
4. Spot-check `data/agent_summary.json` for 3 tickers, confirm new signal names appear with real probs.

## Out of scope

- Real GGUF inference for the 3 new wrappers — scaffolds return HOLD. Inference work scheduled as follow-up.
- Brier/reliability binning UI.
- `custom-trading-lora.gguf` audit — unknown provenance, untouched.
