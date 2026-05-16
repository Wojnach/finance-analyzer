# /fgl: fix-everything session — 2026-05-16

## Goal

Close LLM shadow-enrollment tail. Last session shipped Steps 1-6 of
`/root/.claude/plans/no-we-don-t-these-glowing-ullman.md`; this session
tackles open items from the post-ship status report.

## Items in scope

| # | Item | Batch | Effort |
|---|---|---|---|
| 1 | Join-rate investigation (resolved during exploration — temporal, not bug) | A | XS |
| 2 | Stale shadow audit (5 entries >30d) | A | S |
| 3 | Frontend tile for `/api/llm-leaderboard` | B | S |
| 4 | `finance_llama` real GGUF inference (flip `_FEATURE_AVAILABLE=True`) | C | M |

Out of scope (future session):
- `cryptotrader_lm` real PEFT LoRA inference
- `meta_trader` Qwen2-36L unsloth + upstream-vote wiring
- `custom-trading-lora.gguf` provenance audit
- qwen3 verdict-diversity probe (today's data shows it working — 7 unique probs, 95% HOLD is real ranging-tape behavior)
- Brier reliability binning UI

## Findings during exploration

### Join rate is temporal

Ran `portfolio.llm_outcome_backfill.backfill()` manually:

```
processed: 14038
written: 209
skipped_already_present: 8978
skipped_too_recent: 4818      # 1d horizon hasn't elapsed
skipped_missing_price: 3
skipped_bad_row: 30
```

Post-backfill join rates:
- old voters (ministral/qwen3/sentiment/news_event/claude_fundamental/forecast): **75%**
- new sentiment splits (trading_hero/finbert/fingpt): **27%** (most rows still <24h)
- scaffolds (cryptotrader_lm/finance_llama/meta_trader): **22%**

Asymptotic behavior. As more time passes, new voters converge to ~75%.
`PF-LLMBackfill` already scheduled hourly. No fix needed beyond
documenting the expected curve.

### Stale shadow audit

| Signal | Real state | Action |
|---|---|---|
| `credit_spread_risk` | 0 samples, not in `_LLM_SIGNALS`, force-HOLD | **Retire** |
| `crypto_macro` | Same | **Retire** |
| `finbert` | 253 samples (post-split), stale `entered_shadow_ts=2026-04-09` | **Refresh** to 2026-05-15 |
| `fingpt` | 119 samples, same stale ts | **Refresh** to 2026-05-15 |
| `kronos` | 0 samples, un-retired 2026-04-21, never emitted | **Leave + document** — separate subprocess-reliability work-stream |

### Frontend tile

`dashboard/static/index.html` is the SPA. Tiles live in `dashboard/static/js/views/*.js`.
Pattern: each view exports a render function returning innerHTML string.
Add `views/llm_leaderboard.js` + router wire + home insert.

### finance_llama inference

Plan:
1. Register `finance-llama-8b` in `portfolio/llama_server._MODEL_CONFIGS`
2. In `portfolio/signals/finance_llama.py`: flip `_FEATURE_AVAILABLE=True`
3. Build Ministral-style prompt
4. Call `query_llama_server("finance-llama-8b", prompt, stop=["</s>"])`
5. Parse action + confidence via same regex fallback ministral/qwen3 use
6. Emit `log_vote()` via `derive_probs_from_result()`

Cycle cost target: 3-4s on GPU. `cycle_modulo=3` per existing registry entry.
Falls back to abstention if llama-server unavailable or VRAM tight (Plex transcode).

## Batches

- **A** — registry hygiene (3 files: registry JSON + helper + docs)
- **B** — frontend tile (3 files: view, router, home wire)
- **C** — finance_llama inference (3 files: signal, llama_server config, tests)

## Verification

Each batch: `pytest -n auto` (changed-files first, then full), codex review
xhigh, fix P1/P2, merge to main, push via cmd.exe, restart loops.

## Risks

| Risk | Mitigation |
|---|---|
| `finance-llama-8b-gguf` not loadable by current llama-server build | Wrap try/except, fallback abstention. Test `model_load_safe()` first. |
| Real inference inflates cycle p95 above 90s | `cycle_modulo=3` cap; smoke-test cycle time |
| New `entered_shadow_ts` breaks accuracy-history continuity | Registry-only field; doesn't touch outcome backfill or accuracy_cache |
| Codex P1 in retired signals | Retired = `status="retired"` only; vote remains force-HOLD — no behavioural change |

## Out-of-band

User said system has been running. Loops restart required after C ships
(touches signal_engine wiring). A & B are dashboard/registry only — no
restart needed.
