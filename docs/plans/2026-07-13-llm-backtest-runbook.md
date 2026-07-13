# LLM Backtest Run — 2026-07-13 — Runbook for any agent

READ THIS FIRST if you are a fresh agent/session picking this up.

## What is running and why

Historical backtest of 4 local LLMs on herc2's RTX 3080 to decide, on
evidence, whether **Fin-R1** (new finance-tuned model) and/or **phi4_mini**
should replace/join the incumbent LLM voters (ministral3 57.6%, qwen3
58.9% live 1-day directional accuracy — from herc2 signal_log.db,
n=6748/4271). phi4_mini had ZERO recorded votes ("shadow since Jun 1" was
never wired), so the user chose replay-on-history instead of weeks of live
shadowing. Same prompts, same snapshots → clean head-to-head.

- **Launched:** 2026-07-13 ~10:40 CEST from the Deck.
- **Expected done:** ~17:30-18:00 CEST same day (3,840 inferences × ~6.5s).
- **Where:** herc2 (Windows, Tailscale `100.78.196.30`), scheduled task
  `PF-LLMBacktest` running `scripts/win/llm-backtest-run.ps1` →
  `scripts/llm_backtest.py`. Survives SSH disconnects.
- **Output:** `Q:\finance-analyzer\data\llm_backtest_results.jsonl`
  (append-only, one JSON per inference). Log:
  `Q:\finance-analyzer\data\llm_backtest_run.log`.
- Models: ministral3, qwen3, phi4_mini, fin_r1 · window 2026-02-01→07-11,
  8h step, BTC+ETH · outcome = +1d realized move.

## Commands (run from the Deck)

```bash
~/projects/finance-analyzer/scripts/deck/run-llm-backtest-on-herc.sh --status   # progress
~/projects/finance-analyzer/scripts/deck/run-llm-backtest-on-herc.sh --results  # fetch + score table
~/projects/finance-analyzer/scripts/deck/run-llm-backtest-on-herc.sh            # (re)launch — RESUMES, skips done pairs
```

## Failure scenarios

| Scenario                                 | Action                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Claude session died / fresh agent        | Nothing lost. `--status` to check; if task no longer running and results incomplete, rerun orchestrator (resumes).                                                                                                                                                                                                                    |
| herc2 crashed / rebooted mid-run         | One-shot task does NOT auto-restart. Rerun orchestrator from Deck (wakes herc2, pulls nothing new, relaunches, resumes).                                                                                                                                                                                                              |
| Gate flag stuck                          | If run died hard, `Q:\finance-analyzer\data\local_llm.disabled.backtest-lifted` may exist instead of `local_llm.disabled`. Restore: `ssh herc2@100.78.196.30 "cd /d Q:\finance-analyzer && ren data\local_llm.disabled.backtest-lifted local_llm.disabled"`. NEVER leave the gate open — herc2 must stay LLM-paused outside this run. |
| Run overrunning past 18:00               | Fine — resumable. `--status` shows `[model] i/N` progress lines. qwen3/fin_r1 are the slow phases (~15-20s per query).                                                                                                                                                                                                                |
| Deck rebooted                            | Irrelevant to the run (it lives on herc2). Deck trading stack auto-restarts via systemd (lingering on).                                                                                                                                                                                                                               |
| Binance/network hiccup during case build | Case building happens once at task start; if it died there, log shows it — just relaunch.                                                                                                                                                                                                                                             |

## When it finishes

1. `--results` → score table (directional acc / abstain% / errors per model).
2. Judgment rule agreed with user: promote a model only if its directional
   accuracy on B/S votes beats incumbents on same-sample comparison with
   sensible n (hundreds of votes). Abstain-rate and parse-error rate matter
   too — a model that never votes is useless as a voter.
3. Append results table to `docs/plans/2026-07-12-llm-audit-and-research.md`,
   update memory (`finance-analyzer.md`).
4. herc2 afterwards: sleep was disabled for the run — user preference is
   herc2 off/asleep when idle. `ssh herc2@100.78.196.30 "shutdown /s /t 0"`
   after confirming gate flag restored (ps1 finally-block does it on normal
   completion).
5. NEXT decision still open (see 2026-07-12 doc): where LLM voters run
   long-term — Deck loop is no-LLM by design, herc2 loops disabled. Leading
   option: herc2 llama-server serving Deck loop over Tailscale.

## Wider context

Full system migration state: `docs/plans/2026-07-12-deck-migration-todo.md`
(everything runs on Deck except LLM/GPU; herc2 = LLM box, off by default).
Model research + rankings: `docs/plans/2026-07-12-llm-audit-and-research.md`.
Memory: `~/.claude/projects/-home-deck/memory/finance-analyzer.md`.
Pending unrelated: BankID re-auth on Deck for Avanza orders (user does it
when ready — metals loop in monitoring mode until then).
