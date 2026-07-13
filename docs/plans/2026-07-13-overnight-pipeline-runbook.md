# Overnight LLM pipeline (2026-07-13 → 14) — runbook for any agent

Fresh agent: read this + `2026-07-13-llm-backtest-runbook.md` (base
mechanics) + `2026-07-12-llm-audit-and-research.md` (results so far).

## Decision policy (user, 2026-07-13)

- **≥60% directional accuracy on a cell (ticker × interval × horizon) or
  the model gets no GPU time.** Below 60% = coin flip = not worth the
  electricity.
- Evaluate ONE model at a time across many conditions; keep only earners.
- qwen3/ministral3: benched for crypto (coin flip there); may earn
  per-ticker slots via `_DISABLED_SIGNAL_OVERRIDES` in signal_engine.py
  if any sweep cell clears 60%.
- phi4_mini already holds one qualifying cell: BTC/ETH @ 1h-context/1d
  horizon = 66.7% (n=108). fin_r1 FAILED crypto (44.8%) — voter ban;
  awaiting its metals cells.

## Pipeline state + what happens next

1. **RUNNING: metals backtest** on herc2 — phi4_mini + fin_r1 on
   XAU-USD/XAG-USD, 1h interval, ~1,930 rows into
   `Q:\finance-analyzer\data\llm_backtest_metals.jsonl`. ETA ~23:30.
   Deck monitor task watches for completion (stall + finish alerts).
2. **THEN: launch phi4 sweep** (from Deck):
   `MODEL=phi4_mini nohup scripts/deck/run-llm-sweep.sh > /tmp/sweep-phi4.log 2>&1 &`
   Sequential phases 15m (from 2026-05-01), 4h, 1d (from 2026-02-01) ×
   BTC/ETH/XAU/XAG → `data\llm_sweep_phi4_mini.jsonl` on herc2. ~4.5h.
   DO NOT launch while another PF-LLMBacktest phase is mid-run — the
   launcher force-recreates the same scheduled task.
3. **THEN (parallel with sweep, safe VRAM-wise): install remote LLM server**
   on herc2: run `scripts/win/install-llama-remote-task.ps1` via ssh
   (admin session): PF-LlamaRemote task, llama-server phi4_mini :8788,
   firewall scoped to Deck TS IP 100.75.67.98. Deck side is ALREADY
   configured + tested (config.json `local_llm.remote`, whitelist
   phi4_mini only; unreachable → 3s abstain; gate flag still blocks all
   local inference). Verify from Deck:
   `curl -s -m 5 http://100.78.196.30:8788/health`
   then watch one loop cycle: `journalctl --user -u pf-dataloop -f` —
   phi4 queries should return text, ministral/qwen abstain.
4. **MORNING: score everything:**
   - metals: `OUT='data\llm_backtest_metals.jsonl' scripts/deck/run-llm-backtest-on-herc.sh --results`
   - sweep: `.venv/bin/python scripts/llm_backtest.py --score /tmp/llm_sweep_phi4_mini.jsonl`
     (sweep script auto-fetches + scores at the end; file also on herc2)
   - Apply the 60% rule per cell → decide phi4's slots; queue fin_r1
     sweep next if its metals cells look alive, else qwen3.
5. herc2 sleep is DISABLED during runs. When pipeline idle: either
   shutdown herc2 (`ssh herc2@100.78.196.30 "shutdown /s /t 0"`) if the
   remote-LLM serving isn't wanted yet, or leave awake serving :8788.

## Failure handling (same patterns as before)

- Any backtest phase: resumable — rerun the same launch command, done
  (model, interval, at, ticker) rows are skipped.
- Stuck task on herc2: `schtasks /end /tn PF-LLMBacktest`, kill stray
  `llama-server.exe`/`python.exe` with CommandLine filter (never blanket
  taskkill python — other tasks live there).
- Gate flag: normal completion restores `data/local_llm.disabled` on
  herc2 (ps1 finally-block). Verify after crashes; the
  `*.backtest-lifted` rename is the lifted state.
- Monitors: known quirk — appended transcript log keeps old 'RUN
  COMPLETE' markers; completion = marker COUNT increments, not presence.
- Session-limit death: everything above is idempotent; re-read this doc.

## Where results feed back

Promotion wiring when a model earns slots: (a) add model to
`local_llm.remote.models` whitelist in Deck config.json, (b) signal-side:
per-ticker enable via `_DISABLED_SIGNAL_OVERRIDES` / DISABLED_SIGNALS in
portfolio/tickers.py + signal_engine.py (existing pattern, see Realized
Skewness → XAU example), (c) restart pf-dataloop. phi4_mini has no signal
module registered as VOTER yet — it exists as shadow entry
(signal_registry.py:424); promotion = move out of DISABLED_SIGNALS for
the earned tickers only.
