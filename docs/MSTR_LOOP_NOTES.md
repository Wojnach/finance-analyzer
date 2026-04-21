# MSTR Loop — Phase B (shadow) operational notes

**Status (as of 2026-04-21):** Phase B shadow mode. `config.PHASE = "shadow"`
in `portfolio/mstr_loop/config.py`. No live Avanza orders placed by this
loop; decisions are logged to `data/mstr_loop_shadow.jsonl` with a
hypothetical `SHADOW_NOTIONAL_SEK = 30_000` position size.

## Why the shadow log file may be missing

`data/mstr_loop_shadow.jsonl` is created on the first in-session
BUY / SELL / PARTIAL_SELL decision, not at loop startup. If the file is
absent, the loop is either:

1. **Outside the session window** (NASDAQ 15:30 – 22:00 CET) — the loop
   exits early at `portfolio/mstr_loop/loop.py:83-87` with
   `outside_session_window` and writes to `data/mstr_loop_poll.jsonl`.
   This is the normal pre-open / post-close state. Nothing to do.
2. **Inside the session but holding HOLD** — no strategy has produced a
   BUY/SELL yet this session. The loop is running, `mstr_loop_poll.jsonl`
   will have recent entries, but `mstr_loop_shadow.jsonl` stays empty
   until a first decision fires.
3. **Not running** — check `cmd.exe /c "schtasks /query /tn PF-MSTRLoop"`.
   If absent, the loop hasn't been scheduled and needs manual start.

## How to verify phase at runtime

At startup the loop logs one line declaring its phase. Grep agent.log
for `mstr_loop:phase=`. If the line is missing, the loop predates the
2026-04-21 startup-marker change.

## Phase transition

Phase B → Phase A (live) requires:

- `config.PHASE = "live"` (or `MSTR_LOOP_PHASE=live` env var).
- At least 90 days of shadow equity curve data showing positive
  expectancy net of simulated fees.
- Human approval. Do NOT flip the phase unattended — see
  `scripts/mstr_loop_scorecard.py` which pairs shadow BUY/SELL into
  round-trips and reports win-rate + expectancy.

Transition is reversible: flipping back to `"shadow"` immediately stops
new live orders; existing positions are held to their natural exits.

## Reading the shadow log

```bash
python scripts/mstr_loop_scorecard.py
```

Shows per-strategy win rate, mean R multiple, and equity curve. Tail the
raw log for debug:

```bash
tail -f data/mstr_loop_shadow.jsonl
```
