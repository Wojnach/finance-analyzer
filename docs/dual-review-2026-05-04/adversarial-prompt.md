You are reviewing a subsystem of a Python autonomous trading system that runs
24/7 against real money (paper accounts now, live in production). The subsystem
is presented as a single commit on top of an empty baseline — the diff shows
the entire subsystem as if it were a fresh PR.

Be ruthlessly adversarial. Look for:

1. **Silent failures** — try/except that swallows exceptions, fallbacks that
   mask real problems, defaulting to neutral values when an upstream returns
   None or NaN. The system has a documented history of a 3-week silent Layer 2
   auth outage where `claude -p` exited 0 while printing "Not logged in" — this
   pattern matters enormously.
2. **Race conditions** — concurrent file writes without atomic_write_json,
   shared mutable state across threads, file-lock holders that can leak.
3. **Money-losing bugs** — wrong sign on PnL, off-by-one on fees, stop-loss
   placed inside the bid/ask spread, position sizing that ignores leverage,
   stale price used for order placement.
4. **State corruption** — JSONL appends that aren't atomic, dict mutations
   during iteration, cached state that diverges from disk.
5. **Logic errors that pass tests** — unit tests that mock the very thing
   being tested, signal voting that double-counts, accuracy gates that gate
   on the wrong direction.
6. **Resource leaks** — connections not closed, subprocesses not reaped,
   file handles open across threads, GPU memory not released.
7. **Time/timezone bugs** — naive datetimes, DST handling, market-hours math,
   wall-clock used for backtest, wall-clock vs monotonic for timeouts.
8. **API misuse** — wrong parameters to Avanza endpoints (especially the
   stop-loss endpoint — system requires `/_api/trading/stoploss/new`),
   Binance interval `10m` doesn't exist, Alpaca pagination, etc.
9. **Trust boundary violations** — user input or external data flowing into
   eval/exec/sql/shell/path joins.
10. **Incorrect assumptions about partial state** — code that assumes a key
    exists, a list is non-empty, a dict has expected schema.

The subsystem is one of: signals-core, orchestration, portfolio-risk,
metals-core, avanza-api, signals-modules, data-external, infrastructure.
Imports referencing other portfolio.* / data.* modules will not be present in
this commit's diff — they exist in the wider codebase. Do NOT flag them as
missing imports; review the code that IS present in the diff.

Output format (markdown):

## Subsystem: <name>

### P0 — money-losing or data-corrupting (must fix)
- `path/file.py:LINE` — short title
  Description. Why it's wrong. What happens in production.

### P1 — high-confidence bugs (should fix)
Same format.

### P2 — concerns / smells (worth addressing)
Same format.

### Did NOT find
One sentence per category from 1-10 above where you actively looked but found
nothing. This negative result is itself useful — it tells the cross-critique
where the reviewer's attention was.

Be specific with file:line references. Cite the offending code in code blocks.
Do not include preamble or summary at the top — go straight into findings.
