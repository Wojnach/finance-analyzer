# PLAN — Swedbank book: monitoring ledger + trajectories + dashboard tab

> Supersedes the 2026-06-10/11 audit-fix campaign (completed; archived to
> `docs/plans/2026-06-10-audit-fix-campaign.md`).

Campaign date: 2026-07-31. Worktree `.worktrees/swedbank-book`, branch
`feature/swedbank-book`. Protocol: `/fgl` → `docs/GUIDELINES.md`.

## Privacy constraint — read before adding anything to this doc

**`github.com/Wojnach/finance-analyzer` is a PUBLIC repository** (verified 2026-07-31:
anonymous HTTPS GET returns 200). This subsystem handles the operator's _real_ brokerage
positions, unlike the simulated Patient/Bold portfolios.

Rules, decided by the operator 2026-07-31:

1. **No real position data in git — ever.** No quantities, no cost basis, no account
   totals, no P&L, no account labels. Not in code, not in docs, not in commit messages,
   not in test fixtures.
2. `data/swedbank_book.json` and `data/swedbank_*.json` are **gitignored**. The book is
   generated locally from the operator's own snapshot and never committed.
3. Tests use **synthetic** holdings. Never seed a fixture from the live book.
4. Accounts are referred to in code and docs by index (`account_0/1/2`) or by whatever
   label sits in the local, untracked book file. This document uses A/B/C.
5. Never route this data through `dashboard/export_static.py` or
   `dashboard/static/api-data/` — `.gitignore:32-35` records that path is served with
   **no auth**.

This mirrors existing practice: `data/portfolio_state.json`, `portfolio_state_bold.json`
and the Avanza session files are all already excluded for the same reason.

## Goal

Track three real share accounts (A, B, C — 26 distinct instruments) inside
finance-analyzer. **Monitoring and trajectory calculation only.** The operator executes
every order by hand on Avanza. Nothing in this subsystem may place, modify or cancel an
order.

Deliverables:

1. A durable ledger (`data/swedbank_book.json`, untracked) storing qty + cost basis.
2. A standalone monitoring loop that re-prices it and computes signals/trajectories.
3. A dashboard tab, behind `@require_auth`.

## Environment corrections (Deck migration)

`docs/GUIDELINES.md` still documents the herc2/Windows environment. On this machine:

| GUIDELINES says                                      | Actual                                 |
| ---------------------------------------------------- | -------------------------------------- |
| `.venv/Scripts/python.exe`                           | `.venv/bin/python`                     |
| `cmd.exe /c "cd /d Q:\finance-analyzer && git push"` | plain `git push`                       |
| `schtasks /run /tn PF-DataLoop`                      | `systemctl --user restart pf-dataloop` |
| `scripts/win/install-*-task.ps1`                     | `~/.config/systemd/user/pf-*.service`  |

`pytest-timeout` is not installed — `--timeout=N` is rejected by pytest.

## Method — quantities and cost basis are derived, not entered

The operator's broker export gives price, percent-since-purchase and market value per
line, but **no quantity column**. Both missing fields are recoverable:

```
qty        = value_local / (price * fx)          # fx = 1 for SEK-quoted lines
cost_basis = value_local / (1 + since_purch_pct/100)
```

The FX rate is not assumed — it is **solved**. Sweep candidate rates and choose the one
that drives every USD-quoted line to an integer share count simultaneously. On the
2026-07-31 snapshot a single rate satisfied all of them at once, and the reconstructed
cost basis then reconciled against the broker's stated total to under 2 SEK on a
six-figure book — i.e. rounding. That dual agreement (integer quantities _and_
independent total reconciliation) is what makes the derivation trustworthy rather than
a guess.

Cross-check: the broker's quoted closes for that session matched Alpaca's daily bars for
the same date to within rounding across the sampled US names, confirming the snapshot is
genuine and correctly dated.

**Caveat to carry forward:** the percent-since-purchase is denominated in SEK, so it
embeds FX drift. A USD-true cost basis would need the FX rate at each purchase date,
which the export does not carry. `portfolio/fx_rates.py` could backfill it if trade dates
are ever supplied. Until then, cost basis is SEK-true and USD-approximate — the ledger
must label it as such rather than implying otherwise.

## Data sources

### Primary — Avanza, all 26 instruments

Verified 26/26 on 2026-07-31. One code path, keyed on orderbook ID, covering US
equities, Swedish equities, certificates and warrants identically.

**Quotes** — `GET /_api/market-guide/stock/<ob>/quote` (the `/stock/` path works for
every instrument type). Returns `isRealTime: true` with a current `timeOfLast`, plus
`buy, sell, last, spread, highest, lowest, change, changePercent, totalVolumeTraded,
volumeWeightedAveragePrice, updated`.

**History** — `GET /_api/price-chart/stock/<ob>?timePeriod=<p>&resolution=<r>`

| timePeriod                       | default resolution | bars           |
| -------------------------------- | ------------------ | -------------- |
| `today`                          | minute             | ~420           |
| `one_week`                       | ten_minutes        | ~297           |
| `one_month`                      | hour               | ~214           |
| `one_year`                       | day                | ~251           |
| `three_years` + `resolution=day` | day                | ~753           |
| `five_years` + `resolution=day`  | day                | ~1257          |
| `infinity`                       | month              | ~502 (to 1984) |

`resolution` is lowercase and must appear in that period's `availableResolutions`
(echoed in `metadata.resolution`), else a clean HTTP 400. Response:
`{ohlc: [{timestamp, open, high, low, close, totalVolumeTraded}], metadata, from, to,
previousClosingPrice}`.

This is enough history and enough timeframes for the existing signal engine, and it
covers the Stockholm instruments — including a thin warrant — which no other configured
source can see.

> An earlier draft claimed Avanza quotes were delayed. That was wrong: it read the
> _search-index_ price rather than the quote endpoint, and compared against an Alpaca
> sample taken 20 minutes earlier during a session where one holding ranged ~8% intraday.

### Fallback — Alpaca, US names only

When the Avanza session is unavailable, the 19 US instruments fail over to Alpaca via
`price_source.fetch_klines` (verified 19/19). Alpaca is IEX-only and carries no bid/ask,
so a fallback quote is marked `source: "alpaca:fallback"` and the UI must show that the
bid/ask/spread columns are unavailable rather than blanking them silently.

The 7 Stockholm instruments have **no fallback**. On session loss they degrade to
last-good-price with an explicit age stamp. The book still totals; it never claims
freshness it does not have.

FX: `portfolio.fx_rates.fetch_usd_sek()` — note the name is `fetch_`, not `get_`.

## Instrument identity is a correctness hazard

Resolving instruments by search at runtime returned the **wrong instrument for 2 of 19**
US names during exploration:

- One query returned zero hits because the search string used the legal name rather than
  the ticker.
- One matched the wrong **share class** — Class C instead of Class A, two different
  securities with near-identical names and nearly identical prices.

A third name has a decoy second listing at a different orderbook ID.

Since the operator places orders by hand from this UI, and the orderbook ID is now both
the pricing key _and_ the deep-link target, a wrong ID would mis-price a position **and**
misdirect a real order.

**Decision: orderbook IDs are pinned in a static, human-reviewed table
(`portfolio/swedbank/instruments.py`). No runtime search resolution, ever.** A test
asserts each pinned ID still resolves to the expected instrument name, so an upstream
change fails loudly instead of silently repricing.

## Marking rule — last vs mid

`last` is unreliable on thin instruments. During exploration one warrant had traded 15
units on the day, with `last` sitting ~3% above the live bid/ask mid — stale by hours.
Marking at `last` would have overstated that position by roughly that margin. Another
holding showed a spread near 1,7%.

Rule: mark at `last` when it falls inside the bid/ask; otherwise mark at **mid** and flag
the row `stale_last`. Always surface `spread` — the operator executes manually, so the
spread is a real personal cost, not a footnote.

## Constraints discovered

1. **`pf-dataloop` is `inactive`.** Layer 1 is down; `health_state.json`,
   `signal_log.jsonl`, `agent_summary*.json` may be frozen. The new loop must be
   standalone and must not read Layer 1 outputs as if live.
2. **Formatter trap.** These files are not black-clean; the PostToolUse hook rewrites the
   whole file if touched with Edit/Write. Patch them with `python3 - <<'EOF'` heredoc
   scripts only: `dashboard/app.py`, `portfolio/signal_engine.py`,
   `dashboard/system_status.py`, `portfolio/accuracy_stats.py`,
   `dashboard/trading_status.py`, `portfolio/loop_processes.py`.
3. **The dashboard is internet-facing** (Cloudflare tunnel + CF Access JWT + token).
   The new route uses `@require_auth`, matching `/api/avanza_account` precedent.
   Approved by the operator 2026-07-31.
4. **Main-loop cycle cost.** Seven tickers were removed on 2026-04-09 purely to hold
   cycle p50 down. Adding 26 instruments to Tier 1 would repeat that mistake — hence a
   separate loop.
5. **Shared Avanza session with the real-money metals loop.** The metals loop places and
   cancels real stop-losses through the same authenticated session this loop polls.

   **Measured 2026-07-31:** a full 26-instrument quote refresh takes **1.5 s sequential**
   (median 20 ms/call, 26/26 success, first call ~950 ms as browser-context warmup).
   At a 60 s cycle that is a **2.5% duty cycle** on the shared session.

   **Decision: sequential only, never concurrent.** Concurrency was the mechanism by
   which this loop could have contended for the Playwright context or tripped rate
   limiting while metals was mid-order. At 1.5 s/cycle there is no performance reason to
   parallelise, so the risk is designed out rather than mitigated. A test asserts the
   pricing layer issues no concurrent Avanza calls.

   Remaining coupling to address in the Premortem: session _expiry_ mid-cycle, and
   whether this loop's failures could trigger a browser-recovery path that disturbs
   metals. Mitigation direction: this loop never calls recovery — on any session error it
   degrades to last-good-price and backs off, leaving recovery to the trading loops.

   Current state: all loops (`pf-dataloop`, `pf-metalsloop`, `pf-cryptoloop`,
   `pf-oilloop`, `pf-mstrloop`, `pf-golddigger`) are **inactive and disabled**; only
   `pf-dashboard` runs. The system is dormant, so nothing contends today — but the design
   must hold when metals is re-enabled.

6. Static asset changes require bumping the `sw.js` CACHE version.

## Asset-class registration — why we touch neither `tickers.py` nor `signal_engine.py`

`_compute_applicable_count` (signal_engine.py:1727) classifies purely by global set
membership:

```python
is_crypto = ticker in CRYPTO_SYMBOLS
is_metal  = ticker in METALS_SYMBOLS
is_stock  = ticker in STOCK_SYMBOLS      # == {"MSTR"}
...
if sig in _NON_STOCK_SIGNALS and is_stock: continue   # orderbook_flow
```

This creates a trap with no good option among the obvious two:

- **Leave the new tickers unregistered** → all three flags are `False`, so an equity
  reads as "not a stock" and `orderbook_flow` (declared metals+crypto only) is applied
  to it. Same class of defect as the asset-class mislabeling fixed in b5d2026b /
  597176b2.
- **Add them to `STOCK_SYMBOLS`** → the main loop is unaffected (it iterates `SYMBOLS`,
  main.py:455/477), but `STOCK_SYMBOLS` is _not_ inert. `alpha_vantage.py:238,314`
  iterates it for the fundamentals refresh against a hard **25 calls/day** quota;
  19 extra tickers exhausts it and starves Tier-1. It also pulls in
  `earnings_calendar.py:158,164,189`, `market_timing.py:312` and `reporting.py:202`.

**Decision: use neither.** `portfolio/swedbank/instruments.py` carries an explicit
`asset_class` field per instrument, and the swedbank signal runner computes applicability
from that field — the same rules, parameterised instead of global.

Consequences, all desirable:

- Zero lines changed in `tickers.py` and `signal_engine.py` → zero Tier-1 blast radius.
- Neither of the two heredoc-only files is touched for the signal work.
- The already-failing `test_signal_pipeline.py::TestVoteCountIntegrity` assertions about
  stock applicable-counts are unaffected.
- Alpha Vantage quota untouched.

## Architecture

### 1. Ledger — `portfolio/swedbank/`

Stores `qty` + `cost_basis` + pinned routing. **Never stores value** — value is always
derived, so the file cannot go stale.

```
portfolio/swedbank/
  __init__.py
  instruments.py   pinned instrument table (26 entries), single source of truth
  book.py          schema, load/save via file_utils atomic helpers, revalue()
  snapshot.py      PURE: parse export -> solve FX -> derive qty/cost -> diff vs stored
  pricing.py       Avanza quotes + history, Alpaca fallback, FX, mark/mid rule,
                   staleness stamps, graceful degradation on session expiry
  cli.py           python -m portfolio.swedbank {sync,show}
```

`snapshot.py` is pure (no I/O) so it can be exhaustively unit-tested on synthetic books.

### 2. Monitoring loop — `data/swedbank_loop.py` + `pf-swedbank.service`

Clones the `data/oil_loop.py` satellite pattern. Structurally incapable of trading: the
package imports no order module, and a test asserts that no symbol from
`avanza_orders` is reachable from `portfolio.swedbank`.

systemd unit mirrors `pf-oilloop.service` (WorkingDirectory, PYTHONPATH,
`Restart=always`, `RestartSec=30`).

### 3. Dashboard tab — `/api/swedbank` + `dashboard/static/js/views/swedbank.js`

New SPA view alongside the existing 21, registered in the hash router and the More menu.
Every price carries `as_of` / `stale_s` / `source`; the UI renders age and never shows a
stale number as live.

## Open questions (explorer agents in flight)

- [x] **ANSWERED.** Avanza exposes full OHLCV per orderbook ID — the Stockholm
      instruments can carry technical signals.
- [ ] Avanza session contention with the real-money metals loop (highest priority).
- [x] **ANSWERED.** Entry point is
      `signal_engine.generate_signal(ind, ticker, config, timeframes, df, horizon)`
      (signal*engine.py:3490). Indicators come from
      `portfolio/indicators.py::compute_indicators(df, horizon)`. It only \_warns* on a
      missing ticker — registry membership is not required to call it. See
      "Asset-class registration" below for why we still must not rely on that.
- [ ] Which of the 15 active signals genuinely apply to a plain equity.
- [ ] Whether logging 26 new instruments pollutes Tier-1 accuracy stats.
- [ ] Exact dashboard tab registration checklist.

## Test baseline (captured before any change, 2026-07-31)

`pytest tests/ -n auto` → **38 failed, 11 270 passed, 32 skipped** in 211s.
Pre-existing failures span 18 files; heaviest are `test_portfolio.py` (5),
`test_metals_loop_pre_sell_cancel.py` (4), `test_metals_loop_autonomous.py` (3),
`test_loop_contract.py` (3), `test_forecast_timeout.py` (3),
`test_forecast_circuit_breaker.py` (3). Full list kept locally outside the repo.
Any failure beyond this set is attributable to this campaign.

## Execution order

| Batch | Contents                                                              |
| ----- | --------------------------------------------------------------------- |
| 1     | `portfolio/swedbank/` ledger core + pinned instruments table + tests  |
| 2     | Pricing layer (Avanza primary, Alpaca fallback, FX, mark/mid) + tests |
| 3     | Local seed of `data/swedbank_book.json`; CLI sync/show                |
| 4     | Signals + trajectories for the universe                               |
| 5     | Monitoring loop + systemd unit                                        |
| 6     | `/api/swedbank` route (heredoc patch) + tests                         |
| 7     | SPA view + router + More menu + sw.js bump                            |
| 8     | Adversarial review, full pytest, merge, push                          |

## Premortem

Written from failure modes actually encountered or demonstrated during exploration, not
speculated. A dispatched premortem agent is still in flight; its findings get appended.

### P0-1 — Real positions leak into a public repo, permanently

**Chain:** The subsystem's natural artifacts (`data/swedbank_book.json`, a seeded test
fixture, an exported JSON, a worked example in a commit message) contain real quantities
and cost basis. `docs/GUIDELINES.md:21` makes push _mandatory_
("work that isn't merged and pushed didn't happen"). `github.com/Wojnach/finance-analyzer`
answers anonymous GET with 200 — it is public. Push publishes the operator's positions
irreversibly; git history survives later deletion, and public repos are mirrored within
minutes.

**Already happened in this session.** The first two plan commits contained real account
totals, cost basis and account labels. Caught before push and rewritten via
`git reset --soft`.

**Severity: P0.** **Hook:** `.gitignore` entries added (verified with `git check-ignore`);
privacy constraint section at the top of this plan; a pre-merge grep for the operator's
figures across the diff. Tests must construct synthetic books, never load the live file.

### P0-2 — Wrong orderbook ID sends a real manual order to the wrong security

**Chain:** Resolving instruments by Avanza search at runtime returned the wrong
instrument for 2 of 19 US names during exploration — one zero-hit (legal name vs ticker),
one **wrong share class** (Class C returned for a Class A holding; near-identical names,
near-identical prices, so the error is invisible on inspection). A third name has a decoy
second listing. Because the orderbook ID is both the pricing key and the deep-link the
operator clicks to trade, a wrong ID mis-values the position _and_ routes a real order to
the wrong security.

**Severity: P0.** **Hook:** IDs pinned in `instruments.py`; a test asserts every pinned ID
still resolves to its expected instrument name, so an upstream change fails loudly rather
than silently repricing. No runtime search resolution anywhere in the code path.

### P1-3 — Stale `last` on thin instruments silently overstates the book

**Chain:** Avanza returns `last` even when the instrument has barely traded. Measured:
one warrant had traded 15 units on the day with `last` **+4.04% above live mid** (it grew
from +3.2% within an hour as the market moved and the warrant did not). A second
instrument sat +1.05% off mid. Marking at `last` inflates those positions with no error,
no exception, exit code 0.

**Severity: P1 (silent wrong output).** **Hook:** mark at mid when `last` falls outside
bid/ask; set a `stale_last` flag on the row; surface `spread` in the UI. Unit test with a
synthetic quote whose `last` sits outside the spread, asserting mid is chosen and the flag
is set.

### P1-4 — Registering asset class exhausts the Alpha Vantage quota and starves Tier-1

**Chain:** The intuitive fix for asset-class gating is adding the new equities to
`STOCK_SYMBOLS`. The main loop is unaffected (it iterates `SYMBOLS`), so this looks free
and tests would pass. But `alpha_vantage.py:238,314` iterates `STOCK_SYMBOLS` for the
daily fundamentals refresh against a hard **25 requests/day** limit. Nineteen extra
tickers exhausts the quota, and Tier-1's fundamentals silently go stale — a Tier-1
degradation caused entirely by a monitoring-only subsystem, surfacing days later as
`fundamentals_cache.json` staleness rather than as an error.

**Severity: P1.** **Hook:** decided against touching the global sets at all (see
"Asset-class registration"). Test asserts `portfolio.swedbank` imports do not mutate
`tickers.STOCK_SYMBOLS`/`CRYPTO_SYMBOLS`/`METALS_SYMBOLS`.

### P1-5 — Unregistered ticker mislabeled as non-stock, wrong signals applied

**Chain:** The mirror of P1-4. Leave tickers out of the global sets and
`_compute_applicable_count` (signal_engine.py:1727) evaluates `is_crypto/is_metal/is_stock`
all `False`, so the `_NON_STOCK_SIGNALS` guard (`orderbook_flow`) never fires and an
order-book microstructure signal designed for metals/crypto votes on an equity. Produces
plausible-looking output that is quietly wrong.

**Severity: P1.** **Hook:** explicit `asset_class` per instrument; applicability computed
from that field; test asserting `orderbook_flow` is excluded for `asset_class="equity"`.

### P2-6 — Avanza session contention with the real-money metals loop

**Chain:** `metals_loop` places and cancels real stop-losses through the same
authenticated session. A monitoring loop polling 26 instruments could contend for the
Playwright browser context, trip rate limiting, or trigger a browser-recovery path while
metals is mid-order — turning a monitoring feature into a failed stop-loss.

**Measured:** a full 26-instrument sweep is 1.5 s sequential (median 20 ms/call), i.e. a
2.5% duty cycle at 60 s. **Severity reduced from P0 to P2 by measurement** — there is no
performance reason to use concurrency, so the contention mechanism is removed rather than
managed.

**Hook:** sequential-only, asserted by test; this loop never invokes browser recovery —
on any session error it degrades to last-good-price and backs off, leaving recovery to
the trading loops. Log a structured `swedbank_session_degraded` line rather than writing
to `critical_errors.jsonl`, so it cannot burn the fix-agent backoff budget.

### P2-7 — Layer 1 assumed alive when it is not

**Chain:** `pf-dataloop` is currently `inactive` and _disabled_, alongside every other
loop; only `pf-dashboard` runs. Code that reads `health_state.json`, `signal_log.jsonl`
or `agent_summary*.json` would consume month-old frozen data as if current. The
2026-07-18 redesign doc records this exact confusion already biting the dashboard
("Claude Fundamental 100%" was a frozen lifetime counter read as live health).

**Severity: P2.** **Hook:** the swedbank subsystem reads none of those files. Anything it
does surface carries `as_of`/`age_sec`.

### P2-8 — Parallel Claude sessions corrupt the book

**Chain:** The operator routinely runs 3+ sessions against this repo. Two concurrent
`swedbank sync` runs, or a sync racing the loop's price-cache write, could interleave
writes to `data/swedbank_book.json`.

**Severity: P2.** **Hook:** all writes through `file_utils.atomic_write_json`; the loop
writes prices to a _separate_ cache file and never to the book; `sync` takes the same
O_CREAT|O_EXCL singleton lock the oil loop uses (`data/oil_loop.py:98`).

### Plan changes required

1. `.gitignore` + privacy section — **done** before any code (P0-1).
2. Pinned instrument table with a resolve-verification test (P0-2).
3. Mark-at-mid rule with `stale_last` flag (P1-3).
4. Explicit `asset_class`; do not touch the global ticker sets (P1-4, P1-5).
5. Sequential-only pricing, no browser recovery, structured degradation log (P2-6).
6. No reads of Layer-1 state files (P2-7).
7. Atomic writes + singleton lock on sync; loop never writes the book (P2-8).
