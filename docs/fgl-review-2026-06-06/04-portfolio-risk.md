# Adversarial review — portfolio-risk subsystem (2026-06-06)

Reviewed as a fresh PR: portfolio_mgr, portfolio_validator, risk_management,
trade_guards, trade_validation, trade_risk_classifier, kelly_sizing,
equity_curve, monte_carlo, monte_carlo_risk, circuit_breaker,
warrant_portfolio, cost_model, journal, journal_index,
decision_outcome_tracker, outcome_tracker, backtester.

## P0
(none confirmed — atomic I/O, locks, fail-safe drawdown, and append-only
journals all hold under the read paths examined.)

## P1
- portfolio/warrant_portfolio.py:199-280: `record_warrant_transaction` does an
  unlocked load → mutate → `save_warrant_state` read-modify-write on
  portfolio_state_warrants.json. portfolio_mgr.update_state serializes the two
  500K portfolios with per-file locks; the warrant book has NONE. The metals
  loop (grid_fisher / fin_snipe) can record fills concurrently → last-writer-wins
  drops a transaction and corrupts units/avg-entry (money accounting). → Route
  warrant writes through a per-file lock (mirror portfolio_mgr._get_lock /
  update_state), or add a `update_warrant_state(mutate_fn)` with the same lock.
- portfolio/warrant_portfolio.py:215-280: even single-threaded, the mutation is
  not crash-atomic across the transaction-append + holdings-update: `save_warrant_state`
  is called once at the end (good), but `state["transactions"].append(txn)` then
  the holdings math share one dict, so a raised exception in the avg-in block
  (e.g. bad input) leaves `txn` appended but holdings un-updated only in memory —
  not persisted, so benign — BUT a SELL on a missing key returns AFTER appending
  txn to the in-memory list without saving (line 264 `return`), so no persistence
  either. Confirm intended: SELL-refused path silently drops the recorded txn. →
  Either persist the refusal or don't append before validation.
- portfolio/portfolio_mgr.py:228-234 (update_state) + 175 (quarantine): on
  unrecoverable corruption `_load_state_from` returns a FRESH 500K default; the
  caller's `mutate_fn` then applies a BUY/SELL to that fresh state and the very
  next `_atomic_write_json` overwrites the (already-quarantined) file with a
  reset-to-500K portfolio. Quarantine preserves evidence but the live book is
  silently reset to initial capital mid-session — money created/destroyed vs
  prior real state. → In update_state, if the load came from the corrupt→default
  path, refuse to write (raise/skip the mutation) instead of persisting a reset
  book over a real one.
- portfolio/risk_management.py:121-186 `_resolve_fx_rate` + check_drawdown: the
  cash-only fallback when agent_summary is empty (lines 252-271) makes the
  drawdown breaker OPTIMISTIC for underwater holdings (already self-documented as
  a known blind spot). With a stale/missing feed and losing positions the breaker
  under-reports drawdown and won't trip. This is a fail-toward-trading risk on the
  one guard whose whole job is to halt. → Treat "holdings present but no live
  price" as a hard non-finite/halt case (or use avg_cost fallback like
  `_compute_portfolio_value` does) rather than cash-only.
- portfolio/monte_carlo_risk.py:408 + 485-491: `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` takes the raw value with NO sanity band. The rest of
  the subsystem rejects out-of-band/1.0 fx via `_resolve_fx_rate` (risk_management
  P1-15). A stale agent_summary embedding `fx_rate: 1.0` makes every *_sek VaR/CVaR
  understate SEK risk ~10x. → Use `risk_management._resolve_fx_rate` (or the same
  [7,15] band) here too.

## P2
- portfolio/risk_management.py:96-98 `_streaming_max`: `val = entry.get(value_key, 0); if val > peak`. A JSONL entry with `value_key: null` makes `None > float` raise
  TypeError, which is NOT caught (the surrounding try only catches OSError) →
  check_drawdown crashes for that portfolio, disabling the breaker until the line
  ages out. log_portfolio_value currently always writes numeric, but any other
  writer / hand-edit weaponizes it. → coerce: `if isinstance(val,(int,float)) and val > peak`.
- portfolio/trade_guards.py:264-330 record_trade: cooldown timestamp is set for
  BOTH BUY and SELL under key `{strategy}:{ticker}`. A SELL therefore arms the
  per-ticker cooldown and a subsequent re-entry BUY is blocked for the full
  (escalated) window — and conversely the same ticker can't be SOLD then quickly
  re-BOUGHT on a bounce (the documented bounce-reentry strategy in memory). Verify
  this is intended; if SELL should not gate re-entry, key cooldown on BUY only or
  track direction.
- portfolio/circuit_breaker.py:88-100 allow_request: the OPEN→HALF_OPEN transition
  mutates state inside `allow_request` (a query). Two threads can both read OPEN,
  but the lock serializes them so only the first flips to HALF_OPEN and returns
  True; the second sees HALF_OPEN and returns False — correct. No bug, but note
  `_half_open_probe_sent` is set but never read in allow_request (dead flag) →
  remove or assert on it to prevent future double-probe regressions.
- portfolio/kelly_sizing.py:83-88: break-even round-trips (pnl_pct == 0) are
  bucketed as losses (`p <= 0`) with magnitude 0, deflating win_rate and pulling
  avg_loss toward 0. With several flat trades win_prob/edge is understated → Kelly
  under-sizes. → bucket `== 0` separately or exclude from win/loss.
- portfolio/equity_curve.py:489-490 vs 463-464: win/loss *counts and ratios* use
  `pnl_pct` (gross, price-only) while profit_factor/total_pnl use `pnl_sek` (net of
  fees). A round-trip that's +0.1% gross but fee-negative counts as a WIN in
  win_rate/expectancy yet a LOSS in profit_factor — internally inconsistent
  reporting. → pick one basis (prefer net SEK) for win/loss classification.
- portfolio/monte_carlo.py:150 / monte_carlo_risk.py:227: when `seed is None`,
  `np.random.default_rng(None)` is non-deterministic by design, but
  compute_portfolio_var / simulate_all default seed=None, so VaR/stop-prob figures
  jitter run-to-run (the reporting.py caller passes no seed). Acceptable for
  Monte Carlo but means two cycles 60s apart can report materially different
  knockout/stop probabilities on small n. → pass a per-cycle deterministic seed
  for reproducible audit, or raise n_paths.

## P3
- portfolio/agent_invocation.py:1012: f-string builds `tkr}\{strat}` with a literal
  backslash (Windows-path artifact) in the guard-context message — cosmetic, leaks
  into the Layer 2 prompt.
- portfolio/backtester.py:92-95: self-documented look-ahead bias (accuracy_data
  built from full log incl. future outcomes). TODO acknowledges it; results are
  optimistic until walk-forward rebuild lands. Keep flagged so no one cites these
  numbers as live edge.
- portfolio/risk_management.py:781 check_concentration_risk: `proposed_alloc =
  min(total_value*alloc_pct, cash)` then only warns above 40% and never blocks
  (returns None otherwise) — advisory only, consistent with design, but the
  function name implies a gate it doesn't enforce.
- portfolio/journal.py: writes only the context markdown (atomic_write_text); the
  append-only journal itself is written elsewhere via atomic_append_jsonl — journal
  mutation invariant holds. No action; noted for completeness.

## Risk summary
No P0 corruption/fail-open confirmed: the two 500K books use locked atomic
read-modify-write, the drawdown breaker fails safe on non-finite values, and the
journals are append-only. The real exposure is the warrant portfolio (unlocked
concurrent writes can drop/corrupt money state) and two breaker/VaR fail-toward-
trading paths (cash-only drawdown fallback, unsanitized VaR fx_rate) that quietly
understate risk when the price feed is stale.
