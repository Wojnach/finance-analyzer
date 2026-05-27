## Summary

**Counts:** 2 P0 · 6 P1 · 5 P2 · 3 P3

**Top 3 themes**

1. **FX-rate fallback regression in two surviving call-sites.** The P1-15
   fix in `risk_management.py` introduced a sanitised `_resolve_fx_rate`
   that rejects out-of-band `fx_rate` values (incl. the legacy `1.0`
   literal) and walks a cache chain. Two production paths
   (`monte_carlo_risk.compute_portfolio_var` and
   `exit_optimizer.compute_exit_plan_from_summary`) still call
   `agent_summary.get("fx_rate", DEFAULT)` directly, defeating the gate.
2. **Portfolio mis-valuation hides drawdown when prices are stale.** In
   the absence of a live USD quote, `_compute_portfolio_value` falls
   back to `avg_cost_usd` and reports zero P&L; in the absence of an
   agent_summary at all, `check_drawdown` falls back to cash-only —
   both paths cause the drawdown circuit-breaker to under-state losses
   silently.
3. **Risk gates are advisory not enforced.** Concentration check,
   correlation check, regime-mismatch check, and MINI-barrier proximity
   (memory/grudge "feedback_mini_stoploss") all return `severity:
   "warning"` rather than `block` — Layer 2 sees a warning string but
   no `should_block_trade()`-equivalent intercepts the trade. Per-
   strategy concentration also misses cross-strategy aggregation
   (Patient + Bold + Warrants combined).

**Biggest risk:** if the live ticker price for an active holding is
missing for any reason, `risk_management._compute_portfolio_value`
silently uses `avg_cost_usd` (line 209-212), making the portfolio
value look flat. Drawdown stays at 0% no matter how deep the underlying
sells off, so the 50% block in `agent_invocation.py` will never trip
during the very outage that justified it.

---

### [P0] Drawdown circuit breaker blinded by single missing ticker price
**File:** `Q:/finance-analyzer/portfolio/risk_management.py:201-212`
**Issue:** `_compute_portfolio_value` walks `holdings` and, when
`signals.get(ticker, {}).get("price_usd")` is missing/zero, falls
back to `pos.get("avg_cost_usd", 0)` and adds `shares * avg_cost * fx`
to the holdings value. The position is therefore valued at *entry*
price, not market. If silver craters 30% but the XAG signal block is
missing/stale for a cycle (Binance FAPI hiccup, slow yfinance), the
patient/bold values printed to `portfolio_value_history.jsonl` will
look unchanged and `check_drawdown` will report 0% drawdown. The
`check_drawdown` cash-only fallback at line 260-270 logs a WARNING when
`summary` is *entirely* empty, but the per-ticker missing-price path
swallows the failure silently (single-line `logger.warning` then carries
on as if the price were realised). Compounded by the fact that the
peak-value cache `_peak_cache` is monotonically non-decreasing, any
mis-valued spike during a stale-price window becomes the all-time peak
and inflates the denominator forever.
**Impact:** The 50% drawdown block in `agent_invocation._check_drawdown_gate`
becomes silently non-functional during the exact data-stale conditions
where it most needs to fire (user accepts 10-20% knockout risk per
`feedback_risk_tolerance.md`; 50% is the line where the user wants the
loop to stop). Real-money signals continue to be fired into Telegram
during what is actually an undetected 40%+ drawdown.
**Fix:** Drop the `avg_cost_usd` fallback. If the live price is missing,
either (a) propagate a "stale_value=True" flag in the return dict and
have `check_drawdown` treat any portfolio with `stale_value=True`
as fail-safe (block downstream), or (b) use the last *good* market price
from `portfolio_value_history.jsonl` (already on disk) with an age cap.
Either way, surface the staleness in `data/critical_errors.jsonl` so
the auto-fix-agent dispatcher picks it up.
**Confidence:** 92

---

### [P0] `monte_carlo_risk` and `exit_optimizer` bypass the P1-15 FX fallback gate
**File:** `Q:/finance-analyzer/portfolio/monte_carlo_risk.py:408`
       and `Q:/finance-analyzer/portfolio/exit_optimizer.py:718`
**Issue:** The May-02 P1-15 fix added `_resolve_fx_rate()` in
`risk_management.py` so that any agent_summary `fx_rate` outside the
sanity band (FX_RATE_MIN=7.0, FX_RATE_MAX=15.0) — including the
legacy literal `1.0` — falls back via cache → hardcoded 10.50 instead
of being trusted at face value. Two surviving production call-sites
still do the unguarded `agent_summary.get("fx_rate", X)` pattern:

* `compute_portfolio_var` line 408 — used for VaR/CVaR reporting and
  surfaced in `/api/risk`.
* `compute_exit_plan_from_summary` line 718 — used to value
  MINI-future exit P&L per `MarketSnapshot.usdsek`.

A stale or buggy `fx_rate: 1.0` in `agent_summary.json` (the exact
failure mode P1-15 documents) flows through unchecked and understates
USD→SEK valuations by ~10x. In `monte_carlo_risk` the resulting VaR
and CVaR numbers shown on the dashboard are 1/10th of reality; in
`exit_optimizer` every exit candidate's `pnl_sek` is off by ~10x,
which corrupts the EV ranking that drives the recommended exit.
**Impact:** Layer 2 reads `/api/risk` and exit plan summaries when
making trade decisions. A 10x VaR understatement could trigger
aggressive sizing during a regime where the circuit breaker should
have fired (real VaR 200K SEK, reported as 20K SEK looks routine).
Exit recommendations may pick a far quantile because its EV is
proportionally suppressed alongside the market exit's EV — but the
*ordering* survives, so the impact is bounded to dashboards/journals
rather than execution. Still P0 because (a) the original P1-15 bug
was already P1-class and these two callers were never patched, and
(b) operators relying on `/api/risk` for de-risking decisions get
wrong numbers.
**Fix:** Replace the raw `.get("fx_rate", X)` with
`from portfolio.risk_management import _resolve_fx_rate; fx_rate =
_resolve_fx_rate(agent_summary)` (or promote `_resolve_fx_rate` to
public `fx_rates.resolve_fx_rate` to break the layering). Add a
regression test: pass `{"fx_rate": 1.0, "signals": {...}}` and assert
the resolver returns the cached/fallback value.
**Confidence:** 95

---

### [P1] MINI-barrier proximity rule (≥3% buffer) not codified in risk gates
**File:** `Q:/finance-analyzer/portfolio/risk_management.py` (no
implementation); `Q:/finance-analyzer/portfolio/trade_guards.py` (no
implementation)
**Issue:** `memory/feedback_mini_stoploss.md` calls out as CRITICAL
that stop-losses must never be placed within 3% of a MINI warrant's
financing/barrier level. The only codification of barrier proximity is
in `exit_optimizer._compute_risk_flags` line 372-377 — and that's a
*flag*, not a gate — and `_apply_risk_overrides` line 430-435 which
forces a market exit when ALREADY held. **There is no pre-trade gate**
in `risk_management.py` / `trade_guards.py` that prevents the system
from placing a stop within the 3% danger zone in the first place. The
golddigger / iskbets paths have their own ad-hoc handling, but the
portfolio-risk subsystem (which gates Layer 2 invocations) does not.
Compounded by `compute_stop_levels` (line 374) using only 2×ATR — for
a tight 0.5% ATR XAG day, a 1% stop on a 5x cert with barrier ~3-5%
away is exactly the failure mode the grudge documents.
**Impact:** Grudge-class failure mode (already burned the user once).
The system can recommend stops that get walked through on a normal
intraday wick.
**Fix:** Add `check_barrier_proximity(ticker, stop_price_usd,
financing_level_usd)` to risk_management.py returning
`severity: "block"` when stop is within 3% of financing. Wire it into
`compute_all_risk_flags` and into the stop-placement codepaths in
`avanza_session` / `golddigger.bot`.
**Confidence:** 88

---

### [P1] Concentration / correlation / regime-mismatch checks are advisory only
**File:** `Q:/finance-analyzer/portfolio/risk_management.py:776-788,
833-865, 791-830`
**Issue:** `check_concentration_risk` (>40% of portfolio),
`check_correlation_risk` (correlated names already held), and
`check_regime_mismatch` (BUY in trending-down with low RVOL) all return
`severity: "warning"` only. `compute_all_risk_flags` aggregates them
into a `flags` list but there is no `should_block_trade()` equivalent
that intercepts on warning. In `trade_guards.py`, by contrast,
`ticker_cooldown` and `position_rate_limit` set `severity: "block"`
and there IS a `should_block_trade()` (line 394) which is wired into
the agent invocation path. The risk-flag warnings are surfaced to
Layer 2 as plain text and the LLM is trusted to honour them — same
class of trust-the-LLM failure pattern as the headless `PF_HEADLESS_AGENT`
prompt that caused the 2026-04-16 outage.
**Impact:** Concentration past 40% in a single ticker, correlated
double-up (e.g. BTC + ETH + MSTR all bought together), and counter-
trend BUYs all proceed if Layer 2 doesn't notice the warning string
in the prompt. Combined with the per-strategy-only scope (next
finding), realistic 80%+ effective single-name exposure is possible.
**Fix:** Promote concentration > 40% and correlation conflicts to
`severity: "block"`, wire a `should_block_risk()` helper analogous to
`trade_guards.should_block_trade()`, and call both before subprocess
spawn in `agent_invocation`.
**Confidence:** 85

---

### [P1] Concentration check is per-strategy; cross-strategy aggregate exposure unbounded
**File:** `Q:/finance-analyzer/portfolio/risk_management.py:729-788`
**Issue:** `check_concentration_risk` takes a single `portfolio`
argument and computes `concentration_pct = new_position_value /
total_value * 100`. The caller in `compute_all_risk_flags` (line
954-958) iterates `[("patient", patient_pf), ("bold", bold_pf)]` and
checks each in isolation. Warrants portfolio is ignored entirely.
Result: XAG can be 39% of Patient + 39% of Bold + 100% of Warrants
without firing a single flag. Per the project rule "Concentration
check is portfolio-proportional, not absolute" this is technically
in spec, but the proportional check is being applied at the wrong
layer (per-strategy instead of total household exposure).
**Impact:** A correlated-shock event in XAG/silver (which the user
explicitly identifies as 0.8-conviction primary focus) takes down all
three portfolios simultaneously with no advance warning, exceeding
the user's stated 10-20% knockout-risk tolerance at the household
level.
**Fix:** Add a `check_household_concentration` that sums the
ticker's value across all three portfolios (patient, bold, warrants
via `warrant_portfolio.load_warrant_state()`) divided by the total
household value, with a threshold of (say) 50%.
**Confidence:** 87

---

### [P1] `running_extremes` uses hardcoded seed 42 — MC results deterministic
**File:** `Q:/finance-analyzer/portfolio/price_targets.py:125`
**Issue:** `rng = np.random.default_rng(42)` is hardcoded. Every call
to `running_extremes` for any ticker, any time, generates the same
n_paths × n_steps standard-normal draws. The percentile output looks
randomised because it depends on price/vol/drift inputs, but the
underlying Brownian noise is the same draw each call. Two consequences:
(1) the p10/p25/p50/p75/p90 quantiles consumed by `compute_targets` as
target candidates are a single realisation, not a distribution
estimate, so confidence in their accuracy is over-stated; (2) tests
pass trivially without exercising the stochastic behaviour. Note that
`monte_carlo.py`, `monte_carlo_risk.py`, and `exit_optimizer.py` all
correctly accept `seed: int | None` and pass through to
`default_rng`, so this is unique to `price_targets`.
**Impact:** Quantile-based limit-order placement (e.g.
`compute_targets` for the fishing system) effectively bets the
same Brownian path every session. Practical impact: the candidate
target list is biased toward whatever the seed-42 draw happens to
favour, and antithetic variates aren't used here either (only
`exit_optimizer` and `monte_carlo` do that).
**Fix:** Plumb `seed: int | None = None` through `running_extremes`
and `compute_targets`. Default `None` for production. Tests that need
reproducibility can pass an explicit seed.
**Confidence:** 90

---

### [P1] `warrant_pnl` has no BEAR/SHORT direction handling
**File:** `Q:/finance-analyzer/portfolio/warrant_portfolio.py:96`
**Issue:** `implied_pnl_pct = underlying_change * leverage`. The
warrant catalog in `fin_fish.py` includes BEAR/SHORT products (e.g.
`BEAR_SILVER_X5_AVA_12`, `BEAR_GULD_X5_VON4`) that profit when the
underlying falls. The `holding` dict has fields `leverage`,
`underlying`, but no `direction` field, and `record_warrant_transaction`
does not capture one. A BEAR cert booked with positive leverage will
return inverted P&L: a 2% underlying drop yields the formula
`-0.02 * 5 = -10%` instead of the correct +10%. `get_warrant_summary`
(called in production from `reporting.py`) propagates this directly
to `/api/warrants` and the dashboard. `record_warrant_transaction`
itself is not currently invoked from production code (tests only), so
new state isn't being written through this path, but **any state that
already exists or arrives from another writer is mis-reported**.
**Impact:** P1 and not P0 only because the metals subsystem's primary
writer is `fin_snipe_manager` / `grid_fisher`, which maintain their
own state files, and `portfolio_state_warrants.json` is mostly
unpopulated today. If/when the canonical warrant tracker is wired up
(or a manual BEAR position is hand-edited in), reported P&L silently
flips sign.
**Fix:** Add `direction: "LONG" | "SHORT"` to the holding dict, default
LONG for backward compatibility, multiply `implied_pnl_pct` by
`-1` when SHORT, and have `record_warrant_transaction` accept and
persist a `direction` argument.
**Confidence:** 86

---

### [P1] `compute_stop_levels` assumes long-only positions
**File:** `Q:/finance-analyzer/portfolio/risk_management.py:374`
**Issue:** `stop_price = entry_price * (1 - 2 * atr_pct / 100)`. For a
SHORT position the stop should be `entry * (1 + 2 * atr/100)` (above
entry, the price at which the short loses). The Patient/Bold
portfolios use BUY/SELL semantics where SELL closes a long, not opens
a short, so today this is dormant. Combined with the previous BEAR-
direction finding, however, if SHORT semantics ever land in
`holdings`, the stop-loss math is upside-down — `triggered = current
< stop` would fire on a winning short instead of a losing one.
**Impact:** Latent; activated by any code path that adds short
positions to the standard `holdings` dict (e.g., if Layer 2 were
extended to short crypto/metals directly).
**Fix:** Read a `direction` from the holding (default LONG), branch
the stop formula, and branch the `triggered` comparison.
**Confidence:** 80

---

### [P2] `transactions` list grows unbounded inside portfolio_state JSON
**File:** `Q:/finance-analyzer/portfolio/portfolio_mgr.py:21-26` (no
pruning); `portfolio/warrant_portfolio.py:214` (append-only)
**Issue:** Every BUY/SELL appends to the in-memory `transactions` list
and the whole dict is rewritten via `atomic_write_json`. No cap, no
rotation. Six months of moderate activity at 5 trades/day = ~900
entries; a year of grid-fisher partial fills could be thousands. Each
write rewrites the entire file (tempfile + fsync + replace) and
`validate_portfolio` performs O(transactions²) work in the avg-cost
check (line 226-243). The validator scales fine at current volume but
will degrade noticeably past ~5K entries.
**Impact:** Operational (file size, validate latency, JSON parse time
on every `load_state()`). Doesn't cause a wrong number, just slower
work that eventually delays the 60s loop cycle.
**Fix:** Add a `prune_transactions(max_entries=2000)` helper and call
it from `save_state` every N writes, or archive older entries to a
separate `transactions_archive.jsonl` and keep only the recent window
in the live file.
**Confidence:** 82

---

### [P2] `_peak_cache` peak monotonicity has no NaN guard on the read path
**File:** `Q:/finance-analyzer/portfolio/risk_management.py:91-110`
**Issue:** `_streaming_max` reads `val = entry.get(value_key, 0)`
and compares with `>` against the cached peak. JSON-NaN can be written
by a corrupt upstream — `json.dumps(float('nan'))` produces `NaN`
which most JSON parsers reject, but Python's `json.loads` accepts it
by default. A NaN value compared with `>` is always False, so it
doesn't poison the peak; but if NaN ever lands in `current_value`
(via `_compute_portfolio_value` from a bad price), the
`math.isfinite` guard at line 291 catches it for `current_value` —
yet does NOT re-check `peak_value` after the cache hot path returns.
The `_streaming_max` path returns a float that bypasses NaN
validation if the cached peak was already non-finite somehow (e.g.
test fixture that wrote NaN to history then warmed the cache).
**Impact:** Low. The fail-safe at line 291 (return 100% drawdown)
catches the common case. Edge cases remain where a poisoned cache
survives.
**Fix:** Validate `peak` with `math.isfinite` inside `_streaming_max`
before storing in `_peak_cache`, and on cache-hit before returning.
**Confidence:** 80

---

### [P2] `record_trade()` `consecutive_losses` increments on any negative pnl_pct, no minimum threshold
**File:** `Q:/finance-analyzer/portfolio/trade_guards.py:281-285`
**Issue:** `if pnl_pct < 0: consecutive_losses += 1`. A -0.01% round-
trip (essentially a scratch, possibly just spread + courtage) counts
as a "loss" and contributes to escalating the cooldown multiplier
(1x → 2x → 4x → 8x). The Avanza warrant spread is 0.4-0.5% per leg
which alone can push a scratch trade into "loss" classification.
**Impact:** Cooldowns escalate spuriously after a streak of break-
even scratches, blocking legitimate re-entries. Less severe than
missing the escalation entirely, but it does mean the system gets
more conservative than the data warrants.
**Fix:** Define a noise threshold (e.g. `LOSS_THRESHOLD_PCT = -0.3`)
below which scratches don't increment the loss streak, and
optionally don't reset it either.
**Confidence:** 83

---

### [P2] `equity_curve._daily_returns` uses last-value-per-day; intraday peak-to-trough drawdowns invisible
**File:** `Q:/finance-analyzer/portfolio/equity_curve.py:82-110, 199-213`
**Issue:** `_daily_returns` groups by date and keeps only the last
value per day. The max-drawdown calculation at lines 199-213 walks
`values` (not daily-aggregated), but `daily_rets` (used for
volatility, Sharpe, Sortino, best/worst day) reflects daily close-to-
close only. A 25% intraday round-trip that closes flat is reported as
zero return that day, contributing to a deceptively low realised
volatility and inflated Sharpe.
**Impact:** Sharpe/Sortino reported on `/api/equity-curve` look
better than realised risk warrants. Won't trigger a wrong trade
decision directly but skews backtest comparisons and
strategy-attribution reports.
**Fix:** Compute high-water-mark drawdown from the raw `values`
sequence (already done in max_dd), then also compute intraday-aware
realised vol from the same sequence rather than from daily aggregates.
**Confidence:** 80

---

### [P2] `WARRANT_COSTS` underestimates real spread for high-spread BEAR products
**File:** `Q:/finance-analyzer/portfolio/cost_model.py:64-70`
**Issue:** `WARRANT_COSTS` uses `spread_bps=40.0` (0.4% half-spread)
as a single value for all warrants. The actual catalog in
`fin_fish.py` reports `spread_pct=2.2` for `BEAR_GULD_X5_VON4` (Vontobel
issuer) versus 0.5 for AVA-issued products. The cost model is consumed
by `exit_optimizer._compute_pnl_sek` which underprices exit costs for
non-AVA warrants by ~4x.
**Impact:** EV ranking of exit candidates is biased: limit exits look
better than market exits because the cost model under-charges both
sides equally — but the *absolute* error favours holding too long on
high-spread products.
**Fix:** Either look up the per-instrument `spread_pct` from
`WARRANT_CATALOG` and build a per-instrument `CostModel` at the
call-site, or add a `WARRANT_HIGH_SPREAD_COSTS` model and switch
based on issuer.
**Confidence:** 80

---

### [P3] `compute_metrics` annualisation uses 365 unconditionally for both crypto and stock days
**File:** `Q:/finance-analyzer/portfolio/equity_curve.py:23, 230-249`
**Issue:** `ANNUALIZATION_DAYS = 365` is global. The comment justifies
this by "portfolio runs 24/7" (crypto), but the equity curve mixes
crypto-driven moves with US-stock-driven moves (MSTR) and metals.
A mixed-portfolio Sharpe inflates the annual-vol estimate relative to
stock-only convention. Acceptable choice but worth a one-line comment
on the dashboard so consumers don't compare against stock benchmarks.
**Confidence:** 80

---

### [P3] `transaction_cost_analysis` returns approximate P&L without flagging it loudly
**File:** `Q:/finance-analyzer/portfolio/risk_management.py:686-704`
**Issue:** When `has_open_positions` is True, `pnl = cash -
initial_value` is returned with a `pnl_note: "approximate (excludes
unrealized gains/losses)"`. The `fees_as_pct_of_pnl` ratio is then
computed against this approximate denominator and rounded to four
decimal places — looks precise, isn't. Dashboards reading this number
won't surface the note.
**Confidence:** 80

---

### [P3] `update_state` mutate_fn return-value contract is undocumented and silently swallows None
**File:** `Q:/finance-analyzer/portfolio/portfolio_mgr.py:136-159`
**Issue:** `result = mutate_fn(state); if result is not None: state =
result`. A mutate_fn that mistakenly returns the holdings dict instead
of the full state, or that returns a partial dict, gets persisted
verbatim, overwriting cash_sek/transactions etc. Equally a mutate_fn
that *intended* to replace state but returned None has its mutation
silently kept (it modified `state` in-place). Both contracts are
"work" but they don't compose safely.
**Fix:** Document that mutate_fn must either mutate in-place
(returning None) or return a fully-formed state dict, and add an
assert / shape check on the returned dict before write.
**Confidence:** 80
