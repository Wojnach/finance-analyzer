# Portfolio-Risk Adversarial Review

Scope: 19 files in worktree `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/`.
Reviewer focused on bugs that cost money or corrupt state.

## P0 findings

`portfolio/warrant_portfolio.py:182-265`: 🔴 `record_warrant_transaction` performs read-modify-write
of `portfolio_state_warrants.json` with **no lock** of any kind — neither the per-file
`threading.Lock` that `portfolio_mgr.update_state` uses, nor any cross-process file lock.
`load_warrant_state` (line 198) → mutate (lines 215-263) → `save_warrant_state` (line 265).
`grid_fisher`, `fin_snipe`, `fin_snipe_manager`, and `metals_swing` are all confirmed callers
running in DIFFERENT PROCESSES (PF-MetalsLoop, PF-DataLoop, dashboard handlers). A grid-fisher
fill that arrives while a fin_snipe close is mid-write will silently drop the close: both load
identical state, both mutate, last writer wins, the closed lot reappears as a phantom open
position. This is direct money loss / state corruption.

`portfolio/exit_optimizer.py:325-332`: 🔴 `_compute_pnl_sek` for warrants WITHOUT
`financing_level` (the BEAR-cert / non-MINI branch) computes
`warrant_move = pct_move * position.leverage`, treating `leverage` as a signed long-leverage
factor. `Position` has no `direction` field — BULL and BEAR certs are indistinguishable.
A BEAR cert at leverage=5 with the underlying rising 1% gets `warrant_move = +5%` (P&L SEK
positive) when the actual warrant lost ~5%. `grid_fisher.py` explicitly trades BEAR
certificates (`BULL=LONG, BEAR=SHORT` comments at lines 336 and 1740). If any caller invokes
`compute_exit_plan` for a BEAR-cert position, the recommended exit, EV ranking, and
`market_exit.pnl_sek` are all sign-flipped — system will "take profit" on a losing trade and
"hold the winner" while it bleeds out. The MINI/financing_level branch (line 319) is safe
because it computes `exit_warrant_sek = (exit_price_usd - financing_level) × fx`, but the
fallback at line 325 is not.

`portfolio/portfolio_mgr.py:136-159`: 🔴 `update_state` provides only a process-local
`threading.Lock` (`_state_locks[str(path)]`). The dashboard process, Layer 2 subprocess
(`claude -p`), `mstr_loop`, `crypto_loop`, `oil_loop`, and the main loop all read & write
`portfolio_state.json` / `portfolio_state_bold.json`. `os.replace` is atomic for the WRITE,
but the read-mutate-write window between two processes is unlocked. Layer 2 books a BUY,
meanwhile the dashboard's `/api/validate-portfolio` POST handler mutates and saves the old
state it loaded — Layer 2's BUY vanishes. CLAUDE.md mandates atomic I/O but does not call
out cross-process locking; this is the actual gap. `process_lock.py` exists in the repo
(used elsewhere) but is not wired into `update_state`.

## P1 findings

`portfolio/kelly_sizing.py:92, 99-103`: `_compute_trade_stats` computes
`avg_buy_price = total_cost / total_shares_bought` where `total_cost = sum(total_sek)`. Per
`portfolio_validator.py:71-72` and CLAUDE.md's transaction schema, `BUY.total_sek` is
**full allocation including fee** and `SELL.total_sek` is **net proceeds after fee deducted**.
So `pnl_pct = (sell_net_per_share − buy_gross_per_share)/buy_gross_per_share` double-counts
fees (subtracted once on the buy via inflated buy price, once on the sell via reduced
proceeds). Historical win-rate, avg_win, avg_loss feeding Kelly are biased pessimistic →
under-sizing for the patient strategy and incorrect win/loss ratio. Fix: subtract `fee_sek`
from `total_sek` on the BUY leg, add it back on the SELL leg, or use `shares × price_usd`
directly.

`portfolio/risk_management.py:320-395`: `compute_stop_levels` and `compute_probabilistic_stops`
compute a flat `entry_price * (1 - 2*atr_pct/100)` stop with ATR capped at 15%. **The
function is unaware of MINI warrant `financing_level` / knock-out barriers.** Memory note
explicitly: "NEVER place stop-losses near MINI warrant barriers" — this is enforced ONLY
inside `exit_optimizer._apply_risk_overrides`, not at the stop-computation layer. Any caller
that uses the output of `compute_stop_levels` to PLACE an order (or to gate trade entry on
ATR-stop distance) will get a stop that may sit below the knock-out barrier, guaranteeing a
knockout fill before the stop triggers. Same problem in `check_atr_stop_proximity`
(line 868) — flags positions as "dangerously close to stop" but cannot tell if the stop is
itself BEHIND the barrier.

`portfolio/equity_curve.py:494-528`: `compute_trade_metrics` uses `pnl_pct` (gross %) to
classify wins vs losses and to compute streaks, expectancy, win_loss_ratio — but uses
`pnl_sek` (NET of fees, per the P0-6 fix comment at line 405) for `profit_factor` and
`total_pnl_sek`. After fees a trade can have `pnl_pct > 0` and `pnl_sek < 0`. Then it
counts as a "win" in `wins[]`, extends the current_wins streak, inflates the
`win_loss_ratio`, but reduces gross_profit in profit_factor calculation. The
two metrics report contradictory truths from the same trade. Fix: classify wins by
`pnl_sek > 0`, not `pnl_pct > 0`.

`portfolio/monte_carlo_risk.py:408`: `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)`
bypasses the sanity-band check from `risk_management._resolve_fx_rate` (which exists
specifically to reject `fx_rate=1.0` stale values per the P1-15 incident comment). If
`agent_summary.fx_rate == 1.0` (the legacy default that the resolver was added to reject),
all SEK VaR/CVaR reporting is **10× understated**. Layer 2 sees `var_95_sek ≈ -150 SEK`
when reality is `-1500 SEK` and over-sizes accordingly. Fix: call `_resolve_fx_rate(agent_summary)`
or import & reuse it.

`portfolio/risk_management.py:189-214`: `_compute_portfolio_value` falls back to
`avg_cost_usd` on a per-ticker basis when `signals[ticker].price_usd` is missing. The
warning fires per-ticker (good) but the **drawdown circuit breaker at agent_invocation.py:895
still uses the resulting value as authoritative**. A held position with a stale price feed
is valued at cost, so unrealized losses are invisible to `check_drawdown`. The 50%
`_DRAWDOWN_BLOCK_PCT` can be hit silently. The fail-safe path (line 248-269) only triggers
when `summary` is COMPLETELY empty — not when individual ticker prices are stale. Fix:
when ANY held-ticker price is missing, mark the drawdown reading as unreliable and either
fail-safe (block) or warn loudly. Currently it logs a warning and proceeds.

`portfolio/monte_carlo_risk.py:160-378`: `PortfolioRiskSimulator.__init__` filters with
`p.get("shares", 0) != 0` (line 188), allowing negative-share (short) positions. But
`drawdown_probability` (line 370-377) computes `total_value = sum(shares * price)` which
goes negative for net-short portfolios, then `loss_threshold = -total_value * pct/100`
flips sign, inverting the comparison. No system path currently shorts, but the filter says
the wrong thing — should be `> 0` until shorting is supported and explicitly tested. Same
inconsistency: `compute_portfolio_var` (line 416) DOES filter `if shares <= 0: continue`,
so the contract is ambiguous.

`portfolio/correlation_priors.py:8-11`: Only **two pairs** defined:
`(BTC-USD, ETH-USD)=0.75` and `(XAG-USD, XAU-USD)=0.85`. MSTR is a known high-correlation
leveraged proxy for BTC (regularly cited in this repo's prophecy and CLAUDE.md). With no
prior, `monte_carlo_risk.build_correlation_matrix` falls back to 0 correlation between
MSTR and BTC/ETH. A portfolio long both BTC and MSTR will have its VaR understated by
~30-40% — the joint left tail is fatter than the diagonal-only model captures. Add
`(MSTR, BTC-USD): 0.80` and a copper/silver, gold/yields, etc. set, or use rolling
historical correlation by default.

`portfolio/kelly_metals.py:43-46, 221`: `MAX_POSITION_FRACTION = 0.95` caps a single
warrant allocation at 95% of buying power. With leverage=5 and Kelly recommending
`position_fraction > 1.0` (typical for half-Kelly on a 52% win-rate signal with
asymmetric reward), the cap fires routinely → 95% of cash into ONE 5x cert = 4.75x
gross exposure on the buying power. Memory states user wants 5x leverage and accepts
10-20% knockout risk, but Kelly with 0.95 cap on a 5x cert can produce >40% loss in
a single bad day (5×8% underlying drawdown). Recommend `MAX_POSITION_FRACTION = 0.30`
(or scale by `1/leverage` so effective exposure is bounded).

`portfolio/risk_management.py:764, 776`: `check_concentration_risk` hard-codes
`alloc_pct = 0.30 if strategy == "bold" else 0.15` of TOTAL portfolio value and
flags only at >40%. The 40% threshold ignores leverage — a 40% allocation in a 5x
warrant is 200% gross effective exposure. Per-instrument-class concentration limits
needed (warrants 10-15% max, crypto 25%, stocks 30%).

`portfolio/kelly_sizing.py:269-270, 323` vs `portfolio/risk_management.py:764-765`:
The two modules disagree on what 30%/15% means. `kelly_sizing` computes
`max_alloc = cash_sek * alloc_frac` (% of CASH), `check_concentration_risk` computes
`proposed_alloc = min(total_value * alloc_pct, cash)` (% of TOTAL_VALUE). When the
portfolio is heavily invested (cash low, total_value high), Kelly will return a tiny
`max_alloc` while concentration_check assumes the trade could be much larger. The
concentration flag is misleading.

`portfolio/portfolio_validator.py:214-243`: Check 8 (`avg_cost_usd` consistency)
recomputes weighted average from ALL historical BUY transactions on a ticker without
FIFO-pairing matched SELL portions. After a position is partially sold and then
re-bought at a different price, `avg_cost_usd` (as maintained by the portfolio_mgr,
typically reset on SELL-to-zero then re-set on next BUY) will legitimately differ
from "weighted avg of all historical BUYs". Validator will flag this as an error
even though it's correct. Either align the validator with the portfolio_mgr's
avg-cost methodology, or document and FIFO-track the pairing in the validator.

## P2 findings

`portfolio/monte_carlo_risk.py:204, 228`: `self._trading_days` hardcoded to 365 in
`PortfolioRiskSimulator`, but `compute_portfolio_var` derives per-ticker vol/drift with
`trading_days_for_ticker(ticker)` (252 for stocks). The drift was scaled to a 252-day
year, then `T = horizon_days / 365` rescales it back inconsistently. For MSTR (the
only Tier-1 stock), this slightly under-estimates 1-day vol. Small but a real bias.

`portfolio/exit_optimizer.py:718`: `fx_rate = agent_summary.get("fx_rate", 10.85)`
hardcodes a default that diverges from `FX_RATE_FALLBACK = 10.50` (per the
`risk_management._resolve_fx_rate` comment). Two modules, two fallback values, both
plausible but neither sanity-checked here. Same comment at line 54
(`MarketSnapshot.usdsek: float = 10.85`).

`portfolio/equity_curve.py:23` & `monte_carlo.py:34-36`: `ANNUALIZATION_DAYS = 365`
for the equity curve (24/7 portfolio), but `TRADING_DAYS_STOCKS = 252` lives in
monte_carlo. MSTR-only metrics on the equity curve get annualized at 365 when
MSTR really trades 252 days/year — Sharpe inflated by sqrt(365/252) ≈ 1.20 on
MSTR-dominated periods. Mostly cosmetic since the portfolio is mixed.

`portfolio/risk_management.py:382`: `triggered = current_price < stop_price if current_price > 0 else False`
silently returns `triggered=False` when current_price is 0/missing. A missing price
during a stop-out flash crash would show "stop not triggered" in dashboards — should
be `None` or surface the missing-data state.

`portfolio/trade_validation.py:60`: `min_order_sek = 1000.0` enforced at the
validation layer, but `kelly_sizing.py:326` ALSO has a `if rec_sek < 1000: rec_sek = 0`.
Two-layer enforcement with hardcoded constants — change one, forget the other,
silent drift. Pull to a shared constant.

`portfolio/kelly_metals.py:188`: `if wc is not None and 0 < wc < 1: win_rate = wc`
treats signal `weighted_confidence` (confidence in the signal's chosen DIRECTION) as
a win probability. They are not the same quantity. A signal with 100% confidence in
its BUY call still has a 47% historical accuracy. Source string says
`weighted_confidence ({win_rate:.1%})` which is misleading.

`portfolio/circuit_breaker.py:31-32`: `recovery_timeout` is a mutable instance
attribute that grows on every failed probe (`min(... * 2, max)`). On reset
(`record_success` line 51) it returns to `_base_recovery_timeout`. Looks fine, but
the `recovery_timeout` field is also exposed as a property (no, it isn't — `state`
is, but `recovery_timeout` isn't). Public read of `cb.recovery_timeout` returns
internal mutated state. Minor encapsulation issue.

`portfolio/exit_optimizer.py:567`: `if target <= market.price * 0.999: continue`
drops candidates below current price. For a SHORT/SELL exit (we hold a BEAR cert and
want to close at lower underlying), the entire candidate-generation logic is inverted
— targets should be BELOW current price. Compounds the P0 BEAR-cert direction bug.

`portfolio/cumulative_tracker.py:142-175`: `_find_closest_price` scans the entire
`snapshots` list per ticker per target_ts call. With 3 windows × N tickers, this is
O(snapshots × tickers × 3). At hourly snapshots over a year (~8760 entries) and 15
tickers, that's ~400k iterations per `compute_rolling_changes` call. Negligible at
current scale but will degrade.

`portfolio/decision_outcome_tracker.py:32-45`: `load_jsonl(DECISIONS_FILE)` reads
the ENTIRE decisions JSONL into memory then slices `[-max_entries:]`. Use
`load_jsonl_tail` from file_utils (it exists for exactly this).

## P3 findings

`portfolio/portfolio_mgr.py:55`: `path.with_suffix(f".json.bak{i - 1}" if i > 2 else ".json.bak")`
is fragile string templating. If a portfolio file isn't `.json` (e.g., test
fixture with no extension), `with_suffix` raises. Tests are presumably OK but
this is unnecessary cleverness.

`portfolio/risk_management.py:817`: `regime_mismatch` check on SELL trends-up uses
`volume_ratio < 1.5` but on BUY trends-down uses the same `< 1.5`. Different
"reversal volume" thresholds for entry vs exit might be defensible (asymmetric
liquidity preferences), but is undocumented. Minor.

`portfolio/equity_curve.py:495`: `losses = [t for t in trips if t["pnl_pct"] <= 0]`
includes break-even trades (pnl_pct == 0) as losses. Standard convention is
break-even = neither win nor loss.

`portfolio/instrument_profile.py:36-42`: `_SILVER_IGNORED` includes `"ml"` and
`"custom_lora"` which are noted as "disabled globally" — but per CLAUDE.md the
DISABLED_SIGNALS list is the source of truth. Duplicating the list in this profile
risks drift when one is updated and the other isn't.

`portfolio/monte_carlo.py:91`: `p_up = max(0.01, min(0.99, p_up))` silently
clamps. A signal with p_up=0.0 (certain DOWN) becomes 0.01 (almost certain DOWN).
Fine in practice but no logging of the clamp.

`portfolio/trade_guards.py:170`: `if consecutive >= 2: warnings.append(...)` —
the "warning" message is always emitted as a fresh warning every call, will be
counted in `by_guard` summary even though it carries no actionable information.
Mild noise in the dashboard.

`portfolio/exposure_coach.py:89`: `new_entries = not (zone == "danger" and regime in ("trending-down", "high-vol"))`
— only blocks new entries in the very narrowest case (danger zone AND specific
bearish regime). A "danger" zone with `ranging` regime allows new entries. May
be intentional but worth a comment.

## Cross-cutting observations

1. **Cross-process state isolation is broken in two places** (warrant_portfolio, portfolio_mgr).
   Either route everything through a single writer process with an SPSC queue, or
   wire `process_lock.py` (which already exists in the repo) into
   `update_state`/`save_warrant_state`. The 8-worker ThreadPoolExecutor concurrency
   has been addressed (BUG-219, PR-R4-4, etc.) but cross-process is the next layer.

2. **Stop-loss arithmetic is split across three modules** with different awareness
   of warrant barriers: `risk_management.compute_stop_levels` (ignores barriers),
   `exit_optimizer._apply_risk_overrides` (handles barriers), `kelly_metals`
   (loss-frac sized but doesn't know barrier distance). Memory note says NEVER
   place stops near MINI barriers — this needs to be a single chokepoint with
   `financing_level` as a mandatory parameter for any warrant stop.

3. **fx_rate handling is duplicated and inconsistent**: `risk_management._resolve_fx_rate`
   has the sanity-band + cache fallback chain; `monte_carlo_risk:408`,
   `exit_optimizer:718`, `MarketSnapshot.usdsek` default all bypass it. The
   1.0-stale-rate footgun the resolver was added to prevent is reachable from
   any path that doesn't import the resolver.

4. **Win/loss accounting is inconsistent between modules**: `equity_curve` uses
   pnl_pct for win classification but pnl_sek for profit_factor. `kelly_sizing`
   uses total_sek (with fees baked in). The result: every downstream "what's our
   win rate" answer disagrees with every other one by ~1-3 percentage points,
   and the difference is correlated with fee load.

5. **Kelly sizing is unbounded by leverage**. `kelly_metals` caps at 95% of
   buying power without scaling by cert leverage. The Patient strategy via
   `kelly_sizing.recommended_size` caps at 15% of cash but doesn't know if the
   ticker is a warrant or spot — same 15% applied to BTC or to a 10x warrant
   is wildly different effective risk.

6. **Drawdown circuit breaker has stale-feed blind spots**: per-ticker price
   missingness silently falls back to cost basis (line 211 warn but doesn't fail
   the reading), making the 50% block threshold dependent on the price feed
   being fully populated. A feed outage during a sharp drawdown could let
   trading continue.

## Files reviewed

Absolute paths in worktree:

- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/portfolio_mgr.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/portfolio_validator.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/risk_management.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/trade_guards.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/trade_validation.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/kelly_sizing.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/kelly_metals.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/equity_curve.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/circuit_breaker.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/monte_carlo.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/monte_carlo_risk.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/cost_model.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/exit_optimizer.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/exposure_coach.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/cumulative_tracker.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/decision_outcome_tracker.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/warrant_portfolio.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/trade_risk_classifier.py`
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/instrument_profile.py`

Cross-referenced (not in scope but cited):

- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/file_utils.py` (atomic_write_json behavior)
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/agent_invocation.py:883-920` (drawdown gate)
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/correlation_priors.py` (correlation matrix priors)
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/grid_fisher.py:336,1740` (BEAR cert handling)
- `Q:/finance-analyzer-worktrees/review-portfolio-risk/portfolio/fx_rates.py` (FX_RATE_FALLBACK)
