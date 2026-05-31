# Portfolio-Risk Review

Adversarial whole-file review of the portfolio-risk subsystem (read-only from
`Q:/fa-rev-0531`). Scope: `portfolio_mgr.py`, `risk_management.py`,
`trade_guards.py`, `equity_curve.py`, `monte_carlo.py`, `monte_carlo_risk.py`,
`kelly_sizing.py`, `kelly_metals.py`, `circuit_breaker.py`,
`trade_validation.py`, `portfolio_validator.py`, `warrant_portfolio.py`,
`cost_model.py`.

Findings reported at confidence >= the project's bar, scored against the
stated invariants (atomic I/O, leverage correctness, drawdown breaker actually
halting entries, no silent fail-open on risk gates). Many prior-review fixes
(C8/C7 locks, FX fallback chain, streaming peak, Sortino-by-total-N, FIFO net
fees) are present and correct; those are not re-listed.

---

## P0 — direct money loss / state corruption / risk-gate silently failing open

### P0-1 `warrant_portfolio.py:199-280` — cross-process lost-update on the LEVERAGED warrant state (no lock, read-modify-write)

`record_warrant_transaction()` does `state = load_warrant_state()` →
mutate holdings/transactions in memory → `save_warrant_state(state)`
(`atomic_write_json`). The write is atomic (no torn file), but there is **no
lock of any kind** — not the in-process `threading.Lock` that
`portfolio_mgr.update_state` uses, and nothing cross-process.

Causal chain: the warrant book is the LEVERAGED money (5x/10x). It is touched
by the metals loop (separate process — `data/metals_loop.py`), by Layer 2
(`claude -p` subprocess) and by reporting. Two concurrent
`record_warrant_transaction` calls (or one of those racing a metals-loop write)
both load the same `state`, each appends its own txn / averages-in its own
units, and the second `save_warrant_state` wins — silently dropping the first
transaction *and* leaving `holdings[config_key].units` / `entry_price_sek`
reflecting only one of the two BUYs. Because `warrant_pnl` derives stop-loss
reference and knockout proximity from `entry_price_sek` /
`underlying_entry_price_usd`, a lost average-in corrupts both the position size
and the stop reference on a leveraged instrument. This is the exact
"load, mutate in memory, write back without locking → lost update" failure the
invariants call out, on the highest-leverage book in the system.
→ Route every warrant mutation through a locked read-modify-write. Easiest:
add an `update_warrant_state(mutate_fn)` mirroring
`portfolio_mgr.update_state` (per-file `threading.Lock`) AND wrap the
load+save in a cross-process file lock (`file_utils.jsonl_sidecar_lock`-style
sidecar, or a `.lock` on `portfolio_state_warrants.json`) since the metals loop
is a different OS process where a `threading.Lock` is useless.

### P0-2 `portfolio_mgr.py:29-159` — `update_state` lock is in-process only; cross-process writers (main loop / metals loop / Layer 2) can still lose updates on Patient/Bold state

`_state_locks` are `threading.Lock` objects. The docstring on `update_state`
claims it makes the read-modify-write "safe" to "prevent concurrent callers
from overwriting each other's mutations," and the C8 comment frames this as the
concurrency fix. That guarantee holds only *within one process*. CLAUDE.md
explicitly states portfolio state is written by three independent OS processes
(main loop, metals loop, Layer 2 subprocess). A `threading.Lock` provides zero
mutual exclusion across processes.

Causal chain: Layer 2 (`agent_invocation.py` subprocess) loads Patient state,
debits `cash_sek`, appends a BUY txn, writes. If the main loop or another
subprocess does its own load→mutate→`atomic_write_json` of the same file in the
overlapping window, one full state object overwrites the other — a whole
transaction (cash debit + holding) is lost, or duplicated, with no torn file to
flag it. The `atomic_write_json` rename only prevents *partial* files; it does
nothing about the **stale-read / last-writer-wins** race. The main loop's own
`save_state` at `main.py:796` is currently gated behind
`if not STATE_FILE.exists()`, which limits exposure today, but `update_state`
is presented as a general safe primitive and the lock label oversells it.
→ Add a cross-process file lock (sidecar `.lock` + `msvcrt.locking`/`flock`,
the primitive already in `file_utils.jsonl_sidecar_lock`) around
`_load_state_from`+`_atomic_write_json` in `update_state`, `save_state`,
`save_bold_state`. At minimum, downgrade the docstring/comments so callers do
not assume cross-process safety that does not exist.

### P0-3 `risk_management.py:935-1001` + `kelly_*`/`trade_guards` — the drawdown circuit breaker is computed but nothing in this subsystem ENFORCES it on new entries

`check_drawdown()` returns `{"breached": bool, ...}` with a correct, fail-safe
computation (NaN guard, streaming true peak, conservative-but-warned cash
fallback). But within the reviewed risk subsystem nothing consumes `breached`
to block a BUY. `compute_all_risk_flags()` (the aggregator Layer 2 reads) calls
concentration / correlation / regime / ATR-proximity checks — it never calls
`check_drawdown` and emits no `drawdown` flag. `trade_guards.should_block_trade`
only inspects cooldown / position-rate "block" warnings. Kelly sizing
(`kelly_sizing.recommended_size`, `kelly_metals.recommended_metals_size`) never
references drawdown state.

Causal chain: the invariant is "drawdown circuit breaker must actually halt new
entries when tripped." If the only enforcement is a soft signal in the Layer 2
prompt (an LLM that may or may not honor it), then on a >20% drawdown the system
can keep opening positions — the breaker is computed and displayed but not wired
to a hard go/no-go. This is a risk-gate that fails *open*.
→ Surface `check_drawdown(...)["breached"]` into `compute_all_risk_flags` as a
`severity:"block"` flag (or a dedicated `is_entry_blocked()` consumed by the
execution path), and have the trade-execution code refuse BUYs when breached.
Verify with a test that a breached drawdown produces a hard block, not just a
warning string. (If enforcement lives in an out-of-scope execution module,
confirm it reads `breached` — I found no such consumer in the 13 files.)

---

## P1 — wrong risk number / over-sizing / silent degradation

### P1-1 `kelly_metals.py:211-221` — leveraged Kelly position fraction can be wildly oversized; cap is 95% of buying power on a 5x cert

Step 4 computes `cert_loss_frac = avg_loss * leverage / 100` then
`position_fraction = half_kelly / cert_loss_frac`. With the module defaults
(XAG `avg_loss=2.43%`, `leverage=5` → `cert_loss_frac=0.1215`) and a healthy
half-Kelly of, say, 0.12, `position_fraction = 0.12/0.1215 ≈ 0.99`, clamped to
`MAX_POSITION_FRACTION = 0.95`. So the "Kelly-safe" recommendation is to put
**95% of buying power into a single 5x silver warrant**.

Two problems: (a) the formula divides half-Kelly (a fraction of *capital* to
risk) by the *per-cert loss fraction*, which converts a "risk 12% of capital"
edge into "deploy ~99% of capital" — the leverage is effectively applied as a
*sizing multiplier on notional* rather than constraining it. (b) The 0.95 cap
directly contradicts the user's documented risk posture and the metals stop
rules; a 95% allocation to one leveraged cert is a knockout-risk bomb. Note the
metals loop's prior fixed sizing was "30% of buying power" (per this module's
own docstring) — the new math routinely exceeds 3x that.
→ Re-derive: leveraged Kelly fraction of capital should be roughly
`f_capital = half_kelly / leverage` (so 5x leverage *reduces* the cash
deployed, not amplifies it), and the hard cap should reflect user posture
(e.g. <=0.30, not 0.95). Add a unit test asserting that increasing `leverage`
*decreases* `position_sek` for fixed edge.

### P1-2 `kelly_sizing.py:100-128, 254-262` — consensus-accuracy win-prob path has no sample-size gate; a tiny-sample 100% accuracy flows straight into full Kelly

`_get_signal_accuracy` returns `acc_data["consensus"]["accuracy"]` with only a
`> 0` check — no minimum sample count. `kelly_metals._get_ticker_accuracy`
correctly requires `total >= 30`; the per-ticker signal path in
`_get_ticker_signal_accuracy` requires `samples >= 5`. But when those return
`None`, `recommended_size` falls back to `_get_signal_accuracy`, which will
happily pass a `win_prob` derived from a 5-sample consensus (CLAUDE.md lists
"COT Positioning — 100% 1d, 5 sam"). `kelly_fraction(0.9, ...)` then yields a
huge fraction; half-Kelly of that, times cash, is the recommended size (capped
only at 15%/30% max-alloc, which is itself large).
→ Gate `_get_signal_accuracy` (and the consensus fallback in
`recommended_size`) on a minimum sample count (>=30, matching kelly_metals);
return `None`/0.5 below threshold so sizing stays at the conservative floor.
"Win-rate estimated from too few samples" is one of the named Kelly footguns.

### P1-3 `monte_carlo_risk.py:288-377` — VaR/CVaR/exposure assume all positions are LONG; a short would invert P&L and the gates would understate loss

`PortfolioRiskSimulator` filters on `shares != 0` (`:188`, so negatives are
kept), but `portfolio_pnl` computes `shares * price * (exp(return) - 1)` and
`total_exposure`/`drawdown_probability` use `sum(shares * price)`. For a short
(`shares < 0`) the P&L sign is inverted (a price rise should *lose* money for a
short, but `exp(return)-1 > 0` times negative shares yields a "loss" on an up
move and a "gain" on a crash) and `total_value`/`total_exposure` go negative,
so `drawdown_probability` returns 0.0 (its `total_value <= 0` guard) — i.e. the
gate silently reports zero downside for a short book. `compute_portfolio_var`
upstream filters `shares <= 0` (`:415-417`) so shorts are dropped entirely
there, meaning a short position contributes **nothing** to portfolio VaR.
Simulated portfolios may be long-only today, but the warrant/cert world has
BEAR certs and the metals subsystem shorts; any short exposure is invisible to
this VaR.
→ Decide explicitly: either assert/document long-only and reject `shares < 0`
loudly, or model shorts correctly (P&L `= shares * price * (exp(r)-1)` is
actually correct for signed shares; the bug is the `total_exposure`/`drawdown`
using signed sum — use `sum(abs(shares)*price)` for exposure and gross notional
for the drawdown denominator). Don't silently drop shorts from VaR.

### P1-4 `monte_carlo.py:338-364` / used by VaR drift — directional probability has no per-ticker sample/accuracy gate and is fabricated from `weighted_confidence`

`_get_directional_probability` maps action+confidence to `0.5 ± conf*0.3`. This
drift feeds both `simulate_ticker` and (via `compute_portfolio_var`) the VaR
drift term. A confidently-wrong signal pushes drift the wrong way and shrinks
modeled tail loss on the side you're actually exposed to, making VaR optimistic
exactly when a strong-but-wrong signal is in play. There's no clamp tying drift
magnitude to realized per-ticker accuracy.
→ For *risk* (VaR/CVaR) prefer zero or conservative drift rather than
signal-implied drift; reserve signal drift for the directional MC bands. At
minimum gate `p_up` deviation from 0.5 on sample-backed accuracy.

### P1-5 `risk_management.py:742-801` — `check_concentration_risk` uses `avg_cost_usd` as the price fallback, understating concentration of a winner; threshold 40% only ever returns a soft "warning"

`existing_price = signals.get(ticker,{}).get("price_usd", existing.get("avg_cost_usd",0))`.
When the live price is missing it values an existing position at entry cost. For
a position that has run up, this *understates* current concentration, so a
top-up that would actually breach 40% can slip under. Separately, the only
output is `severity:"warning"` — concentration is never a hard block (ties back
to P0-3: the gate informs but does not enforce).
→ When live price is missing, skip the check or fail-safe (treat as
concentrated / block), don't silently substitute stale entry cost; and decide
whether 40%+ concentration should hard-block for at least the Patient strategy.

---

## P2 — correctness / robustness, lower blast radius

### P2-1 `risk_management.py:466` — `compute_probabilistic_stops` annualizes ATR with `sqrt(trading_days/14)` but the branch only sets `trading_days` for crypto/metals via `inst_type in ("crypto","metals")` while warrants are classified `inst_type="warrant"` (`:454`), so XAG/XAU warrants get the 252 stock factor

`inst_type` becomes `"warrant"` for XAG/XAU (`:452-456`), then
`trading_days = 365.0 if inst_type in ("crypto","metals") else 252.0` →
metals warrants use 252, undercounting annualized vol for a 24/7-priced
underlying. Inconsistent with `monte_carlo.trading_days_for_ticker` (metals=365)
and the rest of the module.
→ Treat `"warrant"` (XAG/XAU underlying) as 365 for vol annualization, or map
to the underlying's class.

### P2-2 `monte_carlo.py:264-335` — default `seed=None` is correct, but `simulate_ticker` re-uses the *same* `seed` across every horizon loop iteration when a seed is passed

When `seed` is provided (tests, reproducibility), every horizon builds a fresh
`MonteCarloEngine(seed=seed)` with identical seed → the horizons share the same
underlying normal draws, so `expected_return_1d` and `expected_return_3d` are
perfectly correlated rather than independent samples. Harmless in production
(`seed=None`) but will mask variance/independence in any seeded analysis or
test that compares horizons.
→ Offset the seed per horizon (`seed + h_index`) as `simulate_all` already does
per ticker.

### P2-3 `equity_curve.py:188` & `:552` — annualized return uses `pow(last/first, 1/years)` on raw equity which is fine, but `days_elapsed >= 1` gate means a <1-day curve silently reports `annualized_return_pct=None` while `total_return_pct` is populated; Calmar then can't compute

Not a bug per se, but the asymmetry means early-life portfolios (the system
started 2026-02-11; curves can be short after a reset) get partial metrics with
no signal as to why Calmar/annualized are `None`. Minor; flag only because
risk-adjusted comparison (`compare_strategies`) silently omits `sharpe_leader`
when either side is `None`.
→ Document/handle the short-curve case explicitly.

### P2-4 `trade_guards.py:103-128, 379-385` — TOCTOU between the `check_overtrading_guards` read and `record_trade` write across processes

State is loaded under `_state_lock` (in-process) at `:126-127`, the guard
decision is made on that snapshot, and `record_trade` later writes under the
same in-process lock. Across the metals loop / Layer 2 process boundary the lock
doesn't apply, so two processes can both pass the cooldown/position-rate gate on
the same stale snapshot and both trade — the rate limit can be exceeded by the
number of concurrent processes. Lower severity than P0 because these are guards
(soft) and the window is small, but it's the same class as P0-1/P0-2.
→ Cross-process lock on `trade_guard_state.json`, or accept-and-document that
guards are per-process best-effort.

### P2-5 `kelly_sizing.py:39-52` & `kelly_metals.py:208-209` — full Kelly is computed and exposed (`kelly_pct`), and `recommended_size` uses *half* Kelly (good), but `kelly_metals` Step 4 applies the leverage transform to `half_kelly` and then can clamp to 0.95 (see P1-1) — the "half" safety is undone by the leverage division

Cross-reference to P1-1; noting here that the half-Kelly conservatism present in
`kelly_sizing` is effectively neutralized in `kelly_metals` by the leverage
math. Keep the half-Kelly, fix the leverage transform.

### P2-6 `cost_model.py` / `equity_curve._pair_round_trips` — round-trip P&L is net of recorded `fee_sek` (good), but the MC stop / Kelly avg-loss estimates do NOT subtract `cost_model` round-trip costs

`WARRANT_COSTS.round_trip_pct()` is ~1.0% (40bps half-spread + 10bps slippage,
×2). Kelly `avg_loss`/`avg_win` come from raw price-% outcomes
(`kelly_metals._get_outcome_stats` uses `change_pct` directly; `kelly_sizing`
uses `pnl_pct` which is gross price move per the P0-6 note in
`equity_curve.py:393-395`). So the edge fed to Kelly ignores ~1% round-trip cost
on warrants — on small-edge metals scalps that's a meaningful overstatement of
the payoff ratio `b`, inflating recommended size.
→ Net `cost_model.round_trip_pct()` out of `avg_win`/into `avg_loss` before
computing the Kelly fraction for the leveraged instruments.

---

## Notes / verified-OK (not findings)

- `file_utils.atomic_write_json` is correct (tempfile + fsync + `os.replace`,
  symlink-resolved). Torn writes are not a risk for single-process writers.
- `check_drawdown` NaN/Inf fail-safe (`:291-303`) and streaming true-peak
  (`_streaming_max`) are correct and well-guarded; the cash-only fallback is
  WARNING-surfaced rather than silently optimistic.
- `equity_curve` Sortino divides by total N (`:248-249`) — standard; Sharpe
  units are internally consistent (pct→decimal). Annualization=365 matches the
  24/7 loop.
- `monte_carlo_risk` C9 fix (Gaussian marginals via `norm.ppf`, t-copula for
  dependence) is correct; `_nearest_psd` Higham clip is sound.
- `circuit_breaker` (data-source breaker, NOT the portfolio drawdown breaker)
  is a correct thread-safe state machine with single HALF_OPEN probe and capped
  exponential backoff. Name collision with "drawdown circuit breaker" is a
  documentation hazard but not a bug.
- `kelly_fraction` correctly clamps negative edge to 0 and caps at 1.
- `warrant_pnl` leverage math (`underlying_change * leverage`, floored at 0 for
  knockout) is correct for a single position; the problem is the unlocked
  *state* path (P0-1), not the per-position math.
