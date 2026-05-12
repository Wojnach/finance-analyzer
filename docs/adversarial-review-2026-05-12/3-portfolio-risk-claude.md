# Claude adversarial review: portfolio-risk (2026-05-12)

## Summary

Re-read the full in-scope set (portfolio/{portfolio_mgr,portfolio_validator,trade_guards,
trade_validation,trade_risk_classifier,risk_management,monte_carlo,monte_carlo_risk,
equity_curve,exit_optimizer,kelly_sizing,kelly_metals,exposure_coach,warrant_portfolio,
cost_model,instrument_profile,stats}.py and portfolio/strategies/*.py).

The four P0s from 2026-05-11 are **all still present unchanged**: three places enforce
`min_order_sek = 500` instead of the Avanza-mandated 1000 SEK; the ATR-stop function
hard-caps ATR at 15% even on 5x certs where the user explicitly accepts 10-20% knockout
risk and where the resulting stop can be inside the MINI financing barrier. Nothing in
the May 12 diff cluster touched portfolio-risk — these blockers carry over verbatim.

New observations this pass (not flagged on May 11):
- `kelly_metals.py:229` separately enforces its own `MIN_TRADE_SEK = 500.0` floor; this
  is a fourth copy of the same constant, not a third (the 2026-05-11 partition counted
  trade_validation, kelly_sizing, kelly_metals — but kelly_metals enforces it at the
  *check site* line 229, not just at the constant line 44).
- `monte_carlo_risk.compute_portfolio_var:419` bypasses `_resolve_fx_rate` while
  `risk_management._resolve_fx_rate` exists in the same package — duplicate FX
  fallback paths, only one of which validates the sanity band [7,15].
- `exit_optimizer._compute_pnl_sek:339` charges costs **once on exit** so EVs are
  overstated by ~half the round-trip; this confirms the May 11 finding and is still
  unfixed.
- `portfolio_mgr._rotate_backups:54-60` uses `path.with_suffix(".json.bakN")` which
  silently mangles paths whose stem contains dots — `portfolio_state_warrants.json`
  doesn't today but a future filename like `portfolio_state.v2.json` would mis-rotate.

## P0 — Blockers

- **portfolio/trade_validation.py:32** — `min_order_sek: float = 500.0` (unchanged
  since 2026-05-11). The CLAUDE.md project rule and `.claude/rules/metals-avanza.md`
  both state **"Every Avanza order must be ≥1000 SEK to avoid minimum courtage"**.
  Callers that take the default ship 500-999 SEK orders, which pass `validate_trade`,
  then either fall below the courtage threshold (silent revenue leak — flat 1 SEK
  courtage = 0.1-0.2%) or get rejected later by `avanza/trading.MIN_ORDER_SEK` after
  the orchestrator has cleared cash and cancelled stops. Fix: default to 1000.0; ideally
  import the canonical constant from `portfolio.avanza.trading`.

- **portfolio/kelly_sizing.py:326** — `if rec_sek < 500: rec_sek = 0` (unchanged).
  Same defect as above, copy-pasted into the Kelly path. Half-Kelly on a small Bold
  cash bucket returns e.g. 750 SEK → consumer places → Avanza either rejects or eats
  the courtage on a sub-minimum lot.

- **portfolio/kelly_metals.py:44 + line 229** — `MIN_TRADE_SEK = 500.0` and
  `if position_sek < MIN_TRADE_SEK: position_sek = 0.0`. This routes to the **metals
  warrant ladder** where the courtage minimum bites hardest. 500-999 SEK ladders get
  placed, billed flat 1 SEK courtage (0.1-0.2%), tagged as "Marja Folcke probe" trades
  with no profitability. Fix: raise to 1000.0; preferably import the canonical broker
  minimum so all three sites move in lockstep.

- **portfolio/risk_management.py:374** — `atr_pct = min(atr_pct, 15.0)` (unchanged).
  For a MINI silver warrant with underlying running 4% daily ATR × 5x leverage, the
  user explicitly accepts 10-20% knockout risk per `memory/feedback_risk_tolerance.md`.
  Capping at 15% generates stops too tight to survive an intraday wick. Worse: when
  `entry_price * (1 - 2 * atr_pct/100)` falls below the MINI financing level, the
  function **never compares the two** and emits a stop that's already inside the
  knockout barrier — directly violating the
  `memory/feedback_mini_stoploss.md` rule ("Never place stop-losses near MINI warrant
  barriers"). Fix: pass `financing_level` and floor `stop_price = max(stop_price,
  financing_level * 1.03)`; raise the cap to 25% to accommodate 5x certs.

## P1 — High

- **portfolio/risk_management.py:902-911** (`check_atr_stop_proximity`) — flags
  `distance_in_atr < 1.0`. For a $30 entry with 4% ATR, the trigger threshold is one
  daily wick from the stop — by then the stop is already going to fire on any normal
  intraday move. The flag is reactive, not preventive. Combined with the `atr_pct =
  min(atr_pct, 15.0)` cap, MINI cert stops trip silently before this proximity flag
  even activates. Fix: trigger at 1.5-2.0x ATR.

- **portfolio/risk_management.py:466-468** (`compute_probabilistic_stops`) — clamps
  `stop_price = entry_price * 0.01` when `stop_price <= 0`. A 99%-drop "stop" trivially
  never fires, silently disabling the probabilistic stop for any leveraged warrant whose
  ATR×2 puts stop at or below zero. Caller reads `stop_hit_prob = 0.0` as "safe" and
  skips the de-risk path. Fix: return `None` for the position and log; do not fabricate
  a stop level.

- **portfolio/risk_management.py:292-304** — NaN guard returns `breached=True,
  current_drawdown_pct=100.0` (fail-safe), but `peak_value` is rounded to `0.0` when
  non-finite (line 301). However, on the same-call path the function does **not write
  the sentinel back to `_peak_cache`** (it `return`s before line 108). Re-reading more
  carefully: the NaN return path bypasses the cache write entirely — good. The real
  remaining bug is that the **cache may still hold a stale finite peak from a prior
  call** while current_value is NaN, and the next call after recovery will reuse that
  stale offset. Lower severity than 2026-05-11 review credited but still worth a
  comment.

- **portfolio/monte_carlo_risk.py:419** — `fx_rate = agent_summary.get("fx_rate",
  FX_RATE_FALLBACK)`. **Bypasses** `_resolve_fx_rate`. A stale agent_summary that
  legitimately contains `fx_rate = 1.0` (the original P1-15 root cause) understates
  VaR_SEK and CVaR_SEK by ~10x and Layer 2 receives reassuring numbers right when
  the FX feed broke. The fix exists in the same package — just import and call
  `_resolve_fx_rate`. Same blind spot exists at **exit_optimizer.py:719**
  (`compute_exit_plan_from_summary` default 10.85) and **exit_optimizer.py:54**
  (`MarketSnapshot.usdsek` default 10.85).

- **portfolio/exit_optimizer.py:303-340** (`_compute_pnl_sek`) — deducts `cost =
  costs.total_cost_sek(exit_value)` **once on exit**. `entry_value = entry_price_sek *
  qty` carries no buy-side cost. For warrants at 40 bps half-spread + 10 bps slippage =
  1.0% round-trip, every quantile's EV is overstated by ~50 bps × position. A small-edge
  exit recommendation can flip sign once the entry-side cost is included. Fix: either
  subtract `costs.total_cost_sek(entry_value)` once in `_compute_pnl_sek`, or document
  loudly that `entry_price_sek` MUST be the cost-grossed-up price (it isn't, per
  `equity_curve._pair_round_trips` which subtracts buy_fee_share separately).

- **portfolio/exit_optimizer.py:617-621** — Hold-to-close EV computed from **5
  percentiles** of terminal prices, not the full 5000-path distribution:
  `np.mean(terminal_pnls)` over the 10/25/50/75/90 quantile P&Ls. That's a coarse
  trapezoidal integral biased toward the median; in fat-tailed cases (warrants near
  barrier) the true tail mean is far lower. Fix: vectorise the P&L formula and apply
  over all 5000 terminal samples.

- **portfolio/exit_optimizer.py:373-378, 397-400, 446** — Knockout buffer hard-coded
  at `financing_level * 1.03` (3%). The user accepts **10-20% knockout risk** per
  `memory/feedback_risk_tolerance.md`; the override at line 446 then **forces market
  exit** at every wick inside 3%. This is exactly the over-aggressive de-risking the
  feedback memory warns against. Fix: thread buffer through `instrument_profile` and
  default to 5-8% for 5x certs.

- **portfolio/trade_guards.py:140-167** — Cooldown comparison uses wall-clock
  `datetime.now(UTC)` against ISO timestamps from `data/trade_guard_state.json`. If
  NTP slews the clock backwards, or an operator manually edits the file, the cooldown
  is silently bypassed — the `except (ValueError, TypeError): pass` swallows malformed
  timestamps without falling back to "block" behaviour. The C4 wiring check (line 363)
  only fires AFTER portfolios already have transactions, so the first BUY between
  process restart and the first `record_trade` call runs unguarded. Fix: reject reads
  where `mtime > now`; on parse failure, treat as cooldown-active rather than passing
  through.

- **portfolio/risk_management.py:765-770** (`check_concentration_risk`) — computes
  `proposed_alloc = min(total_value * alloc_pct, cash)`. If `cash == 0` (fully
  invested Bold portfolio) proposed_alloc is 0 → concentration_pct excludes any new
  entry — the check **never fires** when it's most needed. Combined with the
  trade_validation.max_cash_pct check that scales with `cash_available` not portfolio
  value, an exhausted-cash portfolio passes both checks while pyramiding into a single
  ticker. Fix: drop the `min(..., cash)` clamp and surface a separate "insufficient
  cash" diagnostic.

- **portfolio/trade_guards.py:50-69** (`_portfolios_have_transactions`) — reads three
  separate state files via `load_json` with `default={}`. If `portfolio_state.json` is
  mid-write (atomic rename window), `load_json` returns `{}` and the C4 wiring check
  returns False, suppressing the warning. The retry inside `load_json` mitigates this,
  but the side-effect is that the C4 sanity check is **silent during state writes**.
  Not actively harmful but worth a comment.

## P2 — Medium

- **portfolio/trade_validation.py:75-81** — `cash_pct = order_value / cash_available
  × 100` caps at 50% of **cash**, not of **portfolio value**. After several earlier
  buys, `cash_available` shrinks proportionally so a 50% cash check passes even as
  the position becomes 80%+ of portfolio. The 40% portfolio-level concentration check
  in `risk_management.check_concentration_risk` exists but is not chained into
  `validate_trade`. Fix: pass `portfolio_value`, compute against that.

- **portfolio/equity_curve.py:494-495** — `losses = [t for t in trips if t["pnl_pct"]
  <= 0]` classifies break-even trades (pnl=0) as losses. Skews
  `max_consecutive_losses` upward. Cosmetic but persists in operator dashboards.
  Fix: `< 0`.

- **portfolio/monte_carlo.py:392-399** (`simulate_all`) — `seed + i` where `i =
  enumerate(tickers)`. Adding a new ticker to the focus list shifts every subsequent
  ticker's seed → reruns produce different VaR for unrelated tickers. Reproducibility
  brittle. Fix: hash ticker name into the seed: `seed ^ zlib.crc32(ticker.encode())`.

- **portfolio/monte_carlo_risk.py:200** — `positions.get("shares", 0) != 0` admits
  negative shares. The cost / P&L path assumes long-only (`max(exit_warrant_sek, 0)`
  in exit_optimizer, exposure ceiling, concentration). A negative-shares entry
  slipping in (corrupted state) produces nonsensical VaR. Fix: `> 0`.

- **portfolio/warrant_portfolio.py:80** — `if not holding or not current_underlying_usd
  or not fx_rate: return None`. Python truthiness: `current_underlying_usd = 0.0` and
  `fx_rate = 0.0` both short-circuit. Caller silently records `pnl=None`, total_value
  stays at 0, position invisible to operator unless they read the raw dict. Fix:
  explicit `is None` and `<= 0` checks with logged warning.

- **portfolio/trade_risk_classifier.py:81** — `regime_score = _REGIME_SCORES.get(
  regime_lower, 0)`. Silent permissive fallback: unknown regime string (typo, new
  label, missing JSON field) treated as "trending-up" / zero risk. Fix: log unknown
  regimes and default to 2 (ranging) — the safer prior.

- **portfolio/exposure_coach.py:71-73** — `zone = market_health.get("zone",
  "healthy")` and `score = market_health.get("score", 50)`. Permissive fallback:
  malformed `market_health` dict (missing `zone`) yields a 1.0 ceiling. Combined
  with Kelly sizing using `exposure_ceiling` as a multiplier on cash, upstream data
  corruption **maximises** position size. Fix: default to "caution" (0.6).

- **portfolio/monte_carlo_risk.py:381-388** — `loss_threshold = -total_value *
  threshold_pct / 100.0`. `total_value` is gross exposure; the metric name
  `drawdown_5pct_prob` suggests portfolio drawdown but the denominator is exposure,
  not capital. Mismatch when cash is large relative to positions.

- **portfolio/risk_management.py:838-870** (`check_correlation_risk`) — only inspects
  *current* holdings. A BUY of ETH-USD one minute after a SELL of BTC-USD will not
  flag correlation risk because the SELL already drained the holding. The
  `trade_guards.record_trade` history is the right place to detect this. Fix: check
  recent SELL records for correlated tickers in the last cooldown window.

- **portfolio/portfolio_mgr.py:54-60** — `_rotate_backups` uses `path.with_suffix(...)`.
  For a path whose **stem contains a dot** (e.g. a hypothetical
  `portfolio_state.v2.json`), `with_suffix` only replaces the final `.json` segment,
  silently producing paths like `portfolio_state.v2.json.bak2` when the intent was a
  numbered rotation. Current filenames don't trigger this but it's a footgun.

- **portfolio/portfolio_mgr.py:108-113** — `_save_state_to` acquires the lock, calls
  `_rotate_backups` then `_atomic_write_json`. If `_atomic_write_json` fails *after*
  `_rotate_backups` rotated `.bak → .bak2`, the original file is still intact (atomic
  write didn't replace it) but `.bak` was overwritten with the same content. Net
  effect: one fewer historical backup than expected after a write failure. Cosmetic.

## P3 — Low

- **portfolio/exit_optimizer.py:54** — `MarketSnapshot.usdsek: float = 10.85`.
  Hardcoded fallback. Same reasoning as the May 11 P1-15 in risk_management.
  Low impact because most callers pass live fx, but a degenerate `MarketSnapshot()`
  constructor call gets 10.85.

- **portfolio/equity_curve.py:228-249** — `daily_vol` computed at line 231 from
  `daily_rets` (percent) is unused afterwards (line 232 stores into result, but
  Sharpe re-derives `daily_std_dec` from `daily_rets_dec`). Two parallel std
  computations on the same data, one in % and one in decimal. Harmless but indicates
  piecemeal editing.

- **portfolio/portfolio_validator.py:130** — Holdings-mismatch tolerance is
  `relative_diff < 0.01` only when `actual_shares == 0`. A dust position (0.0001
  shares after rounding) bypasses validation only if it was also removed from
  holdings; otherwise the error fires for a tiny but mathematically-real diff.

- **portfolio/kelly_metals.py:215-217** — `cert_loss_frac = avg_loss * leverage /
  100.0` then `position_fraction = half_kelly / cert_loss_frac`. For leverage=5,
  avg_loss=2.4%, `cert_loss_frac = 0.12`, so position_fraction = `half_kelly / 0.12`.
  With a moderate edge this exceeds 1.0 and is clipped at MAX_POSITION_FRACTION=0.95
  — the half-Kelly safety margin is **lost**. Effectively the leverage-adjustment
  inverts Kelly back into full-Kelly whenever the cert_loss_frac is small. Documented
  intent vs effective output diverge.

- **portfolio/cost_model.py:67** — `spread_bps=40.0` for warrants is conservative for
  XAU but optimistic for XAG (live order books show 0.6-1.5% half-spreads on Avanza
  certs). Per-ticker cost models would be more accurate.

- **portfolio/strategies/orchestrator.py:116** — `time.sleep(0.5)` is the floor on
  tick granularity. A strategy with `poll_interval_seconds < 0.5` cannot tick faster;
  fine for current strategies but the contract isn't documented.

- **portfolio/monte_carlo.py:175-180** — Odd-`n_paths` extra path uses
  `rng.standard_normal(1)` AFTER the antithetic block already consumed half the
  stream; with `seed=None`, the engine is non-deterministic even if re-used.

- **portfolio/strategies/elongir_strategy.py:78** and
  **golddigger_strategy.py:94** — both fall back to `fx = 10.5` (not 10.50, and
  not the FX_RATE_FALLBACK constant of 10.50). Use `portfolio.fx_rates.FX_RATE_FALLBACK`.

- **portfolio/instrument_profile.py** — defines `typical_daily_range_pct: 5.0` for
  silver, `2.9` for gold. CLAUDE.md and accuracy stats suggest XAG averages 4-6%; 5.0
  is fine but the value is hard-coded and not derived from rolling ATR.

## Status of prior P0s (2026-05-11)

| Prior P0 | Status |
|----------|--------|
| `trade_validation.py:32` default `min_order_sek = 500.0` | **STILL PRESENT** — no diff |
| `kelly_sizing.py:326` `if rec_sek < 500: rec_sek = 0` | **STILL PRESENT** — no diff |
| `kelly_metals.py:44` `MIN_TRADE_SEK = 500.0` (and enforcement at line 229) | **STILL PRESENT** — no diff |
| `risk_management.py:374` `atr_pct = min(atr_pct, 15.0)` | **STILL PRESENT** — no diff |

None of the four blockers from 2026-05-11 has been addressed. The May 12 session
appears to have touched signals-core, infrastructure docs, metals state, and morning
briefing files (per gitStatus) but not the portfolio-risk subsystem.

## Tests missing

- **Min-order-size invariant**: Parametrize `trade_validation.validate_trade(...,
  volume=X, price=Y)` over (warrant, stock, crypto) and assert `valid=False` for
  `X*Y < 1000`. Currently no test covers this defaulted path.

- **ATR stop vs MINI financing barrier**: Feed `compute_stop_levels({ticker:
  {shares: N, avg_cost_usd: E}}, signals_with_atr)` with a financing level passed
  in, and assert `stop_price >= financing_level * 1.03`. The function doesn't take
  financing level today — the test would force the API change.

- **Drawdown breaker NaN guard cache integrity**: Write a JSONL with `NaN` values,
  call `check_drawdown`, and verify (a) the function returns `breached=True,
  current_drawdown_pct=100.0` and (b) the next call with a valid current_value
  doesn't carry forward a poisoned `_peak_cache` entry.

- **Kelly leverage inversion**: `recommended_metals_size(leverage=5, avg_loss=2,
  win_rate=0.6, ...)` — assert `position_fraction <= half_kelly_pct`. Today this
  fails: leverage amplifies the position fraction above half-Kelly.

- **Cooldown bypass on clock jump**: Backdate `ticker_trades[…]` to a *future*
  timestamp and verify the guard rejects the trade (not silently passes via the
  bare `except`). Also test with malformed timestamp strings.

- **VaR with non-long shares**: Assert `compute_portfolio_var({shares: -1, ...})`
  rejects or zeroes shorts rather than computing long-only VaR on a short.

- **Exit-optimizer round-trip cost**: Assert the EV of an immediate buy + market
  sell at the *same price* is **strictly negative**, not zero. Today
  `_compute_pnl_sek` only deducts exit-side cost, so this round-trip evaluates to
  `-exit_cost` but should be `-(entry_cost + exit_cost)`.

- **Hold-to-close EV from full path distribution**: Compare `hold_candidate.ev_sek`
  computed from 5 percentiles vs. from the full 5000 terminal samples; assert
  agreement within 1% on a thin-tailed scenario and divergence on a fat-tailed
  one (verifying the quantile-mean bias).

- **`monte_carlo_risk.compute_portfolio_var` FX sanity gate**: Feed an
  `agent_summary` with `fx_rate = 1.0` and assert the SEK-denominated VaR uses
  `FX_RATE_FALLBACK` or a cached value, not 1.0. Today it uses 1.0.

- **`portfolio_mgr._rotate_backups` non-trivial stems**: Create a file
  `portfolio_state.v2.json` and assert backups land at `portfolio_state.v2.json.bak{N}`
  rather than mangled paths.

- **`check_concentration_risk` with cash=0**: Bold portfolio fully invested,
  proposed BUY of an existing holding — assert the function still computes
  `concentration_pct` against `total_value * alloc_pct`, not `min(..., 0)`.
