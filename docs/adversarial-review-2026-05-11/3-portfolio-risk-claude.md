# Claude adversarial review: portfolio-risk

## Summary

Read the full in-scope set. The system has matured (atomic I/O, per-file locks, FX
fallback chain, FIFO P&L with net-of-fees, t-copula VaR). The remaining bugs cluster in
four areas: (1) min-order-size is enforced as **500 SEK in three places** but the project
rule and the actual Avanza floor is **1000 SEK**, so sizing engines emit orders the broker
will burn courtage on; (2) ATR-stop math hard-caps ATR at 15 % even for 5x certs where
the user explicitly accepts 10-20 % knockout risk — and the proximity flag is mis-signed;
(3) MINI-warrant **financing-level barrier** is never compared to the 3 % rule in
`compute_stop_levels`; only the exit optimizer checks it; (4) Monte Carlo `simulate_ticker`
uses `seed` directly but `simulate_all` increments it deterministically — different paths
across calls if a ticker order changes. Plus several quieter issues with reload-bypass on
cooldowns, NaN propagation, and an EV-of-hold computed from quantiles (not paths).

## P0 — Blockers

- portfolio/trade_validation.py:32 — Why it bites: `min_order_sek: float = 500.0` while
  the project rule (CLAUDE.md and metals-avanza.md) and `portfolio/avanza/trading.py:75`
  both enforce **1000 SEK**. Callers that take the default ship 500-999 SEK orders, which
  pass validation, then get rejected by `avanza_session.place_order` with `ValueError`
  AFTER the orchestrator has already cleared cash and cancelled stops. Fix: default to
  1000.0 and pin to `portfolio/avanza/trading.MIN_ORDER_SEK` via import.

- portfolio/kelly_sizing.py:326 — Same defect, copy-paste:
  ```
  if rec_sek < 500:
      rec_sek = 0  # Below minimum trade size
  ```
  Kelly recommends e.g. 750 SEK on a small account → consumer places it → Avanza rejects.
  Fix: raise to 1000 (or import the canonical constant).

- portfolio/kelly_metals.py:44 — `MIN_TRADE_SEK = 500.0` again. This one routes to the
  metals warrant ladder where the courtage minimum bites hardest (no commission until
  1000 SEK). Net effect: 500-999 SEK ladders get placed, billed flat 1 SEK courtage
  (0.1-0.2 %) and tagged as the "Marja Folcke" probe traders see no profitability on.

- portfolio/risk_management.py:374 — `atr_pct = min(atr_pct, 15.0)`. For a MINI silver
  warrant with the underlying running at 4 % daily ATR × 5x leverage, the stop level
  ignores 5-15 % wicks the user explicitly accepts. Worse, when `entry_price * (1 - 2 *
  atr_pct/100)` falls **below the MINI financing level** the function never compares the
  two and returns a stop that is mathematically inside the knockout barrier — violates the
  "NEVER stop-loss within 3 % of MINI warrant barrier" rule directly. Fix: pass
  `financing_level` and floor `stop_price = max(stop_price, financing_level * 1.03)`.

## P1 — High

- portfolio/risk_management.py:902-911 — `check_atr_stop_proximity` computes
  `distance_in_atr = (current - stop) / atr_value` and flags `< 1.0`. For an entry at $30
  with 4 % ATR, stop = $27.60 and a current price of $28 yields `distance_in_atr = 0.33`
  — but `1.0x ATR` is **already inside the noise band**: the flag fires only after we are
  closer than one daily wick from getting filled, which means the warning is reactive, not
  preventive. Combined with the `atr_pct = min(atr_pct, 15.0)` cap, MINI cert stops will
  trip silently before the proximity flag activates. Fix: trigger at 1.5-2.0x ATR.

- portfolio/risk_management.py:466-468 — `compute_probabilistic_stops` clamps
  `stop_price = entry_price * 0.01` when `stop_price ≤ 0`. That's a 99 %-drop "stop"
  which trivially never fires, **silently disabling the probabilistic stop** for any
  position whose ATR×2 puts the stop at or below 0 (high-vol warrants, leveraged certs
  with low entry). Caller treats a probability of 0 as "safe" and skips the de-risk path.

- portfolio/risk_management.py:296-304 — NaN guard returns
  `"breached": True, "current_drawdown_pct": 100.0`. Fail-safe in principle, but
  `peak_value` is then rounded with `round(peak_value, 2) if math.isfinite(peak_value)
  else 0.0` — a `peak_value=0.0` written back to history corrupts the JSONL cache and the
  next call streams from byte-offset zero with floor=initial_value, so the breaker stays
  pinned to 100 % until manual repair. Fix: don't persist the sentinel — return without
  caching.

- portfolio/monte_carlo_risk.py:419 — `fx_rate = agent_summary.get("fx_rate",
  FX_RATE_FALLBACK)`. Unlike `risk_management._resolve_fx_rate` this does **not** check
  the sanity band; a stale agent_summary that legitimately contains `fx_rate = 1.0`
  (P1-15 root cause) understates VaR_SEK by ~10x and the Layer 2 prompt receives
  reassuring numbers. Fix: route through `_resolve_fx_rate`.

- portfolio/monte_carlo.py:395-401 — `_get_directional_probability` falls through to
  `extra.get("_weighted_confidence")` and treats it as P(up). For a `SELL` action the
  code computes `0.5 - conf * 0.3` — bounded at 0.2 — but for **HOLD it returns 0.5**.
  A signal stack that has just flipped from BUY to HOLD with high confidence still drives
  the MC drift to zero, hiding the freshly-signalled tail. Could give Layer 2 the wrong
  picture immediately after a regime change. Acceptable but worth documenting.

- portfolio/exit_optimizer.py:303-340 — `_compute_pnl_sek` deducts cost **once on exit**.
  The buy-side cost (spread, slippage, courtage) is never accounted for. For warrants at
  40 bps half-spread + 10 bps slippage = 1.0 % round-trip, the EV of every quantile is
  overstated by ~half the round-trip cost. Fix: subtract `costs.total_cost_sek(entry_value)`
  once in `_compute_pnl_sek` or document that `entry_price_sek` is post-cost.

- portfolio/exit_optimizer.py:617-621 — Hold-to-close EV is computed from **5 percentiles
  of terminal prices**, not the full path distribution: `np.mean(terminal_pnls)` over the
  10/25/50/75/90 quantile P&Ls. That's a 5-point trapezoidal integral biased toward the
  median; in fat-tailed cases (warrants near barrier) the true tail mean is far lower.
  Fix: `hold_ev = float(np.mean([_compute_pnl_sek(...) for p in terminal]))` — vectorise
  the P&L formula and apply over all 5000 paths.

- portfolio/exit_optimizer.py:397-400 — Knockout buffer is hard-coded `financing_level *
  1.03` (3 %). Adversarial scenario: a XAG 5x cert with financing at $28.50 trades at
  $29.10 (2 % buffer). `_compute_risk_flags` will flag `KNOCKOUT_DANGER` for everything
  under 3 % distance — but the 3 % buffer is also used for `stop_buffer` in the
  risk-override branch (line 446). The override then **forces market exit** at every wick
  inside 3 % even though the user accepts 10-20 % knockout risk. Fix: make buffer a
  parameter, default it from `instrument_profile`.

- portfolio/trade_guards.py:140-167 — Cooldown uses wall-clock `datetime.now(UTC)`
  compared to ISO timestamps from disk. If a Telegram operator manually edits
  `data/trade_guard_state.json` (or NTP slews the clock), the cooldown is bypassable.
  More importantly, **on process restart the cooldown is honoured but
  `record_trade` wiring is checked only via the C4 "positive proof" flag** at
  trade_guards.py:255 — which is set by the first call this process. Between restart
  and the first BUY/SELL the guards run with empty state if the file was missing,
  silently. Fix: monotonic clock not feasible (persistence), but verify the file's
  presence + reject reads where `mtime > now` (clock jump backwards).

- portfolio/portfolio_mgr.py:54-60 — Backup rotation iterates with `path.with_suffix`
  but `STATE_FILE = .../portfolio_state.json` already has `.json` suffix; `with_suffix`
  **replaces** the suffix, so `path.with_suffix(".json.bak2")` from a path that contains
  a dot in the stem (none of the current files have that, but `portfolio_state_warrants`
  is borderline) will silently mangle. Not an active bug today but a footgun.

- portfolio/risk_management.py:911 — Empty `held_correlated` check missing for the case
  where BOTH correlated tickers are flat — `check_correlation_risk` returns None as
  expected, but the BUY into ETH after a SELL of BTC last hour will not flag because
  holdings.shares is checked at this instant only (no history window).

## P2 — Medium

- portfolio/trade_validation.py:75-81 — `cash_pct = order_value / cash_available × 100`
  caps at `max_cash_pct=50%` of **cash**, not of **portfolio value**. After several
  earlier buys, `cash_available` shrinks, so a 50 % cash check passes even as the
  position becomes a 80 % portfolio concentration. The concentration check in
  `risk_management.check_concentration_risk` exists separately at the 40 % portfolio
  level, but the two are not chained. Fix: pass `portfolio_value`, compute against that.

- portfolio/equity_curve.py:494-495 — `losses = [t for t in trips if t["pnl_pct"] <= 0]`
  classifies break-even trades (pnl=0) as losses. This skews
  `max_consecutive_losses` upward and is a silent reporting bias. Fix: `< 0`.

- portfolio/monte_carlo.py:266-339 — `simulate_ticker` uses `seed` as-is. In
  `simulate_all` (line 392-399) the seed becomes `seed + i` where `i` is the index in
  `sorted(tickers)`. Adding a ticker shifts every subsequent seed → reruns produce
  different VaR for unrelated tickers. Reproducibility is brittle. Fix: hash ticker name
  into the seed: `seed ^ zlib.crc32(ticker.encode())`.

- portfolio/monte_carlo_risk.py:200 — `positions.get("shares", 0) != 0` admits negative
  shares (short positions). The cost / P&L path assumes long-only (`max(exit_warrant_sek,
  0)` in exit_optimizer, exposure ceiling, concentration). A negative-shares entry slipping
  in (e.g. corrupted state) would produce nonsensical VaR. Fix: `> 0`.

- portfolio/risk_management.py:734-793 — `check_concentration_risk` computes
  `proposed_alloc = min(total_value * alloc_pct, cash)`. If `cash == 0` (fully invested
  Bold portfolio) proposed_alloc is 0 → concentration_pct excludes any new entry → the
  check **never fires** when it's most needed. Fix: assume `proposed_alloc =
  total_value * alloc_pct` and warn separately if cash short.

- portfolio/warrant_portfolio.py:80 — `if not holding or not current_underlying_usd or
  not fx_rate: return None`. Python truthiness: `current_underlying_usd = 0.0` silently
  returns None — fine — but `fx_rate = 0` also short-circuits and the caller in
  `get_warrant_summary` records the position with `pnl=None` so total_value stays at 0,
  invisible to the operator unless they read the raw dict. Better: explicit `is None`
  checks and a logged warning.

- portfolio/trade_risk_classifier.py:81 — `regime_score = _REGIME_SCORES.get(regime_lower,
  0)`. Silent permissive fallback: an unknown regime string (typo, new label, JSON missing
  the field) is treated as "trending-up" (0 risk). Fix: log unknown and default to 2
  (ranging) — the safer prior.

- portfolio/exposure_coach.py:71 — `zone = market_health.get("zone", "healthy")`. Same
  permissive fallback: a malformed market_health dict (missing `zone`) yields a 1.0
  ceiling. Combined with Kelly sizing using `exposure_ceiling` as a multiplier on cash,
  upstream data corruption maximises position size. Fix: default to "caution" (0.6).

- portfolio/monte_carlo_risk.py:387-388 — `loss_threshold = -total_value * threshold_pct
  / 100.0`. `total_value` is the **gross** exposure across positions; for a portfolio with
  long-only positions this is fine, but the drawdown_probability metric is compared to
  PnL which can exceed total_value (long convex moves). The framing
  "drawdown_5pct_prob" suggests portfolio drawdown but the denominator is exposure, not
  capital. Mismatch when cash is large relative to positions.

## P3 — Low

- portfolio/exit_optimizer.py:54 — `usdsek: float = 10.85`. Hard-coded fallback. Same
  reasoning as P1-15 in risk_management: caller passes a stale snapshot and gets
  ~3 % off. Low impact because most callers compute fx from agent_summary.

- portfolio/equity_curve.py:243-249 — Sharpe re-derives `daily_std_dec` from
  `daily_rets_dec` but at line 231 already computed `daily_vol` from a separate `mean_ret`
  in percent units. Two parallel std computations on the same data, one divided by 100.
  The double work is harmless but indicates the function has been edited piecemeal.

- portfolio/portfolio_validator.py:130 — Holdings-mismatch tolerance is `relative_diff <
  0.01` (1 %) only when `actual_shares == 0`. A leftover dust position (0.0001 shares
  after rounding) bypasses validation only if also removed from holdings — otherwise the
  error message fires with a tiny diff that's actually safe.

- portfolio/kelly_metals.py:215-217 — `cert_loss_frac = avg_loss * leverage / 100.0` then
  `position_fraction = half_kelly / cert_loss_frac`. For leverage=5 and avg_loss=2.4 %,
  `cert_loss_frac = 0.12`, so position_fraction = `half_kelly / 0.12`. With a strong edge
  this exceeds 1.0 and is then clipped at MAX_POSITION_FRACTION=0.95 — but the half-Kelly
  intent is lost. Effectively the leverage-adjustment inverts Kelly back into full Kelly
  whenever the edge is moderate. Documented intent vs effective output diverge.

- portfolio/cost_model.py:67 — `spread_bps=40.0` for warrants is conservative for XAU but
  **optimistic for XAG**. Live order books show 0.6-1.5 % half-spreads. Worth per-ticker
  cost models.

- portfolio/strategies/orchestrator.py:116 — `time.sleep(0.5)` is the floor on tick
  granularity. A strategy with `poll_interval_seconds < 0.5` cannot tick faster; this is
  fine for current strategies but the contract isn't documented.

- portfolio/monte_carlo.py:177-179 — Odd-`n_paths` extra path uses `rng.standard_normal(1)`
  AFTER the antithetic block already consumed half the stream — fine, but if `seed` is
  None and the engine is re-used the extra path is non-deterministic. Not a real bug.

## Tests missing

- Min-order-size invariant: assert `trade_validation.validate_trade(..., volume=X,
  price=Y)` returns invalid for `X*Y < 1000` for warrant / stock / crypto paths.
- ATR stop vs MINI financing level: feed
  `compute_stop_levels({ticker: {shares: N, avg_cost_usd: E, financing_level: F}})`
  and assert `stop_price >= financing_level * 1.03`.
- Drawdown breaker NaN persistence: write `nan` into the history JSONL and verify the
  next call does NOT poison the byte-offset cache.
- Kelly leverage inversion: `recommended_metals_size(leverage=5, avg_loss=2)` — assert
  that `position_fraction <= half_kelly_pct` (the leverage shouldn't grow position).
- Cooldown bypass on file mutation: backdate `ticker_trades[…]` to 24 h ago and verify
  guard reload still respects the elapsed time (it does, but the assertion is missing).
- VaR with negative shares: assert `compute_portfolio_var` rejects or zeroes a short
  holding rather than computing fake long-only VaR.
- Exit optimizer round-trip cost: assert that the EV of an immediate buy + market sell
  at the same price is **negative**, not zero.

## Cross-cut observations

- Three different `MIN_TRADE_SEK / min_order_sek / 500` constants. None imports
  `portfolio.avanza.trading.MIN_ORDER_SEK` (the canonical broker minimum). Make that the
  single source of truth and have every sizing module import it.
- `_resolve_fx_rate` is excellent and should be the only path to FX. `monte_carlo_risk.py`
  and `exit_optimizer.compute_exit_plan_from_summary` both bypass it.
- The "MINI warrant within 3 % of barrier" rule is hard-coded in three places
  (`exit_optimizer:375`, `:446`, `:397`) and absent from `risk_management.compute_stop_levels`.
  Pull into `instrument_profile.get_barrier_buffer(ticker)` so it's consistent and
  configurable per cert series.
- `trade_guards` and `portfolio_mgr` use different lock disciplines: per-file lock vs a
  module-level `_state_lock`. A single read-modify-write helper would prevent the next
  concurrency bug.
- Several risk modules silently default to "permissive" on missing data (regime → 0,
  zone → "healthy", fx_rate → 1.0 historically). The fail-closed posture should be made
  explicit and centralised.
