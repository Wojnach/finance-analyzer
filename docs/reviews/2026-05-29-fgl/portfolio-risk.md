# Portfolio-Risk Subsystem — Adversarial Review (2026-05-29 FGL)

Reviewer: code-review agent (Opus 4.8)
Scope: portfolio state I/O, risk management, position sizing, trade guards, exits, P&L.
Files: portfolio_mgr, portfolio_validator, warrant_portfolio, risk_management,
trade_guards, trade_validation, trade_risk_classifier, circuit_breaker, equity_curve,
monte_carlo, monte_carlo_risk, kelly_sizing, kelly_metals, cost_model, exit_optimizer,
price_targets, decision_outcome_tracker, outcome_tracker, cumulative_tracker.

## Counts
- P0: 1
- P1: 6
- P2: 8
- P3: 5

---

## P0 — data corruption / wrong-size trade / risk gate failing open

- `portfolio/warrant_portfolio.py:257`: P0: SELL with `config_key not in holdings` is a SILENT NO-OP that still saves the appended SELL transaction. The `elif action == "SELL" and config_key in holdings` has no `else` — selling a position that isn't in `holdings` (e.g. key mismatch, or a position recovered from a partial state) records the transaction line but never reduces units, leaving phantom holdings and a transaction ledger that no longer reconciles (`validate_portfolio` Check 4 will later flag "Holdings mismatch"). Worse, a SELL of MORE units than held silently clamps to delete (line 260 `if remaining <= 0: del`) without recording the over-sell, so the cash/units books diverge. → On SELL, if `config_key not in holdings` OR `units > existing.units`, refuse and log CRITICAL (do not append the transaction); never silently drop the position-reduction half of a recorded trade.

---

## P1 — incorrect risk/P&L math, race, missing guard

- `portfolio/risk_management.py:285`: P1: drawdown circuit breaker raises `peak_value` to `current_value` when `current_value > peak_value`, so a corrupted/over-stated current value (e.g. fx_rate momentarily mis-resolved high, or a spurious price) permanently inflates the peak in this call and — because `_streaming_max` caches the peak per file offset — can mask a real subsequent drawdown. The new peak is not validated against a sane ceiling. → Clamp/validate `current_value` against `initial_value * plausible_max` before letting it set a new peak, or only set peak from logged history (which is what the breaker is supposed to track), not the live (possibly mispriced) value.

- `portfolio/risk_management.py:373`: P1: ATR stop uses `atr_pct = min(atr_pct, 15.0)` then `stop_distance = max(2*atr_pct, 3.0)`, but ATR is capped to 15% AFTER which 2x → 30% max stop. For 5x warrants the rules require ≥15% stops on the cert; this computes stops on the UNDERLYING and never accounts for leverage, so a "3% min" underlying stop = 15% cert move on a 5x — exactly the barrier-proximity class the metals rules warn against. `compute_stop_levels` is leverage-blind. → Pass leverage into stop computation for warrant tickers, or document that these stops are underlying-only and must not be forwarded to warrant stop placement.

- `portfolio/kelly_sizing.py:270`: P1: `_compute_trade_stats` win/loss uses `pnl_pct` which `_pair_round_trips` documents as GROSS price-move (fee-excluded), while losses bucket uses `p <= 0` (ties counted as losses, fine) — but avg_win/avg_loss feeding Kelly are gross, so Kelly systematically over-bets because real net edge after the ~0.5–0.8% round-trip warrant spread is lower. With WARRANT round-trip cost ~1.0% and typical avg_win ~3%, ignoring costs inflates `b` and the Kelly fraction. → Use net `pnl_sek`-derived percentages (or subtract round_trip_pct from avg_win / add to avg_loss) before computing Kelly.

- `portfolio/kelly_metals.py:215`: P1: leveraged position fraction `position_fraction = half_kelly / cert_loss_frac` can exceed 1.0 for small `cert_loss_frac`, and is only capped at `MAX_POSITION_FRACTION = 0.95`. With `avg_loss=2.43`, `leverage=5` → `cert_loss_frac=0.1215`; a half-Kelly of 0.12 gives fraction ~1.0 → 95% of buying power into a single 5x cert. This is mathematically "Kelly-consistent" only if avg_loss truly bounds the loss, but warrants can gap/knock-out beyond avg_loss, so 95% sizing is a ruin risk. → Add an absolute per-trade cap well below 0.95 for leveraged certs (e.g. 0.30) independent of Kelly, matching the patient/bold alloc caps used elsewhere.

- `portfolio/equity_curve.py:248`: P1: Sortino downside deviation divides `sum(squared_devs)` by `len(daily_rets_dec)` (all observations) — the comment claims this is "standard", but it pairs a downside-only numerator deviation with a full-sample denominator inconsistently with how `mean_excess` is computed, and more importantly it silently produces a smaller downside deviation (→ inflated Sortino) when there are few losing days. For a strategy with mostly flat days this overstates risk-adjusted performance the user may act on. → Confirm intended convention; if using the "target downside deviation" form, denominator should be number of below-target observations, and the choice should be documented/tested, not asserted in a comment.

- `portfolio/risk_management.py:147`: P1: `_resolve_fx_rate` caches every in-band summary fx_rate to disk via `atomic_write_json` on EVERY call (`check_drawdown`, `_compute_portfolio_value`, `check_concentration_risk`, `log_portfolio_value`). Under the 8-worker loop this is a high-frequency write of the same tiny file from multiple threads/cycles; `atomic_write_json` rename is atomic but the repeated rotate/replace churn on a hot path is wasteful and can interleave with reads returning the previous value. Functionally tolerable but it is an unnecessary write amplification on a reliability-critical loop. → Only persist the cache when the value actually changes (compare to last cached rate) and/or throttle to once per N seconds.

---

## P2 — robustness

- `portfolio/warrant_portfolio.py:96`: P2: `warrant_pnl` computes `underlying_change * leverage` with no direction term — correct for BULL/MINI-L certs only. A BEAR/SHORT cert stored with positive `leverage` would report inverted (wrong-sign) P&L. Currently the metals catalog is bull-only so it is latent, but nothing enforces it. → Store a signed leverage (negative for bear) or an explicit `direction` field and multiply by it; add an assertion/test so a bear cert can't be tracked here silently.

- `portfolio/risk_management.py:251`: P2: when `agent_summary` is empty, drawdown falls back to cash-only value (logged as WARNING) — explicitly NOT conservative for underwater holdings. The breaker can read tiny drawdown while positions are deeply underwater and a stale feed persists. The WARNING is the only mitigation. → Consider falling back to last-known holdings value (avg_cost) rather than dropping holdings entirely, so the breaker degrades pessimistically.

- `portfolio/trade_guards.py:126`: P2: `check_overtrading_guards` reads state under `_state_lock` but the read-decide path is not transactional with `record_trade`'s write. Two BUYs evaluated concurrently can both pass the position rate-limit (read old state) before either records, defeating the limit=1 guard under the 8-worker loop. The lock only protects the load, not the check-then-act. → Acquire the lock for the whole check, or make the limit enforcement happen inside `record_trade` (reject-and-rollback) rather than advisory.

- `portfolio/monte_carlo_risk.py:408`: P2: `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` takes the raw value at face value — does NOT use the `_resolve_fx_rate` sanity band the rest of risk_management adopted (P1-15). A stale `fx_rate: 1.0` in agent_summary understates SEK VaR ~10x. → Route through `risk_management._resolve_fx_rate`.

- `portfolio/monte_carlo_risk.py:431`: P2: VaR models every held position as a LONG GBM (drift derived from `p_up`, exposure = shares*price). It never reads position direction; the system is long-only today so OK, but a SELL/short holding (if ever added) would have inverted P&L and the VaR sign would be meaningless. → Document the long-only assumption at the function contract, or read direction.

- `portfolio/exit_optimizer.py:582`: P2: EV for a limit candidate uses `fill_prob*pnl + (1-fill_prob)*fallback_pnl` where `fallback_pnl` is the MEDIAN-terminal hold P&L. But the limit and the fallback are not mutually exclusive outcomes correctly partitioned — if the limit doesn't fill, the realized terminal isn't the median, it's the conditional-on-not-hitting-target terminal (lower). This biases limit EV upward vs. a true hold. → Use the conditional terminal mean given "max < target" as the fallback, not the unconditional median.

- `portfolio/kelly_sizing.py:297`: P2: `rec_sek = min(half_kelly*cash*exposure_ceiling, max_alloc)` then `if rec_sek < 1000: rec_sek = 0`. `recommended_size` returns 0 silently when Kelly is small — a caller that treats 0 as "no opinion" vs "explicitly don't trade" can't distinguish. Combined with `win_prob` defaulting to 0.5 / accuracy fallbacks, a degenerate sizing of 0 looks identical to a data-missing 0. → Return a status field ("below_min" vs "no_edge" vs "ok") rather than overloading 0.

- `portfolio/outcome_tracker.py:286`: P2: `candidates = h[h.index.date <= target_date]` for yfinance can pick a bar AFTER the target if the index tz conversion is off by a day at horizon boundaries (the very bug the 05-28 comment fixed for one case). The `<=` plus `.iloc[-1]` is correct only if the index is sorted ascending — yfinance usually is, but it isn't asserted. → Add `h = h.sort_index()` defensively before slicing.

- `portfolio/equity_curve.py:490`: P2: `losses = [t for t in trips if t["pnl_pct"] <= 0]` counts breakeven (0%) trades as losses for win/loss-ratio and expectancy, while `wins` requires `> 0`. A flat round-trip drags down win-rate and avg_loss (avg of zeros). Minor metric distortion. → Decide on a tie convention and apply it consistently across win-rate, streaks (line 507 uses `> 0` so streaks and win/loss disagree on ties).

---

## P3 — nits

- `portfolio/portfolio_mgr.py:165`: P3: `portfolio_value` returns cash-only on bad fx_rate but logs only WARNING; consistent with the explicit design but means a bad fx silently undervalues. Acceptable; ensure callers don't treat this as authoritative.

- `portfolio/trade_validation.py:84`: P3: spread check `((ask-bid)/bid)` uses bid as denominator; mid is more standard. Negligible at tight spreads.

- `portfolio/cost_model.py:51`: P3: `total_cost_pct()` excludes the min_fee floor (documented), so for small orders the % understates true cost. Fine given the docstring, but EV math in exit_optimizer uses `total_cost_sek` (correct) — keep callers off `total_cost_pct` for sizing.

- `portfolio/price_targets.py:125`: P3: `running_extremes` hard-codes `rng = default_rng(42)` — deterministic but identical seed across all tickers/sessions means correlated MC noise if results are ever aggregated. For per-target fill prob it's fine. → Consider deriving seed from ticker+ts for independence.

- `portfolio/trade_risk_classifier.py:81`: P3: unknown regime strings map to score 0 (treated as safest, trending-up). A typo'd/novel regime ("crash") silently scores as benign. → Default unknown regime to a non-zero (cautious) score or log.

---

## Notes / verified-OK
- Atomic I/O: portfolio_mgr, warrant_portfolio, trade_guards all use `atomic_write_json` / `atomic_append_jsonl`. outcome_tracker's manual rewrite holds the sidecar lock and preserves concurrent appends correctly.
- `check_drawdown` non-finite guard (line 291) correctly fails SAFE (treats NaN/Inf as 100% drawdown).
- circuit_breaker HALF_OPEN single-probe + exponential backoff logic is sound and thread-safe.
- monte_carlo_risk C9 fix (norm.ppf not t_dist.ppf for marginals) is correct; t-copula dependence + Gaussian GBM marginals is the right construction.
- Positions are long-only across the reviewed modules (shares > 0 enforced); SHORT direction-blindness is latent, not active.
</content>
</invoke>
