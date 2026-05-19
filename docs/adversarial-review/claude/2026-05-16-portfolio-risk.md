# Adversarial Code Review: Portfolio-Risk Subsystem (2026-05-16)

## [P1] kelly_metals.py:245 — Log argument can go negative, crashes at runtime

**File:** portfolio/kelly_metals.py:245
**Bug:** When `f * cert_loss_frac > 1`, the log argument `1 - f * cert_loss_frac` becomes negative.

When `adjusted_fraction = 0.95` and `cert_loss_frac = 1.5`, then `1 - 0.95*1.5 = -0.425`. The `max(1e-10, ...)` guard only prevents zero, not negative values. This causes `math.log(negative)` → ValueError at runtime.

**Why it matters:** kelly_metals uses `position_fraction = half_kelly / cert_loss_frac` (L217). With extreme parameters (low avg_loss, high leverage), position_fraction can exceed 1.0 before the MAX_POSITION_FRACTION cap. The log crashes on high-loss scenarios, silently failing the metals sizing recommendation during peak drawdown (worst time).

**Fix:** Clamp the log argument: `math.log(max(1e-10, min(0.9999, 1 - f * cert_loss_frac)))`.

---

## [P1] kelly_metals.py:217 — Position fraction unbounded before MAX cap

**File:** portfolio/kelly_metals.py:217
**Bug:** Position sizing divides by cert_loss_frac without bounds:

If `half_kelly = 0.20` and `cert_loss_frac = 0.01`, then `position_fraction = 20.0`. The min() cap constrains it to 0.95, but the cap is applied post-calculation. If avg_loss is garbage (0.001% from corrupted signal_log.db), position_fraction becomes 200, then capped silently.

**Why it matters:** Silent loss of signal quality. A bad avg_loss estimate is masked by clamping, not rejected. Operators have no feedback their position is sized on corrupted data.

**Fix:** Validate `cert_loss_frac` is in band (0.01 to 0.50) before division. If out of band, fall back to default and log a warning.

---

## [P2] kelly_metals.py:243-246 — Win probability not clamped before growth calculation

**File:** portfolio/kelly_metals.py:243-246
**Bug:** Log-growth uses win_rate without validation. If corrupted to > 1.0 or < 0.0, log arguments become invalid.

**Why it matters:** Edge case producing misleading growth projections. Low severity because blended win_rate on L176 usually constrains it.

**Fix:** Clamp `win_rate = max(0.01, min(0.99, win_rate))` before log-growth.

---

## [P2] kelly_sizing.py:313 — kelly_fraction result not re-validated after clamping

**File:** portfolio/kelly_sizing.py:313-315
**Bug:** `full_kelly = kelly_fraction(win_prob, avg_win, avg_loss)` returns [0,1] by design, but if NaN/Inf propagates upstream, it passes through unchecked.

**Why it matters:** Defensive programming gap. A downstream function could silently amplify unbounded kelly_fraction.

**Fix:** Add explicit guard: `full_kelly = max(0.0, min(1.0, full_kelly))`.

---

## [P2] risk_management.py:376-378 — Stop-loss distance uses current price, not entry price

**File:** portfolio/risk_management.py:376-378
**Bug:** `atr_value = current_price * atr_pct / 100` makes distance-to-stop order-dependent. On a 50% winner, the perceived distance ratio compresses even though entry-stop distance is unchanged.

**Why it matters:** Risk audit flags are inconsistent. Identical entry/stop levels show different "danger zone" warnings depending on current price. This violates invariants for risk assessment.

**Fix:** Use entry price: `atr_value = entry_price * atr_pct / 100`.

---

## [P2] monte_carlo_risk.py:436 — ATR fallback uses `or`, loses zero as valid

**File:** portfolio/monte_carlo_risk.py:436
**Bug:** `atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)` treats 0.0 as missing, not as extremely stable.

**Why it matters:** A deliberately-set zero volatility (stalled warrant) gets replaced with 2.0%, inflating VaR estimates.

**Fix:** Use explicit None check: `atr_pct = (extra.get("atr_pct") or ticker_data.get("atr_pct")) or _atr_default_for_ticker(ticker)`.

---

## [P2] equity_curve.py:390 — pnl_pct calculated before fee deduction

**File:** portfolio/equity_curve.py:390
**Bug:** `pnl_pct` is gross of fees but `pnl_sek` is net. A +2.5% move with 1% fees reports as 50% of a win_loss_ratio denominator (using gross %), inflating apparent edge by the fee amount.

**Why it matters:** Asymmetry in Calmar and expectancy calculations. Fee impact is underestimated in win_rate thresholds but visible in absolute P&L.

**Fix:** Either recompute pnl_pct net of fees, or document gross/net asymmetry and adjust thresholds.

---

## [P2] portfolio_mgr.py:172 — Missing holdings silently skipped if price is zero

**File:** portfolio/portfolio_mgr.py:162-180
**Bug:** Holdings with price=0 or price<=0 (line 171 condition fails) are silently skipped, not logged. Portfolio value underestimates by skipped shares.

**Why it matters:** Stale price data causes silent undervaluation. Drawdown circuit breaker can trip falsely, halting trading even when real P&L is positive.

**Fix:** Log a warning for each skipped holding: `elif shares > 0: logger.warning("No valid price for %s", ticker)`.

---

## [P2] trade_guards.py:363 — C4 wiring check happens outside lock

**File:** portfolio/trade_guards.py:361-368
**Bug:** C4 check loads state, releases lock, then checks. Between unlock and check, another thread can record a trade and the warning fires as false positive. Also, _portfolios_have_transactions() does lock-free load_json(), racing with concurrent writes.

**Why it matters:** Noisy false-positive alerts. C4 wiring check becomes unreliable under concurrency.

**Fix:** Move C4 check inside the lock.

---

## [P2] risk_management.py:752 — Concentration check uses stale FX rate fallback

**File:** portfolio/risk_management.py:749-762
**Bug:** If agent_summary is stale, _resolve_fx_rate falls back to FX_RATE_FALLBACK (10.50). For multi-day holds, this can be 10% off market. Portfolio value denominator becomes unreliable, affecting concentration thresholds.

**Why it matters:** FX staleness bleeds into concentration risk. A position at 40% threshold can flip to 35% or 45% depending on FX fallback, causing spurious warnings.

**Fix:** Log when using fallback FX rate, surface staleness in flag message.

---

## [P3] monte_carlo.py:309-310 — Stop price can be zero or negative

**File:** portfolio/monte_carlo.py:309-310
**Bug:** `stop_price = price * (1 - 2 * atr_pct / 100)`. If atr_pct >= 50%, stop_price <= 0. Downstream, probability_below(stop_price) treats negative threshold as matching almost all paths, inflating p_stop_hit_1d to 0.99.

**Why it matters:** Misleading stop-hit probabilities. Silver at 6% daily ATR gives negative stops, reported as "will definitely hit" (p=0.99).

**Fix:** Add floor: `stop_price = max(price * 0.01, price * (1 - 2 * atr_pct / 100))`.

---

## [P3] trade_validation.py:85 — Spread calculation doesn't validate ask > bid

**File:** portfolio/trade_validation.py:85
**Bug:** If ask < bid (inverted quotes), spread_pct becomes negative, passes the `> max_spread_pct` check, approves trade as "good spread".

**Why it matters:** Silent approval on corrupted quotes. Inverted bid/ask approved with 0% spread.

**Fix:** Validate before spread calc: `if ask < bid: return ValidationResult(False, "Invalid quotes: ask < bid")`.

---

## SUMMARY

- P1 (critical): 2 findings
- P2 (high): 8 findings
- P3 (low): 2 findings

**Total: 12 findings**

No issues found in portfolio_validator.py, trade_risk_classifier.py, or equity_curve.py metric calculations.

