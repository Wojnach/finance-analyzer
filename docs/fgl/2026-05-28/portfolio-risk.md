# Adversarial Review — portfolio-risk subsystem

Scope: 12 files in worktree `Q:\fa-fgl\portfolio-risk` (diff `fgl-baseline..HEAD`,
all 4259 lines net-new). Cross-checked against callers in `Q:\finance-analyzer`.

## Count summary

| Severity | Count |
|----------|-------|
| P0 (money loss / crash / corruption) | 2 |
| P1 (wrong under realistic conditions) | 9 |
| P2 (latent) | 8 |
| P3 (minor) | 6 |

P0/P1 headline:
1. `kelly_sizing.py:91-104` P0 — round-trip P&L uses a global avg buy price applied to **every** sell including sells that precede their matching buys (look-ahead). Win-rate / avg-win / avg-loss feeding Kelly are wrong → systematically mis-sized recommendations.
2. `risk_management.py:374,465,897` P0 — ATR stop for leveraged warrants is computed as a fixed `entry*(1-2*atr/100)` with no knockout-barrier awareness and no 3%-min-distance floor; for low-ATR silver certs this puts the stop within a couple percent of price, violating the documented "never stop within 3% of bid" rule and risking placement at/through the barrier.
3. `monte_carlo.py:359-362` P1 — directional drift derived from `0.5 ± conf*0.3` is an arbitrary fabricated edge injected into VaR/exit GBM; overstates upside, understates downside risk for held longs.
4. `trade_validation.py` / `trade_risk_classifier.py` P1 — both "block invalid trade" modules have **no production caller** (tests only). They return verdicts nothing enforces.
5. `portfolio_validator.py` P1 — `validate_portfolio` is only wired to the read-only dashboard endpoint; corruption (negative cash, holdings/transaction mismatch) is reported but never blocks a save or a trade.

---

## P0 — money loss / crash / corruption

`portfolio/kelly_sizing.py:91-104`: P0 sizing/P&L: `_compute_trade_stats` computes ONE
weighted-average buy price across **all** BUY transactions for a ticker, then derives a
P&L% for **every** SELL against that single average — including SELLs that happened
*before* later BUYs (look-ahead) and ignoring lot/FIFO order. Win-rate, avg_win_pct and
avg_loss_pct are therefore wrong, and they feed `kelly_fraction` → `recommended_sek`.
A ticker that round-tripped profitably then was re-bought higher can be scored as a
loser (or vice-versa), mis-sizing the position. Fix: reuse the correct FIFO matcher
`equity_curve._pair_round_trips()` (already in this diff) instead of the average-price
shortcut.

`portfolio/risk_management.py:374` (and `:465`, `:897`): P0 stop-loss on leveraged certs:
`stop_price = entry_price * (1 - 2*atr_pct/100)` with `atr_pct` only capped on the high
side (min(atr,15)). There is no LOWER floor and no knockout-barrier check. For a silver
warrant whose underlying ATR reads ~1%, the stop sits at entry*(1-0.02) = within 2% of
entry — directly violating the project rule "NEVER place a stop-loss within 3% of current
bid" (memory/feedback_mini_stoploss.md) and risking a stop set near/below the financing
(knockout) level where it fills instantly at a bad price (Mar 3 incident). Fix: enforce a
minimum stop distance (>=3% of price) AND pass/subtract the instrument's knockout barrier;
for warrants never emit a stop within barrier+buffer. Note these are advisory outputs
today, but `compute_stop_levels`/`compute_probabilistic_stops` are exactly the levels a
human/Layer 2 acts on.

---

## P1 — wrong under realistic conditions

`portfolio/monte_carlo.py:359-362`: P1 risk-model: `_get_directional_probability` maps a
HOLD/BUY/SELL + confidence into `p_up = 0.5 ± conf*0.3`, then `drift_from_probability`
turns that into GBM drift used by VaR (`monte_carlo_risk`) and stop-hit probabilities. This
manufactures an edge that the accuracy gates explicitly distrust (<55% short-term). For a
held long it biases drift positive, lowering modelled stop-hit prob and VaR — overconfident
risk numbers. Fix: default p_up=0.5 (zero drift) for VaR; only use directional drift for
opportunity sizing, never for downside risk.

`portfolio/trade_validation.py:22`: P1 dead safety control: `validate_trade` (cash
sufficiency, position-size cap, spread cap, price-deviation cap) is imported nowhere in
production — only `tests/test_trade_validation.py`. Order-placing paths never call it, so
none of these limits are enforced. Fix: wire it into the order path in
`avanza_orders.py` / the portfolio BUY/SELL execution before placement, treating
`valid=False` as a hard block.

`portfolio/trade_risk_classifier.py:29`: P1 dead safety control: `classify_trade_risk`
has no production caller (tests + review scaffolding only). The HIGH/MEDIUM/LOW score is
computed by nobody at decision time. Fix: call it in `agent_invocation`/reporting and
surface/act on HIGH.

`portfolio/portfolio_validator.py:13`: P1 non-enforcing validation: `validate_portfolio`
detects negative cash, holdings/transaction share mismatch, fee mismatch, duplicate txns —
but the only caller is `dashboard/app.py:1071` (a GET-style POST endpoint that just returns
errors). `portfolio_mgr.save_state`/`update_state` never run it, so a corrupt
read-modify-write is persisted silently. Fix: run `validate_portfolio` inside
`update_state` after mutation and refuse to write (or write to a quarantine file) on error.

`portfolio/risk_management.py:757-758` & `770-771`: P1 currency/price fallback in
concentration: when a held ticker has no live price, it falls back to `avg_cost_usd` for
`total_value` but the proposed-alloc and concentration % then mix a stale cost basis with
live cash — a position deeply underwater shows near-cost concentration, underreporting
risk. Also `proposed_alloc = min(total_value*alloc_pct, cash)` uses *total_value* (cash +
holdings) as the sizing base while capping at cash, so as holdings grow the proposed buy
grows beyond the strategy's intended cash-fraction until clipped at 100% of cash. Fix: size
off cash (or off a clearly-defined equity base) consistently, and flag stale-price
holdings instead of valuing at cost.

`portfolio/kelly_sizing.py:296-310`: P1 sizing fallback: when no trade history exists,
avg_win/avg_loss default to `atr*1.5 / atr*1.0` (b=1.5 fixed). Combined with a win_prob
sourced from signal accuracy that can be 0.55-0.60, this yields a non-trivial Kelly with
**zero** realized evidence. The `min 1000 SEK` floor only zeroes tiny sizes; a 500K cash
book with win_prob 0.58, b=1.5 gives full Kelly ≈ 0.30 → half-Kelly 15% = 75K SEK on a
fabricated edge. Fix: require a minimum realized-sample count before returning non-zero
size, or hard-cap to quarter-Kelly when stats are ATR-derived.

`portfolio/equity_curve.py:188` & `:557`: P1 annualization blow-up: annualized return uses
`pow(last/first, 1/years)`. With the loop logging multiple entries per day and only a few
days of history, `years` is tiny (e.g. 0.01), so a +5% move annualizes to astronomically
large/meaningless numbers that then feed `calmar_ratio`. There is no guard for
`years < ~0.1`. Fix: return None (or cap) when `days_elapsed` is below a sane window
(e.g. < 30 days) before annualizing.

`portfolio/monte_carlo_risk.py:408`: P1 currency mixup risk: `compute_portfolio_var` reads
`fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` — the **raw** `.get` pattern that
`risk_management._resolve_fx_rate` was specifically introduced (P1-15) to replace because a
stale/missing `fx_rate=1.0` understates SEK ~10x. VaR/CVaR `_sek` figures will be ~10x too
small if `fx_rate` is absent or 1.0. Fix: route through `_resolve_fx_rate`.

`portfolio/kelly_sizing.py:142-143`: P1 accuracy-as-winprob with no gate: `_get_signal_accuracy`
returns consensus accuracy directly as win probability with only a `> 0` check — a 0.47
sub-gate accuracy (which the engine force-HOLDs) still produces a positive win_prob fed to
Kelly here. No min-sample / min-accuracy gate mirrors the engine's 47%/50% rule. Fix: apply
the same accuracy floor before using it as p.

`portfolio/risk_management.py:807-819`: P1 regime-mismatch blind spot: SELL-in-uptrend
only flags when `volume_ratio` is present AND <1.5; missing volume (None) is treated as
"not a mismatch", so the most common case (no RVOL data) silently passes a counter-regime
sell. Acceptable for BUY (documented) but inconsistent guard. Fix: at minimum emit an
info-severity flag when volume data is missing on a counter-regime trade.

---

## P2 — latent

`portfolio/portfolio_mgr.py:162-180`: P2 `portfolio_value` accepts shares/price via `.get`
but never validates `shares` is finite/positive-typed beyond `>0`; a string or NaN in
`shares` slips into `shares*price*fx` only caught by the broad `except`. Minor since state
is validated elsewhere, but `_validated_state` does not coerce per-holding `shares`.

`portfolio/monte_carlo_risk.py:188,303,371`: P2 negative-share handling inconsistent.
`PortfolioRiskSimulator` filters `shares != 0` (keeps shorts) but `compute_portfolio_var`
upstream filters `shares <= 0` (drops shorts). The simulator's short support is therefore
dead in the system path; if ever fed shorts directly, `drawdown_probability`'s
`total_value = sum(shares*price)` goes negative and the function returns 0.0 (no risk) for
a net-short book. Fix: use abs/exposure for the denominator.

`portfolio/monte_carlo_risk.py:248`: P2 t-copula df fixed at 4 with no `df>2` guard at the
API boundary; `chisquare(df)` with df<=0 (if a caller overrides) would crash. Low prob but
unvalidated.

`portfolio/equity_curve.py:65-110`: P2 daily-return grouping takes the **last** value per
calendar day; with 24/7 logging and irregular cycle times, a missing-day gap produces a
single jumbo "daily" return that inflates volatility/Sharpe denominators. No gap handling.

`portfolio/risk_management.py:96-98`: P2 `_streaming_max` peak compares `val > peak` with
`val = entry.get(value_key, 0)` — a missing key yields 0, harmless for a max, but a
**negative** portfolio value (impossible normally) would be ignored; also no `isfinite`
check on streamed vals so a NaN in history poisons the cached peak silently (only the final
`check_drawdown` isfinite guard catches it, after caching).

`portfolio/trade_guards.py:131-135`: P2 cooldown multiplier read from
`consecutive_losses[strategy]` but `record_trade` only increments on SELL with non-None
pnl_pct; a BUY-then-stopout recorded as two SELLs, or a SELL recorded without pnl_pct,
leaves the streak un-updated. Escalation can under-count real losses.

`portfolio/circuit_breaker.py`: P2 this is an **API** circuit breaker (data-source
failures), not a drawdown/trading breaker — fine, but its name collides conceptually with
the trading drawdown breaker and it is not wired to halt trading. No bug, but confirm no
caller mistakes it for risk control.

`portfolio/monte_carlo.py:91`: P2 `p_up` clamped to [0.01,0.99]; at the clamp `norm.ppf`
gives ±2.33 → very large drift when multiplied by `sqrt(trading_days)` (≈19 for 365). A
0.99 p_up yields enormous annualized drift, making short-horizon GBM bands wildly skewed.
Fix: tighten the clamp (e.g. [0.4,0.6]) for risk use.

---

## P3 — minor

`portfolio/kelly_sizing.py:110`: P3 breakeven (`p <= 0`) counted as a loss in win/avg-loss
stats. Skews Kelly slightly pessimistic. Use `< 0`.

`portfolio/equity_curve.py:495`: P3 same breakeven-as-loss (`pnl_pct <= 0`) in
win/loss/streak classification.

`portfolio/risk_management.py:382`: P3 `triggered = current_price < stop_price` is a strict
`<`; a price exactly at the stop is not triggered. Negligible.

`portfolio/cost_model.py:49-55`: P3 `total_cost_pct` excludes the min-fee floor, so for
small trades the reported pct understates true cost; docstring notes it but callers may not.

`portfolio/monte_carlo.py:174-177`: P3 odd-`n_paths` extra draw uses a fresh `rng`
standard_normal but the antithetic symmetry is already broken for the odd path — cosmetic,
negligible bias at n=10000.

`portfolio/portfolio_mgr.py:50`: P3 `_rotate_backups` skips backup when file size is 0 but
not when the file is present-and-truncated-to-valid-empty-json (`{}`); a corrupt-to-empty
overwrite could rotate a good backup out over 3 cycles. Very low prob.

---

## Verified NOT bugs (checked to avoid false positives)

- Drawdown breach IS wired: `agent_invocation.py:891-902` hard-blocks invocation at
  `current_drawdown_pct > 50%` (matches user's "only de-risk at 50%+"). The `breached`
  field itself (threshold 20) is unused, but the block uses raw pct — intentional.
- `check_drawdown` has explicit NaN/Inf fail-safe (returns breached=True, 100% DD).
- `_pair_round_trips` (equity_curve) is genuine FIFO and nets buy+sell fees correctly
  (P0-6 fix verified); pnl_pct gross / pnl_sek net split is intentional and documented.
- `_resolve_fx_rate` correctly rejects 1.0 and walks summary→cache→10.50 fallback.
- portfolio_mgr `update_state` holds the per-file lock across the full read-modify-write
  (concurrency-safe); `_get_lock` is itself guarded.
- `kelly_fraction` correctly clamps negative edge to 0 (no shorting) and rejects
  win_prob outside (0,1).
