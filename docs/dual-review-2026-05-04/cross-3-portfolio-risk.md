# Cross-critique — portfolio-risk

## Codex findings Claude missed

| Codex finding | Why Claude missed it |
|---|---|
| `monte_carlo.py:328` — `p_stop_hit_*` uses **terminal price** P(S_T < stop) instead of path minimum P(min S_t < stop). For volatile multi-day horizons this **understates** stop-hit probability — exits and risk warnings miss intra-horizon stop-outs. | Claude reviewed Monte Carlo for annualization (P0-2) but didn't check whether the "stop hit" measurement actually tests path-touching. This is a classic options-pricing trap (path-dependent vs European) that Codex caught directly. |
| `warrant_portfolio.py:100-103` — `current_implied_sek` can go **negative** when underlying drops > 100/leverage % (e.g. 5x cert with >20% drop). Negative warrant value is impossible (warrants knock out at zero). Negative mark flows into `total_value_sek` and portfolio summaries. | **Critical money-math bug Claude completely missed.** Claude reviewed `warrant_portfolio.py` and even flagged `pnl_pct` units inconsistency, but did not test the boundary case of a knocked-out warrant. Codex caught it via boundary analysis. |
| `monte_carlo.py:304-307` — `drift = drift_from_probability(p_up)` is calibrated for 1d horizon but reused for default 3d run. P(up) drifts from 0.60 (1d target) to 0.67 (3d actual). Multi-day MC bands miscalibrated. | Claude flagged annualization (P0-2) but missed the drift recalibration question. Codex showed both axes are wrong: vol scaling AND drift target. |
| `trade_guards.py:286-291` — `new_position_timestamps` appends an entry for every BUY, including adds-to-existing-position. After scaling into a position, the next *unrelated* entry can be blocked as if a fresh position had been opened. | Claude reviewed `trade_guards.py` but didn't differentiate "new position" vs "add". Codex verified the bug with a runtime probe (the log shows the test confirming a block fires after one add-to-existing followed by a new ticker). |
| `cumulative_tracker.py:129-130` — rolling windows compared against wall-clock `now` instead of `snapshots[-1]` timestamp. A stale snapshot file reports change_1d=0.0 even when the last two samples are 24h apart. | Subtle when-was-this-stamped bug. Claude focused on math correctness but didn't audit the *anchor* of rolling windows. |
| `portfolio_mgr.py:68-69` — `_DEFAULT_STATE` shallow-copied; `holdings` and `transactions` are shared with the module-level default. **Mutating one fresh state contaminates later `load_state()` calls** with ghost holdings/transactions in the same process. | Classic Python shared-mutable-default. Claude knew the pattern but didn't trigger on this specific instance. Codex confirmed via runtime test (the log shows `s2 transactions [{'x': 1}]` after `s1` mutation). |
| `portfolio_mgr.py:21-26 / portfolio_validator.py` — `total_fees_sek` not seeded in `_DEFAULT_STATE` but `validate_portfolio()` treats absence as an error. Every fresh portfolio starts invalid. | Claude saw `portfolio_validator.py` (P2-2) but didn't notice this specific schema gap. Codex verified with runtime probe. |

## Claude findings Codex missed

| Claude finding | Why Codex missed it |
|---|---|
| `monte_carlo.py:63` / `risk_management.py:461-462` — `volatility_from_atr` hardcodes `sqrt(252/period)` regardless of candle period. **Underestimates annualized vol by ~5.9x for 1h candles.** | Codex flagged the drift recalibration but didn't notice the volatility scaling bug. **Claude's finding is more severe** (5.9x vol error vs 0.07 drift error). They compound. |
| `monte_carlo_risk.py:426` — `compute_portfolio_var` uses raw `agent_summary.get("fx_rate", 10.0)` instead of `_resolve_fx_rate()`. | Codex didn't audit the fx_rate fallback. Project-specific (the fix was made elsewhere in this PR-equivalent). |
| `kelly_sizing.py:91-95` — `_compute_trade_stats` uses **blended average** of all buys (not FIFO), distorting win_rate / avg_win_pct / avg_loss_pct → wrong Kelly. `equity_curve._pair_round_trips()` has correct FIFO. | Codex didn't compare Kelly's stats computation against equity_curve's. Project-specific knowledge gap. |
| `risk_management.py:374` — ATR stop level anchored to `entry_price`, not `current_price`. Position +20% from entry with 3% ATR has stop at `entry*0.94` — gives up 26% of unrealized gains. | Both the math and the anchor are subtly wrong. Codex didn't trace this. |
| `equity_curve.py:492-493` — `profit_factor` uses net SEK (post-fee) but `win_rate`/`expectancy` use gross pct (pre-fee). Trade with `pnl_pct > 0` but `pnl_sek < 0` (fees ate gain) is win for win_rate but loss for profit_factor. | Codex didn't audit the consistency of the basis (gross/net) across stats. |
| `trade_validation.py:76` — TOCTOU on cash check between concurrent Layer 2 invocations. | Codex didn't model concurrent T1+T2 invocation. |
| `kelly_metals.py:215-217` — silent zero position when `cert_loss_frac=0` (no losses in history). | Codex didn't probe the Kelly defensive zero path. |

## Disagreements

None directly. The reviews are **highly complementary**:
- **Codex** caught money-losing math bugs (warrant negative mark, terminal vs path stop-hit, drift miscalibration) and infrastructure bugs (shared default state, missing schema field, rolling window anchor, trade_guards over-counting).
- **Claude** caught annualization scaling, FIFO vs blended Kelly stats, ATR stop anchoring, fee-basis inconsistency, TOCTOU on cash check.

Both volatility-related findings (Claude's `sqrt(252/14)` and Codex's drift recalibration) are real and orthogonal — fix both.

## What both missed (likely)

- **Multi-strategy Patient/Bold concurrent edits to portfolio_state.{strategy}.json** — Claude flagged in "Did NOT find" that per-file locks are correct. Both passed.
- **Round-trip fee accounting on warrant moves with FX changes intra-position** — neither reviewer asked whether `total_fees_sek` recovers correctly when FX changes between buy and sell of the same warrant.
- **Knockout barrier handling at exactly-zero implied price** — both flagged warrants but neither asked what `pnl_pct = -100%` produces when the leverage formula compounds it (e.g., `current_implied_sek = entry * (1 + (-1.05))` = negative, which Codex caught, but the *post-clamp* downstream — once Codex's fix lands — needs auditing).

## Reconciled verdict

**P0 (must fix — money-math bugs):**
1. **(Codex)** `warrant_portfolio.py:100-103` warrant mark goes negative on knockout — clamp at zero. **Confidence: 95%, money-losing.**
2. **(Claude)** `monte_carlo.py:63` `volatility_from_atr` hardcodes 252-day annualization for non-daily candles — 5.9x vol underestimation. Tail risk hidden.
3. **(Codex)** `monte_carlo.py:328` stop-hit probability uses terminal not path — understates stop-out risk on volatile horizons.
4. **(Claude)** `monte_carlo_risk.py:426` raw fx_rate fallback bypasses `_resolve_fx_rate()`.
5. **(Codex)** `portfolio_mgr.py:68-69` shallow-copied `_DEFAULT_STATE` shares mutable list with module — process-wide ghost transactions.

**P1:**
6. (Codex) `monte_carlo.py:304-307` 1d-calibrated drift reused for 3d run — multi-day MC bias.
7. (Claude) `kelly_sizing.py:91-95` blended-avg not FIFO → wrong Kelly inputs.
8. (Claude) `risk_management.py:374` ATR stop anchored to entry not current price.
9. (Claude) `equity_curve.py:492-493` mixed gross/net basis for win_rate vs profit_factor.
10. (Codex) `trade_guards.py:286-291` over-counts BUY adds toward new-position quota.
11. (Codex) `cumulative_tracker.py:129-130` rolling windows anchored to wall-clock not snapshot timestamp.
12. (Codex) `portfolio_mgr.py:21-26` `total_fees_sek` missing from `_DEFAULT_STATE` but required by validator.
13. (Claude) `trade_validation.py:76` TOCTOU on cash check.

**P2:**
14. (Claude) `kelly_metals.py:215-217` silent zero on `cert_loss_frac=0`.
15. (Claude) `portfolio_validator.py:236-243` `avg_cost_usd` validation includes already-sold shares.
16. (Claude) `risk_management.py:461-462` same vol annualization bug inlined separately.
17. (Claude) `warrant_portfolio.py:259-261` silent oversell clamp.
