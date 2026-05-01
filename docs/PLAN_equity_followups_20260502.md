# Plan — Equity Curve / Warrant Portfolio Follow-ups (2026-05-02)

## Context

Two outstanding adversarial-review findings from
`docs/PLAN_adversarial_followups_20260502.md`. Both are correctness
fixes in portfolio-metrics surface area — pnl is wrong on the dashboard
and in journals because fees are not netted; warrant avg-in keeps a
stale stop-loss reference because `underlying_entry_price_usd` is not
re-anchored.

## Scope

Two files only:
- `portfolio/equity_curve.py` (P0-6: `_pair_round_trips` line 384)
- `portfolio/warrant_portfolio.py` (PR-P1-1: `record_warrant_transaction` lines 218-227)

Plus their test files. No other portfolio/* file is in scope.

## Findings

### P0-6 — `pnl_sek` excludes fees (`equity_curve.py:384`)

Current code (lines 382-389):
```python
buy_price = buy["price_per_share"]
pnl_pct = ((sell_price_per_share - buy_price) / buy_price * 100) if buy_price > 0 else 0
pnl_sek = (sell_price_per_share - buy_price) * matched

# Proportional fees — use original buy quantity as denominator
buy_fee_share = (buy["fee_sek"] * matched / buy["original_shares"]) if buy["original_shares"] > 0 else 0
sell_fee_share = (sell_fee * matched / sell_shares) if sell_shares > 0 else 0
```

`pnl_sek` is **gross** of fees, but is reported alongside `fee_sek` (the
sum of both buy_fee_share and sell_fee_share) in the same dict. The
dashboard / journal consumers see two fields that look like they should
sum to a "net" P&L, but `pnl_sek` is already supposed to be the net.

#### Cascade

`pnl_sek` is consumed by `compute_trade_metrics()`:
- `gross_profit = sum(t["pnl_sek"] for t in trips if t["pnl_sek"] > 0)`
- `gross_loss = abs(sum(t["pnl_sek"] for t in trips if t["pnl_sek"] < 0))`
- `profit_factor = gross_profit / gross_loss`
- `total_pnl_sek = sum(t["pnl_sek"] for t in trips)`
- Calmar ratio (annualized return / max DD computed from a mini equity
  curve seeded with `t["pnl_sek"]`)

`pnl_pct` is **also** gross-of-fees but the plan doc only calls out
`pnl_sek`. Per the plan's instruction, scope is `pnl_sek` only — leave
`pnl_pct` as gross (label-percent of price move). This matches the user
mental model: "pnl_pct" = price change, "pnl_sek" = realised SEK after
costs.

#### Sharpe / Sortino

Sharpe and Sortino in `compute_metrics()` are computed from
`portfolio_value_history.jsonl`, NOT from round-trip pnl. They are
unaffected by this change. Plan doc mentions them as cascaded — they're
not directly cascaded but they're indirectly affected because the
portfolio-value snapshots upstream presumably already net fees (cash
deduction at trade time). So Sharpe/Sortino computed from value
snapshots are correct independent of this fix; this fix only corrects
**round-trip** P&L attribution.

#### Fix

```python
pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share
```

Move the fee-share computation BEFORE the pnl_sek computation, then
subtract.

### PR-P1-1 — Warrant avg-in stale `underlying_entry_price_usd`

Current code (lines 218-227):
```python
if config_key in holdings:
    # Average in
    existing = holdings[config_key]
    old_units = existing.get("units", 0)
    old_price = existing.get("entry_price_sek", 0)
    new_units = old_units + units
    if new_units > 0:
        avg_price = (old_units * old_price + units * price_sek) / new_units
        existing["units"] = new_units
        existing["entry_price_sek"] = round(avg_price, 2)
```

`existing["underlying_entry_price_usd"]` is never updated. So if you
buy 100 units at silver $79.50 then add 100 units at silver $85.00,
the holding still records `underlying_entry_price_usd: 79.50`. The
stop-loss reference price (consumed by metals stop logic) is then
anchored at the FIRST entry, not the volume-weighted average — the
position will trip its hard-stop sooner than intended.

#### Fix

Same volume-weighted average as for `entry_price_sek`:
```python
old_underlying = existing.get("underlying_entry_price_usd", 0)
if old_underlying > 0 and underlying_price_usd > 0:
    avg_underlying = (old_units * old_underlying + units * underlying_price_usd) / new_units
    existing["underlying_entry_price_usd"] = round(avg_underlying, 4)
```

Guard against missing/zero values: if either price is zero, fall back
to keeping the existing underlying entry (defensive — degenerate input).

## Test plan (TDD)

### `tests/test_equity_curve_fifo.py` — P0-6

Add new test class `TestPnlSekNetOfFees` with 6 tests:
1. **test_simple_full_match_pnl_net_of_fees** — single BUY/SELL, verify
   `pnl_sek == gross_pnl - buy_fee - sell_fee`.
2. **test_partial_sell_pnl_net_of_proportional_fees** — BUY 100, SELL 30,
   verify `pnl_sek` subtracts `0.3 * buy_fee + sell_fee`.
3. **test_multi_partial_sell_total_pnl_net** — BUG-37 case extended:
   sum of `pnl_sek` across both partial trips equals
   `gross - total_buy_fee - total_sell_fee`.
4. **test_zero_fee_pnl_unchanged** — fee_sek=0, pnl_sek matches old
   gross behavior.
5. **test_pnl_pct_unaffected** — `pnl_pct` is still gross (price-change
   only), unchanged by this fix.
6. **test_fee_sek_field_unchanged** — `fee_sek` field in trip dict still
   reports total fees (unchanged).

### `tests/test_portfolio_metrics.py` — P0-6 cascade

Add new test class `TestProfitFactorNetOfFees` with 4 tests covering
the downstream metric contracts:
1. **test_profit_factor_uses_net_pnl** — controlled fixture where gross
   profit and net profit differ; verify profit_factor reflects net.
2. **test_total_pnl_sek_is_net** — `compute_trade_metrics` total_pnl_sek
   equals sum of net pnl_sek across trips.
3. **test_expectancy_pct_uses_pnl_pct_not_pnl_sek** — expectancy is
   computed from pnl_pct (gross %), NOT pnl_sek — so it's unaffected.
   Document this contract.
4. **test_calmar_uses_net_pnl** — Calmar's mini equity curve uses
   `t["pnl_sek"]`, so should reflect net.

### `tests/test_warrant_portfolio.py` — PR-P1-1

Add new test class `TestWarrantAvgInUnderlyingEntry` with 4 tests:
1. **test_avg_in_updates_underlying_entry** — buy 100 @ silver $80, then
   100 more @ silver $90, verify `underlying_entry_price_usd ≈ 85`
   (volume-weighted).
2. **test_avg_in_unequal_volumes** — buy 100 @ $80, then 300 @ $84,
   verify `underlying_entry == 83` (weighted = (100*80 + 300*84)/400).
3. **test_avg_in_zero_underlying_no_change** — degenerate input
   (underlying_price_usd=0) keeps existing underlying entry intact.
4. **test_avg_in_existing_underlying_zero_falls_back** — if existing
   underlying_entry is 0 (corrupted state), don't crash.

## Test status verification

After fixes:
1. Run `tests/test_equity_curve.py` — should be 100% green
2. Run `tests/test_equity_curve_fifo.py` — should be 100% green
3. Run `tests/test_portfolio_metrics.py` — verify cascade metric tests
4. Run `tests/test_warrant_portfolio.py` — should be 100% green
5. Grep for any other test file using `pnl_sek` from round trips —
   verify no regressions.

## Canonical numbers (P0-6 magnitude documentation)

Reference trade: BUY 1 BTC @ 600,000 SEK with 600 SEK fee, SELL 1 BTC
@ 660,000 SEK with 660 SEK fee.

- **Pre-fix**: `pnl_sek = (660000 - 600000) * 1 = +60000.00` (gross)
- **Post-fix**: `pnl_sek = 60000 - 600 - 660 = +58740.00` (net)
- **Magnitude**: 1260 SEK overstatement per trade (~2.1% of pnl).
- **profit_factor cascade**: in a fixture with 2 wins of 60K and 1 loss
  of -40K (all with 1000 SEK fees per side):
  - Pre-fix: gross_profit=120000, gross_loss=40000, PF=3.00
  - Post-fix: net_profit=120000-4000=116000, net_loss=40000+2000=42000, PF=2.76
  - **~8% overstatement** of profit_factor on a typical mix.

## Worktree

`/mnt/q/finance-analyzer-equity-p0p1` on branch
`fix/equity-p0p1-followups-20260502`.

## Commit cadence

One commit per fix:
1. `fix(equity_curve): P0-6 — net pnl_sek of fees in round-trip pairing`
2. `fix(warrant_portfolio): PR-P1-1 — avg-in updates underlying_entry_price_usd`
3. `docs(plan): equity-followups-20260502 plan + canonical numbers`
