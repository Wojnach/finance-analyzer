# Microstructure-Features Walk-Forward Backtest — Input-Side Pivot Design

**Date:** 2026-07-20
**Status:** Design only — driver not built yet.
**Goal:** Stop varying the model (LLM matrix plateaued at coin-flip-ish); vary the INPUTS.
Feed historically reconstructable microstructure features into the existing
`scripts/xgb_backtest.py` walk-forward architecture and measure whether they beat the
indicator-only baseline on the same grid, same rows, same scorer.

---

## 1. Data availability — verified verdicts (2026-07-20, empirical)

All verdicts below were verified by direct calls against the live endpoints and the
`data.binance.vision` S3 bucket listing (`https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=...`),
not from docs alone.

### 1.0 Hard constraint found: metals listing dates

`GET https://fapi.binance.com/fapi/v1/exchangeInfo` `onboardDate`:

| Symbol             | Onboarded      | Consequence                                       |
| ------------------ | -------------- | ------------------------------------------------- |
| BTCUSDT (FAPI)     | 2019-09-08     | full 2025-07..2026-07 window OK                   |
| ETHUSDT (FAPI)     | 2019-11-27     | full window OK                                    |
| **XAUUSDT (FAPI)** | **2025-12-11** | **no Binance data of any kind before 2025-12-11** |
| **XAGUSDT (FAPI)** | **2026-01-07** | **no Binance data before 2026-01-07**             |

The requested 2025-07..2026-07 window is only fully coverable for crypto. Metals get
~6–7 months (minus 200-candle warmup). yfinance `GC=F`/`SI=F` could extend price history
but carries **no** taker-buy/funding/OI fields, so the microstructure backtest for metals
is capped at the listing dates. Accept this; don't fake it with proxies.

### 1.1 Klines with taker-buy fields — YES, full window, already fetched

- Spot REST `https://api.binance.com/api/v3/klines`, FAPI REST
  `https://fapi.binance.com/fapi/v1/klines`: every kline row carries
  `quote_vol, trades, taker_buy_base_vol, taker_buy_quote_vol` (fields 8–11).
  `scripts/llm_backtest.py:fetch_klines_1h` already retrieves them (see §2).
- Bulk: `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/...` and
  `data/futures/um/{monthly,daily}/klines/<SYM>/<interval>/` — same 12 columns.
- **This is the primary OFI proxy: per-candle taker-buy imbalance.** Zero new
  infrastructure needed for Stage 0.

### 1.2 aggTrades — YES (bulk), heavy

- Bulk monthly + daily zips, verified present through 2026-06 (monthly) / July 2026 (daily):
  - `https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-06.zip` (475 MB)
  - `data/futures/um/monthly/aggTrades/BTCUSDT/...` (668 MB/month)
  - `data/futures/um/monthly/aggTrades/XAUUSDT/XAUUSDT-aggTrades-2025-12.zip` .. `2026-06.zip` (~110 MB/month)
- REST `/api/v3/aggTrades` supports time-windowed paging but is infeasible for a year.
- Fields: price, qty, timestamp, `isBuyerMaker` → signed trades → VPIN, signed volume,
  large-trade share, trade-throughs. **Stage-2 material** (disk/CPU cost, see §4/§6).

### 1.3 Funding rate history — YES, full window, trivial

- REST `https://fapi.binance.com/fapi/v1/fundingRate?symbol=...&startTime=...&limit=1000`
  — verified returning data from 2025-07-01 (BTCUSDT) and from listing (XAUUSDT).
  Full year = ~1100 rows/symbol (8h settlements) = 2 requests.
- Bulk mirror: `data/futures/um/monthly/fundingRate/<SYM>/` (verified for XAUUSDT 2025-12..2026-06).
- Optional continuous proxy: `data/futures/um/monthly/premiumIndexKlines/<SYM>/1m/` (verified).

### 1.4 Open interest history — YES via bulk `metrics`, NO via REST beyond 30 days

- REST `/futures/data/openInterestHist` rejects `startTime` older than ~30 days
  (verified: error `-1130`). The repo's `futures_data.get_open_interest_history` is
  therefore useless for backfill.
- **Bulk `metrics` dataset**: `data/futures/um/daily/metrics/<SYM>/<SYM>-metrics-YYYY-MM-DD.zip`
  (~11 KB/day, 288 rows at 5-min). Verified columns:
  `create_time, symbol, sum_open_interest, sum_open_interest_value,
count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
count_long_short_ratio, sum_taker_long_short_vol_ratio`.
  Verified coverage: BTCUSDT through 2025-07 and earlier; XAUUSDT 2025-12-11..2026-07-18;
  XAGUSDT from 2026-01-07.

### 1.5 Long/short ratio + taker buy/sell ratio — YES via same `metrics` files

REST `/futures/data/globalLongShortAccountRatio` etc. have the same 30-day cap; the
`metrics` files above carry top-trader L/S (accounts + positions), global L/S, and taker
buy/sell volume ratio at 5-min for the full window. One download pipeline covers §1.4 + §1.5.

### 1.6 Liquidation data — NO, not reconstructable

- REST `allForceOrders` was removed by Binance years ago; the websocket `forceOrder`
  stream is throttled (≤1 msg/sec/symbol) and only forward-looking.
- The historical `liquidationSnapshot` bulk dataset is **gone**: S3 listing under
  `data/futures/um/daily/liquidationSnapshot/` returns 0 keys (verified 2026-07-20).
- Verdict: exclude. Paid vendors (Coinglass API, Tardis.dev) have it; out of scope for
  a local/no-SaaS setup.

### 1.7 Order-book depth history — NOT replayable (confirmed), with one partial exception

- **Spot:** `data/spot/daily/` contains only `aggTrades`, `klines`, `trades` (verified).
  No depth, no bookTicker. L1/L2 spot history does not exist publicly.
- **Futures bookTicker** (L1 best bid/ask): bulk dumps exist but **publication stopped
  April 2024** (verified: `data/futures/um/monthly/bookTicker/BTCUSDT/` ends at
  `2024-04.zip`; XAUUSDT has none at all). No L1 for our window →
  `microstructure.compute_ofi` (Cont et al., needs L1 sizes) and `spread_zscore`
  are **not** reconstructable offline.
- **Partial exception — futures `bookDepth` dataset**:
  `data/futures/um/daily/bookDepth/<SYM>/` (verified for XAUUSDT from 2025-12-11,
  XAGUSDT from 2026-01-07, BTCUSDT; ~0.5 MB/day). Columns:
  `timestamp, percentage, depth, notional` — bid/ask depth at ±1..±5% bands, sampled
  every ~25–30 s. Supports a **band depth-imbalance** feature
  (ln(bid_notional) − ln(ask_notional) at ±1%), NOT tick-level OFI.
- Alternatives for true L2 replay: [Tardis.dev](https://docs.tardis.dev/historical-data-details/binance-futures)
  reconstructs full order-book state from archived streams (paid). Rejected: external
  SaaS, against project constraints.
- **Design consequence: kline taker-buy imbalance is the OFI proxy** (primary), with
  bookDepth band imbalance as a Stage-2 supplement.

### 1.8 Cross-asset via yfinance — YES with caveats

DXY (`DX-Y.NYB`), SPY at 1h: yfinance serves hourly bars for ~730 days — covers the
window. Session gaps (nights/weekends/holidays) require as-of merge + staleness flags,
not naive alignment. Repo precedent: `portfolio/metals_cross_assets.py` (already fetches
SPY/GVZ/copper via `price_source`/yfinance).

Sources:

- https://github.com/binance/binance-public-data (official bulk-data repo; documents klines/trades/aggTrades)
- https://data.binance.vision (verified listings quoted above)
- https://docs.tardis.dev/historical-data-details/binance-futures (L2 replay alternative, paid)

---

## 2. Repo assets — reusable-code map

### Pure feature math — reusable offline

| Code                                                                                    | Offline reuse                                                                                                                       |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `portfolio/microstructure.py:41` `trade_flow_imbalance`                                 | YES — feed aggTrades-derived `{qty, sign}` lists (Stage 2)                                                                          |
| `portfolio/microstructure.py:63` `compute_vpin`                                         | YES — pure, volume-bucketed; aggTrades input (Stage 2)                                                                              |
| `portfolio/microstructure.py:172` `detect_trade_throughs`                               | YES — aggTrades input (Stage 2)                                                                                                     |
| `portfolio/microstructure.py:23` `depth_imbalance`                                      | ADAPT — same log-ratio math against bookDepth ±1% band notionals (Stage 2)                                                          |
| `portfolio/microstructure.py:107` `compute_ofi`                                         | NO — needs L1 snapshot sequence; not reconstructable (§1.7)                                                                         |
| `portfolio/microstructure.py:151` `spread_zscore`                                       | NO — needs spread history; not reconstructable                                                                                      |
| `portfolio/microstructure_state.py:127` `get_multiscale_ofi` + `:146` flow-acceleration | CONCEPT ONLY — port the fast-minus-slow windowing to rolling taker-imbalance columns; the ring-buffer code itself is live-loop-only |
| `portfolio/metals_orderbook.py:21` `SYMBOL_MAP`, `:116` `isBuyerMaker→sign` convention  | YES — reuse mapping + sign convention for aggTrades parsing                                                                         |
| `portfolio/metals_orderbook.py:48,91` snapshot fetchers                                 | NO — live REST snapshots, nothing historical                                                                                        |

### Data plumbing

- `portfolio/futures_data.py:184` `get_funding_rate_history` — right endpoint, but fetches
  only the latest `limit` rows (no `startTime` paging). Backtest needs its own paged
  fetcher; reuse the endpoint/params shape.
- `portfolio/futures_data.py:69` `get_open_interest_history` and `:97-181` L/S ratio
  fetchers — dead ends for backfill (30-day REST cap, §1.4); replace with bulk `metrics`.
- `portfolio/data_collector.py:67-71` `_BINANCE_KLINE_COLS` — live kline fetch **keeps**
  `quote_vol, trades, taker_buy_vol, taker_buy_quote_vol` columns; `:94-95` casts only
  OHLCV to float, so taker columns sit in the DataFrame as strings, unused.
- `scripts/llm_backtest.py:53-108` `fetch_klines_1h` — **does NOT drop any columns**:
  all 12 kline fields land in the DataFrame as `qv, n, tbb, tbq, ig` (`:88-104`); only
  OHLCV is cast to float (`:105-106`). The taker-buy data has been fetched all along —
  `build_context` (`:206`) and `xgb_backtest.feature_frame` just never read it.
  **The pivot driver needs one extra cast, not a new fetcher.**

### Walk-forward architecture to reuse verbatim (`scripts/xgb_backtest.py`)

- `:43` `feature_frame` — pattern for causal feature construction (extend, don't replace)
- `:70` `label_arrays` — label + resolve-time semantics
- `:106-133` fetch/pad logic (delegates to `lb.fetch_klines_1h`)
- `:144-181` grid construction (mirrors `llm_backtest.run`, same `--start/--end/--step-hours/--interval`)
- `:183-189` resume dedup on `(model, interval, at, ticker)`
- `:191-207` `train()` expanding window, `resolve <= at` no-leakage mask
- `:232-243` row schema (`model, interval, arm, at, ticker, vote, conf, outcome_*`) —
  identical rows → results land in the same matrix and `llm_backtest.py --score` works unchanged
- `:29-38` `XGB_PARAMS`, `:39` `MIN_TRAIN_ROWS`, `:40` `MIN_HIST` — keep identical for
  a clean ablation (only inputs vary)

---

## 3. Proposed feature set (all causal, all reconstructable)

Naming: features are per-candle columns added to `feature_frame`'s existing 12
(rsi, macd_hist, ema_gap_pct, bb_pos, vol_ratio, chg24, lr1/3/6/12, fng) — the baseline
stays in as the control group.

### Group A — kline microstructure (Stage 0; no new data source)

| Feature                         | Definition                                                                  |
| ------------------------------- | --------------------------------------------------------------------------- |
| `taker_imb`                     | `2*tbb/volume − 1` (signed OFI proxy per candle)                            |
| `taker_imb_r4`, `taker_imb_r24` | rolling mean of `taker_imb` over 4 / 24 candles                             |
| `taker_imb_z`                   | `taker_imb_r4` z-scored vs 100-candle rolling distribution                  |
| `flow_accel`                    | `taker_imb_r4 − taker_imb_r24` (ports `get_multiscale_ofi` fast-minus-slow) |
| `trade_intensity_z`             | trade count `n` z vs 100-candle rolling                                     |
| `avg_trade_size_z`              | `volume/n` z vs rolling (large-trade proxy)                                 |
| `vol_accel`                     | `log(vol_r4 / vol_r24)` volume acceleration                                 |
| `clv`                           | `(2*close − high − low)/(high − low)` intrabar close location               |

### Group B — realized-vol structure (Stage 0)

| Feature      | Definition                                                                   |
| ------------ | ---------------------------------------------------------------------------- |
| `rv_ratio`   | realized vol (std of lr1) 6-candle vs 48-candle                              |
| `pk_vol_z`   | Parkinson vol from high/low, z vs rolling                                    |
| `ret_skew24` | rolling 24-candle return skewness (repo precedent: realized_skewness signal) |

### Group C — perp positioning (Stage 1; REST funding + bulk metrics, merge_asof)

| Feature                        | Definition                                         |
| ------------------------------ | -------------------------------------------------- |
| `funding`                      | last settled funding rate (known at `fundingTime`) |
| `funding_delta`                | change vs previous settlement                      |
| `funding_z30`                  | z vs trailing 30-day settlements                   |
| `oi_norm`                      | `sum_open_interest` / its 7-day rolling mean       |
| `oi_d1h`, `oi_d24h`            | log OI change over 1h / 24h (from 5-min metrics)   |
| `taker_ls_ratio`               | `sum_taker_long_short_vol_ratio`, 1h mean + z      |
| `top_ls_pos`, `top_ls_pos_d24` | top-trader position L/S ratio, level + 24h delta   |
| `global_ls`                    | `count_long_short_ratio` level                     |

### Group D — cross-asset (Stage 1; yfinance / `price_source`)

| Feature                          | Definition                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------- |
| `dxy_r1`, `dxy_r24`              | DXY 1h / 24h log return (last closed bar, as-of)                                |
| `spy_r1`, `spy_r24`, `spy_stale` | SPY returns + staleness flag (sessions)                                         |
| `btc_r1`, `btc_r24`              | BTC return as exogenous column on ETH/XAU/XAG rows                              |
| `gs_ratio_z`                     | gold/silver ratio z (metals rows only; repo precedent `metals_cross_assets.py`) |

### Group E — bulk tick/depth (Stage 2, only if Stage 0/1 shows lift)

| Feature                               | Source + reused code                                      |
| ------------------------------------- | --------------------------------------------------------- |
| `vpin`                                | aggTrades per-bar buckets → `microstructure.compute_vpin` |
| `signed_vol_pct`, `large_trade_share` | aggTrades → `trade_flow_imbalance` + size deciles         |
| `through_cnt`                         | aggTrades → `detect_trade_throughs`                       |
| `bd_imb_1pct`, `bd_imb_5pct`          | bookDepth band log-imbalance (adapted `depth_imbalance`)  |
| `bd_imb_d1h`                          | 1h change in band imbalance                               |

Notes:

- Crypto rows keep spot klines (Groups A/B from spot, as in the existing matrix); Group C/E
  attach the PERP's positioning to the spot row — that is a cross-market feature by
  construction, flagged as such in results.
- Metals rows are FAPI-native for everything.
- NaN policy: Group C/D merged `merge_asof(direction="backward")` with per-source staleness
  caps (metrics 15 min, funding 8h+5m, DXY/SPY 2h during sessions); beyond cap → NaN and
  the existing `valid` mask drops the row, except SPY/DXY off-session which uses last close +
  `*_stale=1` indicator so weekends aren't discarded for 24/7 assets.

---

## 4. Driver plan — `scripts/micro_backtest.py`

One new script, cloned from `xgb_backtest.py` (264 lines), same CLI, same grid, same
rows, same resume, same scorer. Differences only:

1. **Fetch**: reuse `lb.fetch_klines_1h` (already returns tbb/tbq/n/qv), then cast the
   four extra columns to float in a small `cast_micro(df)` helper. Do not modify
   `llm_backtest.py` (concurrent-editing rule: new file, no edits to tracked machinery).
2. **External data helpers** (cached under `data/micro_cache/`, parquet):
   - `fetch_funding(symbol, start, end)` — REST paged `startTime`, 1000/req (~2 req/yr/symbol).
   - `fetch_metrics(symbol, start, end)` — download daily `metrics` zips from
     data.binance.vision, concat → one parquet per symbol (~4 MB/yr). Skip-and-log missing days.
   - Stage 2 only: `fetch_bookdepth(symbol, ...)` (same pattern, ~0.5 MB/day) and
     `fetch_aggtrades_month(symbol, month)` — stream-process one monthly zip at a time into
     per-bar aggregates, delete the zip afterwards (bounded disk: ≤1 GB transient).
3. **`feature_frame(df, fng, extras, feature_set)`** — baseline columns (copied verbatim
   for parity) + groups gated by `--feature-set`:
   `base` (parity control) | `kline` (=base+A+B) | `full` (=base+A+B+C+D) | `tick` (adds E).
4. **Model naming**: `model = f"xgbmicro-{feature_set}-{h}h"` → rows coexist with
   `xgboost-<h>h` and all LLM rows in the same JSONL matrix; the untouched
   `llm_backtest.py --score` prints them side by side.
5. **Identical**: `XGB_PARAMS`, `MIN_TRAIN_ROWS=200`, `MIN_HIST=120`, expanding-window
   `train()` with `resolve <= at`, grid/outcome/resume/abstain semantics, arm "A".
6. **No-leakage additions**: metrics rows shifted +5 min (publication lag); funding attaches
   at `fundingTime`; DXY/SPY use last _closed_ hourly bar; every Group A/B rolling op is
   `rolling`/`ewm`/`shift` on candles `<= t`, mirroring the documented convention at
   `xgb_backtest.py:44-46`.

Planned invocations (Stage 0 first):

```bash
python scripts/micro_backtest.py --feature-set base  --interval 1h --start 2025-08-01 --end 2026-07-11 --tickers BTC-USD,ETH-USD --out data/xgb_backtest_results.jsonl
python scripts/micro_backtest.py --feature-set kline --interval 1h ...   # A/B ablation
python scripts/micro_backtest.py --feature-set full  --interval 1h ...   # +C/D
python scripts/micro_backtest.py --feature-set full  --interval 4h ...
# metals (window auto-truncates to listing + warmup):
python scripts/micro_backtest.py --feature-set full --interval 1h --tickers XAU-USD,XAG-USD --start 2025-12-20 --end 2026-07-11
```

---

## 5. Staged validation plan

- **Stage 0 — crypto 1h, klines only (hours of work, no new data).**
  `base` vs `kline` on BTC/ETH, 2025-08..2026-07, step 8h (~2×1000 grid points).
  `base` re-run guards against any drift vs the old `xgboost-*` rows. Gate to continue:
  `kline` ≥ `base` + 2pp directional accuracy at the 24h horizon, or ≥53% at 1h/3h,
  paired on identical grid stamps (binomial CI on the delta; same-row pairing makes
  this a McNemar-style comparison, cheap to compute from the JSONL).
- **Stage 1 — crypto 1h + 4h, `full` set.**
  Cost context (why 53–69%): with round-trip cost c and mean |move| r̄ at horizon h,
  break-even directional accuracy ≈ (1 + c/r̄)/2. At 1h crypto moves are small relative
  to spot taker fees+spread → hurdle lands at the high end (~60s–69%); at 4h/24h the
  hurdle falls toward 53–55%. Compute the exact per-horizon hurdle from
  `portfolio/cost_model.py` inputs when scoring, and report cost-adjusted accuracy,
  not raw accuracy alone (also report vote rate — a model that only abstains is useless).
- **Stage 2 — tick/depth features (`tick` set), crypto first.**
  Only if Stage 0/1 shows lift. ~6 GB transient download per symbol-year (aggTrades),
  processed month-by-month on the Deck (or herc2 if awake — optional per current setup).
- **Stage 3 — metals.** `full` on XAU (from 2025-12-20) and XAG (from 2026-01-16), 1h.
  Metals hurdle must use warrant costs (Avanza courtage+spread from `cost_model.py`),
  which are far above crypto spot — expect the hurdle near the top of the band.
  Thin sample (~600 grid points/symbol) → wider CIs; treat as directional evidence only.
- **Holdout discipline:** freeze 2026-05-15..2026-07-11 as untouched holdout; feature-set
  selection happens on data before that; a single final run on the holdout decides.

---

## 6. Open risks

1. **Metals window is short and unfixable** — XAUUSDT 2025-12-11+, XAGUSDT 2026-01-07+
   (Binance listing dates). ~600 grid points at 1h/8h-step; 4h/1d intervals mostly
   infeasible for metals (warmup eats the window).
2. **Spot/perp mismatch (crypto)** — funding/OI/bookDepth describe the USDT-M perp;
   labels and klines are spot. Signal may dilute across the basis. Metals don't have
   this problem (perp-native).
3. **Pre-existing as-of convention** — `build_context`/`feature_frame` read the as-of
   candle's close at its open_time. Internally consistent (outcome is anchored to the
   same close), but keep it byte-identical in the new driver or comparability with the
   existing matrix breaks.
4. **`metrics` data quality** — occasional missing days in the bulk dumps; ratio-column
   definitions (count* vs sum* variants) are semi-documented. Treat as opaque features,
   forward-fill ≤15 min, drop beyond.
5. **Multiple comparisons** — 4 feature sets × 3 horizons × 2 intervals × 4 tickers will
   produce a spurious winner by chance. Holdout (Stage 5 discipline) + paired deltas
   only; no cherry-picking horizons post hoc.
6. **Deck resources** — Stage 0/1 trivial (metrics ≈ 4 MB/yr/symbol). Stage 2 aggTrades
   is tens of GB transient + hours of CPU; needs month-streaming and possibly herc2.
7. **yfinance fragility** — DXY/SPY hourly limited to ~730 days, session gaps, silent
   schema changes; staleness flags mitigate, but cross-asset features may be the first
   to break on re-runs.
8. **Break-even band is venue-dependent** — the 53–69% targets shift with the cost
   model chosen; publish the hurdle alongside accuracy in every scoring report.
9. **Binance bulk availability is not contractual** — bookTicker dumps silently stopped
   in 2024; `liquidationSnapshot` disappeared entirely. Cache all downloaded raw zips'
   parquet derivatives in `data/micro_cache/` so a future disappearance doesn't strand
   the backtest.
