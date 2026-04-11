# Agent Review: signals-modules — Round 5 (2026-04-11)

**Agent**: feature-dev:code-reviewer
**Files reviewed**: 26 (all files in portfolio/signals/)
**Duration**: ~173s

---

## Findings (9 total: 1 P0, 3 P1, 3 P2, 2 P3)

### P0

**SM-R5-5** forecast.py:100-103 — _PREDICTION_DEDUP_EVICT_AGE defined but eviction never implemented
- `_PREDICTION_DEDUP_EVICT_AGE = 600` constant exists but no code iterates `_last_prediction_ts` to evict stale entries
- Memory leak in long-running loop: ~2100 entries/hour (7 TF × 5 tickers × 60 cycles)
- Fix: Add eviction sweep in prediction logging path

### P1

**SM-R5-1** (pre-existing H17) volume_flow.py:61-69 — VWAP cumulative from bar 0
- Confirmed still open. Produces systematic BUY bias in uptrending assets.

**SM-R5-2** crypto_macro.py:228,281 — OPTIONS_TTL used before module-level definition
- Works at runtime (Python resolves at call time) but NameError risk in circular import
- Fix: Move constant to line ~30 with other constants

**SM-R5-7** [CRITICAL NEW] news_event.py:255-265 — "cut" fallthrough counts bearish headlines as bullish
- Any headline with "cut" not matching "rate cut" or "guidance cut" falls through to `pos += 1`
- "job cut", "profit cut", "earnings cut", "rating cut" all counted as POSITIVE sentiment
- **Direct signal inversion affecting trading decisions**
- Fix: Route unknown "cut" to `neg += 1`, whitelist only "rate cut"/"tax cut" as positive

### P2

**SM-R5-8** cot_positioning.py:54-58 — Relative paths for data files
- `load_json("data/{metal}_deep_context.json")` uses relative path
- Silently returns default (HOLD) if CWD is wrong

**SM-R5-9** credit_spread.py:283-289 — Raw open("config.json") violates Rule 4
- Uses relative path + raw open() instead of file_utils.load_json

**SM-R5-10** volatility.py:86-93 — BB squeeze + breakout double-count on release
- Both sub-signals fire in same direction on squeeze-release bar
- 2/7 vote from single event instead of 1/7

### P3

**SM-R5-11** volume_flow.py:289 — Default price_up=True on NaN creates BUY bias
- First bar gets spurious BUY from Volume RSI when price direction unknown

**SM-R5-12** futures_flow.py:65,92 — price_start truthy guard too broad
- `if price_start and price_end > price_start` suppresses signal when price_start==0

---

## Regression Check
- A-SM-1 (gap-fill guard): HOLDING — fix correctly prevents inverted BUY on widening gap
- A-SM-2 (GARCH schema): HOLDING — GARCH key present in empty schema
- FOMC conflict (calendar vs econ_calendar): FALSE POSITIVE — they govern different time windows
