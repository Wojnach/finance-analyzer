# Agent Adversarial Review: metals-core

**Agent**: feature-dev:code-reviewer
**Subsystem**: metals-core (19,014 lines, 21 files — LARGEST subsystem)
**Duration**: ~430 seconds
**Findings**: 21 (6 P0, 9 P1, 4 P2, 2 P3)

---

## P0 Findings (CRITICAL — Active Money-Losing Bugs)

### A-MC-1: HARD_STOP_CERT_PCT=0.05 — Stops Fire on Normal 1% Noise at 5x [P0]
- **File**: `portfolio/fin_snipe_manager.py:61,472`
- **Description**: 5% cert stop at 5x leverage = 1% underlying move. Silver moves 1-2% in 15-min candles routinely. Violates CLAUDE.md rule: "5x leverage certs need -15%+ stops."
- **Impact**: Guaranteed stop-outs on normal volatility. Positions exit during routine retracements.
- **Fix**: Set `HARD_STOP_CERT_PCT = 0.15` (minimum) or compute dynamically from leverage.

### A-MC-2: usdsek=1.0 Hardcoded in Exit Optimizer — All SEK P&L Wrong by ~10x [P0]
- **File**: `portfolio/fin_snipe_manager.py:420`
- **Description**: `MarketSnapshot(usdsek=1.0)` — real rate is ~10.3. All SEK targets from exit optimizer are off by 10x.
- **Impact**: Every limit sell and profit target is priced using phantom exchange rate. Actively running NOW.
- **Fix**: Pass live FX rate from `fx_rates.get_fx_rate()` or `shared_state`.

### A-MC-3: ORB Window 1 Hour Wrong During CEST (Apr-Oct) [P0]
- **File**: `portfolio/orb_predictor.py:32-34`
- **Description**: `MORNING_START_UTC=8` hardcoded for CET winter. During CEST (today is Apr 10), the actual 09:00-11:00 CET window is 07:00-09:00 UTC. System observes 10:00-12:00 CET instead.
- **Impact**: All ORB predictions wrong for 7 months of the year. Active NOW.
- **Fix**: Compute UTC offset dynamically using `zoneinfo.ZoneInfo("Europe/Stockholm")`.

### A-MC-4: entry_ts Always Set to now() — HOLD_TIME_EXTENDED Permanently Disabled [P0]
- **File**: `portfolio/fin_snipe_manager.py:409`
- **Description**: `entry_ts=dt.datetime.now(dt.UTC)` passed on every cycle, not the actual entry time. Exit optimizer always sees hold time ≈ 0.
- **Impact**: Positions never receive "held too long" escalation. 3-5h hold limit unenforced.
- **Fix**: Retrieve actual entry timestamp from instrument_state or snipe state.

### A-MC-5: Raw open() on layer2_journal.jsonl — Torn Reads [P0]
- **File**: `data/metals_loop.py:~2150`
- **Description**: Reads JSONL with raw `open()` + `readlines()`. Concurrent Layer 2 writes produce truncated lines.
- **Fix**: Use `file_utils.load_jsonl_tail()`.

### A-MC-6: Raw open() + json.load() for Cert Catalog at Startup [P0]
- **File**: `data/metals_loop.py:~1487`
- **Description**: Catalog loaded with raw JSON, not atomic I/O. Concurrent write corruption at startup.
- **Fix**: Use `file_utils.load_json()`.

---

## P1 Findings

### A-MC-7: Hardware Trailing Stop trigger_price=HARDWARE_TRAILING_PCT — Ambiguous Units [P1]
- **File**: `data/metals_loop.py:4094`
- **Description**: Passes percentage value (e.g. 5.0) as `trigger_price`. If API interprets as absolute SEK price, stop is set to 5 SEK — never triggers.
- **Fix**: Verify Avanza API contract for PERCENTAGE + FOLLOW_DOWNWARDS. Log full payload.

### A-MC-8: MIN_STOP_DISTANCE_PCT=1.0 Violates 3% Rule [P1]
- **File**: `portfolio/fin_snipe_manager.py:64`
- **Description**: Allows stops 1% below bid. Rule says "NEVER place stop within 3%."
- **Fix**: Set `MIN_STOP_DISTANCE_PCT = 3.0`.

### A-MC-9: Warrant Portfolio BUY Averaging Skips underlying_entry_price_usd [P1]
- **File**: `portfolio/warrant_portfolio.py:219-228`
- **Description**: Second BUY updates `entry_price_sek` (weighted avg) but NOT `underlying_entry_price_usd`. P&L calculations use wrong denominator.

### A-MC-10: persist_state() Double-Appends OFI — Z-Scores Inflated [P1]
- **File**: `portfolio/microstructure_state.py:191-199`
- **Description**: `get_microstructure_state()` called from persist_state appends OFI as side effect. 5-6 copies per cycle. Compressed variance → inflated z-scores → false signals.

### A-MC-11: datetime.now() Without TZ in Fish Engine — Hours Off by 1-2h [P1]
- **File**: `data/metals_loop.py:~2227`

### A-MC-12: Ask Price Used as current_price for Long Positions — Stop Distances Understated [P1]
- **File**: `portfolio/fin_snipe.py:~85`

### A-MC-13: iskbets Gate Defaults to APPROVE on LLM Parse Failure [P1]
- **File**: `portfolio/iskbets.py:287`
- **Description**: `approved = True` default. Any failure (network, parse, empty) silently approves trade.
- **Fix**: Default to `False`. Require explicit "APPROVE" in response.

### A-MC-14: fill_probability_buy Zero-Drift Formula With Non-Zero Drift [P1]
- **File**: `portfolio/price_targets.py:~106`

### A-MC-15: Default leverage=4.76 for Silver Fast-Tick — Wrong for Most Instruments [P1]
- **File**: `data/metals_loop.py:~1061`

---

## P2 Findings

### A-MC-16: orb_postmortem.py Raw open() on JSONL History [P2]
### A-MC-17: orb_postmortem.py Raw open() on Predictions JSON [P2]
### A-MC-18: assert in Production Path in fin_fish.py [P2]
### A-MC-19: vol_scalar Inverted — Narrows Levels in Volatile Markets [P2]
- **File**: `data/metals_loop.py:~1950`
- **Description**: `vol_scalar = median_atr / current_atr` — high vol produces scalar < 1, narrowing dip targets. Should expand them.
- **Fix**: Invert to `current_atr / median_atr`.

---

## P3 Findings
### A-MC-20: orb_backtest Directional Accuracy Uses Predicted Values [P3]
### A-MC-21: Leave-N-Out Slightly Conservative [P3]
