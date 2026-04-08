# Adversarial Review: metals-core (Agent Findings)

Reviewer: Code-reviewer subagent (feature-dev:code-reviewer)
Date: 2026-04-08
The most critical subsystem — 6561-line god file trading real money.

---

## CRITICAL

### CM1. `NameError: config_data` crashes strategy orchestrator at startup [100% confidence]
**File**: `data/metals_loop.py:5936`

`load_strategies(config_data)` called but `config_data` never defined. Actual config is
`config` (line 540). NameError every startup. Caught by except block → orchestrator
silently fails. Any code depending on `_strategy_orchestrator` breaks silently.

**Fix**: Replace `config_data` with `config`.

### CM2. `order["price"]` KeyError in stop-fill accounting [100% confidence]
**File**: `data/metals_loop.py:4024`

Stop order dicts use `"trigger"` and `"sell"` keys, never `"price"`. When a stop fires,
`order["price"]` → KeyError inside try/except. Position state updates partially, but
subsequent stop levels in same iteration are skipped. Produces confusing error log
masking real fill events.

**Fix**: `order.get('trigger', order.get('sell', '?'))`.

### CM3. `read_signal_data` uses raw `open()` — reads partial JSON mid-write [95% confidence]
**File**: `data/metals_loop.py:1036-1049`

Violates Critical Rule #4. `agent_summary.json` written every 60s by main loop. Raw
`open()/json.load()` during mid-write → JSONDecodeError → returns `{}`. With active
position, signal data cleared → sell triggers suppressed.

**Fix**: Use `from portfolio.file_utils import load_json`.

### CM4. Silver velocity alert fires twice at 5-minute epoch boundary [85% confidence]
**File**: `data/metals_loop.py:909-915`

Dedup key `vel_{int(time.time() // 300)}` changes at epoch boundary. Drop starting 2s
before boundary → alert fires, key rolls over, fires again. Double alert + potential
double `_autonomous_decision` invocation.

---

## HIGH

### HM1. Silver alert thresholds never reset on recovery — single-alert-per-level forever [92% confidence]
**File**: `data/metals_loop.py:893-898`

`_silver_alerted_levels` accumulates across position lifetime. `_silver_reset_session()`
defined but NEVER CALLED. After drop → recovery → second drop, no re-alert fires.
User relies on these alerts for position protection.

### HM2. Silver fast-tick warrant % uses linear approximation — understates loss near barrier [88% confidence]
**File**: `data/metals_loop.py:870-875`

`warrant_pct = pct_change * leverage`. For MINI warrants near knock-out, actual loss
follows `(price - financing_level) / (entry - financing_level)`, not linear. At large
deviations (10% drop, 4.76x MINI), real loss can exceed displayed estimate by 20-50%.
False sense of safety.

### HM3. ORB morning range constants hardcoded for winter UTC — wrong 6 months/year [90% confidence]
**File**: `portfolio/orb_predictor.py:32-35`

`MORNING_START_UTC = 8` (= 09:00 CET winter). In summer CEST (UTC+2), 09:00 is 07:00 UTC.
Constants capture 10:00-12:00 CEST instead of 09:00-11:00. ORB predictions wrong from
March to October.

### HM4. Trade queue fallback stop at 15% below — no barrier distance check [88% confidence]
**File**: `data/metals_loop.py:3749-3755`

When Layer 2 doesn't include `stop_trigger`, fallback `exec_price * 0.85` (15% below).
For MINI warrants, this may be below knock-out barrier (worthless) or within 3% minimum
required by memory rules. No barrier check on fallback.

### HM5. Fish engine tick uses `datetime.now()` for CET — wrong if system clock is UTC [85% confidence]
**File**: `data/metals_loop.py:1986-1994`

`datetime.datetime.now()` returns system local time. If UTC, `hour_cet` is 1-2h wrong.
Fish engine may open positions after warrant market close (21:55 CET).

### HM6. EOD fishing sell uses naive local time — wrong timezone on UTC system [87% confidence]
**File**: `data/metals_loop.py:6059-6074`

21:50 time check uses `datetime.datetime.now()` without timezone. On UTC system,
fires at 21:50 UTC = 22:50/23:50 CET (after close). `is_market_hours()` returns False
→ EOD sell logic never triggers. Positions not sold by EOD.

---

## MEDIUM

### MM1. `price_history` is list with O(n) `pop(0)` — should be deque [82% confidence]
**File**: `data/metals_loop.py:550`

120-element list with `pop(0)` every 60s. Comment says "circular buffer" but uses wrong
data structure. No lock protecting concurrent reads.

### MM2. microstructure_state `persist_state()` double-appends OFI to history [80% confidence]
**File**: `portfolio/microstructure_state.py:194-199`

`persist_state()` calls `get_microstructure_state()` which calls `record_ofi()`.
If called on same cycle as another `get_microstructure_state()`, OFI appended twice.
Contaminates z-score history.

### MM3. exit_optimizer timezone-naive entry_ts raises TypeError [80% confidence]
**File**: `portfolio/exit_optimizer.py:390-391`

`market.asof_ts` (aware) minus `position.entry_ts` (potentially naive from fromisoformat).
TypeError on subtraction → no exit plan returned.

### MM4. Stop cleanup crash between cancel and save leaves stale state [82% confidence]
**File**: `data/metals_loop.py:2993-3006`

`del stop_state[key]` before `_save_stop_orders`. Crash between → file still has old entry
→ restart re-cancels already-cancelled orders or re-places stops for sold position.

### MM5. `fill_probability` returns 1.0 unconditionally for target <= price — overconfident buy estimate [80% confidence]
**File**: `portfolio/price_targets.py:65-68`

Near-price buy targets get reflected to slightly above price due to float → 1.0 returned
instead of real probability.

---

## LOW

### LM1. ORB percentile uses floor index — systematically conservative estimates [80% confidence]
**File**: `portfolio/orb_predictor.py:339-349`

`int(len * pct / 100)` truncates → biased toward lower values. P25 of 5 elements
returns index 1 (P20). All ORB targets conservatively biased.

### LM2. metals_cross_assets yfinance stale data on Monday morning [80% confidence]
**File**: `portfolio/metals_cross_assets.py:27-39`

Monday morning: `df.iloc[-1]` is Friday's close. "Today" change shows Friday→Thursday,
misleading for overnight metals signal.

---

## Cross-Critique: Claude Direct vs Metals-Core Agent

### Agent found that Claude missed:
1. **CM1**: `config_data` NameError — total miss (100% confidence crash bug)
2. **CM2**: `order["price"]` KeyError on stop fill — total miss (100% confidence)
3. **CM3**: Raw `open()` violating rule #4 — missed (I noted god file generically)
4. **CM4**: Velocity alert double-fire at epoch boundary — total miss
5. **HM1**: Silver alert thresholds never reset — total miss (defined but never called)
6. **HM2**: MINI warrant linear approximation — total miss (math correctness)
7. **HM3**: ORB constants wrong for summer — total miss (DST-specific)
8. **HM4**: Trade queue 15% fallback no barrier check — missed
9. **HM5**: Fish engine uses naive datetime for CET — missed
10. **HM6**: EOD sell uses wrong timezone — missed
11. **MM2**: microstructure double-append OFI — missed
12. **MM4**: Stop cleanup crash state — missed

### Claude found that agent confirmed:
1. **C2**: God file systemic risk — confirmed as the overarching theme
2. **H8/H9**: Playwright page death, memory leak — not specifically re-raised
3. **H10**: Kelly override on no-edge — not in scope (fish engine Kelly is separate)
4. **M7**: ORB private method call — confirmed as part of broader ORB analysis

### Net assessment:
The metals-core agent found **4 CRITICAL + 6 HIGH + 5 MEDIUM + 2 LOW = 17 net-new issues**.
CM1 (`config_data` NameError) and CM2 (`order["price"]` KeyError) are **confirmed crash bugs**
with 100% confidence — they will fire every time their code paths execute. These are not
theoretical; they are active production bugs.

This was the hardest subsystem to review (6561 lines) and the agent was **overwhelmingly
stronger** — finding concrete crash bugs that a broad review of the god file couldn't catch.
