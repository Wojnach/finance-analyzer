# Agent Adversarial Review: portfolio-risk

**Agent**: feature-dev:code-reviewer
**Subsystem**: portfolio-risk (4,281 lines, 14 files)
**Duration**: ~280 seconds
**Findings**: 15 (3 P0, 6 P1, 4 P2, 2 P3)

---

## P0 Findings (CRITICAL)

### A-PR-1: record_trade() Never Called in Production — Trade Guards Non-Functional [P0]
- **File**: `portfolio/trade_guards.py:177`
- **Description**: The entire overtrading-prevention system (per-ticker cooldowns, consecutive-loss escalation, position rate limits) depends on `record_trade()` being called after every trade. A codebase-wide grep shows it's called ONLY in test files. `trade_guard_state.json` has no entries. Cooldowns are always 0. The code itself has a self-aware warning at line 278-285 that this is broken.
- **Impact**: Layer 2 can re-enter the same position multiple times per hour. Loss escalation multiplier permanently 1x. Zero overtrading protection in production.
- **Fix**: Call `record_trade(ticker, direction, strategy, pnl_pct)` wherever trades are executed — at minimum in the journal writer and agent_invocation.py trade processing.

### A-PR-2: Drawdown Peak Only Scans Last 2000 History Entries (~33h) [P0]
- **File**: `portfolio/risk_management.py:99`
- **Description**: `check_drawdown()` reads only the last 2000 entries from `portfolio_value_history.jsonl` (written every 60s = ~33h coverage). If the true peak occurred earlier, it's truncated. The circuit breaker becomes blind to drawdowns over older peaks.
- **Impact**: A portfolio at 470K that peaked at 600K three days ago shows only 1.1% drawdown instead of 21.7%. Circuit breaker does not trip.
- **Fix**: Maintain a separate `portfolio_peak.json` updated atomically on every `log_portfolio_value()` call. Never truncate the peak.

### A-PR-3: portfolio_validator.py Uses Raw json.load() — Violates Atomic I/O Rule [P0]
- **File**: `portfolio/portfolio_validator.py:258`
- **Description**: Opens portfolio state with raw `open()` + `json.load()` instead of `file_utils.load_json()`. Can read partial file during an ongoing atomic write, causing false corruption reports.
- **Impact**: False validation errors that could trigger erroneous recovery from stale backups.
- **Fix**: Replace with `load_json(path, default=None)`.

---

## P1 Findings

### A-PR-4: VaR Uses Fixed fx_rate — No USD/SEK Correlation [P1]
- **File**: `portfolio/monte_carlo_risk.py:502-505`
- **Description**: SEK VaR applies static fx_rate snapshot. Doesn't model USD/SEK correlation during risk-off.
- **Impact**: Underestimates SEK losses by ~3K when USD weakens simultaneously with crypto crashes.

### A-PR-5: kelly_metals Returns Non-Zero Growth With Zero Position [P1]
- **File**: `portfolio/kelly_metals.py:241-250`
- **Description**: `monthly_growth_pct` can be non-zero when `position_sek = 0` and `units = 0`. No guard on `leverage = 0`.
- **Impact**: Misleading growth projections in Telegram notifications.

### A-PR-6: Kelly P&L Uses All-Time Average Instead of FIFO [P1]
- **File**: `portfolio/kelly_sizing.py:84-103`
- **Description**: Win-rate estimation averages all BUY prices instead of FIFO matching. Masks individual lot P&L.
- **Impact**: Biased win probability → systematically wrong position sizes.
- **Fix**: Reuse `equity_curve._pair_round_trips()` for correct FIFO matching.

### A-PR-7: Concentration Limit Warning Never Blocks [P1]
- **File**: `portfolio/risk_management.py:589-601`
- **Description**: Returns `severity: "warning"` only, never `"block"`. No code path converts concentration warnings to trade blocks. 100% single-ticker positions possible.
- **Fix**: Route concentration flags through `should_block_trade()` or add enforcement threshold.

### A-PR-8: Trade Guards Check/Record Not Atomic — Race Condition [P1]
- **File**: `portfolio/trade_guards.py:52, 177`
- **Description**: `check_overtrading_guards()` and `record_trade()` each independently load/save state. Concurrent Layer 2 sessions can both pass the check before either records.
- **Fix**: File-level lock around the load-check-write cycle.

### A-PR-9: Calmar Ratio Uses Trade-P&L Not Mark-to-Market [P1]
- **File**: `portfolio/equity_curve.py:519-530`
- **Description**: Computes drawdown from cumulative round-trip P&L, ignoring unrealized gains/losses. Appears too favorable.

---

## P2 Findings

### A-PR-10: Monte Carlo ATR Annualization Wrong for Hourly Candles [P2]
- **File**: `portfolio/monte_carlo.py:54-57`
- **Description**: Uses `sqrt(252/period)` assuming daily candles. For hourly candles, correct factor is `sqrt(252*24/period)` — 4.9x larger. Understates volatility by ~5x.
- **Impact**: All Monte Carlo simulations overconfident. Stop-hit probability underestimated.

### A-PR-11: Consecutive Loss Counter Has No Time-Based Decay [P2]
- **File**: `portfolio/trade_guards.py:28`
- **Description**: After 4+ consecutive losses, cooldown multiplier is permanently 8x. No reset mechanism.

### A-PR-12: Cash Reconciliation 1 SEK Tolerance Too Tight [P2]
- **File**: `portfolio/portfolio_validator.py:84`

### A-PR-13: Kelly 500 SEK Minimum vs CLAUDE.md 1000 SEK [P2]
- **File**: `portfolio/kelly_sizing.py:291`
- **Description**: Kelly recommends trades at 500 SEK, below CLAUDE.md 1000 SEK minimum.

---

## P3 Findings

### A-PR-14: Monte Carlo `or` Operator Masks ATR=0 [P3]
- **File**: `portfolio/monte_carlo.py:281`

### A-PR-15: Circuit Breaker Naming Confusion [P3]
- **File**: `portfolio/circuit_breaker.py`
- **Description**: Named "circuit breaker" but is an API retry guard. The drawdown circuit breaker is separate in risk_management.py. Drawdown breaker has no enforcement in non-metals code.
