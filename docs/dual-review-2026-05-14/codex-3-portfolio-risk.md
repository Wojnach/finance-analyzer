# Adversarial Review — 3 portfolio-risk (second-reviewer / codex-substitute)

> Codex CLI quota was exhausted at start of session. This review is produced by a
> second Claude subagent with isolated context as a substitute second opinion.

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/kelly_metals.py:215-221` — Kelly leverage division explodes to the 0.95 cap on any positive edge
  ```python
  cert_loss_frac = avg_loss * leverage / 100.0
  if cert_loss_frac > 0:
      position_fraction = half_kelly / cert_loss_frac
  ...
  position_fraction = min(position_fraction, MAX_POSITION_FRACTION)
  ```
  With defaults `avg_loss=2.43%` (XAG), `leverage=5`, `cert_loss_frac=0.1215`. Any half_kelly ≥ 0.122 (i.e. full_kelly ≥ 24%, which a 55%/3:2 R:R already produces) saturates the cap at 0.95. Production effect: every metals trade with any positive edge sizes to 95% of buying power in a 5x warrant — 4.75× notional exposure. A 4% adverse move in XAG = 20% capital loss, and the warrant's MINI knockout barrier (~20% on a 5x) is reachable in one session. The cap doesn't actually protect against the warrant blowing through stops. The Kelly-fraction-as-bankroll-risk-tolerance conversion to position-fraction-of-buying-power is sound math, but in combination with leverage the cap of 0.95 is the wrong floor.

- `portfolio/risk_management.py:344-394` — `compute_stop_levels` ignores leverage; warrants get unprotected stops
  ```python
  atr_pct = sig.get("atr_pct", 0)
  atr_pct = min(atr_pct, 15.0)
  stop_price = entry_price * (1 - 2 * atr_pct / 100)
  ```
  ATR is the underlying's ATR (e.g. XAG ~4%). For a held 5x warrant, the stop is set 2*4=8% below underlying entry — but the warrant itself moves 5x faster (40% drawdown). The cap at 15% only worsens this for high-vol underlyings. This is the exact "position sizing that ignores leverage" pattern flagged in the priorities list, applied to stop placement. Compounded by line 382 `triggered = current_price < stop_price` which only models LONG stops — BEAR certs (held as positive units against rising underlying) are never triggered by this function. Production effect: stops trigger far past the warrant's 50%+ drawdown, or never trigger for BEAR holdings.

- `portfolio/trade_validation.py:67-81` — `max_cash_pct=50%` position cap is computed on order_value, not leverage-adjusted notional
  ```python
  if cash_pct > max_cash_pct:
      return ValidationResult(
          False,
          f"Position too large: {cash_pct:.1f}% of cash (max {max_cash_pct:.1f}%)",
      )
  ```
  `order_value = price * volume` is the SEK cost of the warrant cert, not the leveraged exposure. A 50% allocation in a 10x warrant is 5x bankroll notional and still passes validation. The function has no `leverage` parameter and no warrant-aware path. Production effect: warrant trades flagged "safe" can in fact knockout the entire account on a single underlying move.

- `portfolio/monte_carlo_risk.py:419` — `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` accepts stale 1.0 from summary
  ```python
  fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)
  ```
  No sanity-band check. `risk_management._resolve_fx_rate` rejects fx_rate outside [7,15] (defending against the documented 2026-05-02 P1-15 bug where 1.0 understated SEK valuations 10x), but this VaR computation bypasses that guard entirely. If agent_summary still embeds fx_rate=1.0 (legacy writer, mid-rotation, fx_rates.py crash during cycle), all `*_sek` values returned by `compute_portfolio_var` are 10x understated — VaR reads as 1/10 the real number and the risk dashboard says everything's fine while real-SEK risk is 10x. Production effect: silent VaR underreporting, identical class to the false-circuit-breaker bug already fixed in `risk_management.py`.

## P1 — high-confidence bugs (should fix)

- `portfolio/equity_curve.py:467-495` — profit_factor uses pnl_sek (net of fees) but win_rate/streaks use pnl_pct (gross)
  ```python
  gross_profit = sum(t["pnl_sek"] for t in trips if t["pnl_sek"] > 0)
  gross_loss = abs(sum(t["pnl_sek"] for t in trips if t["pnl_sek"] < 0))
  ...
  wins = [t for t in trips if t["pnl_pct"] > 0]
  losses = [t for t in trips if t["pnl_pct"] <= 0]
  ```
  After the P0-6 (2026-05-02) change to make `pnl_sek` net of fees while leaving `pnl_pct` gross, the win/loss buckets diverge. A trade with +0.1% gross but -50 SEK after fees is a "win" for win_rate, max_consecutive_wins, and expectancy, but a "loss" for profit_factor and total_pnl_sek. The streak counter (lines 511-519) likewise uses `pnl_pct > 0`. Production effect: profit_factor and total_pnl_sek can show losses while win_rate shows >50% — dashboard inconsistency that hides fee drag from the operator.

- `portfolio/equity_curve.py:314-426` — FIFO matcher pre-queues ALL buys before processing any sell, ignoring timestamps
  ```python
  buy_queues = defaultdict(list)
  for tx in transactions:
      if tx.get("action") == "BUY":
          ...
  for tx in transactions:
      if tx.get("action") != "SELL":
          continue
  ```
  A sell at time t=5 can match a buy at t=10 if the buy appears earlier in the transactions list. Two passes — first builds the queue, second processes sells. There's no temporal guard. Real portfolios append in transaction order so this usually works, but if the list is ever sorted by ticker (or reloaded from a different ordering), round-trips silently pair against future buys. The portfolio_validator catches negative-share states but not this temporal violation in P&L computation. Production effect: misallocated cost basis, wrong P&L on rebalanced portfolios.

- `portfolio/kelly_sizing.py:106` and `portfolio/kelly_metals.py:97` — Kelly fitted on samples of 2 and 30
  ```python
  if len(pnl_list) < 2:
      return None
  ```
  and
  ```python
  if len(rows) < 30:
      return None
  ```
  With 2 round-trips, the avg_win/avg_loss inputs to `kelly_fraction()` are effectively single observations. At 30 trades the 95% CI on a 55% win rate is roughly ±18% — Kelly sized at the upper bound is double the lower bound. `recommended_size` then uses these unreliable estimates to scale position size up to 30% of cash (bold) or 15% (patient). Production effect: noisy, overconfident sizing driven by lucky streaks. Should require minimum n=50 with confidence-interval haircut.

- `portfolio/risk_management.py:273-276` — Bold-vs-patient detection by substring match on portfolio_path
  ```python
  pf_name = pathlib.Path(portfolio_path).stem
  is_bold = "bold" in pf_name
  value_key = "bold_value_sek" if is_bold else "patient_value_sek"
  ```
  Any path containing "bold" (e.g. `portfolio_state_bold_BACKUP.json`, `embolden_test.json`, `bold_strat.json`) is routed to the bold value key. Conversely, if someone renames to `bold/portfolio_state.json` it would still be bold via the substring. Production effect: when called from tests or with a non-canonical path, peak-tracking reads from the wrong column and the drawdown breaker compares apples to oranges.

- `portfolio/trade_validation.py:84-92` — Spread check passes if `bid > ask` (crossed market)
  ```python
  if bid is not None and ask is not None and bid > 0:
      spread_pct = ((ask - bid) / bid) * 100
      if spread_pct > max_spread_pct:
          ...
  ```
  No assertion that `ask >= bid`. If quotes are crossed (Avanza/Binance feed glitch, milliseconds during rapid moves), `ask - bid` is negative → spread_pct negative → no spread warning. The validator declares the trade valid even though the book is malformed. Production effect: trades sent into corrupt order books pass validation. Should check `ask > bid` explicitly and reject crossed markets.

- `portfolio/risk_management.py:255-270` — Cash-only fallback when agent_summary is empty is silently bullish
  ```python
  current_value = portfolio.get("cash_sek", initial_value)
  ```
  After logging a WARNING, the function returns cash as the current value. But peak_value is read from history and was computed with holdings. So current_value (cash only) ≪ peak_value (cash + holdings at past high) → current_drawdown_pct overstates drawdown. The comment says "the circuit breaker will never trip" but the actual effect is the opposite: drawdown reads as MASSIVE (current = small cash, peak = large past total), tripping the breaker spuriously on every cycle where agent_summary is briefly empty. Production effect: false-positive drawdown breaches every time the summary is being rotated.

- `portfolio/monte_carlo.py:88-97` — `drift_from_probability` calibrated only for T=1/252 but result used for multi-day horizons
  ```python
  mu = volatility * z * math.sqrt(252.0) + 0.5 * volatility**2
  ```
  The formula sets `mu` such that P(S_T > S_0) = p_up exactly when T=1/252. When the same `mu` is then plugged into `simulate_ticker` for `horizons=[1,3]` (3 days), the 3-day probability of ending above spot is materially higher than the input p_up (drift compounds linearly while diffusion compounds with sqrt). For p_up=0.6, vol=0.5, a 3-day path biases ~67% above spot vs the user-specified 60%. Production effect: simulate_ticker over-reports upside probabilities at the 3d horizon — directly affects `p_stop_hit_3d` and `expected_return_3d` used by reporting.

## P2 — concerns / smells (worth addressing)

- `portfolio/portfolio_mgr.py:35-41` — Per-file lock dict has no eviction
  Every unique path string ever requested gets a permanent threading.Lock. Tests that pass tmp_path objects accumulate them. Long-running processes that rotate file names leak Locks indefinitely. Not material in production with 3 fixed state files, but a smell.

- `portfolio/portfolio_mgr.py:53-60` — Backup rotation copies in wrong direction on edge case
  ```python
  for i in range(_MAX_BACKUPS, 1, -1):
      src = path.with_suffix(f".json.bak{i - 1}" if i > 2 else ".json.bak")
      dst = path.with_suffix(f".json.bak{i}")
  ```
  Rotation: 3←2, 2←1. For i=3, src=.bak2, dst=.bak3 (overwrites .bak3, good). For i=2, src=.bak, dst=.bak2 (overwrites .bak2, good). Then line 60 copies current→.bak. OK actually — this logic is correct. But the conditional `i - 1` if i > 2 else .bak is fragile; if `_MAX_BACKUPS` is increased to 4 a future maintainer must regenerate this. Use explicit numeric suffix throughout.

- `portfolio/trade_guards.py:78-100` — `_get_cooldown_multiplier` halves on integer division of elapsed_hours
  ```python
  halvings = int(elapsed_hours // LOSS_DECAY_HOURS)
  base = max(1, base >> halvings)
  ```
  Bit-shift on an int is fine, but `LOSS_ESCALATION` values are 1,1,2,4,8 — for halvings=4 (4 days idle), `8 >> 4 = 0` → max(1,0). The decay reaches 1x at 4 days, which is reasonable but the bit-shift is opaque and breaks if anyone makes LOSS_ESCALATION values non-powers-of-2.

- `portfolio/portfolio_validator.py:118-137` — share_diff tolerance permits silent rounding loss
  ```python
  if actual_shares == 0 and ticker not in holdings and relative_diff < 0.01:
      continue
  ```
  Up to 1% of bought shares can vanish silently. For 1000 shares at 100 SEK, that's up to 1000 SEK of "rounding" passed as OK. Real partial-sell rounding is sub-share — should be tighter (1e-4 or absolute SEK floor).

- `portfolio/equity_curve.py:184-189` — Annualization extrapolates from <1 year of data
  ```python
  if days_elapsed >= 1:
      years = days_elapsed / 365.25
      annualized = (pow(last_val / first_val, 1 / years) - 1) * 100
  ```
  With 7 days of data and +5% return, annualized = (1.05)^52 - 1 = +1190%. Dashboard shows insane numbers when curve is short. Should require `days_elapsed >= 60` or 90 before reporting annualized.

- `portfolio/kelly_metals.py:42` — `_DEFAULT_WIN_RATE = 0.52` is positive-edge by fiat
  When all real data sources fail (no cache, no DB, no agent_summary confidence), code falls back to win_rate=52%, which guarantees a positive Kelly fraction and a recommended position. Cold-start sizing should be `0.50` (no edge) or refuse to size.

- `portfolio/equity_curve.py:494-505` — `losses = [t for t in trips if t["pnl_pct"] <= 0]` puts break-even trades in losses
  Anywhere pnl_pct = exactly 0 (rare but possible after fees zero out) is counted as a loss for win/loss-ratio purposes but contributes 0 to avg_loss_pct. Cosmetic but skews win-rate metrics.

- `portfolio/risk_management.py:725-730` — `CORRELATED_PAIRS` hardcodes BTC↔ETH and XAG↔XAU only
  MSTR↔BTC is documented in CLAUDE.md as a leveraged BTC proxy with high correlation, but `check_correlation_risk` doesn't flag MSTR+BTC simultaneous BUY as concentrated. Same with oil↔metals during inflationary regimes. Missing the third explicit pair.

## Did NOT find

1. Silent failures — `risk_management._compute_portfolio_value` has clear WARNING on cash-only fallback; `_streaming_max` has lock + warning on OS error; trade_guards C4 wiring warning. No try/except: pass swallowing.
2. Race conditions — `portfolio_mgr` has per-file locks; `risk_management._peak_cache_lock` protects PR-P1-2; `trade_guards._state_lock`; `circuit_breaker._lock`. Concurrency is mostly handled.
3. Money-losing PnL sign — `equity_curve._pair_round_trips` correctly subtracts buy fees from sell proceeds; `warrant_pnl` correctly computes underlying_change*leverage.
4. State corruption — atomic_write_json / atomic_append_jsonl used consistently; portfolio_validator runs reconciliation checks.
5. Logic errors that pass tests — TestPnlSekNetOfFees referenced; backup rotation has tests; but reviewed for net-vs-gross inconsistency anyway (logged as P1).
6. Resource leaks — `cumulative_tracker._get_last_snapshot_ts` uses `with open()`; sqlite connections in `kelly_metals._get_outcome_stats` are closed in success path (but NOT in exception path — minor leak smell).
7. Time/timezone bugs — `trade_guards` defensively converts naive to UTC at every parse; `equity_curve._parse_ts` does the same; circuit_breaker uses time.monotonic correctly.
8. API misuse — no Avanza endpoint calls or Binance interval strings in this subsystem; SQLite query is parameterized.
9. Trust boundary violations — no eval/exec/shell; SQLite uses parameterized queries; portfolio_path is path joins on fixed DATA_DIR.
10. Partial-state assumptions — `_validated_state` defends dict shape; `validate_portfolio` has explicit None checks; `portfolio_value` guards against missing keys.
