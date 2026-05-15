# Cross-Critique — 3 portfolio-risk

## Agreement — high-confidence findings (both reviewers)

- **`portfolio/risk_management.py:374` — `compute_stop_levels` LONG-only and leverage-blind, P0.** Both reviewers identify the exact two-bug compound: (a) `current_price < stop_price` only triggers LONG (BEAR certs never tripped); (b) the 2*atr_pct stop uses *underlying* ATR on leveraged warrants, so the cert is already 20-40% down before the underlying stop hits. Codex frames it as "stops trigger far past 50% drawdown or never trigger for BEAR holdings"; Claude says the same. **Independent rediscovery — confidence very high.** Action: detect direction from `pos["direction"]`/instrument metadata, multiply atr_pct by leverage for cert positions.

- **`portfolio/monte_carlo_risk.py:419` — `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` bypasses `_resolve_fx_rate` sanity band, P0.** Both reviewers pinpoint the same line. Both reference the documented P1-15 (2026-05-02) bug. Both note `risk_management._resolve_fx_rate` rejects fx_rate outside [7,15] but the VaR path is bypassed. **High-confidence, one-line fix.** Action: replace with `_resolve_fx_rate(agent_summary)`.

- **`portfolio/equity_curve.py:467-505` — profit_factor uses `pnl_sek` net but `wins/losses/streaks` use `pnl_pct` gross, P1.** Same line ranges, same diagnosis, same fix. Codex frames it as "post-2026-05-02 net-of-fees migration created the divergence". **Independent rediscovery.** Action: unify on net.

- **Kelly inputs are corrupt (P0).** Claude flags `kelly_sizing._compute_trade_stats:84-100` aggregates all buys before processing sells (temporal leakage). Codex flags the same shape at `equity_curve._pair_round_trips:314-426` (sell-at-t=5 matches buy-at-t=10 if list-ordered first). Same family — temporal-order assumption made by the matcher. **Both right.** Action: explicit `.sort(key=timestamp)` at entry, plus FIFO matcher in equity_curve.

## Codex found, Claude missed

- **`portfolio/kelly_metals.py:215-221` — leverage division saturates to 0.95 cap on any positive edge (P0).** Codex's catch: `cert_loss_frac = avg_loss * leverage / 100.0` with XAG defaults (avg_loss=2.43%, leverage=5) = 0.1215. Any half_kelly ≥ 0.122 (i.e. full_kelly ≥ 24%, which a 55%/3:2 R:R produces routinely) saturates at MAX_POSITION_FRACTION=0.95 → 4.75× notional exposure on a 5x warrant. Claude flagged "kelly_fraction doesn't cap at half-Kelly" (P2) but missed the much sharper consequence. **Codex right, elevate to P0.**

- **`portfolio/trade_validation.py:67-81` — `max_cash_pct=50%` is on order_value, not leverage-adjusted notional (P0).** Claude didn't open trade_validation.py. Codex: a 50% allocation in a 10x warrant is 5x bankroll notional and still passes "safe". **Real P0 — gate that all manual trades pass through is leverage-blind.**

- **`portfolio/trade_validation.py:84-92` — spread check passes if `bid > ask` (crossed market).** Claude missed entirely. Real P1 — bad-data path lets trades through on feed glitches. One-line fix.

- **`portfolio/risk_management.py:255-270` — cash-only fallback when agent_summary empty produces FALSE-POSITIVE drawdown breaches.** Codex's catch is subtle: current_value=cash (small) vs peak_value=cash+holdings (large) → drawdown massive → spurious circuit-breaker trip every cycle the summary rotates. **Real P1; Claude focused on FX path instead but this is adjacent.**

- **`portfolio/monte_carlo.py:88-97` — `drift_from_probability` calibrated for T=1/252, used for 3-day horizons.** Codex: p_up=0.6 input produces actual 67% at 3d. Direct hit on `p_stop_hit_3d` and `expected_return_3d` reporting. Claude didn't open monte_carlo.py. **Real P1, mathematical specificity Codex got right.**

- **`portfolio/risk_management.py:273-276` — bold-vs-patient detected by substring "bold" in path.** Codex's catch: `bold_state_BACKUP.json` or `embolden.json` route to wrong column. Real P1 for test isolation; could trip in prod if a backup path is ever passed.

- **Multiple P2 quality issues (Codex only).** `kelly_metals._DEFAULT_WIN_RATE = 0.52` (positive-edge by fiat — should be 0.50 cold-start). `equity_curve` annualization extrapolates from <1y data → +1190% dashboard numbers on 7d. `risk_management.CORRELATED_PAIRS` missing MSTR↔BTC. `portfolio_validator` 1% share-diff tolerance permits ~1000 SEK silent loss. `portfolio_mgr` per-file lock dict has no eviction. All real, none P0.

## Claude found, Codex missed

- **`portfolio/kelly_metals.py:198-205` — fallback path's "source" log lies when `outcome_stats["win_rate"]` is set but `avg_win=0` (all-loss stretch).** Codex flagged "n<30 returns None" (P1) but Claude's point is sharper: the default-fallback path *runs* on zero-win streaks but the source attribution still claims real DB stats. Operator misled. Claude is right — narrow but real.

- **`portfolio/risk_management.py:88-110` — `_streaming_max` cached-peak offset isn't updated when read fails.** Claude's catch: if the read fault is permanent (locked rotated file), reads forever from stale offset. Codex didn't notice. Real P2.

- **`portfolio/risk_management.py:96` — `val = entry.get(value_key, 0)` defaults to 0 silently contributing to peak detection.** Claude is right — missing key → 0 → silently incomplete peak history. Codex missed.

- **`portfolio/trade_guards.py:126-128` — lock released between `_load_state` read and write.** Two patient+bold threads in the same cycle can race and lose one cooldown update. Real P1 for ticker-pool concurrency. Codex acknowledged trade_guards locking elsewhere but missed this specific scope hole.

## Disagreements

None on substance. Codex elevates kelly_metals leverage saturation to P0 (one-line cap miscalibration); Claude had `kelly_fraction half-Kelly` at P2. **Codex's framing wins** — the 4.75× notional exposure is a money-losing severity that warrants P0 treatment.

## What BOTH missed (third pass)

- **`portfolio/risk_management.py` `compute_correlation_risk`** is only flagged by Codex as missing MSTR↔BTC pair (P2). Neither reviewer checked whether the correlation gate computes correlation from realized returns or just lookups the hardcoded list. If hardcoded only, *every* new ticker is invisible to the gate until manually added.

- **`portfolio/kelly_sizing.py` `recommended_size`** — neither reviewer checked the interaction with `risk_management.max_drawdown_action`. If Kelly says 30% and drawdown gate says HALT, do they compose correctly? Without source review, unknown.

- **`portfolio/portfolio_mgr.py` was explicitly skipped by Claude** ("not read but inferred"). Codex only reviewed P2 backup rotation. Neither reviewer audited the SHARES vs CASH atomicity — the write order in `record_transaction` matters: if cash decrements before shares increment and crash happens between, the JSONL is consistent but state.json shows phantom cash. Untested.

- **`portfolio/equity_curve.py` `_parse_ts`** — both said timezone handling is correct. Neither checked whether `ts` from older transactions (pre-UTC migration) is still naive-local-time in `data/portfolio_state*.json`. If yes, the equity-curve interpretation drifts by CET offset for any pre-migration trade.

## Verdict

P0 list after cross: **5 confirmed** (LONG-only stops, leverage-blind stops, fx_rate bypass, Kelly leverage saturation, max_cash_pct leverage-blind).
P1 list after cross: **~8 confirmed** (gross vs net, FIFO temporal leak, cooldown lock scope, crossed-market spread, cash-only false drawdown, MC 1-day calibration extrapolation, bold-path substring, _streaming_max stale offset).
P2 list after cross: ~10 (defaults, annualization, correlation pairs, share rounding, lock eviction).

Portfolio-risk has the **highest concentration of P0 money-losing bugs** of any subsystem reviewed.
