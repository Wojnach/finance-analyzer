# Adversarial Review — 3 portfolio-risk (main-thread Claude, independent)

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/risk_management.py:374` — `compute_stop_levels` is LONG-only.
  ```python
  stop_price = entry_price * (1 - 2 * atr_pct / 100)
  ...
  triggered = current_price < stop_price if current_price > 0 else False
  ```
  No direction check on the holding. A BEAR warrant (short underlying) has stops *above* entry, not below. With current code a BEAR cert's stop is always under its bid, so `triggered` never fires, the position runs without a stop, and a 10-15% adverse underlying move → 50-150% cert loss before the user notices. Bear-only check: does this subsystem ever see BEAR positions? Grid fisher places BEAR legs; iskbets places BEAR. Any holding with `direction="SHORT"` (or BEAR cert) walks past this function with no protection. Fix: detect direction from `pos["direction"]` or instrument metadata, flip the sign.

- `portfolio/risk_management.py:373-374` — same function uses UNDERLYING ATR for stop placement on leveraged warrants.
  ```python
  atr_pct = min(atr_pct, 15.0)
  stop_price = entry_price * (1 - 2 * atr_pct / 100)
  ```
  `atr_pct` comes from `agent_summary["signals"][ticker]["atr_pct"]` which is the *underlying's* ATR. A 5x BULL SILVER cert moves 5x as fast as XAG; placing a 2*ATR stop on the underlying value means the cert is already down ~20-30% before the stop trips. Per CLAUDE.md "wider stop-loss defaults for leveraged certificates (min -15% for 5x certs)" is on the Tier-2 backlog — risk_management currently ships the loose pattern. Either multiply atr_pct by `leverage` for cert positions, or refuse to compute stops here for warrant holdings (delegate to grid_fisher/fish_engine which already handle leverage).

- `portfolio/monte_carlo_risk.py:419` — `compute_portfolio_var` reads `fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)` without the sanity band.
  ```python
  fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)
  ```
  `risk_management._resolve_fx_rate` (line 121-150) was created EXACTLY to prevent stale `fx_rate=1.0` in agent_summary from understating SEK figures by 10x — see the comment in that function citing P1-15. The VaR path bypasses it. If a stale agent_summary embeds fx_rate=1.0 (legacy default), `var_95_sek` reports 1/10 the actual SEK risk, and downstream operators (Telegram digest, dashboard) see a tiny number where the real risk is 10x larger. Replace with `_resolve_fx_rate(agent_summary)`.

- `portfolio/kelly_sizing.py:84-100` — `_compute_trade_stats` doesn't FIFO-match buys to sells; it pre-aggregates ALL buys for a ticker into a single weighted average price *before* iterating sells.
  ```python
  total_shares_bought = sum(b.get("shares", 0) for b in buys)
  total_cost = sum(b.get("total_sek", 0) for b in buys)
  ...
  avg_buy_price = total_cost / total_shares_bought
  for sell in sells:
      ...
  ```
  Consequence: a SELL that happens before a later BUY gets matched against an avg that *includes* the buy that came after — temporal leakage. Win/loss flags per round-trip are wrong, so the win_rate fed into `kelly_fraction` is wrong, and Kelly sizes the next position off corrupt edge stats. Fix: use the FIFO matcher from `equity_curve._pair_round_trips:314-426` which is already correct (matches by timestamp order).

## P1 — high-confidence bugs (should fix)

- `portfolio/equity_curve.py:494-504` — `wins` / `losses` classified by `pnl_pct` (GROSS, price-only) but `profit_factor` (line 468-472), `total_pnl_sek`, `expectancy_sek` use `pnl_sek` (NET of fees). Possible outcome: trade with `pnl_pct=+0.1%` but fees > gross → `pnl_sek<0`. It appears in `wins` for streak / win-rate / win_loss_ratio but in the loss pool for `profit_factor`. Dashboard contradiction. Pick one — net (pnl_sek > 0) is the right business definition.

- `portfolio/kelly_metals.py:198-205` — fallback path uses `_DEFAULT_AVG_WIN/LOSS[ticker]` *only* when `outcome_stats` is None or zero. After a fresh DB rotation or insufficient samples, defaults kick in, but the `source` log line still says `"signal_log.db (...)"` if `outcome_stats["win_rate"]` was set even with zero wins. Quote `outcome_stats["avg_win_pct"] > 0 AND avg_loss_pct > 0`: an all-loss stretch produces `avg_win=0`, falls to defaults silently, no operator alert. Surface `"defaults_used_due_to_zero_wins"` in source.

- `portfolio/equity_curve.py:367-374` — `_pair_round_trips` iterates transactions in list order, expecting them to be chronologically sorted on disk. `atomic_append_jsonl` preserves write order, but if the file was ever manually edited (or a corrupt write was repaired), an out-of-order BUY could match against a SELL that happened BEFORE the BUY's wall-clock time. Need a `.sort(key=lambda tx: tx["timestamp"])` pass at function entry. Defensive but cheap.

- `portfolio/trade_guards.py:126-128` — `with _state_lock: state = _load_state()` reads state under lock but immediately releases. Several blocks downstream mutate `state` then write — if two threads pass the lock check simultaneously, both see the same baseline. The actual write at `_save_state` (line 47) is `atomic_write_json` which is last-writer-wins. Two trades from patient+bold in the same cycle can lose one of their cooldown updates. Re-acquire lock for the full read-modify-write cycle.

- `portfolio/risk_management.py:88-110` — `_streaming_max` opens history file with `open(history_path, encoding="utf-8")`, parses line-by-line. If a write is in progress (atomic_append_jsonl uses temp + rename so it's safe on POSIX, but Windows file locks during rename can cause OSError mid-read), we return the *cached* peak via the except branch — correct. But the cached peak's offset isn't updated. Subsequent calls re-attempt from the same offset; if the underlying issue is permanent (rotated and locked), we read from a stale offset forever. Add a "consecutive-read-fail" reset.

- `portfolio/risk_management.py:96` — `val = entry.get(value_key, 0)` defaults to 0. An entry that's structurally valid JSON but missing `patient_value_sek` (e.g., partial schema migration) silently contributes 0 to peak detection. If 0 is below current value, no harm — but the peak history is now silently incomplete. Use `None` and skip; log when `None` count > 0.

- `portfolio/portfolio_mgr.py` (not read but inferred): per CLAUDE.md "atomic state I/O only", any direct `json.dump(open(...))` should be flagged. Verify by grep — `grep -n "json.dump\|open(.*[wW]" portfolio/portfolio_mgr.py`.

## P2 — concerns / smells (worth addressing)

- `portfolio/kelly_metals.py:38-41` — hard-coded `_DEFAULT_AVG_WIN/LOSS` for XAG/XAU only. New tickers (BTC-USD, ETH-USD, MSTR) fall to `.get(ticker, 3.0)` / `.get(ticker, 2.5)` at lines 203-204 with NO source attribution — the log says "default W=3.00% L=2.50%" indistinguishable from a deliberate XAG default. Promote defaults to a config dict with per-ticker fallbacks, log the lookup key.

- `portfolio/kelly_sizing.py:39-52` — `kelly_fraction(p, w, l)` correctly clamps to [0, 1] but doesn't cap at half-Kelly. Caller (`recommended_metals_size`) uses `half_kelly = full_kelly / 2.0` — fine — but other callers grep "kelly_fraction" might use full. Document the contract.

- `portfolio/risk_management.py:373` — `atr_pct = min(atr_pct, 15.0)` cap. If real ATR is 25% (crisis vol), the stop is clamped to 30% wide instead of 50% wide — but the underlying is moving 25% per period anyway. Effectively the cap is "trip the stop on day 1 of a crisis." For metals crisis regimes this might be the correct conservative default; document why.

- `portfolio/equity_curve.py:23-24` — `ANNUALIZATION_DAYS = 365` for Sharpe. Sound for 24/7 crypto+metals, but the curve combines patient (mixed stocks + crypto) and bold (same). When the curve hits stock-only periods (after `MSTR` removal), 365 over-annualizes. Acceptable approximation; mention in metric tooltip.

- `portfolio/monte_carlo_risk.py:77` — `if min_len >= 20` for correlation estimation. 20 observations gives a 95% CI on correlation of ±0.43 — basically useless. Raise to 60 or report SE alongside.

- `portfolio/trade_guards.py:78-81` — `if consecutive_losses >= 4: base = LOSS_ESCALATION[4]` then else lookup. The dict already caps at 4, so the early-return is fine but the comment promises "exponential" (1,1,2,4,8). 4 losses = 8x cooldown = 30min × 8 = 4h. With 5 losses, still 8x. Acceptable.

## Did NOT find

1. **Silent failures**: see P0/P1 above (fx_rate=1.0 path, kelly_metals defaults).
2. **Race conditions**: `_peak_cache_lock` is correct; trade_guards lock scope is too narrow (P1).
3. **Money-losing bugs**: stop-loss direction (P0), warrant leverage (P0), Kelly inputs (P0).
4. **State corruption**: atomic_write_json used everywhere I checked.
5. **Logic errors that pass tests**: BEAR stop direction (P0) — tests likely only assert LONG paths.
6. **Resource leaks**: no subprocess / handle ownership in this subsystem.
7. **Time/timezone bugs**: equity_curve UTC normalization at line 60 correct.
8. **API misuse**: no broker API calls in this subsystem.
9. **Trust boundary**: no external input flowing into eval/exec/shell here.
10. **Incorrect partial-state assumptions**: `pos.get("avg_cost_usd", 0)` etc. defensive everywhere.
