# Cross-critique — metals-core

## Codex findings Claude missed

| Codex finding | Why Claude missed it |
|---|---|
| `data/metals_llm.py:27-28` — Importing `metals_llm` from `metals_loop.py` **unconditionally `chdir`s the entire process** to `Q:/finance-analyzer` and prepends to sys.path. In any other checkout (CI, worktree, this very review's `Q:/fa-review`!) it raises `FileNotFoundError` (caller only catches `ImportError`). When the path *does* exist but isn't the current checkout, all relative reads/writes go to the *wrong* repo. | **Critical config-bleed bug Claude missed entirely.** Claude reviewed metals_loop import logic but didn't notice the embedded chdir. The bug is invisible in a normal Q:\finance-analyzer checkout because the path exists and matches CWD. It only manifests in worktrees, CI, or moved installations — exactly this review setup. |
| `data/metals_warrant_refresh.py:171-173` — Avanza market-guide returns numeric fields as **value objects** (e.g. `{"value": 12.34, "unit": "SEK"}`); the rest of the codebase unwraps with `_v`/`_value` helpers. Here `bid` and `ask` stay as dicts → `bid <= 0` raises `TypeError` → `refresh_warrant_catalog()` aborts → `load_catalog_or_fetch()` falls back to **stale cache forever**. **Live warrant discovery is broken.** | Claude focused on stop-loss flow but didn't audit warrant catalog refresh. Codex caught the API shape bug. |
| `portfolio/exit_optimizer.py:617-621` — `hold_ev` is supposed to be the mean terminal P&L, but the code averages P&L at the 10/25/50/75/90th percentiles. **Not the expected value** — for skewed payoff distributions (knock-out warrants are very skewed), this materially mis-ranks `hold_to_close` vs market/limit exits. | Claude reviewed `exit_optimizer.py` but didn't audit the EV math itself. Statistical bug. |
| `portfolio/orb_predictor.py:384-388` — MINI long intrinsic value floored at zero is missing → `intrinsic_target` can go negative → `format_prediction()` produces losses worse than -100% and negative `warrant_price_factor`. | Claude flagged warrant barriers in `orb_predictor` per memory rules but didn't trace the math down to format. |
| `data/metals_execution_engine.py:137-141` — `chronos_24h_pct` is a **24h return** but converted with `sqrt(252)` (volatility scaling). 1% becomes 0.16 instead of 2.52 (≈16x too small). Inconsistent with `portfolio.fin_fish` which uses `*252`. | Numerical convention bug Claude missed. |

## Claude findings Codex missed

| Claude finding | Why Codex missed it |
|---|---|
| `data/metals_loop.py:4793-4802` — `_handle_buy_fill` calls `place_stop_loss` with `trigger_type`/`value_type` kwargs that the function signature **does not accept**. **The "hardware trailing stop" advertised in config has never worked.** Every queue-path fill is left without broker-side trailing protection. | Codex didn't trace the kwargs against the signature. **Silent feature death — exact pattern of the March-April outage** (advertised feature silently inert). |
| `data/metals_swing_trader.py:142, 2738` — Imports `place_stop_loss` from `portfolio.avanza_control` but unpacks the return as `(success, stop_id)`. `avanza_control.place_stop_loss` returns `(ok, dict)`. `pos["stop_order_id"]` stores a dict; later `_delete_stop_loss(page, ACCOUNT_ID, dict)` fails silently. **Hardware stops on swing trader positions are never cancelled on exit.** | Codex didn't compare the two `place_stop_loss` implementations side-by-side. Same return-shape mismatch class as Codex's avanza-api findings. |
| `data/metals_swing_trader.py:3151` — `_execute_sell` doesn't cancel stop-loss BEFORE placing sell. `short.sell.not.allowed` reject loop possible (volume reserved by stop). Fish engine has the correct pattern; swing trader doesn't. | Codex didn't compare the two execution paths. |
| `data/metals_swing_trader.py:537-538` — `_cet_hour()` fallback uses fixed UTC+1, wrong from late March to late October. | Codex didn't audit the timezone fallback. |
| `data/metals_swing_trader.py:2426, 2758` — `close_cet = 21.0 + 55/60` hardcoded across DST gap weeks. | Codex didn't audit the DST gap. |
| `data/fish_engine.py:654-655` — bare `except Exception: pass` on layer2 vote tactic (silent swallow). | Codex didn't audit fish_engine error paths. Documented bug history makes this material. |

## Disagreements

None. The reviews target almost completely disjoint code:
- **Codex**: import-time side effects (metals_llm chdir), API shape bugs (warrant_refresh value objects), math errors (exit_optimizer EV, orb_predictor floor, metals_execution_engine drift scaling).
- **Claude**: stop-loss kwargs/return-shape misuse (metals_loop, metals_swing_trader), DST/timezone bugs, fish_engine silent swallows.

Both are critical. **The dual-review thesis is strongly validated here.**

## What both missed (likely)

- **`data/silver_monitor.py`** — neither reviewer flagged anything. Claude's "Did NOT find" notes silver_monitor was merged into metals_loop in v10, but the file still exists in the tree (was committed but unused?). Worth an explicit dead-code audit.
- **`portfolio/iskbets.py` quick-gamble path** — both reviewers skipped or under-reviewed this. It places real orders.
- **`portfolio/fin_snipe.py` and `fin_snipe_manager.py`** — neither flagged. Bid/exit ladder logic is high-leverage; deserves its own focused review.
- **Concurrent `metals_loop` + `silver_fomc_loop`** — Claude noted the v10 merge consolidates silver_monitor; neither asked whether `silver_fomc_loop.py` (if active) can race against `metals_loop` order placement.

## Reconciled verdict

**P0 (must fix — silent feature death + money-losing math):**
1. **(Claude)** `metals_loop.py:4793-4802` hardware trailing stop kwargs unrecognized — feature has never worked.
2. **(Claude)** `metals_swing_trader.py:142, 2738` wrong import (`avanza_control.place_stop_loss` returns dict, not stop_id). Hardware stops on swing trader never cancelled.
3. **(Codex)** `metals_warrant_refresh.py:171-173` value-object dict not unwrapped → live warrant discovery broken, stale cache used.
4. **(Codex)** `metals_llm.py:27-28` import-time chdir + sys.path mutation → process-wide config bleed, breaks any non-canonical checkout.
5. **(Codex)** `orb_predictor.py:384-388` warrant intrinsic not floored → impossible negative outputs.
6. **(Codex)** `metals_execution_engine.py:137-141` Chronos 24h return scaled as volatility (16x too small).

**P1:**
7. (Claude) `metals_swing_trader.py:3151` no cancel-before-sell → potential stop reservation conflict loop.
8. (Codex) `exit_optimizer.py:617-621` `hold_ev` averages percentiles, not mean (skew-sensitive).
9. (Claude) `metals_swing_trader.py:537-538` DST fallback wrong half the year.
10. (Claude) `metals_swing_trader.py:2426, 2758` hardcoded 21:55 EOD (DST gap weeks).

**P2:**
11. (Claude) `fish_engine.py:654-655` bare `except: pass` on layer2 vote.
12. (Claude) `metals_loop.py:4907-4915` proximity guard skipped on missing live bid.
