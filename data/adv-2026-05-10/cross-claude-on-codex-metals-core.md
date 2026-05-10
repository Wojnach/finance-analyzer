# Claude critique of codex findings — metals-core

## Verdicts

- [P0] Include the support modules — data/metals_loop.py:212-217
  Verdict: CONFIRMED
  Reason: Imports like `metals_shared`, `portfolio.file_utils`, `portfolio.loop_contract` are real hard dependencies present in the repo. Codex reviewed a fresh worktree checkout missing these support modules during the git safety check (dubious ownership error at lines 16-36 of codex report). This is a real issue for a clean checkout, though modules exist in main.

- [P2] Avoid changing working directory on import — data/metals_loop.py:209
  Verdict: CONFIRMED
  Reason: `os.chdir(BASE_DIR)` at module import time mutates global process state. Violates documented contract: "safe to import from pytest". Callers that expect relative paths or spawn subprocesses will behave differently after import.

- [P1] Don't build BEAR ladders from long-side defaults — portfolio/fin_snipe.py:160-166
  Verdict: CONFIRMED
  Reason: `build_intraday_ladder()` call omits `direction_sign` parameter. Function signature (portfolio/metals_ladder.py:87) defaults to `direction_sign: int = 1`. BEAR products need `direction_sign=-1` to invert buy/sell targets. Without it, inverse warrants get wrong bid/exit targets.

- [P1] Translate optimizer exits with inverse sign for BEAR holdings — portfolio/fin_snipe_manager.py:492-497
  Verdict: CONFIRMED
  Reason: `translate_underlying_target()` call at line 492 omits `direction_sign` parameter. Defaults to long-only (`direction_sign=1`). For BEAR short positions, favorable price drop (lower underlying) should increase warrant value, but long-only formula produces lower exit price instead.

- [P1] Skip BEAR MINI warrants once spot crosses barrier — portfolio/fin_fish.py:732-735
  Verdict: CONFIRMED
  Reason: When `direction == "SHORT"` and `spot >= barrier`, code executes bare `pass` (line 735) instead of `continue`. Knocked-out BEAR MINIs proceed to `evaluate_warrants()` line 742+ and enter results. Guard at 737-738 only checks distance, not knockout state. Allows recommending impossible trades.

## New findings (mine)

- [P1] No direction detection before ladder builder call — portfolio/fin_snipe.py:160
  Snapshot captures `name` field from Avanza (line 176) which contains direction hints ("BEAR SILVER", "BULL GULD", etc.), but this field is never parsed to extract BEAR vs BULL. Should detect "BEAR" in instrument name or via market.get("isBullCertificate")/similar indicator, then pass `direction_sign=-1` to `build_intraday_ladder()` at line 160. Currently all products treated as long.

- [P1] Entry underlying back-calculation ignores position direction — portfolio/fin_snipe_manager.py:365-373
  `_estimate_entry_underlying()` uses formula `(current_price / entry_price - 1) / leverage`. For BEAR products, if warrant price went from 100→110 SEK, the underlying actually dropped (short position gained). Formula needs sign reversal for short: `(entry_price / current_price - 1) / leverage` or track position direction as a parameter. Without this, `translate_underlying_target()` at line 492 still produces inverted results even if direction_sign is later corrected.

- [P0] Missing support module (metals_shared) breaks fresh checkout — data/metals_loop.py:212
  Import `from metals_shared import get_cet_time` fails on a clean checkout because `data/metals_shared.py` is not included in the diff. Codex encountered git ownership error first, but this is the real blocker: the module must be committed to the branch or the import restructured.

## Summary

- Confirmed: 5
- Partial: 0
- False-positive: 0
- New: 3
