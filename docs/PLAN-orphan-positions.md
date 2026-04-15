# PLAN — Adopt Orphan Avanza Positions Into SwingTrader

Branch: `fix/ingest-orphan-positions`
Date: 2026-04-15

## Context

Today `BULL SILVER X5 AVA 4` (ob_id=1650161, 97u, entry 10.27 SEK, live ~10.14)
peaked at +5.78% PnL and faded back to -1.27% with **no exit executed**. The
5% trailing stop that the operator expects to lock in gains never engaged
because the position is not tracked by the working exit machinery.

Root cause (from exploration):

- The position lives in the legacy `POSITIONS` dict
  (`data/metals_positions_state.json`, key `bull_silver_x5`) populated by
  `detect_holdings()` in `data/metals_loop.py:1777` when Avanza holdings
  include a hardcoded `KNOWN_WARRANT_OB_IDS` entry that is NOT tagged
  `_managed_by=swing_trader`.
- Every trailing-stop path attached to legacy POSITIONS is gated by
  `STOP_ORDER_ENABLED=False` (line 399) + `HARDWARE_TRAILING_ENABLED=True`
  (line 425). The legacy trailing-stop writer never runs. No Avanza-side
  stop-loss order exists for this instrument (verified live).
- The working `SwingTrader` (`data/metals_swing_trader.py`) manages
  positions in `data/metals_swing_state.json`, places Avanza-side hardware
  stop-losses in `_set_stop_loss()` (line 1376), and has active trailing
  + momentum-exit logic. But it owns only positions it buys itself — no
  ingestion API for pre-held positions.

Goal: make every Avanza-held metals position a first-class SwingTrader
position so the exit machinery always covers it.

## Design

Two coordinated changes:

### 1. `SwingTrader.ingest_position()` — new public method

New method in `data/metals_swing_trader.py` between `_reconcile_swing_positions`
(line 514) and `_verify_recent_fills` (line 579).

```python
def ingest_position(
    self,
    ob_id: str,
    units: int,
    entry_price: float,
    underlying_price: float,
    direction: str = "LONG",
    set_stop_loss: bool = True,
) -> str | None:
    """Adopt an already-held Avanza position into SwingTrader management.

    Returns pos_id on success, None if catalog lookup fails or duplicate.
    """
```

Behavior:

1. Reject if any position in `self.state["positions"]` already has this `ob_id`.
2. Look up warrant metadata by ob_id:
   - Iterate `self.warrant_catalog` for matching `ob_id`.
   - Fall back to `KNOWN_WARRANT_OB_IDS` via `lookup_known_warrant(ob_id)`.
3. Build position dict matching `_execute_buy` shape (lines 1326-1344):
   - `pos_id = f"pos_{int(time.time())}"`
   - `entry_price`, `entry_underlying=underlying_price`
   - `peak_underlying=underlying_price`, `trough_underlying=underlying_price`
   - `trailing_active=False`, `stop_order_id=None`
   - `fill_verified=True` (already live)
   - `buy_order_id=None`
   - `"ingested": True, "ingested_ts": _now_utc().isoformat()`
4. Do NOT decrement `cash_sek` — not charged this session.
5. `_save_state(self.state)`; append `_log_trade` with `action="INGEST"`.
6. If `set_stop_loss=True` and not DRY_RUN, call `self._set_stop_loss(pos_id)`.
7. Send Telegram: `SWING INGEST {name} {units}u @ {entry} — stop @ N SEK`.

### 2. Runtime migration from evaluate_and_execute

Add `self._orphans_migrated: bool = False` in `__init__`, and prepend to
`evaluate_and_execute(prices, signal_data)`:

```python
if not self._orphans_migrated:
    self._migrate_orphans(prices)
    self._orphans_migrated = True
```

New private method `_migrate_orphans(prices)`:

1. `held = fetch_page_positions(self.page, ACCOUNT_ID)`; return if None.
2. `existing = {str(p["ob_id"]) for p in self.state["positions"].values()}`.
3. Import `lookup_known_warrant` from `metals_loop`.
4. For each `(ob_id, info)` in held:
   - Skip if `ob_id in existing`.
   - Skip if `ob_id in FISHING_OB_IDS` (fish engine owns it).
   - `meta = lookup_known_warrant(ob_id)`; skip if None.
   - Skip if `meta.get("_managed_by")` is set and != `"swing_trader"` (legacy-owned).
   - `direction = _infer_direction(meta.get("name", ""))`.
   - `underlying = meta.get("underlying")`.
   - `und_price = self._get_ticker_underlying_price(underlying, prices)` or 0.
     - If 0, skip; next tick will retry.
   - `pos_id = self.ingest_position(ob_id, info["units"], info["avg_price"], und_price, direction)`.
5. Log summary.

`_infer_direction(name: str) -> str`: returns `"SHORT"` if `name` starts
with `BEAR`, contains `MINI S`, `TURBO S`, or `SHORT`; otherwise `"LONG"`.

### 3. Cross-module catalog helper

In `data/metals_loop.py` just below `KNOWN_WARRANT_OB_IDS` (after line 1577):

```python
def lookup_known_warrant(ob_id: str | int) -> dict | None:
    """Return KNOWN_WARRANT_OB_IDS metadata by ob_id, or None.

    Module-level helper for cross-module lookup from metals_swing_trader.
    """
    return KNOWN_WARRANT_OB_IDS.get(str(ob_id))
```

SwingTrader imports it lazily inside `_migrate_orphans` to avoid pulling in
Playwright at module import time.

### 4. Tag legacy hardcoded entries as swing-managed

Edit `data/metals_loop.py:1516-1527`. Add `"_managed_by": "swing_trader"` to
all five hardcoded entries (silver301, silver_sg, gold, bear_silver_x5,
bull_silver_x5). This routes future `detect_holdings()` auto-creations
through the swing path instead of legacy.

### 5. Mark migrated legacy entry inactive

After successful ingestion, deactivate the matching legacy entry so its
`active` flag no longer attracts any leftover legacy exit logic. Inside
`ingest_position` tail:

```python
from portfolio.file_utils import load_json, atomic_write_json
legacy_path = "data/metals_positions_state.json"
legacy = load_json(legacy_path, {})
changed = False
for k, v in legacy.items():
    if isinstance(v, dict) and str(v.get("ob_id")) == str(ob_id) and v.get("active"):
        v["active"] = False
        v["sold_reason"] = "migrated_to_swing"
        v["sold_ts"] = _now_utc().isoformat()
        changed = True
if changed:
    atomic_write_json(legacy_path, legacy)
```

### 6. Safety gate

Add `SWING_INGEST_ORPHANS = True` to `data/metals_swing_config.py`. Import
in swing_trader; short-circuit `_migrate_orphans` when False.

### 7. Adopt existing Avanza stop-loss (defensive)

Before `_set_stop_loss` places a new stop, check whether Avanza already has
an active stop-loss for this ob_id. If yes, adopt its id instead of placing
a duplicate. Looking up via `api_get("/_api/trading/rest/orders")` — the
`stopLosses` key if present, or filtering `orders` by `type=="STOP_LOSS"`.

Defer this to a helper `_find_existing_stop_loss(ob_id) -> str | None`
used by both `ingest_position` and (optionally) `_set_stop_loss` in future.

## Files changed

| File | Change |
|------|--------|
| `data/metals_swing_trader.py` | +~180 lines: `ingest_position`, `_migrate_orphans`, `_infer_direction`, `_find_existing_stop_loss`, hook in `evaluate_and_execute`, `_orphans_migrated` flag |
| `data/metals_loop.py` | +5 lines: `_managed_by` tags on 5 entries; +5 lines: `lookup_known_warrant` helper |
| `data/metals_swing_config.py` | +1 line: `SWING_INGEST_ORPHANS=True` |
| `tests/test_metals_swing_ingest.py` | NEW: unit tests |

## Risks

1. **Double stop-loss** — mitigated by `_find_existing_stop_loss` (§7).
2. **Wrong direction inference** — name-based heuristic works for current
   catalog. Log choice; operator can correct via state edit if wrong.
3. **Missing underlying price at first tick** — migration retries next tick.
4. **Unit drift** — trust Avanza (source of truth).
5. **Cash not decremented on ingestion** — intentional: we're adopting, not
   buying. Verified by Avanza sync.

## Tests

`tests/test_metals_swing_ingest.py` uses `tmp_path` + monkeypatching:

1. `test_ingest_creates_pos_with_all_required_fields`
2. `test_ingest_rejects_duplicate_ob_id`
3. `test_ingest_looks_up_catalog_by_ob_id_via_swing_catalog`
4. `test_ingest_falls_back_to_known_warrants`
5. `test_ingest_skips_unknown_ob_id_and_returns_none`
6. `test_ingest_calls_set_stop_loss_when_requested_not_dry_run`
7. `test_ingest_skips_set_stop_loss_on_dry_run`
8. `test_ingest_does_not_decrement_cash`
9. `test_ingest_writes_trade_record_with_action_INGEST`
10. `test_ingest_marks_legacy_state_inactive`
11. `test_migrate_orphans_skips_already_tracked_positions`
12. `test_migrate_orphans_ingests_bull_silver_x5_example`
13. `test_migrate_orphans_skips_fishing_ob_ids`
14. `test_migrate_orphans_skips_missing_underlying_price`
15. `test_migrate_orphans_only_runs_once_per_process`
16. `test_infer_direction_bear_warrant_is_short`
17. `test_infer_direction_default_long`

## Execution

1. Worktree `../fa-orphan-positions` on branch `fix/ingest-orphan-positions`.
2. Commit plan.
3. Write failing tests; commit.
4. Implement `ingest_position` + tests 1-10; commit.
5. Implement `_migrate_orphans` + `_infer_direction` + tests 11-17; commit.
6. Add `lookup_known_warrant` + `_managed_by` tags + config flag; commit.
7. Run full suite `-n auto`; commit any incidental fixes.
8. Codex adversarial review; address P1/P2; commit.
9. Merge into main, push via cmd.exe git, restart `PF-MetalsLoop`.
10. Verify bull_silver_x5 appears in `swing_state.positions` within 2 minutes
    and Avanza shows new stop-loss on ob_id 1650161.
11. Clean up worktree.

## Verification (post-ship)

```bash
tail -f data/metals_loop_out.txt | grep -iE "SWING INGEST|_migrate_orphans"

.venv/Scripts/python.exe -c "
from portfolio.file_utils import load_json
s = load_json('data/metals_swing_state.json', {})
for k,v in s.get('positions',{}).items():
    print(k, v.get('warrant_name'), v.get('units'), v.get('ob_id'), 'stop:', v.get('stop_order_id'), 'ingested:', v.get('ingested'))
"
```
