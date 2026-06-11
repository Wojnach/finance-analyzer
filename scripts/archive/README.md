# scripts/archive/ — retired one-off operational scripts

> ⚠️ **DANGER: these scripts contain hardcoded LIVE account and orderbook IDs
> and place/cancel/modify REAL Avanza orders when executed. They are kept for
> reference only. Do NOT run them.** They were moved here from `data/` in the
> 2026-06-10 audit (batch 12) because runnable order-placing scripts sitting in
> the runtime-state directory are discoverable footguns — autonomous agents grep
> `data/` for existing functionality before writing code (a `CLAUDE.md` rule).

Each is a point-in-time debug/fix session artifact. None is imported by any
module (verified by grep at archive time). If you need the behavior, read the
script for reference and re-implement against the current `avanza_session.py` /
`avanza_orders.py` API rather than running the archived copy — the hardcoded IDs
and warrant names are stale.

| Script | What it did | Hardcoded |
|--------|-------------|-----------|
| `gold_sell_debug.py` | Debug a gold-warrant sell: verify position, check IDs, retry order | account `1625505`, orderbook `2308943` |
| `gold_sell_final.py` | Delete existing orders then sell `BULL GULD X20 AVA 6` | account, orderbook, warrant name |
| `gold_sell_retry.py` | Delete a failed order and retry the gold sell | account, orderbook, warrant name |
| `layer2_invoke.py` | One-shot that appends a hardcoded Feb-2026 Layer 2 journal entry (references META/SMCI/MU/NVDA — instruments removed from the system in Mar 2026) | journal payload |
| `place_stoploss_once.py` | One-time cascading stop-loss placement for silver/gold warrants via the correct `/_api/trading/stoploss/new` API | account, warrant IDs |

**Convention going forward:** one-off operational scripts should live in an
untracked `scratch/` directory, or be deleted in the same session that used
them. Git history is the archive — see the commits that introduced/fixed these
(`4adeec2d`, etc.) if you need the exact prior state.
