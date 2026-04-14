# Subsystem 4: Metals Core — Round 5 Findings

## CRITICAL (P1)

**MC-R5-1** — Stop-loss placed at entry-minus-5%, not current-bid-minus-3%. MIN_STOP_DISTANCE_PCT=1%.
`fin_snipe_manager.py:61,514-523`. Violates "never place stop within 3% of current bid" rule.
Fix: Change MIN_STOP_DISTANCE_PCT from 1.0 to 3.0, compute distance from current bid.

**MC-R5-2** — BEAR MINI knockout check silently bypassed with `pass`.
`fin_fish.py:732-735`. If spot >= barrier (warrant knocked out), code does nothing.
Fix: Replace `pass` with `continue`.

**MC-R5-3** — Limit sell + stop-loss both cover 100% position volume. Overfill risk.
`fin_snipe_manager.py:948-984`. If both fill, creates a short position on 5x leveraged cert.
Fix: Cap stop volume to position_volume minus open limit sell volume.

## HIGH (P2)

**MC-R5-4** — session_hours_remaining() hardcodes 21:55, ignoring DST and API todayClosingTime.
**MC-R5-5** — ORB predictor uses hardcoded winter UTC offsets — wrong by 1h in summer.
**MC-R5-6** — Stop proximity check silently skipped when current_bid=0.
**MC-R5-7** — fish_monitor_smart.py reads JSONL with raw read_text() — race with metals loop.

## MEDIUM (P3)

**MC-R5-8** — Loop passes static hours_remaining=6.0, never recalculates session time.
**MC-R5-9** — 820 min warrant constant — acceptable if remaining minutes input is correct.
**MC-R5-10** — LONG fishing negative barrier distance accidentally correct but semantically wrong.
