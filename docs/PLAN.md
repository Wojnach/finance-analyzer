# Fix: Fish Engine Position Reconciliation + Trade Logging

## Date: 2026-04-07

## Problem

The fish engine tracks its position internally (`_fish_engine.position`) but never
verifies against actual Avanza holdings. When a stop-loss triggers on Avanza's side,
the engine doesn't know — its state file goes stale, and it either:
- Tries to sell a position that doesn't exist
- Refuses to buy because it thinks it already has a position
- Reports incorrect P&L

The swing trader's `POSITIONS` dict has reconciliation via `detect_holdings()` (every 30s)
and `_verify_position_holdings()` (at startup), but the fish engine is excluded from both.

Additionally, trade logging is incomplete: SELL entries log price_sek=0, no per-trade P&L,
no underlying price, no exit reason.

## Root Cause

Two parallel position tracking systems with no bridge:
1. `POSITIONS` dict (swing trader) — reconciled via `detect_holdings()` → Avanza API
2. `_fish_engine.position` — only updated when engine itself buys/sells

## What Could Break

- If reconciliation is too aggressive, it could clear a valid position during a brief
  API glitch → fix: only clear after confirmed "not held" from successful API call
- If confirm_exit is called with incorrect P&L → fix: estimate from entry cert price
- Concurrent access → not an issue, metals loop is single-threaded

## Fix Plan

### Batch 1: fish_engine.py changes
1. Add `force_close_position(reason, exit_cert_price=0)` method for external closes
2. Enrich `_log_trade` with: direction, entry_underlying, exit_underlying, exit_reason, pnl_sek
3. Pass exit_cert_price properly in confirm_exit

### Batch 2: metals_loop.py changes
1. In `detect_holdings()`: after POSITIONS reconciliation, check fish engine position
2. In `_verify_position_holdings()`: also verify fish engine position at startup
3. Persist engine state after reconciliation
4. Send Telegram alert on external close

### Batch 3: Tests
1. Test force_close_position clears state + logs
2. Test detect_holdings reconciles fish engine
3. Test startup verification reconciles fish engine
4. Test enriched trade logging fields

## Execution Order

1. Tests first (batch 3) — write failing tests
2. fish_engine.py (batch 1) — add force_close + enrich logging
3. metals_loop.py (batch 2) — hook into reconciliation
4. Run tests, fix failures
5. Codex review
6. Merge + push
