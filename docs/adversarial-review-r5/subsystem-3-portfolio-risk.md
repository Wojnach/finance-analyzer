# Subsystem 3: Portfolio Risk — Round 5 Findings

## THE BIG ANSWER: Risk management is theater.

Every risk function exists, is well-tested, and has zero production callers.

## CRITICAL (P1)

**PR-R5-1** — check_drawdown() STILL never called (confirmed R4 unfixed). risk_management.py:86.
**PR-R5-2** — record_trade() has no production callers. trade_guards.py:177.
**PR-R5-3** — Inconsistent min trade size: 1000 SEK (avanza_session), 500 (trade_validation, kelly_sizing, kelly_metals).
**PR-R5-4** — validate_trade() has zero production callers. trade_validation.py:22.
**PR-R5-5** — classify_trade_risk() has zero callers anywhere. trade_risk_classifier.py:29.
**PR-R5-6** — kelly_sizing.recommended_size() not called for Patient/Bold portfolios.

## HIGH (P2)

**PR-R5-7** — exposure_coach new_entries_allowed flag is advisory only, never enforced.
**PR-R5-8** — Monte Carlo p_stop_hit uses terminal price only (not path-dependent).
**PR-R5-9** — Backup rotation propagates corruption when source file is corrupt non-empty JSON.
**PR-R5-10** — ATR proximity check fires on HOLD tickers with action="CHECK" (misleading).
**PR-R5-11** — Kelly win rate uses average buy price, not FIFO — disagrees with equity_curve.

## MEDIUM (P3)

**PR-R5-13** — portfolio_value_history.jsonl unbounded growth; _streaming_max reads entire file.
