# Agent Review — Avanza API (2026-05-26)

Reviewer: caveman:cavecrew-reviewer
Tools: Read, Grep, Bash (no Write — summary returned inline; main thread saved to file)

## Findings

portfolio/avanza_session.py:701: P1: `get_positions()` returns pension account. Missing `ALLOWED_ACCOUNT_IDS` filter causes callers to receive account 2674244 mixed with trading positions. [REPEAT]

portfolio/avanza_session.py:638: P1: `cancel_order()` has no account whitelist. All other trading functions guard; this one mutates without checking `ALLOWED_ACCOUNT_IDS`. Add guard before line 638. [REPEAT]

portfolio/avanza_client.py:358: P2: `_place_order` TOTP path skips 1000 SEK minimum. Session path enforces (avanza_session:595); TOTP skips. Orders 500–999 SEK bypass guard and incur high fees. [REPEAT]

portfolio/avanza_session.py:147: P2: `_get_playwright_context()` resource leak on exception. If `new_context()` throws, `_pw_instance` / `_pw_browser` assigned but not cleaned. Wrap 147–152 in try/finally. [REPEAT]

portfolio/avanza_session.py:89: P2: Session expiry uses `<=` instead of `<`. Rejects tokens 1 second early. Change to `if exp < now:`. [REPEAT]

## Top 3

1. Account-whitelist drift across `get_positions()` / `cancel_order()` — leak vectors for "only trade ISK 1625505" rule. Centralize whitelist into a decorator or shared guard rather than per-function inclusion.
2. TOTP vs session-path divergence on minimum order size — two-path code collecting bugs. Collapse to a single `_validate_order` checked before either path.
3. All 5 findings are **REPEATS from May 24 review** — none addressed. Avanza-api subsystem is currently the most stale relative to its review burden. Backlog item.

## totals: 5 bugs (0 new, 5 repeat from May 24 review)
