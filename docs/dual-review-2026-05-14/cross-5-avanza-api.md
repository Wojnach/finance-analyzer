# Cross-Critique — 5 avanza-api

## Agreement — high-confidence findings (both reviewers)

- **`portfolio/avanza/trading.py:80-92` — unified `place_order` bypasses whitelist + MAX_ORDER ceiling, P0.** Both flag the asymmetry between legacy `avanza_session.place_*_order` (whitelist + 50K SEK cap + leg warning + `avanza_order_lock`) and the new unified path (only the 1000 SEK fee-oriented check). Codex extends with explicit code blocks proving the gap and notes `__init__.py:29` makes the unified path canonical. **Independent rediscovery — very high confidence.** Action: mirror whitelist + MAX_ORDER guard + lock onto unified path.

- **`portfolio/avanza/account.py:64-94` — `get_buying_power` silent-zero on miss, P0.** Both flag the exact same code: zero-filled `AccountCash` returned when account not in overview. Codex extends with the explicit reference to the C7 fix in `avanza_session.py:385-539` (whose docstring promises None on failure) — the unified package regressed back to silent zeros. **Cross-validates strongly.** Action: raise on miss (or return None) so callers can distinguish "API failed" from "balance is legitimately zero".

- **`portfolio/avanza_session.py:88-95` — `expires_at` unparseable falls through "with caution" (P1).** Both flag. Codex extends with the dead `EXPIRY_BUFFER_MINUTES = 30` constant and explicit reference to the 3-week silent-auth pattern. **Both reviewers right.** Action: fail closed on parse error AND wire the buffer.

- **`portfolio/avanza/scanner.py` BankID path ignores `itype_str` filter (P1).** Both flag (Claude's "per the subagent finding"; Codex's exact line). Real silent miscategorization.

## Codex found, Claude missed

- **`portfolio/avanza_session.py:720-811` — `place_stop_loss` has NO MAX_ORDER guard (P0).** Stop-losses, when triggered, fire a full sell. A `volume * sell_price` of 250K SEK passes (the 1000-SEK warn-only is fee-oriented, not safety-oriented). Plus `place_trailing_stop` (`:814-846`) passes `trigger_price=trail_percent` with NO validation that `0 < trail_percent < 100` — a sign error from LLM-generated rules silently accepted. **Claude missed entirely — real P0.** Action: 50K SEK cap on stop legs, validation on trail_percent.

- **`portfolio/avanza/streaming.py:79-85` — channel string built from raw account IDs.** Codex frames as P1 trust-boundary concern with no whitelist check on subscribed account. Information disclosure (pension positions stream to local subscriber). Claude flagged streaming at P2 but with weaker framing. **Codex's framing is sharper.**

- **`portfolio/avanza_session.py:653-664` — `get_open_orders` returns empty list on double-fetch failure (P1).** Critical for grid fisher / fin_snipe / metals_loop "no in-flight orders → place new" precondition. API outage → empty list → duplicate buy-ladders. **Real silent-failure, Claude missed.** Compare to existing `get_stop_losses_strict` / `get_stop_losses` two-API pattern this same file already established.

- **`portfolio/avanza/tick_rules.py:124-126` — fallback to last tick entry hides API shape drift (P1).** Truncated tick ladder → wrong tick size silently used → "ogiltigt pris" rejections. **Narrow but real, Claude missed.**

- **`portfolio/avanza_orders.py:351-403` — `_execute_confirmed_order` doesn't re-check expiry before placing.** User confirm-after-expiry race. Minor UX bug; Claude didn't open this file.

- **`portfolio/avanza_session.py:1064-1070` — `cancel_all_stop_losses_for` busy-waits without rate limit, holds `_pw_lock` for up to 3s.** Pathological serialization on cancel storms across multi-loop deployment. Claude didn't flag the lock contention.

- **`portfolio/avanza/types.py:67-73` — `Quote.from_api` `0.0`-is-falsy gotcha on bid.** `spread = 0.0` silently when bid is 0. Caller `if spread > MAX_SPREAD: skip` doesn't skip. **Real P2.**

- **`portfolio/avanza_session.py:633-645` — `cancel_order` doesn't validate account_id against whitelist (P2).** Whitelist applies to every mutation principle broken. **Right call.**

- **`portfolio/avanza_session.py:1031-1032` — `import copy` inside hot function.** Cosmetic, but in a safety-critical locked section. Claude missed.

- **`portfolio/avanza/streaming.py:144-150` — WebSocket reconnect doesn't re-handshake clientId on stale (P2).** Minute-scale quote gaps. Real, narrow.

## Claude found, Codex missed

- **`portfolio/avanza/trading.py:65-78` — order total guard at 1000 SEK applies to LIMIT orders but `cancel_order` and `modify_order` don't carry it.** Codex covered cancel separately but not the modify path. Claude is right: a caller modifying a 1500 SEK leg to 200 SEK qty bypasses the placement check.

- **`portfolio/avanza/account.py:74-90` — `get_overview_raw()` called once per `get_buying_power` invocation, 6+ calls per cycle, no caching.** Claude flagged the rate-limit concern. Codex didn't.

- **`portfolio/avanza_session.py:759-762` — non-trailing MONETARY stop must have sell_price > 0 → ValueError when bid=0.** Claude flags the interaction with grid_fisher's P0-4 floor: failure mode is "exception" not "skip & retry". Codex didn't surface this.

- **`portfolio/avanza/account.py:139-140` — `account_id` filter is client-side AFTER fetching all transactions across both ISK and pension.** Claude's catch: pension history leaks into ISK-only caches. Codex didn't.

- **`portfolio/avanza_session.py:768-775` — sub-1000 SEK leg warning isn't structured for dashboard filtering.** Claude's P1 — operator can't separate intended sub-1000 from real bugs. Codex didn't.

- **`portfolio/avanza_session.py:752` — `valid_until` default of 8 days, Friday-placed stops expire Sunday (day 9).** Claude's catch: align to next-Tuesday rollover. Codex didn't.

## Disagreements

None substantive. Both reviewers concur on the major P0s (unified-path safety gap, silent-zero buying_power). Severity matches.

## What BOTH missed (third pass)

- **`portfolio/avanza/client.py` `AvanzaClient.get_instance()` singleton.** Neither reviewer audited what happens if config is reloaded mid-run. If `client.account_id` is cached on first construction and config changes, subsequent calls to `place_order(account_id=None)` use stale account.

- **`portfolio/avanza_session.py` `_with_browser_recovery` block.** Codex briefly mentions it; neither audited whether the recovery's storage-state path uses atomic write or a raw `open(...).write()` that could corrupt the auth cookie file mid-recovery.

- **`portfolio/avanza_orders.py` Telegram CONFIRM token flow.** Codex flagged the race; neither audited whether the token entropy is sufficient (HMAC vs random hex), and whether tokens are leaked in `data/telegram_messages.jsonl` (which is sometimes shared in screenshots).

- **`portfolio/avanza_account_check.py` (new 2026-05-11).** Claude flagged `DISALLOWED_CATEGORY_FRAGMENTS = []` at P2. Neither reviewer asked whether this file's check is actually invoked before any trade — if the integration is incomplete, the empty list is irrelevant because the gate is bypassed.

- **TOTP secret storage**: `portfolio/avanza_session.py` references `TOTP_SECRET` env. Neither reviewer audited where the secret is read from (config.json risk?), and whether session storage state in `data/avanza_session.json` (or wherever) has the right file permissions.

## Verdict

P0 list after cross: **3 confirmed** (unified place_order whitelist+cap gap, place_stop_loss MAX_ORDER + trail_percent validation, get_buying_power silent zero).
P1 list after cross: **~10 confirmed** (expires_at parse failure, scanner type filter, get_open_orders silent empty, tick_rules fallback, get_buying_power per-call rate, sell_price=0 ValueError, sub-1000 leg metric, valid_until weekend, account_id transaction leak, _pw_lock cancel storm).
P2 list after cross: ~9.

Avanza-api is the **broker-side blast-radius layer** — every P0 here writes real-money orders. The unified-package regression is the single most concerning finding (active migration silently drops safety nets).
