# Avanza API Subsystem Review — 2026-05-21

**Baseline:** 604f0ef1 · **Scope:** 8 files, 2.5K LOC · **Severity filters:** P1=breaking, P2=silent/race, P3=micro

---

## Critical Findings (P1 — Real-Money Loss Risk)

### Session Auth

**avanza_client.py:68** — Singleton cache never invalidates. _session_client = True is set once, never checked again. If session expires mid-loop, the flag stays true and every subsequent call assumes valid auth. Add re-verification on 401 responses or expire cache after 1h.

**avanza_session.py:89** — Off-by-one session expiry. Check is `if exp <= now:` but should be `<`. Token valid through 23:59:59 triggers reauth one second early. Change to `if exp < now:`.

**avanza_account_check.py:221-248** — Cache bypass on expiry. verify_default_account() with use_cache=True returns cached "ok: True" forever. If BankID session expired after cache populated, orders placed on dead session. Add timestamp to cache, re-verify if >1h.

**avanza_session.py:620** — Lock busy swallowed silently. place_buy_order() wraps POST in avanza_order_lock() but never catches OrderLockBusyError. If lock times out, exception propagates and order is NOT placed. Catch separately, log "busy, retry next cycle".

**avanza_session.py:749-752** — Untrusted account re-arming. rearm_stop_losses_from_snapshot() uses account_id from snapshot without re-validating against ALLOWED_ACCOUNT_IDS. Snapshot could route stops to pension account. Re-validate account_id before placing.

---

## High-Risk Findings (P2 — Silent Failures / Races)

### Exception Handling & Propagation

**avanza_session.py:258-266** — Playwright resource leak on retry failure. api_get() catches exceptions in _op. On 401, calls close_playwright(). But if retry throws, exception propagates without close. Wrap L220-232 in try/finally ensuring close_playwright() on any retry exception.

**avanza_session.py:314-324** — CSRF token staleness under concurrency. api_post() L315 calls _get_csrf(ctx) AFTER re-entering _pw_lock. But _get_csrf() reads ctx.cookies() which is NOT stable under lock — concurrent thread's close_playwright() can invalidate _pw_context. Use _get_csrf(ctx) passthrough to ensure CSRF and POST use same fresh context.

**avanza_session.py:385-539** — Silent failure cascade in get_buying_power(). Function swallows all exceptions and returns None. But callers like place_stop_loss() never check if the return is None before sizing. Worst-case: stop-loss placed with zero buying_power. Require callers to check for None before guard conditions.

**avanza_session.py:649-664** — Silent endpoint fallback on error. get_open_orders() tries two endpoints; on failure, returns empty list same as "no orders". Caller cannot distinguish API down from flat account. Only fallback on 404, propagate network errors.

**avanza_session.py:676-717** — Missing type guard on get_positions(). api_get() returns parsed JSON but no validation it's a dict. If API returns array or null, data.get("withOrderbook") crashes. Add type guard at start.

**avanza_orders.py:220-288** — Telegram offset advance skips auth. _check_telegram_confirm() increments offset BEFORE checking sender auth. If allowed_user_id set and sender unauthorized, offset advances anyway. Move offset increment AFTER auth pass.

**avanza_orders.py:260-268** — Offset file corruption on exception. Fallback from load_json to read_text() doesn't validate JSON was broken. If JSON valid but load_json returns None, fallback reads and parses as int. Change: check isinstance(offset_data, dict) explicitly, only fallback if None.

**avanza_control.py:174-193** — Lock not visually released on exception. delete_order_live() acquires lock then calls page.evaluate(). If evaluate throws, context manager still releases lock (safe) but code structure unclear. Add comment or restructure for clarity.

**avanza_resilient_page.py:158-174** — Lost traceback on relaunch failure. evaluate() catches exception, checks is_browser_dead_error. If relaunch fails, original exception context lost. Add `raise ... from exc` to preserve chain.

---

## Moderate Findings (P2 — Design Issues)

### Data Validation

**avanza_session.py:759** — Missing default for value_type. Parameter can be None, but check at L759 uses value_type == "MONETARY" which is False if None. Non-trailing stops with unspecified value_type incorrectly pass sell_price<=0 check. Add default at function signature.

**avanza_account_check.py:285-306** — Substring match fragility. _category_disallowed() does fragment in norm.lower() without word boundary. If Avanza adds "RiskSparingISK", substring match fires incorrectly. Use regex or exact match list.

**avanza_session.py:1074-1084** — Timeout semantics unclear. Verification poll exits at max_wait with stale remaining list. "verified_cancelled" filter removes IDs that still exist, but if timeout fired, "still exist" may be incomplete. Snapshot rollback fails. Document: remaining list is stale on timeout.

### Caching & State

**avanza_client.py:242-279** — Account ID cache never expires. Global _account_id cached forever. If user swaps SIM/BankID tokens, stale ID persists. Add 24h TTL and explicit reset on 401.

**avanza_orders.py:204-217** — Partial commit + save failure race. check_pending_orders() executes order (may POST), then saves pending. If Avanza accepts but save fails, disk out-of-sync. Wrap execute in try/except, ensure state saved regardless.

### Signal Clarity

**avanza_session.py:1142-1147** — Function docstring outdated. Docstring says "status (SUCCESS/FAILED)" but code returns dict with status/rearmed/failed. Fix docstring.

**avanza_session.py:192** — Incomplete HTTP status check. Returns resp.ok without validating status. 204 No Content is OK but resp.json() on empty body crashes. Check resp.status in (200, 204) explicitly.

---

## Patterns

1. **Session state cached without invalidation**: _session_client, _account_id, verify_default_account cache all live forever. Missing re-verification on 401/403.
2. **Exception silencing on fallback paths**: get_open_orders, cancel_order, get_positions return empty/false on any error. Callers can't retry appropriately.
3. **Off-by-one date logic**: Session expiry uses <= instead of <.
4. **Concurrent access under partial lock**: CSRF fetch inside re-entered RLock, but CSRF validity not stable across POST if context rotated.
5. **Untrusted snapshot operations**: Snapshots re-armed without re-validation against whitelists.

---

**Summary:** 5 P1 (cache invalidation, off-by-one expiry, lock busy, untrusted snapshot), 10 P2 (exception handling, races, design gaps). Session state management is highest-risk area.

