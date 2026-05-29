# avanza-api — FGL adversarial review (2026-05-29)

> Reconstructed by the orchestrator from the subagent's returned summary — the
> review subagent (caveman:cavecrew-reviewer) completed its analysis but failed to
> write this file itself (Write-tool error mid-run). Findings are verbatim from its
> result. Line numbers should be re-confirmed before acting.

**Counts:** P0: 2 · P1: 8 · P2: 0

## P0
- `portfolio/avanza_orders.py:379`: SELL/confirm path saves `result.get("orderId", "?")`
  as `avanza_order_id` when orderId is missing (`:384` persists it). Creates an
  unreferenceable order — later cancel/track searches for `order_id="?"` at the broker
  and fails. → Reject orders with missing orderId; never accept `"?"` placeholder.
- `portfolio/avanza_session.py:542-561,590`: `place_buy_order()` does not validate
  `orderbook_id` is non-empty/numeric before POSTing; empty `orderbook_id=""` is sent
  as-is. Account whitelist guard is correct (rejects non-ISK) but runs after a silent
  str-coercion. → Validate orderbook_id non-empty + numeric before send.

## P1
- `avanza_session.py:180-192`: `verify_session()` returns `resp.ok`; a server-side
  revoked session that returns 200 with an empty/error body reads as healthy. → check
  body content, not just status.
- `avanza_session.py:258-265`: `api_get()` raises on `not resp.ok` but not on an empty
  200 body; callers parsing JSON crash or process `{}` silently. → reject empty body.
- `avanza_session.py:285-342`: `api_post()` returns `{"raw": body}` on JSON-parse
  failure without an error flag; a 200 with `orderRequestStatus="REJECTION"` is treated
  as success. → surface status field + parse-failure explicitly.
- `avanza_orders.py:367-376` + `avanza_control.py:355-359`: caller does not check
  `result is None` before `.get()` → AttributeError if place_*_order returns None. →
  validate not-None before attribute access.
- `avanza_order_lock.py:85-101`: `finally` calls `lock.release()` unconditionally even
  when `acquire()` raised (timeout/busy) — release-without-hold is UB on some platforms.
  → guard release on an acquired flag.
- `avanza_orders.py:204-207,332`: confirm-token match is case-sensitive lowercase-hex;
  `CONFIRM abC123` is silently dropped by `_HEX_TOKEN_RE`. → case-insensitive or document.
- `avanza_orders.py:378-380`: response fields (`orderRequestStatus`/`orderId`/`message`)
  not validated for presence; a 200 `{}` yields status="UNKNOWN", order limbo. → require
  all three; reject as malformed otherwise.
- `avanza_session.py:285-340`: 200-with-error-JSON not caught (duplicate angle of the
  api_post finding) — parse JSON then check status field before returning success.

## Cross-critique note (orchestrator)
The session-verify / api_get / api_post "200-but-broken" cluster is the avanza-side
instance of the system-wide **silent-failure-on-success-status** theme (same class as
the orchestration "exit 0 but no journal" and data-external "stale cache served as
fresh"). Worth a single shared hardening: a `_require_valid_response()` helper that
asserts non-empty body + expected status field, used by every Avanza call site.
The session has in fact been expired since 2026-05-23 (`avanza_account_mismatch` in
critical_errors.jsonl) — finding #1 (verify_session false-OK) is consistent with a
live, currently-unremediated outage.
