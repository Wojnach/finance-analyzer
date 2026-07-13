"""Avanza session management — load, validate, and use BankID-captured sessions.

Uses Playwright's saved storage state to make authenticated API calls via a
headless browser context. This ensures cookies and TLS session match what
Avanza expects (replaying cookies via requests library causes 401s).

This is the preferred auth method until TOTP credentials are configured.
"""

import concurrent.futures
import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from portfolio.avanza_order_lock import avanza_order_lock
from portfolio.avanza_resilient_page import is_browser_dead_error
from portfolio.file_utils import load_json

logger = logging.getLogger("portfolio.avanza_session")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSION_FILE = DATA_DIR / "avanza_session.json"
STORAGE_STATE_FILE = DATA_DIR / "avanza_storage_state.json"
API_BASE = "https://www.avanza.se"

# Minimum remaining session life before we consider it expired (minutes)
EXPIRY_BUFFER_MINUTES = 30

# Default trading account
DEFAULT_ACCOUNT_ID = "1625505"

# Whitelist of permitted account IDs — never trade outside these.
# Derived from DEFAULT_ACCOUNT_ID so a single config change updates both
# the default routing target and the H7 order-placement guard. If we
# ever need to permit additional accounts (multi-strategy, sub-accounts),
# add them here explicitly — but keep DEFAULT_ACCOUNT_ID as the single
# source of truth for "where do orders go by default."
ALLOWED_ACCOUNT_IDS = {DEFAULT_ACCOUNT_ID}

# Module-level Playwright context (lazy-initialized, reused across calls)
# BUG-129: Protected by _pw_lock to prevent concurrent access corruption
# A-AV-1 (2026-04-11): Upgraded to RLock so api_get/api_post/api_delete can
# wrap their *entire* request flow under the lock — they call
# _get_playwright_context() (which itself acquires the lock) inside the
# critical section. The previous Lock would deadlock; RLock is reentrant
# for the same thread. Without this, Playwright's sync_api was being used
# concurrently from main loop's 8-worker pool + metals 10s fast-tick,
# corrupting trade responses (e.g. CONFIRM stolen by wrong request).
_pw_lock = threading.RLock()
_pw_instance = None
_pw_browser = None
_pw_context = None


class AvanzaSessionError(Exception):
    """Raised when session is missing, expired, or invalid."""


class AvanzaSessionTimeout(AvanzaSessionError):
    """Raised when a session call exceeded its per-call timeout."""


# ---------------------------------------------------------------------------
# Single-worker session executor — Playwright thread pinning
# ---------------------------------------------------------------------------
# 2026-06-12 (audit B4 fix 1): sync Playwright objects are bound to the
# thread that created them; calls from any other thread raise greenlet
# "cannot switch to a different thread (which happens to have exited)".
# Three competing initializers used to race for the singleton context
# (avanza_account_check's transient pool, GridFisher's 'grid-fisher-session'
# worker, metals_loop main-thread stop paths) — whichever won bound the
# context to its thread and broke everyone else. Empirically: 16,719
# session_call_error entries in grid_fisher_decisions.jsonl 2026-05-11 →
# 2026-06-09, the grid market-maker completely blind on 9 full days.
#
# Fix: ALL Playwright traffic is marshalled onto ONE long-lived worker
# thread owned by this module (`_run_on_session_thread`). Callers on any
# thread (including asyncio loops — sync Playwright inside a running loop
# raises otherwise) submit and wait. Premortem hooks (binding):
#   (a) queue-wait + per-call duration recorded (see session_call_stats());
#   (b) queue wait > _QUEUE_WAIT_CRITICAL_S → critical_errors.jsonl entry;
#   (c) per-call timeout so one hung call can't silently block callers
#       forever (the worker itself may stay busy — the queue-stall
#       escalation in (b) is what surfaces that);
#   (d) re-entrant guard — a submit from the worker thread itself raises
#       immediately (1-worker pool would deadlock).
# N consecutive failures escalate to critical_errors + Telegram so the
# subsystem can never again be silently dead for a month.

_SESSION_CALL_TIMEOUT_S = 90.0
_QUEUE_WAIT_WARN_S = 5.0
_QUEUE_WAIT_CRITICAL_S = 30.0
_CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 10
_ESCALATION_COOLDOWN_S = 600.0  # min seconds between repeat escalations

_executor_lock = threading.Lock()
_session_executor: concurrent.futures.ThreadPoolExecutor | None = None
_session_thread_id: int | None = None

_stats_lock = threading.Lock()
_last_call_stats: dict[str, Any] = {}
_consecutive_failures = 0
_last_queue_stall_escalation_mono = 0.0
_last_failure_escalation_mono = 0.0
_last_mutation_timeout_escalation_mono = 0.0


def _record_mutation_timeout(verb: str, path: str) -> None:
    """2026-06-12 (review fix 18d9d0cc #2): a timed-out MUTATING call
    (POST/DELETE) may STILL succeed at the broker — the worker thread
    cannot be interrupted mid-request, so the order/cancel can land after
    we stopped waiting. Local state then doesn't know about a live broker
    order until the next reconcile pass picks reality up. This entry makes
    that residual risk visible in critical_errors.jsonl instead of silent.
    Critical entries are cooled down (warning log always fires)."""
    global _last_mutation_timeout_escalation_mono
    logger.warning(
        "avanza_session: %s %s timed out — the mutation may still have "
        "been applied at the broker; reconcile next tick",
        verb,
        path,
    )
    now_mono = time.monotonic()
    with _stats_lock:
        if now_mono - _last_mutation_timeout_escalation_mono <= _ESCALATION_COOLDOWN_S:
            return
        _last_mutation_timeout_escalation_mono = now_mono
    _record_critical(
        "avanza_mutation_timeout",
        f"{verb} {path} timed out after {_SESSION_CALL_TIMEOUT_S:.0f}s — "
        "the order/cancel may still exist at the broker; verify via "
        "reconcile / open-orders before re-issuing",
        {"verb": verb, "path": path},
    )


def _ensure_session_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Create (once) and return the module's single session worker."""
    global _session_executor, _session_thread_id
    with _executor_lock:
        if _session_executor is None:
            _session_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="avanza-session",
            )
            # Resolve the worker's thread id eagerly so the re-entrant
            # guard works from the very first real call.
            _session_thread_id = _session_executor.submit(threading.get_ident).result(
                timeout=10
            )
        return _session_executor


def _on_session_thread() -> bool:
    return (
        _session_thread_id is not None and threading.get_ident() == _session_thread_id
    )


def session_call_stats() -> dict[str, Any]:
    """Snapshot of the most recent session call's queue/duration metrics.

    Keys: ``op_name``, ``queue_wait_s``, ``duration_s``, ``ok``, ``ts``,
    ``consecutive_failures``. Exposed for tests + ops probes (hook (a)).
    """
    with _stats_lock:
        snap = dict(_last_call_stats)
        snap["consecutive_failures"] = _consecutive_failures
        return snap


def _record_critical(category: str, message: str, context: dict | None) -> None:
    """Best-effort critical_errors.jsonl append. Never raises."""
    try:
        from portfolio.claude_gate import record_critical_error  # noqa: PLC0415

        record_critical_error(category, "portfolio.avanza_session", message, context)
    except Exception:  # noqa: BLE001
        logger.debug("avanza_session: critical_errors append failed", exc_info=True)


def _alert_telegram(message: str) -> None:
    """Best-effort Telegram alert (pattern from avanza_account_check)."""
    try:
        from portfolio.telegram_notifications import send_telegram  # noqa: PLC0415

        cfg = load_json("config.json") or {}
        if cfg.get("telegram", {}).get("token"):
            send_telegram(message, cfg)
    except Exception:  # noqa: BLE001
        logger.debug("avanza_session: telegram alert skipped", exc_info=True)


def _note_call(
    op_name: str, queue_wait_s: float, duration_s: float | None, ok: bool
) -> None:
    """Record per-call metrics and run the escalation hooks (a)/(b)."""
    global _consecutive_failures
    global _last_queue_stall_escalation_mono, _last_failure_escalation_mono
    now_mono = time.monotonic()
    with _stats_lock:
        _last_call_stats.clear()
        _last_call_stats.update(
            {
                "op_name": op_name,
                "queue_wait_s": round(queue_wait_s, 3),
                "duration_s": round(duration_s, 3) if duration_s is not None else None,
                "ok": ok,
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        if ok:
            _consecutive_failures = 0
        else:
            _consecutive_failures += 1
        failures = _consecutive_failures
        queue_stall = (
            queue_wait_s > _QUEUE_WAIT_CRITICAL_S
            and now_mono - _last_queue_stall_escalation_mono > _ESCALATION_COOLDOWN_S
        )
        if queue_stall:
            _last_queue_stall_escalation_mono = now_mono
        failure_escalate = (
            failures >= _CONSECUTIVE_FAILURE_ALERT_THRESHOLD
            and now_mono - _last_failure_escalation_mono > _ESCALATION_COOLDOWN_S
        )
        if failure_escalate:
            _last_failure_escalation_mono = now_mono

    if queue_wait_s > _QUEUE_WAIT_WARN_S:
        logger.warning(
            "avanza_session: %s queued %.1fs behind the session worker "
            "(duration=%s ok=%s)",
            op_name,
            queue_wait_s,
            duration_s,
            ok,
        )
    else:
        logger.debug(
            "avanza_session: %s queue_wait=%.3fs duration=%s ok=%s",
            op_name,
            queue_wait_s,
            duration_s,
            ok,
        )
    if queue_stall:
        _record_critical(
            "avanza_session_queue_stall",
            f"Session call {op_name} waited {queue_wait_s:.1f}s for the "
            "avanza-session worker — a previous call is likely hung",
            {"op_name": op_name, "queue_wait_s": round(queue_wait_s, 1)},
        )
    if failure_escalate:
        msg = (
            f"avanza_session: {failures} consecutive session-call failures "
            f"(latest: {op_name}). Avanza order/stop management is "
            "effectively DOWN — investigate session auth / browser health."
        )
        _record_critical(
            "avanza_session_consecutive_failures",
            msg,
            {"failures": failures, "op_name": op_name},
        )
        _alert_telegram(f"*AVANZA SESSION DEGRADED*\n{msg}")


def _run_on_session_thread(
    fn: Callable[[], Any], *, op_name: str, timeout: float | None = None
) -> Any:
    """Execute ``fn`` on the dedicated session worker thread.

    Raises:
        RuntimeError: if called FROM the worker thread (hook (d) — a
            1-worker pool would deadlock waiting on itself).
        AvanzaSessionTimeout: if the call exceeds the per-call timeout
            (hook (c)). NOTE: the worker may still be busy executing the
            hung call — subsequent callers will see queue-wait growth and
            hook (b) escalates that to critical_errors.jsonl.
    """
    executor = _ensure_session_executor()
    if _on_session_thread():
        raise RuntimeError(
            f"avanza_session: re-entrant session call {op_name!r} from the "
            "session worker thread — would deadlock the 1-worker executor. "
            "Call the underlying Playwright helpers directly instead."
        )
    effective_timeout = _SESSION_CALL_TIMEOUT_S if timeout is None else timeout
    submitted_mono = time.monotonic()
    started: dict[str, float] = {}

    def _instrumented():
        started["t"] = time.monotonic()
        return fn()

    future = executor.submit(_instrumented)
    ok = False
    try:
        result = future.result(timeout=effective_timeout)
        ok = True
        return result
    except concurrent.futures.TimeoutError:
        future.cancel()  # only helps if still queued; a running call hangs the worker
        raise AvanzaSessionTimeout(
            f"avanza_session call {op_name!r} timed out after "
            f"{effective_timeout:.0f}s"
        ) from None
    finally:
        ended_mono = time.monotonic()
        start_t = started.get("t")
        queue_wait = (
            (start_t - submitted_mono)
            if start_t is not None
            else (ended_mono - submitted_mono)
        )
        duration = (ended_mono - start_t) if start_t is not None else None
        _note_call(op_name, queue_wait, duration, ok)


def load_session() -> dict:
    """Load saved BankID session metadata from disk.

    Returns:
        Session dict with expiry info, customer_id, etc.

    Raises:
        AvanzaSessionError: if file missing, unreadable, or expired.
    """
    if not SESSION_FILE.exists():
        raise AvanzaSessionError(
            f"No session file found at {SESSION_FILE}. "
            "Run: python scripts/avanza_login.py"
        )

    data = load_json(SESSION_FILE)
    if data is None:
        raise AvanzaSessionError(f"Failed to read session file: {SESSION_FILE}")

    # Check expiry
    expires_at = data.get("expires_at")
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at)
            now = datetime.now(UTC)
            if exp <= now:
                raise AvanzaSessionError(
                    f"Session expired at {expires_at}. "
                    "Run: python scripts/avanza_login.py"
                )
        except ValueError:
            logger.warning(
                "Cannot parse expires_at %r — cannot verify expiry, proceeding with caution",
                expires_at,
            )

    if not STORAGE_STATE_FILE.exists():
        raise AvanzaSessionError(
            f"No storage state file at {STORAGE_STATE_FILE}. "
            "Run: python scripts/avanza_login.py"
        )

    return data


def session_remaining_minutes() -> float | None:
    """Get minutes remaining on the current session, or None if no session."""
    try:
        data = load_json(SESSION_FILE)
        if data is None:
            return None
        expires_at = data.get("expires_at")
        if not expires_at:
            return None
        exp = datetime.fromisoformat(expires_at)
        now = datetime.now(UTC)
        return (exp - now).total_seconds() / 60.0
    except Exception as e:
        logger.warning("Failed to compute session minutes remaining: %s", e)
        return None


def is_session_expiring_soon(threshold_minutes: float = 60.0) -> bool:
    """Check if session will expire within the given threshold.

    Returns True if session is expired, expiring soon, or doesn't exist.
    """
    remaining = session_remaining_minutes()
    if remaining is None:
        return True
    return remaining < threshold_minutes


def _get_playwright_context():
    """Get or create a headless Playwright browser context with saved auth state."""
    global _pw_instance, _pw_browser, _pw_context

    with _pw_lock:
        if _pw_context is not None:
            return _pw_context

        # Validate session first
        load_session()

        from playwright.sync_api import sync_playwright

        _pw_instance = sync_playwright().start()
        _pw_browser = _pw_instance.chromium.launch(headless=True)
        _pw_context = _pw_browser.new_context(
            storage_state=str(STORAGE_STATE_FILE),
            locale="sv-SE",
        )
        return _pw_context


def _close_playwright_inline():
    """Tear down Playwright resources on the CURRENT thread.

    Only safe when the current thread owns the context (i.e. the session
    worker). External callers must use :func:`close_playwright`.
    """
    global _pw_instance, _pw_browser, _pw_context
    with _pw_lock:
        if _pw_context:
            try:
                _pw_context.close()
            except Exception as e:
                logger.debug("Context close failed: %s", e)
            _pw_context = None
        if _pw_browser:
            try:
                _pw_browser.close()
            except Exception as e:
                logger.debug("Browser close failed: %s", e)
            _pw_browser = None
        if _pw_instance:
            try:
                _pw_instance.stop()
            except Exception as e:
                logger.debug("Playwright stop failed: %s", e)
            _pw_instance = None


def close_playwright():
    """Clean up Playwright resources.

    2026-06-12 (audit B4 fix 1): teardown is marshalled to the session
    worker thread when one exists — closing a sync-Playwright context from
    a foreign thread raises the same greenlet error this module now
    prevents. Runs inline when already on the worker (the 401 recovery
    path inside api_get/api_post calls this from the worker itself) or
    when no worker was ever started (nothing thread-bound exists yet).
    """
    if _session_thread_id is None or _on_session_thread():
        return _close_playwright_inline()
    try:
        return _run_on_session_thread(
            _close_playwright_inline,
            op_name="close_playwright",
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort teardown
        logger.warning(
            "avanza_session: marshalled close_playwright failed (%s) — "
            "clearing refs inline",
            exc,
        )
        return _close_playwright_inline()


def verify_session() -> bool:
    """Verify that the session is valid by making a lightweight API call.

    Returns:
        True if session is valid, False otherwise.
    """

    # A-AV-1: Hold _pw_lock for the entire context+request flow.
    # ctx.request.* is NOT thread-safe; concurrent callers must serialize.
    # 2026-06-12 (audit B4 fix 1): runs on the session worker thread so the
    # context is created/used with the same thread affinity as api_*.
    def _verify():
        with _pw_lock:
            ctx = _get_playwright_context()
            resp = ctx.request.get(f"{API_BASE}/_api/position-data/positions")
            return resp.ok

    try:
        return _run_on_session_thread(_verify, op_name="verify_session")
    except Exception as e:
        logger.warning("Session verification failed: %s", e)
        close_playwright()
        return False


# 2026-04-13: Auto-recovery wrapper for api_get/api_post/api_delete.
# The singleton Playwright browser held in _pw_context occasionally dies
# mid-flight (OS sleep, memory pressure, external BankID re-auth by the
# user, cookie-jar corruption under heavy concurrency). When that happens
# every subsequent ctx.request.* call throws TargetClosedError until the
# process restarts. The pre-existing 401/403 path already knows to call
# close_playwright() so the next request re-launches; we extend the same
# pattern to browser-dead errors.
#
# Keeps the singleton + _pw_lock (BUG-129 / A-AV-1). The whole retry runs
# under the RLock so a concurrent thread cannot partially observe the
# teardown/relaunch. _get_playwright_context also acquires the lock but
# it's reentrant for the same thread.
def _with_browser_recovery(op: Callable[[Any], Any], *, op_name: str) -> Any:
    """Run ``op(ctx)`` on the session worker thread under ``_pw_lock``; on
    browser-dead error, teardown + relaunch + retry once. Propagate all
    other exceptions unchanged.

    ``op`` is called with the current Playwright context. The op is responsible
    for making the actual ctx.request.* call and handling HTTP-level errors.

    2026-06-12 (audit B4 fix 1): the entire locked flow is marshalled to the
    module's single session worker thread (see ``_run_on_session_thread``),
    so the Playwright context is always created AND used on the same
    long-lived thread. The teardown/relaunch inside also happens on that
    thread (``close_playwright`` runs inline when already on the worker).
    """

    def _locked_op():
        with _pw_lock:
            ctx = _get_playwright_context()
            try:
                return op(ctx)
            except Exception as exc:
                if not is_browser_dead_error(exc):
                    raise
                logger.warning(
                    "avanza_session: browser dead on %s (%r) — teardown + relaunch + retry",
                    op_name,
                    exc,
                )
                close_playwright()
                ctx = _get_playwright_context()
                return op(ctx)

    return _run_on_session_thread(_locked_op, op_name=op_name)


# --- API convenience functions ---


def api_get(path: str, **kwargs) -> Any:
    """Make an authenticated GET request to Avanza API.

    Args:
        path: API path (e.g., "/_api/position-data/positions")

    Returns:
        Parsed JSON response.

    Raises:
        AvanzaSessionError: if session is invalid.
    """
    # A-AV-1: Hold _pw_lock for the entire request. Playwright's sync_api
    # is NOT thread-safe and the metals fast-tick + main 8-worker pool race.
    # 2026-04-13: Wrapped in _with_browser_recovery so TargetClosedError
    # (browser died mid-flight) triggers a teardown + relaunch + retry.
    url = f"{API_BASE}{path}" if path.startswith("/") else path

    def _op(ctx):
        resp = ctx.request.get(url)
        if resp.status == 401:
            close_playwright()
            raise AvanzaSessionError(
                "Session returned 401 Unauthorized. "
                "Run: python scripts/avanza_login.py"
            )
        if not resp.ok:
            raise RuntimeError(f"Avanza API error {resp.status}: {resp.text()[:500]}")
        return resp.json()

    return _with_browser_recovery(_op, op_name=f"GET {path}")


def _get_csrf(ctx=None) -> str:
    """Extract CSRF token from Playwright context cookies.

    If ``ctx`` is provided (e.g. from inside an already-locked _with_recovery
    block) it is used directly — avoids re-entering the RLock and avoids a
    stale context reference after a relaunch. Otherwise acquires the lock
    and fetches a fresh context.
    """
    if ctx is not None:
        for c in ctx.cookies():
            if c["name"] == "AZACSRF":
                return c["value"]
        raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")

    # A-AV-1: ctx.cookies() reads Playwright internal state — needs lock.
    # 2026-06-12 (audit B4 fix 1): when no ctx was passed we are NOT inside
    # an already-marshalled op, so route to the session worker thread.
    def _read_csrf():
        with _pw_lock:
            inner_ctx = _get_playwright_context()
            for c in inner_ctx.cookies():
                if c["name"] == "AZACSRF":
                    return c["value"]
            raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")

    if _on_session_thread():
        return _read_csrf()
    return _run_on_session_thread(_read_csrf, op_name="_get_csrf")


def api_post(path: str, payload: dict) -> Any:
    """Make an authenticated POST request to Avanza API.

    Automatically includes the X-SecurityToken (CSRF) header.

    Args:
        path: API path (e.g., "/_api/trading-critical/rest/order/new")
        payload: Request body dict.

    Returns:
        Parsed JSON response.
    """
    # A-AV-1: Hold lock across CSRF read + POST so a concurrent request
    # cannot rotate the cookie jar mid-flight.
    # 2026-04-13: Wrapped in _with_browser_recovery. CSRF is read from the
    # same ctx used for the POST, so a relaunch picks up fresh cookies in
    # both places atomically (no stale-CSRF-against-fresh-context mismatch).
    url = f"{API_BASE}{path}" if path.startswith("/") else path
    body_data = json.dumps(payload)

    def _op(ctx):
        csrf = _get_csrf(ctx)
        resp = ctx.request.post(
            url,
            data=body_data,
            headers={
                "Content-Type": "application/json",
                "X-SecurityToken": csrf,
            },
        )
        if resp.status == 401:
            close_playwright()
            raise AvanzaSessionError(
                "Session returned 401 Unauthorized. "
                "Run: python scripts/avanza_login.py"
            )
        if resp.status == 403:
            close_playwright()
            raise AvanzaSessionError(
                "Session returned 403 Forbidden — CSRF token may be stale. "
                "Run: python scripts/avanza_login.py"
            )
        body = resp.text()
        try:
            return json.loads(body)
        except (json.JSONDecodeError, TypeError):
            if not resp.ok:
                raise RuntimeError(
                    f"Avanza API error {resp.status}: {body[:500]}"
                ) from None
            return {"raw": body}

    try:
        return _with_browser_recovery(_op, op_name=f"POST {path}")
    except AvanzaSessionTimeout:
        # 2026-06-12 (review fix 18d9d0cc #2): POST is mutating — see
        # _record_mutation_timeout for the residual broker-side risk.
        _record_mutation_timeout("POST", path)
        raise


def api_delete(path: str) -> Any:
    """Make an authenticated DELETE request to Avanza API.

    Automatically includes the X-SecurityToken (CSRF) header.

    Args:
        path: API path (e.g., "/_api/trading/stoploss/{stop_id}")

    Returns:
        Dict with ``http_status`` and ``ok`` keys.
    """
    # A-AV-1: Hold lock across CSRF read + DELETE.
    # 2026-04-13: Wrapped in _with_browser_recovery (see api_get/api_post).
    url = f"{API_BASE}{path}" if path.startswith("/") else path

    def _op(ctx):
        csrf = _get_csrf(ctx)
        resp = ctx.request.delete(
            url,
            headers={
                "Content-Type": "application/json",
                "X-SecurityToken": csrf,
            },
        )
        if resp.status == 401:
            close_playwright()
            raise AvanzaSessionError(
                "Session returned 401 Unauthorized. "
                "Run: python scripts/avanza_login.py"
            )
        return {
            "http_status": resp.status,
            "ok": 200 <= resp.status < 300 or resp.status == 404,
        }

    try:
        return _with_browser_recovery(_op, op_name=f"DELETE {path}")
    except AvanzaSessionTimeout:
        # 2026-06-12 (review fix 18d9d0cc #2): DELETE is mutating — see
        # _record_mutation_timeout for the residual broker-side risk.
        _record_mutation_timeout("DELETE", path)
        raise


# --- Trading convenience functions ---


def get_buying_power(account_id: str | None = None) -> dict | None:
    """Get buying power and account value for an account.

    2026-04-09 (Bug C7 fix): ported the multi-shape + multi-field-ID fallback
    pattern from ``data/metals_avanza_helpers.fetch_account_cash`` after Avanza
    changed the ``/_api/account-overview/overview/categorizedAccounts`` response
    shape mid-day. The endpoint used to return a single top-level key
    ``categorizedAccounts`` (an array of categories each with an ``accounts``
    child). The new shape exposes three top-level keys simultaneously:
    ``categories`` (new categorized path), ``accounts`` (flat list of all user
    accounts), and ``loans``. At the same time, the per-account ID field
    renamed from ``accountId`` to ``id`` (the other Avanza endpoints such as
    ``position-data/positions`` already use ``id`` — see ``get_positions``).

    Previously this function assumed the legacy shape + legacy ID field, so on
    the new shape the iteration walked an empty list, then hit ``cats[0]`` on
    an empty list (IndexError) or — if the shape still exposed the legacy key
    but with no matches — silently returned fake numbers derived from the
    first category's totalValue. That made callers like ``fish_straddle`` and
    ``fish_monitor_live`` size positions off wrong cash balances.

    We now try all three shapes (legacy categorized → flat → new categorized)
    and all four known ID fields (``accountId``, ``id``, ``accountNumber``,
    ``number``), taking whichever finds the target account first. On any
    failure path we return ``None`` so callers can distinguish "API call failed"
    from "balance is legitimately zero" — callers must now explicitly handle
    the ``None`` case (previously they silently got ``buying_power=0``, which
    was a dangerous silent failure).

    Args:
        account_id: Avanza account ID (default: ``DEFAULT_ACCOUNT_ID``).

    Returns:
        Dict with ``buying_power``, ``total_value``, ``own_capital`` (all SEK)
        on success. ``None`` on any failure (HTTP error, account not found,
        shape drift, etc.). Failures are logged with enough diagnostic context
        (sample keys, counts per shape) to identify the next shape drift
        without guessing.
    """
    aid = str(account_id or DEFAULT_ACCOUNT_ID)

    try:
        data = api_get("/_api/account-overview/overview/categorizedAccounts")
    except Exception as e:
        logger.warning(
            "get_buying_power: api_get raised account_id=%s exception=%r",
            aid,
            e,
        )
        return None

    if not isinstance(data, dict):
        logger.warning(
            "get_buying_power: unexpected response type account_id=%s type=%s",
            aid,
            type(data).__name__,
        )
        return None

    def _v(obj):
        """Unwrap Avanza {value: N} wrappers → N, else return obj as-is."""
        if isinstance(obj, dict) and "value" in obj:
            return obj["value"]
        return obj

    def _get_acc_id(acc: dict) -> str | None:
        """Try every known ID field in order — matches fetch_account_cash.

        Order preserved from the reference JS implementation so a regression
        hitting one file makes the other equally easy to diagnose.
        """
        if not isinstance(acc, dict):
            return None
        for key in ("accountId", "id", "accountNumber", "number"):
            val = acc.get(key)
            if val is not None:
                return str(val)
        return None

    def _get_balance(acc: dict, primary: str, alternates: tuple[str, ...]):
        """Try primary balance field, fall back to alternates.

        2026-04-09: we haven't confirmed whether `buyingPower` survived the
        shape change, so we try common alternates if the primary is missing.
        Mirrors getBalance() in fetch_account_cash.
        """
        p = _v(acc.get(primary))
        if p is not None:
            return p
        for alt in alternates:
            x = _v(acc.get(alt))
            if x is not None:
                return x
        return None

    def _make_result(acc: dict) -> dict:
        return {
            "buying_power": _get_balance(
                acc,
                "buyingPower",
                ("buyingPowerAvailable", "availableCash", "availableFunds"),
            ),
            "total_value": _get_balance(
                acc,
                "totalValue",
                ("accountTotalValue", "totalHoldings"),
            ),
            "own_capital": _get_balance(
                acc,
                "ownCapital",
                ("netDeposit", "selfOwnedCapital"),
            ),
        }

    ids_seen: list[str] = []
    sample_account_keys: list[str] | None = None

    def _check_account(acc: dict) -> dict | None:
        nonlocal sample_account_keys
        if sample_account_keys is None and isinstance(acc, dict):
            sample_account_keys = list(acc.keys())
        acc_id = _get_acc_id(acc)
        if acc_id is not None:
            ids_seen.append(acc_id)
        if acc_id == aid:
            return _make_result(acc)
        return None

    # Path A (legacy, pre-2026-04-09): data.categorizedAccounts[].accounts[]
    legacy_cats = data.get("categorizedAccounts") or []
    for cat in legacy_cats:
        for acc in cat.get("accounts") or []:
            r = _check_account(acc)
            if r is not None:
                return r

    # Path B (new flat shape, 2026-04-09): data.accounts[]
    flat_accounts = data.get("accounts") or []
    for acc in flat_accounts:
        r = _check_account(acc)
        if r is not None:
            return r

    # Path C (new categorized shape, 2026-04-09): data.categories[].accounts[]
    new_cats = data.get("categories") or []
    for cat in new_cats:
        for acc in cat.get("accounts") or []:
            r = _check_account(acc)
            if r is not None:
                return r

    # Total miss — log the full diagnostic so the next shape drift is obvious.
    logger.warning(
        "get_buying_power: no_account_match account_id=%s "
        "legacy_category_count=%d flat_account_count=%d new_category_count=%d "
        "ids_seen=%s sample_account_keys=%s top_level_keys=%s",
        aid,
        len(legacy_cats),
        len(flat_accounts),
        len(new_cats),
        ids_seen,
        sample_account_keys,
        list(data.keys()),
    )
    return None


def place_buy_order(
    orderbook_id: str,
    price: float,
    volume: int,
    account_id: str | None = None,
    valid_until: str | None = None,
) -> dict:
    """Place a limit BUY order on Avanza.

    Args:
        orderbook_id: Avanza orderbook ID.
        price: Limit price in SEK.
        volume: Number of units (int >= 1).
        account_id: Defaults to DEFAULT_ACCOUNT_ID.
        valid_until: ISO date string. Defaults to today (day order).

    Returns:
        Dict with orderRequestStatus, orderId, message.
    """
    return _place_order("BUY", orderbook_id, price, volume, account_id, valid_until)


def place_sell_order(
    orderbook_id: str,
    price: float,
    volume: int,
    account_id: str | None = None,
    valid_until: str | None = None,
) -> dict:
    """Place a limit SELL order on Avanza."""
    return _place_order("SELL", orderbook_id, price, volume, account_id, valid_until)


def _place_order(
    side: str,
    orderbook_id: str,
    price: float,
    volume: int,
    account_id: str | None = None,
    valid_until: str | None = None,
) -> dict:
    """Internal: place a BUY or SELL limit order."""
    if not orderbook_id or not str(orderbook_id).strip():
        raise ValueError(f"orderbook_id is required, got {orderbook_id!r}")
    ob_str = str(orderbook_id).strip()
    if not ob_str.isdigit():
        raise ValueError(f"orderbook_id must be numeric, got {orderbook_id!r}")
    if volume < 1:
        raise ValueError(f"volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    # H7: account whitelist guard
    effective_account_id = str(account_id or DEFAULT_ACCOUNT_ID)
    if effective_account_id not in ALLOWED_ACCOUNT_IDS:
        raise ValueError(
            f"Refusing to trade on non-whitelisted account {effective_account_id!r}"
        )

    # H8: minimum order size guard
    # 2026-06-12 (audit B4 fix 6): position-closing SELLs are exempt. The
    # 1000-SEK floor exists to avoid minimum-courtage fee drag on ENTRIES;
    # an unexitable sub-1000 SEK lot (partial fill, or a 1200-SEK leg whose
    # price dropped) is strictly worse than paying minimum courtage once —
    # eod_market_flat previously retried the blocked sell forever after the
    # stop was already cancelled. Mirrors place_stop_loss's warn-don't-raise
    # (2026-04-17). A SELL only qualifies when live positions confirm we
    # hold at least the order volume; lookup failure fails CLOSED (guard
    # still raises) so an unverifiable sell can't bypass the floor.
    order_total = round(volume * price, 2)
    if order_total < 1000.0:
        closing_sell = False
        if side == "SELL":
            try:
                held = 0
                for pos in get_positions():
                    if str(pos.get("orderbook_id") or "") != ob_str:
                        continue
                    if str(pos.get("account_id") or "") != effective_account_id:
                        continue
                    held += int(pos.get("volume") or 0)
                closing_sell = held >= volume
            except Exception as exc:  # noqa: BLE001 — fail closed on lookup error
                logger.warning(
                    "_place_order: could not verify closing-sell exemption for "
                    "ob=%s vol=%d (%s) — keeping 1000 SEK guard",
                    ob_str,
                    volume,
                    exc,
                )
        if not closing_sell:
            raise ValueError(
                f"Order total {order_total:.2f} SEK below minimum 1000 SEK"
            )
        logger.warning(
            "_place_order: sub-minimum closing SELL allowed: %.2f SEK "
            "(vol=%d price=%.3f ob=%s) — minimum-courtage fee applies",
            order_total,
            volume,
            price,
            ob_str,
        )

    # BUG-211: maximum order size guard — prevents full-account exposure from
    # a single malformed call (LLM hallucination, unit error, runaway loop).
    # 50K SEK is ~25% of a 200K ISK account; adjust via config if needed.
    MAX_ORDER_TOTAL_SEK = 50_000.0
    if order_total > MAX_ORDER_TOTAL_SEK:
        raise ValueError(
            f"Order total {order_total:.2f} SEK exceeds maximum {MAX_ORDER_TOTAL_SEK:.0f} SEK"
        )

    payload = {
        "accountId": effective_account_id,
        "orderbookId": ob_str,
        "side": side,
        "condition": "NORMAL",
        "price": price,
        "validUntil": valid_until or date.today().isoformat(),
        "volume": volume,
    }
    # 2026-04-13: cross-process lock — metals_loop + golddigger + fin_snipe
    # must not race on buying_power. 2s fail-fast; busy peer aborts the order
    # (caller retries next cycle).
    with avanza_order_lock(op=f"place_order/{side}/{orderbook_id}"):
        result = api_post("/_api/trading-critical/rest/order/new", payload)
    status = result.get("orderRequestStatus", "UNKNOWN")
    if status != "SUCCESS":
        logger.warning(
            "Order %s failed: %s — %s", side, status, result.get("message", "")
        )
    else:
        logger.info(
            "Order %s placed: %dx @ %.3f SEK (id=%s)",
            side,
            volume,
            price,
            result.get("orderId", "?"),
        )
    return result


def cancel_order(order_id: str, account_id: str | None = None) -> dict:
    """Cancel an open order.

    IMPORTANT: Uses POST (not DELETE verb) — Avanza API change 2026-03-24.
    """
    payload = {
        "accountId": str(account_id or DEFAULT_ACCOUNT_ID),
        "orderId": str(order_id),
    }
    # 2026-04-13: cross-process order lock — cancel is a mutation, same
    # concurrency concern as place_order (don't want two cancels racing).
    with avanza_order_lock(op=f"cancel_order/{order_id}"):
        return api_post("/_api/trading-critical/rest/order/delete", payload)


def get_open_orders(account_id: str | None = None) -> list[dict]:
    """Get all open (unfilled) orders for an account.

    2026-06-12 (audit B4 fix 2): a read failure now RAISES instead of
    returning ``[]``. The old silent-[] fallback made "Avanza 5xx on both
    endpoints" indistinguishable from "no open orders", which let
    GridFisher's reconcile mass-misclassify still-resting orders as
    filled/cancelled (and let the spike-rollback path in metals_loop treat
    an unverifiable cancel as terminal). Callers must treat the exception
    as "order book state UNKNOWN — skip this cycle", never as empty.

    Raises:
        AvanzaSessionError: session invalid (401), call timeout, or both
            order endpoints failed with HTTP errors.
    """
    aid = str(account_id or DEFAULT_ACCOUNT_ID)
    try:
        # 2026-07-13: Avanza removed the account-scoped route AND the old
        # deals-and-orders route ("No static resource" 404 on both). The
        # surviving endpoint returns all accounts' orders — filter locally.
        data = api_get("/_api/trading/rest/orders")
        orders = data.get("orders", []) if isinstance(data, dict) else data
        return [
            o
            for o in orders
            if str(o.get("accountId", o.get("account", {}).get("id", ""))) in (aid, "")
        ]
    except RuntimeError as primary_exc:
        # Fallback: the pre-2026-07 account-scoped route, in case Avanza
        # reverts or the new route disappears.
        try:
            data = api_get(f"/_api/trading/rest/order/account/{aid}")
            if isinstance(data, list):
                return data
            return data.get("orders", data.get("openOrders", []))
        except RuntimeError as fallback_exc:
            logger.warning(
                "Could not fetch open orders (primary=%s fallback=%s)",
                primary_exc,
                fallback_exc,
            )
            raise AvanzaSessionError(
                f"Could not fetch open orders for account {aid}: "
                f"primary={primary_exc}; fallback={fallback_exc}"
            ) from fallback_exc


def get_quote(orderbook_id: str) -> dict:
    """Get bid/ask/last quote for an instrument. Fast single-endpoint call.

    Returns:
        Dict with buy, sell, last, changePercent, highest, lowest.
    """
    return api_get(f"/_api/market-guide/stock/{orderbook_id}/quote")


def get_positions() -> list[dict]:
    """Get all positions via session-based auth.

    Returns:
        List of position dicts with name, value, profit, etc.
    """
    data = api_get("/_api/position-data/positions")
    positions = []
    for entry in data.get("withOrderbook", []):
        inst = entry.get("instrument", {})
        orderbook = inst.get("orderbook", {})
        quote = orderbook.get("quote", {})
        volume_obj = entry.get("volume", {})
        value_obj = entry.get("value", {})
        acquired_obj = entry.get("acquiredValue", {})
        account = entry.get("account", {})

        vol = volume_obj.get("value", 0) if isinstance(volume_obj, dict) else volume_obj
        val = value_obj.get("value", 0) if isinstance(value_obj, dict) else value_obj
        acq = (
            acquired_obj.get("value", 0)
            if isinstance(acquired_obj, dict)
            else acquired_obj
        )
        latest = quote.get("latest", {})
        last_price = latest.get("value", 0) if isinstance(latest, dict) else latest
        change_pct_obj = quote.get("changePercent", {})
        change_pct = (
            change_pct_obj.get("value", 0)
            if isinstance(change_pct_obj, dict)
            else change_pct_obj
        )

        positions.append(
            {
                "name": inst.get("name", orderbook.get("name", "")),
                "orderbook_id": str(orderbook.get("id", "")),
                "instrument_id": str(inst.get("id", "")),
                "type": inst.get("type", orderbook.get("type", "")),
                "volume": vol,
                "value": val,
                "acquired_value": acq,
                "profit": val - acq if val and acq else 0,
                "profit_percent": ((val - acq) / acq * 100) if acq else 0,
                "currency": inst.get("currency", "SEK"),
                "last_price": last_price,
                "change_percent": change_pct,
                "account_id": account.get("id", ""),
                "account_type": account.get("type", ""),
            }
        )
    return positions


def place_stop_loss(
    orderbook_id: str,
    trigger_price: float,
    sell_price: float,
    volume: int,
    account_id: str | None = None,
    valid_days: int = 8,
    trigger_type: str = "LESS_OR_EQUAL",
    value_type: str = "MONETARY",
) -> dict:
    """Place a hardware stop-loss order on Avanza.

    IMPORTANT: Uses /_api/trading/stoploss/new, NOT the regular order API.

    Args:
        orderbook_id: Avanza orderbook ID.
        trigger_price: Price at which to trigger the stop-loss.
            For FOLLOW_DOWNWARDS with PERCENTAGE, this is the trail %.
        sell_price: Price to sell at when triggered.
            For trailing stops (FOLLOW_DOWNWARDS), set to 0 (market).
        volume: Number of units to sell.
        account_id: Defaults to DEFAULT_ACCOUNT_ID.
        valid_days: Days until the stop-loss expires (default 8).
        trigger_type: LESS_OR_EQUAL, MORE_OR_EQUAL, FOLLOW_DOWNWARDS, FOLLOW_UPWARDS.
        value_type: MONETARY (absolute price) or PERCENTAGE.

    Returns:
        Dict with status, stoplossOrderId.
    """
    acct = str(account_id or DEFAULT_ACCOUNT_ID)
    if acct not in ALLOWED_ACCOUNT_IDS:
        raise ValueError(
            f"Refusing to place stop-loss on non-whitelisted account {acct!r}"
        )
    valid_until = (date.today() + timedelta(days=valid_days)).isoformat()

    # BUG-223: trailing stops (FOLLOW_DOWNWARDS/UPWARDS) legitimately use
    # sell_price=0 (market order on trigger). Non-trailing MONETARY stops
    # must have sell_price > 0 — a zero sell_price would execute as a market
    # sell at whatever price exists, potentially the worst available price.
    _TRAILING_TYPES = {"FOLLOW_DOWNWARDS", "FOLLOW_UPWARDS"}
    if (
        trigger_type not in _TRAILING_TYPES
        and value_type == "MONETARY"
        and sell_price <= 0
    ):
        raise ValueError(
            f"Non-trailing stop-loss requires sell_price > 0, got {sell_price}"
        )

    # 2026-04-17: stops below Avanza's 1000 SEK min-courtage threshold still
    # succeed at the API but carry outsized fees. Cascaded-stop callers
    # (metals_loop) can legitimately produce sub-1000 legs, so warn rather
    # than raise — surface fee inefficiency without breaking live stops.
    if value_type == "MONETARY" and sell_price > 0:
        leg_total = round(volume * sell_price, 2)
        if leg_total < 1000.0:
            logger.warning(
                "place_stop_loss leg %.2f SEK below 1000 SEK courtage threshold "
                "(vol=%d sell=%.3f ob=%s)",
                leg_total,
                volume,
                sell_price,
                orderbook_id,
            )

    payload = {
        "parentStopLossId": "0",
        "accountId": acct,
        "orderBookId": str(orderbook_id),
        "stopLossTrigger": {
            "type": trigger_type,
            "value": trigger_price,
            "validUntil": valid_until,
            "valueType": value_type,
            "triggerOnMarketMakerQuote": True,
        },
        "stopLossOrderEvent": {
            "type": "SELL",
            "price": sell_price,
            "volume": volume,
            "validDays": valid_days,
            "priceType": value_type,
            "shortSellingAllowed": False,
        },
    }
    # 2026-04-13: cross-process order lock. Stop-loss placement is
    # especially race-sensitive because cancel-before-place flows are
    # common (see user memory: cancel existing stop BEFORE placing new sell).
    with avanza_order_lock(op=f"place_stop_loss/{orderbook_id}"):
        result = api_post("/_api/trading/stoploss/new", payload)
    status = result.get("status", "UNKNOWN")
    if status == "SUCCESS":
        logger.info(
            "Stop-loss placed: %s trigger=%.3f sell=%.3f vol=%d (id=%s)",
            trigger_type,
            trigger_price,
            sell_price,
            volume,
            result.get("stoplossOrderId", "?"),
        )
    else:
        logger.warning("Stop-loss failed: %s — %s", status, result)
    return result


def place_trailing_stop(
    orderbook_id: str,
    trail_percent: float,
    volume: int,
    account_id: str | None = None,
    valid_days: int = 8,
) -> dict:
    """Place a hardware trailing stop-loss that Avanza manages automatically.

    The stop follows the price downward by trail_percent%. If the instrument
    drops trail_percent% from its peak since placement, the stop triggers a
    market sell.

    Args:
        orderbook_id: Avanza orderbook ID.
        trail_percent: Trailing distance as percentage (e.g. 5.0 for 5%).
        volume: Number of units to sell.
        account_id: Defaults to DEFAULT_ACCOUNT_ID.
        valid_days: Days until the stop expires (default 8).

    Returns:
        Dict with status, stoplossOrderId.
    """
    return place_stop_loss(
        orderbook_id=orderbook_id,
        trigger_price=trail_percent,
        sell_price=0,
        volume=volume,
        account_id=account_id,
        valid_days=valid_days,
        trigger_type="FOLLOW_DOWNWARDS",
        value_type="PERCENTAGE",
    )


def get_stop_losses() -> list[dict]:
    """Get all active stop-loss orders.

    Returns ``[]`` on read failure for backward compatibility with
    callers that treat empty as "nothing to monitor". Code that needs
    to distinguish "no stops" from "could not read stops" must use
    :func:`get_stop_losses_strict` instead — or a False return from
    that function will leave the caller unable to make safety
    decisions like cancel-before-sell.
    """
    try:
        data = api_get("/_api/trading/stoploss")
        return data if isinstance(data, list) else []
    except RuntimeError:
        logger.warning("Could not fetch stop-losses")
        return []


def get_stop_losses_strict() -> list[dict]:
    """Get all active stop-loss orders, raising on any read failure.

    Use this in safety-critical paths (e.g., before a sell) where
    "could not read" must NOT be silently treated as "no stops exist".
    A swallowed read error there would let the dependent sell proceed
    against still-encumbered volume, producing the very
    ``short.sell.not.allowed`` error this module exists to prevent.

    Raises:
        RuntimeError: if the underlying ``api_get`` call fails or
            returns a non-list shape.
    """
    data = api_get("/_api/trading/stoploss")
    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected stop-loss response shape: {type(data).__name__}"
        )
    return data


def cancel_stop_loss(stop_id: str, account_id: str | None = None) -> dict:
    """Cancel a single stop-loss order by ID.

    Idempotent: HTTP 404 (already gone) is treated as success since the
    end-state is identical from the caller's perspective.

    Uses DELETE /_api/trading/stoploss/{accountId}/{stopId}, which is the
    correct endpoint per portfolio/avanza_control.py:206. Do NOT use the
    regular order cancel API — it returns "crossing prices" errors for
    stop-losses (March 3 incident).

    Args:
        stop_id: Avanza stop-loss ID (e.g. "A2^1773297348702^1346781").
        account_id: Avanza account ID. Defaults to ``DEFAULT_ACCOUNT_ID``.

    Returns:
        Dict with keys ``status`` ("SUCCESS"/"FAILED"), ``http_status`` (int),
        and ``stop_id`` (str). Errors that prevent the call from running
        (network, missing CSRF, etc.) yield ``status="FAILED"`` with
        ``http_status=0`` and an ``error`` key describing the cause.
    """
    if not stop_id:
        return {
            "status": "FAILED",
            "http_status": 0,
            "stop_id": "",
            "error": "empty stop_id",
        }
    acct = str(account_id or DEFAULT_ACCOUNT_ID)
    try:
        # 2026-04-13: cross-process order lock — SL cancel is mutating.
        # See cancel_order / place_stop_loss for rationale.
        with avanza_order_lock(op=f"cancel_stop_loss/{stop_id}"):
            result = api_delete(f"/_api/trading/stoploss/{acct}/{stop_id}")
    except Exception as exc:  # noqa: BLE001 — propagate as structured failure
        logger.error("cancel_stop_loss(%s) raised: %s", stop_id, exc, exc_info=True)
        return {
            "status": "FAILED",
            "http_status": 0,
            "stop_id": stop_id,
            "error": str(exc),
        }
    http_status = int(result.get("http_status", 0)) if isinstance(result, dict) else 0
    # 2xx = deleted; 404 = already gone (triggered/expired/cancelled). Both succeed.
    ok = (200 <= http_status < 300) or http_status == 404
    if ok:
        logger.info("cancel_stop_loss(%s) -> %s", stop_id, http_status)
    else:
        logger.warning(
            "cancel_stop_loss(%s) failed: http=%s result=%s",
            stop_id,
            http_status,
            result,
        )
    return {
        "status": "SUCCESS" if ok else "FAILED",
        "http_status": http_status,
        "stop_id": stop_id,
    }


def cancel_all_stop_losses_for(
    orderbook_id: str,
    account_id: str | None = None,
    max_wait: float = 3.0,
    poll_interval: float = 0.5,
) -> dict:
    """Cancel every active stop-loss for ``orderbook_id`` and verify clearance.

    The "verify" step is the critical part: Avanza's DELETE returns 200 OK
    immediately, but the encumbered volume on the position is not released
    until the SL actually disappears from the position view. Without polling,
    a follow-up SELL still gets ``short.sell.not.allowed``. We therefore
    re-query ``get_stop_losses_strict()`` every ``poll_interval`` seconds
    until none remain for the target orderbook (or ``max_wait`` is exceeded).

    **Fail-closed semantics**: if the stop-loss list cannot be read (network
    error, 5xx, malformed response), the function returns ``status="FAILED"``
    rather than silently treating "could not read" as "no stops exist".
    A safety-critical caller deciding whether to proceed with a sell MUST
    NOT be misled into believing the path is clear when reality is unknown.

    The function is idempotent and safe to call when no SLs exist — it
    short-circuits to ``status="SUCCESS"`` without any DELETE calls.

    Args:
        orderbook_id: Avanza orderbook ID to clear.
        account_id: Account filter. ``None`` means accept any account.
        max_wait: Maximum total wall-clock seconds to wait for clearance.
        poll_interval: Seconds between re-query attempts.

    Returns:
        Dict with:
            - ``status``: "SUCCESS" (cleared), "PARTIAL" (some cancelled, some
              still showing after timeout), or "FAILED" (no cancels succeeded
              and stops still present, OR the SL list could not be read).
            - ``cancelled``: list of stop_ids the DELETE call accepted.
            - ``remaining``: list of stop_ids still present after the wait.
            - ``snapshot``: list of full stop-loss dicts that were present at
              the start of the cancel sequence. Callers can use this to
              **re-arm** identical stops if the dependent sell fails — the
              cancel/sell sequence is otherwise rollbackable but leaves the
              position naked on partial-completion failure.
            - ``elapsed_seconds``: float, total time spent in this call.
            - ``error``: optional, present only when ``status="FAILED"`` due
              to a read error rather than cancel failures.
    """
    started = time.monotonic()
    target_ob = str(orderbook_id)
    aid_filter = str(account_id) if account_id is not None else None

    def _filter_for_ob(stops: list[dict]) -> list[dict]:
        out = []
        for sl in stops:
            if not isinstance(sl, dict):
                continue
            ob = (sl.get("orderbook") or {}).get("id")
            if str(ob) != target_ob:
                continue
            if aid_filter is not None:
                acct = (sl.get("account") or {}).get("id")
                if str(acct) != aid_filter:
                    continue
            out.append(sl)
        return out

    # Initial fetch — fail closed on read errors. A safety-critical caller
    # cannot tell "no stops" apart from "API down" without this distinction.
    try:
        all_stops = get_stop_losses_strict()
    except Exception as exc:  # noqa: BLE001 — convert to structured failure
        elapsed = time.monotonic() - started
        logger.error(
            "cancel_all_stop_losses_for(%s): cannot read stop-loss list: %s",
            target_ob,
            exc,
        )
        return {
            "status": "FAILED",
            "cancelled": [],
            "remaining": [],
            "snapshot": [],
            "elapsed_seconds": elapsed,
            "error": f"read_error: {exc}",
        }

    initial = _filter_for_ob(all_stops)
    if not initial:
        return {
            "status": "SUCCESS",
            "cancelled": [],
            "remaining": [],
            "snapshot": [],
            "elapsed_seconds": time.monotonic() - started,
        }

    # Snapshot full dicts before cancelling so a caller can re-arm if the
    # dependent sell fails downstream. We deep-copy to insulate against any
    # downstream mutation of the returned structure.
    import copy as _copy

    snapshot = [_copy.deepcopy(sl) for sl in initial]

    # Issue cancels for every matching stop. Use the SL's own account id when
    # available — Avanza's DELETE endpoint requires the account that owns the
    # stop, which may differ from DEFAULT_ACCOUNT_ID for multi-account users.
    cancelled: list[str] = []
    for sl in initial:
        sid = sl.get("id") or ""
        if not sid:
            continue
        sl_acct = (sl.get("account") or {}).get("id") or account_id
        result = cancel_stop_loss(sid, account_id=sl_acct)
        if result.get("status") == "SUCCESS":
            cancelled.append(sid)

    # Poll until cleared or timeout. Re-query is also fail-closed — if the
    # API stops responding mid-poll, treat the orderbook as "may still have
    # stops" rather than declaring victory.
    remaining: list[str] = []
    poll_read_failed = False
    while True:
        try:
            poll_stops = get_stop_losses_strict()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cancel_all_stop_losses_for(%s): poll read failed: %s",
                target_ob,
                exc,
            )
            poll_read_failed = True
            # We don't know if the stops are gone. Fail closed.
            remaining = [
                sl.get("id", "")
                for sl in initial
                if sl.get("id") and sl.get("id") not in cancelled
            ]
            break
        still = _filter_for_ob(poll_stops)
        remaining = [s.get("id", "") for s in still if s.get("id")]
        if not remaining:
            break
        if (time.monotonic() - started) >= max_wait:
            break
        time.sleep(poll_interval)

    elapsed = time.monotonic() - started

    # CODEX-7 finding 1: critical filter — a DELETE-accepted id can still
    # be in `remaining` if the verification poll observed it alive
    # (broker rejected the cancel late, or the DELETE was acknowledged
    # but never propagated). The set we expose to callers as the
    # rollback set MUST be the VERIFIED-cleared set:
    #     verified = cancelled - remaining
    # Re-arming a stop that is still alive would create a duplicate
    # at the broker, recreating the exact over-encumbered failure mode
    # this whole module exists to prevent.
    remaining_set = set(remaining)
    cancelled = [c for c in cancelled if c not in remaining_set]

    if not remaining and not poll_read_failed:
        status = "SUCCESS"
        logger.info(
            "cancel_all_stop_losses_for(%s): cleared %d stops in %.2fs",
            target_ob,
            len(cancelled),
            elapsed,
        )
    elif cancelled and not poll_read_failed:
        status = "PARTIAL"
        logger.warning(
            "cancel_all_stop_losses_for(%s): PARTIAL — verified_cancelled=%s remaining=%s elapsed=%.2fs",
            target_ob,
            cancelled,
            remaining,
            elapsed,
        )
    else:
        status = "FAILED"
        logger.error(
            "cancel_all_stop_losses_for(%s): FAILED — cancelled=%s remaining=%s read_failed=%s",
            target_ob,
            cancelled,
            remaining,
            poll_read_failed,
        )
        # When the verification poll failed, we don't actually know which
        # DELETEs took effect. The list of DELETE-accepted ids is
        # broker-acknowledged but NOT verified-cleared. Drop them all to
        # be safe on the rollback side.
        if poll_read_failed:
            cancelled = []
    return {
        "status": status,
        "cancelled": cancelled,
        "remaining": remaining,
        "snapshot": snapshot,
        "elapsed_seconds": elapsed,
    }


def rearm_stop_losses_from_snapshot(snapshot: list[dict]) -> dict:
    """Re-place stop-losses from the snapshot returned by
    :func:`cancel_all_stop_losses_for`.

    Used to roll back a cancel-then-sell sequence when the sell fails:
    we cancelled the stops to clear the volume, the sell didn't go through,
    and the position is now naked. Re-arming restores the original
    protection so we are no worse off than before the attempt.

    Notes on best-effort behavior:

    - Each re-arm is independent. If one fails, the others still try.
    - The new stop-loss IDs differ from the originals — Avanza issues
      fresh IDs on every place. Callers tracking IDs in local state must
      replace, not deduplicate.
    - ``valid_days`` is computed from the snapshot's ``trigger.validUntil``
      field where present, falling back to 8 days. The trigger semantics
      and price/volume are preserved exactly.

    Args:
        snapshot: List of stop-loss dicts as returned in
            ``cancel_all_stop_losses_for(...)["snapshot"]``.

    Returns:
        Dict with:
            - ``status``: "SUCCESS" (all re-armed), "PARTIAL" (some failed),
              "FAILED" (none succeeded), or "SUCCESS" (snapshot was empty).
            - ``rearmed``: list of new stop_ids placed.
            - ``failed``: list of original stop_ids that could not be re-armed.
    """
    if not snapshot:
        return {"status": "SUCCESS", "rearmed": [], "failed": []}

    rearmed: list[str] = []
    failed: list[str] = []
    today_iso = date.today()

    for sl in snapshot:
        if not isinstance(sl, dict):
            continue
        original_id = sl.get("id", "")
        try:
            ob_id = (sl.get("orderbook") or {}).get("id")
            account = (sl.get("account") or {}).get("id")
            trigger = sl.get("trigger") or {}
            order = sl.get("order") or {}
            trigger_value = trigger.get("value")
            trigger_type = trigger.get("type", "LESS_OR_EQUAL")
            value_type = trigger.get("valueType", "MONETARY")
            sell_price = order.get("price")
            volume = order.get("volume")

            # Compute valid_days from validUntil if present, else default 8.
            valid_days = 8
            valid_until = trigger.get("validUntil")
            if valid_until:
                try:
                    parsed = datetime.strptime(valid_until, "%Y-%m-%d").date()
                    delta = (parsed - today_iso).days
                    if delta > 0:
                        valid_days = delta
                except (ValueError, TypeError):
                    pass

            if not (
                ob_id
                and trigger_value is not None
                and sell_price is not None
                and volume
            ):
                logger.warning(
                    "rearm_stop_losses: snapshot entry missing fields: %s", sl
                )
                failed.append(original_id)
                continue

            result = place_stop_loss(
                orderbook_id=str(ob_id),
                trigger_price=float(trigger_value),
                sell_price=float(sell_price),
                volume=int(volume),
                account_id=account,
                valid_days=valid_days,
                trigger_type=str(trigger_type),
                value_type=str(value_type),
            )
            if result.get("status") == "SUCCESS":
                new_id = result.get("stoplossOrderId", "")
                rearmed.append(new_id)
                logger.info(
                    "rearm_stop_losses: replaced %s -> %s (ob=%s vol=%s)",
                    original_id,
                    new_id,
                    ob_id,
                    volume,
                )
            else:
                logger.warning(
                    "rearm_stop_losses: place_stop_loss failed for original %s: %s",
                    original_id,
                    result,
                )
                failed.append(original_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rearm_stop_losses: exception for original %s: %s",
                original_id,
                exc,
                exc_info=True,
            )
            failed.append(original_id)

    if not failed:
        status = "SUCCESS"
    elif rearmed:
        status = "PARTIAL"
    else:
        status = "FAILED"
    return {"status": status, "rearmed": rearmed, "failed": failed}


def get_instrument_price(orderbook_id: str) -> dict[str, Any]:
    """Get price info for a specific instrument.

    Args:
        orderbook_id: Avanza orderbook ID (numeric string)

    Returns:
        Dict with lastPrice, changePercent, etc.
    """
    # Try stock first, then fund, then certificate/warrant
    for instrument_type in ("stock", "certificate", "fund", "exchange_traded_fund"):
        try:
            data = api_get(
                f"/_api/market-guide/{instrument_type}/{orderbook_id}",
            )
            return data
        except Exception as e:
            logger.warning(
                "Market guide lookup failed for %s/%s: %s",
                instrument_type,
                orderbook_id,
                e,
            )
            continue

    # Fallback: generic orderbook endpoint
    return api_get(f"/_api/orderbook/{orderbook_id}")
