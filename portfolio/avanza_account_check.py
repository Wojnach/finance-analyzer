"""Startup verification that DEFAULT_ACCOUNT_ID points to a trading account.

User reported on 2026-05-11 that the dashboard's account view for
account 1625505 (the hard-coded ``DEFAULT_ACCOUNT_ID`` in
``portfolio/avanza_session.py``) showed Beammwave / NextEra / Vertiv
holdings — typical ISK long-term holdings, not the warrant-trading
positions the bots are supposed to manage. If the configured account is
an ISK, every live order has been routing to the wrong place.

This module fetches the categorized-accounts overview at startup,
finds the configured account, and inspects its category. If the
category matches an ISK / pension / insurance pattern (where leveraged
warrants cannot legally trade in Sweden anyway), it logs a critical
error, emits a Telegram alert, and raises ``AccountCategoryMismatch``
so the caller can fail closed before the first order placement.

Override knob: setting the env var ``PF_SKIP_ACCOUNT_CHECK=1`` makes
the verification log-and-continue instead of raising. Use during a
known-bad window only — the warning still lands in
``data/critical_errors.jsonl`` so the issue stays surfaced.

Result is cached for the lifetime of the process so the metals loop,
golddigger, and grid fisher can each call ``verify_default_account()``
during their respective inits without re-hammering the API.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import threading
from typing import Optional

from portfolio.file_utils import atomic_append_jsonl, load_jsonl_tail

logger = logging.getLogger("portfolio.avanza_account_check")

# Session-expiry de-escalation (2026-06-01). An Avanza BankID session lasts
# ~24 h and only a human relogin (`python scripts/avanza_login.py`) can renew
# it. Logging that expected, operational state as a loop-critical error every
# day (a) clutters the unresolved-critical list and (b) spawns a
# PF-FixAgentDispatcher fix agent (Read/Edit/Bash only) that cannot possibly fix
# it. So a plain session expiry is journaled at level "warning" / category
# "avanza_session_expired" instead. BUT a session that stays dead — the exact
# 3-week silent-outage class CLAUDE.md warns about — must NOT hide: if >=N
# unresolved expiries span >H hours with no successful relogin in between, the
# next one re-escalates to a real critical (premortem #5).
SESSION_EXPIRED_CATEGORY = "avanza_session_expired"
_PERSISTENT_EXPIRY_MIN_PRIORS = 2     # this expiry + 2 unresolved priors = 3rd
_PERSISTENT_EXPIRY_MIN_AGE_H = 24.0   # ...and the oldest is >24 h old
_AUTO_RESOLVE_WINDOW_DAYS = 7         # match check_critical_errors' 7-day window
_AVANZA_ALERT_CATEGORIES = ("avanza_account_mismatch", SESSION_EXPIRED_CATEGORY)

# Category strings that disqualify an account from leveraged-warrant
# trading. Empty by default — 2026-05-11 user confirmation made it
# clear that the original premise ("ISK forbids warrants") was wrong:
# Swedish ISK accounts legally hold warrants, certificates, and ETPs;
# the tax treatment differs but the trades are routine. Same for
# Kapitalförsäkring (insurance wrapper) and Tjänstepension — those
# can trade leveraged instruments too. The verifier now confirms the
# account *exists* under the configured ID and logs its category; the
# operator owns the decision about what to trade where.
#
# If a future category genuinely cannot trade warrants (e.g. a
# restricted minor account), add the fragment here. Substring match
# is case-insensitive.
DISALLOWED_CATEGORY_FRAGMENTS: tuple[str, ...] = ()


CRITICAL_ERRORS_LOG = "data/critical_errors.jsonl"
SKIP_ENV_VAR = "PF_SKIP_ACCOUNT_CHECK"

_cache_lock = threading.Lock()
_cache_result: Optional[dict] = None


class AccountCategoryMismatch(RuntimeError):
    """Raised when the configured DEFAULT_ACCOUNT_ID lands on a
    disqualified category (ISK / pension / insurance)."""


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _walk_accounts(data: dict):
    """Yield ``(category_label, account_dict)`` for every account in
    every known response shape. Mirrors the multi-path walk that
    ``avanza_session.get_buying_power`` uses so we cover shape drift.
    """
    if not isinstance(data, dict):
        return
    # Path A: legacy categorizedAccounts[].accounts[]
    for cat in (data.get("categorizedAccounts") or []):
        label = cat.get("name") or cat.get("type") or cat.get("category") or ""
        for acc in (cat.get("accounts") or []):
            yield str(label), acc
    # Path B: new flat shape — accounts[]. Each account carries its
    # own category field.
    for acc in (data.get("accounts") or []):
        label = (acc.get("type") or acc.get("category")
                 or acc.get("accountType") or "")
        yield str(label), acc
    # Path C: new categorized shape — categories[].accounts[]
    for cat in (data.get("categories") or []):
        label = (cat.get("name") or cat.get("type")
                 or cat.get("category") or "")
        for acc in (cat.get("accounts") or []):
            yield str(label), acc


def _extract_account_id(acc: dict) -> Optional[str]:
    """Pick whichever ID field this account exposes. Same precedence
    order as ``avanza_session.get_buying_power``."""
    if not isinstance(acc, dict):
        return None
    for key in ("accountId", "id", "accountNumber", "number"):
        val = acc.get(key)
        if val is not None:
            return str(val)
    return None


def _category_disallowed(label: str) -> bool:
    """Match the category label against the disqualifying fragments
    case-insensitively. Empty label → not disqualified (caller logs
    a warning instead)."""
    if not label:
        return False
    norm = label.lower()
    return any(fragment in norm for fragment in DISALLOWED_CATEGORY_FRAGMENTS)


def _record_critical_error(
    account_id: str,
    label: str,
    reason: str,
    *,
    level: str = "critical",
    category: str = "avanza_account_mismatch",
) -> None:
    """Append an avanza account-check alert to the critical-errors journal.

    Defaults preserve the historical behaviour (critical /
    avanza_account_mismatch) for account_not_found and disallowed_category. The
    fetch_failed branch overrides level/category to de-escalate a routine
    session expiry (see SESSION_EXPIRED_CATEGORY notes above).
    """
    entry = {
        "ts": _now_iso(),
        "level": level,
        "category": category,
        "caller": "portfolio.avanza_account_check",
        "message": (
            f"DEFAULT_ACCOUNT_ID={account_id} category={label!r} "
            f"reason={reason}"
        ),
        "context": {
            "account_id": account_id,
            "category_label": label,
            "reason": reason,
        },
    }
    try:
        atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
    except Exception:  # noqa: BLE001
        logger.exception("failed to append critical_errors entry")


def _is_session_expiry(reason: str) -> bool:
    """True only for the routine BankID-session-expired fetch failure.

    Narrow on purpose (premortem #5): a generic Playwright/DNS/5xx error must
    NOT be mistaken for an expiry and silently downgraded out of the critical
    list. Requires the fetch_failed prefix AND the explicit expiry phrase the
    avanza_session layer emits.
    """
    r = (reason or "").lower()
    return r.startswith("fetch_failed:") and "session expired" in r


def _parse_ts(ts: str) -> Optional[_dt.datetime]:
    """Tolerant ISO-8601 parse → aware UTC datetime, or None. Handles both the
    '…Z' and '…+00:00' shapes that appear in critical_errors.jsonl."""
    if not ts:
        return None
    try:
        dt = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _recent_journal() -> list[dict]:
    """Bounded tail of the critical-errors journal (best-effort, never raises)."""
    try:
        return load_jsonl_tail(CRITICAL_ERRORS_LOG, max_entries=4000) or []
    except Exception:  # noqa: BLE001
        return []


def _resolved_ts_set(entries: list[dict]) -> set[str]:
    """Original timestamps already closed by a follow-up resolution line."""
    return {e["resolves_ts"] for e in entries if e.get("resolves_ts")}


def _is_persistent_expiry(account_id: str, now: _dt.datetime) -> bool:
    """A session expiry is 'persistent' when prior unresolved expiry warnings
    for this account already span > _PERSISTENT_EXPIRY_MIN_AGE_H hours. A real
    relogin produces an ok=True verify which auto-resolves them, so surviving
    unresolved expiries mean the session never came back — re-escalate."""
    entries = _recent_journal()
    resolved = _resolved_ts_set(entries)
    ages_h: list[float] = []
    for e in entries:
        if e.get("category") != SESSION_EXPIRED_CATEGORY:
            continue
        if (e.get("context") or {}).get("account_id") != account_id:
            continue
        ts = e.get("ts")
        if not ts or ts in resolved:
            continue
        dt = _parse_ts(ts)
        if dt is None:
            continue
        ages_h.append((now - dt).total_seconds() / 3600.0)
    if len(ages_h) < _PERSISTENT_EXPIRY_MIN_PRIORS:
        return False
    return max(ages_h) >= _PERSISTENT_EXPIRY_MIN_AGE_H


def _auto_resolve_prior_alerts(account_id: str, category_label: str) -> None:
    """On a successful verify, close prior unresolved avanza alerts for THIS
    account so a relogin clears the backlog instead of leaving it to pile up.

    Bounded + idempotent (premortem #6/#7): only entries from the last 7 days,
    only matching account_id, and only those without an existing resolution —
    so re-running across multiple loop processes appends nothing the second
    time. Never raises into the verify path."""
    try:
        entries = _recent_journal()
        if not entries:
            return
        resolved = _resolved_ts_set(entries)
        now = _dt.datetime.now(_dt.timezone.utc)
        cutoff = now - _dt.timedelta(days=_AUTO_RESOLVE_WINDOW_DAYS)
        targets: list[str] = []
        for e in entries:
            if e.get("category") not in _AVANZA_ALERT_CATEGORIES:
                continue
            if e.get("level") not in ("critical", "warning"):
                continue
            if (e.get("context") or {}).get("account_id") != account_id:
                continue
            ts = e.get("ts")
            if not ts or ts in resolved or ts in targets:
                continue
            dt = _parse_ts(ts)
            if dt is None or dt < cutoff:
                continue
            targets.append(ts)
        for ts in targets:
            atomic_append_jsonl(CRITICAL_ERRORS_LOG, {
                "ts": _now_iso(),
                "level": "info",
                "category": "resolution",
                "caller": "portfolio.avanza_account_check",
                "resolution": (
                    f"Avanza account {account_id} re-verified OK "
                    f"(category={category_label!r}) — session re-authenticated."
                ),
                "resolves_ts": ts,
                "message": "Auto-resolved avanza alert on successful verify",
                "context": {"account_id": account_id},
            })
        if targets:
            logger.info(
                "auto-resolved %d prior avanza alert(s) for account %s",
                len(targets), account_id,
            )
    except Exception:  # noqa: BLE001
        logger.debug("avanza auto-resolve skipped", exc_info=True)


def _send_telegram(message: str) -> None:
    """Best-effort Telegram alert. Imports lazily so test paths that
    don't have the config can still exercise the helper."""
    try:
        from portfolio.file_utils import load_json  # noqa: PLC0415
        from portfolio.telegram_notifications import send_telegram  # noqa: PLC0415
        cfg = load_json("config.json") or {}
        if cfg.get("telegram", {}).get("token"):
            send_telegram(message, cfg)
    except Exception:  # noqa: BLE001
        logger.debug("telegram alert skipped", exc_info=True)


def _api_get_categorized_accounts():
    """Pulled out so tests can mock without touching avanza_session
    auth/playwright init.

    Runs the sync_playwright fetch on a single-worker thread pool so
    callers inside an asyncio context (notably ``metals_loop`` which
    has an LLM worker + Playwright swing-trader page sharing the main
    thread's event loop) don't trip ``Playwright Sync API inside the
    asyncio loop``. Without the worker, every verifier invocation
    inside metals_loop returned ``fetch_failed`` — the helper would
    never actually catch an ISK mismatch in production. Same pattern
    ``GridFisher._safe_session_call`` uses.
    """
    import concurrent.futures  # noqa: PLC0415

    from portfolio.avanza_session import api_get  # noqa: PLC0415

    def _runner():
        return api_get("/_api/account-overview/overview/categorizedAccounts")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="avanza-acct-check",
    ) as pool:
        return pool.submit(_runner).result(timeout=30)


def verify_default_account(
    account_id: Optional[str] = None,
    *,
    raise_on_mismatch: bool = True,
    use_cache: bool = True,
) -> dict:
    """Confirm the configured Avanza account is a trading-class account.

    Returns a dict describing the verification result:
      ``{"ok": bool, "account_id": str, "category": str, "reason": str}``

    When ``ok`` is ``False`` and ``raise_on_mismatch`` is True, raises
    :class:`AccountCategoryMismatch`. The ``PF_SKIP_ACCOUNT_CHECK=1``
    env var overrides ``raise_on_mismatch`` to ``False`` so an
    operator can keep the loops running while they investigate.

    Args:
        account_id: Account ID to verify. Defaults to
            ``avanza_session.DEFAULT_ACCOUNT_ID`` so most callers can
            simply call ``verify_default_account()``.
        raise_on_mismatch: When True, raise on disallowed category.
            Setting ``PF_SKIP_ACCOUNT_CHECK=1`` env always wins and
            forces ``False``.
        use_cache: When True (default), reuse a previous successful
            verification for the same account ID within this process.
            Failed verifications are NOT cached — they re-check on each
            call so a fixed config picks up immediately.
    """
    global _cache_result

    if account_id is None:
        from portfolio.avanza_session import DEFAULT_ACCOUNT_ID  # noqa: PLC0415
        account_id = DEFAULT_ACCOUNT_ID
    account_id = str(account_id)

    with _cache_lock:
        if (use_cache and _cache_result
                and _cache_result.get("account_id") == account_id
                and _cache_result.get("ok")):
            return dict(_cache_result)

    skip_raise = os.environ.get(SKIP_ENV_VAR, "").strip() in ("1", "true", "yes")
    effective_raise = raise_on_mismatch and not skip_raise

    try:
        data = _api_get_categorized_accounts()
    except Exception as exc:  # noqa: BLE001
        # Codex P2 fix (2026-05-11): a transient categorizedAccounts
        # outage (DNS flap, 5xx, auth blip) must NOT permanently brick
        # the grid fisher for the rest of the process. Only positive
        # mismatches (disallowed_category / account_not_found) fail
        # closed. fetch_failed downgrades to a logged warning + a
        # critical_errors entry so the operator can still see the
        # outage. Callers can retry verification before the first
        # order placement; if Avanza is genuinely unreachable, the
        # actual order placement will fail too and the order-side
        # guards still apply.
        reason = f"fetch_failed:{exc!s}"
        result = {
            "ok": False,
            "account_id": account_id,
            "category": "",
            "reason": reason,
        }
        if _is_session_expiry(reason):
            now = _dt.datetime.now(_dt.timezone.utc)
            if _is_persistent_expiry(account_id, now):
                # Session has stayed dead > 24 h across multiple checks — a
                # relogin would have cleared the priors. Re-escalate so it can't
                # hide as routine noise (the 3-week silent-outage class).
                logger.error(
                    "verify_default_account: account=%s session expired and NOT "
                    "recovering (>24h, multiple attempts) — escalating to CRITICAL",
                    account_id,
                )
                _record_critical_error(
                    account_id, "", f"persistent_session_expiry:{exc!s}",
                    level="critical", category="avanza_account_mismatch",
                )
            else:
                # Routine ~24 h expiry — operational, needs a human BankID
                # relogin. Journal as a warning so it stays visible but does not
                # trip check_critical_errors / the fix-agent dispatcher.
                logger.warning(
                    "verify_default_account: account=%s BankID session expired "
                    "(operational — run scripts/avanza_login.py) — logged as warning",
                    account_id,
                )
                _record_critical_error(
                    account_id, "", reason,
                    level="warning", category=SESSION_EXPIRED_CATEGORY,
                )
            return result
        # Genuine transient outage (DNS flap, 5xx, auth blip) — keep CRITICAL so
        # the operator sees a real connectivity problem.
        logger.warning(
            "verify_default_account: fetch failed account=%s err=%r — "
            "NOT raising (transient outage path), logged critical",
            account_id, exc,
        )
        _record_critical_error(account_id, "", reason)
        return result

    found_label: Optional[str] = None
    seen_ids: list[str] = []
    for label, acc in _walk_accounts(data):
        acc_id = _extract_account_id(acc)
        if acc_id is not None:
            seen_ids.append(acc_id)
        if acc_id == account_id:
            found_label = label
            break

    if found_label is None:
        result = {
            "ok": False,
            "account_id": account_id,
            "category": "",
            "reason": "account_not_found",
            "seen_ids": seen_ids,
        }
        logger.warning(
            "verify_default_account: %s not in categorizedAccounts. "
            "seen=%s", account_id, seen_ids,
        )
        _record_critical_error(account_id, "", "account_not_found")
        _send_telegram(
            f"⚠️ Avanza account {account_id} not found in "
            f"categorizedAccounts. Saw: {seen_ids[:8]}"
        )
        if effective_raise:
            raise AccountCategoryMismatch(
                f"Account {account_id} not present in Avanza "
                f"categorizedAccounts response (saw {seen_ids})"
            )
        return result

    if _category_disallowed(found_label):
        result = {
            "ok": False,
            "account_id": account_id,
            "category": found_label,
            "reason": "disallowed_category",
        }
        logger.error(
            "verify_default_account: %s lives in disallowed category %r — "
            "warrants cannot trade here", account_id, found_label,
        )
        _record_critical_error(account_id, found_label, "disallowed_category")
        _send_telegram(
            f"🚨 Avanza account {account_id} category {found_label!r} "
            f"disallows warrant trading. Update DEFAULT_ACCOUNT_ID."
        )
        if effective_raise:
            raise AccountCategoryMismatch(
                f"Account {account_id} category {found_label!r} disallows "
                f"warrant trading"
            )
        return result

    result = {
        "ok": True,
        "account_id": account_id,
        "category": found_label or "<empty>",
        "reason": "trading_class",
    }
    logger.info(
        "verify_default_account: %s category=%s OK",
        account_id, found_label or "<empty>",
    )
    # A successful verify means the BankID session is alive again — close any
    # prior unresolved avanza alerts for this account so a relogin clears the
    # backlog. Best-effort, idempotent, bounded to the last 7 days.
    _auto_resolve_prior_alerts(account_id, found_label or "<empty>")
    with _cache_lock:
        _cache_result = dict(result)
    return result


def reset_cache() -> None:
    """Drop the cached verification — useful in tests."""
    global _cache_result
    with _cache_lock:
        _cache_result = None
