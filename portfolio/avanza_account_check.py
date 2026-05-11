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

from portfolio.file_utils import atomic_append_jsonl

logger = logging.getLogger("portfolio.avanza_account_check")

# Category strings that disqualify an account from leveraged-warrant
# trading. Avanza category labels are inconsistent across endpoints; we
# do case-insensitive substring matching so a label like
# ``"INVESTERINGSSPARKONTO"`` or ``"Investeringssparkonto"`` or
# ``"ISK"`` is all caught by a single rule. Update the list when a new
# disqualifying category surfaces.
DISALLOWED_CATEGORY_FRAGMENTS: tuple[str, ...] = (
    "investeringsspar",  # ISK — Swedish equity savings, warrants disallowed
    "kapitalfors",        # KF — leveraged certs not allowed in pension wrappers
    "kapitalförs",
    "tjanstepens",        # tjänstepension
    "tjänstepens",
    "pension",
    "isk",                # short form, also seen in some responses
)


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


def _record_critical_error(account_id: str, label: str, reason: str) -> None:
    entry = {
        "ts": _now_iso(),
        "level": "critical",
        "category": "avanza_account_mismatch",
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
    auth/playwright init."""
    from portfolio.avanza_session import api_get  # noqa: PLC0415
    return api_get("/_api/account-overview/overview/categorizedAccounts")


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
        result = {
            "ok": False,
            "account_id": account_id,
            "category": "",
            "reason": f"fetch_failed:{exc!s}",
        }
        logger.warning(
            "verify_default_account: fetch failed account=%s err=%r",
            account_id, exc,
        )
        _record_critical_error(account_id, "", result["reason"])
        if effective_raise:
            raise AccountCategoryMismatch(
                f"Could not verify account {account_id}: {exc}"
            ) from exc
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
    with _cache_lock:
        _cache_result = dict(result)
    return result


def reset_cache() -> None:
    """Drop the cached verification — useful in tests."""
    global _cache_result
    with _cache_lock:
        _cache_result = None
