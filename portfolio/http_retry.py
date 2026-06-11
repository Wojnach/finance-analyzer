"""HTTP retry utility with exponential backoff for finance-analyzer API calls."""

import logging
import random
import re
import time

import requests

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
FATAL_STATUS = {400, 401, 403, 404, 405, 410, 422}

# 2026-06-11: hard cap on any single retry sleep. Telegram 429 flood-waits
# (parameters.retry_after) are uncapped — hundreds to thousands of seconds —
# and _do_send_telegram / telegram_poller call fetch_with_retry synchronously
# from Layer-1 cycle paths. An uncapped sleep (further doubled by additive
# jitter) blocks the loop thread far past the 300s heartbeat staleness
# threshold, tripping the loop-contract watchdog restart. Cap the effective
# sleep so a flood-wait degrades to "retries exhausted -> None" (caller logs
# the message unsent; it is already persisted to telegram_messages.jsonl)
# instead of stalling the loop.
MAX_RETRY_SLEEP_S = 90.0

_SECRET_URL_RE = re.compile(r"/bot[0-9]+:[A-Za-z0-9_-]+/")


def _redact_url(url):
    """Mask Telegram bot tokens and similar secrets in URLs before logging."""
    return _SECRET_URL_RE.sub("/bot***/", url)


def _capped_sleep(wait, deadline):
    """Clamp *wait* to MAX_RETRY_SLEEP_S (and to any remaining *deadline*
    budget). Returns the actual time slept, or None if there is no budget
    left to sleep at all (caller should stop retrying)."""
    wait = min(wait, MAX_RETRY_SLEEP_S)
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        wait = min(wait, remaining)
    time.sleep(wait)
    return wait


def fetch_with_retry(url, method="GET", retries=DEFAULT_RETRIES,
                     backoff=DEFAULT_BACKOFF, backoff_factor=DEFAULT_BACKOFF_FACTOR,
                     timeout=30, headers=None, params=None, json_body=None,
                     session=None, total_deadline=None):
    """Make an HTTP request with exponential backoff retry.

    Returns response object on success, None on all retries exhausted.

    total_deadline: optional wall-clock budget (seconds) for the whole
        retry sequence. Once exceeded, no further sleeps are taken and the
        call returns None. Independent of the per-sleep MAX_RETRY_SLEEP_S
        cap (2026-06-11) which always applies.
    """
    requester = session or requests
    deadline = (time.monotonic() + total_deadline) if total_deadline else None

    for attempt in range(retries + 1):
        try:
            if method.upper() == "GET":
                resp = requester.get(url, headers=headers, params=params, timeout=timeout)
            elif method.upper() == "POST":
                resp = requester.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
            else:
                resp = requester.request(method, url, headers=headers, params=params, timeout=timeout)

            if resp.status_code not in RETRYABLE_STATUS:
                if resp.status_code in FATAL_STATUS:
                    logger.warning("HTTP %s (fatal) from %s — not retrying",
                                   resp.status_code, _redact_url(url))
                return resp

            if attempt < retries:
                wait = backoff * (backoff_factor ** attempt)
                if resp.status_code == 429:
                    try:
                        retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
                        # 2026-06-11: coerce to float — a malformed body could
                        # carry a string (or anything); fall back to the
                        # backoff wait rather than propagating a non-numeric
                        # value into the cap/sleep arithmetic.
                        wait = float(retry_after)
                    except Exception:
                        pass  # keep the exponential backoff wait
                wait += random.uniform(0, wait)
                slept = _capped_sleep(wait, deadline)
                if slept is None:
                    logger.error("HTTP %s from %s — retry deadline exhausted",
                                 resp.status_code, _redact_url(url))
                    return None
                logger.warning("HTTP %s from %s, retry %d/%d in %.1fs",
                               resp.status_code, _redact_url(url), attempt + 1, retries, slept)
            else:
                logger.error("HTTP %s from %s after %d retries",
                             resp.status_code, _redact_url(url), retries)
                return None

        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < retries:
                wait = backoff * (backoff_factor ** attempt)
                wait += random.uniform(0, wait)
                slept = _capped_sleep(wait, deadline)
                if slept is None:
                    logger.error("Request to %s failed — retry deadline exhausted: %s",
                                 _redact_url(url), e)
                    return None
                logger.warning("%s from %s, retry %d/%d in %.1fs",
                               e.__class__.__name__, _redact_url(url), attempt + 1, retries, slept)
            else:
                logger.error("Request failed after %d retries: %s - %s",
                             retries, _redact_url(url), e)
                return None

    return None


def fetch_json(url, *, method="GET", retries=DEFAULT_RETRIES, default=None,
               label="", headers=None, params=None, timeout=30, session=None,
               json_body=None, total_deadline=None, **kwargs):
    """Fetch URL and return parsed JSON, or ``default`` on any failure.

    Combines fetch_with_retry() + raise_for_status() + .json() into one call.

    2026-06-11: previously accepted **kwargs but silently dropped them, so a
    caller writing ``fetch_json(url, method='POST', json_body={...})`` got a
    body-less request with no error — a silent-failure trap. ``json_body`` and
    ``total_deadline`` are now explicit and forwarded; any *other* unexpected
    keyword raises TypeError instead of being swallowed.
    """
    if kwargs:
        raise TypeError(
            f"fetch_json() got unexpected keyword argument(s): "
            f"{', '.join(sorted(kwargs))}"
        )
    resp = fetch_with_retry(url, method=method, retries=retries, timeout=timeout,
                            headers=headers, params=params, session=session,
                            json_body=json_body, total_deadline=total_deadline)
    if resp is None:
        if label:
            logger.warning("[%s] request returned None", label)
        return default
    try:
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        if label:
            logger.warning("[%s] HTTP %s or JSON parse error: %s", label,
                           getattr(resp, 'status_code', '?'), e)
        return default
