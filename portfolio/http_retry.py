"""HTTP retry utility with exponential backoff for finance-analyzer API calls."""

import logging
import random
import time

import requests

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def fetch_with_retry(url, method="GET", retries=DEFAULT_RETRIES,
                     backoff=DEFAULT_BACKOFF, backoff_factor=DEFAULT_BACKOFF_FACTOR,
                     timeout=30, headers=None, params=None, json_body=None,
                     session=None):
    """Make an HTTP request with exponential backoff retry.

    Returns response object on success, None on all retries exhausted.
    """
    requester = session or requests

    for attempt in range(retries + 1):
        try:
            if method.upper() == "GET":
                resp = requester.get(url, headers=headers, params=params, timeout=timeout)
            elif method.upper() == "POST":
                resp = requester.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
            else:
                resp = requester.request(method, url, headers=headers, params=params, timeout=timeout)

            if resp.status_code not in RETRYABLE_STATUS:
                return resp

            if attempt < retries:
                wait = backoff * (backoff_factor ** attempt)
                jitter = random.uniform(0, wait * 0.1)
                wait += jitter
                logger.warning("HTTP %s from %s, retry %d/%d in %.1fs",
                               resp.status_code, url, attempt + 1, retries, wait)
                time.sleep(wait)
            else:
                logger.error("HTTP %s from %s after %d retries",
                             resp.status_code, url, retries)
                return None

        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < retries:
                wait = backoff * (backoff_factor ** attempt)
                jitter = random.uniform(0, wait * 0.1)
                wait += jitter
                logger.warning("%s from %s, retry %d/%d in %.1fs",
                               e.__class__.__name__, url, attempt + 1, retries, wait)
                time.sleep(wait)
            else:
                logger.error("Request failed after %d retries: %s - %s",
                             retries, url, e)
                return None

    return None


def fetch_json(url, *, method="GET", retries=DEFAULT_RETRIES, default=None,
               label="", headers=None, params=None, timeout=30, session=None,
               **kwargs):
    """Fetch URL and return parsed JSON, or ``default`` on any failure.

    Combines fetch_with_retry() + raise_for_status() + .json() into one call.
    """
    resp = fetch_with_retry(url, method=method, retries=retries, timeout=timeout,
                            headers=headers, params=params, session=session)
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
