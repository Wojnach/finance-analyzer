"""HTTP retry utility with exponential backoff for finance-analyzer API calls."""

import time
import requests
import logging

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
    last_exc = None

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
                logger.warning(f"HTTP {resp.status_code} from {url}, retry {attempt+1}/{retries} in {wait:.1f}s")
                time.sleep(wait)
            else:
                logger.error(f"HTTP {resp.status_code} from {url} after {retries} retries")
                return resp

        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < retries:
                wait = backoff * (backoff_factor ** attempt)
                logger.warning(f"Request error {e.__class__.__name__} from {url}, retry {attempt+1}/{retries} in {wait:.1f}s")
                time.sleep(wait)
            else:
                logger.error(f"Request failed after {retries} retries: {url} - {e}")
                return None

    return None
