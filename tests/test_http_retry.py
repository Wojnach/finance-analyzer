"""Comprehensive tests for HTTP retry module.

Covers:
- Successful request on first try
- Retry on transient errors (ConnectionError, Timeout)
- Retry on 5xx status codes (and 429)
- No retry on 4xx status codes (400, 401, 403, 404)
- Max retries exhausted returns None (both exceptions and retryable status codes)
- Exponential backoff timing (mock sleep)
- POST method support (json_body parameter)
- Custom headers passed through
- Timeout parameter passed through
- Custom session support
- Params parameter passed through
- Other HTTP methods (PUT, DELETE)
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from portfolio.http_retry import (
    DEFAULT_BACKOFF,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_RETRIES,
    RETRYABLE_STATUS,
    fetch_json,
    fetch_with_retry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code=200):
    """Create a MagicMock response with the given status code."""
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def _patch_requests():
    """Return a context manager that patches both requests module and time.sleep."""
    return patch("portfolio.http_retry.requests"), patch("portfolio.http_retry.time.sleep")


# ---------------------------------------------------------------------------
# 1. Successful request on first try
# ---------------------------------------------------------------------------

class TestSuccessOnFirstTry:
    def test_get_returns_200_immediately(self):
        """A 200 response should be returned without any retries."""
        mock_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data")

        assert result is mock_resp
        assert result.status_code == 200
        assert mock_req.get.call_count == 1

    def test_get_returns_201_immediately(self):
        """Non-error status codes like 201 should be returned immediately."""
        mock_resp = _mock_response(201)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data")

        assert result.status_code == 201
        assert mock_req.get.call_count == 1

    def test_get_returns_204_immediately(self):
        """204 No Content should be returned immediately."""
        mock_resp = _mock_response(204)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data")

        assert result.status_code == 204
        assert mock_req.get.call_count == 1


# ---------------------------------------------------------------------------
# 2. Retry on transient errors (ConnectionError, Timeout)
# ---------------------------------------------------------------------------

class TestRetryOnTransientErrors:
    def test_retries_on_connection_error_then_succeeds(self):
        """ConnectionError on first attempt, success on second."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep:
            mock_req.ConnectionError = requests.ConnectionError
            mock_req.Timeout = requests.Timeout
            mock_req.get.side_effect = [requests.ConnectionError("conn refused"), ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=1.0)

        assert result.status_code == 200
        assert mock_req.get.call_count == 2
        assert mock_sleep.call_count == 1

    def test_retries_on_timeout_then_succeeds(self):
        """Timeout on first attempt, success on second."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep:
            mock_req.ConnectionError = requests.ConnectionError
            mock_req.Timeout = requests.Timeout
            mock_req.get.side_effect = [requests.Timeout("read timed out"), ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=1.0)

        assert result.status_code == 200
        assert mock_req.get.call_count == 2
        assert mock_sleep.call_count == 1

    def test_multiple_connection_errors_then_succeeds(self):
        """Multiple ConnectionErrors before a successful response."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.ConnectionError = requests.ConnectionError
            mock_req.Timeout = requests.Timeout
            mock_req.get.side_effect = [
                requests.ConnectionError("fail 1"),
                requests.ConnectionError("fail 2"),
                ok_resp,
            ]
            result = fetch_with_retry("https://api.example.com/data", retries=3, backoff=0.01)

        assert result.status_code == 200
        assert mock_req.get.call_count == 3

    def test_mixed_timeout_and_connection_error(self):
        """Timeout followed by ConnectionError followed by success."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.ConnectionError = requests.ConnectionError
            mock_req.Timeout = requests.Timeout
            mock_req.get.side_effect = [
                requests.Timeout("timeout"),
                requests.ConnectionError("conn error"),
                ok_resp,
            ]
            result = fetch_with_retry("https://api.example.com/data", retries=3, backoff=0.01)

        assert result.status_code == 200
        assert mock_req.get.call_count == 3


# ---------------------------------------------------------------------------
# 3. Retry on 5xx status codes (and 429)
# ---------------------------------------------------------------------------

class TestRetryOnServerErrors:
    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
    def test_retries_on_each_retryable_status(self, status):
        """Each status in RETRYABLE_STATUS should trigger a retry."""
        fail_resp = _mock_response(status)
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.side_effect = [fail_resp, ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=0.01)

        assert result.status_code == 200
        assert mock_req.get.call_count == 2

    def test_retryable_status_set_contains_expected_codes(self):
        """Verify the retryable status set has the expected codes."""
        assert {429, 500, 502, 503, 504} == RETRYABLE_STATUS

    def test_retry_on_500_then_success(self):
        """Internal Server Error retried and succeeds."""
        fail_resp = _mock_response(500)
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.side_effect = [fail_resp, ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=1, backoff=0.01)

        assert result.status_code == 200

    def test_retry_on_502_then_success(self):
        """Bad Gateway retried and succeeds."""
        fail_resp = _mock_response(502)
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.side_effect = [fail_resp, ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=1, backoff=0.01)

        assert result.status_code == 200

    def test_retry_on_429_rate_limit(self):
        """429 Too Many Requests should be retried."""
        fail_resp = _mock_response(429)
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.side_effect = [fail_resp, ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=1, backoff=0.01)

        assert result.status_code == 200

    def test_mixed_retryable_statuses_then_success(self):
        """Different retryable statuses across attempts, then success."""
        resp_503 = _mock_response(503)
        resp_429 = _mock_response(429)
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.side_effect = [resp_503, resp_429, ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=3, backoff=0.01)

        assert result.status_code == 200
        assert mock_req.get.call_count == 3


# ---------------------------------------------------------------------------
# 4. No retry on 4xx status codes
# ---------------------------------------------------------------------------

class TestNoRetryOnClientErrors:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 409, 422])
    def test_no_retry_on_client_error(self, status):
        """Client errors (4xx, except 429) should be returned immediately without retry."""
        mock_resp = _mock_response(status)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data", retries=3)

        assert result.status_code == status
        assert mock_req.get.call_count == 1  # no retry

    def test_400_bad_request_no_retry(self):
        """400 specifically should not be retried."""
        mock_resp = _mock_response(400)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data", retries=5)

        assert result.status_code == 400
        assert mock_req.get.call_count == 1

    def test_401_unauthorized_no_retry(self):
        """401 should not be retried."""
        mock_resp = _mock_response(401)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data", retries=3)

        assert result.status_code == 401
        assert mock_req.get.call_count == 1

    def test_404_not_found_no_retry(self):
        """404 should not be retried."""
        mock_resp = _mock_response(404)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data", retries=3)

        assert result.status_code == 404
        assert mock_req.get.call_count == 1


# ---------------------------------------------------------------------------
# 5. Max retries exhausted
# ---------------------------------------------------------------------------

class TestMaxRetriesExhausted:
    def test_returns_none_when_exceptions_exhaust_retries(self):
        """When all retries fail with ConnectionError, return None."""
        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.ConnectionError = requests.ConnectionError
            mock_req.Timeout = requests.Timeout
            mock_req.get.side_effect = requests.ConnectionError("persistent failure")
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=0.01)

        assert result is None
        # initial attempt + 2 retries = 3 calls
        assert mock_req.get.call_count == 3

    def test_returns_none_when_timeout_exhausts_retries(self):
        """When all retries fail with Timeout, return None."""
        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.ConnectionError = requests.ConnectionError
            mock_req.Timeout = requests.Timeout
            mock_req.get.side_effect = requests.Timeout("timeout")
            result = fetch_with_retry("https://api.example.com/data", retries=3, backoff=0.01)

        assert result is None
        assert mock_req.get.call_count == 4  # initial + 3 retries

    def test_returns_none_when_retryable_status_exhausts_retries(self):
        """When all retries get retryable status, return None (BUG-32 fix)."""
        fail_resp = _mock_response(503)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.return_value = fail_resp
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=0.01)

        assert result is None
        assert mock_req.get.call_count == 3

    def test_returns_none_with_zero_retries_on_exception(self):
        """With retries=0, a single exception returns None immediately."""
        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep:
            mock_req.ConnectionError = requests.ConnectionError
            mock_req.Timeout = requests.Timeout
            mock_req.get.side_effect = requests.ConnectionError("fail")
            result = fetch_with_retry("https://api.example.com/data", retries=0, backoff=0.01)

        assert result is None
        assert mock_req.get.call_count == 1
        assert mock_sleep.call_count == 0  # no sleep with 0 retries

    def test_returns_none_with_zero_retries_on_retryable_status(self):
        """With retries=0, a retryable status code returns None immediately (BUG-32 fix)."""
        fail_resp = _mock_response(500)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep:
            mock_req.get.return_value = fail_resp
            result = fetch_with_retry("https://api.example.com/data", retries=0, backoff=0.01)

        assert result is None
        assert mock_req.get.call_count == 1
        assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# 6. Exponential backoff timing (mock sleep)
# ---------------------------------------------------------------------------

class TestExponentialBackoff:
    def test_backoff_doubles_each_retry(self):
        """With backoff=1.0 and factor=2.0, sleeps should be ~1.0, ~2.0, ~4.0 (plus jitter)."""
        fail_resp = _mock_response(503)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep, \
             patch("portfolio.http_retry.random.uniform", return_value=0):
            mock_req.get.return_value = fail_resp
            fetch_with_retry(
                "https://api.example.com/data",
                retries=3, backoff=1.0, backoff_factor=2.0,
            )

        assert mock_sleep.call_count == 3
        sleep_times = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_times == pytest.approx([1.0, 2.0, 4.0])

    def test_backoff_with_custom_base(self):
        """With backoff=0.5 and factor=2.0, sleeps should be ~0.5, ~1.0, ~2.0."""
        fail_resp = _mock_response(500)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep, \
             patch("portfolio.http_retry.random.uniform", return_value=0):
            mock_req.get.return_value = fail_resp
            fetch_with_retry(
                "https://api.example.com/data",
                retries=3, backoff=0.5, backoff_factor=2.0,
            )

        sleep_times = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_times == pytest.approx([0.5, 1.0, 2.0])

    def test_backoff_with_factor_3(self):
        """With backoff=1.0 and factor=3.0, sleeps should be ~1.0, ~3.0, ~9.0."""
        fail_resp = _mock_response(502)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep, \
             patch("portfolio.http_retry.random.uniform", return_value=0):
            mock_req.get.return_value = fail_resp
            fetch_with_retry(
                "https://api.example.com/data",
                retries=3, backoff=1.0, backoff_factor=3.0,
            )

        sleep_times = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_times == pytest.approx([1.0, 3.0, 9.0])

    def test_backoff_on_connection_error(self):
        """Backoff timing also applies to connection error retries."""
        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep, \
             patch("portfolio.http_retry.random.uniform", return_value=0):
            mock_req.ConnectionError = requests.ConnectionError
            mock_req.Timeout = requests.Timeout
            mock_req.get.side_effect = requests.ConnectionError("fail")
            fetch_with_retry(
                "https://api.example.com/data",
                retries=2, backoff=1.0, backoff_factor=2.0,
            )

        sleep_times = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_times == pytest.approx([1.0, 2.0])

    def test_no_sleep_on_success(self):
        """No sleep should occur when the first attempt succeeds."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep:
            mock_req.get.return_value = ok_resp
            fetch_with_retry("https://api.example.com/data", retries=3, backoff=1.0)

        assert mock_sleep.call_count == 0

    def test_no_sleep_on_non_retryable_error(self):
        """No sleep should occur when a non-retryable status is returned."""
        mock_resp = _mock_response(404)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep:
            mock_req.get.return_value = mock_resp
            fetch_with_retry("https://api.example.com/data", retries=3, backoff=1.0)

        assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# 7. POST method support (json_body parameter)
# ---------------------------------------------------------------------------

class TestPostMethod:
    def test_post_calls_requests_post(self):
        """Method='POST' should use requests.post, not requests.get."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.post.return_value = ok_resp
            result = fetch_with_retry(
                "https://api.example.com/data",
                method="POST",
            )

        assert result.status_code == 200
        mock_req.post.assert_called_once()
        mock_req.get.assert_not_called()

    def test_post_passes_json_body(self):
        """json_body should be passed as the 'json' kwarg to requests.post."""
        ok_resp = _mock_response(200)
        body = {"key": "value", "count": 42}

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.post.return_value = ok_resp
            fetch_with_retry(
                "https://api.example.com/data",
                method="POST",
                json_body=body,
            )

        _, kwargs = mock_req.post.call_args
        assert kwargs["json"] == body

    def test_post_method_case_insensitive(self):
        """Method parameter should be case-insensitive."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.post.return_value = ok_resp
            fetch_with_retry("https://api.example.com/data", method="post")

        mock_req.post.assert_called_once()

    def test_post_with_retry_on_server_error(self):
        """POST requests should also retry on retryable status codes."""
        fail_resp = _mock_response(500)
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.post.side_effect = [fail_resp, ok_resp]
            result = fetch_with_retry(
                "https://api.example.com/data",
                method="POST",
                json_body={"data": True},
                retries=2,
                backoff=0.01,
            )

        assert result.status_code == 200
        assert mock_req.post.call_count == 2


# ---------------------------------------------------------------------------
# 8. Custom headers passed through
# ---------------------------------------------------------------------------

class TestCustomHeaders:
    def test_headers_passed_to_get(self):
        """Custom headers should be forwarded to requests.get."""
        ok_resp = _mock_response(200)
        headers = {"Authorization": "Bearer token123", "Accept": "application/json"}

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = ok_resp
            fetch_with_retry("https://api.example.com/data", headers=headers)

        _, kwargs = mock_req.get.call_args
        assert kwargs["headers"] == headers

    def test_headers_passed_to_post(self):
        """Custom headers should be forwarded to requests.post."""
        ok_resp = _mock_response(200)
        headers = {"X-API-Key": "secret"}

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.post.return_value = ok_resp
            fetch_with_retry(
                "https://api.example.com/data",
                method="POST",
                headers=headers,
            )

        _, kwargs = mock_req.post.call_args
        assert kwargs["headers"] == headers

    def test_none_headers_passed_when_not_specified(self):
        """When no headers are provided, None should be passed."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = ok_resp
            fetch_with_retry("https://api.example.com/data")

        _, kwargs = mock_req.get.call_args
        assert kwargs["headers"] is None


# ---------------------------------------------------------------------------
# 9. Timeout parameter passed through
# ---------------------------------------------------------------------------

class TestTimeoutParameter:
    def test_default_timeout_is_30(self):
        """Default timeout of 30 seconds should be passed to requests."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = ok_resp
            fetch_with_retry("https://api.example.com/data")

        _, kwargs = mock_req.get.call_args
        assert kwargs["timeout"] == 30

    def test_custom_timeout_passed_to_get(self):
        """Custom timeout should be forwarded to requests.get."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = ok_resp
            fetch_with_retry("https://api.example.com/data", timeout=60)

        _, kwargs = mock_req.get.call_args
        assert kwargs["timeout"] == 60

    def test_custom_timeout_passed_to_post(self):
        """Custom timeout should be forwarded to requests.post."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.post.return_value = ok_resp
            fetch_with_retry(
                "https://api.example.com/data",
                method="POST",
                timeout=10,
            )

        _, kwargs = mock_req.post.call_args
        assert kwargs["timeout"] == 10

    def test_timeout_preserved_across_retries(self):
        """The same timeout value should be used on every retry attempt."""
        fail_resp = _mock_response(503)
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.side_effect = [fail_resp, fail_resp, ok_resp]
            fetch_with_retry(
                "https://api.example.com/data",
                retries=3, timeout=45, backoff=0.01,
            )

        for c in mock_req.get.call_args_list:
            assert c.kwargs["timeout"] == 45


# ---------------------------------------------------------------------------
# Additional coverage: custom session, params, other methods, defaults
# ---------------------------------------------------------------------------

class TestCustomSession:
    def test_session_used_instead_of_requests_module(self):
        """When a session is provided, it should be used instead of the requests module."""
        mock_session = MagicMock()
        ok_resp = _mock_response(200)
        mock_session.get.return_value = ok_resp

        result = fetch_with_retry("https://api.example.com/data", session=mock_session)

        assert result.status_code == 200
        mock_session.get.assert_called_once()

    def test_session_post(self):
        """Session should be used for POST requests too."""
        mock_session = MagicMock()
        ok_resp = _mock_response(200)
        mock_session.post.return_value = ok_resp

        result = fetch_with_retry(
            "https://api.example.com/data",
            method="POST",
            json_body={"x": 1},
            session=mock_session,
        )

        assert result.status_code == 200
        mock_session.post.assert_called_once()
        _, kwargs = mock_session.post.call_args
        assert kwargs["json"] == {"x": 1}

    def test_session_retries_on_error(self):
        """Session-based requests should also retry on retryable statuses."""
        mock_session = MagicMock()
        fail_resp = _mock_response(502)
        ok_resp = _mock_response(200)
        mock_session.get.side_effect = [fail_resp, ok_resp]

        with patch("portfolio.http_retry.time.sleep"):
            result = fetch_with_retry(
                "https://api.example.com/data",
                retries=2,
                backoff=0.01,
                session=mock_session,
            )

        assert result.status_code == 200
        assert mock_session.get.call_count == 2


class TestParamsParameter:
    def test_params_passed_to_get(self):
        """Query params should be forwarded to requests.get."""
        ok_resp = _mock_response(200)
        params = {"symbol": "BTCUSDT", "interval": "1h"}

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = ok_resp
            fetch_with_retry("https://api.example.com/data", params=params)

        _, kwargs = mock_req.get.call_args
        assert kwargs["params"] == params

    def test_params_passed_to_post(self):
        """Query params should be forwarded to requests.post."""
        ok_resp = _mock_response(200)
        params = {"action": "submit"}

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.post.return_value = ok_resp
            fetch_with_retry(
                "https://api.example.com/data",
                method="POST",
                params=params,
            )

        _, kwargs = mock_req.post.call_args
        assert kwargs["params"] == params


class TestOtherMethods:
    def test_generic_method_uses_request(self):
        """Non-GET/POST methods should use requests.request()."""
        ok_resp = _mock_response(200)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.request.return_value = ok_resp
            result = fetch_with_retry(
                "https://api.example.com/data",
                method="PUT",
            )

        assert result.status_code == 200
        mock_req.request.assert_called_once()
        args, kwargs = mock_req.request.call_args
        assert args[0] == "PUT"
        assert args[1] == "https://api.example.com/data"

    def test_delete_method(self):
        """DELETE method should use requests.request()."""
        ok_resp = _mock_response(204)

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.request.return_value = ok_resp
            result = fetch_with_retry(
                "https://api.example.com/resource/123",
                method="DELETE",
            )

        assert result.status_code == 204
        args, _ = mock_req.request.call_args
        assert args[0] == "DELETE"


class TestDefaultConstants:
    def test_default_retries(self):
        """DEFAULT_RETRIES should be 3."""
        assert DEFAULT_RETRIES == 3

    def test_default_backoff(self):
        """DEFAULT_BACKOFF should be 1.0."""
        assert DEFAULT_BACKOFF == 1.0

    def test_default_backoff_factor(self):
        """DEFAULT_BACKOFF_FACTOR should be 2.0."""
        assert DEFAULT_BACKOFF_FACTOR == 2.0

    def test_default_retries_used_when_not_specified(self):
        """When retries is not specified, DEFAULT_RETRIES (3) should be used."""
        fail_resp = _mock_response(503)

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.return_value = fail_resp
            fetch_with_retry("https://api.example.com/data")

        # 1 initial + 3 retries = 4 total calls
        assert mock_req.get.call_count == 4


# ---------------------------------------------------------------------------
# fetch_json() tests
# ---------------------------------------------------------------------------

class TestFetchJsonSuccess:
    def test_returns_parsed_json_on_200(self):
        """Successful 200 response returns parsed JSON body."""
        mock_resp = _mock_response(200)
        mock_resp.json.return_value = {"key": "value"}
        mock_resp.raise_for_status.return_value = None

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_json("https://api.example.com/data")

        assert result == {"key": "value"}

    def test_returns_list_json(self):
        """JSON arrays are returned correctly."""
        mock_resp = _mock_response(200)
        mock_resp.json.return_value = [1, 2, 3]
        mock_resp.raise_for_status.return_value = None

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            result = fetch_json("https://api.example.com/data")

        assert result == [1, 2, 3]

    def test_passes_params_and_headers(self):
        """Parameters and headers are forwarded to fetch_with_retry."""
        mock_resp = _mock_response(200)
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None

        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            fetch_json("https://api.example.com/data",
                        headers={"X-Key": "abc"}, params={"q": "1"})

        _, kwargs = mock_req.get.call_args
        assert kwargs["headers"] == {"X-Key": "abc"}
        assert kwargs["params"] == {"q": "1"}


class TestFetchJsonNoneResponse:
    def test_returns_default_on_none_response(self):
        """When fetch_with_retry returns None, return the default value."""
        with patch("portfolio.http_retry.requests") as mock_req:
            mock_req.get.return_value = _mock_response(503)
            mock_req.get.return_value = None
            # Simulate fetch_with_retry returning None
            with patch("portfolio.http_retry.fetch_with_retry", return_value=None):
                result = fetch_json("https://api.example.com/data")

        assert result is None

    def test_returns_custom_default_on_none_response(self):
        """Custom default value is returned when response is None."""
        with patch("portfolio.http_retry.fetch_with_retry", return_value=None):
            result = fetch_json("https://api.example.com/data", default=[])

        assert result == []

    def test_returns_empty_dict_default(self):
        """Empty dict default works."""
        with patch("portfolio.http_retry.fetch_with_retry", return_value=None):
            result = fetch_json("https://api.example.com/data", default={})

        assert result == {}


class TestFetchJsonNon200:
    def test_returns_default_on_non_200_status(self):
        """Non-200 status (raise_for_status raises) returns default."""
        mock_resp = _mock_response(404)
        mock_resp.raise_for_status.side_effect = Exception("404 Client Error")

        with patch("portfolio.http_retry.fetch_with_retry", return_value=mock_resp):
            result = fetch_json("https://api.example.com/data")

        assert result is None

    def test_returns_custom_default_on_non_200(self):
        """Custom default returned on non-200."""
        mock_resp = _mock_response(500)
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")

        with patch("portfolio.http_retry.fetch_with_retry", return_value=mock_resp):
            result = fetch_json("https://api.example.com/data", default="fallback")

        assert result == "fallback"


class TestFetchJsonInvalidBody:
    def test_returns_default_on_invalid_json(self):
        """When .json() raises, return default."""
        mock_resp = _mock_response(200)
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")

        with patch("portfolio.http_retry.fetch_with_retry", return_value=mock_resp):
            result = fetch_json("https://api.example.com/data")

        assert result is None


class TestFetchJsonLogging:
    def test_logs_warning_with_label_on_none_response(self):
        """When label is provided and response is None, a warning is logged."""
        with patch("portfolio.http_retry.fetch_with_retry", return_value=None), \
             patch("portfolio.http_retry.logger") as mock_logger:
            fetch_json("https://api.example.com/data", label="test_api")

        mock_logger.warning.assert_called_once()
        assert "test_api" in mock_logger.warning.call_args[0][1]

    def test_no_warning_without_label_on_none_response(self):
        """When no label and response is None, no warning is logged."""
        with patch("portfolio.http_retry.fetch_with_retry", return_value=None), \
             patch("portfolio.http_retry.logger") as mock_logger:
            fetch_json("https://api.example.com/data")

        mock_logger.warning.assert_not_called()

    def test_logs_warning_with_label_on_http_error(self):
        """When label is provided and raise_for_status fails, a warning is logged."""
        mock_resp = _mock_response(500)
        mock_resp.raise_for_status.side_effect = Exception("500 error")

        with patch("portfolio.http_retry.fetch_with_retry", return_value=mock_resp), \
             patch("portfolio.http_retry.logger") as mock_logger:
            fetch_json("https://api.example.com/data", label="my_svc")

        mock_logger.warning.assert_called_once()
        assert "my_svc" in mock_logger.warning.call_args[0][1]

    def test_no_warning_without_label_on_http_error(self):
        """When no label and raise_for_status fails, no warning is logged."""
        mock_resp = _mock_response(500)
        mock_resp.raise_for_status.side_effect = Exception("500 error")

        with patch("portfolio.http_retry.fetch_with_retry", return_value=mock_resp), \
             patch("portfolio.http_retry.logger") as mock_logger:
            fetch_json("https://api.example.com/data")

        mock_logger.warning.assert_not_called()
