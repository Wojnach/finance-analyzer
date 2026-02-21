"""Tests for HTTP retry module.

Covers:
- Successful request on first try
- Retry on retryable status codes (429, 500, 502, 503, 504)
- Retry on connection errors
- Retry on timeout errors
- Returns None after all retries exhausted
- Non-retryable status codes (400, 401, 404) returned immediately
- Backoff timing (exponential)
- POST method support
- Custom session support
"""

import pytest
from unittest.mock import patch, MagicMock, call
import requests

from portfolio.http_retry import fetch_with_retry, RETRYABLE_STATUS


class TestFetchWithRetrySuccess:
    def test_returns_response_on_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("portfolio.http_retry.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data")

        assert result is not None
        assert result.status_code == 200

    def test_non_retryable_error_returned_immediately(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("portfolio.http_retry.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data", retries=3)

        assert result.status_code == 404
        assert mock_requests.get.call_count == 1  # no retries


class TestFetchWithRetryRetries:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retries_on_retryable_status(self, status):
        fail_resp = MagicMock()
        fail_resp.status_code = status
        ok_resp = MagicMock()
        ok_resp.status_code = 200

        with patch("portfolio.http_retry.requests") as mock_requests, \
             patch("portfolio.http_retry.time.sleep"):
            mock_requests.get.side_effect = [fail_resp, ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=0.01)

        assert result.status_code == 200
        assert mock_requests.get.call_count == 2

    def test_retries_on_connection_error(self):
        ok_resp = MagicMock()
        ok_resp.status_code = 200

        with patch("portfolio.http_retry.requests") as mock_requests, \
             patch("portfolio.http_retry.time.sleep"):
            mock_requests.ConnectionError = requests.ConnectionError
            mock_requests.Timeout = requests.Timeout
            mock_requests.get.side_effect = [requests.ConnectionError("fail"), ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=0.01)

        assert result.status_code == 200

    def test_retries_on_timeout(self):
        ok_resp = MagicMock()
        ok_resp.status_code = 200

        with patch("portfolio.http_retry.requests") as mock_requests, \
             patch("portfolio.http_retry.time.sleep"):
            mock_requests.ConnectionError = requests.ConnectionError
            mock_requests.Timeout = requests.Timeout
            mock_requests.get.side_effect = [requests.Timeout("timeout"), ok_resp]
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=0.01)

        assert result.status_code == 200

    def test_returns_none_after_all_retries_on_exception(self):
        with patch("portfolio.http_retry.requests") as mock_requests, \
             patch("portfolio.http_retry.time.sleep"):
            mock_requests.ConnectionError = requests.ConnectionError
            mock_requests.Timeout = requests.Timeout
            mock_requests.get.side_effect = requests.ConnectionError("persistent failure")
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=0.01)

        assert result is None
        assert mock_requests.get.call_count == 3  # initial + 2 retries

    def test_returns_last_response_after_all_retries_on_status(self):
        fail_resp = MagicMock()
        fail_resp.status_code = 503

        with patch("portfolio.http_retry.requests") as mock_requests, \
             patch("portfolio.http_retry.time.sleep"):
            mock_requests.get.return_value = fail_resp
            result = fetch_with_retry("https://api.example.com/data", retries=2, backoff=0.01)

        assert result.status_code == 503
        assert mock_requests.get.call_count == 3


class TestFetchWithRetryBackoff:
    def test_exponential_backoff_timing(self):
        fail_resp = MagicMock()
        fail_resp.status_code = 503

        with patch("portfolio.http_retry.requests") as mock_requests, \
             patch("portfolio.http_retry.time.sleep") as mock_sleep:
            mock_requests.get.return_value = fail_resp
            fetch_with_retry("https://api.example.com/data", retries=3, backoff=1.0, backoff_factor=2.0)

        # Should sleep with exponential delays: 1.0, 2.0, 4.0
        assert mock_sleep.call_count == 3
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == pytest.approx([1.0, 2.0, 4.0])


class TestFetchWithRetryMethods:
    def test_post_method(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("portfolio.http_retry.requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = fetch_with_retry("https://api.example.com/data",
                                       method="POST", json_body={"key": "value"})

        assert result.status_code == 200
        mock_requests.post.assert_called_once()

    def test_custom_session(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.get.return_value = mock_resp

        result = fetch_with_retry("https://api.example.com/data", session=mock_session)

        assert result.status_code == 200
        mock_session.get.assert_called_once()
