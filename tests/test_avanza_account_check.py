"""Tests for portfolio.avanza_account_check.verify_default_account
plus the related ALLOWED_ACCOUNT_IDS / DEFAULT_ACCOUNT_ID invariant."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from portfolio import avanza_account_check as ac


@pytest.fixture(autouse=True)
def _reset_cache():
    ac.reset_cache()
    yield
    ac.reset_cache()


@pytest.fixture
def critical_errors_path(tmp_path, monkeypatch):
    path = tmp_path / "critical_errors.jsonl"
    monkeypatch.setattr(ac, "CRITICAL_ERRORS_LOG", str(path))
    return path


# ---------------------------------------------------------------------------
# Category match helpers
# ---------------------------------------------------------------------------


class TestCategoryDisallowed:
    @pytest.mark.parametrize("label", [
        "INVESTERINGSSPARKONTO",
        "Investeringssparkonto",
        "ISK",
        "Kapitalförsäkring",
        "kapitalforsakring",
        "Tjänstepension",
        "PENSION",
    ])
    def test_disallowed_labels(self, label):
        assert ac._category_disallowed(label) is True

    @pytest.mark.parametrize("label", [
        "AKTIE_DEPÅ", "Aktiedepå", "AF", "Depå",
        "Equity Account", "",
    ])
    def test_allowed_labels(self, label):
        assert ac._category_disallowed(label) is False


# ---------------------------------------------------------------------------
# verify_default_account — happy path
# ---------------------------------------------------------------------------


class TestVerifyOk:
    def test_trading_category_returns_ok(self, critical_errors_path):
        response = {
            "categories": [
                {"name": "AKTIE_DEPÅ", "accounts": [
                    {"id": "9999999", "type": "DEPÅ"},
                ]},
            ],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            result = ac.verify_default_account("9999999")
        assert result["ok"] is True
        assert "AKTIE_DEPÅ" in result["category"]
        assert not critical_errors_path.exists()

    def test_cache_reused_on_second_call(self, critical_errors_path):
        response = {"accounts": [{"id": "9999999", "type": "Aktiedepå"}]}
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response) as mock_api:
            r1 = ac.verify_default_account("9999999")
            r2 = ac.verify_default_account("9999999")
        assert r1["ok"] and r2["ok"]
        mock_api.assert_called_once()

    def test_cache_bypassed_on_different_account(self, critical_errors_path):
        response_a = {"accounts": [{"id": "111", "type": "Aktiedepå"}]}
        response_b = {"accounts": [{"id": "222", "type": "Aktiedepå"}]}
        with patch.object(ac, "_api_get_categorized_accounts",
                          side_effect=[response_a, response_b]) as mock_api:
            ac.verify_default_account("111")
            ac.verify_default_account("222")
        assert mock_api.call_count == 2

    def test_legacy_categorized_shape(self, critical_errors_path):
        response = {
            "categorizedAccounts": [
                {"name": "AKTIE_DEPÅ", "accounts": [
                    {"accountId": "111", "type": "Depå"},
                ]},
            ],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            result = ac.verify_default_account("111")
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Disallowed category — raises by default
# ---------------------------------------------------------------------------


class TestVerifyDisallowed:
    def test_isk_category_raises(self, critical_errors_path):
        response = {
            "categories": [
                {"name": "INVESTERINGSSPARKONTO", "accounts": [
                    {"id": "1625505", "type": "ISK"},
                ]},
            ],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            with patch.object(ac, "_send_telegram"):
                with pytest.raises(ac.AccountCategoryMismatch, match="1625505"):
                    ac.verify_default_account("1625505")
        # critical_errors entry was appended
        assert critical_errors_path.exists()
        entry = json.loads(critical_errors_path.read_text().strip())
        assert entry["category"] == "avanza_account_mismatch"
        assert entry["context"]["account_id"] == "1625505"
        assert "investerings" in entry["context"]["category_label"].lower()

    def test_pension_category_raises(self, critical_errors_path):
        response = {
            "accounts": [{"id": "999", "type": "Tjänstepension"}],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            with patch.object(ac, "_send_telegram"):
                with pytest.raises(ac.AccountCategoryMismatch):
                    ac.verify_default_account("999")

    def test_raise_disabled_returns_dict(self, critical_errors_path):
        response = {
            "categories": [
                {"name": "ISK", "accounts": [{"id": "111"}]},
            ],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            with patch.object(ac, "_send_telegram"):
                result = ac.verify_default_account(
                    "111", raise_on_mismatch=False,
                )
        assert result["ok"] is False
        assert result["reason"] == "disallowed_category"

    def test_skip_env_var_disables_raise(self, critical_errors_path,
                                         monkeypatch):
        monkeypatch.setenv(ac.SKIP_ENV_VAR, "1")
        response = {
            "categories": [
                {"name": "INVESTERINGSSPARKONTO",
                 "accounts": [{"id": "111"}]},
            ],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            with patch.object(ac, "_send_telegram"):
                # Must NOT raise even though raise_on_mismatch defaults to True
                result = ac.verify_default_account("111")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Account not found
# ---------------------------------------------------------------------------


class TestVerifyNotFound:
    def test_unknown_id_raises(self, critical_errors_path):
        response = {
            "accounts": [{"id": "999", "type": "Aktiedepå"}],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            with patch.object(ac, "_send_telegram"):
                with pytest.raises(ac.AccountCategoryMismatch,
                                    match="not present"):
                    ac.verify_default_account("1234")

    def test_unknown_id_logs_seen_ids(self, critical_errors_path):
        response = {
            "accounts": [
                {"id": "111", "type": "Aktiedepå"},
                {"id": "222", "type": "Aktiedepå"},
            ],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            with patch.object(ac, "_send_telegram"):
                result = ac.verify_default_account(
                    "999", raise_on_mismatch=False,
                )
        assert result["reason"] == "account_not_found"
        assert set(result["seen_ids"]) == {"111", "222"}


# ---------------------------------------------------------------------------
# Fetch failure — fail closed
# ---------------------------------------------------------------------------


class TestFetchFailure:
    def test_fetch_failure_does_not_raise(self, critical_errors_path):
        # Codex P2 fix 2026-05-11: transient categorizedAccounts outage
        # downgrades to a warning instead of permanently bricking the
        # caller. Positive mismatches still raise.
        with patch.object(ac, "_api_get_categorized_accounts",
                          side_effect=RuntimeError("dns down")):
            with patch.object(ac, "_send_telegram"):
                result = ac.verify_default_account("1625505")
        assert result["ok"] is False
        assert result["reason"].startswith("fetch_failed")
        # critical_errors entry still gets appended so the outage is surfaced.
        entry = json.loads(critical_errors_path.read_text().strip())
        assert entry["context"]["reason"].startswith("fetch_failed")

    def test_fetch_failure_with_skip_env(self, critical_errors_path,
                                          monkeypatch):
        # Even with skip env, fetch_failed returns a non-OK dict — no raise.
        monkeypatch.setenv(ac.SKIP_ENV_VAR, "1")
        with patch.object(ac, "_api_get_categorized_accounts",
                          side_effect=RuntimeError("dns down")):
            with patch.object(ac, "_send_telegram"):
                result = ac.verify_default_account("1625505")
        assert result["ok"] is False
        assert result["reason"].startswith("fetch_failed")

    def test_fetch_failure_not_cached(self, critical_errors_path):
        # A transient failure must not poison the process cache — the
        # next call should retry the API rather than return the stale
        # failure.
        call_count = {"n": 0}

        def maybe_fail(*_a, **_kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("dns down")
            return {"accounts": [{"id": "1625505", "type": "Aktiedepå"}]}

        with patch.object(ac, "_api_get_categorized_accounts",
                          side_effect=maybe_fail):
            with patch.object(ac, "_send_telegram"):
                r1 = ac.verify_default_account("1625505")
                r2 = ac.verify_default_account("1625505")
        assert r1["ok"] is False
        assert r2["ok"] is True
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Defensive shape handling
# ---------------------------------------------------------------------------


class TestAllowedAccountIdsInvariant:
    """Codex P1 fix 2026-05-11: ALLOWED_ACCOUNT_IDS must derive from
    DEFAULT_ACCOUNT_ID so an operator changing the default to fix the
    ISK-mismatch issue does not also have to remember to update the H7
    order-placement whitelist."""

    def test_default_id_is_in_whitelist(self):
        from portfolio import avanza_session
        assert avanza_session.DEFAULT_ACCOUNT_ID in (
            avanza_session.ALLOWED_ACCOUNT_IDS
        )


class TestShapeHandling:
    def test_empty_response_treated_as_not_found(self, critical_errors_path):
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value={}):
            with patch.object(ac, "_send_telegram"):
                result = ac.verify_default_account(
                    "111", raise_on_mismatch=False,
                )
        assert result["reason"] == "account_not_found"

    def test_non_dict_response_treated_as_not_found(self, critical_errors_path):
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=["nonsense"]):
            with patch.object(ac, "_send_telegram"):
                result = ac.verify_default_account(
                    "111", raise_on_mismatch=False,
                )
        assert result["reason"] == "account_not_found"

    def test_accountnumber_field_resolves(self, critical_errors_path):
        response = {
            "accounts": [
                {"accountNumber": "555", "type": "Aktiedepå"},
            ],
        }
        with patch.object(ac, "_api_get_categorized_accounts",
                          return_value=response):
            result = ac.verify_default_account("555")
        assert result["ok"] is True
