"""CF-Access JWT verification tests.

Verifies that ``dashboard.cf_access.verify_cf_jwt`` is strict — it
returns None on every failure path, so callers fail closed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import dashboard.cf_access as cfa


@pytest.fixture
def rsa_keypair():
    """Generate a throwaway RSA keypair so tests sign real JWTs.

    Using a real keypair (rather than HMAC) ensures the test exercises
    the same RS256 path that production CF Access uses; an HMAC stub
    would silently miss algorithm-confusion bugs.
    """
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, pub_pem


def _make_jwt(priv_key, *, aud, email, iss="https://t.example/cdn-cgi/access",
              exp_offset=600):
    import time
    now = int(time.time())
    payload = {
        "aud": aud,
        "iss": iss,
        "email": email,
        "iat": now,
        "exp": now + exp_offset,
    }
    return jwt.encode(payload, priv_key, algorithm="RS256")


@pytest.fixture(autouse=True)
def _clear_jwks_cache():
    """Each test starts with a fresh JWKS cache."""
    cfa._JWKS_CLIENT_CACHE.clear()
    yield
    cfa._JWKS_CLIENT_CACHE.clear()


def _patch_jwks_client(public_key_pem: bytes):
    """Return a context that makes _get_jwks_client return a mock whose
    get_signing_key_from_jwt() yields our test public key."""
    mock_signing_key = MagicMock()
    # PyJWT's signing_key.key just needs to be a valid public key object
    # for jwt.decode; pass the PEM bytes through serialization.
    mock_signing_key.key = serialization.load_pem_public_key(public_key_pem)
    mock_client = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
    return patch.object(cfa, "_get_jwks_client", return_value=mock_client)


class TestConfigGating:
    def test_returns_none_when_team_domain_missing(self):
        result = cfa.verify_cf_jwt("anything", "user@x.com",
                                    team_domain=None, aud_tag="aud123")
        assert result is None

    def test_returns_none_when_aud_missing(self):
        result = cfa.verify_cf_jwt("anything", "user@x.com",
                                    team_domain="t.cloudflareaccess.com",
                                    aud_tag=None)
        assert result is None

    def test_returns_none_when_token_empty(self):
        result = cfa.verify_cf_jwt("", "user@x.com",
                                    team_domain="t.cloudflareaccess.com",
                                    aud_tag="aud123")
        assert result is None

    def test_returns_none_when_email_empty(self):
        result = cfa.verify_cf_jwt("xxx", "",
                                    team_domain="t.cloudflareaccess.com",
                                    aud_tag="aud123")
        assert result is None


class TestSignatureVerification:
    def test_valid_jwt_passes(self, rsa_keypair):
        priv, pub = rsa_keypair
        token = _make_jwt(priv, aud="aud123", email="user@x.com")
        with _patch_jwks_client(pub):
            claims = cfa.verify_cf_jwt(
                token, "user@x.com",
                team_domain="t.cloudflareaccess.com",
                aud_tag="aud123",
            )
        assert claims is not None
        assert claims["email"] == "user@x.com"
        assert claims["aud"] == "aud123"

    def test_bad_signature_fails(self, rsa_keypair):
        """Token signed with a different key must NOT verify against
        the public key returned by the JWKs client."""
        priv, _ = rsa_keypair
        # Make a SECOND keypair; we'll sign with priv1 and try to verify
        # against pub2.
        other_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        other_pub = other_priv.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        token = _make_jwt(priv, aud="aud123", email="user@x.com")
        with _patch_jwks_client(other_pub):
            claims = cfa.verify_cf_jwt(
                token, "user@x.com",
                team_domain="t.cloudflareaccess.com",
                aud_tag="aud123",
            )
        assert claims is None

    def test_wrong_audience_fails(self, rsa_keypair):
        priv, pub = rsa_keypair
        token = _make_jwt(priv, aud="WRONG", email="user@x.com")
        with _patch_jwks_client(pub):
            claims = cfa.verify_cf_jwt(
                token, "user@x.com",
                team_domain="t.cloudflareaccess.com",
                aud_tag="aud123",
            )
        assert claims is None

    def test_expired_jwt_fails(self, rsa_keypair):
        priv, pub = rsa_keypair
        token = _make_jwt(priv, aud="aud123", email="user@x.com",
                          exp_offset=-1)
        with _patch_jwks_client(pub):
            claims = cfa.verify_cf_jwt(
                token, "user@x.com",
                team_domain="t.cloudflareaccess.com",
                aud_tag="aud123",
            )
        assert claims is None

    def test_email_header_claim_mismatch_fails(self, rsa_keypair):
        """An attacker who somehow obtained a valid JWT for user A
        must not be able to replay it with a header claiming user B.
        This is the impersonation-replay defense."""
        priv, pub = rsa_keypair
        token = _make_jwt(priv, aud="aud123", email="alice@x.com")
        with _patch_jwks_client(pub):
            claims = cfa.verify_cf_jwt(
                token, "bob@x.com",
                team_domain="t.cloudflareaccess.com",
                aud_tag="aud123",
            )
        assert claims is None


class TestIsConfigured:
    def test_both_set(self):
        assert cfa.is_cf_access_configured({
            "cf_access_team_domain": "t.cloudflareaccess.com",
            "cf_access_aud_tag": "aud123",
        })

    def test_one_missing(self):
        assert not cfa.is_cf_access_configured({
            "cf_access_team_domain": "t.cloudflareaccess.com",
        })
        assert not cfa.is_cf_access_configured({
            "cf_access_aud_tag": "aud123",
        })

    def test_empty_strings(self):
        assert not cfa.is_cf_access_configured({
            "cf_access_team_domain": "  ",
            "cf_access_aud_tag": "",
        })
