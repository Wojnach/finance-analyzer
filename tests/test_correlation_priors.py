"""Tests for portfolio.correlation_priors — single source of truth for asset correlations."""

from portfolio.correlation_priors import CORRELATION_PRIORS, get_correlated_pairs, get_prior


def test_get_prior_known_pair():
    assert get_prior("BTC-USD", "ETH-USD") == 0.75
    assert get_prior("ETH-USD", "BTC-USD") == 0.75


def test_get_prior_unknown_pair():
    assert get_prior("BTC-USD", "MSTR") == 0.0


def test_get_correlated_pairs_bidirectional():
    pairs = get_correlated_pairs()
    assert "BTC-USD" in pairs["ETH-USD"]
    assert "ETH-USD" in pairs["BTC-USD"]
    assert "XAU-USD" in pairs["XAG-USD"]
    assert "XAG-USD" in pairs["XAU-USD"]


def test_get_correlated_pairs_covers_all_priors():
    pairs = get_correlated_pairs()
    for (a, b) in CORRELATION_PRIORS:
        assert b in pairs[a]
        assert a in pairs[b]


def test_priors_are_positive():
    for pair, value in CORRELATION_PRIORS.items():
        assert 0.0 < value <= 1.0, f"{pair} has invalid correlation {value}"
