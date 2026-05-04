"""Tests for portfolio/signal_state_since.py — per-(ticker, signal) flip times."""

from portfolio.signal_state_since import update_state_since


_T0 = "2026-05-05T18:00:00+00:00"
_T1 = "2026-05-05T18:01:00+00:00"
_T2 = "2026-05-05T18:02:00+00:00"


def test_cold_start_seeds_all_with_now():
    out = update_state_since(None, {"BTC-USD": {"rsi": "BUY", "macd": "HOLD"}}, _T0)
    assert out["updated_at"] == _T0
    assert out["votes"]["BTC-USD"]["rsi"] == {"vote": "BUY", "since": _T0}
    assert out["votes"]["BTC-USD"]["macd"] == {"vote": "HOLD", "since": _T0}


def test_unchanged_vote_preserves_since():
    prev = {
        "updated_at": _T0,
        "votes": {"BTC-USD": {"rsi": {"vote": "BUY", "since": _T0}}},
    }
    out = update_state_since(prev, {"BTC-USD": {"rsi": "BUY"}}, _T1)
    assert out["updated_at"] == _T1
    assert out["votes"]["BTC-USD"]["rsi"] == {"vote": "BUY", "since": _T0}


def test_flipped_vote_resets_since_to_now():
    prev = {
        "updated_at": _T0,
        "votes": {"BTC-USD": {"rsi": {"vote": "BUY", "since": _T0}}},
    }
    out = update_state_since(prev, {"BTC-USD": {"rsi": "SELL"}}, _T1)
    assert out["votes"]["BTC-USD"]["rsi"] == {"vote": "SELL", "since": _T1}


def test_buy_to_hold_to_buy_counts_as_two_flips():
    prev = {
        "updated_at": _T0,
        "votes": {"BTC-USD": {"rsi": {"vote": "BUY", "since": _T0}}},
    }
    after_hold = update_state_since(prev, {"BTC-USD": {"rsi": "HOLD"}}, _T1)
    assert after_hold["votes"]["BTC-USD"]["rsi"] == {"vote": "HOLD", "since": _T1}

    after_buy_again = update_state_since(after_hold, {"BTC-USD": {"rsi": "BUY"}}, _T2)
    assert after_buy_again["votes"]["BTC-USD"]["rsi"] == {"vote": "BUY", "since": _T2}


def test_new_ticker_seeds_with_now():
    prev = {
        "updated_at": _T0,
        "votes": {"BTC-USD": {"rsi": {"vote": "BUY", "since": _T0}}},
    }
    out = update_state_since(
        prev,
        {"BTC-USD": {"rsi": "BUY"}, "ETH-USD": {"rsi": "SELL"}},
        _T1,
    )
    assert out["votes"]["BTC-USD"]["rsi"]["since"] == _T0
    assert out["votes"]["ETH-USD"]["rsi"] == {"vote": "SELL", "since": _T1}


def test_dropped_ticker_is_removed():
    prev = {
        "updated_at": _T0,
        "votes": {
            "BTC-USD": {"rsi": {"vote": "BUY", "since": _T0}},
            "OLD-USD": {"rsi": {"vote": "BUY", "since": _T0}},
        },
    }
    out = update_state_since(prev, {"BTC-USD": {"rsi": "BUY"}}, _T1)
    assert "OLD-USD" not in out["votes"]
    assert out["votes"]["BTC-USD"]["rsi"]["since"] == _T0


def test_new_signal_on_existing_ticker_seeds_with_now():
    prev = {
        "updated_at": _T0,
        "votes": {"BTC-USD": {"rsi": {"vote": "BUY", "since": _T0}}},
    }
    out = update_state_since(
        prev, {"BTC-USD": {"rsi": "BUY", "macd": "SELL"}}, _T1,
    )
    assert out["votes"]["BTC-USD"]["rsi"]["since"] == _T0
    assert out["votes"]["BTC-USD"]["macd"] == {"vote": "SELL", "since": _T1}


def test_lowercase_or_falsy_vote_normalised():
    prev = {
        "updated_at": _T0,
        "votes": {"BTC-USD": {"rsi": {"vote": "BUY", "since": _T0}}},
    }
    out = update_state_since(prev, {"BTC-USD": {"rsi": "buy"}}, _T1)
    assert out["votes"]["BTC-USD"]["rsi"] == {"vote": "BUY", "since": _T0}

    out2 = update_state_since(None, {"BTC-USD": {"rsi": None}}, _T0)
    assert out2["votes"]["BTC-USD"]["rsi"] == {"vote": "HOLD", "since": _T0}


def test_corrupt_prev_treated_as_cold_start():
    for bad_prev in ({}, {"votes": "not-a-dict"}, "string", None, []):
        out = update_state_since(bad_prev, {"BTC-USD": {"rsi": "BUY"}}, _T0)
        assert out["votes"]["BTC-USD"]["rsi"] == {"vote": "BUY", "since": _T0}


def test_non_dict_ticker_value_skipped():
    out = update_state_since(
        None, {"BTC-USD": {"rsi": "BUY"}, "JUNK": "not-a-dict"}, _T0,  # type: ignore[arg-type]
    )
    assert "JUNK" not in out["votes"]
    assert out["votes"]["BTC-USD"]["rsi"]["since"] == _T0


def test_empty_current_votes_returns_empty_payload():
    out = update_state_since(None, {}, _T0)
    assert out == {"updated_at": _T0, "votes": {}}
