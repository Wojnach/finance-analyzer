"""Tests for shared notification wording helpers."""


def test_format_vote_summary_with_hold_count():
    from portfolio.notification_text import format_vote_summary

    assert format_vote_summary(4, 2, 6) == "4 buy / 2 sell / 6 hold"


def test_humanize_thesis_status():
    from portfolio.notification_text import humanize_thesis_status

    assert humanize_thesis_status("THREATENED") == "Thesis threatened"
    assert humanize_thesis_status("INTACT") == "Thesis intact"


def test_format_portfolio_context():
    from portfolio.notification_text import format_portfolio_context

    result = format_portfolio_context(
        495000,
        -1.2,
        458000,
        -8.4,
        bold_holdings=" · NVDA 50 shares",
        consensus_accuracy=0.48,
    )

    assert "Patient portfolio 495K SEK (-1%)" in result
    assert "Bold portfolio 458K SEK (-8%) · NVDA 50 shares" in result
    assert "Consensus accuracy 48%" in result
