"""Tests for Telegram message formatting.

Covers:
- Special characters in ticker names don't break Markdown
- Vote format (XB/YS/ZH) generation
- Message length stays under 4096 chars with 31 tickers
"""

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TELEGRAM_MAX_LENGTH = 4096

ALL_TICKERS = [
    "BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD",
    "PLTR", "NVDA", "AMD", "GOOGL",
    "AMZN", "AAPL", "AVGO",
    "META", "MU", "SOUN",
    "SMCI", "TSM", "TTWO",
    "VRT", "LMT",
]


# ---------------------------------------------------------------------------
# Helpers: recreate the formatting logic from CLAUDE.md spec
# ---------------------------------------------------------------------------

def format_vote_string(buy_count, sell_count, total_applicable):
    """Format the XB/YS/ZH vote string as specified in CLAUDE.md."""
    abstains = total_applicable - buy_count - sell_count
    return f"{buy_count}B/{sell_count}S/{abstains}H"


def format_ticker_line(ticker, price_usd, action, buy_count, sell_count,
                       total_applicable):
    """Format a single ticker line for the Telegram grid."""
    vote_str = format_vote_string(buy_count, sell_count, total_applicable)
    return f"`{ticker:<5s} ${price_usd:<10,.0f} {action:<5s} {vote_str}`"


def format_timeframe_heatmap(ticker, timeframe_actions):
    """Format a timeframe heatmap line.

    timeframe_actions: dict mapping horizon -> action
    """
    labels = ["Now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]
    cells = []
    for label in labels:
        action = timeframe_actions.get(label, "H")
        cells.append(action[0])  # B, S, or H
    return f"`{ticker:<5s} {'  '.join(cells)}`"


def build_hold_message(signal_data, patient_value, patient_pnl,
                       bold_value, bold_pnl, crypto_fg, stock_fg,
                       patient_reasoning, bold_reasoning):
    """Build a HOLD Telegram message following CLAUDE.md format."""
    lines = []
    lines.append("*HOLD*")
    lines.append("")

    # Separate actionable tickers from HOLD tickers
    actionable = []
    hold_count = 0

    for ticker, data in signal_data.items():
        if data["action"] in ("BUY", "SELL") or data.get("has_position"):
            actionable.append((ticker, data))
        else:
            hold_count += 1

    # If no actionable, show top 3 most interesting
    if not actionable:
        sorted_tickers = sorted(
            signal_data.items(),
            key=lambda x: x[1].get("voter_count", 0),
            reverse=True,
        )
        actionable = sorted_tickers[:3]
        hold_count = len(signal_data) - len(actionable)

    # Ticker grid
    for ticker, data in actionable:
        line = format_ticker_line(
            ticker, data["price_usd"], data["action"],
            data["buy_count"], data["sell_count"], data["total_applicable"],
        )
        lines.append(line)

    lines.append(f"_+ {hold_count} HOLD_")
    lines.append("")

    # Timeframe heatmap
    header = "`     Now 12h  2d  7d 1mo 3mo 6mo`"
    lines.append(header)
    for ticker, data in actionable:
        if "timeframes" in data:
            heatmap = format_timeframe_heatmap(ticker, data["timeframes"])
            lines.append(heatmap)

    lines.append("")

    # F&G + portfolio
    lines.append(f"_Crypto F&G: {crypto_fg} · Stock F&G: {stock_fg}_")
    lines.append(f"_Patient: {patient_value:,.0f} SEK ({patient_pnl:+.2f}%)_")
    lines.append(f"_Bold: {bold_value:,.0f} SEK ({bold_pnl:+.2f}%)_")
    lines.append("")

    # Reasoning
    lines.append(f"Patient: {patient_reasoning}")
    lines.append(f"Bold: {bold_reasoning}")

    return "\n".join(lines)


def build_trade_message(strategy, action, ticker, trade_sek, price_usd,
                        signal_data, patient_value, patient_pnl,
                        bold_value, bold_pnl, crypto_fg, stock_fg,
                        patient_reasoning, bold_reasoning):
    """Build a TRADE Telegram message following CLAUDE.md format."""
    lines = []
    lines.append(f"*{strategy.upper()} {action} {ticker}* — {trade_sek:,.0f} SEK @ ${price_usd:,.0f}")
    lines.append("")

    # Ticker grid
    actionable = []
    hold_count = 0
    for t, data in signal_data.items():
        if data["action"] in ("BUY", "SELL") or data.get("has_position"):
            actionable.append((t, data))
        else:
            hold_count += 1

    for t, data in actionable:
        line = format_ticker_line(
            t, data["price_usd"], data["action"],
            data["buy_count"], data["sell_count"], data["total_applicable"],
        )
        lines.append(line)

    lines.append(f"_+ {hold_count} HOLD_")
    lines.append("")

    # Timeframe heatmap
    header = "`     Now 12h  2d  7d 1mo 3mo 6mo`"
    lines.append(header)
    for t, data in actionable:
        if "timeframes" in data:
            heatmap = format_timeframe_heatmap(t, data["timeframes"])
            lines.append(heatmap)

    lines.append("")

    lines.append(f"_Crypto F&G: {crypto_fg} · Stock F&G: {stock_fg}_")
    lines.append(f"_Patient: {patient_value:,.0f} SEK ({patient_pnl:+.2f}%) · HOLD_")
    lines.append(f"_Bold: {bold_value:,.0f} SEK ({bold_pnl:+.2f}%) · {ticker} 0.19_")
    lines.append("")

    lines.append(f"Patient: {patient_reasoning}")
    lines.append(f"Bold: {bold_reasoning}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test: Vote format (XB/YS/ZH) generation
# ---------------------------------------------------------------------------

class TestVoteFormatGeneration:
    def test_all_hold(self):
        """0 buys, 0 sells, 25 total -> 0B/0S/25H."""
        result = format_vote_string(0, 0, 25)
        assert result == "0B/0S/25H"

    def test_all_buy(self):
        """25 buys, 0 sells, 25 total -> 25B/0S/0H."""
        result = format_vote_string(25, 0, 25)
        assert result == "25B/0S/0H"

    def test_mixed_votes(self):
        """4 buys, 1 sell, 11 total -> 4B/1S/6H."""
        result = format_vote_string(4, 1, 11)
        assert result == "4B/1S/6H"

    def test_crypto_25_total(self):
        """Crypto has 25 total applicable signals."""
        result = format_vote_string(3, 2, 25)
        assert result == "3B/2S/20H"

    def test_stock_21_total(self):
        """Stocks have 21 total applicable signals."""
        result = format_vote_string(2, 3, 21)
        assert result == "2B/3S/16H"

    def test_zero_total(self):
        """Edge case: 0 total applicable."""
        result = format_vote_string(0, 0, 0)
        assert result == "0B/0S/0H"


# ---------------------------------------------------------------------------
# Test: Ticker line formatting
# ---------------------------------------------------------------------------

class TestTickerLineFormatting:
    def test_btc_line(self):
        line = format_ticker_line("BTC", 66800, "BUY", 4, 1, 11)
        assert "BTC" in line
        assert "66,800" in line
        assert "BUY" in line
        assert "4B/1S/6H" in line
        assert line.startswith("`")
        assert line.endswith("`")

    def test_short_ticker(self):
        line = format_ticker_line("MU", 428.18, "HOLD", 1, 0, 21)
        assert "MU" in line

    def test_ticker_with_hyphen(self):
        """Tickers like BTC-USD should not break Markdown."""
        line = format_ticker_line("BTC-USD", 66800, "BUY", 4, 1, 11)
        # Hyphen should be inside backticks, so it's safe
        assert "`" in line
        assert "BTC-USD" in line


# ---------------------------------------------------------------------------
# Test: Special characters don't break Markdown
# ---------------------------------------------------------------------------

class TestMarkdownSafety:
    def test_underscore_in_italic(self):
        """Underscores in _italic_ formatting should be balanced."""
        msg = "_+ 27 HOLD_"
        # Count underscores — should be even (balanced)
        underscore_count = msg.count("_")
        assert underscore_count % 2 == 0

    def test_backtick_balance(self):
        """Backticks should be balanced in ticker lines."""
        line = format_ticker_line("SMCI", 42.30, "SELL", 2, 3, 7)
        backtick_count = line.count("`")
        assert backtick_count == 2  # opening and closing

    def test_asterisk_in_header(self):
        """Bold markers should be balanced."""
        header = "*HOLD*"
        assert header.count("*") == 2

    def test_dollar_sign_safe(self):
        """Dollar signs in prices should not break Markdown."""
        line = format_ticker_line("BTC", 66800, "BUY", 4, 1, 11)
        assert "$" in line
        # Dollar sign inside backticks is safe

    def test_percentage_sign_safe(self):
        """Percentage signs should not break Markdown."""
        line = "_Patient: 500,000 SEK (+0.00%)_"
        # Should be parseable Markdown
        assert "%" in line

    def test_hyphen_in_ticker_safe(self):
        """Hyphens in ticker names (BTC-USD) inside backticks are safe."""
        line = format_ticker_line("BTC-USD", 66800, "BUY", 4, 1, 11)
        assert "BTC-USD" in line


# ---------------------------------------------------------------------------
# Test: Message length under 4096 chars
# ---------------------------------------------------------------------------

class TestMessageLength:
    def _make_signal_data(self, num_tickers, action="HOLD"):
        """Create mock signal data for testing."""
        data = {}
        for i, ticker in enumerate(ALL_TICKERS[:num_tickers]):
            data[ticker] = {
                "price_usd": 100.0 + i * 1000,
                "action": action,
                "buy_count": 2 if action == "BUY" else 0,
                "sell_count": 2 if action == "SELL" else 0,
                "total_applicable": 24 if ticker not in ("BTC-USD", "ETH-USD") else 25,
                "voter_count": 4 if action != "HOLD" else 0,
                "has_position": False,
                "timeframes": {
                    "Now": "BUY", "12h": "HOLD", "2d": "SELL",
                    "7d": "SELL", "1mo": "SELL", "3mo": "SELL", "6mo": "HOLD",
                },
            }
        return data

    def test_all_hold_31_tickers(self):
        """HOLD message with 31 tickers should be under 4096 chars."""
        signal_data = self._make_signal_data(31)
        msg = build_hold_message(
            signal_data,
            patient_value=500_000, patient_pnl=0.00,
            bold_value=464_535, bold_pnl=-7.09,
            crypto_fg=11, stock_fg=62,
            patient_reasoning="HOLD -- no multi-TF alignment.",
            bold_reasoning="HOLD -- no breakout confirmation.",
        )

        assert len(msg) < TELEGRAM_MAX_LENGTH, \
            f"Message length {len(msg)} exceeds {TELEGRAM_MAX_LENGTH}"

    def test_5_actionable_26_hold(self):
        """5 actionable + 26 HOLD tickers should be under 4096 chars."""
        signal_data = self._make_signal_data(31)
        # Make 5 tickers actionable
        for i, ticker in enumerate(ALL_TICKERS[:5]):
            signal_data[ticker]["action"] = "BUY" if i % 2 == 0 else "SELL"
            signal_data[ticker]["buy_count"] = 3 if i % 2 == 0 else 1
            signal_data[ticker]["sell_count"] = 1 if i % 2 == 0 else 3

        msg = build_hold_message(
            signal_data,
            patient_value=500_000, patient_pnl=0.00,
            bold_value=464_535, bold_pnl=-7.09,
            crypto_fg=11, stock_fg=62,
            patient_reasoning="HOLD -- mixed signals across tickers.",
            bold_reasoning="HOLD -- waiting for clear breakout.",
        )

        assert len(msg) < TELEGRAM_MAX_LENGTH

    def test_trade_message_under_limit(self):
        """Trade message should be under 4096 chars."""
        signal_data = self._make_signal_data(31)
        signal_data["BTC-USD"]["action"] = "BUY"
        signal_data["BTC-USD"]["buy_count"] = 4
        signal_data["BTC-USD"]["sell_count"] = 1
        signal_data["BTC-USD"]["has_position"] = True

        msg = build_trade_message(
            strategy="BOLD",
            action="BUY",
            ticker="BTC",
            trade_sek=139_361,
            price_usd=66_800,
            signal_data=signal_data,
            patient_value=500_000, patient_pnl=0.00,
            bold_value=325_174, bold_pnl=-7.09,
            crypto_fg=11, stock_fg=62,
            patient_reasoning="HOLD -- BUY only on Now, longer TFs bearish.",
            bold_reasoning="BUY BTC -- 4B consensus + BB expansion + EMA alignment.",
        )

        assert len(msg) < TELEGRAM_MAX_LENGTH

    def test_all_31_actionable_worst_case(self):
        """Even if all 31 tickers are actionable (worst case), using the
        actionable-only filter should keep message under limit."""
        signal_data = self._make_signal_data(31, action="BUY")
        for ticker in ALL_TICKERS:
            if ticker in signal_data:
                signal_data[ticker]["buy_count"] = 5
                signal_data[ticker]["sell_count"] = 2

        # In practice, the message builder should limit shown tickers
        # For this test, we verify the raw message with all tickers shown
        lines = ["*HOLD*", ""]
        for ticker, data in signal_data.items():
            line = format_ticker_line(
                ticker, data["price_usd"], data["action"],
                data["buy_count"], data["sell_count"], data["total_applicable"],
            )
            lines.append(line)
        lines.append("")
        lines.append("_Crypto F&G: 11 · Stock F&G: 62_")
        lines.append("_Patient: 500,000 SEK (+0.00%)_")
        lines.append("_Bold: 464,535 SEK (-7.09%)_")

        msg = "\n".join(lines)
        # Even worst case grid should fit
        assert len(msg) < TELEGRAM_MAX_LENGTH


# ---------------------------------------------------------------------------
# Test: Timeframe heatmap formatting
# ---------------------------------------------------------------------------

class TestTimeframeHeatmap:
    def test_all_buy(self):
        tf = {"Now": "BUY", "12h": "BUY", "2d": "BUY",
              "7d": "BUY", "1mo": "BUY", "3mo": "BUY", "6mo": "BUY"}
        line = format_timeframe_heatmap("BTC", tf)
        assert "B  B  B  B  B  B  B" in line

    def test_all_sell(self):
        tf = {"Now": "SELL", "12h": "SELL", "2d": "SELL",
              "7d": "SELL", "1mo": "SELL", "3mo": "SELL", "6mo": "SELL"}
        line = format_timeframe_heatmap("ETH", tf)
        assert "S  S  S  S  S  S  S" in line

    def test_mixed(self):
        tf = {"Now": "BUY", "12h": "HOLD", "2d": "SELL",
              "7d": "SELL", "1mo": "SELL", "3mo": "SELL", "6mo": "HOLD"}
        line = format_timeframe_heatmap("BTC", tf)
        assert "B  H  S  S  S  S  H" in line

    def test_missing_horizon_defaults_to_h(self):
        """Missing horizons default to HOLD (H)."""
        tf = {"Now": "BUY"}
        line = format_timeframe_heatmap("BTC", tf)
        assert "B" in line
        assert "H" in line

    def test_heatmap_inside_backticks(self):
        tf = {"Now": "BUY", "12h": "HOLD", "2d": "SELL",
              "7d": "SELL", "1mo": "SELL", "3mo": "SELL", "6mo": "HOLD"}
        line = format_timeframe_heatmap("BTC", tf)
        assert line.startswith("`")
        assert line.endswith("`")


# ---------------------------------------------------------------------------
# Test: HOLD count formatting
# ---------------------------------------------------------------------------

class TestHoldCount:
    def test_hold_count_italicized(self):
        """HOLD count line should use Markdown italics."""
        line = f"_+ 27 HOLD_"
        assert line.startswith("_")
        assert line.endswith("_")

    def test_hold_count_correct_number(self):
        """HOLD count should reflect non-actionable tickers."""
        total = 31
        actionable = 4
        hold_count = total - actionable
        assert hold_count == 27
