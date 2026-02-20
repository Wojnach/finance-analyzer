"""Position sizing using Kelly criterion.

Kelly fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win

Uses signal accuracy as win probability and historical trade outcomes for avg_win/avg_loss.
Returns recommended position size as fraction of portfolio.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PATIENT_FILE = DATA_DIR / "portfolio_state.json"
BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"


def _load_json(path):
    """Load a JSON file, returning empty dict on failure."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {}


def kelly_fraction(win_prob, avg_win_pct, avg_loss_pct):
    """Compute Kelly criterion fraction.

    The Kelly fraction gives the optimal bet size as a fraction of bankroll
    that maximizes long-run geometric growth rate.

    Formula: f* = (p * b - q) / b
    where p = win probability, q = 1 - p, b = avg_win / avg_loss

    Args:
        win_prob: Probability of winning (0.0 to 1.0).
        avg_win_pct: Average win as a positive percentage (e.g. 2.5 for +2.5%).
        avg_loss_pct: Average loss as a positive percentage (e.g. 1.8 for -1.8%).

    Returns:
        float: Kelly fraction (0.0 to 1.0). Clamped to [0, 1].
            Returns 0.0 if inputs are invalid or edge is negative.
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    if avg_win_pct <= 0 or avg_loss_pct <= 0:
        return 0.0

    # b = ratio of avg win to avg loss
    b = avg_win_pct / avg_loss_pct
    q = 1.0 - win_prob

    # Kelly formula: f* = (p * b - q) / b
    kelly = (win_prob * b - q) / b

    # Clamp to [0, 1] — negative Kelly means negative edge, don't bet
    return max(0.0, min(1.0, kelly))


def _compute_trade_stats(transactions, ticker=None):
    """Compute win rate and average win/loss from historical transactions.

    Pairs BUY and SELL transactions to compute realized P&L per round-trip.

    Args:
        transactions: List of transaction dicts from portfolio state.
        ticker: If specified, filter to this ticker only. None = all tickers.

    Returns:
        dict: {win_rate, avg_win_pct, avg_loss_pct, total_trades, wins, losses}
              Returns None if insufficient data (fewer than 2 round-trips).
    """
    # Group transactions by ticker
    from collections import defaultdict
    buys_by_ticker = defaultdict(list)
    sells_by_ticker = defaultdict(list)

    for t in transactions:
        t_ticker = t.get("ticker", "")
        if ticker and t_ticker != ticker:
            continue
        action = t.get("action", "")
        if action == "BUY":
            buys_by_ticker[t_ticker].append(t)
        elif action == "SELL":
            sells_by_ticker[t_ticker].append(t)

    # Compute P&L for each sell vs weighted average buy price
    pnl_list = []
    for t_ticker, sells in sells_by_ticker.items():
        buys = buys_by_ticker.get(t_ticker, [])
        if not buys:
            continue

        # Compute weighted average buy price (in SEK per share)
        total_shares_bought = sum(b.get("shares", 0) for b in buys)
        total_cost = sum(b.get("total_sek", 0) for b in buys)
        if total_shares_bought <= 0:
            continue
        avg_buy_price = total_cost / total_shares_bought

        for sell in sells:
            sell_shares = sell.get("shares", 0)
            sell_total = sell.get("total_sek", 0)
            if sell_shares <= 0:
                continue
            sell_price_per_share = sell_total / sell_shares
            pnl_pct = (sell_price_per_share - avg_buy_price) / avg_buy_price * 100
            pnl_list.append(pnl_pct)

    if len(pnl_list) < 2:
        return None

    wins = [p for p in pnl_list if p > 0]
    losses = [abs(p) for p in pnl_list if p <= 0]

    win_rate = len(wins) / len(pnl_list) if pnl_list else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return {
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "total_trades": len(pnl_list),
        "wins": len(wins),
        "losses": len(losses),
    }


def _get_signal_accuracy(agent_summary, ticker=None):
    """Extract consensus accuracy from agent_summary.

    Uses signal_accuracy_1d.consensus.accuracy as the win probability proxy.
    Falls back to weighted_confidence from the ticker's signal data.

    Args:
        agent_summary: Parsed agent_summary.json dict.
        ticker: Ticker to get accuracy for (used for weighted_confidence fallback).

    Returns:
        float: Estimated win probability (0.0 to 1.0).
    """
    # Primary: use overall consensus accuracy
    acc_data = agent_summary.get("signal_accuracy_1d", {})
    consensus_acc = acc_data.get("consensus", {}).get("accuracy")
    if consensus_acc is not None and consensus_acc > 0:
        return consensus_acc

    # Fallback: use weighted_confidence from the ticker's signals
    if ticker:
        signals = agent_summary.get("signals", {})
        ticker_data = signals.get(ticker, {})
        weighted_conf = ticker_data.get("weighted_confidence")
        if weighted_conf is not None:
            return weighted_conf

    # Last resort: 50/50
    return 0.5


def _get_ticker_signal_accuracy(agent_summary, ticker):
    """Get the accuracy of the most relevant signals for a specific ticker.

    Computes a weighted average of signal accuracies, weighted by each signal's
    normalized weight. Only considers signals that are actively voting (non-HOLD).

    Args:
        agent_summary: Parsed agent_summary.json dict.
        ticker: Ticker symbol.

    Returns:
        float: Weighted signal accuracy (0.0 to 1.0), or None if insufficient data.
    """
    acc_data = agent_summary.get("signal_accuracy_1d", {})
    sig_accuracies = acc_data.get("signals", {})
    sig_weights = agent_summary.get("signal_weights", {})

    signals = agent_summary.get("signals", {})
    ticker_data = signals.get(ticker, {})
    extra = ticker_data.get("extra", {}) if isinstance(ticker_data, dict) else {}
    votes = extra.get("_votes", {})

    if not votes or not sig_accuracies:
        return None

    weighted_sum = 0.0
    weight_total = 0.0

    for sig_name, vote in votes.items():
        if vote == "HOLD":
            continue
        sig_acc = sig_accuracies.get(sig_name, {})
        accuracy = sig_acc.get("accuracy", 0.5)
        samples = sig_acc.get("samples", 0)
        if samples < 5:
            continue  # unreliable

        weight = sig_weights.get(sig_name, {}).get("normalized_weight", 1.0)
        weighted_sum += accuracy * weight
        weight_total += weight

    if weight_total <= 0:
        return None

    return weighted_sum / weight_total


def recommended_size(ticker, portfolio_path=None, agent_summary=None, strategy="patient"):
    """Compute recommended position size using Kelly criterion.

    Combines signal accuracy (as win probability) with historical trade
    performance (avg win/loss) to compute optimal position sizing.

    Args:
        ticker: Ticker symbol to compute sizing for.
        portfolio_path: Path to portfolio state JSON. If None, uses default
            based on strategy.
        agent_summary: Parsed agent_summary dict. If None, loads from file.
        strategy: "patient" or "bold" - determines default portfolio and max alloc.

    Returns:
        dict: {
            kelly_pct: Full Kelly fraction (0-1),
            half_kelly_pct: Half Kelly (more conservative),
            quarter_kelly_pct: Quarter Kelly (most conservative),
            recommended_sek: Recommended trade size in SEK (using half Kelly),
            max_alloc_sek: Maximum allocation per strategy rules,
            win_prob: Estimated win probability used,
            avg_win_pct: Average win percentage used,
            avg_loss_pct: Average loss percentage used,
            source: Description of data source used for estimates,
        }
    """
    # Load portfolio
    if portfolio_path is None:
        portfolio_path = BOLD_FILE if strategy == "bold" else PATIENT_FILE
    portfolio = _load_json(portfolio_path)
    cash_sek = portfolio.get("cash_sek", 0)
    transactions = portfolio.get("transactions", [])

    # Load agent summary
    if agent_summary is None:
        agent_summary = _load_json(AGENT_SUMMARY_FILE)

    # Max allocation per strategy rules
    alloc_frac = 0.30 if strategy == "bold" else 0.15
    max_alloc = cash_sek * alloc_frac

    # Estimate win probability
    # Priority: ticker-specific weighted signal accuracy > consensus accuracy > 50%
    win_prob = _get_ticker_signal_accuracy(agent_summary, ticker)
    source = f"weighted signal accuracy for {ticker}"

    if win_prob is None:
        win_prob = _get_signal_accuracy(agent_summary, ticker)
        source = "consensus accuracy"

    # Estimate avg win/loss from historical trades
    trade_stats = _compute_trade_stats(transactions, ticker=ticker)
    if trade_stats is None:
        # Try all tickers if not enough ticker-specific data
        trade_stats = _compute_trade_stats(transactions, ticker=None)

    if trade_stats and trade_stats["avg_win_pct"] > 0 and trade_stats["avg_loss_pct"] > 0:
        avg_win = trade_stats["avg_win_pct"]
        avg_loss = trade_stats["avg_loss_pct"]
        source += f" + trade history ({trade_stats['total_trades']} trades)"
    else:
        # Default estimates based on typical crypto/stock moves
        # Use ATR from agent summary if available
        signals = agent_summary.get("signals", {})
        ticker_data = signals.get(ticker, {})
        atr_pct = ticker_data.get("atr_pct", 1.5) if isinstance(ticker_data, dict) else 1.5

        # Assume avg win = 1.5x ATR, avg loss = 1x ATR (realistic risk/reward)
        avg_win = atr_pct * 1.5
        avg_loss = atr_pct * 1.0
        source += f" + ATR-based estimates (win={avg_win:.1f}%, loss={avg_loss:.1f}%)"

    # Compute Kelly
    full_kelly = kelly_fraction(win_prob, avg_win, avg_loss)
    half_kelly = full_kelly / 2.0
    quarter_kelly = full_kelly / 4.0

    # Recommended size = half Kelly * cash, capped at max allocation
    rec_sek = min(half_kelly * cash_sek, max_alloc)

    # Minimum trade size check
    if rec_sek < 500:
        rec_sek = 0  # Below minimum trade size

    return {
        "kelly_pct": round(full_kelly, 4),
        "half_kelly_pct": round(half_kelly, 4),
        "quarter_kelly_pct": round(quarter_kelly, 4),
        "recommended_sek": round(rec_sek, 0),
        "max_alloc_sek": round(max_alloc, 0),
        "win_prob": round(win_prob, 4),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "source": source,
    }


def print_sizing_report(tickers=None, strategy="patient"):
    """Print Kelly sizing recommendations for given tickers.

    Args:
        tickers: List of ticker symbols. If None, uses all from agent_summary.
        strategy: "patient" or "bold".
    """
    agent_summary = _load_json(AGENT_SUMMARY_FILE)
    if tickers is None:
        tickers = list(agent_summary.get("signals", {}).keys())

    if not tickers:
        print("No tickers found in agent_summary.json")
        return

    print(f"=== Kelly Sizing Report ({strategy.title()}) ===")
    print()
    print(
        f"{'Ticker':<10} {'Kelly%':>7} {'Half-K%':>8} {'Rec SEK':>10} "
        f"{'Max SEK':>10} {'Win Prob':>9}"
    )
    print(
        f"{'------':<10} {'------':>7} {'-------':>8} {'-------':>10} "
        f"{'-------':>10} {'--------':>9}"
    )

    for ticker in sorted(tickers):
        rec = recommended_size(ticker, agent_summary=agent_summary, strategy=strategy)
        print(
            f"{ticker:<10} {rec['kelly_pct']*100:>6.1f}% {rec['half_kelly_pct']*100:>7.1f}% "
            f"{rec['recommended_sek']:>10,.0f} {rec['max_alloc_sek']:>10,.0f} "
            f"{rec['win_prob']*100:>8.1f}%"
        )

    print()
    print("Note: Recommended size uses Half Kelly (more conservative).")
    print("Full Kelly is optimal but volatile. Quarter Kelly is safest.")


if __name__ == "__main__":
    import sys

    strategy = "bold" if "--bold" in sys.argv else "patient"
    tickers = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not tickers:
        tickers = None
    print_sizing_report(tickers=tickers, strategy=strategy)
