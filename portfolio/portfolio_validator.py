"""Portfolio state validation utilities.

Validates portfolio state files for data integrity, ensuring that cash,
holdings, fees, and transaction records are all internally consistent.
"""

import json
import pathlib
from collections import defaultdict


def validate_portfolio(portfolio: dict) -> list[str]:
    """Validate portfolio state integrity.

    Performs comprehensive checks on a portfolio state dict to ensure
    all invariants hold. Returns a list of error messages -- an empty
    list means the portfolio is valid.

    Checks:
    1. Cash is non-negative
    2. All shares are non-negative
    3. Cash reconciliation: initial - sum(BUY allocs) + sum(SELL net_proceeds) = cash_sek
    4. Holdings shares reconciliation: sum(BUY shares) - sum(SELL shares) per ticker
    5. Fee reconciliation: total_fees_sek = sum(all fee_sek in transactions)
    6. No duplicate timestamps in transactions (same ticker + same timestamp)
    7. Transaction field completeness
    8. Holdings avg_cost_usd consistency

    Args:
        portfolio: Full portfolio state dict with keys:
            - cash_sek, holdings, transactions, initial_value_sek, total_fees_sek

    Returns:
        list of error message strings (empty = valid).
    """
    errors = []

    # --- Extract fields with defaults ---
    cash_sek = portfolio.get("cash_sek")
    holdings = portfolio.get("holdings", {})
    transactions = portfolio.get("transactions", [])
    initial_value = portfolio.get("initial_value_sek", 500_000)
    total_fees_sek = portfolio.get("total_fees_sek")

    # --- Check 0: Required fields exist ---
    if cash_sek is None:
        errors.append("Missing required field: cash_sek")
        cash_sek = 0
    if total_fees_sek is None:
        errors.append("Missing or null field: total_fees_sek (should be 0 if no fees)")
        total_fees_sek = 0
    if "initial_value_sek" not in portfolio:
        errors.append("Missing required field: initial_value_sek")

    # --- Check 1: Cash is non-negative ---
    if cash_sek < 0:
        errors.append(f"Cash is negative: {cash_sek:.2f} SEK")

    # --- Check 2: All shares are non-negative ---
    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares < 0:
            errors.append(f"Negative shares for {ticker}: {shares}")
        avg_cost = pos.get("avg_cost_usd")
        if avg_cost is not None and avg_cost < 0:
            errors.append(f"Negative avg_cost_usd for {ticker}: {avg_cost}")

    # --- Check 3: Cash reconciliation ---
    # cash = initial - sum(BUY total_sek) + sum(SELL total_sek)
    # Note: BUY total_sek = full allocation (including fee)
    # SELL total_sek = net proceeds (after fee deducted)
    total_buy_alloc = 0.0
    total_sell_proceeds = 0.0
    for tx in transactions:
        action = tx.get("action", "")
        total_sek = tx.get("total_sek", 0) or 0
        if action == "BUY":
            total_buy_alloc += total_sek
        elif action == "SELL":
            total_sell_proceeds += total_sek

    expected_cash = initial_value - total_buy_alloc + total_sell_proceeds
    cash_diff = abs(expected_cash - cash_sek)
    if cash_diff > 1.0:  # Allow 1 SEK tolerance for floating point
        errors.append(
            f"Cash reconciliation failed: expected {expected_cash:.2f} SEK "
            f"(initial {initial_value} - buys {total_buy_alloc:.2f} + sells {total_sell_proceeds:.2f}), "
            f"got {cash_sek:.2f} SEK (diff: {cash_diff:.2f})"
        )

    # --- Check 4: Holdings shares reconciliation ---
    # For each ticker: net_shares = sum(BUY shares) - sum(SELL shares)
    ticker_bought = defaultdict(float)
    ticker_sold = defaultdict(float)
    for tx in transactions:
        ticker = tx.get("ticker", "")
        action = tx.get("action", "")
        shares = tx.get("shares", 0) or 0
        if action == "BUY":
            ticker_bought[ticker] += shares
        elif action == "SELL":
            ticker_sold[ticker] += shares

    # Check tickers that appear in transactions
    all_tx_tickers = set(ticker_bought.keys()) | set(ticker_sold.keys())
    for ticker in all_tx_tickers:
        expected_shares = ticker_bought[ticker] - ticker_sold[ticker]

        # Get actual shares from holdings
        if ticker in holdings:
            actual_shares = holdings[ticker].get("shares", 0)
        else:
            actual_shares = 0

        # Compare (with tolerance for floating point and rounding from repeated partial sells)
        share_diff = abs(expected_shares - actual_shares)
        if share_diff > 1e-6:
            if expected_shares <= 1e-9 and actual_shares == 0:
                # Both effectively zero -- OK (sold all, removed from holdings)
                continue
            if expected_shares <= 1e-9 and ticker not in holdings:
                # Fully sold, ticker removed from holdings -- OK
                continue
            # Tolerance for small remainders from repeated partial sells (e.g.,
            # multiple 50% sells that don't sum exactly to total bought due to
            # floating-point rounding). Allow up to 1% of total bought shares.
            total_bought = ticker_bought[ticker]
            relative_diff = share_diff / total_bought if total_bought > 0 else float("inf")
            if actual_shares == 0 and ticker not in holdings and relative_diff < 0.01:
                # Small remainder from rounding, ticker removed -- acceptable
                continue
            errors.append(
                f"Holdings mismatch for {ticker}: expected {expected_shares:.8f} shares "
                f"(bought {ticker_bought[ticker]:.8f} - sold {ticker_sold[ticker]:.8f}), "
                f"got {actual_shares:.8f} in holdings (diff: {share_diff:.8f})"
            )

    # Check for holdings tickers not in transactions
    for ticker in holdings:
        shares = holdings[ticker].get("shares", 0)
        if shares > 0 and ticker not in all_tx_tickers:
            errors.append(
                f"Holdings contains {ticker} with {shares} shares but no matching transactions"
            )

    # --- Check 5: Fee reconciliation ---
    computed_fees = 0.0
    tx_with_fees = 0
    tx_without_fees = 0
    for tx in transactions:
        fee = tx.get("fee_sek")
        if fee is not None:
            computed_fees += fee
            tx_with_fees += 1
        else:
            tx_without_fees += 1

    if tx_without_fees > 0 and tx_with_fees > 0:
        errors.append(
            f"Inconsistent fee tracking: {tx_with_fees} transactions have fee_sek, "
            f"{tx_without_fees} do not"
        )

    # Only compare fees if transactions have fee_sek fields
    if tx_with_fees > 0:
        fee_diff = abs(computed_fees - total_fees_sek)
        if fee_diff > 0.01:  # 0.01 SEK tolerance
            errors.append(
                f"Fee reconciliation failed: sum of transaction fees = {computed_fees:.2f} SEK, "
                f"total_fees_sek = {total_fees_sek:.2f} SEK (diff: {fee_diff:.2f})"
            )
    elif len(transactions) > 0 and total_fees_sek == 0:
        # Transactions exist but no fee tracking at all -- warn
        errors.append(
            "No fee_sek fields in any transaction and total_fees_sek is 0, "
            "but transactions exist. Fees may not be tracked."
        )

    # --- Check 6: No duplicate timestamps per ticker ---
    seen_tx = set()
    for i, tx in enumerate(transactions):
        key = (tx.get("ticker", ""), tx.get("timestamp", ""), tx.get("action", ""))
        if key in seen_tx:
            errors.append(
                f"Duplicate transaction at index {i}: {key[2]} {key[0]} at {key[1]}"
            )
        seen_tx.add(key)

    # --- Check 7: Transaction field completeness ---
    required_tx_fields = ["timestamp", "ticker", "action", "shares", "price_usd",
                          "total_sek", "reason"]
    recommended_tx_fields = ["price_sek", "fee_sek", "confidence", "fx_rate"]

    for i, tx in enumerate(transactions):
        for field in required_tx_fields:
            if field not in tx or tx[field] is None:
                errors.append(f"Transaction {i} missing required field: {field}")

        # Validate action value
        action = tx.get("action", "")
        if action not in ("BUY", "SELL"):
            errors.append(f"Transaction {i} has invalid action: '{action}' (expected BUY or SELL)")

        # Validate shares > 0
        shares = tx.get("shares", 0)
        if shares is not None and shares <= 0:
            errors.append(f"Transaction {i} has non-positive shares: {shares}")

        # Validate total_sek > 0
        total = tx.get("total_sek", 0)
        if total is not None and total <= 0:
            errors.append(f"Transaction {i} has non-positive total_sek: {total}")

    # --- Check 8: Holdings avg_cost_usd consistency ---
    # For tickers with multiple BUY transactions, verify avg_cost is plausible
    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue
        avg_cost = pos.get("avg_cost_usd")
        if avg_cost is None:
            errors.append(f"Holdings {ticker} missing avg_cost_usd")
            continue

        # Compute weighted average from BUY transactions
        total_cost = 0.0
        total_bought = 0.0
        for tx in transactions:
            if tx.get("ticker") != ticker or tx.get("action") != "BUY":
                continue
            tx_shares = tx.get("shares", 0) or 0
            tx_price = tx.get("price_usd", 0) or 0
            total_cost += tx_shares * tx_price
            total_bought += tx_shares

        if total_bought > 0:
            expected_avg = total_cost / total_bought
            avg_diff_pct = abs(expected_avg - avg_cost) / expected_avg * 100 if expected_avg > 0 else 0
            if avg_diff_pct > 1.0:  # More than 1% off
                errors.append(
                    f"Holdings {ticker} avg_cost_usd ({avg_cost:.4f}) differs from "
                    f"computed weighted average ({expected_avg:.4f}) by {avg_diff_pct:.2f}%"
                )

    return errors


def validate_portfolio_file(path: str) -> list[str]:
    """Validate a portfolio state JSON file.

    Convenience wrapper that loads the file and runs validate_portfolio().

    Args:
        path: Path to the portfolio state JSON file.

    Returns:
        list of error message strings (empty = valid).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            portfolio = json.load(f)
    except FileNotFoundError:
        return [f"Portfolio file not found: {path}"]
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in {path}: {e}"]

    return validate_portfolio(portfolio)


def validate_all() -> dict[str, list[str]]:
    """Validate both patient and bold portfolio files.

    Returns:
        dict with keys "patient" and "bold", each containing a list of errors.
    """
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    return {
        "patient": validate_portfolio_file(str(data_dir / "portfolio_state.json")),
        "bold": validate_portfolio_file(str(data_dir / "portfolio_state_bold.json")),
    }


if __name__ == "__main__":
    results = validate_all()
    for strategy, errs in results.items():
        print(f"\n{'='*60}")
        print(f"  {strategy.upper()} PORTFOLIO VALIDATION")
        print(f"{'='*60}")
        if errs:
            for e in errs:
                print(f"  ERROR: {e}")
        else:
            print("  VALID - all checks passed")
    print()
