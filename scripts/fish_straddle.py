"""Straddle fishing — place limit orders at floor AND ceiling, catch extremes.

No direction prediction needed. Fish both sides. Market tells us which fills.
Best for: chop days, post-crash consolidation, ranging markets.

Strategy:
  1. Compute floor (-X%) and ceiling (+Y%) from current price
  2. Place limit buy BULL at floor, limit buy BEAR at ceiling
  3. When one fills, cancel the other
  4. Hold until SMA20 (mean reversion target) or +1% bounce
  5. If neither fills by 18:55, cancel both (cost: 0 SEK)

Usage:
  .venv/Scripts/python.exe scripts/fish_straddle.py [--floor-pct 5] [--ceil-pct 2]

Start via Windows for autonomous operation:
  powershell Start-Process python -ArgumentList '-u','scripts/fish_straddle.py' -WindowStyle Hidden
"""
import time, json, requests, datetime, sys, os, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

FAPI = 'https://fapi.binance.com/fapi/v1/ticker/price'
LOG = 'data/fish_straddle.log'
BULL_OB = '1650161'  # BULL SILVER X5 AVA 4 (no knockout)
BEAR_OB = '2286417'  # BEAR SILVER X5 AVA 12 (no knockout)
LEVERAGE = 5.0

# Default levels from historical analysis
DEFAULT_FLOOR_PCT = 3.0   # buy BULL when silver dips 3% (fills 40% of days)
DEFAULT_CEIL_PCT = 2.0    # buy BEAR when silver spikes 2% (fills 50% of days)
CANCEL_HOUR, CANCEL_MIN = 18, 55  # cancel unfilled orders
SESSION_END_H, SESSION_END_M = 21, 45


def log_msg(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    try:
        print(line, flush=True)
    except:
        pass
    try:
        with open(LOG, 'a') as f:
            f.write(line + '\n')
    except:
        pass


def fetch_price():
    try:
        return float(requests.get(f'{FAPI}?symbol=XAGUSDT', timeout=5).json()['price'])
    except:
        return None


def get_sma20():
    """Get SMA20 from 15-min bars as mean reversion target."""
    try:
        import yfinance as yf
        import numpy as np
        si = yf.download('SI=F', period='3d', interval='15m', progress=False)
        if si.empty:
            return None
        close = si['Close'].values.flatten()
        return float(np.mean(close[-20:]))
    except:
        return None


def place_limit_buy(ob_id, price_sek, volume):
    """Place limit buy order. Returns order ID or None."""
    try:
        from portfolio.avanza_session import place_buy_order
        r = place_buy_order(ob_id, price=price_sek, volume=volume)
        if r.get('orderRequestStatus') == 'SUCCESS':
            return r.get('orderId')
    except Exception as e:
        log_msg(f'Order error: {e}')
    return None


def cancel_order(order_id):
    """Cancel an open order."""
    try:
        from portfolio.avanza_session import cancel_order as avanza_cancel
        avanza_cancel(str(order_id))
        log_msg(f'Cancelled order {order_id}')
    except Exception as e:
        log_msg(f'Cancel error: {e}')


def check_order_filled(order_id):
    """Check if an order has been filled."""
    try:
        from portfolio.avanza_session import api_get
        orders = api_get('/_api/trading/rest/orders')
        if isinstance(orders, list):
            for o in orders:
                if str(o.get('orderId', '')) == str(order_id):
                    state = o.get('orderState', '')
                    if state in ('ACTIVE', 'NEW', 'PENDING'):
                        return False  # still open
            return True  # not in open orders = filled or cancelled
        # If orders is a string/error, check positions instead
        return None
    except:
        return None


def check_position(ob_id):
    """Check if we hold a position in this instrument."""
    try:
        from portfolio.avanza_session import get_positions
        for p in get_positions():
            if p.get('orderbook_id') == ob_id and p.get('volume', 0) > 0:
                return {
                    'volume': p['volume'],
                    'value': p.get('value', 0),
                    'profit': p.get('profit', 0),
                }
    except:
        pass
    return None


def sell_position(ob_id, volume, reason):
    """Sell a position at market bid."""
    try:
        from portfolio.avanza_session import place_sell_order, get_quote
        q = get_quote(ob_id)
        bid = float(q.get('buy', 0))
        if bid > 0:
            r = place_sell_order(ob_id, price=bid, volume=volume)
            log_msg(f'SELL({reason}): {volume}u@{bid} [{r.get("orderRequestStatus", "?")}]')
            return bid * volume
    except Exception as e:
        log_msg(f'Sell error: {e}')
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Straddle fishing')
    parser.add_argument('--floor-pct', type=float, default=DEFAULT_FLOOR_PCT)
    parser.add_argument('--ceil-pct', type=float, default=DEFAULT_CEIL_PCT)
    parser.add_argument('--budget', type=float, default=0, help='Budget per side (0=auto)')
    args = parser.parse_args()

    floor_pct = args.floor_pct
    ceil_pct = args.ceil_pct

    # Get current price and compute levels
    price = fetch_price()
    if not price:
        log_msg('Cannot fetch price, aborting')
        return

    floor_price = price * (1 - floor_pct / 100)
    ceil_price = price * (1 + ceil_pct / 100)

    log_msg(f'=== STRADDLE FISHING ===')
    log_msg(f'Silver: ${price:.2f}')
    log_msg(f'FLOOR: ${floor_price:.2f} (-{floor_pct}%) -> buy BULL X5')
    log_msg(f'CEIL:  ${ceil_price:.2f} (+{ceil_pct}%) -> buy BEAR X5')
    log_msg(f'Cancel unfilled at {CANCEL_HOUR}:{CANCEL_MIN:02d}')
    log_msg(f'Session end: {SESSION_END_H}:{SESSION_END_M:02d}')

    # Get cert prices and compute volumes
    try:
        from portfolio.avanza_session import get_quote, get_buying_power
        bp = float(get_buying_power().get('buying_power', 0))
        budget = args.budget if args.budget > 0 else bp * 0.45  # 45% per side

        bull_q = get_quote(BULL_OB)
        bear_q = get_quote(BEAR_OB)
        bull_ask = float(bull_q.get('sell', 0))
        bear_ask = float(bear_q.get('sell', 0))

        if bull_ask <= 0 or bear_ask <= 0:
            log_msg('Cannot get cert prices, aborting')
            return

        bull_vol = int(budget / bull_ask)
        bear_vol = int(budget / bear_ask)

        log_msg(f'Budget: {budget:.0f} SEK per side (BP: {bp:.0f})')
        log_msg(f'BULL: {bull_vol}u @ {bull_ask} = {bull_vol*bull_ask:.0f} SEK')
        log_msg(f'BEAR: {bear_vol}u @ {bear_ask} = {bear_vol*bear_ask:.0f} SEK')
    except Exception as e:
        log_msg(f'Setup error: {e}')
        return

    # Place both limit orders
    # Note: we can't place limit orders based on UNDERLYING price on Avanza
    # Avanza limit orders are on the CERT price. So we need to estimate
    # what the cert price would be when underlying hits our level.
    #
    # For daily certs (no barrier), cert price moves ~proportionally:
    # BULL cert at floor = current_cert * (1 - floor_pct * leverage / 100)... approximately
    # But this is imprecise. Better approach: just monitor price and buy at market
    # when underlying hits our level.

    log_msg(f'Monitoring for level hits (market buy when triggered)...')

    bull_filled = False
    bear_filled = False
    active = None  # {'side': 'BULL'/'BEAR', 'ob': ..., 'vol': ..., 'cert': ..., 'entry': ..., 't': ...}
    sma20 = get_sma20()
    last_sma_refresh = time.time()

    while True:
        try:
            dt = datetime.datetime.now()
            h, m = dt.hour, dt.minute
            ts = dt.strftime('%H:%M:%S')

            # Session end
            if h >= SESSION_END_H and m >= SESSION_END_M:
                if active:
                    pos = check_position(active['ob'])
                    if pos:
                        sell_position(active['ob'], pos['volume'], 'SESSION END')
                log_msg('Session end')
                break

            # Cancel time — no new entries
            past_cancel = h > CANCEL_HOUR or (h == CANCEL_HOUR and m >= CANCEL_MIN)

            p = fetch_price()
            if not p:
                time.sleep(30)
                continue

            # Refresh SMA20 every 30 min
            if time.time() - last_sma_refresh > 1800:
                sma20 = get_sma20()
                last_sma_refresh = time.time()

            if active:
                # We're holding a position — manage exit
                d = active['side']
                ep = active['entry']
                if d == 'BULL':
                    mv = (p - ep) / ep * 100
                else:
                    mv = (ep - p) / ep * 100
                cm = mv * LEVERAGE
                cpnl = active['cert'] * active['vol'] * cm / 100

                exit_reason = None
                # Exit at SMA20 (mean reversion target)
                if sma20 and d == 'BULL' and p >= sma20:
                    exit_reason = f'SMA20 ${sma20:.2f}'
                elif sma20 and d == 'BEAR' and p <= sma20:
                    exit_reason = f'SMA20 ${sma20:.2f}'
                # Exit at +1% underlying bounce
                elif mv >= 1.0:
                    exit_reason = f'TP +{mv:.1f}%'
                # Stop loss at -2%
                elif mv <= -2.0:
                    exit_reason = f'SL {mv:.1f}%'
                # Time stop
                elif int((time.time() - active['t']) / 60) >= 120:
                    exit_reason = '2h hold'

                if exit_reason:
                    pos = check_position(active['ob'])
                    if pos:
                        sell_position(active['ob'], pos['volume'], exit_reason)
                    active = None
                    log_msg(f'Exit: {exit_reason} | P&L: {cpnl:+.0f} SEK ({cm:+.1f}%)')
                else:
                    sma_str = f' SMA20=${sma20:.2f}' if sma20 else ''
                    log_msg(f'${p:.2f} {d} {mv:+.1f}%/{cm:+.0f}% P&L:{cpnl:+.0f}{sma_str}')

            else:
                # Check if either level was hit
                floor_str = f'F=${floor_price:.2f}'
                ceil_str = f'C=${ceil_price:.2f}'

                if p <= floor_price and not bull_filled and not past_cancel:
                    # FLOOR HIT — buy BULL at market
                    log_msg(f'!!! FLOOR HIT ${p:.2f} <= ${floor_price:.2f} — buying BULL')
                    try:
                        from portfolio.avanza_session import place_buy_order, get_quote as gq
                        q = gq(BULL_OB)
                        ask = float(q.get('sell', 0))
                        if ask > 0:
                            r = place_buy_order(BULL_OB, price=ask, volume=bull_vol)
                            log_msg(f'BUY BULL: {bull_vol}u@{ask} [{r.get("orderRequestStatus", "?")}]')
                            active = {'side': 'BULL', 'ob': BULL_OB, 'vol': bull_vol, 'cert': ask, 'entry': p, 't': time.time()}
                            bull_filled = True
                    except Exception as e:
                        log_msg(f'Buy error: {e}')

                elif p >= ceil_price and not bear_filled and not past_cancel:
                    # CEILING HIT — buy BEAR at market
                    log_msg(f'!!! CEILING HIT ${p:.2f} >= ${ceil_price:.2f} — buying BEAR')
                    try:
                        from portfolio.avanza_session import place_buy_order, get_quote as gq
                        q = gq(BEAR_OB)
                        ask = float(q.get('sell', 0))
                        if ask > 0:
                            r = place_buy_order(BEAR_OB, price=ask, volume=bear_vol)
                            log_msg(f'BUY BEAR: {bear_vol}u@{ask} [{r.get("orderRequestStatus", "?")}]')
                            active = {'side': 'BEAR', 'ob': BEAR_OB, 'vol': bear_vol, 'cert': ask, 'entry': p, 't': time.time()}
                            bear_filled = True
                    except Exception as e:
                        log_msg(f'Buy error: {e}')

                else:
                    dist_floor = (p - floor_price) / p * 100
                    dist_ceil = (ceil_price - p) / p * 100
                    log_msg(f'${p:.2f} | {floor_str} ({dist_floor:.1f}% away) | {ceil_str} ({dist_ceil:.1f}% away)')

        except Exception as e:
            log_msg(f'ERROR: {e}')
            traceback.print_exc()

        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log_msg(f'FATAL: {e}')
        traceback.print_exc()
