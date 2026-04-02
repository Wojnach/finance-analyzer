"""Live fishing monitor — runs until session end, self-heartbeat, auto-trade.

Usage: .venv/Scripts/python.exe -u scripts/fish_monitor_live.py
"""
import time, json, requests, datetime, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

FAPI = 'https://fapi.binance.com/fapi/v1/ticker/price'
HB = 'data/fish_heartbeat.txt'
LOG = 'data/fish_monitor_live.log'

def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def fp():
    try:
        return float(requests.get(f'{FAPI}?symbol=XAGUSDT', timeout=5).json()['price'])
    except:
        return None

def load():
    try:
        with open('data/agent_summary_compact.json') as f:
            s = json.load(f)
        with open('data/agent_summary.json') as f:
            full = json.load(f)
        xag = (s.get('signals') or {}).get('XAG-USD', {})
        ex = xag.get('extra', {})
        mc = s.get('monte_carlo', {}).get('XAG-USD', {})
        enh = full.get('signals', {}).get('XAG-USD', {}).get('enhanced_signals', {})
        ma = '?'
        try:
            with open('data/metals_signal_log.jsonl') as f:
                lines = f.readlines()
            if lines:
                ma = json.loads(lines[-1]).get('signals', {}).get('XAG-USD', {}).get('action', '?')
        except:
            pass
        return {
            'a': xag.get('action', '?'), 'rsi': float(xag.get('rsi', 50)),
            'b': ex.get('_buy_count', 0), 's': ex.get('_sell_count', 0),
            'mc': float(mc.get('p_up', 0.5)), 'ma': ma,
            'news': (enh.get('news_event') or {}).get('action', 'HOLD'),
            'econ': (enh.get('econ_calendar') or {}).get('action', 'HOLD'),
            'f1d': f"{s.get('focus_probabilities', {}).get('XAG-USD', {}).get('1d', {}).get('direction', '?')} {float(s.get('focus_probabilities', {}).get('XAG-USD', {}).get('1d', {}).get('probability', 0.5)):.0%}",
        }
    except Exception as e:
        log(f'Signal load error: {e}')
        return None

session_pnl = -597
active = None
mch = []
lsc = 0
cd = 0
md = 0

log(f'=== FISH MONITOR LIVE | {session_pnl:+.0f} SEK | until 21:45 ===')

while True:
    now = time.time()
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    h = int(ts[:2])
    m = int(ts[3:5])

    # Self-heartbeat
    try:
        with open(HB, 'w') as f:
            f.write(str(now))
    except:
        pass

    # Session end
    if h >= 21 and m >= 45:
        if active:
            sg = load()
            f1d = sg.get('f1d', '? 50%') if sg else '? 50%'
            hold = 'up' in f1d
            if hold:
                log(f'21:45 {f1d} HOLDING OVERNIGHT')
            else:
                try:
                    from portfolio.avanza_session import place_sell_order, get_quote
                    q = get_quote(active['ob'])
                    bid = float(q.get('buy', 0))
                    if bid > 0:
                        place_sell_order(active['ob'], price=bid, volume=active['v'])
                        pnl = (bid - active['c']) * active['v']
                        session_pnl += pnl
                        log(f'SELL(21:45): P&L:{pnl:+.0f} Sess:{session_pnl:+.0f}')
                except Exception as e:
                    log(f'Sell error: {e}')
        log(f'Session end: {session_pnl:+.0f} SEK')
        break

    p = fp()
    if not p:
        time.sleep(30)
        continue

    if now - lsc >= 60:
        lsc = now
        sg = load()
        if not sg:
            time.sleep(30)
            continue

        mch.append(sg['mc'])
        if len(mch) > 3:
            mch.pop(0)
        m2u = len(mch) >= 2 and all(x > 0.70 for x in mch[-2:])
        m2b = len(mch) >= 2 and all(x < 0.30 for x in mch[-2:])

        if active:
            d = active['d']
            ep = active['e']
            mv = (p - ep) / ep * 100 if d == 'LONG' else (ep - p) / ep * 100
            cm = mv * 5
            cpnl = active['c'] * active['v'] * cm / 100
            mn = int((now - active['t']) / 60)

            if d == 'LONG' and sg['ma'] == 'SELL':
                md += 1
            elif d == 'SHORT' and sg['ma'] == 'BUY':
                md += 1
            else:
                md = 0

            mdf = f' !!MD{md}' if md >= 2 else ''
            nf = f' N={sg["news"]}' if sg['news'] != 'HOLD' else ''
            ef = f' E={sg["econ"]}' if sg['econ'] != 'HOLD' else ''
            log(f'${p:.2f} {d} {mv:+.1f}%/{cm:+.0f}% P&L:{cpnl:+.0f} | {sg["a"]} {sg["b"]}B/{sg["s"]}S RSI={sg["rsi"]:.0f} MC={sg["mc"]:.0%} M:{sg["ma"]} | {sg["f1d"]}{mdf}{nf}{ef} [{mn}m]')

            ex = None
            if d == 'LONG':
                if sg['rsi'] > 62 and sg['mc'] < 0.35: ex = 'COMB'
                elif sg['rsi'] > 70: ex = 'RSI'
                elif sg['s'] > sg['b'] + 4: ex = 'SELL flip'
                elif md >= 3: ex = f'MD{md}'
                elif mv >= 2.0: ex = 'TP'
                elif mv <= -3.0: ex = 'SL'
                elif mn >= 120: ex = '2h'
            else:
                if sg['rsi'] < 30: ex = 'RSI'
                elif sg['b'] > sg['s'] + 4: ex = 'BUY flip'
                elif md >= 3: ex = f'MD{md}'
                elif mv >= 2.0: ex = 'TP'
                elif mv <= -3.0: ex = 'SL'
                elif mn >= 120: ex = '2h'

            if ex:
                try:
                    from portfolio.avanza_session import place_sell_order, get_quote
                    q = get_quote(active['ob'])
                    bid = float(q.get('buy', 0))
                    if bid > 0:
                        place_sell_order(active['ob'], price=bid, volume=active['v'])
                        pnl = (bid - active['c']) * active['v']
                        session_pnl += pnl
                        log(f'SELL({ex}): P&L:{pnl:+.0f} Sess:{session_pnl:+.0f}')
                except Exception as e:
                    log(f'Sell error: {e}')
                active = None
                cd = now + 60
                md = 0
        else:
            bu = sg['a'] == 'BUY' and sg['ma'] == 'BUY' and m2u
            be = sg['a'] == 'SELL' and sg['ma'] == 'SELL' and m2b
            fl = ''
            if now > cd and (bu or be):
                dr = 'LONG' if bu else 'SHORT'
                ob = '1650161' if dr == 'LONG' else '2286417'
                try:
                    from portfolio.avanza_session import place_buy_order, get_quote, get_buying_power
                    q = get_quote(ob)
                    ask = float(q.get('sell', 0))
                    bp = float(get_buying_power().get('buying_power', 0))
                    if ask > 0 and bp > 100:
                        vol = int(min(bp * 0.95, 1500) / ask)
                        if vol >= 5:
                            r = place_buy_order(ob, price=ask, volume=vol)
                            active = {'ob': ob, 'd': dr, 'v': vol, 'c': ask, 'e': p, 't': now}
                            md = 0
                            fl = f' >>> {dr} {vol}u@{ask}'
                            log(f'BUY: {vol}u@{ask} [{r.get("orderRequestStatus", "?")}]')
                except Exception as e:
                    log(f'Buy error: {e}')
            log(f'${p:.2f} | {sg["a"]} {sg["b"]}B/{sg["s"]}S RSI={sg["rsi"]:.0f} MC={sg["mc"]:.0%} M:{sg["ma"]} | {sg["f1d"]}{fl}')
    else:
        if active:
            d = active['d']
            mv = (p - active['e']) / active['e'] * 100 if d == 'LONG' else (active['e'] - p) / active['e'] * 100
            cpnl = active['c'] * active['v'] * mv * 5 / 100
            log(f'${p:.2f} {mv:+.1f}% P&L:{cpnl:+.0f}')
        else:
            log(f'${p:.2f}')

    time.sleep(30)
