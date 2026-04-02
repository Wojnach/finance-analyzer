"""Live fishing monitor — autonomous, survives disconnects, trades until 21:45.

Fully self-contained. Detects existing positions on startup. Never crashes
(triple try/except). Writes to log file. Self-heartbeat. Auto-sells at 21:45
unless signals say hold overnight.

Start:  powershell Start-Process python -ArgumentList '-u','scripts/fish_monitor_live.py' -WindowStyle Hidden
Check:  tail data/fish_monitor_live.log
Kill:   powershell "Get-Process python* | ? {(Get-CimInstance Win32_Process -Filter 'ProcessId=$($_.Id)').CommandLine -match 'fish_monitor_live'} | Stop-Process -Force"
"""
import time, json, requests, datetime, sys, os, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

FAPI = 'https://fapi.binance.com/fapi/v1/ticker/price'
HB = 'data/fish_heartbeat.txt'
LOG = 'data/fish_monitor_live.log'
SESSION_END_H, SESSION_END_M = 21, 45
SILVER_KEYWORDS = ['BULL SILVER', 'BEAR SILVER', 'TURBO L SILVER', 'TURBO S SILVER', 'MINI S SILVER', 'MINI L SILVER']

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

def heartbeat():
    try:
        with open(HB, 'w') as f:
            f.write(str(time.time()))
    except:
        pass

def fetch_price():
    try:
        return float(requests.get(f'{FAPI}?symbol=XAGUSDT', timeout=5).json()['price'])
    except:
        return None

def load_signals():
    try:
        with open('data/agent_summary_compact.json') as f:
            s = json.load(f)
        with open('data/agent_summary.json') as f:
            full = json.load(f)
        xag = (s.get('signals') or {}).get('XAG-USD', {})
        ex = xag.get('extra', {})
        mc = s.get('monte_carlo', {}).get('XAG-USD', {})
        focus = s.get('focus_probabilities', {}).get('XAG-USD', {})
        enh = full.get('signals', {}).get('XAG-USD', {}).get('enhanced_signals', {})
        ma = '?'
        try:
            with open('data/metals_signal_log.jsonl') as f:
                lines = f.readlines()
            if lines:
                ma = json.loads(lines[-1]).get('signals', {}).get('XAG-USD', {}).get('action', '?')
        except:
            pass
        f1d_data = focus.get('1d', {})
        f1d_dir = f1d_data.get('direction', '?')
        f1d_prob = float(f1d_data.get('probability', 0.5))
        # News/event details (Gap 1+2 fix)
        news_sig = enh.get('news_event') or {}
        econ_sig = enh.get('econ_calendar') or {}
        news_ind = news_sig.get('indicators', {})
        econ_ind = econ_sig.get('indicators', {})

        # Extract event proximity and type
        event_hours = econ_ind.get('proximity_hours_until', 999)
        event_name = econ_ind.get('risk_nearest_event', econ_ind.get('proximity_next_event', ''))
        event_impact = econ_ind.get('type_event_impact', '')
        high_impact_near = econ_ind.get('risk_high_impact_within_4h', False)

        # Extract news severity
        news_severity = news_ind.get('severity_max_severity', '')
        news_keywords = news_ind.get('severity_keywords_found', [])
        news_articles = news_ind.get('total_headlines', news_ind.get('velocity_article_count', 0))
        news_velocity_base = news_ind.get('velocity_baseline', 0)

        out = {
            'a': xag.get('action', '?'), 'rsi': float(xag.get('rsi', 50)),
            'b': ex.get('_buy_count', 0), 's': ex.get('_sell_count', 0),
            'mc': float(mc.get('p_up', 0.5)), 'ma': ma,
            'news': news_sig.get('action', 'HOLD'),
            'econ': econ_sig.get('action', 'HOLD'),
            'f1d': f'{f1d_dir} {f1d_prob:.0%}', 'f1d_dir': f1d_dir, 'f1d_prob': f1d_prob,
            # Event details
            'event_hours': event_hours if isinstance(event_hours, (int, float)) else 999,
            'event_name': str(event_name)[:30] if event_name else '',
            'event_impact': str(event_impact),
            'high_impact_near': bool(high_impact_near),
            # News details
            'news_severity': str(news_severity),
            'news_keywords': news_keywords if isinstance(news_keywords, list) else [],
            'news_articles': int(news_articles) if news_articles else 0,
            'news_spike': int(news_articles) > int(news_velocity_base or 0) * 2 if news_velocity_base else False,
        }
        # Read persisted headlines for context
        try:
            with open('data/headlines_latest.json') as hf:
                hl_data = json.load(hf)
            if hl_data.get('ticker') == 'XAG-USD' and hl_data.get('headlines'):
                top_hl = hl_data['headlines'][0]
                out['headline_top'] = top_hl.get('title', '')[:80]
                out['headline_severity'] = top_hl.get('severity', '')
                out['headline_sentiment'] = top_hl.get('sentiment', '')
        except Exception:
            pass
        return out
    except Exception as e:
        log_msg(f'Signal load error: {e}')
        return None

def detect_position():
    """Detect existing silver position on Avanza."""
    try:
        from portfolio.avanza_session import get_positions
        for p in get_positions():
            name = p.get('name', '')
            vol = p.get('volume', 0)
            if vol <= 0:
                continue
            for kw in SILVER_KEYWORDS:
                if kw in name.upper():
                    ob_id = p.get('orderbook_id', '')
                    val = p.get('value', 0)
                    profit = p.get('profit', 0)
                    cost = val - profit
                    entry_cert = cost / vol if vol > 0 else 0
                    is_short = any(k in name.upper() for k in ['BEAR', 'MINI S', 'TURBO S'])
                    direction = 'SHORT' if is_short else 'LONG'
                    log_msg(f'Detected position: {name} {vol}u {direction} val={val:.0f} P&L={profit:+.0f}')
                    return {
                        'ob': ob_id, 'd': direction, 'v': vol,
                        'c': entry_cert, 'e': 0, 't': time.time() - 600,
                    }
    except Exception as e:
        log_msg(f'Position detect error: {e}')
    return None

def sell_position(active, reason):
    """Sell the active position. Returns P&L."""
    try:
        from portfolio.avanza_session import place_sell_order, get_quote
        q = get_quote(active['ob'])
        bid = float(q.get('buy', 0))
        if bid > 0:
            r = place_sell_order(active['ob'], price=bid, volume=active['v'])
            pnl = (bid - active['c']) * active['v']
            log_msg(f'SELL({reason}): {active["v"]}u@{bid} P&L:{pnl:+.0f} [{r.get("orderRequestStatus", "?")}]')
            return pnl
    except Exception as e:
        log_msg(f'Sell error: {e}')
    return 0

def buy_position(direction, price):
    """Enter a new position. Returns active dict or None."""
    try:
        from portfolio.avanza_session import place_buy_order, get_quote, get_buying_power
        ob = '1650161' if direction == 'LONG' else '2286417'
        q = get_quote(ob)
        ask = float(q.get('sell', 0))
        bp = float(get_buying_power().get('buying_power', 0))
        if ask <= 0 or bp < 100:
            return None
        vol = int(min(bp * 0.95, 1500) / ask)
        if vol < 5:
            return None
        r = place_buy_order(ob, price=ask, volume=vol)
        nm = 'BULL X5' if direction == 'LONG' else 'BEAR X5'
        log_msg(f'BUY: {vol}u {nm}@{ask} [{r.get("orderRequestStatus", "?")}]')
        return {'ob': ob, 'd': direction, 'v': vol, 'c': ask, 'e': price, 't': time.time()}
    except Exception as e:
        log_msg(f'Buy error: {e}')
    return None

# =========================================================================
# MAIN LOOP — triple try/except, never crashes
# =========================================================================
def main():
    session_pnl = -597
    active = detect_position()
    mch = []
    lsc = 0
    cd = 0
    md = 0

    log_msg(f'=== FISH MONITOR LIVE | {session_pnl:+.0f} SEK | until {SESSION_END_H}:{SESSION_END_M:02d} ===')
    if active:
        log_msg(f'Resuming {active["d"]} {active["v"]}u')
    else:
        log_msg('No position, scanning')

    while True:
        try:
            heartbeat()
            now = time.time()
            dt = datetime.datetime.now()
            h, m = dt.hour, dt.minute

            # --- Session end ---
            if h >= SESSION_END_H and m >= SESSION_END_M:
                if active:
                    sg = load_signals()
                    hold = sg and sg.get('f1d_dir') == 'up' and sg.get('f1d_prob', 0) > 0.55
                    if hold:
                        log_msg(f'21:45 {sg["f1d"]} HOLDING OVERNIGHT')
                    else:
                        pnl = sell_position(active, 'SESSION END')
                        session_pnl += pnl
                        active = None
                log_msg(f'Session end: {session_pnl:+.0f} SEK')
                break

            p = fetch_price()
            if not p:
                time.sleep(30)
                continue

            if now - lsc >= 60:
                lsc = now
                sg = load_signals()
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
                    ep = active['e'] if active['e'] > 0 else p
                    if active['e'] == 0:
                        active['e'] = p  # set entry price if not known
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

                    # Event/news warnings (Gap 1+2 fix)
                    warnings = []
                    event_hours = sg.get('event_hours', 999)
                    event_name = sg.get('event_name', '')
                    if event_hours < 24 and event_name:
                        warnings.append(f'EVENT:{event_name} {event_hours:.0f}h')
                    if sg.get('high_impact_near'):
                        warnings.append('HIGH-IMPACT <4h!')
                    if sg.get('news_severity') in ('critical', 'high'):
                        kw = ','.join(sg.get('news_keywords', [])[:3])
                        warnings.append(f'NEWS:{sg["news_severity"]}({kw})')
                    if sg.get('news_spike'):
                        warnings.append(f'NEWS-SPIKE:{sg.get("news_articles",0)} articles')
                    if sg['news'] != 'HOLD':
                        warnings.append(f'N={sg["news"]}')
                        hl = sg.get('headline_top', '')
                        if hl:
                            warnings.append(f'"{hl[:50]}"')
                    if sg['econ'] != 'HOLD':
                        warnings.append(f'E={sg["econ"]}')
                    warn_str = ' !!' + ' '.join(warnings) if warnings else ''

                    # Max hold time: reduce to 60m if high-impact event within 24h
                    max_hold = 60 if (event_hours < 24 or sg.get('high_impact_near')) else 120

                    log_msg(f'${p:.2f} {d} {mv:+.1f}%/{cm:+.0f}% P&L:{cpnl:+.0f} | {sg["a"]} {sg["b"]}B/{sg["s"]}S RSI={sg["rsi"]:.0f} MC={sg["mc"]:.0%} M:{sg["ma"]} | {sg["f1d"]}{mdf}{warn_str} [{mn}m/{max_hold}m]')

                    ex = None
                    if d == 'LONG':
                        if sg['rsi'] > 62 and sg['mc'] < 0.35:
                            ex = 'COMB'
                        elif sg['rsi'] > 70:
                            ex = 'RSI'
                        elif sg['s'] > sg['b'] + 4:
                            ex = 'SELL flip'
                        elif md >= 3:
                            ex = f'MD{md}'
                        elif mv >= 2.0:
                            ex = 'TP'
                        elif mv <= -3.0:
                            ex = 'SL'
                        elif mn >= max_hold:
                            ex = f'{max_hold}m hold'
                    else:
                        if sg['rsi'] < 30:
                            ex = 'RSI'
                        elif sg['b'] > sg['s'] + 4:
                            ex = 'BUY flip'
                        elif md >= 3:
                            ex = f'MD{md}'
                        elif mv >= 2.0:
                            ex = 'TP'
                        elif mv <= -3.0:
                            ex = 'SL'
                        elif mn >= max_hold:
                            ex = f'{max_hold}m hold'

                    if ex:
                        pnl = sell_position(active, ex)
                        session_pnl += pnl
                        log_msg(f'Session: {session_pnl:+.0f} SEK')
                        # Variable cooldown based on exit conviction
                        # High-conviction directional exits → short cooldown (flip faster)
                        # Low-conviction exits → longer cooldown (rescan needed)
                        if ex in ('COMB', 'SELL flip', 'BUY flip', 'RSI'):
                            cd = now + 5   # near-instant re-entry allowed
                            log_msg(f'High-conviction exit ({ex}) — ready to flip')
                        elif ex.startswith('MD'):
                            cd = now + 60  # standard cooldown
                        else:
                            cd = now + 120  # TP/SL/timeout — need fresh signal
                        active = None
                        md = 0
                else:
                    # Scan for entry
                    bu = sg['a'] == 'BUY' and sg['ma'] == 'BUY' and m2u
                    be = sg['a'] == 'SELL' and sg['ma'] == 'SELL' and m2b
                    fl = ''
                    if now > cd and (bu or be):
                        dr = 'LONG' if bu else 'SHORT'
                        active = buy_position(dr, p)
                        if active:
                            md = 0
                            fl = f' >>> {dr}'
                    # Show warnings when scanning too
                    scan_warns = []
                    ev_h = sg.get('event_hours', 999)
                    ev_n = sg.get('event_name', '')
                    if ev_h < 24 and ev_n:
                        scan_warns.append(f'EVENT:{ev_n} {ev_h:.0f}h')
                    if sg.get('news_severity') in ('critical', 'high'):
                        scan_warns.append(f'NEWS:{sg["news_severity"]}')
                        hl = sg.get('headline_top', '')
                        if hl:
                            scan_warns.append(f'"{hl[:50]}"')
                    if sg['econ'] != 'HOLD':
                        scan_warns.append(f'E={sg["econ"]}')
                    sw = ' !!' + ' '.join(scan_warns) if scan_warns else ''
                    log_msg(f'${p:.2f} | {sg["a"]} {sg["b"]}B/{sg["s"]}S RSI={sg["rsi"]:.0f} MC={sg["mc"]:.0%} M:{sg["ma"]} | {sg["f1d"]}{sw}{fl}')
            else:
                if active:
                    d = active['d']
                    ep = active['e'] if active['e'] > 0 else p
                    mv = (p - ep) / ep * 100 if d == 'LONG' else (ep - p) / ep * 100
                    cpnl = active['c'] * active['v'] * mv * 5 / 100
                    log_msg(f'${p:.2f} {mv:+.1f}% P&L:{cpnl:+.0f}')
                else:
                    log_msg(f'${p:.2f}')

        except Exception as e:
            log_msg(f'ERROR (continuing): {e}')
            traceback.print_exc()

        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log_msg(f'FATAL: {e}')
        traceback.print_exc()
