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


def fetch_gold_price():
    try:
        return float(requests.get(f'{FAPI}?symbol=XAUUSDT', timeout=5).json()['price'])
    except:
        return None


# ---------------------------------------------------------------------------
# Gold-leads-silver detection (5-min rolling, 0.85 correlation)
# ---------------------------------------------------------------------------
GOLD_LEAD_MINUTES = 5
GOLD_LEAD_THRESHOLD = 0.5  # gold must move >0.5% in 5 min
SILVER_LAG_THRESHOLD = 0.2  # silver hasn't moved >0.2% yet

class GoldLeadDetector:
    """Detect when gold moves before silver — 5 minute edge."""

    def __init__(self):
        self.gold_prices = []  # [(timestamp, price), ...]
        self.silver_prices = []

    def update(self, gold_price, silver_price):
        now = time.time()
        if gold_price:
            self.gold_prices.append((now, gold_price))
        if silver_price:
            self.silver_prices.append((now, silver_price))
        # Keep 10 min of history
        cutoff = now - 600
        self.gold_prices = [(t, p) for t, p in self.gold_prices if t > cutoff]
        self.silver_prices = [(t, p) for t, p in self.silver_prices if t > cutoff]

    def detect(self):
        """Returns ('LONG', confidence), ('SHORT', confidence), or (None, 0).

        LONG: gold rallied >0.5% in 5 min, silver hasn't followed yet
        SHORT: gold dropped >0.5% in 5 min, silver hasn't followed yet
        """
        if len(self.gold_prices) < 2 or len(self.silver_prices) < 2:
            return None, 0

        now = time.time()
        # Gold 5-min change
        gold_5m_ago = [p for t, p in self.gold_prices if t <= now - GOLD_LEAD_MINUTES * 60]
        if not gold_5m_ago:
            gold_5m_ago = [self.gold_prices[0][1]]
        gold_now = self.gold_prices[-1][1]
        gold_chg = (gold_now - gold_5m_ago[-1]) / gold_5m_ago[-1] * 100

        # Silver recent change (should be smaller if gold is leading)
        silver_5m_ago = [p for t, p in self.silver_prices if t <= now - GOLD_LEAD_MINUTES * 60]
        if not silver_5m_ago:
            silver_5m_ago = [self.silver_prices[0][1]]
        silver_now = self.silver_prices[-1][1]
        silver_chg = (silver_now - silver_5m_ago[-1]) / silver_5m_ago[-1] * 100

        # Gold moved big, silver hasn't followed
        if gold_chg > GOLD_LEAD_THRESHOLD and silver_chg < SILVER_LAG_THRESHOLD:
            confidence = min(1.0, gold_chg / 1.0)  # scale: 0.5%=0.5, 1.0%=1.0
            return 'LONG', round(confidence, 2)
        elif gold_chg < -GOLD_LEAD_THRESHOLD and silver_chg > -SILVER_LAG_THRESHOLD:
            confidence = min(1.0, abs(gold_chg) / 1.0)
            return 'SHORT', round(confidence, 2)

        return None, 0

    def status(self):
        """One-line status for display."""
        if len(self.gold_prices) < 2:
            return ''
        now = time.time()
        gold_5m_ago = [p for t, p in self.gold_prices if t <= now - GOLD_LEAD_MINUTES * 60]
        if not gold_5m_ago:
            return ''
        gold_chg = (self.gold_prices[-1][1] - gold_5m_ago[-1]) / gold_5m_ago[-1] * 100
        return f'Au5m:{gold_chg:+.1f}%'


# ---------------------------------------------------------------------------
# Temporal Pattern — recurring time-of-day edges (Headlands-inspired)
# ---------------------------------------------------------------------------

class TemporalPatternDetector:
    """Detect recurring time-of-day patterns from historical scan."""

    def __init__(self):
        self.patterns = {}  # {(dow, hour): {'direction': 'BULL', 'probability': 75, ...}}
        try:
            with open('data/temporal_patterns.json') as f:
                data = json.load(f)
            for p in data.get('patterns', []):
                key = (p['day'], p['hour_cet'])
                self.patterns[key] = p
            log_msg(f'Temporal: loaded {len(self.patterns)} patterns')
        except Exception:
            log_msg('Temporal: no patterns file')

    def detect(self, min_probability=68):
        """Check if current (day, hour) has a pattern. Returns (direction, prob) or (None, 0)."""
        dt = datetime.datetime.now()
        dow = dt.weekday()  # 0=Mon
        hour = dt.hour
        key = (dow, hour)
        p = self.patterns.get(key)
        if p and p['probability'] >= min_probability:
            return p['direction'], p['probability']
        return None, 0

    def status(self):
        dt = datetime.datetime.now()
        key = (dt.weekday(), dt.hour)
        p = self.patterns.get(key)
        if p:
            return f'T:{p["day_name"]}{p["hour_cet"]:02d}={p["direction"][0]}{p["probability"]:.0f}%'
        return ''


# ---------------------------------------------------------------------------
# Vol-Targeting — scale position size by current volatility
# ---------------------------------------------------------------------------

def compute_vol_scalar():
    """Compute vol-targeting position size scalar.

    Returns a float 0.25-2.0:
    - High vol (ATR >> median) → smaller positions (0.25-0.5)
    - Normal vol → 1.0
    - Low vol → larger positions (1.5-2.0)
    """
    try:
        import yfinance as yf
        import numpy as np
        si = yf.download('SI=F', period='6mo', interval='1d', progress=False)
        if si.empty or len(si) < 30:
            return 1.0
        close = si['Close'].values.flatten()
        high = si['High'].values.flatten()
        low = si['Low'].values.flatten()
        # ATR 14-day
        tr = [max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
              for i in range(1, len(close))]
        atr_14 = np.mean(tr[-14:])
        atr_median = np.median(tr[-126:])  # 6-month median
        if atr_median == 0:
            return 1.0
        ratio = atr_14 / atr_median
        # Scale: ratio=1 → scalar=1, ratio=2 → scalar=0.5, ratio=0.5 → scalar=1.5
        scalar = 1.0 / max(ratio, 0.5)
        scalar = max(0.25, min(2.0, scalar))
        return round(scalar, 2)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# ORB — Opening Range Breakout
# ---------------------------------------------------------------------------

class ORBTracker:
    """Track the opening range (first 15 min) and detect breakouts."""

    def __init__(self):
        self.range_high = 0
        self.range_low = 0
        self.range_formed = False
        self.prices = []
        self.start_time = 0

    def set_range(self, high, low):
        """Manually set the range (e.g., from orb_predictor)."""
        self.range_high = high
        self.range_low = low
        self.range_formed = True
        log_msg(f'ORB range set: ${low:.2f} - ${high:.2f} (width: {(high-low)/low*100:.1f}%)')

    def update(self, price):
        """Feed prices during range formation (first 15 min)."""
        if self.range_formed:
            return
        if not self.start_time:
            self.start_time = time.time()
        self.prices.append(price)
        elapsed = time.time() - self.start_time
        if elapsed >= 900 and len(self.prices) >= 3:  # 15 min
            self.range_high = max(self.prices)
            self.range_low = min(self.prices)
            self.range_formed = True
            log_msg(f'ORB range formed: ${self.range_low:.2f} - ${self.range_high:.2f}')

    def detect(self, price):
        """Returns ('LONG', tp, sl), ('SHORT', tp, sl), or (None, 0, 0)."""
        if not self.range_formed or self.range_high <= self.range_low:
            return None, 0, 0
        rng = self.range_high - self.range_low
        if price > self.range_high:
            tp = price + rng * 0.5  # 50% extension
            sl = self.range_high - rng * 0.6
            return 'LONG', tp, sl
        elif price < self.range_low:
            tp = price - rng * 0.5
            sl = self.range_low + rng * 0.6
            return 'SHORT', tp, sl
        return None, 0, 0

    def status(self):
        if not self.range_formed:
            return 'ORB:forming'
        return f'ORB:${self.range_low:.2f}-${self.range_high:.2f}'

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

def buy_position(direction, price, size_scalar=1.0):
    """Enter a new position with vol-targeted sizing. Returns active dict or None."""
    try:
        from portfolio.avanza_session import place_buy_order, get_quote, get_buying_power
        ob = '1650161' if direction == 'LONG' else '2286417'
        q = get_quote(ob)
        ask = float(q.get('sell', 0))
        bp = float(get_buying_power().get('buying_power', 0))
        if ask <= 0 or bp < 100:
            return None
        budget = min(bp * 0.95, 1500) * size_scalar
        vol = int(budget / ask)
        if vol < 5:
            return None
        r = place_buy_order(ob, price=ask, volume=vol)
        nm = 'BULL X5' if direction == 'LONG' else 'BEAR X5'
        sz = f' (vol-scaled {size_scalar:.1f}x)' if size_scalar != 1.0 else ''
        log_msg(f'BUY: {vol}u {nm}@{ask}{sz} [{r.get("orderRequestStatus", "?")}]')
        return {'ob': ob, 'd': direction, 'v': vol, 'c': ask, 'e': price, 't': time.time()}
    except Exception as e:
        log_msg(f'Buy error: {e}')
    return None

# =========================================================================
# MODE DETECTION — momentum vs straddle
# =========================================================================

def detect_mode():
    """Detect whether to use momentum or straddle mode.

    Returns ('momentum', {}) or ('straddle', {'floor': ..., 'ceil': ...}).
    Works at any time of day — looks at recent data, not session start.
    """
    try:
        import yfinance as yf
        import numpy as np

        # 1. Was yesterday a crash? (>3% drop)
        daily = yf.download('SI=F', period='5d', interval='1d', progress=False)
        if len(daily) >= 2:
            yesterday_chg = (daily['Close'].iloc[-2] - daily['Close'].iloc[-3]) / daily['Close'].iloc[-3] * 100
            yesterday_chg = float(yesterday_chg.iloc[0]) if hasattr(yesterday_chg, 'iloc') else float(yesterday_chg)
            if yesterday_chg < -5:
                log_msg(f'Mode: STRADDLE (yesterday crashed {yesterday_chg:.1f}%)')
                return _compute_straddle_levels()

        # 2. Today's range vs net move (is it chopping?)
        intraday = yf.download('SI=F', period='1d', interval='15m', progress=False)
        if not intraday.empty and len(intraday) >= 8:
            closes = intraday['Close'].values.flatten()
            highs = intraday['High'].values.flatten()
            lows = intraday['Low'].values.flatten()

            day_range = (max(highs) - min(lows)) / min(lows) * 100
            net_move = abs(closes[-1] - closes[0]) / closes[0] * 100

            if day_range > 3 and net_move < 0.5:
                log_msg(f'Mode: STRADDLE (range {day_range:.1f}% but net move {net_move:.1f}% — chop)')
                return _compute_straddle_levels()

        # 3. Check MC stability from recent signal log
        try:
            with open('data/metals_signal_log.jsonl') as f:
                lines = f.readlines()
            recent_mc = []
            for line in lines[-10:]:
                d = json.loads(line)
                xag = d.get('signals', {}).get('XAG-USD', {})
                if xag.get('w_confidence') is not None:
                    # Proxy: if action flips frequently, it's choppy
                    recent_mc.append(xag.get('action', 'HOLD'))
            if recent_mc:
                flips = sum(1 for i in range(1, len(recent_mc)) if recent_mc[i] != recent_mc[i-1])
                if flips >= 4:
                    log_msg(f'Mode: STRADDLE (signal flipped {flips}x in last 10 checks)')
                    return _compute_straddle_levels()
        except Exception:
            pass

    except Exception as e:
        log_msg(f'Mode detection error: {e}')

    log_msg('Mode: MOMENTUM (default)')
    return 'momentum', {}


def _compute_straddle_levels():
    """Compute floor/ceiling levels for straddle mode."""
    try:
        p = fetch_price()
        if p:
            floor_pct = 3.0
            ceil_pct = 2.0
            floor = p * (1 - floor_pct / 100)
            ceil = p * (1 + ceil_pct / 100)
            log_msg(f'Straddle levels: floor ${floor:.2f} (-{floor_pct}%) | ceil ${ceil:.2f} (+{ceil_pct}%)')
            return 'straddle', {'floor': floor, 'ceil': ceil, 'floor_pct': floor_pct, 'ceil_pct': ceil_pct}
    except Exception:
        pass
    return 'momentum', {}


# =========================================================================
# MAIN LOOP — triple try/except, never crashes
# =========================================================================
def main():
    session_pnl = 0
    active = detect_position()
    mch = []
    lsc = 0
    cd = 0

    # Detect mode (works at any time of day)
    mode, straddle_cfg = detect_mode()
    straddle_floor = straddle_cfg.get('floor', 0)
    straddle_ceil = straddle_cfg.get('ceil', 0)
    straddle_bull_filled = False
    straddle_bear_filled = False
    momentum_losses = 0  # track consecutive bad trades for mode switch

    # Tactic modules
    gold_lead = GoldLeadDetector()
    orb = ORBTracker()
    temporal = TemporalPatternDetector()
    vol_scalar = compute_vol_scalar()
    log_msg(f'Vol scalar: {vol_scalar:.2f}x (1.0=normal, <1=high vol, >1=low vol)')
    # Try to load ORB range from orb_predictor
    try:
        from portfolio.orb_predictor import fetch_klines, calculate_morning_range
        klines = fetch_klines(num_batches=1, interval="15m", limit=100)
        if klines:
            from portfolio.orb_predictor import _group_by_day
            days = _group_by_day(klines)
            if days:
                today_candles = days[-1]
                mr = calculate_morning_range(today_candles)
                if mr:
                    orb.set_range(mr.high, mr.low)
    except Exception as e:
        log_msg(f'ORB init: {e}')

    # VWAP for straddle exit target
    vwap = 0
    try:
        with open('data/agent_summary.json') as f:
            full = json.load(f)
        vwap = float((full.get('price_levels', {}).get('XAG-USD', {}) or {}).get('vwap', 0))
        if vwap:
            log_msg(f'VWAP: ${vwap:.2f}')
    except Exception:
        pass
    md = 0

    log_msg(f'=== FISH MONITOR LIVE | {session_pnl:+.0f} SEK | {mode.upper()} mode | until {SESSION_END_H}:{SESSION_END_M:02d} ===')
    if mode == 'straddle':
        log_msg(f'Straddle: floor ${straddle_floor:.2f} | ceil ${straddle_ceil:.2f}')
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

            # Track gold + ORB every cycle
            gold_p = fetch_gold_price()
            gold_lead.update(gold_p, p)
            orb.update(p)

            # Time-of-day gating (tactic 4)
            is_us_session = 14 <= h < 17  # 14:00-17:00 CET = best hours
            is_dead_zone = 10 <= h < 14   # 10:00-14:00 CET = low volume

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

                    gl_str = f' {gold_lead.status()}' if gold_lead.status() else ''
                    orb_str = f' {orb.status()}' if orb.range_formed else ''
                    log_msg(f'${p:.2f} {d} {mv:+.1f}%/{cm:+.0f}% P&L:{cpnl:+.0f} | {sg["a"]} {sg["b"]}B/{sg["s"]}S RSI={sg["rsi"]:.0f} MC={sg["mc"]:.0%} M:{sg["ma"]} | {sg["f1d"]}{mdf}{warn_str}{gl_str}{orb_str} [{mn}m/{max_hold}m]')

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
                        # Track momentum losses for potential mode switch
                        if pnl <= 0:
                            momentum_losses += 1
                        else:
                            momentum_losses = 0
                        # Auto-switch to straddle after 3 consecutive losses
                        if mode == 'momentum' and momentum_losses >= 3:
                            log_msg(f'!! 3 consecutive losses — switching to STRADDLE mode')
                            mode = 'straddle'
                            _, straddle_cfg = _compute_straddle_levels()
                            straddle_floor = straddle_cfg.get('floor', 0)
                            straddle_ceil = straddle_cfg.get('ceil', 0)
                            straddle_bull_filled = False
                            straddle_bear_filled = False
                else:
                    # Scan for entry — depends on mode
                    fl = ''
                    if mode == 'momentum':
                        bu = sg['a'] == 'BUY' and sg['ma'] == 'BUY' and m2u
                        be = sg['a'] == 'SELL' and sg['ma'] == 'SELL' and m2b
                        if now > cd and (bu or be):
                            dr = 'LONG' if bu else 'SHORT'
                            active = buy_position(dr, p, vol_scalar)
                            if active:
                                md = 0
                                fl = f' >>> MOMENTUM {dr}'
                    elif mode == 'straddle':
                        # Check if price hit floor or ceiling
                        past_cancel = h > CANCEL_HOUR or (h == CANCEL_HOUR and m >= CANCEL_MIN)
                        if not past_cancel and straddle_floor > 0 and straddle_ceil > 0:
                            if p <= straddle_floor and not straddle_bull_filled:
                                log_msg(f'!! FLOOR HIT ${p:.2f} <= ${straddle_floor:.2f}')
                                active = buy_position('LONG', p, vol_scalar)
                                if active:
                                    straddle_bull_filled = True
                                    fl = f' >>> STRADDLE LONG (floor)'
                            elif p >= straddle_ceil and not straddle_bear_filled:
                                log_msg(f'!! CEILING HIT ${p:.2f} >= ${straddle_ceil:.2f}')
                                active = buy_position('SHORT', p, vol_scalar)
                                if active:
                                    straddle_bear_filled = True
                                    fl = f' >>> STRADDLE SHORT (ceil)'
                    # --- Gold-leads-silver entry (fires in ANY mode) ---
                    if not active and now > cd:
                        gl_dir, gl_conf = gold_lead.detect()
                        if gl_dir and gl_conf >= 0.5:
                            # Time gating: stronger signal needed in dead zone
                            if not is_dead_zone or gl_conf >= 0.7:
                                log_msg(f'!! GOLD LEAD: {gl_dir} (conf {gl_conf:.0%}, {gold_lead.status()})')
                                active = buy_position(gl_dir, p, vol_scalar)
                                if active:
                                    fl = f' >>> GOLD-LEAD {gl_dir}'

                    # --- ORB breakout entry (fires in ANY mode) ---
                    if not active and now > cd and orb.range_formed:
                        orb_dir, orb_tp, orb_sl = orb.detect(p)
                        if orb_dir:
                            log_msg(f'!! ORB BREAKOUT: {orb_dir} (TP=${orb_tp:.2f} SL=${orb_sl:.2f})')
                            active = buy_position(orb_dir, p, vol_scalar)
                            if active:
                                fl = f' >>> ORB {orb_dir}'

                    # --- Temporal pattern entry (fires in ANY mode) ---
                    if not active and now > cd:
                        tp_dir, tp_prob = temporal.detect(min_probability=68)
                        if tp_dir:
                            log_msg(f'!! TEMPORAL: {tp_dir} {tp_prob:.0f}% ({temporal.status()})')
                            active = buy_position(tp_dir if tp_dir == 'BULL' else 'SHORT' if tp_dir == 'BEAR' else tp_dir, p, vol_scalar)
                            if active:
                                fl = f' >>> TEMPORAL {tp_dir}'

                    # --- Sentiment velocity entry (fires in ANY mode) ---
                    if not active and now > cd and sg.get('news_spike'):
                        # Headlines spiking > 2x baseline
                        sentiment = sg.get('headline_sentiment', '')
                        if sentiment == 'negative':
                            log_msg(f'!! NEWS VELOCITY SPIKE: {sg.get("news_articles",0)} articles, negative -> BEAR')
                            active = buy_position('SHORT', p, vol_scalar)
                            if active:
                                fl = f' >>> NEWS-VELOCITY SHORT'
                        elif sentiment == 'positive':
                            log_msg(f'!! NEWS VELOCITY SPIKE: {sg.get("news_articles",0)} articles, positive -> BULL')
                            active = buy_position('LONG', p, vol_scalar)
                            if active:
                                fl = f' >>> NEWS-VELOCITY LONG'

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
                    mode_str = f' [{mode.upper()}]' if mode == 'straddle' else ''
                    straddle_str = ''
                    if mode == 'straddle' and straddle_floor > 0:
                        dist_f = (p - straddle_floor) / p * 100
                        dist_c = (straddle_ceil - p) / p * 100
                        straddle_str = f' F=${straddle_floor:.2f}({dist_f:.1f}%) C=${straddle_ceil:.2f}({dist_c:.1f}%)'
                    gl_s = f' {gold_lead.status()}' if gold_lead.status() else ''
                    orb_s = f' {orb.status()}' if orb.range_formed else ''
                    tp_s = f' {temporal.status()}' if temporal.status() else ''
                    vs = f' vol:{vol_scalar:.1f}x' if vol_scalar != 1.0 else ''
                    tz = ' [DEAD-ZONE]' if is_dead_zone else (' [US-SESSION]' if is_us_session else '')
                    log_msg(f'${p:.2f} | {sg["a"]} {sg["b"]}B/{sg["s"]}S RSI={sg["rsi"]:.0f} MC={sg["mc"]:.0%} M:{sg["ma"]} | {sg["f1d"]}{sw}{fl}{mode_str}{straddle_str}{gl_s}{orb_s}{tp_s}{vs}{tz}')
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
