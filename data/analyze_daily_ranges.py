"""Analyze historical daily price ranges for XAG and XAU, project onto leveraged warrants."""
import json, statistics, math

with open("data/metals_history.json", "r") as f:
    hist = json.load(f)

# Current positions
POSITIONS = {
    "BULL GULD X8 N": {"underlying": "XAU-USD", "leverage": 8.0, "entry": 972.4, "bid": 954.5},
    "MINI L SILVER AVA 79": {"underlying": "XAG-USD", "leverage": 1.34, "entry": 65.13, "bid": 64.99},  # effective ~5x
    "MINI L SILVER AVA 301": {"underlying": "XAG-USD", "leverage": 4.28, "entry": 20.70, "bid": 20.35},
}

# Effective leverage for MINI warrants: price_move * (underlying_price / (underlying_price - barrier))
# Silver79 barrier=24.92, underlying=95.24 -> eff_lev = 95.24/(95.24-24.92) = 1.35x (matches reported)
# Silver301 barrier=75.03, underlying=95.24 -> eff_lev = 95.24/(95.24-75.03) = 4.71x
# But reported is 4.28x, let's use reported values

for metal_key in ["XAG-USD", "XAU-USD"]:
    data = hist["metals"][metal_key]
    candles = data["daily_ohlcv"]
    stats = data["stats"]

    print(f"\n{'='*70}")
    print(f"  {metal_key} — Daily Range Analysis (YTD 2026, {len(candles)} trading days)")
    print(f"{'='*70}")

    # Calculate daily metrics
    daily_ranges = []       # (high-low)/open as %
    open_to_high = []       # (high-open)/open as % (best case from open)
    open_to_low = []        # (open-low)/open as % (worst case from open)
    close_to_close = []     # close-to-close change %
    true_ranges = []        # max(H-L, |H-prev_C|, |L-prev_C|)

    prev_close = None
    for c in candles:
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]

        range_pct = (h - l) / o * 100
        oth = (h - o) / o * 100
        otl = (o - l) / o * 100

        daily_ranges.append(range_pct)
        open_to_high.append(oth)
        open_to_low.append(otl)

        if prev_close is not None:
            ctc = (cl - prev_close) / prev_close * 100
            close_to_close.append(ctc)
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            true_ranges.append(tr / o * 100)

        prev_close = cl

    # Sort for percentile calculation
    dr_sorted = sorted(daily_ranges)
    oth_sorted = sorted(open_to_high)
    otl_sorted = sorted(open_to_low)
    ctc_sorted = sorted(close_to_close)

    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[f] * (c - k) + data[c] * (k - f)

    print(f"\n  Current price: ${stats['ytd_current']:.2f}")
    print(f"  ATR(14d): {stats['atr_14d_pct']:.2f}%  (${stats['atr_14d']:.2f})")
    print(f"  Avg daily range: {stats['avg_daily_range_pct']:.2f}%")
    print(f"  Up days: {stats['up_days']}/{stats['total_trading_days']} ({stats['up_days']/stats['total_trading_days']*100:.0f}%)")

    print(f"\n  --- Daily Range (High-Low)/Open ---")
    print(f"  {'Percentile':<12} {'Range %':>10} {'$ Move':>10}")
    for p in [10, 25, 50, 75, 90, 95]:
        val = percentile(dr_sorted, p)
        dollar = val / 100 * stats['ytd_current']
        print(f"  P{p:<10} {val:>9.2f}% {dollar:>9.2f}")
    print(f"  {'Min':<12} {min(daily_ranges):>9.2f}%")
    print(f"  {'Max':<12} {max(daily_ranges):>9.2f}%")
    print(f"  {'Mean':<12} {statistics.mean(daily_ranges):>9.2f}%")

    print(f"\n  --- Max Intraday Gain from Open (Open->High) ---")
    print(f"  {'Percentile':<12} {'Gain %':>10}")
    for p in [25, 50, 75, 90, 95]:
        val = percentile(oth_sorted, p)
        print(f"  P{p:<10} {val:>9.2f}%")

    print(f"\n  --- Max Intraday Drop from Open (Open->Low) ---")
    print(f"  {'Percentile':<12} {'Drop %':>10}")
    for p in [25, 50, 75, 90, 95]:
        val = percentile(otl_sorted, p)
        print(f"  P{p:<10} {val:>9.2f}%")

    print(f"\n  --- Close-to-Close Daily Change ---")
    print(f"  {'Percentile':<12} {'Change %':>10}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = percentile(ctc_sorted, p)
        print(f"  P{p:<10} {val:>9.2f}%")

    # Recent 5 days
    recent = candles[-5:]
    print(f"\n  --- Last 5 Days ---")
    print(f"  {'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Range%':>8} {'Chg%':>8}")
    for i, c in enumerate(recent):
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        rng = (h - l) / o * 100
        chg = (cl - (candles[-5+i-1]["close"] if i > 0 else o)) / o * 100
        print(f"  {c['date']:<12} {o:>8.2f} {h:>8.2f} {l:>8.2f} {cl:>8.2f} {rng:>7.2f}% {chg:>7.2f}%")

# ========================================================
# WARRANT IMPACT TABLE
# ========================================================
print(f"\n{'='*70}")
print(f"  WARRANT IMPACT — Expected Daily Moves (Leveraged)")
print(f"{'='*70}")
print(f"\n  How much your warrants can move today based on typical underlying ranges:")

for metal_key in ["XAG-USD", "XAU-USD"]:
    candles = hist["metals"][metal_key]["daily_ohlcv"]

    daily_ranges = []
    open_to_low = []
    open_to_high = []
    for c in candles:
        o, h, l = c["open"], c["high"], c["low"]
        daily_ranges.append((h - l) / o * 100)
        open_to_high.append((h - o) / o * 100)
        open_to_low.append((o - l) / o * 100)

    dr_sorted = sorted(daily_ranges)
    oth_sorted = sorted(open_to_high)
    otl_sorted = sorted(open_to_low)

    # Get relevant positions
    relevant = {k: v for k, v in POSITIONS.items() if v["underlying"] == metal_key}
    if not relevant:
        continue

    print(f"\n  {metal_key} Warrants:")
    print(f"  {'':20} {'Typical Day (P50)':>18} {'Bad Day (P90)':>18} {'Worst Day (P95)':>18}")

    for name, pos in relevant.items():
        lev = pos["leverage"]
        entry = pos["entry"]
        bid = pos["bid"]

        # Median and tail underlying moves
        med_range = percentile(dr_sorted, 50)
        p90_range = percentile(dr_sorted, 90)
        p95_range = percentile(dr_sorted, 95)

        med_drop = percentile(otl_sorted, 50)
        p90_drop = percentile(otl_sorted, 90)
        p95_drop = percentile(otl_sorted, 95)

        med_gain = percentile(oth_sorted, 50)
        p90_gain = percentile(oth_sorted, 90)
        p95_gain = percentile(oth_sorted, 95)

        # Leveraged warrant moves
        med_w = med_range * lev
        p90_w = p90_range * lev
        p95_w = p95_range * lev

        med_d = med_drop * lev
        p90_d = p90_drop * lev
        p95_d = p95_drop * lev

        med_g = med_gain * lev
        p90_g = p90_gain * lev
        p95_g = p95_gain * lev

        short_name = name.replace("MINI L SILVER AVA ", "SILVER").replace("BULL GULD X8 N", "GOLD 8x")

        print(f"\n  {short_name} ({lev}x leverage):")
        print(f"    Full range:     {med_w:>+8.1f}%           {p90_w:>+8.1f}%           {p95_w:>+8.1f}%")
        print(f"    Max drop:       {-med_d:>+8.1f}%           {-p90_d:>+8.1f}%           {-p95_d:>+8.1f}%")
        print(f"    Max gain:       {+med_g:>+8.1f}%           {+p90_g:>+8.1f}%           {+p95_g:>+8.1f}%")

        # SEK impact
        value = bid * (5 if "GULD" in name else 78 if "79" in name else 240)
        print(f"    SEK impact:     {value * med_d/100:>+8.0f} SEK drop    {value * p90_d/100:>+8.0f} SEK drop    {value * p95_d/100:>+8.0f} SEK drop")
        print(f"    Position value: {value:.0f} SEK")

# ========================================================
# TODAY'S SESSION SO FAR
# ========================================================
print(f"\n{'='*70}")
print(f"  TODAY'S SESSION (so far, ~1h into session)")
print(f"{'='*70}")

# Today's underlying range from price_history_recent
with open("data/metals_context.json", "r") as f:
    ctx = json.load(f)

recent = ctx["price_history_recent"]
if recent:
    gold_prices = [r["gold_und"] for r in recent]
    silver_prices = [r["silver79_und"] for r in recent]

    gold_open = gold_prices[0]
    silver_open = silver_prices[0]

    gold_hi = max(gold_prices)
    gold_lo = min(gold_prices)
    silver_hi = max(silver_prices)
    silver_lo = min(silver_prices)

    gold_range = (gold_hi - gold_lo) / gold_open * 100
    silver_range = (silver_hi - silver_lo) / silver_open * 100

    # Compare to typical
    gold_candles = hist["metals"]["XAU-USD"]["daily_ohlcv"]
    silver_candles = hist["metals"]["XAG-USD"]["daily_ohlcv"]

    gold_daily_ranges = sorted([(c["high"]-c["low"])/c["open"]*100 for c in gold_candles])
    silver_daily_ranges = sorted([(c["high"]-c["low"])/c["open"]*100 for c in silver_candles])

    gold_med = percentile(gold_daily_ranges, 50)
    silver_med = percentile(silver_daily_ranges, 50)

    print(f"\n  Gold underlying: ${gold_lo:.2f} - ${gold_hi:.2f} (range: {gold_range:.3f}%)")
    print(f"    Typical daily range: {gold_med:.2f}% -> today so far is {gold_range/gold_med*100:.0f}% of typical")
    print(f"    Remaining potential move: {gold_med - gold_range:.2f}% of underlying = {(gold_med - gold_range)*8:.1f}% on GOLD 8x warrant")

    print(f"\n  Silver underlying: ${silver_lo:.2f} - ${silver_hi:.2f} (range: {silver_range:.3f}%)")
    print(f"    Typical daily range: {silver_med:.2f}% -> today so far is {silver_range/silver_med*100:.0f}% of typical")
    print(f"    Remaining potential move: {silver_med - silver_range:.2f}% of underlying = {(silver_med - silver_range)*4.28:.1f}% on SILVER301 warrant")

# ========================================================
# EU SESSION HOURS (09:00-17:25 CET) — subset analysis
# ========================================================
print(f"\n{'='*70}")
print(f"  EU SESSION RANGE ESTIMATE (Avanza 09:00-17:25 CET, ~8.4h)")
print(f"{'='*70}")
print(f"\n  XAG/XAU trade 24/7 but warrants only during EU hours.")
print(f"  EU session captures roughly 35-50% of 24h range (based on metals")
print(f"  volatility clustering around US open 15:30 CET and London AM fix).")
print(f"  ")
print(f"  Estimated EU-session ranges (40% of 24h range as conservative estimate):")

for metal_key in ["XAG-USD", "XAU-USD"]:
    candles = hist["metals"][metal_key]["daily_ohlcv"]
    daily_ranges = sorted([(c["high"]-c["low"])/c["open"]*100 for c in candles])

    med = percentile(daily_ranges, 50)
    p75 = percentile(daily_ranges, 75)
    p90 = percentile(daily_ranges, 90)

    eu_factor = 0.40
    print(f"\n  {metal_key}:")
    print(f"    24h median range: {med:.2f}% -> EU session: ~{med*eu_factor:.2f}%")
    print(f"    24h P75 range:    {p75:.2f}% -> EU session: ~{p75*eu_factor:.2f}%")
    print(f"    24h P90 range:    {p90:.2f}% -> EU session: ~{p90*eu_factor:.2f}%")

    # Warrant impact for EU session
    relevant = {k: v for k, v in POSITIONS.items() if v["underlying"] == metal_key}
    for name, pos in relevant.items():
        lev = pos["leverage"]
        short_name = name.replace("MINI L SILVER AVA ", "SILVER").replace("BULL GULD X8 N", "GOLD 8x")
        print(f"    -> {short_name} ({lev}x): typical EU drop ~{med*eu_factor*lev/2:.1f}%, bad EU drop ~{p90*eu_factor*lev/2:.1f}%")

print(f"\n{'='*70}")
print(f"  KEY TAKEAWAYS")
print(f"{'='*70}")

# XAG stats
xag_candles = hist["metals"]["XAG-USD"]["daily_ohlcv"]
xag_ranges = sorted([(c["high"]-c["low"])/c["open"]*100 for c in xag_candles])
xag_drops = sorted([(c["open"]-c["low"])/c["open"]*100 for c in xag_candles])
xag_med = percentile(xag_ranges, 50)
xag_p90_drop = percentile(sorted([(c["open"]-c["low"])/c["open"]*100 for c in xag_candles]), 90)

xau_candles = hist["metals"]["XAU-USD"]["daily_ohlcv"]
xau_med = percentile(sorted([(c["high"]-c["low"])/c["open"]*100 for c in xau_candles]), 50)
xau_p90_drop = percentile(sorted([(c["open"]-c["low"])/c["open"]*100 for c in xau_candles]), 90)

print(f"""
  1. Silver (XAG) is WILD: median daily range {xag_med:.1f}%, worst days 15-27%
     -> SILVER301 (4.3x): typical day swings {xag_med*4.28:.0f}%, bad day {xag_p90_drop*4.28:.0f}% drop
     -> SILVER79 (1.3x): typical day swings {xag_med*1.34:.0f}%, bad day {xag_p90_drop*1.34:.0f}% drop

  2. Gold (XAU) is calmer: median daily range {xau_med:.1f}%, worst days 7-10%
     -> GOLD 8x: typical day swings {xau_med*8:.0f}%, bad day {xau_p90_drop*8:.0f}% drop
     -> 8x leverage makes gold as volatile as raw silver!

  3. Today's session so far: TINY moves (<0.4% underlying for both)
     -> ~5-10% of a typical day's range used up in first hour
     -> Most of the daily range still ahead (US session 15:30 CET drives metals)
     -> ISM Manufacturing at 16:00 CET could catalyze the remaining range

  4. Your stops (L3 emergency):
     - Gold: 5.7% from stop -> on a P90 bad day, 8x could move {xau_p90_drop*8:.0f}% (CLOSE)
     - Silver79: 7.8% from stop -> very safe even on P90 day
     - Silver301: 8.2% from stop -> on a P90 bad day, 4.3x could move {xag_p90_drop*4.28:.0f}% (CLOSE)
""")
