import json

with open("Q:/finance-analyzer/data/agent_summary.json", encoding="utf-8") as f:
    data = json.load(f)

signals = data.get("signals", {})
print(f"Timestamp: {data['timestamp']}")
print(f"FX Rate (USD/SEK): {data['fx_rate']}")
print(f"Total tickers: {len(signals)}")
print()

header = (
    f"{'Ticker':<14} {'Act':<5} {'Conf':>5} {'WConf':>5} {'Price':>11} "
    f"{'RSI':>5} {'MACD':>9} {'BB':>8} {'ATR%':>5} {'Regime':<13} "
    f"{'B':>2} {'S':>2} {'Tot':>3} {'Vol':>5} {'Min':<5} {'ML':<5} {'F&G':>4}"
)
print(header)
print("-" * len(header))

for ticker, d in signals.items():
    ex = d.get("extra", {})
    rsi = d.get("rsi")
    macd = d.get("macd_hist")
    atr = d.get("atr_pct")
    vol = ex.get("volume_ratio")
    fg = ex.get("fear_greed")

    rsi_s = f"{rsi:5.1f}" if rsi is not None else "  N/A"
    macd_s = f"{macd:9.2f}" if macd is not None else "      N/A"
    atr_s = f"{atr:5.2f}" if atr is not None else "  N/A"
    vol_s = f"{vol:5.2f}" if vol is not None else "  N/A"
    fg_s = f"{fg:4}" if fg is not None else " N/A"

    print(
        f"{ticker:<14} {d.get('action',''):<5} {d.get('confidence',0):5.3f} "
        f"{d.get('weighted_confidence',0):5.3f} {d.get('price_usd',0):11.3f} "
        f"{rsi_s} {macd_s} {d.get('bb_position',''):>8} {atr_s} "
        f"{d.get('regime',''):13} {ex.get('_buy_count',0):2} {ex.get('_sell_count',0):2} "
        f"{ex.get('_total_applicable',0):3} {vol_s} {ex.get('ministral_action',''):5} "
        f"{ex.get('ml_action',''):5} {fg_s}"
    )

print()
print("=" * 80)
print("TIMEFRAMES (B=BUY S=SELL H=HOLD)")
print("=" * 80)
tf_order = ["now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]
h = f"{'Ticker':<14} " + " ".join(f"{tf:>4}" for tf in tf_order)
print(h)
print("-" * len(h))
for ticker, tf_data in data.get("timeframes", {}).items():
    vals = []
    for tf in tf_order:
        v = tf_data.get(tf, {}).get("action", "?")
        vals.append(f"{'B' if v=='BUY' else 'S' if v=='SELL' else 'H':>4}")
    print(f"{ticker:<14} " + " ".join(vals))
