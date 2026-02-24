import json, datetime, pathlib, requests

BASE = pathlib.Path("Q:/finance-analyzer/data")
now = datetime.datetime.now(datetime.timezone.utc)
ts_now = now.isoformat()

# === 1. Journal Entry ===
entry = {
    "ts": ts_now,
    "trigger": "BTC-USD consensus SELL (51%)",
    "regime": "range-bound",
    "reflection": "Prices completely flat since 03:15 UTC - BTC $67,772->$67,780 (+0.01%), ETH unchanged. Range-bound thesis holds for 11th consecutive check.",
    "continues": "2026-02-21T03:15:01.809821+00:00",
    "decisions": {
        "patient": {"action": "HOLD", "reasoning": "MU +0.8% from entry, 6B/1S with 5/7 TFs BUY. BTC trigger is 4B/4S coin flip. Stocks closed."},
        "bold": {"action": "HOLD", "reasoning": "MU+NVDA positions flat. Both signals strong (6B/1S and 7B/2S). BTC 4B/4S split - noise. No breakout in crypto overnight."}
    },
    "tickers": {
        "BTC-USD": {"outlook": "neutral", "thesis": "4B/4S split - coin flip. MACD -30 deeply negative, but F&G 8 + mean_reversion BUY. No trend, no trade.", "conviction": 0.1, "levels": [65000, 70000]},
        "ETH-USD": {"outlook": "neutral", "thesis": "6B/3S BUY but 1/7 TFs BUY vs 4/7 SELL. Contrarian vs structural conflict.", "conviction": 0.1, "levels": [1900, 2050]},
        "MU": {"outlook": "bullish", "thesis": "6B/1S, 5/7 TFs BUY. Both portfolios long. Holding above $420 support.", "conviction": 0.7, "levels": [420, 435]},
        "NVDA": {"outlook": "bullish", "thesis": "7B/2S, 6/7 TFs BUY. Bold long at $190. Trend structure intact.", "conviction": 0.6, "levels": [185, 195]},
        "TSM": {"outlook": "bullish", "thesis": "7B/1S, 7/7 TFs BUY - strongest alignment. Watching Monday open.", "conviction": 0.7, "levels": [365, 380]},
        "VRT": {"outlook": "bullish", "thesis": "6B/2S, 7/7 TFs BUY. Watching Monday open for volume.", "conviction": 0.6, "levels": [240, 250]},
        "RXRX": {"outlook": "bearish", "thesis": "1B/6S, 7/7 TFs SELL. Unanimous bearish.", "conviction": 0.5, "levels": [3.2, 3.6]}
    },
    "watchlist": [
        "TSM Monday open - confirm 7/7 TF + vol >1.5x for Patient BUY",
        "VRT Monday open - confirm 7/7 TF + vol >1.5x for Bold BUY",
        "MU hold above $420 support; NVDA hold above $185"
    ],
    "prices": {
        "BTC-USD": 67780.32, "ETH-USD": 1961.67, "XAU-USD": 5111.01, "XAG-USD": 84.55,
        "MSTR": 131.03, "PLTR": 134.52, "NVDA": 189.9999, "AMD": 199.8994,
        "BABA": 154.61, "GOOGL": 315.41, "AMZN": 210.0494, "AAPL": 264.38,
        "AVGO": 332.96, "AI": 10.44, "GRRR": 11.4, "IONQ": 31.935,
        "MRVL": 79.46, "META": 655.66, "MU": 426.7, "PONY": 13.5883,
        "RXRX": 3.4365, "SOUN": 7.8398, "SMCI": 32.27, "TSM": 370.77,
        "TTWO": 199.49, "TEM": 58.57, "UPST": 29.42, "VERI": 2.8598,
        "VRT": 243.7, "QQQ": 608.76, "LMT": 658.1588
    }
}

jpath = BASE / "layer2_journal.jsonl"
with open(jpath, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("Journal entry written.")

# === 2. Telegram ===
msg = (
    "*HOLD* \u00b7 NVDA 7B TSM 7B \u00b7 F&G 8/55\n"
    "\n"
    "`MU   $427  BUY   6B/1S/14H BBB\u00b7\u00b7BB`\n"
    "`NVDA $190  BUY   7B/2S/12H BBBBB\u00b7B`\n"
    "`BTC  $68K  SELL  4B/4S/16H S\u00b7B\u00b7SS\u00b7`\n"
    "`TSM  $371  BUY   7B/1S/13H BBBBBBB`\n"
    "`VRT  $244  BUY   6B/2S/13H BBBBBBB`\n"
    "`RXRX $3.4  SELL  1B/6S/14H SSSSSSS`\n"
    "_+17 hold \u00b7 8 sell_\n"
    "\n"
    "_P:500K MU \u00b7 B:465K(-7%) MU+NVDA \u00b7 DXY 98\u2191 \u00b7 10Y 4.09\u2191_\n"
    "BTC 4B/4S coin-flip triggered \u2014 noise. Crypto flat, stocks closed. "
    "MU+NVDA trend intact; TSM+VRT 7/7 TFs \u2014 watching Mon open."
)

# Save locally — category: "analysis" (HOLD) or "trade" (BUY/SELL)
category = "analysis"  # HOLD message — saved only, not sent to Telegram
with open(BASE / "telegram_messages.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"ts": ts_now, "text": msg, "category": category, "sent": category == "trade"}, ensure_ascii=False) + "\n")
print("Telegram message saved.")

# Only send to Telegram if a trade was executed
if category == "trade":
    config = json.load(open("Q:/finance-analyzer/config.json", encoding="utf-8"))
    resp = requests.post(
        f"https://api.telegram.org/bot{config['telegram']['token']}/sendMessage",
        json={"chat_id": config["telegram"]["chat_id"], "text": msg, "parse_mode": "Markdown"}
    )
    print(f"Telegram: {resp.status_code} {resp.json().get('ok', False)}")
else:
    print("HOLD message — saved to JSONL only, not sent to Telegram")
