import json, datetime, pathlib, requests

ts_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

# --- Journal Entry ---
entry = {
    "ts": ts_now,
    "trigger": "XAU-USD BUY->HOLD flip (sustained) + crypto 1h check-in",
    "regime": "range-bound",
    "reflection": "Previous range-bound thesis correct. BTC +0.2% ($67,853->$68,052), ETH +0.3% ($1,962->$1,968), MU flat $426.70. Gold BUY->HOLD flip is noise (price unchanged at $5,113). No breakouts forming.",
    "continues": "2026-02-21T01:24:09.409515+00:00",
    "decisions": {
        "patient": {"action": "HOLD", "reasoning": "MU position stable +0.77% from $423.42 entry. Weekend overnight, stocks stale. Crypto shows weak BUY but no volume (0.4-0.66x), split TFs."},
        "bold": {"action": "HOLD", "reasoning": "MU+NVDA positions from yesterday holding fine. No overnight breakouts. Room for 1 more pos. TSM/VRT 7/7 TF alignment top Monday candidates."}
    },
    "tickers": {
        "BTC-USD": {"outlook": "neutral", "thesis": "Range-bound. Short TFs BUY but 1mo/3mo SELL. Vol 0.66x dead. No breakout.", "conviction": 0.2, "levels": [65000, 70000]},
        "ETH-USD": {"outlook": "neutral", "thesis": "Same as BTC. Weak BUY, vol 0.4x, longer TFs bearish.", "conviction": 0.1, "levels": [1900, 2050]},
        "MU": {"outlook": "bullish", "thesis": "Holding above $420 support. 6B/1S, 5/7 TFs BUY. Position performing.", "conviction": 0.6, "levels": [420, 440]},
        "NVDA": {"outlook": "bullish", "thesis": "6/7 TFs BUY, vol 1.92x. Trend intact, RSI 59.8. Watch continuation Mon.", "conviction": 0.6, "levels": [185, 195]},
        "TSM": {"outlook": "bullish", "thesis": "7/7 TFs BUY. Strongest alignment on board. 7B/1S. Top Monday candidate.", "conviction": 0.7, "levels": [365, 380]},
        "VRT": {"outlook": "bullish", "thesis": "7/7 TFs BUY, 6B/2S. Second strongest setup. Monday candidate.", "conviction": 0.6, "levels": [240, 250]}
    },
    "watchlist": [
        "TSM Monday open - confirm 7/7 TF holds + vol >1.5x for entry",
        "VRT Monday open - 7/7 TFs, need session volume confirm",
        "MU hold above $420 support"
    ],
    "prices": {
        "BTC-USD": 68052.45, "ETH-USD": 1968.42, "XAU-USD": 5112.76, "XAG-USD": 84.53,
        "MSTR": 131.03, "PLTR": 134.52, "NVDA": 189.9999, "AMD": 199.8994,
        "BABA": 154.61, "GOOGL": 315.41, "AMZN": 210.0494, "AAPL": 264.38,
        "AVGO": 332.96, "AI": 10.44, "GRRR": 11.4, "IONQ": 31.935,
        "MRVL": 79.46, "META": 655.66, "MU": 426.7, "PONY": 13.5883,
        "RXRX": 3.4365, "SOUN": 7.8398, "SMCI": 32.27, "TSM": 370.77,
        "TTWO": 199.49, "TEM": 58.57, "UPST": 29.42, "VERI": 2.8598,
        "VRT": 243.7, "QQQ": 608.76, "LMT": 658.1588
    }
}

journal_path = pathlib.Path("data/layer2_journal.jsonl")
with open(journal_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("Journal entry written.")

# --- Telegram Message ---
msg = (
    "*HOLD* \u00b7 TSM 7B NVDA 7B \u00b7 F&G 8/55\n"
    "\n"
    "`NVDA $190  BUY  7B/2S/12H BBBBB\u00b7B`\n"
    "`TSM  $371  BUY  7B/1S/13H BBBBBBB`\n"
    "`MU   $427  BUY  6B/1S/14H BBB\u00b7\u00b7BB`\n"
    "`VRT  $244  BUY  6B/2S/13H BBBBBBB`\n"
    "`GRRR $11   BUY  7B/1S/13H BBBSSS\u00b7`\n"
    "`BTC  $68K  BUY  6B/1S/17H BBB\u00b7SS\u00b7`\n"
    "`RXRX $3.4  SELL 1B/6S/14H SSSSSSS`\n"
    "_+15 hold \u00b7 9 sell_\n"
    "\n"
    "_P:500K \u00b7 B:465K(-7%) MU+NVDA \u00b7 DXY 98\u2191 \u00b7 10Y 4.09\u2191_\n"
    "Weekend overnight. Crypto flat, vol dead. MU/NVDA holding. "
    "TSM+VRT 7/7 TF alignment \u2014 top Monday candidates."
)

# Save telegram message — category: "analysis" (HOLD) or "trade" (BUY/SELL)
category = "analysis"  # HOLD message — saved only, not sent to Telegram
msg_log = pathlib.Path("data/telegram_messages.jsonl")
with open(msg_log, "a", encoding="utf-8") as f:
    f.write(json.dumps({"ts": ts_now, "text": msg, "category": category, "sent": category == "trade"}, ensure_ascii=False) + "\n")
print("Telegram message saved.")

# Only send to Telegram if a trade was executed
if category == "trade":
    config = json.load(open("config.json"))
    resp = requests.post(
        f"https://api.telegram.org/bot{config['telegram']['token']}/sendMessage",
        json={"chat_id": config["telegram"]["chat_id"], "text": msg, "parse_mode": "Markdown"}
    )
    print(f"Telegram sent: {resp.status_code} {resp.json().get('ok', False)}")
else:
    print("HOLD message — saved to JSONL only, not sent to Telegram")
