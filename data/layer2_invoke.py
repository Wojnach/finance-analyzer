import json, datetime, pathlib, requests

ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

# Journal entry
entry = {
    "ts": ts,
    "trigger": "META consensus BUY (63%), SMCI consensus BUY (55%)",
    "regime": "range-bound",
    "reflection": "Range-bound thesis confirmed again. All prices flat in Fri extended hours. MU $427.09->$427.00 (-0.02%), NVDA $189.90->$189.88 (-0.01%). No breakouts forming.",
    "continues": "2026-02-20T23:48:06.071513+00:00",
    "decisions": {
        "patient": {"action": "HOLD", "reasoning": "MU position +0.85% from entry, healthy. AAPL 8B strongest but extended hrs -- wait for Mon regular session. META/SMCI triggers weak."},
        "bold": {"action": "HOLD", "reasoning": "MU/NVDA entries yesterday, both flat near entry. No exits needed -- trend intact. No new breakouts in extended hrs Fri night."}
    },
    "tickers": {
        "BTC-USD": {"outlook": "neutral", "thesis": "Short-term BUY (6B/1S) but 1mo/3mo SELL. Range-bound, no conviction.", "conviction": 0.2, "levels": [65000, 70000]},
        "ETH-USD": {"outlook": "neutral", "thesis": "5B/2S short-term BUY noise, longer-term bearish. Not actionable.", "conviction": 0.1, "levels": []},
        "MU": {"outlook": "bullish", "thesis": "Holding. 6B/1S, 4/7 TFs BUY. Entry $423.42, now $427 (+0.85%). Need regular session to confirm.", "conviction": 0.6, "levels": [420, 435]},
        "NVDA": {"outlook": "bullish", "thesis": "Holding. 4B/1S, 5/7 TFs BUY -- best TF alignment. Flat from entry. Watch for Mon continuation.", "conviction": 0.5, "levels": [185, 195]},
        "AAPL": {"outlook": "bullish", "thesis": "8B/1S (87%), vol 3.3x -- strongest signal on board. Extended hrs only. Must sustain into Mon regular session.", "conviction": 0.4, "levels": [260, 270]},
        "VERI": {"outlook": "bearish", "thesis": "7S/1B, 6/7 TFs SELL, vol 4.4x. Confirmed downtrend, no position.", "conviction": 0.0, "levels": []},
        "AMD": {"outlook": "bearish", "thesis": "6S/2B, 5/7 TFs SELL. Structural deterioration, no position.", "conviction": 0.0, "levels": []}
    },
    "watchlist": [
        "AAPL 8B/1S sustain into Mon regular session with vol confirmation",
        "MU hold above $420 support or break above $435",
        "NVDA breakout above $195 with volume expansion"
    ],
    "prices": {
        "BTC-USD": 67959.45, "ETH-USD": 1966.20, "XAU-USD": 5105.20, "XAG-USD": 84.54,
        "MSTR": 131.35, "PLTR": 134.97, "NVDA": 189.8801, "AMD": 199.90,
        "BABA": 154.54, "GOOGL": 315.55, "AMZN": 209.9986, "AAPL": 264.35,
        "AVGO": 332.4077, "AI": 10.41, "GRRR": 11.37, "IONQ": 32.02,
        "MRVL": 79.47, "META": 654.94, "MU": 427.00, "PONY": 13.58,
        "RXRX": 3.43, "SOUN": 7.835, "SMCI": 32.28, "TSM": 370.49,
        "TTWO": 199.7656, "TEM": 58.60, "UPST": 29.311, "VERI": 2.8301,
        "VRT": 243.60, "QQQ": 608.85, "LMT": 659.79
    }
}

with open("data/layer2_journal.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("Journal entry written")

# Telegram message
msg = """*HOLD* \u00b7 AAPL 8B VERI 7S \u00b7 F&G 7/55

`AAPL $264  BUY   8B/1S/12H BBBSSSB`
`MU   $427  HOLD  6B/1S/14H \u00b7BB\u00b7\u00b7BB`
`NVDA $190  BUY   4B/1S/16H BBBBB\u00b7B`
`META $655  BUY   5B/2S/14H BBBSSB\u00b7`
`SMCI $32   BUY   3B/3S/15H B\u00b7\u00b7\u00b7BSS`
`VERI $2.83 SELL  1B/7S/13H SSSSSSB`
`AMD  $200  SELL  2B/6S/13H SSSSS\u00b7B`
`BTC  $68K  BUY   6B/1S/17H BB\u00b7\u00b7SS\u00b7`
_+6 buy \u00b7 15 hold \u00b7 2 sell_

_P:500K(+0%) MU 19sh \u00b7 B:465K(-7%) MU+NVDA \u00b7 DXY 98\u2191 \u00b7 10Y 4.09\u2191_
Fri extended hrs \u2014 no action. AAPL 8B/vol 3.3x strongest but needs Mon regular session. MU/NVDA flat, holding."""

# Save message locally
log = pathlib.Path("data/telegram_messages.jsonl")
with open(log, "a", encoding="utf-8") as f:
    f.write(json.dumps({"ts": ts, "text": msg}, ensure_ascii=False) + "\n")
print("Telegram message saved")

# Send via Telegram
config = json.load(open("config.json"))
resp = requests.post(
    f"https://api.telegram.org/bot{config['telegram']['token']}/sendMessage",
    json={"chat_id": config["telegram"]["chat_id"], "text": msg, "parse_mode": "Markdown"}
)
print(f"Telegram response: {resp.status_code} {resp.json().get('ok', False)}")
