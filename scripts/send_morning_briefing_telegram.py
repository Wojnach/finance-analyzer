"""Send the morning briefing summary to Telegram."""
import json
import requests

config = json.load(open("config.json"))
token = config["telegram"]["token"]
chat_id = config["telegram"]["chat_id"]

msg = """*🌅 MORNING BRIEFING — May 26, 2026*

*Outlook:* NEUTRAL (conf 0.55)
Memorial Day thin liquidity. US-Iran deal binary risk. Thursday PCE/GDP.

*Key Levels:*
• XAU $4,562 (S:4441 R:4577) — Central banks buying 244t Q1
• XAG $77.5 (S:73 R:83) — GSR 55:1, 6th yr supply deficit
• BTC $77.2K (S:75K R:80K) — Broke descending channel, testing $80K
• ETH $2,128 (S:2085 R:2170) — RSI 36 near oversold
• MSTR $160 (S:146 R:170) — Gap risk Tue from BTC weekend move

*Trade Ideas:* ALL HOLD
No edge — consensus accuracy ≤50% for 4/5 instruments.
Best: BTC 52.7%. Worst: XAG 49.6%.

*Tonight's Improvements (shipped):*
✅ Per-ticker signal overrides:
  - RE-ENABLED: williams\\_vix\\_fix (XAU 76.5%, XAG 60.9%), realized\\_skewness (XAU 60.3%), credit\\_spread\\_risk (BTC/ETH 57.4%)
  - DISABLED: statistical\\_jump\\_regime (XAU 50.2%), cubic\\_trend\\_persistence (XAG 41.6%), realized\\_skewness (XAG 42.9%), crypto\\_evrp (BTC 40.2%)
  - WEIGHT BOOSTS: drift\\_regime\\_gate 1.4x (68.1%), williams\\_vix\\_fix 1.3x
  - WEIGHT CUTS: statistical\\_jump\\_regime 0.7x, crypto\\_evrp 0.6x

*Top Priority (not yet impl):*
1. Wire walk-forward weights into live consensus (+2-4%)
2. IC-based signal weighting (+3-5% metals/ETH)
3. Regime-conditional signal selection (+2-3%)

*Risk Warnings:*
⚠ Iran deal binary — 3-5% gap either direction
⚠ Thursday PCE: >3.2% = hawkish shock
⚠ Memorial Day — avoid new positions until Tue
⚠ Layer 2 agent: 14 triggers, 0 journal entries since May 20
⚠ Avanza session expired since May 23"""

r = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
    timeout=30,
)
print(f"Status: {r.status_code}")
if r.status_code != 200:
    print(r.text)
