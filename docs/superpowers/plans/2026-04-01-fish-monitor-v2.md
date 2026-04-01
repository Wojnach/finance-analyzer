# Fish Monitor v2 — Bake Today's Lessons Into Code

## Context
April 1 fishing session: +389 SEK, 10 trades, 17 lessons. The inline monitoring
scripts worked but were throwaway — not using SmartFishMonitor properly. Key
findings: combined RSI+MC exit works (66.7% backtest win rate), MC whipsaws in
ranging need 2-check stability, RSI entry gate too strict, news/events ignored.

## Backtest Evidence
- Combined LONG exit (RSI>62 + SMA slope<0): 66.7% win rate, -0.41% avg 2h return
- RSI>70 alone: 28.6% de-duped win rate — NOT reliable
- SHORT exit inverse (RSI<38 + slope>0): 33.3% — does NOT work, asymmetric
- Recommendation: implement LONG combined exit, keep RSI<30 for SHORT solo

## Liberation Day (April 2)
- Last year: silver -8.9% in one week, gold/silver ratio >100
- This year: SCOTUS changed tariff legal framework, no major shock expected
- Strategy: avoid new longs on April 2, wait for reaction, buy dips

## Batch 1: SmartFishMonitor exit logic
Files: portfolio/fish_monitor_smart.py

Changes:
1. Add mc_history tracking (list of last 3 MC values)
2. Implement combined LONG exit: RSI>62 AND MC<35%
3. Keep SHORT exit as RSI<30 solo (backtest shows combined doesn't work)
4. Add metals loop disagreement counter — warn at 2+ consecutive disagreements
5. Add macro event flag — read news_event action from agent_summary

## Batch 2: SmartFishMonitor entry logic + display
Files: portfolio/fish_monitor_smart.py

Changes:
1. Entry requires both loops agree + MC stable for 2 checks (>70% or <30%)
2. Remove RSI entry gate — RSI too noisy around 50
3. Shorter cooldown (60s) after combined exit (high confidence reversal)
4. Display: show news_event action, metals loop agreement, trusted signal votes for overnight decisions

## Batch 3: Tomorrow's playbook
Files: data/liberation_day_playbook.json (NEW)

Pre-computed trading rules for April 2:
- No new LONG entries before 14:30 CET (wait for US reaction)
- If gold/silver ratio spikes >65 → strong silver BUY signal
- Wider stops (-3% underlying instead of -2%)
- Preferred instruments ranked by barrier safety

## Batch 4: Tests
Files: tests/test_fish_monitor_smart.py (update)

- Test combined exit fires correctly
- Test MC stability check
- Test metals loop disagreement detection
- Test asymmetric exit (LONG uses combined, SHORT uses RSI<30)
