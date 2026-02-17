# ISKBETS Feature Plan

Intraday "quick gamble" mode for the trading system. User enables it with a target amount
and ticker(s), system monitors realtime and sends Telegram alerts for entry + exit.

## Concept

1. User enables ISKBETS mode targeting a ticker + amount (e.g. MSTR, 100K SEK)
2. System polls more frequently (~30s vs normal 3min), looking for a good entry
3. Layer 2 sends a Telegram alert: "Good entry on MSTR — consider buying"
4. **User buys manually on Avanza**, then confirms to the system: "bought MSTR at $X for 100K SEK"
5. System creates a temp ISKBETS portfolio mirroring the real position
6. System monitors the position and alerts when exit conditions are met
7. User sells manually on Avanza

The ISKBETS portfolio is a shadow of the real Avanza trade — not the simulated 500K portfolios.
It tracks real entry price and real SEK amount so P&L reflects actual exposure.

## Todos (task IDs from session where this was planned)

1. Design ISKBETS feature spec (parameters, entry criteria, exit strategy definition)
2. Build enable/disable mechanism (likely `data/iskbets_config.json`)
3. Implement turbo monitoring loop (shorter poll interval for target tickers)
4. Add entry signal logic + exit strategy calculator (store in `data/iskbets_state.json`)
5. Add Telegram notifications — entry alert with full exit plan, exit alert with result

## Key Design Decisions (to resolve in task 1)

- How is it enabled? CLI flag? Edit a config file? Simple toggle script?
- What counts as a "good entry"? Intraday signals: RSI, BB touch, volume spike, EMA cross?
- Exit strategy: ATR-based stop (2x ATR), % profit target (1.5-2%), time-based (market close)?
- How does user confirm the buy? Telegram reply? Edit a file? Run a command?
- Auto-disable after market close or after trade completes + confirmed exit?

## Telegram Two-Way Communication

Using polling (not webhook) — works on Steam Deck with no public IP.

- Poller calls `getUpdates` every 5s
- Parses incoming messages from user
- Commands:
  - `bought MSTR 129.50 100000` → open ISKBETS position, start monitoring
  - `sold` → close position, log P&L
  - `cancel` → abort ISKBETS mode
- Writes state to `data/iskbets_state.json`
- Runs as a background process alongside the existing fast loop

## Files (to be created)

- `data/iskbets_config.json` — enabled flag, amount, tickers, expiry
- `data/iskbets_state.json` — active trade state, exit plan
- `portfolio/iskbets.py` — monitoring logic (or extend existing main.py)
- `portfolio/telegram_poller.py` — background process, polls getUpdates every 5s, parses commands
