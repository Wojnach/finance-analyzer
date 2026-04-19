# Agent Review: metals-core

## P1 Findings
1. **Hardware trailing stop never placed** — tuple/dict mismatch in _handle_buy_fill (metals_loop.py:4740-4761). Confidence: 100
2. **EOD fish sell blocked** by EMERGENCY_SELL_ENABLED=False (metals_loop.py). Confidence: 88

## P2 Findings
1. **_load_json_state non-atomic read** — violates CLAUDE.md rule 4 (metals_loop.py:538-554). Confidence: 95
2. **4 additional non-atomic reads** scattered through 7634-line file
3. **_silver_alerted_levels not cleared** on new position entry (metals_loop.py:859)
4. **_underlying_prices dict mutation** without synchronization (metals_loop.py:1096)

## P3 Findings
1. HARD_STOP_CERT_PCT=0.05 too tight for 5x warrants (fin_snipe_manager.py:61)
2. exit_optimizer always uses linear approximation, knockout flags never raised
3. price_targets fixed seed 42 kills MC randomness
4. iskbets highest_price fallback overstates P&L
5. Config read at import time, silent failure on partial file
