# Cross-critique — data-external

## Codex findings Claude missed

| Codex finding | Why Claude missed it |
|---|---|
| `data_collector.py:334-339` — `as_completed(..., timeout=...)` doesn't actually stop hung workers. `f.cancel()` doesn't stop futures already running, and exiting `with ThreadPoolExecutor()` calls `shutdown(wait=True)` → hung yfinance call still blocks `collect_timeframes()`. **BUG-179 fix is ineffective.** | Claude reviewed `data_collector.py` for stale-data risks but didn't audit the timeout enforcement. Fundamental misunderstanding of `concurrent.futures` semantics — Codex caught it. |
| `crypto_macro_data.py:137` — `total_pain` is non-negative but `max_pain_value` initialized to `-1`. `total_pain < max_pain_value` is False for every candidate after the first. **`max_pain_strike` always pinned to first strike → bogus max-pain level.** | Claude reviewed `crypto_macro_data.py` for the agent_summary stale-price bug but didn't audit the max-pain search loop. |
| `data_refresh.py:30-31` — `download_klines()` writes into `.../binance/futures` but request goes to **spot `BINANCE_BASE` endpoint**. Downstream consumers treating files as futures bars silently mix spot candles with futures-only metrics (funding/OI). | Claude reviewed precompute paths but didn't audit `data_refresh.py` separately. Wrong-endpoint bug. |

## Claude findings Codex missed

| Claude finding | Why Codex missed it |
|---|---|
| `microstructure_state.py:205-213` — `persist_state()` calls `get_microstructure_state(ticker)` which calls `record_ofi(ticker, ofi)` → **double-counts OFI on every persist tick**. Inflates `_ofi_history` buffer and corrupts z-score distribution. | Codex didn't audit `persist_state` for side effects in a "read" function. Subtle Python bug — read with side effects is a class of bug Codex sometimes misses. |
| `data/crypto_data.py:184-185` vs `mstr_precompute.py:35,37` — Hardcoded MSTR BTC holdings differ by 6% (499K vs 471K) and shares outstanding differ by 25% (229M vs 287M). NAV premium math off by 20%+. **Two sources of truth diverging.** | Codex didn't compare across files. Cross-module consistency check Claude does well. |
| `crypto_precompute.py:185` — `out[key_funding] = float(r.json().get("lastFundingRate", 0))` — `0` default conflates missing field, partial response, and actual zero funding. | Codex didn't audit the funding fallback default. |
| `earnings_calendar.py:48-53` — Self-documented bypass of `_daily_budget_used` counter for AV earnings calls — silently exhausts 25/day quota. | Codex didn't read the comment that calls out the bug. |
| `metals_precompute.py:407-409, 458-460` — COT fetch uses raw `requests.get()` with no retry. CFTC SOCRATA single transient failure marks COT failed for 7 days. | Codex didn't audit the COT fetch path resilience. |
| `crypto_macro_data.py:208-218` — Reads `agent_summary_compact.json` for live BTC/gold prices; violates "always pull live" rule and can be hours stale. | Codex didn't have project-rule context about live-prices-first. |
| `sentiment.py:859` — `ab_key = f"{short}:{datetime.now(UTC).isoformat()}"` non-unique under microsecond concurrency. | Codex didn't probe the AB log key collision risk. |
| `session_calendar.py:82-89` — `_make_session_end()` produces past datetime when called after session close. | Codex didn't audit session_calendar. |

## Disagreements

### Both flagged `econ_dates.py` 14:00 UTC for all events
- **Codex** P2 — explicit: "FOMC days the error is 4-5 hours."
- **Claude** P1 — explicit: "Pre-CPI risk-off doesn't fire until after the print."
- **Same bug, different priority.** Reconcile to **P1**: both reviewers agree on the bug; the higher-priority interpretation wins.

### Both flagged `metals_precompute.py` empty payloads on intermediate runs
- **Codex** P1 (line 137-145): "deep-context files will oscillate between complete and mostly empty every other cycle."
- **Claude** P1 (line 149-256): "Within the 4h cache window, ETF context vanishes — looks like fetch failure."
- **Same bug, same priority.** Strong cross-confirmation.

## What both missed (likely)

- **`fx_rates.py`** — neither flagged anything. USD/SEK is critical for SEK-denominated portfolio valuations and warrant trades. Stale FX in `agent_summary` would silently mis-price every position.
- **`bert_sentiment.py`** — neither flagged. Sentiment model loading concurrency, GPU sharing, and confidence calibration deserve attention.
- **`fomc_dates.py` hardcoded calendar exhaustion** — same risk as Claude's `econ_calendar.py:137` finding (BUY-on-stale-calendar). Neither reviewer extended the same logic to FOMC dates.
- **`gold_precompute.py` / `silver_precompute.py` / `mstr_precompute.py`** — Claude found the MSTR holdings divergence (P0); neither audited gold/silver precomputes for similar hardcoded constants.

## Reconciled verdict

**P0 (must fix — silent data corruption):**
1. **(Claude)** `microstructure_state.py:205-213` `persist_state()` double-counts OFI — corrupts z-score distribution.
2. **(Claude)** `data/crypto_data.py:184-185` MSTR BTC holdings/shares diverge from `mstr_precompute.py` by 6%/25%. NAV premium signal can flip.
3. **(Codex + Claude — same bug)** `metals_precompute.py:137-145/149-256` un-refreshed sources yield None instead of carrying forward last successful values. **Confirmed by both reviewers — high confidence.**

**P1:**
4. (Codex + Claude same bug) `econ_dates.py:155, 180, 224, 273` all events pinned to 14:00 UTC — pre/post-event windows wrong by 4-5h on FOMC, 1.5h on CPI/NFP.
5. (Codex) `data_collector.py:334-339` BUG-179 fix doesn't actually prevent yfinance hangs.
6. (Claude) `crypto_precompute.py:185` funding rate fallback to 0.0 silently masks missing field.
7. (Claude) `earnings_calendar.py:48-53` AV calls bypass budget counter → silent quota exhaustion.
8. (Claude) `metals_precompute.py:407-409, 458-460` COT fetch no retry → 7-day silent failure.
9. (Claude) `crypto_macro_data.py:208-218` reads stale `agent_summary_compact.json` for prices.
10. (Codex) `data_refresh.py:30-31` futures backfill uses spot endpoint.
11. (Codex) `crypto_macro_data.py:137` max_pain_value=-1 init bug → wrong max-pain strike.

**P2:**
12. (Claude) `sentiment.py:859` ab_key collision risk.
13. (Claude) `macro_context.py:287-292` raw open() on config.json.
14. (Claude) `session_calendar.py:82-89` past datetime on session_end after close.
