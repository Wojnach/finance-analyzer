Reading additional input from stdin...
OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\fa-adv-2026-05-11
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019e17a7-4736-7242-99e5-f52266e6e778
--------
user
You are doing an ADVERSARIAL code review of the data-external subsystem (data collectors, sentiment, microstructure, fundamentals) of a quantitative trading system at Q:\finance-analyzer. Sandbox: read-only.

In-scope files:
- portfolio/data_collector.py
- portfolio/fear_greed.py
- portfolio/sentiment.py
- portfolio/bert_sentiment.py
- portfolio/alpha_vantage.py
- portfolio/futures_data.py
- portfolio/funding_rate.py
- portfolio/fx_rates.py
- portfolio/onchain_data.py
- portfolio/news_keywords.py
- portfolio/social_sentiment.py
- portfolio/crypto_macro_data.py
- portfolio/crypto_scheduler.py
- portfolio/earnings_calendar.py
- portfolio/econ_dates.py
- portfolio/fomc_dates.py
- portfolio/seasonality.py
- portfolio/seasonality_updater.py
- portfolio/session_calendar.py
- portfolio/price_source.py
- portfolio/http_retry.py
- portfolio/api_utils.py
- portfolio/data_refresh.py
- portfolio/forecast_signal.py
- portfolio/indicators.py
- portfolio/metals_orderbook.py
- portfolio/microstructure.py
- portfolio/microstructure_state.py
- portfolio/metals_cross_assets.py
- portfolio/tickers.py

Project rules:
- Sources: Binance spot+FAPI (crypto), Alpaca (US stocks), yfinance/Avanza (metals via FAPI), Alpha Vantage (25/day), NewsAPI (100/day), FRED (treasury yields), BGeometrics (15/day BTC on-chain).
- HTTP retry: exponential backoff via http_retry.py — bypassing breaks rate-limit handling.
- Live prices first; cache TTLs must be honored.
- Atomic I/O via file_utils. JSON state file writes never raw.
- Binance: 10m candle interval does NOT exist (error -1120). Use 5m.
- DST-aware market hours.

Adversarial focus:
1. Rate-limit math: requests-per-day counter not persistent across restarts; multi-process counter race.
2. Timezone bugs: tz-naive vs tz-aware datetime mixing; market hours wrong for DST transitions.
3. HTTP retry: backoff too aggressive (DoS source), or too lenient (hangs loop), or swallows 4xx as retryable.
4. Sentiment: BERT model fallback to wrong tokenizer; sentiment score normalization (0..1 vs -1..+1) inconsistent.
5. Funding rate sign: shorts pay vs longs pay convention reversed.
6. fx_rates: stale FX cached > 24h, no validation that rate is sane.
7. price_source: CL=F → Binance FAPI fallback chain — silent fallback to wrong instrument when primary fails.
8. Microstructure: OFI/VPIN window bias, depth imbalance sign convention, persistence of rolling state across process restart.
9. Indicators: pandas-ta vs talib drift; warm-up bars not respected.
10. Onchain BTC: MVRV / SOPR cache key without timestamp -> stale.
11. news_keywords / social_sentiment: keyword regex injection from feed content.
12. tickers.py: ticker normalization (XAG=F vs XAG-USD vs SILVER) inconsistent across modules.

Output format — strict markdown:
# Codex adversarial review: data-external
## Summary
## P0 — Blockers
## P1 — High
## P2 — Medium
## P3 — Low
## Tests missing

Be concrete. Cite line numbers. No false-positive padding.
codex
Reviewing the in-scope data-external modules for concrete failure modes around rate limits, time handling, retries, fallback chains, and state persistence. I’ll read the target files first, map the call paths, then extract only findings that are demonstrably supported by the code.
  → Inspect in-scope files and map data-external call paths
  • Validate adversarial focus areas against concrete code and line numbers
  • Draft severity-ordered review with missing-test gaps
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"retry|backoff|timezone|tz|datetime|funding|fear|greed|BERT|tokenizer|MVRV|SOPR|cache|TTL|Alpha Vantage|NewsAPI|Binance|5m|10m|OFI|VPIN|imbalance|warm-up|talib|pandas_ta|XAG|CL=F|fallback|normalize|normaliz|regex|re\\.compile|re\\.search|re\\.findall|keyword|state|atomic|write_json|json.dump|pickle|market hours|DST|session|FOMC|earnings|FRED|fx\" Q:\\finance-analyzer\\portfolio\\data_collector.py Q:\\finance-analyzer\\portfolio\\fear_greed.py Q:\\finance-analyzer\\portfolio\\sentiment.py Q:\\finance-analyzer\\portfolio\\bert_sentiment.py Q:\\finance-analyzer\\portfolio\\alpha_vantage.py Q:\\finance-analyzer\\portfolio\\futures_data.py Q:\\finance-analyzer\\portfolio\\funding_rate.py Q:\\finance-analyzer\\portfolio\\fx_rates.py Q:\\finance-analyzer\\portfolio\\onchain_data.py Q:\\finance-analyzer\\portfolio\\news_keywords.py Q:\\finance-analyzer\\portfolio\\social_sentiment.py Q:\\finance-analyzer\\portfolio\\crypto_macro_data.py Q:\\finance-analyzer\\portfolio\\crypto_scheduler.py Q:\\finance-analyzer\\portfolio\\earnings_calendar.py Q:\\finance-analyzer\\portfolio\\econ_dates.py Q:\\finance-analyzer\\portfolio\\fomc_dates.py Q:\\finance-analyzer\\portfolio\\seasonality.py Q:\\finance-analyzer\\portfolio\\seasonality_updater.py Q:\\finance-analyzer\\portfolio\\session_calendar.py Q:\\finance-analyzer\\portfolio\\price_source.py Q:\\finance-analyzer\\portfolio\\http_retry.py Q:\\finance-analyzer\\portfolio\\api_utils.py Q:\\finance-analyzer\\portfolio\\data_refresh.py Q:\\finance-analyzer\\portfolio\\forecast_signal.py Q:\\finance-analyzer\\portfolio\\indicators.py Q:\\finance-analyzer\\portfolio\\metals_orderbook.py Q:\\finance-analyzer\\portfolio\\microstructure.py Q:\\finance-analyzer\\portfolio\\microstructure_state.py Q:\\finance-analyzer\\portfolio\\metals_cross_assets.py Q:\\finance-analyzer\\portfolio\\tickers.py" in Q:\finance-analyzer
 succeeded in 682ms:
Q:\finance-analyzer\portfolio\data_refresh.py
Q:\finance-analyzer\portfolio\digest.py
Q:\finance-analyzer\portfolio\econ_dates.py
Q:\finance-analyzer\portfolio\data_collector.py
Q:\finance-analyzer\portfolio\daily_digest.py
Q:\finance-analyzer\portfolio\decision_outcome_tracker.py
Q:\finance-analyzer\portfolio\cumulative_tracker.py
Q:\finance-analyzer\portfolio\crypto_scheduler.py
Q:\finance-analyzer\portfolio\earnings_calendar.py
Q:\finance-analyzer\portfolio\crypto_precompute.py
Q:\finance-analyzer\portfolio\crypto_macro_data.py
Q:\finance-analyzer\portfolio\cost_model.py
Q:\finance-analyzer\portfolio\config_validator.py
Q:\finance-analyzer\portfolio\claude_gate.py
Q:\finance-analyzer\portfolio\circuit_breaker.py
Q:\finance-analyzer\portfolio\iskbets.py
Q:\finance-analyzer\portfolio\bigbet.py
Q:\finance-analyzer\portfolio\forecast_accuracy.py
Q:\finance-analyzer\portfolio\bert_sentiment.py
Q:\finance-analyzer\portfolio\fomc_dates.py
Q:\finance-analyzer\portfolio\backtester.py
Q:\finance-analyzer\portfolio\focus_analysis.py
Q:\finance-analyzer\portfolio\avanza_tracker.py
Q:\finance-analyzer\portfolio\fish_monitor_smart.py
Q:\finance-analyzer\portfolio\avanza_session.py
Q:\finance-analyzer\portfolio\fish_instrument_finder.py
Q:\finance-analyzer\portfolio\instrument_profile.py
Q:\finance-analyzer\portfolio\avanza_resilient_page.py
Q:\finance-analyzer\portfolio\fin_snipe_manager.py
Q:\finance-analyzer\portfolio\indicators.py
Q:\finance-analyzer\portfolio\fin_snipe.py
Q:\finance-analyzer\portfolio\fin_fish.py
Q:\finance-analyzer\portfolio\fin_evolve.py
Q:\finance-analyzer\portfolio\file_utils.py
Q:\finance-analyzer\portfolio\feature_normalizer.py
Q:\finance-analyzer\portfolio\fear_greed.py
Q:\finance-analyzer\portfolio\exposure_coach.py
Q:\finance-analyzer\portfolio\exit_optimizer.py
Q:\finance-analyzer\portfolio\equity_curve.py
Q:\finance-analyzer\portfolio\avanza_order_lock.py
Q:\finance-analyzer\portfolio\avanza_orders.py
Q:\finance-analyzer\portfolio\avanza_control.py
Q:\finance-analyzer\portfolio\avanza_client.py
Q:\finance-analyzer\portfolio\avanza_account_check.py
Q:\finance-analyzer\portfolio\ministral_trader.py
Q:\finance-analyzer\portfolio\ministral_signal.py
Q:\finance-analyzer\portfolio\microstructure_state.py
Q:\finance-analyzer\portfolio\microstructure.py
Q:\finance-analyzer\portfolio\meta_learner.py
Q:\finance-analyzer\portfolio\metals_precompute.py
Q:\finance-analyzer\portfolio\metals_orderbook.py
Q:\finance-analyzer\portfolio\metals_ladder.py
Q:\finance-analyzer\portfolio\metals_cross_assets.py
Q:\finance-analyzer\portfolio\message_throttle.py
Q:\finance-analyzer\portfolio\message_store.py
Q:\finance-analyzer\portfolio\memory_consolidation.py
Q:\finance-analyzer\portfolio\market_timing.py
Q:\finance-analyzer\portfolio\market_health.py
Q:\finance-analyzer\portfolio\main.py
Q:\finance-analyzer\portfolio\macro_context.py
Q:\finance-analyzer\portfolio\loop_health.py
Q:\finance-analyzer\portfolio\ic_computation.py
Q:\finance-analyzer\portfolio\loop_contract.py
Q:\finance-analyzer\portfolio\http_retry.py
Q:\finance-analyzer\portfolio\log_rotation.py
Q:\finance-analyzer\portfolio\health.py
Q:\finance-analyzer\portfolio\logging_config.py
Q:\finance-analyzer\portfolio\grid_tiers.py
Q:\finance-analyzer\portfolio\local_llm_report.py
Q:\finance-analyzer\portfolio\grid_fisher_config.py
Q:\finance-analyzer\portfolio\llm_probability_log.py
Q:\finance-analyzer\portfolio\grid_fisher.py
Q:\finance-analyzer\portfolio\llm_prewarmer.py
Q:\finance-analyzer\portfolio\gpu_gate.py
Q:\finance-analyzer\portfolio\llm_outcome_backfill.py
Q:\finance-analyzer\portfolio\gold_precompute.py
Q:\finance-analyzer\portfolio\llm_calibration.py
Q:\finance-analyzer\portfolio\llm_batch.py
Q:\finance-analyzer\portfolio\llama_server.py
Q:\finance-analyzer\portfolio\linear_factor.py
Q:\finance-analyzer\portfolio\kelly_sizing.py
Q:\finance-analyzer\portfolio\kelly_metals.py
Q:\finance-analyzer\portfolio\journal_index.py
Q:\finance-analyzer\portfolio\journal.py
Q:\finance-analyzer\portfolio\funding_rate.py
Q:\finance-analyzer\portfolio\forecast_signal.py
Q:\finance-analyzer\portfolio\futures_data.py
Q:\finance-analyzer\portfolio\fx_rates.py
Q:\finance-analyzer\portfolio\sentiment.py
Q:\finance-analyzer\portfolio\seasonality_updater.py
Q:\finance-analyzer\portfolio\seasonality.py
Q:\finance-analyzer\portfolio\risk_management.py
Q:\finance-analyzer\portfolio\reporting.py
Q:\finance-analyzer\portfolio\regime_alerts.py
Q:\finance-analyzer\portfolio\reflection.py
Q:\finance-analyzer\portfolio\qwen3_trader.py
Q:\finance-analyzer\portfolio\qwen3_signal.py
Q:\finance-analyzer\portfolio\prophecy.py
Q:\finance-analyzer\portfolio\process_lock.py
Q:\finance-analyzer\portfolio\price_targets.py
Q:\finance-analyzer\portfolio\price_source.py
Q:\finance-analyzer\portfolio\portfolio_validator.py
Q:\finance-analyzer\portfolio\portfolio_mgr.py
Q:\finance-analyzer\portfolio\perception_gate.py
Q:\finance-analyzer\portfolio\outcome_tracker.py
Q:\finance-analyzer\portfolio\orb_predictor.py
Q:\finance-analyzer\portfolio\orb_postmortem.py
Q:\finance-analyzer\portfolio\orb_backtest.py
Q:\finance-analyzer\portfolio\onchain_data.py
Q:\finance-analyzer\portfolio\oil_precompute.py
Q:\finance-analyzer\portfolio\oil_grid_signal.py
Q:\finance-analyzer\portfolio\notification_text.py
Q:\finance-analyzer\portfolio\news_keywords.py
Q:\finance-analyzer\portfolio\multi_agent_layer2.py
Q:\finance-analyzer\portfolio\mstr_precompute.py
Q:\finance-analyzer\portfolio\weekly_digest.py
Q:\finance-analyzer\portfolio\warrant_portfolio.py
Q:\finance-analyzer\portfolio\vector_memory.py
Q:\finance-analyzer\portfolio\trigger.py
Q:\finance-analyzer\portfolio\train_signal_weights.py
Q:\finance-analyzer\portfolio\trade_validation.py
Q:\finance-analyzer\portfolio\trade_risk_classifier.py
Q:\finance-analyzer\portfolio\trade_guards.py
Q:\finance-analyzer\portfolio\tinylora_trainer.py
Q:\finance-analyzer\portfolio\ticker_accuracy.py
Q:\finance-analyzer\portfolio\tickers.py
Q:\finance-analyzer\portfolio\telegram_poller.py
Q:\finance-analyzer\portfolio\telegram_notifications.py
Q:\finance-analyzer\portfolio\subprocess_utils.py
Q:\finance-analyzer\portfolio\elongir\state.py
Q:\finance-analyzer\portfolio\elongir\signal.py
Q:\finance-analyzer\portfolio\elongir\runner.py
Q:\finance-analyzer\portfolio\elongir\risk.py
Q:\finance-analyzer\portfolio\elongir\indicators.py
Q:\finance-analyzer\portfolio\elongir\data_provider.py
Q:\finance-analyzer\portfolio\elongir\config.py
Q:\finance-analyzer\portfolio\elongir\bot.py
Q:\finance-analyzer\portfolio\monte_carlo_risk.py
Q:\finance-analyzer\portfolio\monte_carlo.py
Q:\finance-analyzer\portfolio\ml_trainer.py
Q:\finance-analyzer\portfolio\ml_signal.py
Q:\finance-analyzer\portfolio\mstr_loop\data_provider.py
Q:\finance-analyzer\portfolio\mstr_loop\config.py
Q:\finance-analyzer\portfolio\mstr_loop\execution.py
Q:\finance-analyzer\portfolio\golddigger\bot.py
Q:\finance-analyzer\portfolio\golddigger\augmented_signals.py
Q:\finance-analyzer\portfolio\mstr_loop\loop.py
Q:\finance-analyzer\portfolio\golddigger\config.py
Q:\finance-analyzer\portfolio\signal_registry.py
Q:\finance-analyzer\portfolio\signal_postmortem.py
Q:\finance-analyzer\portfolio\signal_history.py
Q:\finance-analyzer\portfolio\signal_engine.py
Q:\finance-analyzer\portfolio\signal_decay_alert.py
Q:\finance-analyzer\portfolio\signal_db.py
Q:\finance-analyzer\portfolio\strategies\orchestrator.py
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py
Q:\finance-analyzer\portfolio\strategies\elongir_strategy.py
Q:\finance-analyzer\portfolio\strategies\base.py
Q:\finance-analyzer\portfolio\stats.py
Q:\finance-analyzer\portfolio\social_sentiment.py
Q:\finance-analyzer\portfolio\silver_precompute.py
Q:\finance-analyzer\portfolio\signal_weight_optimizer.py
Q:\finance-analyzer\portfolio\signal_weights.py
Q:\finance-analyzer\portfolio\signal_utils.py
Q:\finance-analyzer\portfolio\signal_state_since.py
Q:\finance-analyzer\portfolio\api_utils.py
Q:\finance-analyzer\portfolio\analyze.py
Q:\finance-analyzer\portfolio\alpha_vantage.py
Q:\finance-analyzer\portfolio\alert_budget.py
Q:\finance-analyzer\portfolio\agent_invocation.py
Q:\finance-analyzer\portfolio\accuracy_stats.py
Q:\finance-analyzer\portfolio\accuracy_degradation.py
Q:\finance-analyzer\portfolio\avanza\client.py
Q:\finance-analyzer\portfolio\avanza\auth.py
Q:\finance-analyzer\portfolio\avanza\account.py
Q:\finance-analyzer\portfolio\autonomous.py
Q:\finance-analyzer\portfolio\avanza\scanner.py
Q:\finance-analyzer\portfolio\avanza\market_data.py
Q:\finance-analyzer\portfolio\avanza\search.py
Q:\finance-analyzer\portfolio\avanza\streaming.py
Q:\finance-analyzer\portfolio\shadow_registry.py
Q:\finance-analyzer\portfolio\session_calendar.py
Q:\finance-analyzer\portfolio\sentiment_shadow_backfill.py
Q:\finance-analyzer\portfolio\shared_state.py
Q:\finance-analyzer\portfolio\short_horizon.py
Q:\finance-analyzer\portfolio\mstr_loop\risk.py
Q:\finance-analyzer\portfolio\golddigger\data_provider.py
Q:\finance-analyzer\portfolio\golddigger\risk.py
Q:\finance-analyzer\portfolio\avanza\types.py
Q:\finance-analyzer\portfolio\golddigger\state.py
Q:\finance-analyzer\portfolio\golddigger\signal.py
Q:\finance-analyzer\portfolio\avanza\trading.py
Q:\finance-analyzer\portfolio\golddigger\runner.py
Q:\finance-analyzer\portfolio\avanza\tick_rules.py
Q:\finance-analyzer\portfolio\mstr_loop\state.py
Q:\finance-analyzer\portfolio\mstr_loop\session.py
Q:\finance-analyzer\portfolio\mstr_loop\telegram_report.py
Q:\finance-analyzer\portfolio\mstr_loop\strategies\premium_arb.py
Q:\finance-analyzer\portfolio\mstr_loop\strategies\overnight_gap.py
Q:\finance-analyzer\portfolio\mstr_loop\strategies\momentum_rider.py
Q:\finance-analyzer\portfolio\mstr_loop\strategies\mean_reversion.py
Q:\finance-analyzer\portfolio\mstr_loop\strategies\earnings_play.py
Q:\finance-analyzer\portfolio\mstr_loop\strategies\base.py
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py
Q:\finance-analyzer\portfolio\signals\crypto_macro.py
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py
Q:\finance-analyzer\portfolio\signals\credit_spread.py
Q:\finance-analyzer\portfolio\signals\cot_positioning.py
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py
Q:\finance-analyzer\portfolio\signals\candlestick.py
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py
Q:\finance-analyzer\portfolio\signals\futures_basis.py
Q:\finance-analyzer\portfolio\signals\forecast.py
Q:\finance-analyzer\portfolio\signals\fibonacci.py
Q:\finance-analyzer\portfolio\signals\econ_calendar.py
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py
Q:\finance-analyzer\portfolio\signals\futures_flow.py
Q:\finance-analyzer\portfolio\signals\hurst_regime.py
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py
Q:\finance-analyzer\portfolio\signals\macro_regime.py
Q:\finance-analyzer\portfolio\signals\network_momentum.py
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py
Q:\finance-analyzer\portfolio\signals\oscillators.py
Q:\finance-analyzer\portfolio\signals\realized_skewness.py
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py
Q:\finance-analyzer\portfolio\signals\momentum_factors.py
Q:\finance-analyzer\portfolio\signals\news_event.py
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py
Q:\finance-analyzer\portfolio\signals\momentum.py
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py
Q:\finance-analyzer\portfolio\signals\mean_reversion.py
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py
Q:\finance-analyzer\portfolio\signals\volume_flow.py
Q:\finance-analyzer\portfolio\signals\volatility.py
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py
Q:\finance-analyzer\portfolio\signals\trend.py
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py
Q:\finance-analyzer\portfolio\signals\structure.py
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py
Q:\finance-analyzer\portfolio\signals\smart_money.py
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 721ms:
Q:\finance-analyzer\portfolio\futures_data.py:1:"""Binance FAPI futures data — open interest, long/short ratios, funding history.
Q:\finance-analyzer\portfolio\futures_data.py:4:Uses existing infrastructure: fetch_json, _cached, _binance_limiter.
Q:\finance-analyzer\portfolio\futures_data.py:11:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\futures_data.py:12:from portfolio.shared_state import _binance_limiter, _cached
Q:\finance-analyzer\portfolio\futures_data.py:21:# Cache TTLs
Q:\finance-analyzer\portfolio\futures_data.py:22:_OI_TTL = 300        # 5 min
Q:\finance-analyzer\portfolio\futures_data.py:23:_LS_TTL = 300        # 5 min
Q:\finance-analyzer\portfolio\futures_data.py:24:_FUNDING_TTL = 900   # 15 min
Q:\finance-analyzer\portfolio\futures_data.py:28:    """Fetch JSON from Binance FAPI with rate limiting and retry."""
Q:\finance-analyzer\portfolio\futures_data.py:55:    return _cached(f"futures_oi_{ticker}", _OI_TTL, _fetch)
Q:\finance-analyzer\portfolio\futures_data.py:58:def get_open_interest_history(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:83:    return _cached(f"futures_oi_hist_{ticker}_{period}", _OI_TTL, _fetch)
Q:\finance-analyzer\portfolio\futures_data.py:86:def get_long_short_ratio(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:112:    return _cached(f"futures_ls_{ticker}_{period}", _LS_TTL, _fetch)
Q:\finance-analyzer\portfolio\futures_data.py:115:def get_top_trader_position_ratio(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:141:    return _cached(f"futures_top_pos_{ticker}_{period}", _LS_TTL, _fetch)
Q:\finance-analyzer\portfolio\futures_data.py:144:def get_top_trader_account_ratio(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:170:    return _cached(f"futures_top_acct_{ticker}_{period}", _LS_TTL, _fetch)
Q:\finance-analyzer\portfolio\futures_data.py:173:def get_funding_rate_history(ticker, limit=100):
Q:\finance-analyzer\portfolio\futures_data.py:174:    """Historical funding rates.
Q:\finance-analyzer\portfolio\futures_data.py:176:    Returns: list of {fundingRate, fundingTime, symbol} or None.
Q:\finance-analyzer\portfolio\futures_data.py:184:            f"{BINANCE_FAPI_BASE}/fundingRate",
Q:\finance-analyzer\portfolio\futures_data.py:191:                "fundingRate": float(d["fundingRate"]),
Q:\finance-analyzer\portfolio\futures_data.py:192:                "fundingTime": d["fundingTime"],
Q:\finance-analyzer\portfolio\futures_data.py:198:    return _cached(f"futures_funding_hist_{ticker}", _FUNDING_TTL, _fetch)
Q:\finance-analyzer\portfolio\futures_data.py:205:             top_position_ratio, top_account_ratio, funding_history.
Q:\finance-analyzer\portfolio\futures_data.py:216:        "funding_history": get_funding_rate_history(ticker),
Q:\finance-analyzer\portfolio\futures_data.py:234:                    print(f"  {k}: {json.dumps(v, indent=2)}")
Q:\finance-analyzer\portfolio\fear_greed.py:2:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\fear_greed.py:5:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\fear_greed.py:6:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\fear_greed.py:12:# Sustained fear/greed tracking — used by signal_engine to gate contrarian
Q:\finance-analyzer\portfolio\fear_greed.py:13:# signals during prolonged extreme sentiment regimes (e.g., 46-day fear streaks).
Q:\finance-analyzer\portfolio\fear_greed.py:17:_STREAK_FILE = _BASE_DIR / "data" / "fear_greed_streak.json"
Q:\finance-analyzer\portfolio\fear_greed.py:20:def get_sustained_fear_days() -> int:
Q:\finance-analyzer\portfolio\fear_greed.py:23:    Returns 0 if not in an extreme fear streak, or if tracking data is unavailable.
Q:\finance-analyzer\portfolio\fear_greed.py:27:        if data and data.get("streak_type") == "extreme_fear":
Q:\finance-analyzer\portfolio\fear_greed.py:30:        logger.debug("Could not read fear streak file", exc_info=True)
Q:\finance-analyzer\portfolio\fear_greed.py:34:def update_fear_streak(fg_value: int) -> dict:
Q:\finance-analyzer\portfolio\fear_greed.py:35:    """Update the sustained fear/greed streak tracker.
Q:\finance-analyzer\portfolio\fear_greed.py:37:    Called after each successful F&G fetch. Persists streak state to disk
Q:\finance-analyzer\portfolio\fear_greed.py:45:    now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\fear_greed.py:56:        if prev_type == "extreme_fear":
Q:\finance-analyzer\portfolio\fear_greed.py:60:            data = {"streak_type": "extreme_fear", "streak_days": 1,
Q:\finance-analyzer\portfolio\fear_greed.py:63:        if prev_type == "extreme_greed":
Q:\finance-analyzer\portfolio\fear_greed.py:67:            data = {"streak_type": "extreme_greed", "streak_days": 1,
Q:\finance-analyzer\portfolio\fear_greed.py:77:        atomic_write_json(_STREAK_FILE, data)
Q:\finance-analyzer\portfolio\fear_greed.py:79:        logger.debug("Could not write fear streak file", exc_info=True)
Q:\finance-analyzer\portfolio\fear_greed.py:95:def get_crypto_fear_greed() -> dict:
Q:\finance-analyzer\portfolio\fear_greed.py:97:                      label="crypto_fear_greed")
Q:\finance-analyzer\portfolio\fear_greed.py:103:    # crashed every cycle's fear-greed signal computation silently.
Q:\finance-analyzer\portfolio\fear_greed.py:107:        logger.debug("crypto_fear_greed: API returned no data (maintenance?)")
Q:\finance-analyzer\portfolio\fear_greed.py:111:        logger.debug("crypto_fear_greed: data[0] not a dict, got %r", type(data))
Q:\finance-analyzer\portfolio\fear_greed.py:117:            "timestamp": datetime.fromtimestamp(
Q:\finance-analyzer\portfolio\fear_greed.py:118:                int(data["timestamp"]), tz=UTC
Q:\finance-analyzer\portfolio\fear_greed.py:122:        logger.debug("crypto_fear_greed: malformed entry: %s", e)
Q:\finance-analyzer\portfolio\fear_greed.py:126:def get_stock_fear_greed() -> dict:
Q:\finance-analyzer\portfolio\fear_greed.py:130:    from portfolio.shared_state import yfinance_lock
Q:\finance-analyzer\portfolio\fear_greed.py:169:        "timestamp": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\fear_greed.py:174:def get_fear_greed(ticker=None) -> dict:
Q:\finance-analyzer\portfolio\fear_greed.py:176:        return get_crypto_fear_greed()
Q:\finance-analyzer\portfolio\fear_greed.py:177:    return get_stock_fear_greed()
Q:\finance-analyzer\portfolio\fear_greed.py:182:    result = get_crypto_fear_greed()
Q:\finance-analyzer\portfolio\fear_greed.py:186:    result = get_stock_fear_greed()
Q:\finance-analyzer\portfolio\bert_sentiment.py:1:"""In-process BERT sentiment inference (CryptoBERT, Trading-Hero-LLM, FinBERT).
Q:\finance-analyzer\portfolio\bert_sentiment.py:14:  3. Cold-load the 125M BERT weights from disk (~1-3 s)
Q:\finance-analyzer\portfolio\bert_sentiment.py:17:BERT calls per cycle (2 per ticker * 5 active tickers), we burned 30-100 s
Q:\finance-analyzer\portfolio\bert_sentiment.py:18:per cycle just in BERT sentiment. And none of the three scripts ever called
Q:\finance-analyzer\portfolio\bert_sentiment.py:24:Load each model lazily on first use, cache (tokenizer, model, device, lock)
Q:\finance-analyzer\portfolio\bert_sentiment.py:31:per model so CryptoBERT and FinBERT can run in parallel, but two threads
Q:\finance-analyzer\portfolio\bert_sentiment.py:39:import fails, if a model cache dir is missing, if CUDA OOMs on the load,
Q:\finance-analyzer\portfolio\bert_sentiment.py:46:its current model is loaded. Three BERT models = ~1.5 GB resident. Total:
Q:\finance-analyzer\portfolio\bert_sentiment.py:52:- Retiring the 3 /mnt/q/models/*_infer.py scripts: they stay as the fallback
Q:\finance-analyzer\portfolio\bert_sentiment.py:55:- GPU gate acquisition around the forward pass: BERT forward pass is ~20 ms
Q:\finance-analyzer\portfolio\bert_sentiment.py:80:    "CryptoBERT": {
Q:\finance-analyzer\portfolio\bert_sentiment.py:82:        "cache_dir_win": r"Q:\models\cryptobert",
Q:\finance-analyzer\portfolio\bert_sentiment.py:83:        "cache_dir_linux": "/home/deck/models/cryptobert",
Q:\finance-analyzer\portfolio\bert_sentiment.py:85:        # CryptoBERT's native labels: {0: Bearish, 1: Neutral, 2: Bullish}.
Q:\finance-analyzer\portfolio\bert_sentiment.py:92:        "cache_dir_win": r"Q:\models\trading-hero-llm",
Q:\finance-analyzer\portfolio\bert_sentiment.py:93:        "cache_dir_linux": "/home/deck/models/trading-hero-llm",
Q:\finance-analyzer\portfolio\bert_sentiment.py:99:    "FinBERT": {
Q:\finance-analyzer\portfolio\bert_sentiment.py:101:        "cache_dir_win": r"Q:\models\finbert",
Q:\finance-analyzer\portfolio\bert_sentiment.py:102:        "cache_dir_linux": "/home/deck/models/finbert",
Q:\finance-analyzer\portfolio\bert_sentiment.py:107:        # FinBERT uses a snapshot dir layout (models--ProsusAI--finbert/snapshots/<hash>)
Q:\finance-analyzer\portfolio\bert_sentiment.py:108:        # instead of the Hub cache layout, so we resolve the path differently.
Q:\finance-analyzer\portfolio\bert_sentiment.py:115:# Module-level cache: model_name -> (tokenizer, model, device, lock).
Q:\finance-analyzer\portfolio\bert_sentiment.py:121:def _resolve_cache_dir(config: dict) -> str:
Q:\finance-analyzer\portfolio\bert_sentiment.py:122:    return config["cache_dir_win"] if platform.system() == "Windows" else config["cache_dir_linux"]
Q:\finance-analyzer\portfolio\bert_sentiment.py:125:def _resolve_finbert_snapshot(cache_dir: str, subdir: str) -> str | None:
Q:\finance-analyzer\portfolio\bert_sentiment.py:126:    """FinBERT is saved under cache_dir/models--ProsusAI--finbert/snapshots/<hash>/.
Q:\finance-analyzer\portfolio\bert_sentiment.py:130:    snapshots = glob.glob(os.path.join(cache_dir, subdir, "snapshots", "*"))
Q:\finance-analyzer\portfolio\bert_sentiment.py:137:    BERT models keep LayerNorm running mean/var (and a few other tensors)
Q:\finance-analyzer\portfolio\bert_sentiment.py:152:    or None if accelerate isn't importable. Used in the meta-tensor retry
Q:\finance-analyzer\portfolio\bert_sentiment.py:163:def _model_load_kwargs(name: str, config: dict, cache_dir: str) -> tuple[str, dict]:
Q:\finance-analyzer\portfolio\bert_sentiment.py:166:    Returns (load_path, kwargs). FinBERT uses a snapshot path (no
Q:\finance-analyzer\portfolio\bert_sentiment.py:167:    cache_dir/local_files_only); others use cache_dir + local_files_only.
Q:\finance-analyzer\portfolio\bert_sentiment.py:168:    Falls back to the Hub name for FinBERT if no local snapshot exists.
Q:\finance-analyzer\portfolio\bert_sentiment.py:171:    be reused by the meta-tensor recovery retry path without duplicating
Q:\finance-analyzer\portfolio\bert_sentiment.py:172:    the FinBERT-vs-others branching.
Q:\finance-analyzer\portfolio\bert_sentiment.py:174:    if name == "FinBERT":
Q:\finance-analyzer\portfolio\bert_sentiment.py:175:        snapshot = _resolve_finbert_snapshot(cache_dir, config["snapshot_subdir"])
Q:\finance-analyzer\portfolio\bert_sentiment.py:180:        "cache_dir": cache_dir,
Q:\finance-analyzer\portfolio\bert_sentiment.py:186:    """Load a BERT model + tokenizer. Called under _init_lock.
Q:\finance-analyzer\portfolio\bert_sentiment.py:188:    Returns (tokenizer, model, device, per_model_lock) where device is
Q:\finance-analyzer\portfolio\bert_sentiment.py:197:    cache_dir = _resolve_cache_dir(config)
Q:\finance-analyzer\portfolio\bert_sentiment.py:199:    load_path, load_kwargs = _model_load_kwargs(name, config, cache_dir)
Q:\finance-analyzer\portfolio\bert_sentiment.py:201:    if name == "FinBERT" and load_path == hf_name:
Q:\finance-analyzer\portfolio\bert_sentiment.py:202:        logger.info("Loading BERT model %s via hub name %s (no local snapshot found)", name, hf_name)
Q:\finance-analyzer\portfolio\bert_sentiment.py:203:    elif name == "FinBERT":
Q:\finance-analyzer\portfolio\bert_sentiment.py:204:        logger.info("Loading BERT model %s from snapshot %s", name, load_path)
Q:\finance-analyzer\portfolio\bert_sentiment.py:206:        logger.info("Loading BERT model %s from %s", name, cache_dir)
Q:\finance-analyzer\portfolio\bert_sentiment.py:208:    tokenizer = AutoTokenizer.from_pretrained(load_path, **{
Q:\finance-analyzer\portfolio\bert_sentiment.py:210:        if k in ("cache_dir", "local_files_only")
Q:\finance-analyzer\portfolio\bert_sentiment.py:216:    # Race between Chronos's CUDA load and concurrent BERT loads (commit
Q:\finance-analyzer\portfolio\bert_sentiment.py:219:    # sentiment phase's BERT loads via main.py's ThreadPoolExecutor) can
Q:\finance-analyzer\portfolio\bert_sentiment.py:220:    # leave some FinBERT weights on the `meta` device when accelerate's
Q:\finance-analyzer\portfolio\bert_sentiment.py:224:    # the per-text fallback writes a zero-confidence neutral placeholder
Q:\finance-analyzer\portfolio\bert_sentiment.py:228:    # Detection at load time + one retry with eager-init kwargs flips
Q:\finance-analyzer\portfolio\bert_sentiment.py:231:    # caller's _get_model() doesn't cache, so subsequent predict calls
Q:\finance-analyzer\portfolio\bert_sentiment.py:236:    # which is rare and load-time-only — never during steady-state
Q:\finance-analyzer\portfolio\bert_sentiment.py:240:            "BERT %s loaded with meta tensors (likely accelerate race with "
Q:\finance-analyzer\portfolio\bert_sentiment.py:241:            "concurrent CUDA load); retrying with eager init",
Q:\finance-analyzer\portfolio\bert_sentiment.py:254:                f"BERT {name} still has meta tensors after retry "
Q:\finance-analyzer\portfolio\bert_sentiment.py:265:    # 2026-04-09 (hotfix): BERT models now stay on CPU by default.
Q:\finance-analyzer\portfolio\bert_sentiment.py:267:    # Initial deployment tried to move BERT models to CUDA for ~5-20x per-call
Q:\finance-analyzer\portfolio\bert_sentiment.py:270:    #   BERT (3 models) ~1.5 GB + Chronos-2 ~3.5 GB + llama-server 5 GB
Q:\finance-analyzer\portfolio\bert_sentiment.py:272:    # With BERT + Chronos resident, llama-server's 5 GB finance-llama-8b load
Q:\finance-analyzer\portfolio\bert_sentiment.py:273:    # was timing out / retrying for 200+ s per swap, making cycles LONGER
Q:\finance-analyzer\portfolio\bert_sentiment.py:279:    # forward pass for a 125M BERT is ~100-300 ms per headline, vs ~5-20 ms
Q:\finance-analyzer\portfolio\bert_sentiment.py:283:    # Set BERT_SENTIMENT_USE_GPU=1 in the environment to opt back in to GPU
Q:\finance-analyzer\portfolio\bert_sentiment.py:286:    use_gpu = os.environ.get("BERT_SENTIMENT_USE_GPU", "").strip() in ("1", "true", "TRUE", "yes")
Q:\finance-analyzer\portfolio\bert_sentiment.py:292:            logger.info("BERT model %s moved to CUDA (BERT_SENTIMENT_USE_GPU=1)", name)
Q:\finance-analyzer\portfolio\bert_sentiment.py:294:            logger.warning("BERT model %s failed to move to CUDA, staying on CPU: %s", name, e)
Q:\finance-analyzer\portfolio\bert_sentiment.py:296:        logger.info("BERT model %s staying on CPU (default, avoids VRAM contention with llama-server)", name)
Q:\finance-analyzer\portfolio\bert_sentiment.py:298:    return tokenizer, model, device, threading.Lock()
Q:\finance-analyzer\portfolio\bert_sentiment.py:303:    cached tuple on subsequent calls.
Q:\finance-analyzer\portfolio\bert_sentiment.py:306:        raise KeyError(f"Unknown BERT model: {name!r}. Known: {list(_MODEL_CONFIGS)}")
Q:\finance-analyzer\portfolio\bert_sentiment.py:323:    """Run BERT sentiment inference on a list of texts.
Q:\finance-analyzer\portfolio\bert_sentiment.py:332:    / finbert_infer.py) all pass the full text list to the tokenizer in
Q:\finance-analyzer\portfolio\bert_sentiment.py:335:    the BERT kernel launch overhead is amortized across the batch.
Q:\finance-analyzer\portfolio\bert_sentiment.py:337:    If the batched path fails (e.g. OOM on a huge batch, or tokenizer
Q:\finance-analyzer\portfolio\bert_sentiment.py:350:    tokenizer, model, device, lock = _get_model(model_name)
Q:\finance-analyzer\portfolio\bert_sentiment.py:361:                texts, tokenizer, model, device, max_length, label_map, torch,
Q:\finance-analyzer\portfolio\bert_sentiment.py:365:                "BERT %s batched predict failed, falling back to per-text loop: %s",
Q:\finance-analyzer\portfolio\bert_sentiment.py:369:                texts, tokenizer, model, device, max_length, label_map, torch, model_name,
Q:\finance-analyzer\portfolio\bert_sentiment.py:373:def _predict_batched(texts, tokenizer, model, device, max_length, label_map, torch):
Q:\finance-analyzer\portfolio\bert_sentiment.py:374:    """Single tokenizer + forward pass over the whole batch."""
Q:\finance-analyzer\portfolio\bert_sentiment.py:375:    inputs = tokenizer(
Q:\finance-analyzer\portfolio\bert_sentiment.py:408:def _predict_per_text(texts, tokenizer, model, device, max_length, label_map, torch, model_name):
Q:\finance-analyzer\portfolio\bert_sentiment.py:411:    tokenizer error on one odd input).
Q:\finance-analyzer\portfolio\bert_sentiment.py:417:            inputs = tokenizer(
Q:\finance-analyzer\portfolio\bert_sentiment.py:444:                "BERT %s per-text predict failed for %r: %s",
Q:\finance-analyzer\portfolio\bert_sentiment.py:457:    """Return the list of supported BERT model names (for tests + debugging)."""
Q:\finance-analyzer\portfolio\bert_sentiment.py:467:    """Drop the model cache. Tests only - don't call this in production code.
Q:\finance-analyzer\portfolio\social_sentiment.py:9:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\social_sentiment.py:13:# (subreddit, dedicated) — dedicated: keep all posts; general: filter by keywords
Q:\finance-analyzer\portfolio\social_sentiment.py:29:def _fetch_subreddit(sub, keywords, dedicated, per_sub):
Q:\finance-analyzer\portfolio\social_sentiment.py:40:        if not dedicated and not any(kw in title.lower() for kw in keywords):
Q:\finance-analyzer\portfolio\social_sentiment.py:48:                    datetime.fromtimestamp(created, tz=UTC).isoformat()
Q:\finance-analyzer\portfolio\social_sentiment.py:50:                    else datetime.now(UTC).isoformat()
Q:\finance-analyzer\portfolio\social_sentiment.py:59:def _search_subreddit(sub, keywords, limit=10):
Q:\finance-analyzer\portfolio\social_sentiment.py:60:    query = urllib.parse.quote(" OR ".join(keywords))
Q:\finance-analyzer\portfolio\social_sentiment.py:80:                    datetime.fromtimestamp(created, tz=UTC).isoformat()
Q:\finance-analyzer\portfolio\social_sentiment.py:82:                    else datetime.now(UTC).isoformat()
Q:\finance-analyzer\portfolio\social_sentiment.py:94:    keywords = TICKER_KEYWORDS.get(short, [short.lower()])
Q:\finance-analyzer\portfolio\social_sentiment.py:104:            fetched = _fetch_subreddit(sub, keywords, dedicated, per_sub)
Q:\finance-analyzer\portfolio\social_sentiment.py:112:    # Fallback: if keyword filtering yielded nothing, try search
Q:\finance-analyzer\portfolio\social_sentiment.py:116:                fetched = _search_subreddit(sub, keywords, limit=per_sub)
Q:\finance-analyzer\portfolio\alpha_vantage.py:1:"""Alpha Vantage fundamentals — daily-refresh stock fundamentals cache.
Q:\finance-analyzer\portfolio\alpha_vantage.py:3:Fetches OVERVIEW endpoint data for stock tickers and caches persistently.
Q:\finance-analyzer\portfolio\alpha_vantage.py:11:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\alpha_vantage.py:16:from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\alpha_vantage.py:17:from portfolio.shared_state import _alpha_vantage_limiter
Q:\finance-analyzer\portfolio\alpha_vantage.py:24:CACHE_FILE = DATA_DIR / "fundamentals_cache.json"
Q:\finance-analyzer\portfolio\alpha_vantage.py:28:# Module-level state
Q:\finance-analyzer\portfolio\alpha_vantage.py:29:_cache = {}  # ticker -> normalized fundamentals dict
Q:\finance-analyzer\portfolio\alpha_vantage.py:30:_cache_lock = threading.Lock()
Q:\finance-analyzer\portfolio\alpha_vantage.py:36:def load_persistent_cache():
Q:\finance-analyzer\portfolio\alpha_vantage.py:37:    """Load fundamentals cache from disk on startup."""
Q:\finance-analyzer\portfolio\alpha_vantage.py:38:    global _cache
Q:\finance-analyzer\portfolio\alpha_vantage.py:41:        logger.info("No fundamentals cache found at %s", CACHE_FILE)
Q:\finance-analyzer\portfolio\alpha_vantage.py:44:        with _cache_lock:
Q:\finance-analyzer\portfolio\alpha_vantage.py:45:            _cache = data
Q:\finance-analyzer\portfolio\alpha_vantage.py:46:        logger.info("Loaded fundamentals for %d tickers from cache", len(data))
Q:\finance-analyzer\portfolio\alpha_vantage.py:49:def _save_persistent_cache():
Q:\finance-analyzer\portfolio\alpha_vantage.py:50:    """Write current cache to disk atomically."""
Q:\finance-analyzer\portfolio\alpha_vantage.py:52:        from portfolio.file_utils import atomic_write_json
Q:\finance-analyzer\portfolio\alpha_vantage.py:53:        with _cache_lock:
Q:\finance-analyzer\portfolio\alpha_vantage.py:54:            snapshot = dict(_cache)
Q:\finance-analyzer\portfolio\alpha_vantage.py:55:        atomic_write_json(CACHE_FILE, snapshot)
Q:\finance-analyzer\portfolio\alpha_vantage.py:57:        logger.warning("Failed to save fundamentals cache: %s", e)
Q:\finance-analyzer\portfolio\alpha_vantage.py:61:    """Return cached fundamentals for a ticker, or None if not available."""
Q:\finance-analyzer\portfolio\alpha_vantage.py:62:    with _cache_lock:
Q:\finance-analyzer\portfolio\alpha_vantage.py:63:        return _cache.get(ticker)
Q:\finance-analyzer\portfolio\alpha_vantage.py:67:    """Return all cached fundamentals (for reporting)."""
Q:\finance-analyzer\portfolio\alpha_vantage.py:68:    with _cache_lock:
Q:\finance-analyzer\portfolio\alpha_vantage.py:69:        return dict(_cache)
Q:\finance-analyzer\portfolio\alpha_vantage.py:72:def _normalize_overview(raw):
Q:\finance-analyzer\portfolio\alpha_vantage.py:73:    """Convert Alpha Vantage OVERVIEW response to clean format.
Q:\finance-analyzer\portfolio\alpha_vantage.py:107:        "earnings_growth_yoy": _float(raw.get("QuarterlyEarningsGrowthYOY")),
Q:\finance-analyzer\portfolio\alpha_vantage.py:124:        "_fetched_at": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\alpha_vantage.py:130:    """Fetch OVERVIEW data for a single ticker from Alpha Vantage."""
Q:\finance-analyzer\portfolio\alpha_vantage.py:136:    resp = fetch_with_retry(
Q:\finance-analyzer\portfolio\alpha_vantage.py:151:        logger.warning("Alpha Vantage rate limit hit: %s", data["Note"][:100])
Q:\finance-analyzer\portfolio\alpha_vantage.py:160:    BUG-108: Protected by _cache_lock for thread safety.
Q:\finance-analyzer\portfolio\alpha_vantage.py:163:    today = datetime.now(UTC).strftime("%Y-%m-%d")
Q:\finance-analyzer\portfolio\alpha_vantage.py:164:    with _cache_lock:
Q:\finance-analyzer\portfolio\alpha_vantage.py:172:    """Check if cached data for ticker is older than max_stale_days."""
Q:\finance-analyzer\portfolio\alpha_vantage.py:173:    with _cache_lock:
Q:\finance-analyzer\portfolio\alpha_vantage.py:174:        entry = _cache.get(ticker)
Q:\finance-analyzer\portfolio\alpha_vantage.py:181:        fetched_time = datetime.fromisoformat(fetched_at)
Q:\finance-analyzer\portfolio\alpha_vantage.py:182:        age_seconds = (datetime.now(UTC) - fetched_time).total_seconds()
Q:\finance-analyzer\portfolio\alpha_vantage.py:188:def _cache_age_hours(ticker):
Q:\finance-analyzer\portfolio\alpha_vantage.py:189:    """Return age of cached data in hours, or None if not cached."""
Q:\finance-analyzer\portfolio\alpha_vantage.py:190:    with _cache_lock:
Q:\finance-analyzer\portfolio\alpha_vantage.py:191:        entry = _cache.get(ticker)
Q:\finance-analyzer\portfolio\alpha_vantage.py:198:        fetched_time = datetime.fromisoformat(fetched_at)
Q:\finance-analyzer\portfolio\alpha_vantage.py:199:        return (datetime.now(UTC) - fetched_time).total_seconds() / 3600
Q:\finance-analyzer\portfolio\alpha_vantage.py:214:        logger.info("Alpha Vantage disabled in config")
Q:\finance-analyzer\portfolio\alpha_vantage.py:219:        logger.warning("Alpha Vantage API key not configured")
Q:\finance-analyzer\portfolio\alpha_vantage.py:224:    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)
Q:\finance-analyzer\portfolio\alpha_vantage.py:228:        logger.info("Alpha Vantage circuit breaker active (%s)", _cb.state.value)
Q:\finance-analyzer\portfolio\alpha_vantage.py:233:        logger.info("Alpha Vantage daily budget exhausted (%d/%d)", budget_used, daily_budget)
Q:\finance-analyzer\portfolio\alpha_vantage.py:236:    # Build refresh list: stock tickers not recently cached, not skipped
Q:\finance-analyzer\portfolio\alpha_vantage.py:241:        age = _cache_age_hours(ticker)
Q:\finance-analyzer\portfolio\alpha_vantage.py:242:        if age is None or age > cache_ttl_hours:
Q:\finance-analyzer\portfolio\alpha_vantage.py:271:            normalized = _normalize_overview(raw)
Q:\finance-analyzer\portfolio\alpha_vantage.py:272:            if normalized is None:
Q:\finance-analyzer\portfolio\alpha_vantage.py:273:                logger.warning("Alpha Vantage: empty/error response for %s", ticker)
Q:\finance-analyzer\portfolio\alpha_vantage.py:279:            with _cache_lock:
Q:\finance-analyzer\portfolio\alpha_vantage.py:280:                _cache[ticker] = normalized
Q:\finance-analyzer\portfolio\alpha_vantage.py:286:                        normalized.get("pe_ratio") or 0,
Q:\finance-analyzer\portfolio\alpha_vantage.py:287:                        normalized.get("sector", "?"))
Q:\finance-analyzer\portfolio\alpha_vantage.py:290:            logger.warning("Alpha Vantage fetch failed for %s: %s", ticker, e)
Q:\finance-analyzer\portfolio\alpha_vantage.py:296:        _save_persistent_cache()
Q:\finance-analyzer\portfolio\alpha_vantage.py:305:    Only runs when cache is >24h old. Designed to be called from the main loop.
Q:\finance-analyzer\portfolio\alpha_vantage.py:311:    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)
Q:\finance-analyzer\portfolio\alpha_vantage.py:317:        age = _cache_age_hours(ticker)
Q:\finance-analyzer\portfolio\alpha_vantage.py:318:        if age is None or age > cache_ttl_hours:
Q:\finance-analyzer\portfolio\seasonality_updater.py:3:Fetches 1h klines from Binance FAPI and computes hour-of-day
Q:\finance-analyzer\portfolio\seasonality_updater.py:20:        tickers: List of tickers to update. Defaults to XAG-USD, XAU-USD.
Q:\finance-analyzer\portfolio\seasonality_updater.py:26:        tickers = ["XAG-USD", "XAU-USD"]
Q:\finance-analyzer\portfolio\seasonality_updater.py:52:    """Fetch 1h klines from Binance FAPI for a metals ticker."""
Q:\finance-analyzer\portfolio\seasonality_updater.py:56:    from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\seasonality_updater.py:57:    from portfolio.shared_state import _binance_limiter
Q:\finance-analyzer\portfolio\seasonality_updater.py:76:            "time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
Q:\finance-analyzer\portfolio\crypto_macro_data.py:9:All fetches use http_retry for resilience.
Q:\finance-analyzer\portfolio\crypto_macro_data.py:21:from portfolio.file_utils import atomic_append_jsonl
Q:\finance-analyzer\portfolio\crypto_macro_data.py:22:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\crypto_macro_data.py:23:from portfolio.shared_state import _cached
Q:\finance-analyzer\portfolio\crypto_macro_data.py:30:# Cache TTLs
Q:\finance-analyzer\portfolio\crypto_macro_data.py:31:OPTIONS_TTL = 900       # 15 min — options data changes slowly
Q:\finance-analyzer\portfolio\crypto_macro_data.py:32:RATIO_TTL = 3600        # 1h — gold/btc ratio
Q:\finance-analyzer\portfolio\crypto_macro_data.py:33:NETFLOW_HIST_TTL = 3600 # 1h — netflow history
Q:\finance-analyzer\portfolio\crypto_macro_data.py:65:    import datetime
Q:\finance-analyzer\portfolio\crypto_macro_data.py:104:            return datetime.datetime.strptime(s, "%d%b%y").date()
Q:\finance-analyzer\portfolio\crypto_macro_data.py:108:    now = datetime.date.today()
Q:\finance-analyzer\portfolio\crypto_macro_data.py:194:    return _cached(f"deribit_options_{currency}", OPTIONS_TTL,
Q:\finance-analyzer\portfolio\crypto_macro_data.py:308:        atomic_append_jsonl(RATIO_HISTORY_FILE, entry)
Q:\finance-analyzer\portfolio\crypto_macro_data.py:424:        atomic_append_jsonl(NETFLOW_HISTORY_FILE, entry)
Q:\finance-analyzer\portfolio\crypto_macro_data.py:450:    ratio = _cached("gold_btc_ratio", RATIO_TTL, compute_gold_btc_ratio)
Q:\finance-analyzer\portfolio\crypto_macro_data.py:456:        netflow = _cached("exchange_netflow_trend", NETFLOW_HIST_TTL,
Q:\finance-analyzer\portfolio\fomc_dates.py:1:"""Shared FOMC meeting date constants.
Q:\finance-analyzer\portfolio\fomc_dates.py:3:Single source of truth for all modules that need FOMC dates.
Q:\finance-analyzer\portfolio\fomc_dates.py:10:from datetime import date
Q:\finance-analyzer\portfolio\fomc_dates.py:12:# 2026 FOMC meeting dates (two-day meetings)
Q:\finance-analyzer\portfolio\fomc_dates.py:13:FOMC_DATES_2026 = [
Q:\finance-analyzer\portfolio\fomc_dates.py:24:# 2027 FOMC meeting dates (two-day meetings)
Q:\finance-analyzer\portfolio\fomc_dates.py:25:FOMC_DATES_2027 = [
Q:\finance-analyzer\portfolio\fomc_dates.py:37:FOMC_DATES_ISO = [d.isoformat() for d in FOMC_DATES_2026 + FOMC_DATES_2027]
Q:\finance-analyzer\portfolio\fomc_dates.py:40:FOMC_ANNOUNCEMENT_DATES = [
Q:\finance-analyzer\portfolio\data_refresh.py:7:from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\data_refresh.py:30:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\data_refresh.py:67:    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
Q:\finance-analyzer\portfolio\seasonality.py:17:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\seasonality.py:95:def normalize_volatility(raw_vol: float, hour: int, profile: dict) -> float:
Q:\finance-analyzer\portfolio\seasonality.py:121:    atomic_write_json(_STATE_FILE, profiles)
Q:\finance-analyzer\portfolio\data_collector.py:1:"""Data collection — Binance, Alpaca, yfinance kline fetchers + multi-timeframe collector."""
Q:\finance-analyzer\portfolio\data_collector.py:6:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\data_collector.py:10:import portfolio.shared_state as _ss
Q:\finance-analyzer\portfolio\data_collector.py:13:from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\data_collector.py:27:    "15m": ("15Min", 5),
Q:\finance-analyzer\portfolio\data_collector.py:36:    "15m": ("15m", "5d"),       # yfinance max for intraday <=60d
Q:\finance-analyzer\portfolio\data_collector.py:43:# Multi-timeframe analysis — (label, binance_interval, num_candles, cache_ttl_seconds)
Q:\finance-analyzer\portfolio\data_collector.py:45:    ("Now", "15m", 100, 0),  # ~25h data, refresh every cycle
Q:\finance-analyzer\portfolio\data_collector.py:46:    ("12h", "1h", 100, 300),  # ~4d data, cache 5min
Q:\finance-analyzer\portfolio\data_collector.py:47:    ("2d", "4h", 100, 900),  # ~17d data, cache 15min
Q:\finance-analyzer\portfolio\data_collector.py:48:    ("7d", "1d", 100, 3600),  # ~100d data, cache 1hr
Q:\finance-analyzer\portfolio\data_collector.py:49:    ("1mo", "3d", 100, 14400),  # ~300d data, cache 4hr
Q:\finance-analyzer\portfolio\data_collector.py:50:    ("3mo", "1w", 100, 43200),  # ~2yr data, cache 12hr
Q:\finance-analyzer\portfolio\data_collector.py:51:    ("6mo", "1M", 48, 86400),  # ~4yr data, cache 24hr
Q:\finance-analyzer\portfolio\data_collector.py:55:    ("Now", "15m", 100, 0),
Q:\finance-analyzer\portfolio\data_collector.py:65:# --- Binance API ---
Q:\finance-analyzer\portfolio\data_collector.py:74:def _binance_fetch(base_url, cb, label, symbol, interval="5m", limit=100):
Q:\finance-analyzer\portfolio\data_collector.py:75:    """Shared Binance kline fetcher for spot and FAPI endpoints."""
Q:\finance-analyzer\portfolio\data_collector.py:77:        logger.warning("Binance %s circuit OPEN — skipping %s", label, symbol)
Q:\finance-analyzer\portfolio\data_collector.py:78:        raise ConnectionError(f"Binance {label} circuit open for {symbol}")
Q:\finance-analyzer\portfolio\data_collector.py:80:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\data_collector.py:86:            raise ConnectionError(f"Binance {label} klines request failed for {symbol}")
Q:\finance-analyzer\portfolio\data_collector.py:91:                f"Binance {label} returned empty data for {symbol} {interval}"
Q:\finance-analyzer\portfolio\data_collector.py:96:        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
Q:\finance-analyzer\portfolio\data_collector.py:104:def binance_klines(symbol, interval="5m", limit=100):
Q:\finance-analyzer\portfolio\data_collector.py:108:def binance_fapi_klines(symbol, interval="5m", limit=100):
Q:\finance-analyzer\portfolio\data_collector.py:109:    """Fetch klines from Binance Futures API (for metals like XAUUSDT, XAGUSDT)."""
Q:\finance-analyzer\portfolio\data_collector.py:124:        end = datetime.now(UTC)
Q:\finance-analyzer\portfolio\data_collector.py:126:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\data_collector.py:157:        df["time"] = pd.to_datetime(df["time"])
Q:\finance-analyzer\portfolio\data_collector.py:262:        if _ss._current_market_state in ("closed", "weekend", "holiday"):
Q:\finance-analyzer\portfolio\data_collector.py:263:            logger.debug("%s: using yfinance (market %s)", ticker, _ss._current_market_state)
Q:\finance-analyzer\portfolio\data_collector.py:275:# H11/DC-R3-4: use the module-level lock from shared_state so all modules
Q:\finance-analyzer\portfolio\data_collector.py:276:# (fear_greed, golddigger/data_provider, data_collector) share one lock.
Q:\finance-analyzer\portfolio\data_collector.py:277:from portfolio.shared_state import yfinance_lock as _yfinance_lock
Q:\finance-analyzer\portfolio\data_collector.py:282:    cache_key = f"tf_{source_key}_{label}"
Q:\finance-analyzer\portfolio\data_collector.py:284:        with _ss._cache_lock:
Q:\finance-analyzer\portfolio\data_collector.py:285:            cached = _ss._tool_cache.get(cache_key)
Q:\finance-analyzer\portfolio\data_collector.py:286:            if cached and time.time() - cached["time"] < ttl:
Q:\finance-analyzer\portfolio\data_collector.py:287:                return (label, cached["data"])
Q:\finance-analyzer\portfolio\data_collector.py:290:        if "alpaca" in source and _ss._current_market_state in ("closed", "weekend", "holiday"):
Q:\finance-analyzer\portfolio\data_collector.py:308:            with _ss._cache_lock:
Q:\finance-analyzer\portfolio\data_collector.py:309:                _ss._tool_cache[cache_key] = {"data": entry, "time": time.time()}
Q:\finance-analyzer\portfolio\earnings_calendar.py:1:"""Earnings calendar — proximity gate to prevent trading near earnings.
Q:\finance-analyzer\portfolio\earnings_calendar.py:3:Fetches upcoming earnings dates from yfinance for US stock tickers.
Q:\finance-analyzer\portfolio\earnings_calendar.py:4:When a ticker has earnings within GATE_DAYS, BUY signals are gated to HOLD
Q:\finance-analyzer\portfolio\earnings_calendar.py:7:Cache: per-ticker, 24h TTL (earnings dates don't change intraday).
Q:\finance-analyzer\portfolio\earnings_calendar.py:14:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\earnings_calendar.py:18:logger = logging.getLogger("portfolio.earnings_calendar")
Q:\finance-analyzer\portfolio\earnings_calendar.py:20:# Gate window: force HOLD if earnings within this many calendar days
Q:\finance-analyzer\portfolio\earnings_calendar.py:23:# Cache TTL: 24 hours (earnings dates are stable)
Q:\finance-analyzer\portfolio\earnings_calendar.py:24:EARNINGS_CACHE_TTL = 86400
Q:\finance-analyzer\portfolio\earnings_calendar.py:26:# Per-ticker cache: {ticker: {"data": {...}, "time": epoch}}
Q:\finance-analyzer\portfolio\earnings_calendar.py:27:_earnings_cache: dict[str, dict] = {}
Q:\finance-analyzer\portfolio\earnings_calendar.py:28:_earnings_lock = threading.Lock()
Q:\finance-analyzer\portfolio\earnings_calendar.py:31:def _fetch_earnings_alpha_vantage(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\earnings_calendar.py:32:    """Fetch next earnings date from Alpha Vantage EARNINGS endpoint.
Q:\finance-analyzer\portfolio\earnings_calendar.py:38:        from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\earnings_calendar.py:39:        from portfolio.shared_state import _alpha_vantage_limiter
Q:\finance-analyzer\portfolio\earnings_calendar.py:49:        # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
Q:\finance-analyzer\portfolio\earnings_calendar.py:51:        # Known limitation — earnings fetches consume 1 AV call each but are not
Q:\finance-analyzer\portfolio\earnings_calendar.py:53:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\earnings_calendar.py:71:        today = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\earnings_calendar.py:72:        # Find the next upcoming earnings (reportedDate in the future or very recent)
Q:\finance-analyzer\portfolio\earnings_calendar.py:78:                from datetime import date as _date
Q:\finance-analyzer\portfolio\earnings_calendar.py:83:                        "earnings_date": ed.isoformat(),
Q:\finance-analyzer\portfolio\earnings_calendar.py:93:        logger.debug("Alpha Vantage earnings fetch failed for %s", ticker, exc_info=True)
Q:\finance-analyzer\portfolio\earnings_calendar.py:97:def _fetch_earnings_yfinance(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\earnings_calendar.py:98:    """Fallback: fetch next earnings date from yfinance."""
Q:\finance-analyzer\portfolio\earnings_calendar.py:107:                    earnings_date = cal.get("Earnings Date")
Q:\finance-analyzer\portfolio\earnings_calendar.py:108:                    if isinstance(earnings_date, list) and earnings_date:
Q:\finance-analyzer\portfolio\earnings_calendar.py:109:                        earnings_date = earnings_date[0]
Q:\finance-analyzer\portfolio\earnings_calendar.py:112:                        earnings_date = cal.loc["Earnings Date"].iloc[0]
Q:\finance-analyzer\portfolio\earnings_calendar.py:114:                        earnings_date = None
Q:\finance-analyzer\portfolio\earnings_calendar.py:116:                if earnings_date is not None:
Q:\finance-analyzer\portfolio\earnings_calendar.py:117:                    if hasattr(earnings_date, "date"):
Q:\finance-analyzer\portfolio\earnings_calendar.py:118:                        ed = earnings_date.date()
Q:\finance-analyzer\portfolio\earnings_calendar.py:119:                    elif isinstance(earnings_date, str):
Q:\finance-analyzer\portfolio\earnings_calendar.py:120:                        ed = datetime.fromisoformat(earnings_date).date()
Q:\finance-analyzer\portfolio\earnings_calendar.py:125:                        today = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\earnings_calendar.py:130:                            "earnings_date": ed.isoformat(),
Q:\finance-analyzer\portfolio\earnings_calendar.py:137:        logger.debug("yfinance earnings fetch failed for %s", ticker, exc_info=True)
Q:\finance-analyzer\portfolio\earnings_calendar.py:141:def _fetch_earnings_date(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\earnings_calendar.py:142:    """Fetch next earnings date — Alpha Vantage primary, yfinance fallback.
Q:\finance-analyzer\portfolio\earnings_calendar.py:144:    Returns dict with earnings_date, days_until, timing, or None.
Q:\finance-analyzer\portfolio\earnings_calendar.py:146:    # Primary: Alpha Vantage (already have API key + rate limiter)
Q:\finance-analyzer\portfolio\earnings_calendar.py:147:    result = _fetch_earnings_alpha_vantage(ticker)
Q:\finance-analyzer\portfolio\earnings_calendar.py:152:    return _fetch_earnings_yfinance(ticker)
Q:\finance-analyzer\portfolio\earnings_calendar.py:155:def get_earnings_proximity(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\earnings_calendar.py:156:    """Get cached earnings proximity for a ticker.
Q:\finance-analyzer\portfolio\earnings_calendar.py:161:        dict with earnings_date, days_until, gate_active, timing
Q:\finance-analyzer\portfolio\earnings_calendar.py:162:        or None if no upcoming earnings found.
Q:\finance-analyzer\portfolio\earnings_calendar.py:168:    with _earnings_lock:
Q:\finance-analyzer\portfolio\earnings_calendar.py:169:        cached = _earnings_cache.get(ticker)
Q:\finance-analyzer\portfolio\earnings_calendar.py:170:        if cached and now - cached["time"] < EARNINGS_CACHE_TTL:
Q:\finance-analyzer\portfolio\earnings_calendar.py:171:            return cached["data"]
Q:\finance-analyzer\portfolio\earnings_calendar.py:174:    data = _fetch_earnings_date(ticker)
Q:\finance-analyzer\portfolio\earnings_calendar.py:176:    with _earnings_lock:
Q:\finance-analyzer\portfolio\earnings_calendar.py:177:        _earnings_cache[ticker] = {"data": data, "time": now}
Q:\finance-analyzer\portfolio\earnings_calendar.py:182:def should_gate_earnings(ticker: str) -> bool:
Q:\finance-analyzer\portfolio\earnings_calendar.py:183:    """Check if ticker should be gated to HOLD due to nearby earnings.
Q:\finance-analyzer\portfolio\earnings_calendar.py:192:    prox = get_earnings_proximity(ticker)
Q:\finance-analyzer\portfolio\earnings_calendar.py:199:def get_all_earnings_proximity() -> dict:
Q:\finance-analyzer\portfolio\earnings_calendar.py:200:    """Get earnings proximity for all stock tickers.
Q:\finance-analyzer\portfolio\earnings_calendar.py:207:        prox = get_earnings_proximity(ticker)
Q:\finance-analyzer\portfolio\earnings_calendar.py:213:def clear_cache() -> None:
Q:\finance-analyzer\portfolio\earnings_calendar.py:214:    """Clear the earnings cache (for testing)."""
Q:\finance-analyzer\portfolio\earnings_calendar.py:215:    with _earnings_lock:
Q:\finance-analyzer\portfolio\earnings_calendar.py:216:        _earnings_cache.clear()
Q:\finance-analyzer\portfolio\microstructure_state.py:1:"""Microstructure snapshot accumulator for rolling OFI and spread history.
Q:\finance-analyzer\portfolio\microstructure_state.py:4:The orderbook_flow signal reads the accumulated OFI and spread z-score
Q:\finance-analyzer\portfolio\microstructure_state.py:5:from the persisted state.
Q:\finance-analyzer\portfolio\microstructure_state.py:8:data/microstructure_state.json for cross-process access.
Q:\finance-analyzer\portfolio\microstructure_state.py:18:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\microstructure_state.py:21:logger = logging.getLogger("portfolio.microstructure_state")
Q:\finance-analyzer\portfolio\microstructure_state.py:24:_STATE_FILE = _BASE_DIR / "data" / "microstructure_state.json"
Q:\finance-analyzer\portfolio\microstructure_state.py:26:_MIN_SNAPSHOTS_FOR_OFI = 3
Q:\finance-analyzer\portfolio\microstructure_state.py:28:_MIN_OFI_HISTORY_FOR_ZSCORE = 10
Q:\finance-analyzer\portfolio\microstructure_state.py:29:_MAX_OFI_HISTORY = 120  # ~2h of OFI readings for z-score normalization
Q:\finance-analyzer\portfolio\microstructure_state.py:31:# Multi-scale OFI windows (snapshot counts)
Q:\finance-analyzer\portfolio\microstructure_state.py:32:_OFI_WINDOW_FAST = 5   # ~5 min
Q:\finance-analyzer\portfolio\microstructure_state.py:33:_OFI_WINDOW_MEDIUM = 15  # ~15 min
Q:\finance-analyzer\portfolio\microstructure_state.py:38:# (60s) can both call accumulate_snapshot / get_state concurrently.
Q:\finance-analyzer\portfolio\microstructure_state.py:42:_ofi_history: dict[str, deque] = {}  # rolling OFI values for z-score
Q:\finance-analyzer\portfolio\microstructure_state.py:52:        _ofi_history[ticker] = deque(maxlen=_MAX_OFI_HISTORY)
Q:\finance-analyzer\portfolio\microstructure_state.py:59:        ticker: Canonical ticker (e.g. "XAG-USD")
Q:\finance-analyzer\portfolio\microstructure_state.py:79:    """Compute OFI from accumulated snapshots for a ticker.
Q:\finance-analyzer\portfolio\microstructure_state.py:81:    Returns cumulative OFI over the last N snapshots.
Q:\finance-analyzer\portfolio\microstructure_state.py:87:    if len(snapshots) < _MIN_SNAPSHOTS_FOR_OFI:
Q:\finance-analyzer\portfolio\microstructure_state.py:93:    """Record an OFI value for z-score history tracking.
Q:\finance-analyzer\portfolio\microstructure_state.py:95:    Called once per cycle from get_microstructure_state to avoid
Q:\finance-analyzer\portfolio\microstructure_state.py:104:    """Z-score of OFI relative to its own rolling distribution.
Q:\finance-analyzer\portfolio\microstructure_state.py:115:    if len(history) < _MIN_OFI_HISTORY_FOR_ZSCORE:
Q:\finance-analyzer\portfolio\microstructure_state.py:128:    """Compute OFI at 3 time scales: fast (~5min), medium (~15min), slow (full).
Q:\finance-analyzer\portfolio\microstructure_state.py:138:    ofi_slow = compute_ofi(snapshots) if n >= _MIN_SNAPSHOTS_FOR_OFI else 0.0
Q:\finance-analyzer\portfolio\microstructure_state.py:139:    ofi_medium = compute_ofi(snapshots[-_OFI_WINDOW_MEDIUM:]) if n >= _OFI_WINDOW_MEDIUM else ofi_slow
Q:\finance-analyzer\portfolio\microstructure_state.py:140:    ofi_fast = compute_ofi(snapshots[-_OFI_WINDOW_FAST:]) if n >= _OFI_WINDOW_FAST else ofi_medium
Q:\finance-analyzer\portfolio\microstructure_state.py:142:    # Flow acceleration: compare fast to slow (normalized by snapshot counts).
Q:\finance-analyzer\portfolio\microstructure_state.py:144:    # during warmup (n < _OFI_WINDOW_FAST), fast==slow so acceleration
Q:\finance-analyzer\portfolio\microstructure_state.py:146:    if n >= _OFI_WINDOW_FAST:
Q:\finance-analyzer\portfolio\microstructure_state.py:147:        fast_per_snap = ofi_fast / max(_OFI_WINDOW_FAST - 1, 1)
Q:\finance-analyzer\portfolio\microstructure_state.py:175:def get_microstructure_state(ticker: str) -> dict:
Q:\finance-analyzer\portfolio\microstructure_state.py:176:    """Get current accumulated microstructure state for a ticker.
Q:\finance-analyzer\portfolio\microstructure_state.py:178:    Returns dict with ofi, ofi_zscore, multiscale OFI, and spread_zscore.
Q:\finance-analyzer\portfolio\microstructure_state.py:205:def persist_state() -> None:
Q:\finance-analyzer\portfolio\microstructure_state.py:206:    """Write current microstructure state to disk for cross-process access."""
Q:\finance-analyzer\portfolio\microstructure_state.py:207:    state = {}
Q:\finance-analyzer\portfolio\microstructure_state.py:209:        ms = get_microstructure_state(ticker)
Q:\finance-analyzer\portfolio\microstructure_state.py:211:        state[ticker] = ms
Q:\finance-analyzer\portfolio\microstructure_state.py:212:    if state:
Q:\finance-analyzer\portfolio\microstructure_state.py:213:        atomic_write_json(_STATE_FILE, state)
Q:\finance-analyzer\portfolio\microstructure_state.py:216:def load_persisted_state(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\microstructure_state.py:217:    """Read persisted microstructure state for a ticker.
Q:\finance-analyzer\portfolio\fx_rates.py:3:NOTE: Not migrated to shared_state._cached() because this module has custom
Q:\finance-analyzer\portfolio\fx_rates.py:4:staleness behaviour: Telegram alerting when stale, hardcoded 10.50 fallback,
Q:\finance-analyzer\portfolio\fx_rates.py:5:and ConnectionError semantics. These would be lost with _cached().
Q:\finance-analyzer\portfolio\fx_rates.py:13:from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\fx_rates.py:15:logger = logging.getLogger("portfolio.fx_rates")
Q:\finance-analyzer\portfolio\fx_rates.py:22:# BUG-215: Thread-safe FX cache. Accessed from 8-worker ThreadPoolExecutor.
Q:\finance-analyzer\portfolio\fx_rates.py:23:_fx_lock = threading.Lock()
Q:\finance-analyzer\portfolio\fx_rates.py:24:_fx_cache = {"rate": None, "time": 0}
Q:\finance-analyzer\portfolio\fx_rates.py:30:    with _fx_lock:
Q:\finance-analyzer\portfolio\fx_rates.py:31:        cached_rate = _fx_cache["rate"]
Q:\finance-analyzer\portfolio\fx_rates.py:32:        cached_time = _fx_cache["time"]
Q:\finance-analyzer\portfolio\fx_rates.py:33:    if cached_rate and now - cached_time < 900:  # 15 min cache
Q:\finance-analyzer\portfolio\fx_rates.py:34:        return cached_rate
Q:\finance-analyzer\portfolio\fx_rates.py:36:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\fx_rates.py:50:            with _fx_lock:
Q:\finance-analyzer\portfolio\fx_rates.py:51:                _fx_cache["rate"] = rate
Q:\finance-analyzer\portfolio\fx_rates.py:52:                _fx_cache["time"] = now
Q:\finance-analyzer\portfolio\fx_rates.py:56:    # Fallback to stale cached value
Q:\finance-analyzer\portfolio\fx_rates.py:57:    with _fx_lock:
Q:\finance-analyzer\portfolio\fx_rates.py:58:        cached_rate = _fx_cache["rate"]
Q:\finance-analyzer\portfolio\fx_rates.py:59:        cached_time = _fx_cache["time"]
Q:\finance-analyzer\portfolio\fx_rates.py:60:    if cached_rate:
Q:\finance-analyzer\portfolio\fx_rates.py:61:        age_secs = now - cached_time
Q:\finance-analyzer\portfolio\fx_rates.py:64:            _fx_alert_telegram(age_secs)
Q:\finance-analyzer\portfolio\fx_rates.py:65:        return cached_rate
Q:\finance-analyzer\portfolio\fx_rates.py:66:    # Last resort: hardcoded fallback
Q:\finance-analyzer\portfolio\fx_rates.py:69:    logger.error("Using hardcoded FX fallback rate %.2f SEK — no cached or live rate available", FX_RATE_FALLBACK)
Q:\finance-analyzer\portfolio\fx_rates.py:70:    _fx_alert_telegram(None)
Q:\finance-analyzer\portfolio\fx_rates.py:74:def _fx_alert_telegram(age_secs):
Q:\finance-analyzer\portfolio\fx_rates.py:77:    with _fx_lock:
Q:\finance-analyzer\portfolio\fx_rates.py:78:        last_alert = _fx_cache.get("_last_fx_alert", 0)
Q:\finance-analyzer\portfolio\fx_rates.py:81:        _fx_cache["_last_fx_alert"] = now
Q:\finance-analyzer\portfolio\fx_rates.py:87:            msg = f"_FX WARNING: Using hardcoded fallback rate {FX_RATE_FALLBACK} SEK. No live or cached rate available._"
Q:\finance-analyzer\portfolio\sentiment.py:1:"""Sentiment analysis — multi-model A/B testing with FinGPT, CryptoBERT, and TradingHero.
Q:\finance-analyzer\portfolio\sentiment.py:9:  - CryptoBERT: crypto headlines (legacy primary)
Q:\finance-analyzer\portfolio\sentiment.py:12:  - FinBERT: CPU fallback (shadow)
Q:\finance-analyzer\portfolio\sentiment.py:23:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\sentiment.py:26:from portfolio.file_utils import atomic_append_jsonl
Q:\finance-analyzer\portfolio\sentiment.py:27:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\sentiment.py:35:    CRYPTOBERT_SCRIPT = r"Q:\models\cryptobert_infer.py"
Q:\finance-analyzer\portfolio\sentiment.py:37:    FINBERT_SCRIPT = r"Q:\models\finbert_infer.py"
Q:\finance-analyzer\portfolio\sentiment.py:40:    CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
Q:\finance-analyzer\portfolio\sentiment.py:42:    FINBERT_SCRIPT = "/home/deck/models/finbert_infer.py"
Q:\finance-analyzer\portfolio\sentiment.py:51:    "XAG": "SILVER",
Q:\finance-analyzer\portfolio\sentiment.py:90:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
Q:\finance-analyzer\portfolio\sentiment.py:93:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
Q:\finance-analyzer\portfolio\sentiment.py:100:            "published": datetime.fromtimestamp(
Q:\finance-analyzer\portfolio\sentiment.py:101:                a["published_on"], tz=UTC
Q:\finance-analyzer\portfolio\sentiment.py:107:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
Q:\finance-analyzer\portfolio\sentiment.py:111:# Mapping from short crypto ticker to yfinance symbol for fallback
Q:\finance-analyzer\portfolio\sentiment.py:115:def _fetch_crypto_headlines_yahoo_fallback(ticker, limit=20):
Q:\finance-analyzer\portfolio\sentiment.py:123:            logger.info("[CryptoCompare] fallback to Yahoo Finance for %s: %d articles",
Q:\finance-analyzer\portfolio\sentiment.py:127:        logger.debug("[Yahoo News] crypto fallback error for %s: %s", ticker, e)
Q:\finance-analyzer\portfolio\sentiment.py:153:                "published": pub or datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\sentiment.py:160:    """Fetch headlines from NewsAPI with optional custom search query."""
Q:\finance-analyzer\portfolio\sentiment.py:177:            "published": a.get("publishedAt", datetime.now(UTC).isoformat()),
Q:\finance-analyzer\portfolio\sentiment.py:185:    """Fetch from NewsAPI and track the call against daily quota.
Q:\finance-analyzer\portfolio\sentiment.py:190:    from portfolio.shared_state import newsapi_track_call
Q:\finance-analyzer\portfolio\sentiment.py:198:    """Fetch stock headlines. NewsAPI for priority tickers (metals), Yahoo for the rest."""
Q:\finance-analyzer\portfolio\sentiment.py:199:    from portfolio.shared_state import (
Q:\finance-analyzer\portfolio\sentiment.py:200:        _cached,
Q:\finance-analyzer\portfolio\sentiment.py:208:    # NewsAPI: only for priority tickers during active hours (metals get 20-min refresh)
Q:\finance-analyzer\portfolio\sentiment.py:213:            cached_newsapi = _cached(
Q:\finance-analyzer\portfolio\sentiment.py:222:            if cached_newsapi:
Q:\finance-analyzer\portfolio\sentiment.py:223:                articles.extend(cached_newsapi)
Q:\finance-analyzer\portfolio\sentiment.py:225:            logger.debug("[NewsAPI] error for %s: %s", ticker, e)
Q:\finance-analyzer\portfolio\sentiment.py:227:    # Yahoo Finance: fallback for metals, primary for everything else
Q:\finance-analyzer\portfolio\sentiment.py:244:        logger.debug("[Headlines %s] %d NewsAPI + %d Yahoo = %d total",
Q:\finance-analyzer\portfolio\sentiment.py:256:_INPROC_BERT_MAP = {
Q:\finance-analyzer\portfolio\sentiment.py:257:    CRYPTOBERT_SCRIPT: "CryptoBERT",
Q:\finance-analyzer\portfolio\sentiment.py:259:    FINBERT_SCRIPT: "FinBERT",
Q:\finance-analyzer\portfolio\sentiment.py:266:    Tries the in-process BERT cache first (portfolio.bert_sentiment) because
Q:\finance-analyzer\portfolio\sentiment.py:270:    cache dir is missing.
Q:\finance-analyzer\portfolio\sentiment.py:272:    model_name = _INPROC_BERT_MAP.get(script)
Q:\finance-analyzer\portfolio\sentiment.py:279:            # we end up stuck on the subprocess fallback. sentiment.py already
Q:\finance-analyzer\portfolio\sentiment.py:282:                "In-process BERT %s failed, falling back to subprocess: %s",
Q:\finance-analyzer\portfolio\sentiment.py:287:    # known BERT models, though that doesn't happen today).
Q:\finance-analyzer\portfolio\sentiment.py:290:        input=json.dumps(texts),
Q:\finance-analyzer\portfolio\sentiment.py:302:# data/sentiment_ab_log.jsonl alongside the primary model's vote (CryptoBERT
Q:\finance-analyzer\portfolio\sentiment.py:320:# primary model and FinBERT shadow are still computed inline, but their A/B
Q:\finance-analyzer\portfolio\sentiment.py:353:    2026-04-28: cryptobert_shadow slot added; CryptoBERT was demoted from
Q:\finance-analyzer\portfolio\sentiment.py:370:    """Attach the inline FinBERT shadow result to a pending A/B entry."""
Q:\finance-analyzer\portfolio\sentiment.py:378:    """Attach the inline CryptoBERT shadow result to a pending A/B entry.
Q:\finance-analyzer\portfolio\sentiment.py:380:    Added 2026-04-28 when CryptoBERT was demoted from crypto primary to
Q:\finance-analyzer\portfolio\sentiment.py:421:    leftover state.
Q:\finance-analyzer\portfolio\sentiment.py:472:            # FinBERT shadow (already aggregated inline during get_sentiment).
Q:\finance-analyzer\portfolio\sentiment.py:477:            # CryptoBERT shadow (added 2026-04-28 — was the primary; demoted
Q:\finance-analyzer\portfolio\sentiment.py:491:    """Run FinBERT sentiment inference.
Q:\finance-analyzer\portfolio\sentiment.py:493:    2026-04-09 (fix/bert-inproc-gpu): routes through _run_model so FinBERT
Q:\finance-analyzer\portfolio\sentiment.py:494:    also benefits from the in-process GPU cache. _run_model's _INPROC_BERT_MAP
Q:\finance-analyzer\portfolio\sentiment.py:495:    knows that FINBERT_SCRIPT -> "FinBERT" and will hit bert_sentiment.predict
Q:\finance-analyzer\portfolio\sentiment.py:498:    return _run_model(FINBERT_SCRIPT, texts)
Q:\finance-analyzer\portfolio\sentiment.py:506:    """Group headlines by keyword overlap and time proximity.
Q:\finance-analyzer\portfolio\sentiment.py:518:    from portfolio.news_keywords import score_headline
Q:\finance-analyzer\portfolio\sentiment.py:520:    # Extract keywords per headline
Q:\finance-analyzer\portfolio\sentiment.py:521:    headline_keywords = []
Q:\finance-analyzer\portfolio\sentiment.py:531:        headline_keywords.append(words)
Q:\finance-analyzer\portfolio\sentiment.py:533:    # Simple greedy clustering by keyword overlap
Q:\finance-analyzer\portfolio\sentiment.py:542:        kw_i = headline_keywords[i]
Q:\finance-analyzer\portfolio\sentiment.py:547:            kw_j = headline_keywords[j]
Q:\finance-analyzer\portfolio\sentiment.py:549:            # Cluster if they share 2+ keywords or 1 matched keyword
Q:\finance-analyzer\portfolio\sentiment.py:569:    "inflation", "rate", "cut", "hike", "layoffs", "earnings", "fomc",
Q:\finance-analyzer\portfolio\sentiment.py:591:    """Return per-sentiment weights from keyword scoring + dissemination."""
Q:\finance-analyzer\portfolio\sentiment.py:593:        from portfolio.news_keywords import score_headline
Q:\finance-analyzer\portfolio\sentiment.py:622:    When headlines are provided, score weights from news_keywords.score_headline()
Q:\finance-analyzer\portfolio\sentiment.py:623:    amplify high-impact keywords (tariff/war/crash 3x). dissemination_mult
Q:\finance-analyzer\portfolio\sentiment.py:682:def _filter_relevant_headlines(articles, ticker, *, fallback_n=3):
Q:\finance-analyzer\portfolio\sentiment.py:685:    Uses news_keywords.is_relevant_headline (keyword OR ticker-synonym match)
Q:\finance-analyzer\portfolio\sentiment.py:690:    Falls back to most-recent `fallback_n` if the filter would drop
Q:\finance-analyzer\portfolio\sentiment.py:697:    from portfolio.news_keywords import is_credible_source, is_relevant_headline
Q:\finance-analyzer\portfolio\sentiment.py:714:    # All-irrelevant fallback: keep the most-recent N articles. Sort by the
Q:\finance-analyzer\portfolio\sentiment.py:723:    return sorted_articles[:fallback_n]
Q:\finance-analyzer\portfolio\sentiment.py:730:            "ts": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\sentiment.py:739:        atomic_append_jsonl(AB_LOG_FILE, entry)
Q:\finance-analyzer\portfolio\sentiment.py:753:      1. Crypto primary model swapped CryptoBERT -> Trading-Hero-LLM.
Q:\finance-analyzer\portfolio\sentiment.py:754:         CryptoBERT was 99.1% neutral on 2,817 wire-feed samples (it was
Q:\finance-analyzer\portfolio\sentiment.py:756:         CryptoBERT now runs as a shadow for continuity of the 30d accuracy
Q:\finance-analyzer\portfolio\sentiment.py:766:    Shadow models (logged): CryptoBERT (crypto only), FinGPT, FinBERT
Q:\finance-analyzer\portfolio\sentiment.py:781:    # See module/function docstring for the CryptoBERT demotion rationale.
Q:\finance-analyzer\portfolio\sentiment.py:803:    # fallback so we never go silent on slow-news days.
Q:\finance-analyzer\portfolio\sentiment.py:810:        from portfolio.news_keywords import dissemination_score
Q:\finance-analyzer\portfolio\sentiment.py:859:    ab_key = f"{short}:{datetime.now(UTC).isoformat()}"
Q:\finance-analyzer\portfolio\sentiment.py:867:    # ministral/qwen3 (which go through _cached_or_enqueue's should_enqueue_fn),
Q:\finance-analyzer\portfolio\sentiment.py:868:    # fingpt enqueues directly because it doesn't use the signal cache — it
Q:\finance-analyzer\portfolio\sentiment.py:892:    # Shadow: FinBERT (CPU, fast) — still runs inline because it's cheap
Q:\finance-analyzer\portfolio\sentiment.py:903:                "model": "FinBERT",
Q:\finance-analyzer\portfolio\sentiment.py:909:        logger.debug("FinBERT shadow failed: %s", e)
Q:\finance-analyzer\portfolio\sentiment.py:911:    # Shadow: CryptoBERT — demoted from primary 2026-04-28. Kept as shadow
Q:\finance-analyzer\portfolio\sentiment.py:915:    # buffer; picked up by flush_ab_log alongside FinGPT and FinBERT.
Q:\finance-analyzer\portfolio\sentiment.py:918:            crypto_results = _run_model(CRYPTOBERT_SCRIPT, titles)
Q:\finance-analyzer\portfolio\sentiment.py:924:                    "model": "CryptoBERT",
Q:\finance-analyzer\portfolio\sentiment.py:930:            logger.debug("CryptoBERT shadow failed: %s", e)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:14:on XAG 3h — root cause was 5-day lookbacks evaluated against 3h outcomes
Q:\finance-analyzer\portfolio\metals_cross_assets.py:26:from portfolio.shared_state import _cached
Q:\finance-analyzer\portfolio\metals_cross_assets.py:30:_CROSS_TTL = 300
Q:\finance-analyzer\portfolio\metals_cross_assets.py:31:_GVZ_TTL = 600
Q:\finance-analyzer\portfolio\metals_cross_assets.py:32:# Intraday TTL is shorter — 60m bars refresh at the start of each hour,
Q:\finance-analyzer\portfolio\metals_cross_assets.py:34:_CROSS_INTRADAY_TTL = 180
Q:\finance-analyzer\portfolio\metals_cross_assets.py:41:    commodity futures (HG=F, CL=F) to Binance FAPI for 7.7s-fresh data,
Q:\finance-analyzer\portfolio\metals_cross_assets.py:47:    # Rough period-to-limit mapping — price_source/Binance/Alpaca use row
Q:\finance-analyzer\portfolio\metals_cross_assets.py:67:    # yfinance convention). Backends normalize to lowercase; re-capitalize
Q:\finance-analyzer\portfolio\metals_cross_assets.py:76:def _nocache(func):
Q:\finance-analyzer\portfolio\metals_cross_assets.py:77:    """Mark function so tests can bypass _cached via func.__wrapped__."""
Q:\finance-analyzer\portfolio\metals_cross_assets.py:92:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:109:    return _cached("cross_copper", _CROSS_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:112:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:132:    return _cached("cross_gvz", _GVZ_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:135:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:163:    return _cached("cross_gs_ratio", _CROSS_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:166:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:168:    """WTI Crude Oil futures (CL=F) price and momentum."""
Q:\finance-analyzer\portfolio\metals_cross_assets.py:170:        df = _yf_download("CL=F", period="3mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:181:    return _cached("cross_oil", _CROSS_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:184:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:199:    return _cached("cross_spy", _CROSS_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:221:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:236:    return _cached("cross_copper_intraday", _CROSS_INTRADAY_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:239:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:260:    return _cached("cross_gs_ratio_intraday", _CROSS_INTRADAY_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:263:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:267:        df = _yf_download("CL=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:278:    return _cached("cross_oil_intraday", _CROSS_INTRADAY_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_cross_assets.py:281:@_nocache
Q:\finance-analyzer\portfolio\metals_cross_assets.py:296:    return _cached("cross_spy_intraday", _CROSS_INTRADAY_TTL, _fetch)
Q:\finance-analyzer\portfolio\api_utils.py:15:# Config cache (thread-safe)
Q:\finance-analyzer\portfolio\api_utils.py:17:_config_cache = None
Q:\finance-analyzer\portfolio\api_utils.py:23:    global _config_cache, _config_mtime
Q:\finance-analyzer\portfolio\api_utils.py:29:            if _config_cache is None or mtime != _config_mtime:
Q:\finance-analyzer\portfolio\api_utils.py:31:                    _config_cache = json.load(f)
Q:\finance-analyzer\portfolio\api_utils.py:34:            if _config_cache is None:
Q:\finance-analyzer\portfolio\api_utils.py:36:    return _config_cache
Q:\finance-analyzer\portfolio\api_utils.py:57:    """Get Binance API credentials."""
Q:\finance-analyzer\portfolio\funding_rate.py:2:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\funding_rate.py:3:from portfolio.shared_state import FUNDING_RATE_TTL, _cached
Q:\finance-analyzer\portfolio\funding_rate.py:11:def _fetch_funding_rate(ticker):
Q:\finance-analyzer\portfolio\funding_rate.py:12:    """Fetch and interpret funding rate for a single ticker."""
Q:\finance-analyzer\portfolio\funding_rate.py:18:        label="funding_rate",
Q:\finance-analyzer\portfolio\funding_rate.py:23:    # Adversarial review 04-29 DE-P1-1 (2026-05-02): Binance FAPI premiumIndex
Q:\finance-analyzer\portfolio\funding_rate.py:25:    # deployment / weird symbol state / schema change can return a payload
Q:\finance-analyzer\portfolio\funding_rate.py:28:    # funding signal for the cycle (and 8 of those = whole cycle thread pool
Q:\finance-analyzer\portfolio\funding_rate.py:31:    last_funding = data.get("lastFundingRate")
Q:\finance-analyzer\portfolio\funding_rate.py:33:    if last_funding is None or mark_price is None:
Q:\finance-analyzer\portfolio\funding_rate.py:36:        rate = float(last_funding)
Q:\finance-analyzer\portfolio\funding_rate.py:41:    # Normal funding ~0.01% (0.0001). Thresholds:
Q:\finance-analyzer\portfolio\funding_rate.py:59:def get_funding_rate(ticker):
Q:\finance-analyzer\portfolio\funding_rate.py:62:    return _cached(f"funding_rate_{ticker}", FUNDING_RATE_TTL,
Q:\finance-analyzer\portfolio\funding_rate.py:63:                   _fetch_funding_rate, ticker)
Q:\finance-analyzer\portfolio\funding_rate.py:68:        print(f"{t}: {get_funding_rate(t)}")
Q:\finance-analyzer\portfolio\metals_orderbook.py:1:"""Binance FAPI order book depth and recent trades for metals.
Q:\finance-analyzer\portfolio\metals_orderbook.py:3:Fetches L2 snapshots and trade ticks for XAUUSDT / XAGUSDT.
Q:\finance-analyzer\portfolio\metals_orderbook.py:4:Data feeds into microstructure feature computations (OFI, depth
Q:\finance-analyzer\portfolio\metals_orderbook.py:5:imbalance, VPIN, spread metrics).
Q:\finance-analyzer\portfolio\metals_orderbook.py:16:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\metals_orderbook.py:17:from portfolio.shared_state import _binance_limiter, _cached
Q:\finance-analyzer\portfolio\metals_orderbook.py:23:    "XAG-USD": "XAGUSDT",
Q:\finance-analyzer\portfolio\metals_orderbook.py:28:_DEPTH_TTL = 10
Q:\finance-analyzer\portfolio\metals_orderbook.py:29:_TRADES_TTL = 10
Q:\finance-analyzer\portfolio\metals_orderbook.py:33:    """Fetch JSON from Binance FAPI with rate limiting and retry."""
Q:\finance-analyzer\portfolio\metals_orderbook.py:38:def _nocache(func):
Q:\finance-analyzer\portfolio\metals_orderbook.py:39:    """Mark a function so tests can call func.__wrapped__ to bypass _cached."""
Q:\finance-analyzer\portfolio\metals_orderbook.py:47:@_nocache
Q:\finance-analyzer\portfolio\metals_orderbook.py:49:    """Fetch order book depth snapshot from Binance FAPI.
Q:\finance-analyzer\portfolio\metals_orderbook.py:87:    return _cached(f"depth_{ticker}_{limit}", _DEPTH_TTL, _fetch)
Q:\finance-analyzer\portfolio\metals_orderbook.py:90:@_nocache
Q:\finance-analyzer\portfolio\metals_orderbook.py:92:    """Fetch recent trades from Binance FAPI.
Q:\finance-analyzer\portfolio\metals_orderbook.py:121:    return _cached(f"trades_{ticker}_{limit}", _TRADES_TTL, _fetch)
Q:\finance-analyzer\portfolio\tickers.py:8:from functools import lru_cache
Q:\finance-analyzer\portfolio\tickers.py:14:    # Crypto (Binance spot)
Q:\finance-analyzer\portfolio\tickers.py:17:    # Metals (Binance futures)
Q:\finance-analyzer\portfolio\tickers.py:19:    "XAG-USD": {"binance_fapi": "XAGUSDT"},
Q:\finance-analyzer\portfolio\tickers.py:32:METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
Q:\finance-analyzer\portfolio\tickers.py:74:    # where the system already has edge (XAG 5d consensus 61.2%). The
Q:\finance-analyzer\portfolio\tickers.py:118:                        # on XAG. In oscillator_trend cluster where momentum_factors
Q:\finance-analyzer\portfolio\tickers.py:143:                        # XAG 34.9%, XAU 40.2%, MSTR 42.6%; 5065 total sam). Also weak at
Q:\finance-analyzer\portfolio\tickers.py:155:                        # CryptoBERT predictions are noise. High-volume signal actively hurting
Q:\finance-analyzer\portfolio\tickers.py:158:# 2026-04-11 research session changes:
Q:\finance-analyzer\portfolio\tickers.py:164:# funding: removed from DISABLED — 74.2% at 3h (535 samples) but 29.9% at 1d.
Q:\finance-analyzer\portfolio\tickers.py:171:_DISABLED_REASON_ENTRY_RE = re.compile(
Q:\finance-analyzer\portfolio\tickers.py:174:_DISABLED_REASON_CONT_RE = re.compile(r'^(\s+)#\s*(.*)$')
Q:\finance-analyzer\portfolio\tickers.py:189:@lru_cache(maxsize=1)
Q:\finance-analyzer\portfolio\tickers.py:202:    block_match = re.search(
Q:\finance-analyzer\portfolio\tickers.py:248:# Skipped for US stocks outside market hours to save GPU resources.
Q:\finance-analyzer\portfolio\tickers.py:257:    "fear_greed",
Q:\finance-analyzer\portfolio\tickers.py:261:    "funding",
Q:\finance-analyzer\portfolio\microstructure.py:9:    - VPIN: Volume-synchronized probability of informed trading
Q:\finance-analyzer\portfolio\microstructure.py:10:    - OFI: Order Flow Imbalance from quote changes  [Cont et al. 2014]
Q:\finance-analyzer\portfolio\microstructure.py:23:def depth_imbalance(depth: dict, levels: int | None = None) -> float:
Q:\finance-analyzer\portfolio\microstructure.py:41:def trade_flow_imbalance(trades: list[dict]) -> dict[str, float] | None:
Q:\finance-analyzer\portfolio\microstructure.py:42:    """Compute signed volume imbalance from recent trades.
Q:\finance-analyzer\portfolio\microstructure.py:51:    imbalance = signed_vol / total_vol if total_vol > 0 else 0.0
Q:\finance-analyzer\portfolio\microstructure.py:56:        "imbalance_ratio": imbalance,
Q:\finance-analyzer\portfolio\microstructure.py:64:    """VPIN: average absolute buy-sell imbalance per volume bucket.
Q:\finance-analyzer\portfolio\microstructure.py:66:    High VPIN (>0.6) → toxic flow, likely informed trading.
Q:\finance-analyzer\portfolio\microstructure.py:67:    Low VPIN (<0.3) → balanced, uninformed flow.
Q:\finance-analyzer\portfolio\microstructure.py:76:    imbalances = []
Q:\finance-analyzer\portfolio\microstructure.py:95:                    imbalances.append(
Q:\finance-analyzer\portfolio\microstructure.py:102:    if not imbalances:
Q:\finance-analyzer\portfolio\microstructure.py:104:    return float(np.mean(imbalances))
Q:\finance-analyzer\portfolio\microstructure.py:110:    Implements the Cont et al. (2014) OFI formula.
Q:\finance-analyzer\portfolio\microstructure.py:111:    Positive OFI → net buying pressure.
Q:\finance-analyzer\portfolio\microstructure.py:112:    Negative OFI → net selling pressure.
Q:\finance-analyzer\portfolio\onchain_data.py:3:Fetches MVRV Z-Score, SOPR, NUPL, realized price, exchange netflow,
Q:\finance-analyzer\portfolio\onchain_data.py:18:from datetime import datetime
Q:\finance-analyzer\portfolio\onchain_data.py:22:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\onchain_data.py:23:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\onchain_data.py:24:from portfolio.shared_state import _cached
Q:\finance-analyzer\portfolio\onchain_data.py:30:    """A-DE-5 (2026-04-11): Coerce a cache timestamp to a unix-epoch float.
Q:\finance-analyzer\portfolio\onchain_data.py:32:    Older versions of onchain_cache.json stored "ts" as an ISO 8601 string
Q:\finance-analyzer\portfolio\onchain_data.py:34:    `time.time() - cache_ts < TTL` which crashes with TypeError when ts is
Q:\finance-analyzer\portfolio\onchain_data.py:38:        - "2026-04-11T..."      → parsed via datetime.fromisoformat
Q:\finance-analyzer\portfolio\onchain_data.py:39:        - anything else / fail  → 0.0 (treated as ancient → cache miss)
Q:\finance-analyzer\portfolio\onchain_data.py:41:    Returning 0 on failure is safe: it forces a cache miss, which costs
Q:\finance-analyzer\portfolio\onchain_data.py:54:            return datetime.fromisoformat(iso).timestamp()
Q:\finance-analyzer\portfolio\onchain_data.py:58:    # forces a cache miss, which costs one extra API call but never breaks
Q:\finance-analyzer\portfolio\onchain_data.py:71:CACHE_FILE = DATA_DIR / "onchain_cache.json"
Q:\finance-analyzer\portfolio\onchain_data.py:74:ONCHAIN_TTL = 43200  # 12 hours
Q:\finance-analyzer\portfolio\onchain_data.py:93:# Persistent cache (survives restarts)
Q:\finance-analyzer\portfolio\onchain_data.py:96:def _save_onchain_cache(data):
Q:\finance-analyzer\portfolio\onchain_data.py:97:    """Save on-chain data to persistent cache file."""
Q:\finance-analyzer\portfolio\onchain_data.py:99:        # BUG-125: Use atomic_write_json to prevent corrupt cache on crash
Q:\finance-analyzer\portfolio\onchain_data.py:100:        atomic_write_json(CACHE_FILE, data, ensure_ascii=False)
Q:\finance-analyzer\portfolio\onchain_data.py:102:        logger.warning("Failed to write onchain cache", exc_info=True)
Q:\finance-analyzer\portfolio\onchain_data.py:105:def _load_onchain_cache(max_age_seconds=ONCHAIN_TTL):
Q:\finance-analyzer\portfolio\onchain_data.py:106:    """Load on-chain data from persistent cache if fresh enough."""
Q:\finance-analyzer\portfolio\onchain_data.py:116:        logger.warning("On-chain cache load failed: %s", e, exc_info=True)
Q:\finance-analyzer\portfolio\onchain_data.py:127:    Skips retries on failure since retrying just burns more of the
Q:\finance-analyzer\portfolio\onchain_data.py:137:    """Fetch latest MVRV and MVRV Z-Score."""
Q:\finance-analyzer\portfolio\onchain_data.py:148:    """Fetch latest SOPR (Spent Output Profit Ratio)."""
Q:\finance-analyzer\portfolio\onchain_data.py:239:    # Save to persistent cache
Q:\finance-analyzer\portfolio\onchain_data.py:240:    _save_onchain_cache(result)
Q:\finance-analyzer\portfolio\onchain_data.py:245:    """Get on-chain data for BTC, using in-memory + persistent cache.
Q:\finance-analyzer\portfolio\onchain_data.py:249:    # H12/DC-R3-5: On restart, seed the in-memory cache from the persistent cache
Q:\finance-analyzer\portfolio\onchain_data.py:255:        # this cache stored "ts" as an ISO 8601 string instead of an epoch
Q:\finance-analyzer\portfolio\onchain_data.py:260:        cache_ts = _coerce_epoch(raw_ts)
Q:\finance-analyzer\portfolio\onchain_data.py:261:        if time.time() - cache_ts < ONCHAIN_TTL:
Q:\finance-analyzer\portfolio\onchain_data.py:262:            # Still fresh — pre-populate the in-memory cache so _cached() returns
Q:\finance-analyzer\portfolio\onchain_data.py:264:            from portfolio.shared_state import _cache_lock, _tool_cache
Q:\finance-analyzer\portfolio\onchain_data.py:265:            with _cache_lock:
Q:\finance-analyzer\portfolio\onchain_data.py:266:                if "onchain_btc" not in _tool_cache:
Q:\finance-analyzer\portfolio\onchain_data.py:267:                    _tool_cache["onchain_btc"] = {
Q:\finance-analyzer\portfolio\onchain_data.py:269:                        "time": cache_ts,
Q:\finance-analyzer\portfolio\onchain_data.py:270:                        "ttl": ONCHAIN_TTL,
Q:\finance-analyzer\portfolio\onchain_data.py:272:                    logger.debug("On-chain: seeded in-memory cache from persistent (age %.0fs)",
Q:\finance-analyzer\portfolio\onchain_data.py:273:                                 time.time() - cache_ts)
Q:\finance-analyzer\portfolio\onchain_data.py:277:        # Try persistent cache even without token
Q:\finance-analyzer\portfolio\onchain_data.py:278:        cached = _load_onchain_cache(max_age_seconds=ONCHAIN_TTL * 2)
Q:\finance-analyzer\portfolio\onchain_data.py:279:        if cached:
Q:\finance-analyzer\portfolio\onchain_data.py:280:            logger.debug("No BGeometrics token, using stale cache")
Q:\finance-analyzer\portfolio\onchain_data.py:281:            return cached
Q:\finance-analyzer\portfolio\onchain_data.py:284:    return _cached("onchain_btc", ONCHAIN_TTL, _fetch_all_onchain, token)
Q:\finance-analyzer\portfolio\onchain_data.py:301:    # MVRV Z-Score zones
Q:\finance-analyzer\portfolio\onchain_data.py:311:    # SOPR zones
Q:\finance-analyzer\portfolio\onchain_data.py:329:            interp["nupl_zone"] = "greed"
Q:\finance-analyzer\portfolio\crypto_scheduler.py:11:Pattern: follows digest.py (state file + sentinel + time check).
Q:\finance-analyzer\portfolio\crypto_scheduler.py:21:from datetime import datetime
Q:\finance-analyzer\portfolio\crypto_scheduler.py:24:from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
Q:\finance-analyzer\portfolio\crypto_scheduler.py:32:STATE_FILE = DATA_DIR / "crypto_scheduler_state.json"
Q:\finance-analyzer\portfolio\crypto_scheduler.py:47:def _get_state():
Q:\finance-analyzer\portfolio\crypto_scheduler.py:48:    """Load scheduler state from file."""
Q:\finance-analyzer\portfolio\crypto_scheduler.py:49:    state = load_json(STATE_FILE, default={})
Q:\finance-analyzer\portfolio\crypto_scheduler.py:50:    return state if isinstance(state, dict) else {}
Q:\finance-analyzer\portfolio\crypto_scheduler.py:53:def _save_state(state):
Q:\finance-analyzer\portfolio\crypto_scheduler.py:54:    """Save scheduler state to file."""
Q:\finance-analyzer\portfolio\crypto_scheduler.py:55:    atomic_write_json(STATE_FILE, state)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:70:    tz_name = crypto_cfg.get("timezone", DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:73:        tz = zoneinfo.ZoneInfo(tz_name)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:75:        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:77:    now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:84:    state = _get_state()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:85:    last_ts = state.get("last_report_time", 0)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:90:    last_hour = state.get("last_report_hour", -1)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:91:    last_date = state.get("last_report_date", "")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:140:    fg = summary.get("fear_greed", {})
Q:\finance-analyzer\portfolio\crypto_scheduler.py:148:    # FOMC
Q:\finance-analyzer\portfolio\crypto_scheduler.py:178:        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:179:        now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:209:    lines.append(f"📊 {fg_str} | DXY {dxy_val} ({dxy_chg}%) | VIX {vix_val} | FOMC {fomc_days}d")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:232:    btc_funding = btc_futures.get("funding_rate_pct", "?")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:234:    lines.append(f"Futures: Funding {btc_funding}% | L/S {btc_ls}")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:260:    eth_funding = eth_futures.get("funding_rate_pct", "?")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:262:    lines.append(f"Futures: Funding {eth_funding}% | L/S {eth_ls}")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:275:    # Fundamentals from cache
Q:\finance-analyzer\portfolio\crypto_scheduler.py:277:        fund_cache = load_json(DATA_DIR / "fundamentals_cache.json", default={})
Q:\finance-analyzer\portfolio\crypto_scheduler.py:278:        mstr_fund = fund_cache.get("MSTR", {})
Q:\finance-analyzer\portfolio\crypto_scheduler.py:285:        logger.debug("Failed to load MSTR fundamentals from cache", exc_info=True)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:310:        "ts": datetime.now().astimezone().isoformat(),
Q:\finance-analyzer\portfolio\crypto_scheduler.py:313:        "crypto_fear_greed": crypto_fg,
Q:\finance-analyzer\portfolio\crypto_scheduler.py:323:            "funding_rate": btc_futures.get("funding_rate_pct"),
Q:\finance-analyzer\portfolio\crypto_scheduler.py:331:            "funding_rate": eth_futures.get("funding_rate_pct"),
Q:\finance-analyzer\portfolio\crypto_scheduler.py:371:            atomic_append_jsonl(LOG_FILE, log_entry)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:373:        # Update state
Q:\finance-analyzer\portfolio\crypto_scheduler.py:375:            tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:376:            now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:381:        state = _get_state()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:382:        state["last_report_time"] = time.time()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:383:        state["last_report_hour"] = hour
Q:\finance-analyzer\portfolio\crypto_scheduler.py:384:        state["last_report_date"] = today_str
Q:\finance-analyzer\portfolio\crypto_scheduler.py:385:        _save_state(state)
Q:\finance-analyzer\portfolio\http_retry.py:1:"""HTTP retry utility with exponential backoff for finance-analyzer API calls."""
Q:\finance-analyzer\portfolio\http_retry.py:17:def fetch_with_retry(url, method="GET", retries=DEFAULT_RETRIES,
Q:\finance-analyzer\portfolio\http_retry.py:18:                     backoff=DEFAULT_BACKOFF, backoff_factor=DEFAULT_BACKOFF_FACTOR,
Q:\finance-analyzer\portfolio\http_retry.py:20:                     session=None):
Q:\finance-analyzer\portfolio\http_retry.py:21:    """Make an HTTP request with exponential backoff retry.
Q:\finance-analyzer\portfolio\http_retry.py:25:    requester = session or requests
Q:\finance-analyzer\portfolio\http_retry.py:40:                wait = backoff * (backoff_factor ** attempt)
Q:\finance-analyzer\portfolio\http_retry.py:43:                # H26/HTTP1: Honour Telegram retry_after parameter for 429 responses.
Q:\finance-analyzer\portfolio\http_retry.py:46:                        retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
Q:\finance-analyzer\portfolio\http_retry.py:48:                        retry_after = wait
Q:\finance-analyzer\portfolio\http_retry.py:49:                    wait = retry_after
Q:\finance-analyzer\portfolio\http_retry.py:50:                logger.warning("HTTP %s from %s, retry %d/%d in %.1fs",
Q:\finance-analyzer\portfolio\http_retry.py:60:                wait = backoff * (backoff_factor ** attempt)
Q:\finance-analyzer\portfolio\http_retry.py:63:                logger.warning("%s from %s, retry %d/%d in %.1fs",
Q:\finance-analyzer\portfolio\http_retry.py:75:               label="", headers=None, params=None, timeout=30, session=None,
Q:\finance-analyzer\portfolio\http_retry.py:79:    Combines fetch_with_retry() + raise_for_status() + .json() into one call.
Q:\finance-analyzer\portfolio\http_retry.py:81:    resp = fetch_with_retry(url, method=method, retries=retries, timeout=timeout,
Q:\finance-analyzer\portfolio\http_retry.py:82:                            headers=headers, params=params, session=session)
Q:\finance-analyzer\portfolio\news_keywords.py:1:"""Shared keyword dictionaries and sector mappings for news/event signals.
Q:\finance-analyzer\portfolio\news_keywords.py:4:- Tiered keyword lists with severity weights
Q:\finance-analyzer\portfolio\news_keywords.py:12:from datetime import UTC
Q:\finance-analyzer\portfolio\news_keywords.py:65:    "earnings miss": 1.5,
Q:\finance-analyzer\portfolio\news_keywords.py:66:    "earnings beat": 1.5,
Q:\finance-analyzer\portfolio\news_keywords.py:81:    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), weight)
Q:\finance-analyzer\portfolio\news_keywords.py:98:    "metals": {"XAU-USD", "XAG-USD"},
Q:\finance-analyzer\portfolio\news_keywords.py:140:    """Score a headline by keyword severity.
Q:\finance-analyzer\portfolio\news_keywords.py:143:        (max_weight, matched_keywords) — max_weight is the highest keyword
Q:\finance-analyzer\portfolio\news_keywords.py:144:        weight found (1.0 if no keywords match), matched_keywords lists all
Q:\finance-analyzer\portfolio\news_keywords.py:145:        matched keyword strings.
Q:\finance-analyzer\portfolio\news_keywords.py:162:def keyword_severity(title: str) -> str:
Q:\finance-analyzer\portfolio\news_keywords.py:216:    from datetime import datetime
Q:\finance-analyzer\portfolio\news_keywords.py:224:                ts = datetime.fromtimestamp(pub, tz=UTC)
Q:\finance-analyzer\portfolio\news_keywords.py:228:                ts = datetime.fromisoformat(pub_str)
Q:\finance-analyzer\portfolio\news_keywords.py:252:def get_sector_impact(keyword: str, ticker: str) -> str | None:
Q:\finance-analyzer\portfolio\news_keywords.py:253:    """Get the directional impact of a keyword on a specific ticker.
Q:\finance-analyzer\portfolio\news_keywords.py:257:    impacts = KEYWORD_SECTOR_IMPACT.get(keyword.lower(), {})
Q:\finance-analyzer\portfolio\news_keywords.py:277:#   1. It triggers a keyword from score_headline (weight > 1.0), OR
Q:\finance-analyzer\portfolio\news_keywords.py:279:#      Ethereum/ETH, gold/XAU, silver/XAG, or the stock symbol)
Q:\finance-analyzer\portfolio\news_keywords.py:286:    "XAG": ["xag", "silver"],
Q:\finance-analyzer\portfolio\news_keywords.py:312:        return re.compile(r"\b" + re.escape(short) + r"\b", re.IGNORECASE)
Q:\finance-analyzer\portfolio\news_keywords.py:314:    return re.compile(r"\b(" + pattern + r")\b", re.IGNORECASE)
Q:\finance-analyzer\portfolio\news_keywords.py:317:# Memoize per-ticker patterns; tickers are a fixed small set so the cache
Q:\finance-analyzer\portfolio\news_keywords.py:318:# never grows large. None values are cached too (use sentinel-via-membership).
Q:\finance-analyzer\portfolio\news_keywords.py:352:    tokens = [t for t in re.findall(r"\b[A-Za-z]{2,}\b", stripped) if t.lower() not in {"the", "and", "for", "from"}]
Q:\finance-analyzer\portfolio\indicators.py:8:import portfolio.shared_state as _ss
Q:\finance-analyzer\portfolio\indicators.py:42:    # BUG-209: Guard against zero/negative prices (Binance maintenance, API glitches).
Q:\finance-analyzer\portfolio\indicators.py:156:    # BUG-169: Access regime cache under lock — 8 threads from ThreadPoolExecutor
Q:\finance-analyzer\portfolio\indicators.py:157:    # can call this concurrently. The check-then-clear pattern must be atomic.
Q:\finance-analyzer\portfolio\indicators.py:158:    cache_key = (
Q:\finance-analyzer\portfolio\indicators.py:168:        if _ss._run_cycle_id != _ss._regime_cache_cycle:
Q:\finance-analyzer\portfolio\indicators.py:169:            _ss._regime_cache = {}
Q:\finance-analyzer\portfolio\indicators.py:170:            _ss._regime_cache_cycle = _ss._run_cycle_id
Q:\finance-analyzer\portfolio\indicators.py:171:        if cache_key in _ss._regime_cache:
Q:\finance-analyzer\portfolio\indicators.py:172:            return _ss._regime_cache[cache_key]
Q:\finance-analyzer\portfolio\indicators.py:174:    # Compute outside lock (pure function, no shared state)
Q:\finance-analyzer\portfolio\indicators.py:206:        _ss._regime_cache[cache_key] = result
Q:\finance-analyzer\portfolio\forecast_signal.py:10:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\forecast_signal.py:16:from portfolio.file_utils import atomic_append_jsonl, load_json
Q:\finance-analyzer\portfolio\forecast_signal.py:33:_prophet_cache = {}  # ticker -> last fit time, to avoid refitting every minute
Q:\finance-analyzer\portfolio\forecast_signal.py:67:    """Override the Chronos model (e.g. from config). Resets cached pipeline."""
Q:\finance-analyzer\portfolio\forecast_signal.py:196:    timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="h")
Q:\finance-analyzer\portfolio\forecast_signal.py:272:        # Build dataframe with hourly timestamps (tz-naive, Prophet requirement)
Q:\finance-analyzer\portfolio\forecast_signal.py:273:        now = datetime.now(UTC).replace(tzinfo=None)
Q:\finance-analyzer\portfolio\forecast_signal.py:345:    ts = datetime.now(UTC).isoformat()
Q:\finance-analyzer\portfolio\forecast_signal.py:384:            atomic_append_jsonl(PREDICTIONS_FILE, entry)
Q:\finance-analyzer\portfolio\econ_dates.py:10:- FOMC: imported from fomc_dates.py (not duplicated here)
Q:\finance-analyzer\portfolio\econ_dates.py:15:from datetime import UTC, date, datetime
Q:\finance-analyzer\portfolio\econ_dates.py:17:from portfolio.fomc_dates import FOMC_ANNOUNCEMENT_DATES
Q:\finance-analyzer\portfolio\econ_dates.py:109:# Impact levels: "high" (FOMC, CPI, NFP) or "medium" (GDP)
Q:\finance-analyzer\portfolio\econ_dates.py:113:    "FOMC": {"crypto", "metals", "big_tech", "etf"},
Q:\finance-analyzer\portfolio\econ_dates.py:124:    for d in FOMC_ANNOUNCEMENT_DATES:
Q:\finance-analyzer\portfolio\econ_dates.py:125:        events.append({"date": d, "type": "FOMC", "impact": "high"})
Q:\finance-analyzer\portfolio\econ_dates.py:150:        ref_date = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\econ_dates.py:155:            evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:156:                                      tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:157:            # M5: Use datetime.now(UTC) so past-today events show hours_until=0.
Q:\finance-analyzer\portfolio\econ_dates.py:158:            now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:173:        ref_date = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\econ_dates.py:176:    now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:180:        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:181:                                  tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:195:                              ref_time: datetime | None = None) -> list[dict]:
Q:\finance-analyzer\portfolio\econ_dates.py:202:    after FOMC/CPI/NFP — those releases routinely whipsaw signals for
Q:\finance-analyzer\portfolio\econ_dates.py:219:    now = ref_time if ref_time is not None else datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:224:        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:225:                                  tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:238:    now: datetime | None = None,
Q:\finance-analyzer\portfolio\econ_dates.py:254:    ``datetime.now``. ``events_within_hours`` reads wall clock directly
Q:\finance-analyzer\portfolio\econ_dates.py:258:        now: Optional reference time (UTC). Defaults to ``datetime.now(UTC)``.
Q:\finance-analyzer\portfolio\econ_dates.py:262:            so only FOMC/CPI/NFP trigger; GDP (medium) does not.
Q:\finance-analyzer\portfolio\econ_dates.py:268:        now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:272:        evt_dt = datetime.combine(
Q:\finance-analyzer\portfolio\econ_dates.py:273:            evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC,
Q:\finance-analyzer\portfolio\session_calendar.py:1:"""Session calendar — instrument-specific trading hours and session state.
Q:\finance-analyzer\portfolio\session_calendar.py:3:Provides remaining-session time, session boundaries, and session mismatch
Q:\finance-analyzer\portfolio\session_calendar.py:7:    from portfolio.session_calendar import get_session_info
Q:\finance-analyzer\portfolio\session_calendar.py:8:    info = get_session_info("warrant", underlying="XAG-USD")
Q:\finance-analyzer\portfolio\session_calendar.py:9:    # info.remaining_minutes, info.session_end, info.is_extended, ...
Q:\finance-analyzer\portfolio\session_calendar.py:15:from datetime import UTC, datetime, time, timedelta
Q:\finance-analyzer\portfolio\session_calendar.py:22:    """Trading session state for an instrument.
Q:\finance-analyzer\portfolio\session_calendar.py:25:        session_end: Absolute datetime (UTC) of normal session close.
Q:\finance-analyzer\portfolio\session_calendar.py:26:        extended_end: Absolute datetime (UTC) of extended session close, if applicable.
Q:\finance-analyzer\portfolio\session_calendar.py:29:        is_extended: Whether we're in the extended (evening) session.
Q:\finance-analyzer\portfolio\session_calendar.py:33:    session_end: datetime
Q:\finance-analyzer\portfolio\session_calendar.py:34:    extended_end: datetime | None
Q:\finance-analyzer\portfolio\session_calendar.py:48:# We handle DST for EU sessions too.
Q:\finance-analyzer\portfolio\session_calendar.py:50:def _eu_dst(dt: datetime) -> bool:
Q:\finance-analyzer\portfolio\session_calendar.py:51:    """Check if datetime falls in EU Central European Summer Time (CEST).
Q:\finance-analyzer\portfolio\session_calendar.py:53:    EU DST: last Sunday of March 01:00 UTC → last Sunday of October 01:00 UTC.
Q:\finance-analyzer\portfolio\session_calendar.py:58:    mar31 = datetime(year, 3, 31, tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:60:    dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:63:    oct31 = datetime(year, 10, 31, tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:65:    dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:70:def _cet_offset(dt: datetime) -> int:
Q:\finance-analyzer\portfolio\session_calendar.py:75:def _cet_to_utc(hour: int, minute: int, dt: datetime) -> time:
Q:\finance-analyzer\portfolio\session_calendar.py:76:    """Convert CET time to UTC time object, adjusted for DST on given date."""
Q:\finance-analyzer\portfolio\session_calendar.py:82:def _make_session_end(now: datetime, cet_hour: int, cet_minute: int) -> datetime:
Q:\finance-analyzer\portfolio\session_calendar.py:83:    """Create a UTC datetime for today's session end from CET time."""
Q:\finance-analyzer\portfolio\session_calendar.py:87:    if end.tzinfo is None:
Q:\finance-analyzer\portfolio\session_calendar.py:88:        end = end.replace(tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:115:def get_session_info(instrument_type: str,
Q:\finance-analyzer\portfolio\session_calendar.py:117:                     now: datetime | None = None) -> SessionInfo:
Q:\finance-analyzer\portfolio\session_calendar.py:118:    """Get current session state for an instrument.
Q:\finance-analyzer\portfolio\session_calendar.py:122:        underlying: Underlying ticker for warrants (e.g., "XAG-USD").
Q:\finance-analyzer\portfolio\session_calendar.py:126:        SessionInfo with remaining time, phase, and session boundaries.
Q:\finance-analyzer\portfolio\session_calendar.py:129:        now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:130:    if now.tzinfo is None:
Q:\finance-analyzer\portfolio\session_calendar.py:131:        now = now.replace(tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:135:        # Use midnight as "session end" — effectively infinite session
Q:\finance-analyzer\portfolio\session_calendar.py:140:            session_end=end,
Q:\finance-analyzer\portfolio\session_calendar.py:149:    # US stocks: NYSE hours with DST
Q:\finance-analyzer\portfolio\session_calendar.py:155:        session_end = now.replace(hour=close_utc, minute=0, second=0, microsecond=0)
Q:\finance-analyzer\portfolio\session_calendar.py:157:                   now.replace(hour=open_utc, minute=30, second=0) <= now < session_end)
Q:\finance-analyzer\portfolio\session_calendar.py:159:        remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
Q:\finance-analyzer\portfolio\session_calendar.py:166:            session_end=session_end,
Q:\finance-analyzer\portfolio\session_calendar.py:180:    session_end = _make_session_end(now, ch, cm)
Q:\finance-analyzer\portfolio\session_calendar.py:181:    session_open = _make_session_end(now, oh, om)
Q:\finance-analyzer\portfolio\session_calendar.py:184:    is_open = is_weekday and session_open <= now < session_end
Q:\finance-analyzer\portfolio\session_calendar.py:186:    remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
Q:\finance-analyzer\portfolio\session_calendar.py:192:        # US stock underlying — check US market hours
Q:\finance-analyzer\portfolio\session_calendar.py:193:        us_info = get_session_info("stock_us", now=now)
Q:\finance-analyzer\portfolio\session_calendar.py:197:        session_end=session_end,
Q:\finance-analyzer\portfolio\session_calendar.py:207:def remaining_session_minutes(instrument_type: str = "warrant",
Q:\finance-analyzer\portfolio\session_calendar.py:208:                              now: datetime | None = None) -> float:
Q:\finance-analyzer\portfolio\session_calendar.py:209:    """Shortcut: get remaining minutes for an instrument's session."""
Q:\finance-analyzer\portfolio\session_calendar.py:210:    info = get_session_info(instrument_type, now=now)
Q:\finance-analyzer\portfolio\price_source.py:7:* Commodity underlyings (XAG, XAU, oil)            → Binance FAPI perps
Q:\finance-analyzer\portfolio\price_source.py:8:* Crypto (BTC, ETH)                                → Binance SPOT
Q:\finance-analyzer\portfolio\price_source.py:10:* Treasury yields / macro daily series             → FRED
Q:\finance-analyzer\portfolio\price_source.py:11:* FX (EUR/USD)                                     → Alpha Vantage FX_DAILY
Q:\finance-analyzer\portfolio\price_source.py:16:Binance FAPI XAGUSDT             7.7 s       445 ms    ← PRIMARY
Q:\finance-analyzer\portfolio\price_source.py:23:exists AND the yfinance fallback is not allowed for the requested
Q:\finance-analyzer\portfolio\price_source.py:37:# obvious. The router is alias-aware: 'XAG-USD', 'SI=F', 'XAGUSDT' all
Q:\finance-analyzer\portfolio\price_source.py:38:# resolve to the same Binance FAPI symbol.
Q:\finance-analyzer\portfolio\price_source.py:43:    "XAG-USD": "XAGUSDT", "SI=F": "XAGUSDT", "XAGUSDT": "XAGUSDT",
Q:\finance-analyzer\portfolio\price_source.py:65:    "HG=F",          # copper — no Binance perpetual
Q:\finance-analyzer\portfolio\price_source.py:66:    "DX-Y.NYB",      # DXY pseudo-ticker (Alpha Vantage FX is paid intraday)
Q:\finance-analyzer\portfolio\price_source.py:67:    "EURUSD=X",      # FX — Alpha Vantage paid intraday
Q:\finance-analyzer\portfolio\price_source.py:68:    "^TNX",          # 10y treasury yield (CBOE; FRED has daily DGS10 fallback)
Q:\finance-analyzer\portfolio\price_source.py:72:    # Oil futures — no free real-time alternative (Binance has no oil
Q:\finance-analyzer\portfolio\price_source.py:75:    # CL=F/BZ=F routed to Alpaca (futures unsupported) → fallback to
Q:\finance-analyzer\portfolio\price_source.py:77:    # been silently relying on the same fallback path.
Q:\finance-analyzer\portfolio\price_source.py:78:    "CL=F",          # WTI front-month
Q:\finance-analyzer\portfolio\price_source.py:90:# helpers + a new yfinance helper for the allowed fallback paths.
Q:\finance-analyzer\portfolio\price_source.py:95:    """Normalize yfinance-style intervals to Binance format.
Q:\finance-analyzer\portfolio\price_source.py:97:    yfinance uses '60m' / '90m' — Binance uses '1h' / no 90m equivalent.
Q:\finance-analyzer\portfolio\price_source.py:99:    through unchanged so a direct Binance interval (e.g. '1h') still works.
Q:\finance-analyzer\portfolio\price_source.py:103:        "90m": "1h",   # Binance has no 90m — closest down-sample is 1h
Q:\finance-analyzer\portfolio\price_source.py:142:            "This source lags 10-15 min; upstream caller should be on Binance/Alpaca/FRED if possible.",
Q:\finance-analyzer\portfolio\price_source.py:197:        ticker: Symbol in any recognized alias form (XAG-USD, SI=F,
Q:\finance-analyzer\portfolio\price_source.py:198:            XAGUSDT, BTC-USD, MSTR, ^VIX, etc.)
Q:\finance-analyzer\portfolio\price_source.py:199:        interval: Binance/Alpaca-style interval string
Q:\finance-analyzer\portfolio\price_source.py:200:            ("1m", "5m", "1h", "1d").
Q:\finance-analyzer\portfolio\price_source.py:203:            used only on the yfinance fallback path.
Q:\finance-analyzer\portfolio\price_source.py:221:        # yfinance fallback
Q:\finance-analyzer\portfolio\price_source.py:224:        # If a primary source (Binance/Alpaca) fails AND the ticker isn't
Q:\finance-analyzer\portfolio\price_source.py:239:                    f"({exc!r}), fallback=yfinance ({exc2!r})"
Q:\finance-analyzer\portfolio\price_source.py:251:    :func:`fetch_klines`'s OHLCV contract — e.g. yfinance's earnings
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve narrowed the likely risk areas to `alpha_vantage`, `http_retry`, `fx_rates`, `price_source`, `microstructure`, `onchain_data`, `session_calendar`, and the sentiment stack. Reading those in full now to confirm whether the suspicious patterns are real defects or just implementation choices.
2026-05-11T15:29:00.093462Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\http_retry.py | ForEach-Object {\""'$i	$_"; $i++ }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\http_retry.py | ForEach-Object {\""'$i	$_"; $i++ }' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\alpha_vantage.py | ForEach-Object {\""'$i	$_"; $i++ }' in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\http_retry.py | ForEach-Object {\""'$i	$_"; $i++ }'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\alpha_vantage.py | ForEach-Object {\""'$i	$_"; $i++ }'` rejected: blocked by policy
2026-05-11T15:29:00.098381Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\alpha_vantage.py | ForEach-Object {\""'$i	$_"; $i++ }'` rejected: blocked by policy
2026-05-11T15:29:00.101961Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\fx_rates.py | ForEach-Object {\""'$i	$_"; $i++ }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\fx_rates.py | ForEach-Object {\""'$i	$_"; $i++ }' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\price_source.py | ForEach-Object {\""'$i	$_"; $i++ }' in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\fx_rates.py | ForEach-Object {\""'$i	$_"; $i++ }'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\price_source.py | ForEach-Object {\""'$i	$_"; $i++ }'` rejected: blocked by policy
2026-05-11T15:29:00.102609Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$i=1; Get-Content Q:'"\\finance-analyzer\\portfolio\\price_source.py | ForEach-Object {\""'$i	$_"; $i++ }'` rejected: blocked by policy
codex
The shell policy is rejecting the `Get-Content` form, so I’m switching to `rg -n "^"` for full-file reads with stable line numbers. That keeps the review grounded in exact locations without fighting the shell wrapper.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\http_retry.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\alpha_vantage.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\fx_rates.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\price_source.py" in Q:\finance-analyzer
 succeeded in 769ms:
1:"""HTTP retry utility with exponential backoff for finance-analyzer API calls."""
2:
3:import logging
4:import random
5:import time
6:
7:import requests
8:
9:logger = logging.getLogger(__name__)
10:
11:DEFAULT_RETRIES = 3
12:DEFAULT_BACKOFF = 1.0  # seconds
13:DEFAULT_BACKOFF_FACTOR = 2.0
14:RETRYABLE_STATUS = {429, 500, 502, 503, 504}
15:
16:
17:def fetch_with_retry(url, method="GET", retries=DEFAULT_RETRIES,
18:                     backoff=DEFAULT_BACKOFF, backoff_factor=DEFAULT_BACKOFF_FACTOR,
19:                     timeout=30, headers=None, params=None, json_body=None,
20:                     session=None):
21:    """Make an HTTP request with exponential backoff retry.
22:
23:    Returns response object on success, None on all retries exhausted.
24:    """
25:    requester = session or requests
26:
27:    for attempt in range(retries + 1):
28:        try:
29:            if method.upper() == "GET":
30:                resp = requester.get(url, headers=headers, params=params, timeout=timeout)
31:            elif method.upper() == "POST":
32:                resp = requester.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
33:            else:
34:                resp = requester.request(method, url, headers=headers, params=params, timeout=timeout)
35:
36:            if resp.status_code not in RETRYABLE_STATUS:
37:                return resp
38:
39:            if attempt < retries:
40:                wait = backoff * (backoff_factor ** attempt)
41:                jitter = random.uniform(0, wait * 0.1)
42:                wait += jitter
43:                # H26/HTTP1: Honour Telegram retry_after parameter for 429 responses.
44:                if resp.status_code == 429:
45:                    try:
46:                        retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
47:                    except Exception:
48:                        retry_after = wait
49:                    wait = retry_after
50:                logger.warning("HTTP %s from %s, retry %d/%d in %.1fs",
51:                               resp.status_code, url, attempt + 1, retries, wait)
52:                time.sleep(wait)
53:            else:
54:                logger.error("HTTP %s from %s after %d retries",
55:                             resp.status_code, url, retries)
56:                return None
57:
58:        except (requests.ConnectionError, requests.Timeout) as e:
59:            if attempt < retries:
60:                wait = backoff * (backoff_factor ** attempt)
61:                jitter = random.uniform(0, wait * 0.1)
62:                wait += jitter
63:                logger.warning("%s from %s, retry %d/%d in %.1fs",
64:                               e.__class__.__name__, url, attempt + 1, retries, wait)
65:                time.sleep(wait)
66:            else:
67:                logger.error("Request failed after %d retries: %s - %s",
68:                             retries, url, e)
69:                return None
70:
71:    return None
72:
73:
74:def fetch_json(url, *, method="GET", retries=DEFAULT_RETRIES, default=None,
75:               label="", headers=None, params=None, timeout=30, session=None,
76:               **kwargs):
77:    """Fetch URL and return parsed JSON, or ``default`` on any failure.
78:
79:    Combines fetch_with_retry() + raise_for_status() + .json() into one call.
80:    """
81:    resp = fetch_with_retry(url, method=method, retries=retries, timeout=timeout,
82:                            headers=headers, params=params, session=session)
83:    if resp is None:
84:        if label:
85:            logger.warning("[%s] request returned None", label)
86:        return default
87:    try:
88:        resp.raise_for_status()
89:        return resp.json()
90:    except Exception as e:
91:        if label:
92:            logger.warning("[%s] HTTP %s or JSON parse error: %s", label,
93:                           getattr(resp, 'status_code', '?'), e)
94:        return default
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 774ms:
1:"""Alpha Vantage fundamentals — daily-refresh stock fundamentals cache.
2:
3:Fetches OVERVIEW endpoint data for stock tickers and caches persistently.
4:Free tier: 25 requests/day, 5 requests/minute.
5:
6:Not used for crypto or metals (no OVERVIEW data available).
7:"""
8:
9:import logging
10:import threading
11:from datetime import UTC, datetime
12:from pathlib import Path
13:
14:from portfolio.circuit_breaker import CircuitBreaker
15:from portfolio.file_utils import load_json
16:from portfolio.http_retry import fetch_with_retry
17:from portfolio.shared_state import _alpha_vantage_limiter
18:from portfolio.tickers import STOCK_SYMBOLS
19:
20:logger = logging.getLogger("portfolio.alpha_vantage")
21:
22:BASE_DIR = Path(__file__).resolve().parent.parent
23:DATA_DIR = BASE_DIR / "data"
24:CACHE_FILE = DATA_DIR / "fundamentals_cache.json"
25:
26:AV_BASE_URL = "https://www.alphavantage.co/query"
27:
28:# Module-level state
29:_cache = {}  # ticker -> normalized fundamentals dict
30:_cache_lock = threading.Lock()
31:_daily_budget_used = 0
32:_budget_reset_date = ""  # ISO date string for budget tracking
33:_cb = CircuitBreaker("alpha_vantage", failure_threshold=3, recovery_timeout=300)
34:
35:
36:def load_persistent_cache():
37:    """Load fundamentals cache from disk on startup."""
38:    global _cache
39:    data = load_json(CACHE_FILE)
40:    if data is None:
41:        logger.info("No fundamentals cache found at %s", CACHE_FILE)
42:        return
43:    if isinstance(data, dict):
44:        with _cache_lock:
45:            _cache = data
46:        logger.info("Loaded fundamentals for %d tickers from cache", len(data))
47:
48:
49:def _save_persistent_cache():
50:    """Write current cache to disk atomically."""
51:    try:
52:        from portfolio.file_utils import atomic_write_json
53:        with _cache_lock:
54:            snapshot = dict(_cache)
55:        atomic_write_json(CACHE_FILE, snapshot)
56:    except Exception as e:
57:        logger.warning("Failed to save fundamentals cache: %s", e)
58:
59:
60:def get_fundamentals(ticker):
61:    """Return cached fundamentals for a ticker, or None if not available."""
62:    with _cache_lock:
63:        return _cache.get(ticker)
64:
65:
66:def get_all_fundamentals():
67:    """Return all cached fundamentals (for reporting)."""
68:    with _cache_lock:
69:        return dict(_cache)
70:
71:
72:def _normalize_overview(raw):
73:    """Convert Alpha Vantage OVERVIEW response to clean format.
74:
75:    AV returns "None" strings for missing values and all values as strings.
76:    """
77:    def _float(val, default=None):
78:        if val is None or val == "None" or val == "-" or val == "":
79:            return default
80:        try:
81:            return float(val)
82:        except (ValueError, TypeError):
83:            return default
84:
85:    def _int(val, default=None):
86:        if val is None or val == "None" or val == "-" or val == "":
87:            return default
88:        try:
89:            return int(float(val))
90:        except (ValueError, TypeError):
91:            return default
92:
93:    # Check for error responses
94:    if "Error Message" in raw or "Note" in raw:
95:        return None
96:
97:    # Require at least Symbol to be present
98:    if not raw.get("Symbol"):
99:        return None
100:
101:    result = {
102:        "pe_ratio": _float(raw.get("PERatio")),
103:        "forward_pe": _float(raw.get("ForwardPE")),
104:        "peg_ratio": _float(raw.get("PEGRatio")),
105:        "eps": _float(raw.get("EPS")),
106:        "revenue_growth_yoy": _float(raw.get("QuarterlyRevenueGrowthYOY")),
107:        "earnings_growth_yoy": _float(raw.get("QuarterlyEarningsGrowthYOY")),
108:        "profit_margin": _float(raw.get("ProfitMargin")),
109:        "market_cap": _int(raw.get("MarketCapitalization")),
110:        "sector": raw.get("Sector") if raw.get("Sector") != "None" else None,
111:        "industry": raw.get("Industry") if raw.get("Industry") != "None" else None,
112:        "dividend_yield": _float(raw.get("DividendYield")),
113:        "analyst_target": _float(raw.get("AnalystTargetPrice")),
114:        "analyst_ratings": {
115:            "strong_buy": _int(raw.get("AnalystRatingStrongBuy"), 0),
116:            "buy": _int(raw.get("AnalystRatingBuy"), 0),
117:            "hold": _int(raw.get("AnalystRatingHold"), 0),
118:            "sell": _int(raw.get("AnalystRatingSell"), 0),
119:            "strong_sell": _int(raw.get("AnalystRatingStrongSell"), 0),
120:        },
121:        "beta": _float(raw.get("Beta")),
122:        "w52_high": _float(raw.get("52WeekHigh")),
123:        "w52_low": _float(raw.get("52WeekLow")),
124:        "_fetched_at": datetime.now(UTC).isoformat(),
125:    }
126:    return result
127:
128:
129:def _fetch_overview(ticker, api_key):
130:    """Fetch OVERVIEW data for a single ticker from Alpha Vantage."""
131:    params = {
132:        "function": "OVERVIEW",
133:        "symbol": ticker,
134:        "apikey": api_key,
135:    }
136:    resp = fetch_with_retry(
137:        AV_BASE_URL,
138:        params=params,
139:        timeout=15,
140:        retries=2,
141:    )
142:    if resp is None:
143:        return None
144:    try:
145:        data = resp.json()
146:    except (ValueError, AttributeError):
147:        return None
148:
149:    # Rate limit info check
150:    if isinstance(data, dict) and "Note" in data:
151:        logger.warning("Alpha Vantage rate limit hit: %s", data["Note"][:100])
152:        return None
153:
154:    return data
155:
156:
157:def _check_budget():
158:    """Check and reset daily budget counter. Returns current usage count.
159:
160:    BUG-108: Protected by _cache_lock for thread safety.
161:    """
162:    global _daily_budget_used, _budget_reset_date
163:    today = datetime.now(UTC).strftime("%Y-%m-%d")
164:    with _cache_lock:
165:        if _budget_reset_date != today:
166:            _daily_budget_used = 0
167:            _budget_reset_date = today
168:        return _daily_budget_used
169:
170:
171:def _is_stale(ticker, max_stale_days=5):
172:    """Check if cached data for ticker is older than max_stale_days."""
173:    with _cache_lock:
174:        entry = _cache.get(ticker)
175:    if not entry:
176:        return True
177:    fetched_at = entry.get("_fetched_at")
178:    if not fetched_at:
179:        return True
180:    try:
181:        fetched_time = datetime.fromisoformat(fetched_at)
182:        age_seconds = (datetime.now(UTC) - fetched_time).total_seconds()
183:        return age_seconds > max_stale_days * 86400
184:    except (ValueError, TypeError):
185:        return True
186:
187:
188:def _cache_age_hours(ticker):
189:    """Return age of cached data in hours, or None if not cached."""
190:    with _cache_lock:
191:        entry = _cache.get(ticker)
192:    if not entry:
193:        return None
194:    fetched_at = entry.get("_fetched_at")
195:    if not fetched_at:
196:        return None
197:    try:
198:        fetched_time = datetime.fromisoformat(fetched_at)
199:        return (datetime.now(UTC) - fetched_time).total_seconds() / 3600
200:    except (ValueError, TypeError):
201:        return None
202:
203:
204:def refresh_fundamentals_batch(config):
205:    """Batch-refresh fundamentals for all stock tickers.
206:
207:    Respects rate limits (5/min) and daily budget (25/day).
208:    Returns number of successfully refreshed tickers.
209:    """
210:    global _daily_budget_used
211:
212:    av_config = config.get("alpha_vantage", {})
213:    if not av_config.get("enabled", True):
214:        logger.info("Alpha Vantage disabled in config")
215:        return 0
216:
217:    api_key = av_config.get("api_key", "")
218:    if not api_key:
219:        logger.warning("Alpha Vantage API key not configured")
220:        return 0
221:
222:    daily_budget = av_config.get("daily_budget", 25)
223:    skip_tickers = set(av_config.get("skip_tickers", []))
224:    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)
225:
226:    # Check circuit breaker
227:    if not _cb.allow_request():
228:        logger.info("Alpha Vantage circuit breaker active (%s)", _cb.state.value)
229:        return 0
230:
231:    budget_used = _check_budget()
232:    if budget_used >= daily_budget:
233:        logger.info("Alpha Vantage daily budget exhausted (%d/%d)", budget_used, daily_budget)
234:        return 0
235:
236:    # Build refresh list: stock tickers not recently cached, not skipped
237:    tickers_to_refresh = []
238:    for ticker in sorted(STOCK_SYMBOLS):
239:        if ticker in skip_tickers:
240:            continue
241:        age = _cache_age_hours(ticker)
242:        if age is None or age > cache_ttl_hours:
243:            tickers_to_refresh.append(ticker)
244:
245:    if not tickers_to_refresh:
246:        logger.info("All fundamentals fresh, nothing to refresh")
247:        return 0
248:
249:    # Cap by remaining budget
250:    remaining_budget = daily_budget - budget_used
251:    tickers_to_refresh = tickers_to_refresh[:remaining_budget]
252:
253:    logger.info(
254:        "Refreshing fundamentals for %d tickers (budget: %d/%d used)",
255:        len(tickers_to_refresh), budget_used, daily_budget,
256:    )
257:
258:    success_count = 0
259:    for ticker in tickers_to_refresh:
260:        # Rate limit
261:        _alpha_vantage_limiter.wait()
262:
263:        try:
264:            raw = _fetch_overview(ticker, api_key)
265:            if raw is None:
266:                _cb.record_failure()
267:                if not _cb.allow_request():
268:                    break
269:                continue
270:
271:            normalized = _normalize_overview(raw)
272:            if normalized is None:
273:                logger.warning("Alpha Vantage: empty/error response for %s", ticker)
274:                _cb.record_failure()
275:                if not _cb.allow_request():
276:                    break
277:                continue
278:
279:            with _cache_lock:
280:                _cache[ticker] = normalized
281:                _daily_budget_used += 1  # BUG-108: increment under lock
282:            _cb.record_success()
283:            success_count += 1
284:            logger.info("Refreshed fundamentals for %s (PE=%.1f, sector=%s)",
285:                        ticker,
286:                        normalized.get("pe_ratio") or 0,
287:                        normalized.get("sector", "?"))
288:
289:        except Exception as e:
290:            logger.warning("Alpha Vantage fetch failed for %s: %s", ticker, e)
291:            _cb.record_failure()
292:            if not _cb.allow_request():
293:                break
294:
295:    if success_count > 0:
296:        _save_persistent_cache()
297:        logger.info("Fundamentals refresh complete: %d/%d succeeded", success_count, len(tickers_to_refresh))
298:
299:    return success_count
300:
301:
302:def should_batch_refresh(config):
303:    """Check if a batch refresh should run now.
304:
305:    Only runs when cache is >24h old. Designed to be called from the main loop.
306:    """
307:    av_config = config.get("alpha_vantage", {})
308:    if not av_config.get("enabled", True) or not av_config.get("api_key", ""):
309:        return False
310:
311:    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)
312:
313:    # Check if any stock ticker needs refresh
314:    for ticker in STOCK_SYMBOLS:
315:        if ticker in set(av_config.get("skip_tickers", [])):
316:            continue
317:        age = _cache_age_hours(ticker)
318:        if age is None or age > cache_ttl_hours:
319:            return True
320:
321:    return False
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 775ms:
1:"""Canonical real-time price-source router.
2:
3:yfinance is **LAST RESORT** only. Use it exclusively for CBOE volatility
4:indices (^VIX, ^VIX3M, ^OVX, ^GVZ) where no free live alternative
5:exists. For everything else this module routes to a real-time source:
6:
7:* Commodity underlyings (XAG, XAU, oil)            → Binance FAPI perps
8:* Crypto (BTC, ETH)                                → Binance SPOT
9:* US stocks / ETFs (MSTR, SPY, QQQ, USO, …)        → Alpaca IEX feed
10:* Treasury yields / macro daily series             → FRED
11:* FX (EUR/USD)                                     → Alpha Vantage FX_DAILY
12:
13:Benchmark reference (measured 2026-04-14 13:00 CET):
14:
15:====== source ===============  data age  api latency
16:Binance FAPI XAGUSDT             7.7 s       445 ms    ← PRIMARY
17:Avanza market-guide              tick      1,259 ms
18:yfinance SI=F                   655.4 s      389 ms    ← 85× stale
19:
20:Every consumer of historical / intraday OHLCV should call
21::func:`fetch_klines` rather than importing yfinance directly. The
22:function raises :class:`SourceUnavailableError` if no live alternative
23:exists AND the yfinance fallback is not allowed for the requested
24:ticker.
25:"""
26:from __future__ import annotations
27:
28:import logging
29:
30:import pandas as pd
31:
32:logger = logging.getLogger("portfolio.price_source")
33:
34:
35:# ---------------------------------------------------------------------------
36:# Ticker classification — keep this table explicit so future additions are
37:# obvious. The router is alias-aware: 'XAG-USD', 'SI=F', 'XAGUSDT' all
38:# resolve to the same Binance FAPI symbol.
39:# ---------------------------------------------------------------------------
40:
41:_BINANCE_FAPI = {
42:    # silver
43:    "XAG-USD": "XAGUSDT", "SI=F": "XAGUSDT", "XAGUSDT": "XAGUSDT",
44:    # gold
45:    "XAU-USD": "XAUUSDT", "GC=F": "XAUUSDT", "XAUUSDT": "XAUUSDT",
46:}
47:
48:_BINANCE_SPOT = {
49:    "BTC-USD": "BTCUSDT", "BTCUSDT": "BTCUSDT",
50:    "ETH-USD": "ETHUSDT", "ETHUSDT": "ETHUSDT",
51:}
52:
53:# CBOE-proprietary volatility indices. No free live alternative exists —
54:# these remain on yfinance by design. If you ever find a real-time feed,
55:# update this set AND the router.
56:_CBOE_VOL_INDICES = frozenset({
57:    "^VIX", "^VIX3M", "^OVX", "^GVZ", "^RVX", "^VXN",
58:})
59:
60:# Tickers for which yfinance is the only available free data source.
61:# Calls to these emit DEBUG (not WARNING) so legitimate use doesn't
62:# pollute the log; calls for non-allowed tickers emit WARNING so we
63:# can quantify residual leakage.
64:_YFINANCE_LAST_RESORT = frozenset({
65:    "HG=F",          # copper — no Binance perpetual
66:    "DX-Y.NYB",      # DXY pseudo-ticker (Alpha Vantage FX is paid intraday)
67:    "EURUSD=X",      # FX — Alpha Vantage paid intraday
68:    "^TNX",          # 10y treasury yield (CBOE; FRED has daily DGS10 fallback)
69:    "^TYX",          # 30y treasury yield
70:    "2YY=F",         # 2y treasury yield futures pseudo-ticker
71:    "^FVX",          # 5y treasury yield
72:    # Oil futures — no free real-time alternative (Binance has no oil
73:    # perpetual; Alpaca has the USO ETF but not the underlying futures).
74:    # 2026-05-01: added when oil_loop went live in DRY_RUN — previously
75:    # CL=F/BZ=F routed to Alpaca (futures unsupported) → fallback to
76:    # yfinance with WARNING noise every 60s cycle. oil_precompute had
77:    # been silently relying on the same fallback path.
78:    "CL=F",          # WTI front-month
79:    "BZ=F",          # Brent front-month
80:    "RB=F",          # RBOB gasoline (used for crack-spread context)
81:}) | _CBOE_VOL_INDICES
82:
83:
84:class SourceUnavailableError(RuntimeError):
85:    """Raised when no price source can serve the requested ticker."""
86:
87:
88:# ---------------------------------------------------------------------------
89:# Per-source fetchers — thin wrappers over the existing data_collector
90:# helpers + a new yfinance helper for the allowed fallback paths.
91:# ---------------------------------------------------------------------------
92:
93:
94:def _binance_interval(interval: str) -> str:
95:    """Normalize yfinance-style intervals to Binance format.
96:
97:    yfinance uses '60m' / '90m' — Binance uses '1h' / no 90m equivalent.
98:    This function translates common cases. Unrecognized intervals pass
99:    through unchanged so a direct Binance interval (e.g. '1h') still works.
100:    """
101:    mapping = {
102:        "60m": "1h",
103:        "90m": "1h",   # Binance has no 90m — closest down-sample is 1h
104:        "120m": "2h",
105:    }
106:    return mapping.get(interval, interval)
107:
108:
109:def _fetch_binance_fapi(symbol: str, interval: str, limit: int) -> pd.DataFrame:
110:    from portfolio.data_collector import binance_fapi_klines
111:
112:    return binance_fapi_klines(symbol, interval=_binance_interval(interval), limit=limit)
113:
114:
115:def _fetch_binance_spot(symbol: str, interval: str, limit: int) -> pd.DataFrame:
116:    from portfolio.data_collector import binance_klines
117:
118:    return binance_klines(symbol, interval=_binance_interval(interval), limit=limit)
119:
120:
121:def _fetch_alpaca(ticker: str, interval: str, limit: int) -> pd.DataFrame:
122:    from portfolio.data_collector import alpaca_klines
123:
124:    return alpaca_klines(ticker, interval=interval, limit=limit)
125:
126:
127:def _fetch_yfinance(
128:    ticker: str, interval: str, period: str | None = None, limit: int | None = None,
129:) -> pd.DataFrame:
130:    """Yfinance fetcher. Allowed-list tickers emit DEBUG; everything else
131:    emits WARNING so we can quantify residual leakage."""
132:    import yfinance as yf
133:
134:    if ticker in _YFINANCE_LAST_RESORT:
135:        logger.debug(
136:            "price_source: yfinance for %s (interval=%s, period=%s) — allowed (no live alt)",
137:            ticker, interval, period,
138:        )
139:    else:
140:        logger.warning(
141:            "price_source: falling back to yfinance for %s (interval=%s, period=%s). "
142:            "This source lags 10-15 min; upstream caller should be on Binance/Alpaca/FRED if possible.",
143:            ticker, interval, period,
144:        )
145:    p = period or "5d"
146:    df = yf.download(
147:        ticker, period=p, interval=interval,
148:        progress=False, auto_adjust=True,
149:    )
150:    if isinstance(df.columns, pd.MultiIndex):
151:        df.columns = df.columns.droplevel(1)
152:    if df.empty:
153:        return df
154:    # Normalize column casing (yfinance uses capitalized; our downstream
155:    # code expects lowercase after the alpaca/binance path).
156:    df = df.rename(columns={
157:        "Open": "open", "High": "high", "Low": "low",
158:        "Close": "close", "Volume": "volume",
159:    })
160:    return df.tail(limit) if limit else df
161:
162:
163:# ---------------------------------------------------------------------------
164:# Public API
165:# ---------------------------------------------------------------------------
166:
167:
168:def resolve_source(ticker: str) -> str:
169:    """Return the routing decision for ``ticker``: one of
170:    {"binance_fapi", "binance_spot", "alpaca", "yfinance"}.
171:
172:    Exported for diagnostics and tests.
173:    """
174:    if ticker in _BINANCE_FAPI:
175:        return "binance_fapi"
176:    if ticker in _BINANCE_SPOT:
177:        return "binance_spot"
178:    if ticker in _YFINANCE_LAST_RESORT:
179:        return "yfinance"
180:    # Default assumption: a bare uppercase symbol is a US stock/ETF → Alpaca.
181:    # Anything starting with '^' that isn't a CBOE vol index is an index we
182:    # don't have a mapping for — route to yfinance with warning.
183:    if ticker.startswith("^"):
184:        return "yfinance"
185:    return "alpaca"
186:
187:
188:def fetch_klines(
189:    ticker: str,
190:    interval: str = "1d",
191:    limit: int = 100,
192:    period: str | None = None,
193:) -> pd.DataFrame:
194:    """Fetch OHLCV bars for ``ticker`` from the freshest available source.
195:
196:    Args:
197:        ticker: Symbol in any recognized alias form (XAG-USD, SI=F,
198:            XAGUSDT, BTC-USD, MSTR, ^VIX, etc.)
199:        interval: Binance/Alpaca-style interval string
200:            ("1m", "5m", "1h", "1d").
201:        limit: Maximum rows to return.
202:        period: Optional yfinance-style period string ("5d", "3mo") —
203:            used only on the yfinance fallback path.
204:
205:    Returns:
206:        ``pandas.DataFrame`` with lowercase OHLCV columns
207:        (``open, high, low, close, volume``).
208:
209:    Raises:
210:        SourceUnavailableError: If every applicable source failed.
211:    """
212:    source = resolve_source(ticker)
213:
214:    try:
215:        if source == "binance_fapi":
216:            return _fetch_binance_fapi(_BINANCE_FAPI[ticker], interval, limit)
217:        if source == "binance_spot":
218:            return _fetch_binance_spot(_BINANCE_SPOT[ticker], interval, limit)
219:        if source == "alpaca":
220:            return _fetch_alpaca(ticker, interval, limit)
221:        # yfinance fallback
222:        return _fetch_yfinance(ticker, interval, period=period, limit=limit)
223:    except Exception as exc:
224:        # If a primary source (Binance/Alpaca) fails AND the ticker isn't
225:        # explicitly yfinance-only, fall through to yfinance as emergency
226:        # backup so the loop doesn't lose data entirely. Emit an error log
227:        # so we can see how often the primary is flaking.
228:        if source in ("binance_fapi", "binance_spot", "alpaca"):
229:            logger.error(
230:                "price_source: primary source %s FAILED for %s (%r). "
231:                "Falling back to yfinance. Investigate the primary outage.",
232:                source, ticker, exc,
233:            )
234:            try:
235:                return _fetch_yfinance(ticker, interval, period=period, limit=limit)
236:            except Exception as exc2:
237:                raise SourceUnavailableError(
238:                    f"All sources failed for {ticker}: primary={source} "
239:                    f"({exc!r}), fallback=yfinance ({exc2!r})"
240:                ) from exc2
241:        raise SourceUnavailableError(
242:            f"yfinance failed for {ticker}: {exc!r}"
243:        ) from exc
244:
245:
246:def is_yfinance_allowed(ticker: str) -> bool:
247:    """True if yfinance is an approved primary source for this ticker
248:    (CBOE vol indices + the explicit last-resort list).
249:
250:    Callers that want a direct yfinance path (for reasons beyond
251:    :func:`fetch_klines`'s OHLCV contract — e.g. yfinance's earnings
252:    calendar or fundamentals APIs) should check this first and log a
253:    WARNING if they still go to yfinance for a ticker outside this list.
254:    """
255:    return ticker in _YFINANCE_LAST_RESORT
256:
257:
258:__all__ = [
259:    "fetch_klines",
260:    "resolve_source",
261:    "is_yfinance_allowed",
262:    "SourceUnavailableError",
263:]
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 781ms:
1:"""USD/SEK exchange rate fetching with caching and staleness alerts.
2:
3:NOTE: Not migrated to shared_state._cached() because this module has custom
4:staleness behaviour: Telegram alerting when stale, hardcoded 10.50 fallback,
5:and ConnectionError semantics. These would be lost with _cached().
6:"""
7:
8:import logging
9:import threading
10:import time
11:
12:from portfolio.api_utils import load_config as _load_config
13:from portfolio.http_retry import fetch_with_retry
14:
15:logger = logging.getLogger("portfolio.fx_rates")
16:
17:# Shared FX constants — imported by risk_management.py and monte_carlo_risk.py
18:FX_RATE_FALLBACK = 10.50
19:FX_RATE_MIN = 7.0
20:FX_RATE_MAX = 15.0
21:
22:# BUG-215: Thread-safe FX cache. Accessed from 8-worker ThreadPoolExecutor.
23:_fx_lock = threading.Lock()
24:_fx_cache = {"rate": None, "time": 0}
25:_FX_STALE_THRESHOLD = 7200  # 2 hours — warn if FX rate hasn't been refreshed
26:
27:
28:def fetch_usd_sek():
29:    now = time.time()
30:    with _fx_lock:
31:        cached_rate = _fx_cache["rate"]
32:        cached_time = _fx_cache["time"]
33:    if cached_rate and now - cached_time < 900:  # 15 min cache
34:        return cached_rate
35:    try:
36:        r = fetch_with_retry(
37:            "https://api.frankfurter.app/latest",
38:            params={"from": "USD", "to": "SEK"},
39:            timeout=10,
40:        )
41:        if r is None:
42:            raise ConnectionError("FX rate request failed after retries")
43:        r.raise_for_status()
44:        rate = float(r.json()["rates"]["SEK"])
45:        # BUG-117: Sanity check — SEK/USD should be in 7-15 range historically.
46:        # If outside this range, the API may be returning bad data.
47:        if not (FX_RATE_MIN <= rate <= FX_RATE_MAX):
48:            logger.error("FX rate %.4f SEK/USD outside sane bounds (7-15) — ignoring", rate)
49:        else:
50:            with _fx_lock:
51:                _fx_cache["rate"] = rate
52:                _fx_cache["time"] = now
53:            return rate
54:    except Exception as e:
55:        logger.warning("FX rate fetch failed: %s", e)
56:    # Fallback to stale cached value
57:    with _fx_lock:
58:        cached_rate = _fx_cache["rate"]
59:        cached_time = _fx_cache["time"]
60:    if cached_rate:
61:        age_secs = now - cached_time
62:        if age_secs > _FX_STALE_THRESHOLD:
63:            logger.warning("Using stale FX rate (%.1fh old)", age_secs / 3600)
64:            _fx_alert_telegram(age_secs)
65:        return cached_rate
66:    # Last resort: hardcoded fallback
67:    # BUG-117: Use ERROR level — hardcoded rate may be severely stale.
68:    # Portfolio valuations using this rate could be off by 10-15% if SEK has moved.
69:    logger.error("Using hardcoded FX fallback rate %.2f SEK — no cached or live rate available", FX_RATE_FALLBACK)
70:    _fx_alert_telegram(None)
71:    return FX_RATE_FALLBACK
72:
73:
74:def _fx_alert_telegram(age_secs):
75:    """Send a one-shot Telegram alert about FX rate issues. Fires at most once per 4h."""
76:    now = time.time()
77:    with _fx_lock:
78:        last_alert = _fx_cache.get("_last_fx_alert", 0)
79:        if now - last_alert < 14400:  # 4h cooldown between alerts
80:            return
81:        _fx_cache["_last_fx_alert"] = now
82:    try:
83:        config = _load_config()
84:        if age_secs is not None:
85:            msg = f"_FX WARNING: USD/SEK rate is {age_secs / 3600:.1f}h stale. API may be down._"
86:        else:
87:            msg = f"_FX WARNING: Using hardcoded fallback rate {FX_RATE_FALLBACK} SEK. No live or cached rate available._"
88:        from portfolio.message_store import send_or_store
89:        send_or_store(msg, config, category="error")
90:    except Exception as e:
91:        logger.debug("FX Telegram alert failed: %s", e)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"requests\\.|Session\\(|httpx|urllib\\.request|urlopen|aiohttp|yf\\.download|download\\(|fetch_with_retry|fetch_json\" Q:\\finance-analyzer\\portfolio\\data_collector.py Q:\\finance-analyzer\\portfolio\\fear_greed.py Q:\\finance-analyzer\\portfolio\\sentiment.py Q:\\finance-analyzer\\portfolio\\bert_sentiment.py Q:\\finance-analyzer\\portfolio\\alpha_vantage.py Q:\\finance-analyzer\\portfolio\\futures_data.py Q:\\finance-analyzer\\portfolio\\funding_rate.py Q:\\finance-analyzer\\portfolio\\fx_rates.py Q:\\finance-analyzer\\portfolio\\onchain_data.py Q:\\finance-analyzer\\portfolio\\news_keywords.py Q:\\finance-analyzer\\portfolio\\social_sentiment.py Q:\\finance-analyzer\\portfolio\\crypto_macro_data.py Q:\\finance-analyzer\\portfolio\\crypto_scheduler.py Q:\\finance-analyzer\\portfolio\\earnings_calendar.py Q:\\finance-analyzer\\portfolio\\econ_dates.py Q:\\finance-analyzer\\portfolio\\fomc_dates.py Q:\\finance-analyzer\\portfolio\\seasonality.py Q:\\finance-analyzer\\portfolio\\seasonality_updater.py Q:\\finance-analyzer\\portfolio\\session_calendar.py Q:\\finance-analyzer\\portfolio\\price_source.py Q:\\finance-analyzer\\portfolio\\http_retry.py Q:\\finance-analyzer\\portfolio\\api_utils.py Q:\\finance-analyzer\\portfolio\\data_refresh.py Q:\\finance-analyzer\\portfolio\\forecast_signal.py Q:\\finance-analyzer\\portfolio\\indicators.py Q:\\finance-analyzer\\portfolio\\metals_orderbook.py Q:\\finance-analyzer\\portfolio\\microstructure.py Q:\\finance-analyzer\\portfolio\\microstructure_state.py Q:\\finance-analyzer\\portfolio\\metals_cross_assets.py Q:\\finance-analyzer\\portfolio\\tickers.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_daily_budget_used|_budget_reset_date|cache_ttl|_fx_cache|_FX_STALE_THRESHOLD|Retry-After|retry_after|raise_for_status|fromisoformat|tzinfo=None|replace\\(tzinfo=None\\)|ZoneInfo|DST|timezone|naive|aware|market hours|session_end|session_open|funding|shorts|longs|pay|MVRV|SOPR|ofi|vpin|imbalance|rolling|state|atomic_write_json|load_json|XAG|XAU|SILVER|normalize|alias|keyword|regex|compile|escape\" Q:\\finance-analyzer\\portfolio\\data_collector.py Q:\\finance-analyzer\\portfolio\\fear_greed.py Q:\\finance-analyzer\\portfolio\\sentiment.py Q:\\finance-analyzer\\portfolio\\bert_sentiment.py Q:\\finance-analyzer\\portfolio\\alpha_vantage.py Q:\\finance-analyzer\\portfolio\\futures_data.py Q:\\finance-analyzer\\portfolio\\funding_rate.py Q:\\finance-analyzer\\portfolio\\fx_rates.py Q:\\finance-analyzer\\portfolio\\onchain_data.py Q:\\finance-analyzer\\portfolio\\news_keywords.py Q:\\finance-analyzer\\portfolio\\social_sentiment.py Q:\\finance-analyzer\\portfolio\\crypto_macro_data.py Q:\\finance-analyzer\\portfolio\\crypto_scheduler.py Q:\\finance-analyzer\\portfolio\\earnings_calendar.py Q:\\finance-analyzer\\portfolio\\econ_dates.py Q:\\finance-analyzer\\portfolio\\fomc_dates.py Q:\\finance-analyzer\\portfolio\\seasonality.py Q:\\finance-analyzer\\portfolio\\seasonality_updater.py Q:\\finance-analyzer\\portfolio\\session_calendar.py Q:\\finance-analyzer\\portfolio\\price_source.py Q:\\finance-analyzer\\portfolio\\http_retry.py Q:\\finance-analyzer\\portfolio\\api_utils.py Q:\\finance-analyzer\\portfolio\\data_refresh.py Q:\\finance-analyzer\\portfolio\\forecast_signal.py Q:\\finance-analyzer\\portfolio\\indicators.py Q:\\finance-analyzer\\portfolio\\metals_orderbook.py Q:\\finance-analyzer\\portfolio\\microstructure.py Q:\\finance-analyzer\\portfolio\\microstructure_state.py Q:\\finance-analyzer\\portfolio\\metals_cross_assets.py Q:\\finance-analyzer\\portfolio\\tickers.py" in Q:\finance-analyzer
 succeeded in 1000ms:
Q:\finance-analyzer\portfolio\futures_data.py:4:Uses existing infrastructure: fetch_json, _cached, _binance_limiter.
Q:\finance-analyzer\portfolio\futures_data.py:11:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\futures_data.py:27:def _fetch_json(url, params=None, timeout=10):
Q:\finance-analyzer\portfolio\futures_data.py:30:    return fetch_json(url, params=params, timeout=timeout, label="binance_fapi")
Q:\finance-analyzer\portfolio\futures_data.py:43:        data = _fetch_json(
Q:\finance-analyzer\portfolio\futures_data.py:68:        data = _fetch_json(
Q:\finance-analyzer\portfolio\futures_data.py:96:        data = _fetch_json(
Q:\finance-analyzer\portfolio\futures_data.py:125:        data = _fetch_json(
Q:\finance-analyzer\portfolio\futures_data.py:154:        data = _fetch_json(
Q:\finance-analyzer\portfolio\futures_data.py:183:        data = _fetch_json(
Q:\finance-analyzer\portfolio\alpha_vantage.py:16:from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\alpha_vantage.py:136:    resp = fetch_with_retry(
Q:\finance-analyzer\portfolio\seasonality_updater.py:56:    from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\seasonality_updater.py:64:    data = fetch_json(
Q:\finance-analyzer\portfolio\sentiment.py:27:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\sentiment.py:83:    data = fetch_json(
Q:\finance-analyzer\portfolio\sentiment.py:162:    data = fetch_json(
Q:\finance-analyzer\portfolio\fear_greed.py:6:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\fear_greed.py:96:    body = fetch_json("https://api.alternative.me/fng/", timeout=10,
Q:\finance-analyzer\portfolio\metals_cross_assets.py:37:def _yf_download(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
Q:\finance-analyzer\portfolio\metals_cross_assets.py:96:        df = _yf_download("HG=F", period="3mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:116:        df = _yf_download("^GVZ", period="3mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:139:        gold_df = _yf_download("GC=F", period="6mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:140:        silver_df = _yf_download("SI=F", period="6mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:170:        df = _yf_download("CL=F", period="3mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:188:        df = _yf_download("SPY", period="1mo", interval="1d")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:225:        df = _yf_download("HG=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:243:        gold_df = _yf_download("GC=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:244:        silver_df = _yf_download("SI=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:267:        df = _yf_download("CL=F", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:285:        df = _yf_download("SPY", period="5d", interval="60m")
Q:\finance-analyzer\portfolio\metals_orderbook.py:16:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\metals_orderbook.py:35:    return fetch_json(url, params=params, timeout=timeout, label="metals_orderbook")
Q:\finance-analyzer\portfolio\data_collector.py:13:from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\data_collector.py:80:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\data_collector.py:126:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\data_collector.py:222:    df = yf.download(
Q:\finance-analyzer\portfolio\crypto_macro_data.py:22:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\crypto_macro_data.py:54:    data = fetch_json(url, params=params, timeout=20, retries=2,
Q:\finance-analyzer\portfolio\earnings_calendar.py:38:        from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\earnings_calendar.py:53:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\data_refresh.py:7:from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\data_refresh.py:30:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\onchain_data.py:7:Budget: 6 metrics x 2 refreshes/day = 12 requests.
Q:\finance-analyzer\portfolio\onchain_data.py:23:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\onchain_data.py:132:    return fetch_json(url, headers=headers, params=params, timeout=15,
Q:\finance-analyzer\portfolio\price_source.py:146:    df = yf.download(
Q:\finance-analyzer\portfolio\http_retry.py:17:def fetch_with_retry(url, method="GET", retries=DEFAULT_RETRIES,
Q:\finance-analyzer\portfolio\http_retry.py:58:        except (requests.ConnectionError, requests.Timeout) as e:
Q:\finance-analyzer\portfolio\http_retry.py:74:def fetch_json(url, *, method="GET", retries=DEFAULT_RETRIES, default=None,
Q:\finance-analyzer\portfolio\http_retry.py:79:    Combines fetch_with_retry() + raise_for_status() + .json() into one call.
Q:\finance-analyzer\portfolio\http_retry.py:81:    resp = fetch_with_retry(url, method=method, retries=retries, timeout=timeout,
Q:\finance-analyzer\portfolio\social_sentiment.py:8:import urllib.request
Q:\finance-analyzer\portfolio\social_sentiment.py:32:    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
Q:\finance-analyzer\portfolio\social_sentiment.py:33:    with urllib.request.urlopen(req, timeout=10) as resp:
Q:\finance-analyzer\portfolio\social_sentiment.py:65:    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
Q:\finance-analyzer\portfolio\social_sentiment.py:66:    with urllib.request.urlopen(req, timeout=10) as resp:
Q:\finance-analyzer\portfolio\fx_rates.py:13:from portfolio.http_retry import fetch_with_retry
Q:\finance-analyzer\portfolio\fx_rates.py:36:        r = fetch_with_retry(
Q:\finance-analyzer\portfolio\funding_rate.py:2:from portfolio.http_retry import fetch_json
Q:\finance-analyzer\portfolio\funding_rate.py:14:    data = fetch_json(
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 987ms:
Q:\finance-analyzer\portfolio\futures_data.py:1:"""Binance FAPI futures data — open interest, long/short ratios, funding history.
Q:\finance-analyzer\portfolio\futures_data.py:12:from portfolio.shared_state import _binance_limiter, _cached
Q:\finance-analyzer\portfolio\futures_data.py:173:def get_funding_rate_history(ticker, limit=100):
Q:\finance-analyzer\portfolio\futures_data.py:174:    """Historical funding rates.
Q:\finance-analyzer\portfolio\futures_data.py:176:    Returns: list of {fundingRate, fundingTime, symbol} or None.
Q:\finance-analyzer\portfolio\futures_data.py:184:            f"{BINANCE_FAPI_BASE}/fundingRate",
Q:\finance-analyzer\portfolio\futures_data.py:191:                "fundingRate": float(d["fundingRate"]),
Q:\finance-analyzer\portfolio\futures_data.py:192:                "fundingTime": d["fundingTime"],
Q:\finance-analyzer\portfolio\futures_data.py:198:    return _cached(f"futures_funding_hist_{ticker}", _FUNDING_TTL, _fetch)
Q:\finance-analyzer\portfolio\futures_data.py:205:             top_position_ratio, top_account_ratio, funding_history.
Q:\finance-analyzer\portfolio\futures_data.py:216:        "funding_history": get_funding_rate_history(ticker),
Q:\finance-analyzer\portfolio\seasonality_updater.py:1:"""Compute and persist intraday seasonality profiles for metals.
Q:\finance-analyzer\portfolio\seasonality_updater.py:4:return/volatility profiles. Called periodically to keep profiles fresh.
Q:\finance-analyzer\portfolio\seasonality_updater.py:11:from portfolio.seasonality import compute_hourly_profile, load_profiles, save_profiles
Q:\finance-analyzer\portfolio\seasonality_updater.py:16:def update_seasonality_profiles(tickers: list[str] | None = None) -> dict:
Q:\finance-analyzer\portfolio\seasonality_updater.py:17:    """Fetch klines and recompute seasonality profiles for metals tickers.
Q:\finance-analyzer\portfolio\seasonality_updater.py:20:        tickers: List of tickers to update. Defaults to XAG-USD, XAU-USD.
Q:\finance-analyzer\portfolio\seasonality_updater.py:23:        Dict of ticker -> profile (or None if failed).
Q:\finance-analyzer\portfolio\seasonality_updater.py:26:        tickers = ["XAG-USD", "XAU-USD"]
Q:\finance-analyzer\portfolio\seasonality_updater.py:28:    profiles = load_profiles()
Q:\finance-analyzer\portfolio\seasonality_updater.py:36:            profile = compute_hourly_profile(df)
Q:\finance-analyzer\portfolio\seasonality_updater.py:37:            if profile is not None:
Q:\finance-analyzer\portfolio\seasonality_updater.py:38:                profiles[ticker] = profile
Q:\finance-analyzer\portfolio\seasonality_updater.py:39:                logger.info("Updated seasonality profile for %s (%d hours)",
Q:\finance-analyzer\portfolio\seasonality_updater.py:40:                           ticker, len(profile))
Q:\finance-analyzer\portfolio\seasonality_updater.py:42:                logger.warning("Insufficient data for %s profile", ticker)
Q:\finance-analyzer\portfolio\seasonality_updater.py:46:    if profiles:
Q:\finance-analyzer\portfolio\seasonality_updater.py:47:        save_profiles(profiles)
Q:\finance-analyzer\portfolio\seasonality_updater.py:48:    return profiles
Q:\finance-analyzer\portfolio\seasonality_updater.py:57:    from portfolio.shared_state import _binance_limiter
Q:\finance-analyzer\portfolio\fear_greed.py:5:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\fear_greed.py:26:        data = load_json(_STREAK_FILE)
Q:\finance-analyzer\portfolio\fear_greed.py:37:    Called after each successful F&G fetch. Persists streak state to disk
Q:\finance-analyzer\portfolio\fear_greed.py:43:    data = load_json(_STREAK_FILE, default={}) or {}
Q:\finance-analyzer\portfolio\fear_greed.py:77:        atomic_write_json(_STREAK_FILE, data)
Q:\finance-analyzer\portfolio\fear_greed.py:130:    from portfolio.shared_state import yfinance_lock
Q:\finance-analyzer\portfolio\bert_sentiment.py:191:    # Lazy import so the main loop doesn't pay the torch import cost at
Q:\finance-analyzer\portfolio\bert_sentiment.py:236:    # which is rare and load-time-only — never during steady-state
Q:\finance-analyzer\portfolio\alpha_vantage.py:15:from portfolio.file_utils import load_json
Q:\finance-analyzer\portfolio\alpha_vantage.py:17:from portfolio.shared_state import _alpha_vantage_limiter
Q:\finance-analyzer\portfolio\alpha_vantage.py:28:# Module-level state
Q:\finance-analyzer\portfolio\alpha_vantage.py:29:_cache = {}  # ticker -> normalized fundamentals dict
Q:\finance-analyzer\portfolio\alpha_vantage.py:31:_daily_budget_used = 0
Q:\finance-analyzer\portfolio\alpha_vantage.py:32:_budget_reset_date = ""  # ISO date string for budget tracking
Q:\finance-analyzer\portfolio\alpha_vantage.py:39:    data = load_json(CACHE_FILE)
Q:\finance-analyzer\portfolio\alpha_vantage.py:52:        from portfolio.file_utils import atomic_write_json
Q:\finance-analyzer\portfolio\alpha_vantage.py:55:        atomic_write_json(CACHE_FILE, snapshot)
Q:\finance-analyzer\portfolio\alpha_vantage.py:72:def _normalize_overview(raw):
Q:\finance-analyzer\portfolio\alpha_vantage.py:108:        "profit_margin": _float(raw.get("ProfitMargin")),
Q:\finance-analyzer\portfolio\alpha_vantage.py:162:    global _daily_budget_used, _budget_reset_date
Q:\finance-analyzer\portfolio\alpha_vantage.py:165:        if _budget_reset_date != today:
Q:\finance-analyzer\portfolio\alpha_vantage.py:166:            _daily_budget_used = 0
Q:\finance-analyzer\portfolio\alpha_vantage.py:167:            _budget_reset_date = today
Q:\finance-analyzer\portfolio\alpha_vantage.py:168:        return _daily_budget_used
Q:\finance-analyzer\portfolio\alpha_vantage.py:181:        fetched_time = datetime.fromisoformat(fetched_at)
Q:\finance-analyzer\portfolio\alpha_vantage.py:198:        fetched_time = datetime.fromisoformat(fetched_at)
Q:\finance-analyzer\portfolio\alpha_vantage.py:210:    global _daily_budget_used
Q:\finance-analyzer\portfolio\alpha_vantage.py:224:    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)
Q:\finance-analyzer\portfolio\alpha_vantage.py:228:        logger.info("Alpha Vantage circuit breaker active (%s)", _cb.state.value)
Q:\finance-analyzer\portfolio\alpha_vantage.py:242:        if age is None or age > cache_ttl_hours:
Q:\finance-analyzer\portfolio\alpha_vantage.py:271:            normalized = _normalize_overview(raw)
Q:\finance-analyzer\portfolio\alpha_vantage.py:272:            if normalized is None:
Q:\finance-analyzer\portfolio\alpha_vantage.py:280:                _cache[ticker] = normalized
Q:\finance-analyzer\portfolio\alpha_vantage.py:281:                _daily_budget_used += 1  # BUG-108: increment under lock
Q:\finance-analyzer\portfolio\alpha_vantage.py:286:                        normalized.get("pe_ratio") or 0,
Q:\finance-analyzer\portfolio\alpha_vantage.py:287:                        normalized.get("sector", "?"))
Q:\finance-analyzer\portfolio\alpha_vantage.py:311:    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)
Q:\finance-analyzer\portfolio\alpha_vantage.py:318:        if age is None or age > cache_ttl_hours:
Q:\finance-analyzer\portfolio\sentiment.py:50:    "XAU": "GOLD",
Q:\finance-analyzer\portfolio\sentiment.py:51:    "XAG": "SILVER",
Q:\finance-analyzer\portfolio\sentiment.py:190:    from portfolio.shared_state import newsapi_track_call
Q:\finance-analyzer\portfolio\sentiment.py:199:    from portfolio.shared_state import (
Q:\finance-analyzer\portfolio\sentiment.py:302:# data/sentiment_ab_log.jsonl alongside the primary model's vote (CryptoBERT
Q:\finance-analyzer\portfolio\sentiment.py:421:    leftover state.
Q:\finance-analyzer\portfolio\sentiment.py:506:    """Group headlines by keyword overlap and time proximity.
Q:\finance-analyzer\portfolio\sentiment.py:518:    from portfolio.news_keywords import score_headline
Q:\finance-analyzer\portfolio\sentiment.py:520:    # Extract keywords per headline
Q:\finance-analyzer\portfolio\sentiment.py:521:    headline_keywords = []
Q:\finance-analyzer\portfolio\sentiment.py:531:        headline_keywords.append(words)
Q:\finance-analyzer\portfolio\sentiment.py:533:    # Simple greedy clustering by keyword overlap
Q:\finance-analyzer\portfolio\sentiment.py:542:        kw_i = headline_keywords[i]
Q:\finance-analyzer\portfolio\sentiment.py:547:            kw_j = headline_keywords[j]
Q:\finance-analyzer\portfolio\sentiment.py:549:            # Cluster if they share 2+ keywords or 1 matched keyword
Q:\finance-analyzer\portfolio\sentiment.py:591:    """Return per-sentiment weights from keyword scoring + dissemination."""
Q:\finance-analyzer\portfolio\sentiment.py:593:        from portfolio.news_keywords import score_headline
Q:\finance-analyzer\portfolio\sentiment.py:622:    When headlines are provided, score weights from news_keywords.score_headline()
Q:\finance-analyzer\portfolio\sentiment.py:623:    amplify high-impact keywords (tariff/war/crash 3x). dissemination_mult
Q:\finance-analyzer\portfolio\sentiment.py:685:    Uses news_keywords.is_relevant_headline (keyword OR ticker-synonym match)
Q:\finance-analyzer\portfolio\sentiment.py:686:    plus a credible-source-with-long-title escape hatch (Reuters/Bloomberg/
Q:\finance-analyzer\portfolio\sentiment.py:697:    from portfolio.news_keywords import is_credible_source, is_relevant_headline
Q:\finance-analyzer\portfolio\sentiment.py:705:        # Credible-source escape hatch: long titles from credible outlets are
Q:\finance-analyzer\portfolio\sentiment.py:810:        from portfolio.news_keywords import dissemination_score
Q:\finance-analyzer\portfolio\sentiment.py:915:    # buffer; picked up by flush_ab_log alongside FinGPT and FinBERT.
Q:\finance-analyzer\portfolio\tickers.py:18:    "XAU-USD": {"binance_fapi": "XAUUSDT"},
Q:\finance-analyzer\portfolio\tickers.py:19:    "XAG-USD": {"binance_fapi": "XAGUSDT"},
Q:\finance-analyzer\portfolio\tickers.py:32:METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
Q:\finance-analyzer\portfolio\tickers.py:74:    # where the system already has edge (XAG 5d consensus 61.2%). The
Q:\finance-analyzer\portfolio\tickers.py:118:                        # on XAG. In oscillator_trend cluster where momentum_factors
Q:\finance-analyzer\portfolio\tickers.py:143:                        # XAG 34.9%, XAU 40.2%, MSTR 42.6%; 5065 total sam). Also weak at
Q:\finance-analyzer\portfolio\tickers.py:146:                        # ETH 34.9% (146), MSTR 33.3% (264), XAU N/A. Not salvageable.
Q:\finance-analyzer\portfolio\tickers.py:164:# funding: removed from DISABLED — 74.2% at 3h (535 samples) but 29.9% at 1d.
Q:\finance-analyzer\portfolio\tickers.py:171:_DISABLED_REASON_ENTRY_RE = re.compile(
Q:\finance-analyzer\portfolio\tickers.py:174:_DISABLED_REASON_CONT_RE = re.compile(r'^(\s+)#\s*(.*)$')
Q:\finance-analyzer\portfolio\tickers.py:248:# Skipped for US stocks outside market hours to save GPU resources.
Q:\finance-analyzer\portfolio\tickers.py:261:    "funding",
Q:\finance-analyzer\portfolio\earnings_calendar.py:39:        from portfolio.shared_state import _alpha_vantage_limiter
Q:\finance-analyzer\portfolio\earnings_calendar.py:49:        # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
Q:\finance-analyzer\portfolio\earnings_calendar.py:79:                ed = _date.fromisoformat(rd)
Q:\finance-analyzer\portfolio\earnings_calendar.py:120:                        ed = datetime.fromisoformat(earnings_date).date()
Q:\finance-analyzer\portfolio\seasonality.py:3:Computes average return and volatility profiles by hour-of-day from
Q:\finance-analyzer\portfolio\seasonality.py:17:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\seasonality.py:22:_STATE_FILE = _BASE_DIR / "data" / "seasonality_profiles.json"
Q:\finance-analyzer\portfolio\seasonality.py:23:_MIN_DAYS = 5  # minimum trading days to compute profiles
Q:\finance-analyzer\portfolio\seasonality.py:26:def compute_hourly_profile(klines_1h: pd.DataFrame) -> dict | None:
Q:\finance-analyzer\portfolio\seasonality.py:57:    profile = {}
Q:\finance-analyzer\portfolio\seasonality.py:61:            profile[str(hour)] = {
Q:\finance-analyzer\portfolio\seasonality.py:67:            profile[str(hour)] = {
Q:\finance-analyzer\portfolio\seasonality.py:73:    return profile
Q:\finance-analyzer\portfolio\seasonality.py:76:def detrend_return(raw_return: float, hour: int, profile: dict) -> float:
Q:\finance-analyzer\portfolio\seasonality.py:82:        profile: Hourly profile dict from compute_hourly_profile.
Q:\finance-analyzer\portfolio\seasonality.py:87:    if profile is None:
Q:\finance-analyzer\portfolio\seasonality.py:89:    entry = profile.get(str(hour))
Q:\finance-analyzer\portfolio\seasonality.py:95:def normalize_volatility(raw_vol: float, hour: int, profile: dict) -> float:
Q:\finance-analyzer\portfolio\seasonality.py:101:        profile: Hourly profile dict from compute_hourly_profile.
Q:\finance-analyzer\portfolio\seasonality.py:105:        Returns raw_vol if profile unavailable.
Q:\finance-analyzer\portfolio\seasonality.py:107:    if profile is None:
Q:\finance-analyzer\portfolio\seasonality.py:109:    entry = profile.get(str(hour))
Q:\finance-analyzer\portfolio\seasonality.py:115:def save_profiles(profiles: dict[str, dict]) -> None:
Q:\finance-analyzer\portfolio\seasonality.py:116:    """Persist ticker-keyed profiles to disk.
Q:\finance-analyzer\portfolio\seasonality.py:119:        profiles: Dict keyed by ticker, each value is an hourly profile.
Q:\finance-analyzer\portfolio\seasonality.py:121:    atomic_write_json(_STATE_FILE, profiles)
Q:\finance-analyzer\portfolio\seasonality.py:124:def load_profiles() -> dict:
Q:\finance-analyzer\portfolio\seasonality.py:125:    """Load persisted profiles from disk."""
Q:\finance-analyzer\portfolio\seasonality.py:126:    return load_json(_STATE_FILE) or {}
Q:\finance-analyzer\portfolio\seasonality.py:129:def get_profile(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\seasonality.py:130:    """Load the hourly profile for a specific ticker."""
Q:\finance-analyzer\portfolio\seasonality.py:131:    profiles = load_profiles()
Q:\finance-analyzer\portfolio\seasonality.py:132:    return profiles.get(ticker)
Q:\finance-analyzer\portfolio\data_collector.py:10:import portfolio.shared_state as _ss
Q:\finance-analyzer\portfolio\data_collector.py:43:# Multi-timeframe analysis — (label, binance_interval, num_candles, cache_ttl_seconds)
Q:\finance-analyzer\portfolio\data_collector.py:87:        r.raise_for_status()
Q:\finance-analyzer\portfolio\data_collector.py:109:    """Fetch klines from Binance Futures API (for metals like XAUUSDT, XAGUSDT)."""
Q:\finance-analyzer\portfolio\data_collector.py:140:        r.raise_for_status()
Q:\finance-analyzer\portfolio\data_collector.py:262:        if _ss._current_market_state in ("closed", "weekend", "holiday"):
Q:\finance-analyzer\portfolio\data_collector.py:263:            logger.debug("%s: using yfinance (market %s)", ticker, _ss._current_market_state)
Q:\finance-analyzer\portfolio\data_collector.py:275:# H11/DC-R3-4: use the module-level lock from shared_state so all modules
Q:\finance-analyzer\portfolio\data_collector.py:277:from portfolio.shared_state import yfinance_lock as _yfinance_lock
Q:\finance-analyzer\portfolio\data_collector.py:290:        if "alpaca" in source and _ss._current_market_state in ("closed", "weekend", "holiday"):
Q:\finance-analyzer\portfolio\metals_orderbook.py:3:Fetches L2 snapshots and trade ticks for XAUUSDT / XAGUSDT.
Q:\finance-analyzer\portfolio\metals_orderbook.py:5:imbalance, VPIN, spread metrics).
Q:\finance-analyzer\portfolio\metals_orderbook.py:17:from portfolio.shared_state import _binance_limiter, _cached
Q:\finance-analyzer\portfolio\metals_orderbook.py:22:    "XAU-USD": "XAUUSDT",
Q:\finance-analyzer\portfolio\metals_orderbook.py:23:    "XAG-USD": "XAGUSDT",
Q:\finance-analyzer\portfolio\crypto_macro_data.py:23:from portfolio.shared_state import _cached
Q:\finance-analyzer\portfolio\crypto_macro_data.py:157:            # Max pain = P that MINIMIZES total payout (i.e., max pain for buyers)
Q:\finance-analyzer\portfolio\crypto_macro_data.py:162:        # Max pain for buyers = strike where payout is MINIMIZED
Q:\finance-analyzer\portfolio\crypto_macro_data.py:209:        from portfolio.file_utils import load_json
Q:\finance-analyzer\portfolio\crypto_macro_data.py:210:        summary = load_json(DATA_DIR / "agent_summary_compact.json")
Q:\finance-analyzer\portfolio\crypto_macro_data.py:216:        gold_price = signals.get("XAU-USD", {}).get("price_usd")
Q:\finance-analyzer\portfolio\metals_cross_assets.py:14:on XAG 3h — root cause was 5-day lookbacks evaluated against 3h outcomes
Q:\finance-analyzer\portfolio\metals_cross_assets.py:26:from portfolio.shared_state import _cached
Q:\finance-analyzer\portfolio\metals_cross_assets.py:67:    # yfinance convention). Backends normalize to lowercase; re-capitalize
Q:\finance-analyzer\portfolio\metals_cross_assets.py:106:            "sma20": float(close.rolling(20).mean().iloc[-1]),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:107:            "vs_sma20_pct": float((close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1) * 100),
Q:\finance-analyzer\portfolio\metals_cross_assets.py:123:        mean20 = float(close.rolling(20, min_periods=10).mean().iloc[-1])
Q:\finance-analyzer\portfolio\metals_cross_assets.py:124:        std20 = float(close.rolling(20, min_periods=10).std().iloc[-1])
Q:\finance-analyzer\portfolio\metals_cross_assets.py:154:        sma20 = float(ratio.rolling(20).mean().iloc[-1])
Q:\finance-analyzer\portfolio\metals_cross_assets.py:155:        std20 = float(ratio.rolling(20).std().iloc[-1])
Q:\finance-analyzer\portfolio\metals_cross_assets.py:217:# intraday rolling stats. On weekends/holidays the last ~2 days of bars
Q:\finance-analyzer\portfolio\microstructure.py:11:    - Spread Z-Score: current spread vs rolling distribution
Q:\finance-analyzer\portfolio\microstructure.py:23:def depth_imbalance(depth: dict, levels: int | None = None) -> float:
Q:\finance-analyzer\portfolio\microstructure.py:41:def trade_flow_imbalance(trades: list[dict]) -> dict[str, float] | None:
Q:\finance-analyzer\portfolio\microstructure.py:42:    """Compute signed volume imbalance from recent trades.
Q:\finance-analyzer\portfolio\microstructure.py:51:    imbalance = signed_vol / total_vol if total_vol > 0 else 0.0
Q:\finance-analyzer\portfolio\microstructure.py:56:        "imbalance_ratio": imbalance,
Q:\finance-analyzer\portfolio\microstructure.py:63:def compute_vpin(trades: list[dict], n_buckets: int = 20) -> float | None:
Q:\finance-analyzer\portfolio\microstructure.py:64:    """VPIN: average absolute buy-sell imbalance per volume bucket.
Q:\finance-analyzer\portfolio\microstructure.py:76:    imbalances = []
Q:\finance-analyzer\portfolio\microstructure.py:95:                    imbalances.append(
Q:\finance-analyzer\portfolio\microstructure.py:102:    if not imbalances:
Q:\finance-analyzer\portfolio\microstructure.py:104:    return float(np.mean(imbalances))
Q:\finance-analyzer\portfolio\microstructure.py:107:def compute_ofi(snapshots: list[dict]) -> float:
Q:\finance-analyzer\portfolio\microstructure.py:117:    total_ofi = 0.0
Q:\finance-analyzer\portfolio\microstructure.py:146:        total_ofi += delta_bid - delta_ask
Q:\finance-analyzer\portfolio\microstructure.py:148:    return total_ofi
Q:\finance-analyzer\portfolio\social_sentiment.py:13:# (subreddit, dedicated) — dedicated: keep all posts; general: filter by keywords
Q:\finance-analyzer\portfolio\social_sentiment.py:29:def _fetch_subreddit(sub, keywords, dedicated, per_sub):
Q:\finance-analyzer\portfolio\social_sentiment.py:40:        if not dedicated and not any(kw in title.lower() for kw in keywords):
Q:\finance-analyzer\portfolio\social_sentiment.py:59:def _search_subreddit(sub, keywords, limit=10):
Q:\finance-analyzer\portfolio\social_sentiment.py:60:    query = urllib.parse.quote(" OR ".join(keywords))
Q:\finance-analyzer\portfolio\social_sentiment.py:94:    keywords = TICKER_KEYWORDS.get(short, [short.lower()])
Q:\finance-analyzer\portfolio\social_sentiment.py:104:            fetched = _fetch_subreddit(sub, keywords, dedicated, per_sub)
Q:\finance-analyzer\portfolio\social_sentiment.py:112:    # Fallback: if keyword filtering yielded nothing, try search
Q:\finance-analyzer\portfolio\social_sentiment.py:116:                fetched = _search_subreddit(sub, keywords, limit=per_sub)
Q:\finance-analyzer\portfolio\indicators.py:8:import portfolio.shared_state as _ss
Q:\finance-analyzer\portfolio\indicators.py:71:    bb_mid = close.rolling(20).mean()
Q:\finance-analyzer\portfolio\indicators.py:72:    bb_std = close.rolling(20).std()
Q:\finance-analyzer\portfolio\indicators.py:112:    # RSI rolling percentiles for adaptive thresholds
Q:\finance-analyzer\portfolio\indicators.py:114:    rsi_p20 = rsi_series.rolling(100, min_periods=20).quantile(0.2).iloc[-1]
Q:\finance-analyzer\portfolio\indicators.py:115:    rsi_p80 = rsi_series.rolling(100, min_periods=20).quantile(0.8).iloc[-1]
Q:\finance-analyzer\portfolio\indicators.py:174:    # Compute outside lock (pure function, no shared state)
Q:\finance-analyzer\portfolio\microstructure_state.py:1:"""Microstructure snapshot accumulator for rolling OFI and spread history.
Q:\finance-analyzer\portfolio\microstructure_state.py:5:from the persisted state.
Q:\finance-analyzer\portfolio\microstructure_state.py:8:data/microstructure_state.json for cross-process access.
Q:\finance-analyzer\portfolio\microstructure_state.py:18:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\microstructure_state.py:19:from portfolio.microstructure import compute_ofi, spread_zscore
Q:\finance-analyzer\portfolio\microstructure_state.py:21:logger = logging.getLogger("portfolio.microstructure_state")
Q:\finance-analyzer\portfolio\microstructure_state.py:24:_STATE_FILE = _BASE_DIR / "data" / "microstructure_state.json"
Q:\finance-analyzer\portfolio\microstructure_state.py:38:# (60s) can both call accumulate_snapshot / get_state concurrently.
Q:\finance-analyzer\portfolio\microstructure_state.py:42:_ofi_history: dict[str, deque] = {}  # rolling OFI values for z-score
Q:\finance-analyzer\portfolio\microstructure_state.py:51:    if ticker not in _ofi_history:
Q:\finance-analyzer\portfolio\microstructure_state.py:52:        _ofi_history[ticker] = deque(maxlen=_MAX_OFI_HISTORY)
Q:\finance-analyzer\portfolio\microstructure_state.py:56:    """Add an order book snapshot to the rolling buffer.
Q:\finance-analyzer\portfolio\microstructure_state.py:59:        ticker: Canonical ticker (e.g. "XAG-USD")
Q:\finance-analyzer\portfolio\microstructure_state.py:78:def get_rolling_ofi(ticker: str) -> float:
Q:\finance-analyzer\portfolio\microstructure_state.py:89:    return compute_ofi(snapshots)
Q:\finance-analyzer\portfolio\microstructure_state.py:92:def record_ofi(ticker: str, ofi_val: float) -> None:
Q:\finance-analyzer\portfolio\microstructure_state.py:95:    Called once per cycle from get_microstructure_state to avoid
Q:\finance-analyzer\portfolio\microstructure_state.py:96:    double-appending if get_rolling_ofi is called multiple times.
Q:\finance-analyzer\portfolio\microstructure_state.py:100:        _ofi_history[ticker].append(ofi_val)
Q:\finance-analyzer\portfolio\microstructure_state.py:103:def get_ofi_zscore(ticker: str, current_ofi: float | None = None) -> float:
Q:\finance-analyzer\portfolio\microstructure_state.py:104:    """Z-score of OFI relative to its own rolling distribution.
Q:\finance-analyzer\portfolio\microstructure_state.py:106:    Scores ``current_ofi`` (or the latest recorded value) against the
Q:\finance-analyzer\portfolio\microstructure_state.py:114:        history = list(_ofi_history[ticker])
Q:\finance-analyzer\portfolio\microstructure_state.py:123:    value = current_ofi if current_ofi is not None else arr[-1]
Q:\finance-analyzer\portfolio\microstructure_state.py:127:def get_multiscale_ofi(ticker: str) -> dict:
Q:\finance-analyzer\portfolio\microstructure_state.py:130:    Returns dict with ofi_fast, ofi_medium, ofi_slow, and flow_acceleration
Q:\finance-analyzer\portfolio\microstructure_state.py:138:    ofi_slow = compute_ofi(snapshots) if n >= _MIN_SNAPSHOTS_FOR_OFI else 0.0
Q:\finance-analyzer\portfolio\microstructure_state.py:139:    ofi_medium = compute_ofi(snapshots[-_OFI_WINDOW_MEDIUM:]) if n >= _OFI_WINDOW_MEDIUM else ofi_slow
Q:\finance-analyzer\portfolio\microstructure_state.py:140:    ofi_fast = compute_ofi(snapshots[-_OFI_WINDOW_FAST:]) if n >= _OFI_WINDOW_FAST else ofi_medium
Q:\finance-analyzer\portfolio\microstructure_state.py:142:    # Flow acceleration: compare fast to slow (normalized by snapshot counts).
Q:\finance-analyzer\portfolio\microstructure_state.py:147:        fast_per_snap = ofi_fast / max(_OFI_WINDOW_FAST - 1, 1)
Q:\finance-analyzer\portfolio\microstructure_state.py:148:        slow_per_snap = ofi_slow / max(n - 1, 1) if n > 1 else 0.0
Q:\finance-analyzer\portfolio\microstructure_state.py:154:        "ofi_fast": round(ofi_fast, 4),
Q:\finance-analyzer\portfolio\microstructure_state.py:155:        "ofi_medium": round(ofi_medium, 4),
Q:\finance-analyzer\portfolio\microstructure_state.py:156:        "ofi_slow": round(ofi_slow, 4),
Q:\finance-analyzer\portfolio\microstructure_state.py:175:def get_microstructure_state(ticker: str) -> dict:
Q:\finance-analyzer\portfolio\microstructure_state.py:176:    """Get current accumulated microstructure state for a ticker.
Q:\finance-analyzer\portfolio\microstructure_state.py:178:    Returns dict with ofi, ofi_zscore, multiscale OFI, and spread_zscore.
Q:\finance-analyzer\portfolio\microstructure_state.py:180:    ofi = get_rolling_ofi(ticker)
Q:\finance-analyzer\portfolio\microstructure_state.py:184:    ofi_z = get_ofi_zscore(ticker, current_ofi=ofi)
Q:\finance-analyzer\portfolio\microstructure_state.py:185:    record_ofi(ticker, ofi)
Q:\finance-analyzer\portfolio\microstructure_state.py:187:    ms_ofi = get_multiscale_ofi(ticker)
Q:\finance-analyzer\portfolio\microstructure_state.py:193:        "ofi": ofi,
Q:\finance-analyzer\portfolio\microstructure_state.py:194:        "ofi_zscore": ofi_z,
Q:\finance-analyzer\portfolio\microstructure_state.py:195:        "ofi_fast": ms_ofi["ofi_fast"],
Q:\finance-analyzer\portfolio\microstructure_state.py:196:        "ofi_medium": ms_ofi["ofi_medium"],
Q:\finance-analyzer\portfolio\microstructure_state.py:197:        "ofi_slow": ms_ofi["ofi_slow"],
Q:\finance-analyzer\portfolio\microstructure_state.py:198:        "flow_acceleration": ms_ofi["flow_acceleration"],
Q:\finance-analyzer\portfolio\microstructure_state.py:205:def persist_state() -> None:
Q:\finance-analyzer\portfolio\microstructure_state.py:206:    """Write current microstructure state to disk for cross-process access."""
Q:\finance-analyzer\portfolio\microstructure_state.py:207:    state = {}
Q:\finance-analyzer\portfolio\microstructure_state.py:209:        ms = get_microstructure_state(ticker)
Q:\finance-analyzer\portfolio\microstructure_state.py:211:        state[ticker] = ms
Q:\finance-analyzer\portfolio\microstructure_state.py:212:    if state:
Q:\finance-analyzer\portfolio\microstructure_state.py:213:        atomic_write_json(_STATE_FILE, state)
Q:\finance-analyzer\portfolio\microstructure_state.py:216:def load_persisted_state(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\microstructure_state.py:217:    """Read persisted microstructure state for a ticker.
Q:\finance-analyzer\portfolio\microstructure_state.py:222:    data = load_json(_STATE_FILE)
Q:\finance-analyzer\portfolio\data_refresh.py:42:        r.raise_for_status()
Q:\finance-analyzer\portfolio\forecast_signal.py:16:from portfolio.file_utils import atomic_append_jsonl, load_json
Q:\finance-analyzer\portfolio\forecast_signal.py:272:        # Build dataframe with hourly timestamps (tz-naive, Prophet requirement)
Q:\finance-analyzer\portfolio\forecast_signal.py:273:        now = datetime.now(UTC).replace(tzinfo=None)
Q:\finance-analyzer\portfolio\forecast_signal.py:339:        summary = load_json(AGENT_SUMMARY_FILE)
Q:\finance-analyzer\portfolio\http_retry.py:43:                # H26/HTTP1: Honour Telegram retry_after parameter for 429 responses.
Q:\finance-analyzer\portfolio\http_retry.py:46:                        retry_after = resp.json().get("parameters", {}).get("retry_after", wait)
Q:\finance-analyzer\portfolio\http_retry.py:48:                        retry_after = wait
Q:\finance-analyzer\portfolio\http_retry.py:49:                    wait = retry_after
Q:\finance-analyzer\portfolio\http_retry.py:79:    Combines fetch_with_retry() + raise_for_status() + .json() into one call.
Q:\finance-analyzer\portfolio\http_retry.py:88:        resp.raise_for_status()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:11:Pattern: follows digest.py (state file + sentinel + time check).
Q:\finance-analyzer\portfolio\crypto_scheduler.py:24:from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
Q:\finance-analyzer\portfolio\crypto_scheduler.py:32:STATE_FILE = DATA_DIR / "crypto_scheduler_state.json"
Q:\finance-analyzer\portfolio\crypto_scheduler.py:47:def _get_state():
Q:\finance-analyzer\portfolio\crypto_scheduler.py:48:    """Load scheduler state from file."""
Q:\finance-analyzer\portfolio\crypto_scheduler.py:49:    state = load_json(STATE_FILE, default={})
Q:\finance-analyzer\portfolio\crypto_scheduler.py:50:    return state if isinstance(state, dict) else {}
Q:\finance-analyzer\portfolio\crypto_scheduler.py:53:def _save_state(state):
Q:\finance-analyzer\portfolio\crypto_scheduler.py:54:    """Save scheduler state to file."""
Q:\finance-analyzer\portfolio\crypto_scheduler.py:55:    atomic_write_json(STATE_FILE, state)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:70:    tz_name = crypto_cfg.get("timezone", DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:73:        tz = zoneinfo.ZoneInfo(tz_name)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:75:        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:84:    state = _get_state()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:85:    last_ts = state.get("last_report_time", 0)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:90:    last_hour = state.get("last_report_hour", -1)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:91:    last_date = state.get("last_report_date", "")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:110:    summary = load_json(DATA_DIR / "agent_summary_compact.json")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:178:        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:232:    btc_funding = btc_futures.get("funding_rate_pct", "?")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:234:    lines.append(f"Futures: Funding {btc_funding}% | L/S {btc_ls}")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:260:    eth_funding = eth_futures.get("funding_rate_pct", "?")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:262:    lines.append(f"Futures: Funding {eth_funding}% | L/S {eth_ls}")
Q:\finance-analyzer\portfolio\crypto_scheduler.py:277:        fund_cache = load_json(DATA_DIR / "fundamentals_cache.json", default={})
Q:\finance-analyzer\portfolio\crypto_scheduler.py:310:        "ts": datetime.now().astimezone().isoformat(),
Q:\finance-analyzer\portfolio\crypto_scheduler.py:323:            "funding_rate": btc_futures.get("funding_rate_pct"),
Q:\finance-analyzer\portfolio\crypto_scheduler.py:331:            "funding_rate": eth_futures.get("funding_rate_pct"),
Q:\finance-analyzer\portfolio\crypto_scheduler.py:373:        # Update state
Q:\finance-analyzer\portfolio\crypto_scheduler.py:375:            tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:381:        state = _get_state()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:382:        state["last_report_time"] = time.time()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:383:        state["last_report_hour"] = hour
Q:\finance-analyzer\portfolio\crypto_scheduler.py:384:        state["last_report_date"] = today_str
Q:\finance-analyzer\portfolio\crypto_scheduler.py:385:        _save_state(state)
Q:\finance-analyzer\portfolio\price_source.py:7:* Commodity underlyings (XAG, XAU, oil)            → Binance FAPI perps
Q:\finance-analyzer\portfolio\price_source.py:16:Binance FAPI XAGUSDT             7.7 s       445 ms    ← PRIMARY
Q:\finance-analyzer\portfolio\price_source.py:37:# obvious. The router is alias-aware: 'XAG-USD', 'SI=F', 'XAGUSDT' all
Q:\finance-analyzer\portfolio\price_source.py:43:    "XAG-USD": "XAGUSDT", "SI=F": "XAGUSDT", "XAGUSDT": "XAGUSDT",
Q:\finance-analyzer\portfolio\price_source.py:45:    "XAU-USD": "XAUUSDT", "GC=F": "XAUUSDT", "XAUUSDT": "XAUUSDT",
Q:\finance-analyzer\portfolio\price_source.py:197:        ticker: Symbol in any recognized alias form (XAG-USD, SI=F,
Q:\finance-analyzer\portfolio\price_source.py:198:            XAGUSDT, BTC-USD, MSTR, ^VIX, etc.)
Q:\finance-analyzer\portfolio\news_keywords.py:1:"""Shared keyword dictionaries and sector mappings for news/event signals.
Q:\finance-analyzer\portfolio\news_keywords.py:4:- Tiered keyword lists with severity weights
Q:\finance-analyzer\portfolio\news_keywords.py:79:# Pre-compile patterns sorted by length descending (match longer phrases first)
Q:\finance-analyzer\portfolio\news_keywords.py:81:    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), weight)
Q:\finance-analyzer\portfolio\news_keywords.py:98:    "metals": {"XAU-USD", "XAG-USD"},
Q:\finance-analyzer\portfolio\news_keywords.py:140:    """Score a headline by keyword severity.
Q:\finance-analyzer\portfolio\news_keywords.py:143:        (max_weight, matched_keywords) — max_weight is the highest keyword
Q:\finance-analyzer\portfolio\news_keywords.py:144:        weight found (1.0 if no keywords match), matched_keywords lists all
Q:\finance-analyzer\portfolio\news_keywords.py:145:        matched keyword strings.
Q:\finance-analyzer\portfolio\news_keywords.py:162:def keyword_severity(title: str) -> str:
Q:\finance-analyzer\portfolio\news_keywords.py:183:    """Score how widely news has spread (FinGPT dissemination-aware pattern).
Q:\finance-analyzer\portfolio\news_keywords.py:228:                ts = datetime.fromisoformat(pub_str)
Q:\finance-analyzer\portfolio\news_keywords.py:252:def get_sector_impact(keyword: str, ticker: str) -> str | None:
Q:\finance-analyzer\portfolio\news_keywords.py:253:    """Get the directional impact of a keyword on a specific ticker.
Q:\finance-analyzer\portfolio\news_keywords.py:257:    impacts = KEYWORD_SECTOR_IMPACT.get(keyword.lower(), {})
Q:\finance-analyzer\portfolio\news_keywords.py:277:#   1. It triggers a keyword from score_headline (weight > 1.0), OR
Q:\finance-analyzer\portfolio\news_keywords.py:279:#      Ethereum/ETH, gold/XAU, silver/XAG, or the stock symbol)
Q:\finance-analyzer\portfolio\news_keywords.py:285:    "XAU": ["xau", "gold", "bullion"],
Q:\finance-analyzer\portfolio\news_keywords.py:286:    "XAG": ["xag", "silver"],
Q:\finance-analyzer\portfolio\news_keywords.py:312:        return re.compile(r"\b" + re.escape(short) + r"\b", re.IGNORECASE)
Q:\finance-analyzer\portfolio\news_keywords.py:313:    pattern = "|".join(re.escape(s) for s in syns)
Q:\finance-analyzer\portfolio\news_keywords.py:314:    return re.compile(r"\b(" + pattern + r")\b", re.IGNORECASE)
Q:\finance-analyzer\portfolio\funding_rate.py:3:from portfolio.shared_state import FUNDING_RATE_TTL, _cached
Q:\finance-analyzer\portfolio\funding_rate.py:11:def _fetch_funding_rate(ticker):
Q:\finance-analyzer\portfolio\funding_rate.py:12:    """Fetch and interpret funding rate for a single ticker."""
Q:\finance-analyzer\portfolio\funding_rate.py:18:        label="funding_rate",
Q:\finance-analyzer\portfolio\funding_rate.py:25:    # deployment / weird symbol state / schema change can return a payload
Q:\finance-analyzer\portfolio\funding_rate.py:28:    # funding signal for the cycle (and 8 of those = whole cycle thread pool
Q:\finance-analyzer\portfolio\funding_rate.py:31:    last_funding = data.get("lastFundingRate")
Q:\finance-analyzer\portfolio\funding_rate.py:33:    if last_funding is None or mark_price is None:
Q:\finance-analyzer\portfolio\funding_rate.py:36:        rate = float(last_funding)
Q:\finance-analyzer\portfolio\funding_rate.py:41:    # Normal funding ~0.01% (0.0001). Thresholds:
Q:\finance-analyzer\portfolio\funding_rate.py:42:    #   > 0.03% → overleveraged longs → contrarian SELL
Q:\finance-analyzer\portfolio\funding_rate.py:43:    #   < -0.01% → overleveraged shorts → contrarian BUY
Q:\finance-analyzer\portfolio\funding_rate.py:59:def get_funding_rate(ticker):
Q:\finance-analyzer\portfolio\funding_rate.py:62:    return _cached(f"funding_rate_{ticker}", FUNDING_RATE_TTL,
Q:\finance-analyzer\portfolio\funding_rate.py:63:                   _fetch_funding_rate, ticker)
Q:\finance-analyzer\portfolio\funding_rate.py:68:        print(f"{t}: {get_funding_rate(t)}")
Q:\finance-analyzer\portfolio\fx_rates.py:3:NOTE: Not migrated to shared_state._cached() because this module has custom
Q:\finance-analyzer\portfolio\fx_rates.py:24:_fx_cache = {"rate": None, "time": 0}
Q:\finance-analyzer\portfolio\fx_rates.py:25:_FX_STALE_THRESHOLD = 7200  # 2 hours — warn if FX rate hasn't been refreshed
Q:\finance-analyzer\portfolio\fx_rates.py:31:        cached_rate = _fx_cache["rate"]
Q:\finance-analyzer\portfolio\fx_rates.py:32:        cached_time = _fx_cache["time"]
Q:\finance-analyzer\portfolio\fx_rates.py:43:        r.raise_for_status()
Q:\finance-analyzer\portfolio\fx_rates.py:51:                _fx_cache["rate"] = rate
Q:\finance-analyzer\portfolio\fx_rates.py:52:                _fx_cache["time"] = now
Q:\finance-analyzer\portfolio\fx_rates.py:58:        cached_rate = _fx_cache["rate"]
Q:\finance-analyzer\portfolio\fx_rates.py:59:        cached_time = _fx_cache["time"]
Q:\finance-analyzer\portfolio\fx_rates.py:62:        if age_secs > _FX_STALE_THRESHOLD:
Q:\finance-analyzer\portfolio\fx_rates.py:78:        last_alert = _fx_cache.get("_last_fx_alert", 0)
Q:\finance-analyzer\portfolio\fx_rates.py:81:        _fx_cache["_last_fx_alert"] = now
Q:\finance-analyzer\portfolio\session_calendar.py:1:"""Session calendar — instrument-specific trading hours and session state.
Q:\finance-analyzer\portfolio\session_calendar.py:8:    info = get_session_info("warrant", underlying="XAG-USD")
Q:\finance-analyzer\portfolio\session_calendar.py:9:    # info.remaining_minutes, info.session_end, info.is_extended, ...
Q:\finance-analyzer\portfolio\session_calendar.py:22:    """Trading session state for an instrument.
Q:\finance-analyzer\portfolio\session_calendar.py:25:        session_end: Absolute datetime (UTC) of normal session close.
Q:\finance-analyzer\portfolio\session_calendar.py:33:    session_end: datetime
Q:\finance-analyzer\portfolio\session_calendar.py:48:# We handle DST for EU sessions too.
Q:\finance-analyzer\portfolio\session_calendar.py:53:    EU DST: last Sunday of March 01:00 UTC → last Sunday of October 01:00 UTC.
Q:\finance-analyzer\portfolio\session_calendar.py:76:    """Convert CET time to UTC time object, adjusted for DST on given date."""
Q:\finance-analyzer\portfolio\session_calendar.py:82:def _make_session_end(now: datetime, cet_hour: int, cet_minute: int) -> datetime:
Q:\finance-analyzer\portfolio\session_calendar.py:118:    """Get current session state for an instrument.
Q:\finance-analyzer\portfolio\session_calendar.py:122:        underlying: Underlying ticker for warrants (e.g., "XAG-USD").
Q:\finance-analyzer\portfolio\session_calendar.py:140:            session_end=end,
Q:\finance-analyzer\portfolio\session_calendar.py:149:    # US stocks: NYSE hours with DST
Q:\finance-analyzer\portfolio\session_calendar.py:155:        session_end = now.replace(hour=close_utc, minute=0, second=0, microsecond=0)
Q:\finance-analyzer\portfolio\session_calendar.py:157:                   now.replace(hour=open_utc, minute=30, second=0) <= now < session_end)
Q:\finance-analyzer\portfolio\session_calendar.py:159:        remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
Q:\finance-analyzer\portfolio\session_calendar.py:166:            session_end=session_end,
Q:\finance-analyzer\portfolio\session_calendar.py:180:    session_end = _make_session_end(now, ch, cm)
Q:\finance-analyzer\portfolio\session_calendar.py:181:    session_open = _make_session_end(now, oh, om)
Q:\finance-analyzer\portfolio\session_calendar.py:184:    is_open = is_weekday and session_open <= now < session_end
Q:\finance-analyzer\portfolio\session_calendar.py:186:    remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
Q:\finance-analyzer\portfolio\session_calendar.py:192:        # US stock underlying — check US market hours
Q:\finance-analyzer\portfolio\session_calendar.py:197:        session_end=session_end,
Q:\finance-analyzer\portfolio\onchain_data.py:3:Fetches MVRV Z-Score, SOPR, NUPL, realized price, exchange netflow,
Q:\finance-analyzer\portfolio\onchain_data.py:22:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\onchain_data.py:24:from portfolio.shared_state import _cached
Q:\finance-analyzer\portfolio\onchain_data.py:38:        - "2026-04-11T..."      → parsed via datetime.fromisoformat
Q:\finance-analyzer\portfolio\onchain_data.py:54:            return datetime.fromisoformat(iso).timestamp()
Q:\finance-analyzer\portfolio\onchain_data.py:99:        # BUG-125: Use atomic_write_json to prevent corrupt cache on crash
Q:\finance-analyzer\portfolio\onchain_data.py:100:        atomic_write_json(CACHE_FILE, data, ensure_ascii=False)
Q:\finance-analyzer\portfolio\onchain_data.py:107:    data = load_json(CACHE_FILE)
Q:\finance-analyzer\portfolio\onchain_data.py:137:    """Fetch latest MVRV and MVRV Z-Score."""
Q:\finance-analyzer\portfolio\onchain_data.py:148:    """Fetch latest SOPR (Spent Output Profit Ratio)."""
Q:\finance-analyzer\portfolio\onchain_data.py:156:    """Fetch latest NUPL (Net Unrealized Profit/Loss)."""
Q:\finance-analyzer\portfolio\onchain_data.py:252:    persistent = load_json(CACHE_FILE, default={})
Q:\finance-analyzer\portfolio\onchain_data.py:264:            from portfolio.shared_state import _cache_lock, _tool_cache
Q:\finance-analyzer\portfolio\onchain_data.py:301:    # MVRV Z-Score zones
Q:\finance-analyzer\portfolio\onchain_data.py:311:    # SOPR zones
Q:\finance-analyzer\portfolio\onchain_data.py:317:            interp["sopr_zone"] = "profit_taking"
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
The first pass already surfaced one likely budget-accounting defect and one direct retry bypass in `social_sentiment`. I’m reading the smaller modules in full next, then I’ll go back for targeted context inside the larger ones.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\funding_rate.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\session_calendar.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\microstructure.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\microstructure_state.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\onchain_data.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\social_sentiment.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\earnings_calendar.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\tickers.py" in Q:\finance-analyzer
 succeeded in 1065ms:
1:"""Session calendar — instrument-specific trading hours and session state.
2:
3:Provides remaining-session time, session boundaries, and session mismatch
4:detection for the exit optimizer.
5:
6:Usage:
7:    from portfolio.session_calendar import get_session_info
8:    info = get_session_info("warrant", underlying="XAG-USD")
9:    # info.remaining_minutes, info.session_end, info.is_extended, ...
10:"""
11:
12:from __future__ import annotations
13:
14:from dataclasses import dataclass
15:from datetime import UTC, datetime, time, timedelta
16:
17:from portfolio.market_timing import _is_us_dst
18:
19:
20:@dataclass(frozen=True)
21:class SessionInfo:
22:    """Trading session state for an instrument.
23:
24:    Attributes:
25:        session_end: Absolute datetime (UTC) of normal session close.
26:        extended_end: Absolute datetime (UTC) of extended session close, if applicable.
27:        remaining_minutes: Minutes until effective close (extended if available).
28:        is_open: Whether the instrument is currently tradeable.
29:        is_extended: Whether we're in the extended (evening) session.
30:        underlying_open: Whether the underlying's primary market is open (for warrants).
31:        phase: Human-readable phase: "open", "extended", "pre_open", "closed".
32:    """
33:    session_end: datetime
34:    extended_end: datetime | None
35:    remaining_minutes: float
36:    is_open: bool
37:    is_extended: bool
38:    underlying_open: bool
39:    phase: str
40:
41:
42:# ---------------------------------------------------------------------------
43:# Session definitions (times in UTC)
44:# ---------------------------------------------------------------------------
45:
46:# Avanza commodity warrants: 08:15-21:55 CET = 07:15-20:55 UTC (winter)
47:# CET = UTC+1 (winter), CEST = UTC+2 (summer)
48:# We handle DST for EU sessions too.
49:
50:def _eu_dst(dt: datetime) -> bool:
51:    """Check if datetime falls in EU Central European Summer Time (CEST).
52:
53:    EU DST: last Sunday of March 01:00 UTC → last Sunday of October 01:00 UTC.
54:    """
55:    year = dt.year
56:
57:    # Last Sunday of March
58:    mar31 = datetime(year, 3, 31, tzinfo=UTC)
59:    last_sun_mar = 31 - (mar31.weekday() + 1) % 7
60:    dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)
61:
62:    # Last Sunday of October
63:    oct31 = datetime(year, 10, 31, tzinfo=UTC)
64:    last_sun_oct = 31 - (oct31.weekday() + 1) % 7
65:    dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)
66:
67:    return dst_start <= dt < dst_end
68:
69:
70:def _cet_offset(dt: datetime) -> int:
71:    """Return CET/CEST offset from UTC in hours (1 or 2)."""
72:    return 2 if _eu_dst(dt) else 1
73:
74:
75:def _cet_to_utc(hour: int, minute: int, dt: datetime) -> time:
76:    """Convert CET time to UTC time object, adjusted for DST on given date."""
77:    offset = _cet_offset(dt)
78:    utc_hour = (hour - offset) % 24
79:    return time(utc_hour, minute)
80:
81:
82:def _make_session_end(now: datetime, cet_hour: int, cet_minute: int) -> datetime:
83:    """Create a UTC datetime for today's session end from CET time."""
84:    offset = _cet_offset(now)
85:    utc_hour = cet_hour - offset
86:    end = now.replace(hour=utc_hour, minute=cet_minute, second=0, microsecond=0)
87:    if end.tzinfo is None:
88:        end = end.replace(tzinfo=UTC)
89:    return end
90:
91:
92:# Session specs: (open_cet, close_cet) as (hour, minute) tuples
93:SESSIONS = {
94:    "warrant": {
95:        "open_cet": (8, 15),
96:        "close_cet": (21, 55),
97:        "has_extended": False,  # Already includes evening trading
98:        "description": "Avanza commodity warrants",
99:    },
100:    "stock_se": {
101:        "open_cet": (9, 0),
102:        "close_cet": (17, 25),
103:        "has_extended": False,
104:        "description": "Nasdaq Stockholm equities",
105:    },
106:    "crypto": {
107:        "open_cet": (0, 0),
108:        "close_cet": (23, 59),
109:        "has_extended": False,
110:        "description": "Crypto 24/7",
111:    },
112:}
113:
114:
115:def get_session_info(instrument_type: str,
116:                     underlying: str | None = None,
117:                     now: datetime | None = None) -> SessionInfo:
118:    """Get current session state for an instrument.
119:
120:    Args:
121:        instrument_type: "warrant", "stock_se", "stock_us", "crypto".
122:        underlying: Underlying ticker for warrants (e.g., "XAG-USD").
123:        now: Current UTC time. Defaults to now.
124:
125:    Returns:
126:        SessionInfo with remaining time, phase, and session boundaries.
127:    """
128:    if now is None:
129:        now = datetime.now(UTC)
130:    if now.tzinfo is None:
131:        now = now.replace(tzinfo=UTC)
132:
133:    # Crypto: always open (24/7)
134:    if instrument_type == "crypto":
135:        # Use midnight as "session end" — effectively infinite session
136:        end = (now + timedelta(days=1)).replace(
137:            hour=0, minute=0, second=0, microsecond=0
138:        )
139:        return SessionInfo(
140:            session_end=end,
141:            extended_end=None,
142:            remaining_minutes=(end - now).total_seconds() / 60,
143:            is_open=True,
144:            is_extended=False,
145:            underlying_open=True,
146:            phase="open",
147:        )
148:
149:    # US stocks: NYSE hours with DST
150:    if instrument_type == "stock_us":
151:        us_dst = _is_us_dst(now)
152:        open_utc = 13 if us_dst else 14   # 09:30 ET
153:        close_utc = 20 if us_dst else 21  # 16:00 ET
154:
155:        session_end = now.replace(hour=close_utc, minute=0, second=0, microsecond=0)
156:        is_open = (now.weekday() < 5 and
157:                   now.replace(hour=open_utc, minute=30, second=0) <= now < session_end)
158:
159:        remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
160:        phase = "open" if is_open else "closed"
161:
162:        # Check if underlying is open (for warrants referencing US stocks)
163:        underlying_open = is_open
164:
165:        return SessionInfo(
166:            session_end=session_end,
167:            extended_end=None,
168:            remaining_minutes=remaining,
169:            is_open=is_open,
170:            is_extended=False,
171:            underlying_open=underlying_open,
172:            phase=phase,
173:        )
174:
175:    # EU-based instruments (warrants, Stockholm stocks)
176:    spec = SESSIONS.get(instrument_type, SESSIONS["warrant"])
177:    oh, om = spec["open_cet"]
178:    ch, cm = spec["close_cet"]
179:
180:    session_end = _make_session_end(now, ch, cm)
181:    session_open = _make_session_end(now, oh, om)
182:
183:    is_weekday = now.weekday() < 5
184:    is_open = is_weekday and session_open <= now < session_end
185:
186:    remaining = max(0, (session_end - now).total_seconds() / 60) if is_open else 0
187:    phase = "open" if is_open else "closed"
188:
189:    # For warrants, check if underlying's primary market is open
190:    underlying_open = True  # Metals trade ~24h
191:    if underlying and not underlying.endswith("-USD"):
192:        # US stock underlying — check US market hours
193:        us_info = get_session_info("stock_us", now=now)
194:        underlying_open = us_info.is_open
195:
196:    return SessionInfo(
197:        session_end=session_end,
198:        extended_end=None,
199:        remaining_minutes=remaining,
200:        is_open=is_open,
201:        is_extended=False,
202:        underlying_open=underlying_open,
203:        phase=phase,
204:    )
205:
206:
207:def remaining_session_minutes(instrument_type: str = "warrant",
208:                              now: datetime | None = None) -> float:
209:    """Shortcut: get remaining minutes for an instrument's session."""
210:    info = get_session_info(instrument_type, now=now)
211:    return info.remaining_minutes
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1104ms:
1:from portfolio.api_utils import BINANCE_FAPI_BASE as BINANCE_FAPI
2:from portfolio.http_retry import fetch_json
3:from portfolio.shared_state import FUNDING_RATE_TTL, _cached
4:
5:SYMBOL_MAP = {
6:    "BTC-USD": "BTCUSDT",
7:    "ETH-USD": "ETHUSDT",
8:}
9:
10:
11:def _fetch_funding_rate(ticker):
12:    """Fetch and interpret funding rate for a single ticker."""
13:    symbol = SYMBOL_MAP[ticker]
14:    data = fetch_json(
15:        f"{BINANCE_FAPI}/premiumIndex",
16:        params={"symbol": symbol},
17:        timeout=10,
18:        label="funding_rate",
19:    )
20:    if data is None:
21:        return None
22:
23:    # Adversarial review 04-29 DE-P1-1 (2026-05-02): Binance FAPI premiumIndex
24:    # response is normally {lastFundingRate, markPrice, ...}, but a partial
25:    # deployment / weird symbol state / schema change can return a payload
26:    # missing one of those fields. The pre-fix code did `data["lastFundingRate"]`
27:    # which raised KeyError that bubbled into the worker thread, killing the
28:    # funding signal for the cycle (and 8 of those = whole cycle thread pool
29:    # poisoned). Use .get() + None-return so the signal disappears cleanly
30:    # for ONE cycle instead of crashing.
31:    last_funding = data.get("lastFundingRate")
32:    mark_price = data.get("markPrice")
33:    if last_funding is None or mark_price is None:
34:        return None
35:    try:
36:        rate = float(last_funding)
37:        mark_price_f = float(mark_price)
38:    except (TypeError, ValueError):
39:        return None
40:
41:    # Normal funding ~0.01% (0.0001). Thresholds:
42:    #   > 0.03% → overleveraged longs → contrarian SELL
43:    #   < -0.01% → overleveraged shorts → contrarian BUY
44:    if rate > 0.0003:
45:        action = "SELL"
46:    elif rate < -0.0001:
47:        action = "BUY"
48:    else:
49:        action = "HOLD"
50:
51:    return {
52:        "rate": rate,
53:        "rate_pct": round(rate * 100, 4),
54:        "action": action,
55:        "mark_price": mark_price_f,
56:    }
57:
58:
59:def get_funding_rate(ticker):
60:    if ticker not in SYMBOL_MAP:
61:        return None
62:    return _cached(f"funding_rate_{ticker}", FUNDING_RATE_TTL,
63:                   _fetch_funding_rate, ticker)
64:
65:
66:if __name__ == "__main__":
67:    for t in ["BTC-USD", "ETH-USD", "SOL-USD"]:
68:        print(f"{t}: {get_funding_rate(t)}")
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1112ms:
1:"""Microstructure feature computations for short-term metals prediction.
2:
3:Implements order-flow and market-microstructure metrics from academic literature.
4:All functions are pure — they take raw data and return numeric features.
5:
6:Key features:
7:    - Depth Imbalance: ln(V_bid) - ln(V_ask)  [Lipton et al.]
8:    - Trade Flow Imbalance: signed volume ratio
9:    - VPIN: Volume-synchronized probability of informed trading
10:    - OFI: Order Flow Imbalance from quote changes  [Cont et al. 2014]
11:    - Spread Z-Score: current spread vs rolling distribution
12:"""
13:from __future__ import annotations
14:
15:import logging
16:import math
17:
18:import numpy as np
19:
20:logger = logging.getLogger("portfolio.microstructure")
21:
22:
23:def depth_imbalance(depth: dict, levels: int | None = None) -> float:
24:    """Log ratio of bid vs ask volume: F_t = ln(V_bid) - ln(V_ask).
25:
26:    Positive → bid-heavy (buying pressure).
27:    Negative → ask-heavy (selling pressure).
28:    """
29:    bids = depth.get("bids", [])
30:    asks = depth.get("asks", [])
31:    if levels is not None:
32:        bids = bids[:levels]
33:        asks = asks[:levels]
34:    bid_vol = sum(q for _, q in bids)
35:    ask_vol = sum(q for _, q in asks)
36:    if bid_vol <= 0 or ask_vol <= 0:
37:        return 0.0
38:    return math.log(bid_vol) - math.log(ask_vol)
39:
40:
41:def trade_flow_imbalance(trades: list[dict]) -> dict[str, float] | None:
42:    """Compute signed volume imbalance from recent trades.
43:    Each trade dict must have 'qty' (float) and 'sign' (+1 buyer, -1 seller).
44:    """
45:    if not trades:
46:        return None
47:    buy_vol = sum(t["qty"] for t in trades if t["sign"] == 1)
48:    sell_vol = sum(t["qty"] for t in trades if t["sign"] == -1)
49:    total_vol = buy_vol + sell_vol
50:    signed_vol = buy_vol - sell_vol
51:    imbalance = signed_vol / total_vol if total_vol > 0 else 0.0
52:    return {
53:        "signed_volume": signed_vol,
54:        "buy_volume": buy_vol,
55:        "sell_volume": sell_vol,
56:        "imbalance_ratio": imbalance,
57:        "trade_count": len(trades),
58:        "buy_count": sum(1 for t in trades if t["sign"] == 1),
59:        "sell_count": sum(1 for t in trades if t["sign"] == -1),
60:    }
61:
62:
63:def compute_vpin(trades: list[dict], n_buckets: int = 20) -> float | None:
64:    """VPIN: average absolute buy-sell imbalance per volume bucket.
65:
66:    High VPIN (>0.6) → toxic flow, likely informed trading.
67:    Low VPIN (<0.3) → balanced, uninformed flow.
68:    """
69:    if len(trades) < n_buckets:
70:        return None
71:    total_vol = sum(t["qty"] for t in trades)
72:    if total_vol <= 0:
73:        return None
74:    bucket_size = total_vol / n_buckets
75:
76:    imbalances = []
77:    bucket_buy = 0.0
78:    bucket_sell = 0.0
79:    bucket_vol = 0.0
80:
81:    for t in trades:
82:        qty = t["qty"]
83:        remaining = qty
84:        while remaining > 0:
85:            space = bucket_size - bucket_vol
86:            fill = min(remaining, space)
87:            if t["sign"] == 1:
88:                bucket_buy += fill
89:            else:
90:                bucket_sell += fill
91:            bucket_vol += fill
92:            remaining -= fill
93:            if bucket_vol >= bucket_size - 1e-12:
94:                if bucket_buy + bucket_sell > 0:
95:                    imbalances.append(
96:                        abs(bucket_buy - bucket_sell) / (bucket_buy + bucket_sell)
97:                    )
98:                bucket_buy = 0.0
99:                bucket_sell = 0.0
100:                bucket_vol = 0.0
101:
102:    if not imbalances:
103:        return None
104:    return float(np.mean(imbalances))
105:
106:
107:def compute_ofi(snapshots: list[dict]) -> float:
108:    """Order Flow Imbalance from consecutive order book snapshots.
109:
110:    Implements the Cont et al. (2014) OFI formula.
111:    Positive OFI → net buying pressure.
112:    Negative OFI → net selling pressure.
113:    """
114:    if len(snapshots) < 2:
115:        return 0.0
116:
117:    total_ofi = 0.0
118:    for i in range(1, len(snapshots)):
119:        prev = snapshots[i - 1]
120:        curr = snapshots[i]
121:
122:        prev_bid = prev["best_bid"]
123:        curr_bid = curr["best_bid"]
124:        prev_bid_vol = prev["bids"][0][1] if prev["bids"] else 0.0
125:        curr_bid_vol = curr["bids"][0][1] if curr["bids"] else 0.0
126:
127:        if curr_bid > prev_bid:
128:            delta_bid = curr_bid_vol
129:        elif curr_bid == prev_bid:
130:            delta_bid = curr_bid_vol - prev_bid_vol
131:        else:
132:            delta_bid = -prev_bid_vol
133:
134:        prev_ask = prev["best_ask"]
135:        curr_ask = curr["best_ask"]
136:        prev_ask_vol = prev["asks"][0][1] if prev["asks"] else 0.0
137:        curr_ask_vol = curr["asks"][0][1] if curr["asks"] else 0.0
138:
139:        if curr_ask < prev_ask:
140:            delta_ask = curr_ask_vol
141:        elif curr_ask == prev_ask:
142:            delta_ask = curr_ask_vol - prev_ask_vol
143:        else:
144:            delta_ask = -prev_ask_vol
145:
146:        total_ofi += delta_bid - delta_ask
147:
148:    return total_ofi
149:
150:
151:def spread_zscore(spread_history: list[float], min_samples: int = 5) -> float | None:
152:    """Z-score of current spread vs recent history."""
153:    if len(spread_history) < min_samples:
154:        return None
155:    arr = np.array(spread_history, dtype=float)
156:    mean = arr[:-1].mean()
157:    std = arr[:-1].std()
158:    if std < 1e-12:
159:        # Zero variance: if current matches mean it's normal (0.0),
160:        # otherwise it's an extreme outlier — return large signed value.
161:        diff = arr[-1] - mean
162:        if abs(diff) < 1e-12:
163:            return 0.0
164:        return float(np.sign(diff) * 10.0)
165:    return float((arr[-1] - mean) / std)
166:
167:
168:# ---------------------------------------------------------------------------
169:# Trade-Through Detection (approximate)
170:# ---------------------------------------------------------------------------
171:
172:def detect_trade_throughs(trades: list[dict], threshold_bps: float = 5.0) -> dict:
173:    """Detect trade-throughs: trades that jump across multiple price levels.
174:
175:    A trade-through occurs when a market order is large enough to consume
176:    multiple levels of the order book, causing the execution price to jump
177:    significantly from the previous trade.  We approximate this from
178:    the trades list by detecting price gaps > threshold between consecutive
179:    trades in the same direction.
180:
181:    Args:
182:        trades: List of trade dicts with 'price', 'qty', 'sign'.
183:        threshold_bps: Minimum price jump in basis points to count as
184:                       trade-through (default 5 bps = 0.05%).
185:
186:    Returns:
187:        Dict with buy_throughs, sell_throughs (counts), total_volume_throughs,
188:        and max_gap_bps.  Returns zeros if insufficient trades.
189:    """
190:    if len(trades) < 2:
191:        return {
192:            "buy_throughs": 0,
193:            "sell_throughs": 0,
194:            "total_throughs": 0,
195:            "through_volume": 0.0,
196:            "max_gap_bps": 0.0,
197:        }
198:
199:    buy_throughs = 0
200:    sell_throughs = 0
201:    through_volume = 0.0
202:    max_gap_bps = 0.0
203:
204:    for i in range(1, len(trades)):
205:        prev = trades[i - 1]
206:        curr = trades[i]
207:        mid_price = (prev["price"] + curr["price"]) / 2.0
208:        if mid_price <= 0:
209:            continue
210:        gap_bps = abs(curr["price"] - prev["price"]) / mid_price * 10000
211:
212:        if gap_bps >= threshold_bps and curr["sign"] == prev["sign"]:
213:            # Same-direction large gap = likely trade-through
214:            if curr["sign"] == 1:
215:                buy_throughs += 1
216:            else:
217:                sell_throughs += 1
218:            through_volume += curr["qty"]
219:            max_gap_bps = max(max_gap_bps, gap_bps)
220:
221:    return {
222:        "buy_throughs": buy_throughs,
223:        "sell_throughs": sell_throughs,
224:        "total_throughs": buy_throughs + sell_throughs,
225:        "through_volume": round(through_volume, 4),
226:        "max_gap_bps": round(max_gap_bps, 2),
227:    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1120ms:
1:"""Microstructure snapshot accumulator for rolling OFI and spread history.
2:
3:Called each cycle by metals_loop.py to build order book snapshot history.
4:The orderbook_flow signal reads the accumulated OFI and spread z-score
5:from the persisted state.
6:
7:State is kept in memory (ring buffer) and persisted to
8:data/microstructure_state.json for cross-process access.
9:"""
10:from __future__ import annotations
11:
12:import logging
13:import threading
14:import time
15:from collections import deque
16:from pathlib import Path
17:
18:from portfolio.file_utils import atomic_write_json, load_json
19:from portfolio.microstructure import compute_ofi, spread_zscore
20:
21:logger = logging.getLogger("portfolio.microstructure_state")
22:
23:_BASE_DIR = Path(__file__).resolve().parent.parent
24:_STATE_FILE = _BASE_DIR / "data" / "microstructure_state.json"
25:_MAX_SNAPSHOTS = 60  # ~30-60 min at 30-60s intervals
26:_MIN_SNAPSHOTS_FOR_OFI = 3
27:_MIN_SPREADS_FOR_ZSCORE = 10
28:_MIN_OFI_HISTORY_FOR_ZSCORE = 10
29:_MAX_OFI_HISTORY = 120  # ~2h of OFI readings for z-score normalization
30:
31:# Multi-scale OFI windows (snapshot counts)
32:_OFI_WINDOW_FAST = 5   # ~5 min
33:_OFI_WINDOW_MEDIUM = 15  # ~15 min
34:# slow = all snapshots (full buffer)
35:
36:# In-memory ring buffers per ticker.
37:# Protected by _buffer_lock — metals_loop fast-tick (10s) and main cycle
38:# (60s) can both call accumulate_snapshot / get_state concurrently.
39:_buffer_lock = threading.Lock()
40:_snapshot_buffers: dict[str, deque] = {}
41:_spread_buffers: dict[str, deque] = {}
42:_ofi_history: dict[str, deque] = {}  # rolling OFI values for z-score
43:
44:
45:def _ensure_buffer(ticker: str) -> None:
46:    """Initialize ring buffers for a ticker if needed."""
47:    if ticker not in _snapshot_buffers:
48:        _snapshot_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
49:    if ticker not in _spread_buffers:
50:        _spread_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
51:    if ticker not in _ofi_history:
52:        _ofi_history[ticker] = deque(maxlen=_MAX_OFI_HISTORY)
53:
54:
55:def accumulate_snapshot(ticker: str, depth: dict) -> None:
56:    """Add an order book snapshot to the rolling buffer.
57:
58:    Args:
59:        ticker: Canonical ticker (e.g. "XAG-USD")
60:        depth: Order book depth dict from metals_orderbook.get_orderbook_depth()
61:               Must have: best_bid, best_ask, bids, asks, spread
62:    """
63:    if depth is None:
64:        return
65:    snapshot = {
66:        "best_bid": depth["best_bid"],
67:        "best_ask": depth["best_ask"],
68:        "bids": depth["bids"][:5],   # keep top 5 levels only
69:        "asks": depth["asks"][:5],
70:        "ts": depth.get("ts", int(time.time() * 1000)),
71:    }
72:    with _buffer_lock:
73:        _ensure_buffer(ticker)
74:        _snapshot_buffers[ticker].append(snapshot)
75:        _spread_buffers[ticker].append(depth["spread"])
76:
77:
78:def get_rolling_ofi(ticker: str) -> float:
79:    """Compute OFI from accumulated snapshots for a ticker.
80:
81:    Returns cumulative OFI over the last N snapshots.
82:    Returns 0.0 if insufficient history.
83:    """
84:    with _buffer_lock:
85:        _ensure_buffer(ticker)
86:        snapshots = list(_snapshot_buffers[ticker])
87:    if len(snapshots) < _MIN_SNAPSHOTS_FOR_OFI:
88:        return 0.0
89:    return compute_ofi(snapshots)
90:
91:
92:def record_ofi(ticker: str, ofi_val: float) -> None:
93:    """Record an OFI value for z-score history tracking.
94:
95:    Called once per cycle from get_microstructure_state to avoid
96:    double-appending if get_rolling_ofi is called multiple times.
97:    """
98:    with _buffer_lock:
99:        _ensure_buffer(ticker)
100:        _ofi_history[ticker].append(ofi_val)
101:
102:
103:def get_ofi_zscore(ticker: str, current_ofi: float | None = None) -> float:
104:    """Z-score of OFI relative to its own rolling distribution.
105:
106:    Scores ``current_ofi`` (or the latest recorded value) against the
107:    historical distribution WITHOUT including the current value.
108:    This prevents self-contamination that compresses z-scores toward zero.
109:
110:    Returns 0.0 if insufficient history.
111:    """
112:    with _buffer_lock:
113:        _ensure_buffer(ticker)
114:        history = list(_ofi_history[ticker])
115:    if len(history) < _MIN_OFI_HISTORY_FOR_ZSCORE:
116:        return 0.0
117:    import numpy as np
118:    arr = np.array(history, dtype=float)
119:    mean = arr.mean()
120:    std = arr.std()
121:    if std < 1e-12:
122:        return 0.0
123:    value = current_ofi if current_ofi is not None else arr[-1]
124:    return float((value - mean) / std)
125:
126:
127:def get_multiscale_ofi(ticker: str) -> dict:
128:    """Compute OFI at 3 time scales: fast (~5min), medium (~15min), slow (full).
129:
130:    Returns dict with ofi_fast, ofi_medium, ofi_slow, and flow_acceleration
131:    (fast z-score minus slow z-score — positive = accelerating buying).
132:    """
133:    with _buffer_lock:
134:        _ensure_buffer(ticker)
135:        snapshots = list(_snapshot_buffers[ticker])
136:    n = len(snapshots)
137:
138:    ofi_slow = compute_ofi(snapshots) if n >= _MIN_SNAPSHOTS_FOR_OFI else 0.0
139:    ofi_medium = compute_ofi(snapshots[-_OFI_WINDOW_MEDIUM:]) if n >= _OFI_WINDOW_MEDIUM else ofi_slow
140:    ofi_fast = compute_ofi(snapshots[-_OFI_WINDOW_FAST:]) if n >= _OFI_WINDOW_FAST else ofi_medium
141:
142:    # Flow acceleration: compare fast to slow (normalized by snapshot counts).
143:    # Only meaningful when we have enough snapshots for distinct windows;
144:    # during warmup (n < _OFI_WINDOW_FAST), fast==slow so acceleration
145:    # would produce misleading non-zero values (code review MEDIUM-2).
146:    if n >= _OFI_WINDOW_FAST:
147:        fast_per_snap = ofi_fast / max(_OFI_WINDOW_FAST - 1, 1)
148:        slow_per_snap = ofi_slow / max(n - 1, 1) if n > 1 else 0.0
149:        flow_acceleration = fast_per_snap - slow_per_snap
150:    else:
151:        flow_acceleration = 0.0
152:
153:    return {
154:        "ofi_fast": round(ofi_fast, 4),
155:        "ofi_medium": round(ofi_medium, 4),
156:        "ofi_slow": round(ofi_slow, 4),
157:        "flow_acceleration": round(flow_acceleration, 4),
158:    }
159:
160:
161:def get_spread_zscore(ticker: str) -> float | None:
162:    """Compute spread z-score from accumulated spread history.
163:
164:    Returns z-score of current spread vs recent history.
165:    Returns None if insufficient data.
166:    """
167:    with _buffer_lock:
168:        _ensure_buffer(ticker)
169:        spreads = list(_spread_buffers[ticker])
170:    if len(spreads) < _MIN_SPREADS_FOR_ZSCORE:
171:        return None
172:    return spread_zscore(spreads)
173:
174:
175:def get_microstructure_state(ticker: str) -> dict:
176:    """Get current accumulated microstructure state for a ticker.
177:
178:    Returns dict with ofi, ofi_zscore, multiscale OFI, and spread_zscore.
179:    """
180:    ofi = get_rolling_ofi(ticker)
181:    # Compute z-score BEFORE appending current value to history —
182:    # scoring against a population that includes itself compresses
183:    # the z-score toward zero (code review finding CRITICAL-1).
184:    ofi_z = get_ofi_zscore(ticker, current_ofi=ofi)
185:    record_ofi(ticker, ofi)
186:    sz = get_spread_zscore(ticker)
187:    ms_ofi = get_multiscale_ofi(ticker)
188:    with _buffer_lock:
189:        _ensure_buffer(ticker)
190:        snap_count = len(_snapshot_buffers[ticker])
191:        spread_count = len(_spread_buffers[ticker])
192:    return {
193:        "ofi": ofi,
194:        "ofi_zscore": ofi_z,
195:        "ofi_fast": ms_ofi["ofi_fast"],
196:        "ofi_medium": ms_ofi["ofi_medium"],
197:        "ofi_slow": ms_ofi["ofi_slow"],
198:        "flow_acceleration": ms_ofi["flow_acceleration"],
199:        "spread_zscore": sz if sz is not None else 0.0,
200:        "snapshot_count": snap_count,
201:        "spread_count": spread_count,
202:    }
203:
204:
205:def persist_state() -> None:
206:    """Write current microstructure state to disk for cross-process access."""
207:    state = {}
208:    for ticker in _snapshot_buffers:
209:        ms = get_microstructure_state(ticker)
210:        ms["ts"] = int(time.time() * 1000)
211:        state[ticker] = ms
212:    if state:
213:        atomic_write_json(_STATE_FILE, state)
214:
215:
216:def load_persisted_state(ticker: str) -> dict | None:
217:    """Read persisted microstructure state for a ticker.
218:
219:    Used by orderbook_flow signal when running in a different process
220:    from metals_loop.
221:    """
222:    data = load_json(_STATE_FILE)
223:    if not data or ticker not in data:
224:        return None
225:    entry = data[ticker]
226:    age_ms = int(time.time() * 1000) - entry.get("ts", 0)
227:    if age_ms > 120_000:  # stale if >2 minutes old
228:        return None
229:    return entry
230:
231:
232:def snapshot_count(ticker: str) -> int:
233:    """Return current snapshot buffer size for a ticker."""
234:    with _buffer_lock:
235:        _ensure_buffer(ticker)
236:        return len(_snapshot_buffers[ticker])
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1129ms:
1:"""BGeometrics on-chain data integration for Bitcoin.
2:
3:Fetches MVRV Z-Score, SOPR, NUPL, realized price, exchange netflow,
4:and liquidation data from the free BGeometrics API (bitcoin-data.com).
5:
6:Free tier: 8 requests/hour, 15 requests/day.
7:Budget: 6 metrics x 2 refreshes/day = 12 requests.
8:Cache: 12 hours per metric (on-chain data doesn't change fast).
9:
10:Usage:
11:    from portfolio.onchain_data import get_onchain_data, interpret_onchain
12:    data = get_onchain_data()  # returns dict or None
13:    interp = interpret_onchain(data)  # returns interpretation dict
14:"""
15:
16:import logging
17:import time
18:from datetime import datetime
19:from pathlib import Path
20:
21:from portfolio.api_utils import load_config as _load_config
22:from portfolio.file_utils import atomic_write_json, load_json
23:from portfolio.http_retry import fetch_json
24:from portfolio.shared_state import _cached
25:
26:logger = logging.getLogger("portfolio.onchain_data")
27:
28:
29:def _coerce_epoch(value) -> float:
30:    """A-DE-5 (2026-04-11): Coerce a cache timestamp to a unix-epoch float.
31:
32:    Older versions of onchain_cache.json stored "ts" as an ISO 8601 string
33:    instead of an epoch number. The seeding code below does
34:    `time.time() - cache_ts < TTL` which crashes with TypeError when ts is
35:    a string. This helper accepts:
36:        - int / float           → returned as float
37:        - "1712345678"          → parsed as float
38:        - "2026-04-11T..."      → parsed via datetime.fromisoformat
39:        - anything else / fail  → 0.0 (treated as ancient → cache miss)
40:
41:    Returning 0 on failure is safe: it forces a cache miss, which costs
42:    one extra API call but never silently breaks the on-chain voter.
43:    """
44:    if isinstance(value, (int, float)):
45:        return float(value)
46:    if isinstance(value, str) and value:
47:        try:
48:            return float(value)
49:        except ValueError:
50:            pass
51:        try:
52:            # Handle "Z" suffix for UTC zulu time
53:            iso = value.replace("Z", "+00:00")
54:            return datetime.fromisoformat(iso).timestamp()
55:        except (ValueError, TypeError):
56:            pass
57:    # P1-14 (2026-05-02): defensive fall-through. Returning 0.0 silently
58:    # forces a cache miss, which costs one extra API call but never breaks
59:    # the on-chain voter. Log at DEBUG so operators investigating "why is
60:    # the BGeometrics 15-req/day budget burning every restart?" can see
61:    # this firing in the logs without breaking the safe-default behaviour.
62:    logger.debug(
63:        "_coerce_epoch: unparseable value type=%s repr=%.50r — returning 0.0",
64:        type(value).__name__,
65:        value,
66:    )
67:    return 0.0
68:
69:BASE_DIR = Path(__file__).resolve().parent.parent
70:DATA_DIR = BASE_DIR / "data"
71:CACHE_FILE = DATA_DIR / "onchain_cache.json"
72:
73:API_BASE = "https://bitcoin-data.com"
74:ONCHAIN_TTL = 43200  # 12 hours
75:
76:
77:# ---------------------------------------------------------------------------
78:# Config
79:# ---------------------------------------------------------------------------
80:
81:def _load_config_token():
82:    """Load BGeometrics API token from config.json."""
83:    try:
84:        config = _load_config()
85:        token = config.get("bgeometrics", {}).get("api_token", "")
86:        return token if token else None
87:    except Exception as e:
88:        logger.warning("BGeometrics token load failed: %s", e, exc_info=True)
89:        return None
90:
91:
92:# ---------------------------------------------------------------------------
93:# Persistent cache (survives restarts)
94:# ---------------------------------------------------------------------------
95:
96:def _save_onchain_cache(data):
97:    """Save on-chain data to persistent cache file."""
98:    try:
99:        # BUG-125: Use atomic_write_json to prevent corrupt cache on crash
100:        atomic_write_json(CACHE_FILE, data, ensure_ascii=False)
101:    except Exception:
102:        logger.warning("Failed to write onchain cache", exc_info=True)
103:
104:
105:def _load_onchain_cache(max_age_seconds=ONCHAIN_TTL):
106:    """Load on-chain data from persistent cache if fresh enough."""
107:    data = load_json(CACHE_FILE)
108:    if data is None:
109:        return None
110:    try:
111:        ts = data.get("ts", 0)
112:        if time.time() - ts > max_age_seconds:
113:            return None
114:        return data
115:    except Exception as e:
116:        logger.warning("On-chain cache load failed: %s", e, exc_info=True)
117:        return None
118:
119:
120:# ---------------------------------------------------------------------------
121:# Individual metric fetchers
122:# ---------------------------------------------------------------------------
123:
124:def _api_get(endpoint, token, params=None):
125:    """Make authenticated GET request to BGeometrics API.
126:
127:    Skips retries on failure since retrying just burns more of the
128:    8 req/hour free tier budget.
129:    """
130:    url = f"{API_BASE}{endpoint}"
131:    headers = {"Authorization": f"Bearer {token}"}
132:    return fetch_json(url, headers=headers, params=params, timeout=15,
133:                      retries=0, label=f"bgeometrics:{endpoint}")
134:
135:
136:def _fetch_mvrv(token):
137:    """Fetch latest MVRV and MVRV Z-Score."""
138:    data = _api_get("/v1/mvrv/last", token)
139:    if not data or not isinstance(data, dict):
140:        return None
141:    return {
142:        "mvrv": data.get("mvrv"),
143:        "mvrv_zscore": data.get("mvrvZScore"),
144:    }
145:
146:
147:def _fetch_sopr(token):
148:    """Fetch latest SOPR (Spent Output Profit Ratio)."""
149:    data = _api_get("/v1/sopr/last", token)
150:    if not data or not isinstance(data, dict):
151:        return None
152:    return {"sopr": data.get("sopr")}
153:
154:
155:def _fetch_nupl(token):
156:    """Fetch latest NUPL (Net Unrealized Profit/Loss)."""
157:    data = _api_get("/v1/nupl/last", token)
158:    if not data or not isinstance(data, dict):
159:        return None
160:    return {"nupl": data.get("nupl")}
161:
162:
163:def _fetch_realized_price(token):
164:    """Fetch latest realized price."""
165:    data = _api_get("/v1/realized-price/last", token)
166:    if not data or not isinstance(data, dict):
167:        return None
168:    return {"realized_price": data.get("realizedPrice")}
169:
170:
171:def _fetch_exchange_netflow(token):
172:    """Fetch latest exchange netflow (negative = accumulation)."""
173:    data = _api_get("/v1/exchange-netflow", token, params={"size": 1})
174:    if not data or not isinstance(data, list) or len(data) == 0:
175:        return None
176:    latest = data[0] if isinstance(data[0], dict) else data[-1]
177:    return {"netflow": latest.get("netflow")}
178:
179:
180:def _fetch_liquidations(token):
181:    """Fetch latest BTC liquidation data."""
182:    data = _api_get("/v1/btc-liquidations", token, params={"size": 1})
183:    if not data or not isinstance(data, list) or len(data) == 0:
184:        return None
185:    latest = data[0] if isinstance(data[0], dict) else data[-1]
186:    return {
187:        "long_liquidations": latest.get("longLiquidations"),
188:        "short_liquidations": latest.get("shortLiquidations"),
189:    }
190:
191:
192:# ---------------------------------------------------------------------------
193:# Main aggregator
194:# ---------------------------------------------------------------------------
195:
196:def _safe_float(val):
197:    """Convert API value to float, handling strings and None."""
198:    if val is None:
199:        return None
200:    try:
201:        return float(val)
202:    except (ValueError, TypeError):
203:        return None
204:
205:
206:def _fetch_all_onchain(token):
207:    """Fetch all 6 on-chain metrics and aggregate into a single dict.
208:
209:    Adds 1s delay between requests to respect free tier rate limits (8 req/hr).
210:    """
211:    result = {"ts": time.time()}
212:
213:    fetchers = [
214:        ("mvrv", _fetch_mvrv),
215:        ("sopr", _fetch_sopr),
216:        ("nupl", _fetch_nupl),
217:        ("realized_price", _fetch_realized_price),
218:        ("exchange_netflow", _fetch_exchange_netflow),
219:        ("liquidations", _fetch_liquidations),
220:    ]
221:
222:    any_success = False
223:    for i, (name, fetcher) in enumerate(fetchers):
224:        if i > 0:
225:            time.sleep(1)  # Rate limit: space out requests
226:        try:
227:            data = fetcher(token)
228:            if data:
229:                # Convert string values to float
230:                result.update({k: _safe_float(v) if k != "ts" else v
231:                              for k, v in data.items()})
232:                any_success = True
233:        except Exception:
234:            logger.warning("BGeometrics %s fetch failed", name, exc_info=True)
235:
236:    if not any_success:
237:        return None
238:
239:    # Save to persistent cache
240:    _save_onchain_cache(result)
241:    return result
242:
243:
244:def get_onchain_data():
245:    """Get on-chain data for BTC, using in-memory + persistent cache.
246:
247:    Returns dict with all available metrics, or None if unavailable.
248:    """
249:    # H12/DC-R3-5: On restart, seed the in-memory cache from the persistent cache
250:    # if it is still fresh. This avoids firing 6 BGeometrics API calls immediately
251:    # after every process restart (burns the 15 req/day budget).
252:    persistent = load_json(CACHE_FILE, default={})
253:    if persistent:
254:        # A-DE-5 (2026-04-11): Defensive timestamp parse. Older versions of
255:        # this cache stored "ts" as an ISO 8601 string instead of an epoch
256:        # float. The arithmetic on line below crashes with TypeError when
257:        # ts is a string, which then propagates up and silently disables
258:        # the on-chain BTC voter on the next restart. Detect and convert.
259:        raw_ts = persistent.get("ts", 0) or persistent.get("_fetched_at", 0)
260:        cache_ts = _coerce_epoch(raw_ts)
261:        if time.time() - cache_ts < ONCHAIN_TTL:
262:            # Still fresh — pre-populate the in-memory cache so _cached() returns
263:            # immediately without hitting the API.
264:            from portfolio.shared_state import _cache_lock, _tool_cache
265:            with _cache_lock:
266:                if "onchain_btc" not in _tool_cache:
267:                    _tool_cache["onchain_btc"] = {
268:                        "data": persistent,
269:                        "time": cache_ts,
270:                        "ttl": ONCHAIN_TTL,
271:                    }
272:                    logger.debug("On-chain: seeded in-memory cache from persistent (age %.0fs)",
273:                                 time.time() - cache_ts)
274:
275:    token = _load_config_token()
276:    if not token:
277:        # Try persistent cache even without token
278:        cached = _load_onchain_cache(max_age_seconds=ONCHAIN_TTL * 2)
279:        if cached:
280:            logger.debug("No BGeometrics token, using stale cache")
281:            return cached
282:        return None
283:
284:    return _cached("onchain_btc", ONCHAIN_TTL, _fetch_all_onchain, token)
285:
286:
287:# ---------------------------------------------------------------------------
288:# Interpretation helpers (for Layer 2 context)
289:# ---------------------------------------------------------------------------
290:
291:def interpret_onchain(data):
292:    """Interpret on-chain metrics into human-readable zones.
293:
294:    Returns dict with zone classifications for each available metric.
295:    """
296:    if not data:
297:        return {}
298:
299:    interp = {}
300:
301:    # MVRV Z-Score zones
302:    zscore = _safe_float(data.get("mvrv_zscore"))
303:    if zscore is not None:
304:        if zscore < 1:
305:            interp["mvrv_zone"] = "undervalued"
306:        elif zscore > 7:
307:            interp["mvrv_zone"] = "overheated"
308:        else:
309:            interp["mvrv_zone"] = "neutral"
310:
311:    # SOPR zones
312:    sopr = _safe_float(data.get("sopr"))
313:    if sopr is not None:
314:        if sopr < 0.97:
315:            interp["sopr_zone"] = "capitulation"
316:        elif sopr > 1.05:
317:            interp["sopr_zone"] = "profit_taking"
318:        else:
319:            interp["sopr_zone"] = "neutral"
320:
321:    # NUPL zones
322:    nupl = _safe_float(data.get("nupl"))
323:    if nupl is not None:
324:        if nupl < 0:
325:            interp["nupl_zone"] = "capitulation"
326:        elif nupl > 0.75:
327:            interp["nupl_zone"] = "euphoria"
328:        elif nupl > 0.5:
329:            interp["nupl_zone"] = "greed"
330:        elif nupl > 0.25:
331:            interp["nupl_zone"] = "optimism"
332:        else:
333:            interp["nupl_zone"] = "hope"
334:
335:    # Exchange netflow
336:    netflow = _safe_float(data.get("netflow"))
337:    if netflow is not None:
338:        if netflow < 0:
339:            interp["netflow_signal"] = "accumulation"
340:        else:
341:            interp["netflow_signal"] = "distribution"
342:
343:    return interp
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1157ms:
1:"""Single source of truth for all ticker lists, source mappings, and symbol constants.
2:
3:Every module that needs ticker definitions should import from here instead
4:of maintaining its own copy.
5:"""
6:
7:import re
8:from functools import lru_cache
9:from pathlib import Path
10:
11:# ── Tier 1: Full signals (30 signals, 7 timeframes) ──────────────────────
12:
13:SYMBOLS = {
14:    # Crypto (Binance spot)
15:    "BTC-USD": {"binance": "BTCUSDT"},
16:    "ETH-USD": {"binance": "ETHUSDT"},
17:    # Metals (Binance futures)
18:    "XAU-USD": {"binance_fapi": "XAUUSDT"},
19:    "XAG-USD": {"binance_fapi": "XAGUSDT"},
20:    # US Equities (Alpaca IEX) — MSTR kept as BTC NAV-premium reference for metals_loop
21:    "MSTR": {"alpaca": "MSTR"},
22:    # Removed Mar 15: AMD, GOOGL, AMZN, AAPL, AVGO, META, SOUN, LMT
23:    # Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT
24:    #   Reduces main loop load to stay under 60s cadence. Cycle p50 was 143s with
25:    #   12 tickers — dropping to 5 is expected to bring p50 under target. MSTR retained
26:    #   because data/metals_loop.py uses it for BTC NAV-premium tracking.
27:}
28:
29:# ── Asset-class subsets ───────────────────────────────────────────────────
30:
31:CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
32:METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
33:STOCK_SYMBOLS = {"MSTR"}
34:
35:# All known tickers (union of all subsets)
36:ALL_TICKERS = CRYPTO_SYMBOLS | METALS_SYMBOLS | STOCK_SYMBOLS
37:
38:# ── Derived mappings (all from SYMBOLS — single source of truth) ─────────
39:
40:BINANCE_SPOT_MAP = {
41:    t: sym for t, src in SYMBOLS.items()
42:    for k, sym in src.items() if k == "binance"
43:}
44:BINANCE_FAPI_MAP = {
45:    t: sym for t, src in SYMBOLS.items()
46:    for k, sym in src.items() if k == "binance_fapi"
47:}
48:BINANCE_MAP = {**BINANCE_SPOT_MAP, **BINANCE_FAPI_MAP}
49:
50:# Ticker -> (source_type, symbol) mapping (used by macro_context)
51:TICKER_SOURCE_MAP = {
52:    t: next(iter(src.items())) for t, src in SYMBOLS.items()
53:}
54:
55:# Yahoo Finance symbol mapping — stock tickers map to themselves
56:YF_MAP = {t: t for t in STOCK_SYMBOLS}
57:
58:# ── Signal names (used by outcome_tracker, accuracy_stats) ───────────────
59:# Canonical source is portfolio.signal_registry.get_signal_names().
60:# This static list is kept for backward compatibility with modules that
61:# import SIGNAL_NAMES directly (outcome_tracker, accuracy_stats).
62:
63:# Signals that are force-HOLD (disabled due to poor accuracy).
64:# Kept in SIGNAL_NAMES for historical tracking but excluded from active reports.
65:DISABLED_SIGNALS = {
66:    "ml",               # 41.7% accuracy (1714 sam) — worse than coin flip
67:    "fibonacci",        # 2026-04-29: 43.6% at 1d (17024 sam), 43.3% at 3h (8811 sam).
68:                        # Consistently below coin flip across ALL horizons and tickers
69:                        # with massive sample size. Was accuracy-gated but still computed
70:                        # every cycle (~50ms wasted). Formal disable saves CPU.
71:    # "cot_positioning" re-enabled 2026-04-13 for shadow validation (was
72:    # force-HOLD pending live validation, 0 samples). COT is a weekly signal
73:    # (CFTC Friday release) — expected to contribute mostly at 3d/5d horizons
74:    # where the system already has edge (XAG 5d consensus 61.2%). The
75:    # existing accuracy gate in signal_engine.py auto-disables any signal
76:    # below 45% accuracy once 30+ samples accumulate, so re-enabling is
77:    # self-correcting.
78:    "futures_basis",    # 0 accuracy samples — pending live validation
79:    "hurst_regime",     # pending live validation (added 2026-04-11)
80:    "shannon_entropy",  # pending live validation (added 2026-04-12)
81:    "vix_term_structure",  # pending live validation (added 2026-04-13)
82:    "gold_real_yield_paradox",  # pending live validation (added 2026-04-14)
83:    "cross_asset_tsmom",  # pending live validation (added 2026-04-15)
84:    "copper_gold_ratio",  # pending live validation (added 2026-04-17)
85:    # "statistical_jump_regime" RE-ENABLED 2026-04-29: 52.7% accuracy (110 sam)
86:    # at 1d — above 47% gate, marginal but worth live validation. Shadow-safe
87:    # since 2026-04-18. If it degrades below 47% the accuracy gate auto-disables.
88:    "network_momentum",  # pending live validation (added 2026-04-19)
89:    "ovx_metals_spillover",  # pending live validation (added 2026-04-20)
90:    "xtrend_equity_spillover",  # pending live validation (added 2026-04-21)
91:    "complexity_gap_regime",  # pending live validation (added 2026-04-22)
92:    "realized_skewness",  # KILLED 2026-04-29: 33.3% at 1d (90 sam). Below coin flip.
93:    "mahalanobis_turbulence",  # pending live validation (added 2026-04-24)
94:    "crypto_evrp",  # pending live validation (added 2026-04-25)
95:    "hash_ribbons",  # pending live validation (added 2026-04-26)
96:    "drift_regime_gate",  # pending live validation (added 2026-04-28)
97:    "vol_ratio_regime",  # pending live validation (added 2026-04-29)
98:    "residual_pair_reversion",  # pending live validation (added 2026-04-30)
99:    "williams_vix_fix",  # pending live validation (added 2026-05-01)
100:    "treasury_risk_rotation",  # pending live validation (added 2026-05-07)
101:    "intraday_seasonality",  # pending live validation (added 2026-05-08)
102:    "cubic_trend_persistence",  # pending live validation (added 2026-05-09)
103:    "vwap_zscore_mr",  # pending live validation (added 2026-05-10)
104:    "calendar",         # 2026-05-09: 29.3% recent accuracy — structural BUY bias
105:                        # (6/8 sub-signals BUY-only). Actively harmful. Was per-horizon
106:                        # blacklisted at 1d but still voted at other horizons.
107:    "futures_flow",     # 2026-05-07: 38.3% at 1d (2168 sam). Actively harmful —
108:                        # 12pp worse than coin flip. In cross_asset_flow cluster
109:                        # but still wastes compute. Was accuracy-gated at runtime
110:                        # but formal disable saves ~50ms/cycle.
111:    "trend",            # 2026-05-07: 46.1% at 1d (17880 sam), 40.3% at 3h.
112:                        # Massive sample, consistently below threshold across ALL
113:                        # horizons. 92-100% correlated with ema/macro_regime in
114:                        # pure_trend cluster. In ranging regime (current) this is
115:                        # pure noise. ema (50.0%) is the cluster leader.
116:    "macd",             # 2026-05-07: 44.2% at 1d (6136 sam), 43.7% at 3h.
117:                        # Below threshold across all horizons. Only 5.3% activation
118:                        # on XAG. In oscillator_trend cluster where momentum_factors
119:                        # (53.2%) is the better signal.
120:    # "econ_calendar" RE-ENABLED 2026-04-23. BUG-218 fixed: added post_event_relief
121:    # sub-signal that emits BUY after high-impact events pass (4-24h relief window)
122:    # and during event-free calm windows (>72h to next event). The composite is now
123:    # 5 sub-signals (3 SELL + 1 BUY + 1 neutral) instead of 4 SELL-only.
124:    # 62.6% accuracy before disabling. Accuracy gate will auto-gate if BUY
125:    # signals degrade the composite.
126:    "orderbook_flow",   # 2026-04-11: 51.1% accuracy (360 sam), 93.3% activation rate,
127:                        # no recent data. Pure noise in every consensus decision.
128:                        # Re-evaluate after 2 weeks of accuracy data collection.
129:    # "forecast" RE-ENABLED 2026-04-21. The 36-39% accuracy measured on 2026-04-12
130:    # was polluted by Kronos voting 100% HOLD in shadow mode — Kronos occupied 3 of 6
131:    # slots in _health_weighted_vote whenever its subprocess succeeded, dragging every
132:    # composite vote toward HOLD regardless of Chronos's verdict. With Kronos retired
133:    # in portfolio/signals/forecast.py (same PR), the composite is now Chronos-only.
134:    # Chronos effective accuracy: 1h=45.4%, 24h=52.4% (4d ago). The 47% tiered
135:    # accuracy gate will force-HOLD 1h while letting 24h contribute. Forecast stayed
136:    # in this set for 10 days, which ALSO silenced forecast_predictions.jsonl and
137:    # forecast_health.jsonl because signal_engine.py skips disabled signals before
138:    # invocation — so we lost all shadow/health visibility while the signal was off.
139:    # Re-enabling restores both the signal and the logging. If accuracy degrades
140:    # again post-Kronos-retire, move into REGIME_GATED_SIGNALS (24h-only) rather
141:    # than re-disabling blindly.
142:    "oscillators",      # 2026-04-14: below 45% on ALL tickers at 1d (BTC 35.8%, ETH 36.3%,
143:                        # XAG 34.9%, XAU 40.2%, MSTR 42.6%; 5065 total sam). Also weak at
144:                        # 3h (34-45% per ticker). Regime-gated in ranging but noise everywhere.
145:    "smart_money",      # 2026-04-24: below 40% on ALL Tier 1 tickers at 1d — BTC 39.8% (123),
146:                        # ETH 34.9% (146), MSTR 33.3% (264), XAU N/A. Not salvageable.
147:                        # Cross-ticker consistent failure. 51.6% aggregate masks per-ticker disaster.
148:    "claude_fundamental",  # 2026-05-03: CRASHED to 19.8% recent 1d (222 sam) from 57.9%
149:                        # all-time. Root cause: Opus tier has 95% BUY bias (76/80 votes BUY),
150:                        # Sonnet 73% BUY bias. Haiku 83% abstention (useless). In ranging
151:                        # market these BUY calls are mostly wrong. Bias detectors (added
152:                        # 2026-04-25) couldn't prevent structural LLM bullish lean.
153:                        # Re-enable after fixing bias detector thresholds.
154:    "sentiment",        # 2026-05-03: 33.8% at 3h recent (3629 sam), 45.9% all-time (39579 sam).
155:                        # CryptoBERT predictions are noise. High-volume signal actively hurting
156:                        # consensus. Always in macro_external cluster but dragging down peers.
157:}
158:# 2026-04-11 research session changes:
159:# - orderbook_flow DISABLED: 93.3% active, 51.1% accuracy, 0 recent data. Noise.
160:# - credit_spread_risk ENABLED: 66.9% accuracy (257 sam), BUY 80.3%. Directional
161:#   gate at 40% will auto-gate SELL (49.1%) while allowing strong BUY votes.
162:# - crypto_macro ENABLED: 56.5% accuracy (1273 sam). BUY-biased (93%) so bias
163:#   penalty (0.5x) applies. Provides crypto-specific on-chain edge.
164:# funding: removed from DISABLED — 74.2% at 3h (535 samples) but 29.9% at 1d.
165:# Horizon-gated via REGIME_GATED_SIGNALS to only vote at 3h/4h.
166:
167:# 2026-05-05: Surface the disable reason to the dashboard tooltip by parsing the
168:# inline comments next to each DISABLED_SIGNALS entry. Done via source-file
169:# parsing (rather than a parallel dict) so the comments stay the single source
170:# of truth. Falls back to None if the file shape changes.
171:_DISABLED_REASON_ENTRY_RE = re.compile(
172:    r'^(\s*)"([a-z_][a-z0-9_]*)"\s*,\s*(?:#\s*(.*))?$'
173:)
174:_DISABLED_REASON_CONT_RE = re.compile(r'^(\s+)#\s*(.*)$')
175:
176:
177:def _clean_disabled_reason(lines: list[str]) -> str:
178:    """Join continuation comments and trim to a single short summary."""
179:    if not lines:
180:        return ""
181:    text = " ".join(lines).strip()
182:    for sep in (". ", " — "):
183:        if sep in text:
184:            text = text.split(sep, 1)[0].rstrip(".")
185:            break
186:    return text[:160].rstrip()
187:
188:
189:@lru_cache(maxsize=1)
190:def _parse_disabled_reasons() -> dict[str, str]:
191:    """Parse the DISABLED_SIGNALS literal in this file into {name: reason}.
192:
193:    A continuation comment is recognised when its `#` is indented strictly
194:    further than the entry name's column, which excludes flush-left
195:    separator comments (e.g. the commented-out re-enable notes) from
196:    bleeding into the previous entry's reason.
197:    """
198:    try:
199:        src = Path(__file__).resolve().read_text(encoding="utf-8")
200:    except OSError:
201:        return {}
202:    block_match = re.search(
203:        r'^DISABLED_SIGNALS\s*=\s*\{(.*?)^\}',
204:        src, re.MULTILINE | re.DOTALL,
205:    )
206:    if not block_match:
207:        return {}
208:    out: dict[str, str] = {}
209:    current: str | None = None
210:    current_lines: list[str] = []
211:    entry_indent = 0
212:    for raw in block_match.group(1).splitlines():
213:        m_entry = _DISABLED_REASON_ENTRY_RE.match(raw)
214:        if m_entry:
215:            if current is not None:
216:                out[current] = _clean_disabled_reason(current_lines)
217:            current = m_entry.group(2)
218:            entry_indent = len(m_entry.group(1))
219:            first = (m_entry.group(3) or "").strip()
220:            current_lines = [first] if first else []
221:            continue
222:        m_cont = _DISABLED_REASON_CONT_RE.match(raw)
223:        if m_cont and current is not None:
224:            indent = len(m_cont.group(1))
225:            if indent > entry_indent:
226:                txt = m_cont.group(2).strip()
227:                if txt:
228:                    current_lines.append(txt)
229:    if current is not None:
230:        out[current] = _clean_disabled_reason(current_lines)
231:    return out
232:
233:
234:def get_disabled_reason(signal_name: str) -> str | None:
235:    """Return a short reason for why `signal_name` is disabled, or None.
236:
237:    Returns None for signals not in DISABLED_SIGNALS, and for disabled
238:    signals whose comment was empty or unparseable.
239:    """
240:    if signal_name not in DISABLED_SIGNALS:
241:        return None
242:    reasons = _parse_disabled_reasons()
243:    reason = reasons.get(signal_name)
244:    return reason if reason else None
245:
246:
247:# Signals that require local GPU inference.
248:# Skipped for US stocks outside market hours to save GPU resources.
249:# claude_fundamental excluded — uses remote API, has its own market-hours gate.
250:GPU_SIGNALS = frozenset({"ministral", "qwen3", "forecast"})
251:
252:SIGNAL_NAMES = [
253:    "rsi",
254:    "macd",
255:    "ema",
256:    "bb",
257:    "fear_greed",
258:    "sentiment",
259:    "ministral",
260:    "ml",
261:    "funding",
262:    "volume",
263:    "qwen3",
264:    # custom_lora removed — disabled signal, was polluting accuracy stats
265:    # Enhanced composite signals
266:    "trend",
267:    "momentum",
268:    "volume_flow",
269:    "volatility_sig",
270:    "candlestick",
271:    "structure",
272:    "fibonacci",
273:    "smart_money",
274:    "oscillators",
275:    "heikin_ashi",
276:    "mean_reversion",
277:    "calendar",
278:    "macro_regime",
279:    "momentum_factors",
280:    "news_event",
281:    "econ_calendar",
282:    "forecast",
283:    "claude_fundamental",
284:    "futures_flow",
285:    "crypto_macro",
286:    "orderbook_flow",
287:    "metals_cross_asset",
288:    "dxy_cross_asset",
289:    "cot_positioning",
290:    "credit_spread_risk",
291:    "onchain",
292:    "futures_basis",
293:    "hurst_regime",
294:    "shannon_entropy",
295:    "vix_term_structure",
296:    "gold_real_yield_paradox",
297:    "cross_asset_tsmom",
298:    "copper_gold_ratio",
299:    "statistical_jump_regime",
300:    "network_momentum",
301:    "ovx_metals_spillover",
302:    "xtrend_equity_spillover",
303:    "complexity_gap_regime",
304:    "realized_skewness",
305:    "mahalanobis_turbulence",
306:    "crypto_evrp",
307:    "hash_ribbons",
308:    "drift_regime_gate",
309:    "vol_ratio_regime",
310:    "residual_pair_reversion",
311:    "williams_vix_fix",
312:    "treasury_risk_rotation",
313:    "intraday_seasonality",
314:    "cubic_trend_persistence",
315:    "vwap_zscore_mr",
316:    "btc_proxy",
317:]
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1158ms:
1:"""Earnings calendar — proximity gate to prevent trading near earnings.
2:
3:Fetches upcoming earnings dates from yfinance for US stock tickers.
4:When a ticker has earnings within GATE_DAYS, BUY signals are gated to HOLD
5:to avoid unpredictable binary events.
6:
7:Cache: per-ticker, 24h TTL (earnings dates don't change intraday).
8:"""
9:
10:import logging
11:import threading
12:import time
13:from contextlib import suppress
14:from datetime import UTC, datetime
15:
16:from portfolio.tickers import STOCK_SYMBOLS
17:
18:logger = logging.getLogger("portfolio.earnings_calendar")
19:
20:# Gate window: force HOLD if earnings within this many calendar days
21:GATE_DAYS = 2
22:
23:# Cache TTL: 24 hours (earnings dates are stable)
24:EARNINGS_CACHE_TTL = 86400
25:
26:# Per-ticker cache: {ticker: {"data": {...}, "time": epoch}}
27:_earnings_cache: dict[str, dict] = {}
28:_earnings_lock = threading.Lock()
29:
30:
31:def _fetch_earnings_alpha_vantage(ticker: str) -> dict | None:
32:    """Fetch next earnings date from Alpha Vantage EARNINGS endpoint.
33:
34:    Uses the already-configured AV API key and rate limiter.
35:    """
36:    try:
37:        from portfolio.api_utils import load_config
38:        from portfolio.http_retry import fetch_with_retry
39:        from portfolio.shared_state import _alpha_vantage_limiter
40:
41:        config = load_config()
42:        # C9/DC-R3-1: key lives under config["alpha_vantage"]["api_key"], not
43:        # the flat "alpha_vantage_key" key that doesn't exist.
44:        api_key = config.get("alpha_vantage", {}).get("api_key", "")
45:        if not api_key:
46:            return None
47:
48:        _alpha_vantage_limiter.wait()
49:        # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
50:        # because there is no public increment function exported from that module.
51:        # Known limitation — earnings fetches consume 1 AV call each but are not
52:        # reflected in the budget tracker.  Each ticker only fetches once per 24h.
53:        r = fetch_with_retry(
54:            "https://www.alphavantage.co/query",
55:            params={
56:                "function": "EARNINGS",
57:                "symbol": ticker,
58:                "apikey": api_key,
59:            },
60:            timeout=10,
61:        )
62:        if r is None:
63:            return None
64:        data = r.json()
65:
66:        # AV EARNINGS returns quarterlyEarnings and annualEarnings
67:        quarterly = data.get("quarterlyEarnings", [])
68:        if not quarterly:
69:            return None
70:
71:        today = datetime.now(UTC).date()
72:        # Find the next upcoming earnings (reportedDate in the future or very recent)
73:        for q in quarterly:
74:            rd = q.get("reportedDate")
75:            if not rd or rd == "None":
76:                continue
77:            try:
78:                from datetime import date as _date
79:                ed = _date.fromisoformat(rd)
80:                days_until = (ed - today).days
81:                if days_until >= -1:
82:                    return {
83:                        "earnings_date": ed.isoformat(),
84:                        "days_until": days_until,
85:                        "gate_active": 0 <= days_until <= GATE_DAYS,
86:                        "timing": "unknown",
87:                    }
88:            except (ValueError, TypeError):
89:                continue
90:
91:        return None
92:    except Exception:
93:        logger.debug("Alpha Vantage earnings fetch failed for %s", ticker, exc_info=True)
94:        return None
95:
96:
97:def _fetch_earnings_yfinance(ticker: str) -> dict | None:
98:    """Fallback: fetch next earnings date from yfinance."""
99:    try:
100:        import yfinance as yf
101:
102:        t = yf.Ticker(ticker)
103:        with suppress(Exception):
104:            cal = t.calendar
105:            if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
106:                if isinstance(cal, dict):
107:                    earnings_date = cal.get("Earnings Date")
108:                    if isinstance(earnings_date, list) and earnings_date:
109:                        earnings_date = earnings_date[0]
110:                else:
111:                    if "Earnings Date" in cal.index:
112:                        earnings_date = cal.loc["Earnings Date"].iloc[0]
113:                    else:
114:                        earnings_date = None
115:
116:                if earnings_date is not None:
117:                    if hasattr(earnings_date, "date"):
118:                        ed = earnings_date.date()
119:                    elif isinstance(earnings_date, str):
120:                        ed = datetime.fromisoformat(earnings_date).date()
121:                    else:
122:                        ed = None
123:
124:                    if ed:
125:                        today = datetime.now(UTC).date()
126:                        days_until = (ed - today).days
127:                        if days_until < -5:
128:                            return None
129:                        return {
130:                            "earnings_date": ed.isoformat(),
131:                            "days_until": days_until,
132:                            "gate_active": 0 <= days_until <= GATE_DAYS,
133:                            "timing": "unknown",
134:                        }
135:        return None
136:    except Exception:
137:        logger.debug("yfinance earnings fetch failed for %s", ticker, exc_info=True)
138:        return None
139:
140:
141:def _fetch_earnings_date(ticker: str) -> dict | None:
142:    """Fetch next earnings date — Alpha Vantage primary, yfinance fallback.
143:
144:    Returns dict with earnings_date, days_until, timing, or None.
145:    """
146:    # Primary: Alpha Vantage (already have API key + rate limiter)
147:    result = _fetch_earnings_alpha_vantage(ticker)
148:    if result:
149:        return result
150:
151:    # Fallback: yfinance
152:    return _fetch_earnings_yfinance(ticker)
153:
154:
155:def get_earnings_proximity(ticker: str) -> dict | None:
156:    """Get cached earnings proximity for a ticker.
157:
158:    Only works for STOCK_SYMBOLS — returns None for crypto/metals.
159:
160:    Returns:
161:        dict with earnings_date, days_until, gate_active, timing
162:        or None if no upcoming earnings found.
163:    """
164:    if ticker not in STOCK_SYMBOLS:
165:        return None
166:
167:    now = time.time()
168:    with _earnings_lock:
169:        cached = _earnings_cache.get(ticker)
170:        if cached and now - cached["time"] < EARNINGS_CACHE_TTL:
171:            return cached["data"]
172:
173:    # Fetch fresh
174:    data = _fetch_earnings_date(ticker)
175:
176:    with _earnings_lock:
177:        _earnings_cache[ticker] = {"data": data, "time": now}
178:
179:    return data
180:
181:
182:def should_gate_earnings(ticker: str) -> bool:
183:    """Check if ticker should be gated to HOLD due to nearby earnings.
184:
185:    Returns True if:
186:    - Ticker is a US stock (not crypto/metals)
187:    - Earnings are within GATE_DAYS calendar days
188:    """
189:    if ticker not in STOCK_SYMBOLS:
190:        return False
191:
192:    prox = get_earnings_proximity(ticker)
193:    if prox is None:
194:        return False
195:
196:    return prox.get("gate_active", False)
197:
198:
199:def get_all_earnings_proximity() -> dict:
200:    """Get earnings proximity for all stock tickers.
201:
202:    Returns dict keyed by ticker with proximity info.
203:    Used by reporting.py to enrich agent_summary.
204:    """
205:    result = {}
206:    for ticker in STOCK_SYMBOLS:
207:        prox = get_earnings_proximity(ticker)
208:        if prox:
209:            result[ticker] = prox
210:    return result
211:
212:
213:def clear_cache() -> None:
214:    """Clear the earnings cache (for testing)."""
215:    with _earnings_lock:
216:        _earnings_cache.clear()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1163ms:
1:"""Social media sentiment — Reddit headline fetcher.
2:
3:Uses Reddit's public JSON API, no authentication needed.
4:"""
5:
6:import json
7:import urllib.parse
8:import urllib.request
9:from datetime import UTC, datetime
10:
11:USER_AGENT = "finance-analyzer/1.0 (portfolio intelligence bot)"
12:
13:# (subreddit, dedicated) — dedicated: keep all posts; general: filter by keywords
14:TICKER_SUBREDDITS = {
15:    "BTC": [("Bitcoin", True), ("CryptoCurrency", False)],
16:    "ETH": [("ethereum", True), ("CryptoCurrency", False)],
17:    "PLTR": [("PLTR", True), ("wallstreetbets", False)],
18:    "NVDA": [("wallstreetbets", False), ("stocks", False)],
19:}
20:
21:TICKER_KEYWORDS = {
22:    "BTC": ["bitcoin", "btc"],
23:    "ETH": ["ethereum", "eth"],
24:    "PLTR": ["palantir", "pltr"],
25:    "NVDA": ["nvidia", "nvda"],
26:}
27:
28:
29:def _fetch_subreddit(sub, keywords, dedicated, per_sub):
30:    posts = []
31:    url = f"https://www.reddit.com/r/{sub}/hot.json?limit={per_sub + 5}&raw_json=1"
32:    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
33:    with urllib.request.urlopen(req, timeout=10) as resp:
34:        data = json.loads(resp.read())
35:    for child in data.get("data", {}).get("children", []):
36:        post = child.get("data", {})
37:        title = post.get("title", "").strip()
38:        if not title or post.get("stickied"):
39:            continue
40:        if not dedicated and not any(kw in title.lower() for kw in keywords):
41:            continue
42:        created = post.get("created_utc", 0)
43:        posts.append(
44:            {
45:                "title": title,
46:                "source": f"reddit/r/{sub}",
47:                "published": (
48:                    datetime.fromtimestamp(created, tz=UTC).isoformat()
49:                    if created
50:                    else datetime.now(UTC).isoformat()
51:                ),
52:                "score": post.get("score", 0),
53:                "num_comments": post.get("num_comments", 0),
54:            }
55:        )
56:    return posts
57:
58:
59:def _search_subreddit(sub, keywords, limit=10):
60:    query = urllib.parse.quote(" OR ".join(keywords))
61:    url = (
62:        f"https://www.reddit.com/r/{sub}/search.json"
63:        f"?q={query}&sort=new&restrict_sr=on&limit={limit}&raw_json=1"
64:    )
65:    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
66:    with urllib.request.urlopen(req, timeout=10) as resp:
67:        data = json.loads(resp.read())
68:    posts = []
69:    for child in data.get("data", {}).get("children", []):
70:        post = child.get("data", {})
71:        title = post.get("title", "").strip()
72:        if not title:
73:            continue
74:        created = post.get("created_utc", 0)
75:        posts.append(
76:            {
77:                "title": title,
78:                "source": f"reddit/r/{sub}",
79:                "published": (
80:                    datetime.fromtimestamp(created, tz=UTC).isoformat()
81:                    if created
82:                    else datetime.now(UTC).isoformat()
83:                ),
84:                "score": post.get("score", 0),
85:                "num_comments": post.get("num_comments", 0),
86:            }
87:        )
88:    return posts
89:
90:
91:def get_reddit_posts(ticker, limit=20):
92:    short = ticker.upper().replace("-USD", "")
93:    subreddits = TICKER_SUBREDDITS.get(short, [])
94:    keywords = TICKER_KEYWORDS.get(short, [short.lower()])
95:    if not subreddits:
96:        return []
97:
98:    posts = []
99:    seen = set()
100:    per_sub = max(5, limit // len(subreddits))
101:
102:    for sub, dedicated in subreddits:
103:        try:
104:            fetched = _fetch_subreddit(sub, keywords, dedicated, per_sub)
105:            for p in fetched:
106:                if p["title"] not in seen:
107:                    seen.add(p["title"])
108:                    posts.append(p)
109:        except Exception as e:
110:            print(f"    [Reddit r/{sub}] error: {e}")
111:
112:    # Fallback: if keyword filtering yielded nothing, try search
113:    if not posts:
114:        for sub, _dedicated in subreddits[:1]:
115:            try:
116:                fetched = _search_subreddit(sub, keywords, limit=per_sub)
117:                for p in fetched:
118:                    if p["title"] not in seen:
119:                        seen.add(p["title"])
120:                        posts.append(p)
121:            except Exception as e:
122:                print(f"    [Reddit search r/{sub}] error: {e}")
123:
124:    posts.sort(key=lambda p: p.get("score", 0), reverse=True)
125:    return posts[:limit]
126:
127:
128:if __name__ == "__main__":
129:    for ticker in ["BTC", "ETH", "PLTR", "NVDA"]:
130:        print(f"\n{'='*60}")
131:        print(f"  Reddit posts for {ticker}")
132:        print(f"{'='*60}")
133:        posts = get_reddit_posts(ticker)
134:        print(f"  Found {len(posts)} posts")
135:        for p in posts[:5]:
136:            score = p.get("score", 0)
137:            print(f"  [{score:>5}] {p['source']:>25}  {p['title'][:70]}")
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"MODEL|CryptoBERT|FinBERT|Trading-Hero|label_map|score|prob|neutral|tokenizer|AutoTokenizer|from_pretrained|fallback|predict|sentiment\" Q:\\finance-analyzer\\portfolio\\bert_sentiment.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"bert|CryptoBERT|FinBERT|Trading-Hero|score|normalized|normalize|sentiment|headline|reddit|newsapi|news_keywords|model|fallback|tokenizer|[-+]1|0\\.5|compound\" Q:\\finance-analyzer\\portfolio\\sentiment.py" in Q:\finance-analyzer
 succeeded in 707ms:
1:"""In-process BERT sentiment inference (CryptoBERT, Trading-Hero-LLM, FinBERT).
2-
3-2026-04-09 (fix/bert-inproc-gpu): this module replaces the subprocess.run
4:pattern that portfolio/sentiment.py was using to call three inference scripts
5-under /mnt/q/models/ — cryptobert_infer.py, trading_hero_infer.py, and
6-finbert_infer.py.
7-
8-Why this exists
9----------------
10:The old path spawned a fresh Python subprocess for every sentiment call, and
11-each subprocess had to:
12-  1. Start a Python interpreter (~500 ms)
13-  2. Import torch + transformers (~1.5-2.5 s)
14-  3. Cold-load the 125M BERT weights from disk (~1-3 s)
15-  4. Run inference on CPU (~100-300 ms per headline)
16-That was ~3-10 s per call, of which >90% was fixed spawn overhead. With 10
17-BERT calls per cycle (2 per ticker * 5 active tickers), we burned 30-100 s
18:per cycle just in BERT sentiment. And none of the three scripts ever called
19-`.to("cuda")` despite the main venv having `torch 2.6.0+cu124` with CUDA
20-available, so everything ran on CPU.
21-
22-The fix
23--------
24:Load each model lazily on first use, cache (tokenizer, model, device, lock)
25-in a module-level dict, move the model to CUDA if available, and run forward
26-passes directly in the caller's process. Subsequent calls skip the load and
27-just run inference (~5-20 ms per headline on GPU).
28-
29-Per-model threading.Lock serializes CUDA kernel launches because main.py's
30:ThreadPoolExecutor(8 workers) can call predict() concurrently. The lock is
31:per model so CryptoBERT and FinBERT can run in parallel, but two threads
32-asking for the same model serialize. That's fine — forward pass is ~20 ms
33-and the subprocess cold-load it replaces was ~3-10 s per thread.
34-
35-Fallback contract
36------------------
37:portfolio/sentiment.py wraps every call to this module in try/except and
38-falls back to the old subprocess path on any exception. That means: if torch
39-import fails, if a model cache dir is missing, if CUDA OOMs on the load,
40-we don't crash the main loop — we just lose the speedup for that call and
41-log a warning.
42-
43-VRAM budget
44------------
45-RTX 3080 10 GB. llama-server (ministral3 / qwen3 / fingpt) holds ~5 GB when
46-its current model is loaded. Three BERT models = ~1.5 GB resident. Total:
47-~6.5 GB under load. Kronos/Chronos uses its own gpu_gate and is not
48-co-resident with llama-server. Plenty of headroom.
49-
50-Not in scope
51-------------
52:- Retiring the 3 /mnt/q/models/*_infer.py scripts: they stay as the fallback
53-  path and as CLI debugging tools. Cost is zero (unchanged, not imported
54-  unless the subprocess path is triggered).
55-- GPU gate acquisition around the forward pass: BERT forward pass is ~20 ms
56-  and llama-server's phase runs post-ticker, so there's no co-residency
57:  conflict. If that changes, wrap predict() in a gpu_gate context.
58-- Eager load at module import: keeps import side-effect-free; first call
59-  absorbs the ~5-10 s cold-load cost.
60-"""
61-
62-from __future__ import annotations
63-
64-import logging
65-import os
66-import platform
67-import threading
68-from typing import Any
69-
70:logger = logging.getLogger("portfolio.bert_sentiment")
71-
72-
73-# --- Model configuration ---------------------------------------------------
74-#
75-# Each entry mirrors the hardcoded config of the matching subprocess script
76-# under /mnt/q/models/. Label maps use the same ordering as those scripts so
77-# the output shape is a drop-in replacement for _run_model's subprocess path.
78-
79:_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
80:    "CryptoBERT": {
81-        "hf_name": "ElKulako/cryptobert",
82-        "cache_dir_win": r"Q:\models\cryptobert",
83-        "cache_dir_linux": "/home/deck/models/cryptobert",
84-        "max_length": 128,
85:        # CryptoBERT's native labels: {0: Bearish, 1: Neutral, 2: Bullish}.
86:        # sentiment.py expects positive/negative/neutral, so map at read time.
87:        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
88-        "local_files_only": True,
89-    },
90:    "Trading-Hero-LLM": {
91:        "hf_name": "fuchenru/Trading-Hero-LLM",
92-        "cache_dir_win": r"Q:\models\trading-hero-llm",
93-        "cache_dir_linux": "/home/deck/models/trading-hero-llm",
94-        "max_length": 512,
95:        # Trading-Hero-LLM's labels: {0: neutral, 1: positive, 2: negative}.
96:        "label_map": {0: "neutral", 1: "positive", 2: "negative"},
97-        "local_files_only": True,
98-    },
99:    "FinBERT": {
100-        "hf_name": "ProsusAI/finbert",
101-        "cache_dir_win": r"Q:\models\finbert",
102-        "cache_dir_linux": "/home/deck/models/finbert",
103-        "max_length": 512,
104:        # ProsusAI/finbert's labels: {0: positive, 1: negative, 2: neutral}.
105-        # This matches the hardcoded LABELS list in finbert_infer.py.
106:        "label_map": {0: "positive", 1: "negative", 2: "neutral"},
107:        # FinBERT uses a snapshot dir layout (models--ProsusAI--finbert/snapshots/<hash>)
108-        # instead of the Hub cache layout, so we resolve the path differently.
109-        "local_files_only": False,
110-        "snapshot_subdir": "models--ProsusAI--finbert",
111-    },
112-}
113-
114-
115:# Module-level cache: model_name -> (tokenizer, model, device, lock).
116-# Populated lazily by _get_model. Protected by _init_lock during load.
117-_models: dict[str, tuple[Any, Any, str, threading.Lock]] = {}
118-_init_lock = threading.Lock()
119-
120-
121-def _resolve_cache_dir(config: dict) -> str:
122-    return config["cache_dir_win"] if platform.system() == "Windows" else config["cache_dir_linux"]
123-
124-
125-def _resolve_finbert_snapshot(cache_dir: str, subdir: str) -> str | None:
126:    """FinBERT is saved under cache_dir/models--ProsusAI--finbert/snapshots/<hash>/.
127-    Return the first snapshot path, or None if the layout doesn't match.
128-    """
129-    import glob
130-    snapshots = glob.glob(os.path.join(cache_dir, subdir, "snapshots", "*"))
131-    return snapshots[0] if snapshots else None
132-
133-
134-def _has_meta_tensor(model: Any) -> bool:
--
156-    try:
157-        import accelerate
158-        return getattr(accelerate, "__version__", "unknown")
159-    except ImportError:
160-        return None
161-
162-
163-def _model_load_kwargs(name: str, config: dict, cache_dir: str) -> tuple[str, dict]:
164:    """Resolve the from_pretrained() positional path + kwargs for `name`.
165-
166:    Returns (load_path, kwargs). FinBERT uses a snapshot path (no
167-    cache_dir/local_files_only); others use cache_dir + local_files_only.
168:    Falls back to the Hub name for FinBERT if no local snapshot exists.
169-
170-    Extracted 2026-05-04 (fix/bert-meta-tensor) so the same dispatch can
171-    be reused by the meta-tensor recovery retry path without duplicating
172:    the FinBERT-vs-others branching.
173-    """
174:    if name == "FinBERT":
175-        snapshot = _resolve_finbert_snapshot(cache_dir, config["snapshot_subdir"])
176-        if snapshot is not None:
177-            return snapshot, {}
178-        return config["hf_name"], {}
179-    return config["hf_name"], {
180-        "cache_dir": cache_dir,
181-        "local_files_only": config.get("local_files_only", False),
182-    }
183-
184-
185-def _load_model(name: str) -> tuple[Any, Any, str, threading.Lock]:
186:    """Load a BERT model + tokenizer. Called under _init_lock.
187-
188:    Returns (tokenizer, model, device, per_model_lock) where device is
189-    "cuda" or "cpu".
190-    """
191-    # Lazy import so the main loop doesn't pay the torch import cost at
192:    # startup if no ticker ever calls get_sentiment this run.
193-    import torch
194:    from transformers import AutoModelForSequenceClassification, AutoTokenizer
195-
196:    config = _MODEL_CONFIGS[name]
197-    cache_dir = _resolve_cache_dir(config)
198-    hf_name = config["hf_name"]
199-    load_path, load_kwargs = _model_load_kwargs(name, config, cache_dir)
200-
201:    if name == "FinBERT" and load_path == hf_name:
202-        logger.info("Loading BERT model %s via hub name %s (no local snapshot found)", name, hf_name)
203:    elif name == "FinBERT":
204-        logger.info("Loading BERT model %s from snapshot %s", name, load_path)
205-    else:
206-        logger.info("Loading BERT model %s from %s", name, cache_dir)
207-
208:    tokenizer = AutoTokenizer.from_pretrained(load_path, **{
209-        k: v for k, v in load_kwargs.items()
210-        if k in ("cache_dir", "local_files_only")
211-    })
212:    model = AutoModelForSequenceClassification.from_pretrained(load_path, **load_kwargs)
213-
214-    # 2026-05-04 (fix/bert-meta-tensor): defensive meta-tensor recovery.
215-    #
216-    # Race between Chronos's CUDA load and concurrent BERT loads (commit
217-    # 789cc91c, 2026-05-03 21:08, swapped Chronos/Kronos order so Chronos
218-    # now loads on the first ticker's forecast call concurrent with the
219:    # sentiment phase's BERT loads via main.py's ThreadPoolExecutor) can
220:    # leave some FinBERT weights on the `meta` device when accelerate's
221-    # lazy init interleaves with CUDA init on another thread. Without
222:    # this guard, predict-time forward passes silently fail per-text
223-    # ("Tensor on device meta is not on the expected device cpu!") and
224:    # the per-text fallback writes a zero-confidence neutral placeholder
225:    # for every headline, polluting sentiment_ab_log.jsonl until the
226-    # next process restart.
227-    #
228-    # Detection at load time + one retry with eager-init kwargs flips
229-    # this from silent A/B-log corruption into either a self-healed
230-    # cycle (warning + clean reload) or a loud RuntimeError that the
231:    # caller's _get_model() doesn't cache, so subsequent predict calls
232-    # try again from scratch instead of compounding the corruption.
233-    #
234-    # Cost: one is_meta walk over ~200 parameters per load (<1ms).
235-    # Triggered: only when accelerate's race actually leaves meta tensors,
236-    # which is rare and load-time-only — never during steady-state
237-    # inference.
238-    if _has_meta_tensor(model):
239-        logger.warning(
--
241-            "concurrent CUDA load); retrying with eager init",
242-            name,
243-        )
244-        eager_kwargs = {
245-            **load_kwargs,
246-            "torch_dtype": torch.float32,
247-            "low_cpu_mem_usage": False,
248-        }
249:        model = AutoModelForSequenceClassification.from_pretrained(
250-            load_path, **eager_kwargs,
251-        )
252-        if _has_meta_tensor(model):
253-            raise RuntimeError(
254-                f"BERT {name} still has meta tensors after retry "
255-                f"(accelerate version: {_accelerate_version() or 'not installed'}, "
256-                f"load_path={load_path!r})"
257-            )
--
260-    # uses the equivalent .train(False) spelling — same effect, and we
261-    # prefer it here because the other spelling collides with an unrelated
262-    # security-scanner false positive on a substring match.
263-    model.train(False)
264-
265-    # 2026-04-09 (hotfix): BERT models now stay on CPU by default.
266-    #
267-    # Initial deployment tried to move BERT models to CUDA for ~5-20x per-call
268:    # inference speedup, but that created a VRAM contention problem with
269-    # llama-server's model swap phase (LLM batch Phase 1/2/3). The budget:
270-    #   BERT (3 models) ~1.5 GB + Chronos-2 ~3.5 GB + llama-server 5 GB
271-    #   = ~10 GB = the entire RTX 3080 10GB budget, no margin.
272-    # With BERT + Chronos resident, llama-server's 5 GB finance-llama-8b load
273-    # was timing out / retrying for 200+ s per swap, making cycles LONGER
274-    # than the pre-migration subprocess baseline. See portfolio.log for the
275-    # 21:30 (262s) and 21:48 (429s) cycles on 2026-04-09.
276-    #
--
290-            model = model.to("cuda")
291-            device = "cuda"
292-            logger.info("BERT model %s moved to CUDA (BERT_SENTIMENT_USE_GPU=1)", name)
293-        except Exception as e:
294-            logger.warning("BERT model %s failed to move to CUDA, staying on CPU: %s", name, e)
295-    else:
296-        logger.info("BERT model %s staying on CPU (default, avoids VRAM contention with llama-server)", name)
297-
298:    return tokenizer, model, device, threading.Lock()
299-
300-
301-def _get_model(name: str) -> tuple[Any, Any, str, threading.Lock]:
302-    """Thread-safe lazy accessor. Loads the model on first call, returns the
303-    cached tuple on subsequent calls.
304-    """
305:    if name not in _MODEL_CONFIGS:
306:        raise KeyError(f"Unknown BERT model: {name!r}. Known: {list(_MODEL_CONFIGS)}")
307-
308-    # Fast path: already loaded. Avoid holding _init_lock during forward pass.
309-    entry = _models.get(name)
310-    if entry is not None:
311-        return entry
312-
313-    # Slow path: acquire init lock, double-check, load.
314-    with _init_lock:
315-        entry = _models.get(name)
316-        if entry is None:
317-            entry = _load_model(name)
318-            _models[name] = entry
319-        return entry
320-
321-
322:def predict(model_name: str, texts: list[str]) -> list[dict]:
323:    """Run BERT sentiment inference on a list of texts.
324-
325-    Returns a list of dicts matching the legacy subprocess output shape:
326:        [{"text": <str>, "sentiment": "positive"|"negative"|"neutral",
327:          "confidence": <float>, "scores": {"positive": .., "negative": ..,
328:          "neutral": ..}}, ...]
329-
330-    2026-04-09 (hotfix 2): uses BATCHED tokenize + forward pass. The three
331-    legacy subprocess scripts (cryptobert_infer.py / trading_hero_infer.py
332:    / finbert_infer.py) all pass the full text list to the tokenizer in
333-    one call, which gives one forward pass over a padded tensor instead
334-    of N sequential passes. On CPU the speedup is ~5-10x per call because
335-    the BERT kernel launch overhead is amortized across the batch.
336-
337:    If the batched path fails (e.g. OOM on a huge batch, or tokenizer
338-    edge case), we fall back to a per-text loop so the caller still gets
339-    one result per input. A final safety net emits a zero-confidence
340:    neutral placeholder if even the per-text path fails.
341-    """
342-    if not texts:
343-        return []
344-
345:    # Lazy torch import. If this fails, caller (sentiment.py) catches and
346-    # falls back to subprocess. Don't try to guard here - let the exception
347-    # propagate.
348-    import torch
349-
350:    tokenizer, model, device, lock = _get_model(model_name)
351:    config = _MODEL_CONFIGS[model_name]
352-    max_length = config["max_length"]
353:    label_map = config["label_map"]
354-
355-    # Hold the per-model lock for the whole batch. Batched forward pass
356-    # takes ~100-500 ms on CPU for N<=30 headlines, much better than the
357-    # N sequential passes the earlier version of this code did.
358-    with lock:
359-        try:
360:            return _predict_batched(
361:                texts, tokenizer, model, device, max_length, label_map, torch,
362-            )
363-        except Exception as e:
364-            logger.warning(
365:                "BERT %s batched predict failed, falling back to per-text loop: %s",
366-                model_name, e,
367-            )
368:            return _predict_per_text(
369:                texts, tokenizer, model, device, max_length, label_map, torch, model_name,
370-            )
371-
372-
373:def _predict_batched(texts, tokenizer, model, device, max_length, label_map, torch):
374:    """Single tokenizer + forward pass over the whole batch."""
375:    inputs = tokenizer(
376-        texts,
377-        return_tensors="pt",
378-        truncation=True,
379-        padding=True,
380-        max_length=max_length,
381-    )
382-    if device == "cuda":
383-        inputs = {k: v.to("cuda") for k, v in inputs.items()}
384-    with torch.no_grad():
385-        outputs = model(**inputs)
386:    probs = torch.softmax(outputs.logits, dim=1)  # shape [N, num_labels]
387-
388-    results: list[dict] = []
389:    num_labels = len(label_map)
390-    for i, text in enumerate(texts):
391:        row = probs[i]
392-        label_idx = int(torch.argmax(row).item())
393:        sentiment = label_map[label_idx]
394:        scores = {
395:            label_map[j]: float(row[j].item())
396-            for j in range(num_labels)
397-        }
398-        confidence = float(row[label_idx].item())
399-        results.append({
400-            "text": text[:100],
401:            "sentiment": sentiment,
402-            "confidence": confidence,
403:            "scores": scores,
404-        })
405-    return results
406-
407-
408:def _predict_per_text(texts, tokenizer, model, device, max_length, label_map, torch, model_name):
409-    """Fallback: one forward pass per text. Slower but more resilient to
410-    edge-case failures in the batched path (e.g. OOM on a huge batch or
411:    tokenizer error on one odd input).
412-    """
413-    results: list[dict] = []
414:    num_labels = len(label_map)
415-    for text in texts:
416-        try:
417:            inputs = tokenizer(
418-                text,
419-                return_tensors="pt",
420-                truncation=True,
421-                padding=True,
422-                max_length=max_length,
423-            )
424-            if device == "cuda":
425-                inputs = {k: v.to("cuda") for k, v in inputs.items()}
426-            with torch.no_grad():
427-                outputs = model(**inputs)
428:            probs = torch.softmax(outputs.logits, dim=1)[0]
429:            label_idx = int(torch.argmax(probs).item())
430:            sentiment = label_map[label_idx]
431:            scores = {
432:                label_map[i]: float(probs[i].item())
433-                for i in range(num_labels)
434-            }
435:            confidence = float(probs[label_idx].item())
436-            results.append({
437-                "text": text[:100],
438:                "sentiment": sentiment,
439-                "confidence": confidence,
440:                "scores": scores,
441-            })
442-        except Exception as e:
443-            logger.warning(
444:                "BERT %s per-text predict failed for %r: %s",
445-                model_name, text[:60], e,
446-            )
447-            results.append({
448-                "text": text[:100],
449:                "sentiment": "neutral",
450-                "confidence": 0.0,
451:                "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
452-            })
453-    return results
454-
455-
456-def available_models() -> list[str]:
457-    """Return the list of supported BERT model names (for tests + debugging)."""
458:    return list(_MODEL_CONFIGS)
459-
460-
461-def is_loaded(name: str) -> bool:
462-    """Check whether a model has been lazy-loaded yet (for tests)."""
463-    return name in _models
464-
465-
466-def _reset_for_tests() -> None:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 725ms:
1:"""Sentiment analysis — multi-model A/B testing with FinGPT, CryptoBERT, and TradingHero.
2-
3:Runs multiple sentiment models in shadow mode and logs results for comparison.
4:The primary (voting) model is the legacy model; shadow models are logged to
5:data/sentiment_ab_log.jsonl for accuracy tracking. Once a shadow model proves
6-superior (>60% on 200+ samples), it can be promoted.
7-
8-Models:
9:  - CryptoBERT: crypto headlines (legacy primary)
10:  - Trading-Hero-LLM: stock headlines (legacy primary)
11-  - FinGPT (Finance-Llama-8B / FinGPT-MT): GGUF via llama-cpp-python (shadow)
12:  - FinBERT: CPU fallback (shadow)
13-
14:Phase 3B: Cumulative headline clustering — groups related headlines and scores
15-them as a batch for richer "drumbeat effect" detection.
16-"""
17-
18-import json
19-import logging
20-import platform
21-import subprocess
22-import threading
23-from datetime import UTC, datetime
24-from pathlib import Path
25-
26-from portfolio.file_utils import atomic_append_jsonl
27-from portfolio.http_retry import fetch_json
28-
29:logger = logging.getLogger("portfolio.sentiment")
30-
31-CRYPTO_TICKERS = {"BTC", "ETH"}
32-
33-if platform.system() == "Windows":
34-    MODELS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
35:    CRYPTOBERT_SCRIPT = r"Q:\models\cryptobert_infer.py"
36:    TRADING_HERO_SCRIPT = r"Q:\models\trading_hero_infer.py"
37:    FINBERT_SCRIPT = r"Q:\models\finbert_infer.py"
38-else:
39:    MODELS_PYTHON = "/home/deck/models/.venv/bin/python"
40:    CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
41:    TRADING_HERO_SCRIPT = "/home/deck/models/trading_hero_infer.py"
42:    FINBERT_SCRIPT = "/home/deck/models/finbert_infer.py"
43-
44-DATA_DIR = Path(__file__).resolve().parent.parent / "data"
45:AB_LOG_FILE = DATA_DIR / "sentiment_ab_log.jsonl"
46-
47-TICKER_CATEGORIES = {
48-    "BTC": "BTC",
49-    "ETH": "ETH",
50-    "XAU": "GOLD",
51-    "XAG": "SILVER",
52-    "PLTR": "TECHNOLOGY",
53-    "NVDA": "TECHNOLOGY",
--
69-
70-CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
71-
72-
73-def _is_crypto(ticker):
74-    return ticker.upper().replace("-USD", "") in CRYPTO_TICKERS
75-
76-
77:def _fetch_crypto_headlines(ticker="BTC", limit=20, *, cryptocompare_api_key=None):
78-    category = TICKER_CATEGORIES.get(ticker.upper(), ticker.upper())
79-    url = f"{CRYPTOCOMPARE_URL}&categories={category}"
80-    headers = {"User-Agent": "Mozilla/5.0"}
81-    if cryptocompare_api_key:
82-        headers["Authorization"] = f"Apikey {cryptocompare_api_key}"
83-    data = fetch_json(
84-        url,
85-        headers=headers,
86-        timeout=15,
87:        label="crypto_headlines",
88-    )
89-    if data is None:
90:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
91-    if isinstance(data, dict) and data.get("Response") == "Error":
92-        logger.warning("[CryptoCompare] API error: %s", data.get("Message", "unknown"))
93:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
94-    raw = data.get("Data", [])
95-    articles = list(raw)[:limit] if isinstance(raw, list) else []
96-    parsed = [
97-        {
98-            "title": a["title"],
99-            "source": a.get("source", "unknown"),
100-            "published": datetime.fromtimestamp(
101-                a["published_on"], tz=UTC
102-            ).isoformat(),
103-        }
104-        for a in articles
105-    ]
106-    if not parsed:
107:        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
108-    return parsed
109-
110-
111:# Mapping from short crypto ticker to yfinance symbol for fallback
112-_CRYPTO_YFINANCE_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
113-
114-
115:def _fetch_crypto_headlines_yahoo_fallback(ticker, limit=20):
116:    """Fallback: fetch crypto headlines via yfinance when CryptoCompare fails."""
117-    yf_symbol = _CRYPTO_YFINANCE_MAP.get(ticker.upper())
118-    if not yf_symbol:
119-        return []
120-    try:
121:        articles = _fetch_yahoo_headlines(yf_symbol, limit=limit)
122-        if articles:
123:            logger.info("[CryptoCompare] fallback to Yahoo Finance for %s: %d articles",
124-                        ticker, len(articles))
125-        return articles
126-    except Exception as e:
127:        logger.debug("[Yahoo News] crypto fallback error for %s: %s", ticker, e)
128-        return []
129-
130-
131:def _fetch_yahoo_headlines(ticker, limit=10):
132-    import yfinance as yf
133-
134-    stock = yf.Ticker(ticker)
135-    news = stock.news or []
136-    articles = []
137-    for item in news[:limit]:
138-        content = item.get("content", item)
139-        title = content.get("title", "")
--
151-                "title": title,
152-                "source": source,
153-                "published": pub or datetime.now(UTC).isoformat(),
154-            }
155-        )
156-    return articles
157-
158-
159:def _fetch_newsapi_headlines(ticker, api_key, limit=10, query=None):
160:    """Fetch headlines from NewsAPI with optional custom search query."""
161-    search_q = query or ticker
162-    data = fetch_json(
163:        "https://newsapi.org/v2/everything",
164-        params={"q": search_q, "language": "en", "sortBy": "publishedAt",
165-                "pageSize": limit},
166-        headers={"User-Agent": "Mozilla/5.0", "X-Api-Key": api_key},
167-        timeout=15,
168:        label=f"newsapi:{ticker}",
169-    )
170-    if data is None:
171-        return []
172-    articles = data.get("articles", [])
173-    return [
174-        {
175-            "title": a.get("title", ""),
176-            "source": a.get("source", {}).get("name", "unknown"),
177-            "published": a.get("publishedAt", datetime.now(UTC).isoformat()),
178-        }
179-        for a in articles
180-        if a.get("title")
181-    ]
182-
183-
184:def _fetch_newsapi_with_tracking(ticker, api_key, limit=10, query=None):
185-    """Fetch from NewsAPI and track the call against daily quota.
186-
187-    H9/DC-R3-2: only count against budget when the fetch actually returned data
188-    (not on empty responses or errors), preventing spurious budget exhaustion.
189-    """
190:    from portfolio.shared_state import newsapi_track_call
191:    result = _fetch_newsapi_headlines(ticker, api_key, limit=limit, query=query)
192-    if result:  # only count against budget when we actually got data
193:        newsapi_track_call()
194-    return result
195-
196-
197:def _fetch_stock_headlines(ticker, newsapi_key=None, limit=20):
198:    """Fetch stock headlines. NewsAPI for priority tickers (metals), Yahoo for the rest."""
199-    from portfolio.shared_state import (
200-        _cached,
201:        newsapi_quota_ok,
202:        newsapi_search_query,
203:        newsapi_ttl_for_ticker,
204-    )
205-
206-    articles = []
207-
208-    # NewsAPI: only for priority tickers during active hours (metals get 20-min refresh)
209:    ttl = newsapi_ttl_for_ticker(ticker) if newsapi_key else None
210:    if ttl is not None and newsapi_key and newsapi_quota_ok():
211-        try:
212:            query = newsapi_search_query(ticker)
213:            cached_newsapi = _cached(
214:                f"newsapi_{ticker}",
215-                ttl,
216:                _fetch_newsapi_with_tracking,
217-                ticker,
218:                newsapi_key,
219-                limit,
220-                query,
221-            )
222:            if cached_newsapi:
223:                articles.extend(cached_newsapi)
224-        except Exception as e:
225-            logger.debug("[NewsAPI] error for %s: %s", ticker, e)
226-
227:    # Yahoo Finance: fallback for metals, primary for everything else
228-    if len(articles) < limit:
229-        try:
230-            remaining = max(0, limit - len(articles))
231-            if remaining > 0:
232:                yahoo_articles = _fetch_yahoo_headlines(ticker, limit=remaining)
233-                seen_titles = {a.get("title", "").lower() for a in articles}
234-                for ya in yahoo_articles:
235-                    if ya.get("title", "").lower() not in seen_titles:
236-                        articles.append(ya)
237-                        seen_titles.add(ya.get("title", "").lower())
238-        except Exception as e:
239-            logger.debug("[Yahoo News] error for %s: %s", ticker, e)
240-
241:    newsapi_count = len([a for a in articles if a.get("source", "") != "Yahoo Finance"])
242:    yahoo_count = len(articles) - newsapi_count
243-    if articles:
244-        logger.debug("[Headlines %s] %d NewsAPI + %d Yahoo = %d total",
245:                     ticker, newsapi_count, yahoo_count, len(articles))
246-
247-    return articles[:limit]
248-
249-
250:# 2026-04-09 (fix/bert-inproc-gpu): map subprocess script paths to in-process
251:# model names so _run_model can try the fast in-process path first and fall
252:# back to the old subprocess path on failure. See portfolio/bert_sentiment.py
253:# for the full rationale — short version: subprocess cold-load was ~3-10s per
254-# call, in-process on GPU is ~50-200ms per call, ~20-60x speedup with the
255-# same output shape.
256-_INPROC_BERT_MAP = {
257:    CRYPTOBERT_SCRIPT: "CryptoBERT",
258:    TRADING_HERO_SCRIPT: "Trading-Hero-LLM",
259:    FINBERT_SCRIPT: "FinBERT",
260-}
261-
262-
263:def _run_model(script, texts):
264:    """Run a sentiment model.
265-
266:    Tries the in-process BERT cache first (portfolio.bert_sentiment) because
267:    it avoids the ~3-10 s subprocess spawn + cold-load cost and runs on GPU
268-    if available. Falls back to the legacy subprocess pattern on any failure
269:    so the main loop stays up even if torch/transformers break or a model
270-    cache dir is missing.
271-    """
272:    model_name = _INPROC_BERT_MAP.get(script)
273:    if model_name is not None:
274-        try:
275:            from portfolio.bert_sentiment import predict as _bert_predict
276:            return _bert_predict(model_name, texts)
277-        except Exception as e:
278:            # Log once per (model, exception class) to keep the log clean if
279:            # we end up stuck on the subprocess fallback. sentiment.py already
280-            # has its own logger configured.
281-            logger.warning(
282-                "In-process BERT %s failed, falling back to subprocess: %s",
283:                model_name, e,
284-            )
285-
286-    # Legacy subprocess path (also used if script is not one of the three
287:    # known BERT models, though that doesn't happen today).
288-    proc = subprocess.run(
289-        [MODELS_PYTHON, script],
290-        input=json.dumps(texts),
291-        capture_output=True,
292-        text=True,
293-        timeout=120,
294-    )
295-    if proc.returncode != 0:
296-        raise RuntimeError(f"Model failed: {proc.stderr}")
297-    return json.loads(proc.stdout)
298-
299-
300-# ── Deferred fingpt A/B buffering ──────────────────────────────────────────
301:# Fingpt is a SHADOW sentiment signal — it never votes. Its output lands in
302:# data/sentiment_ab_log.jsonl alongside the primary model's vote (CryptoBERT
303:# for crypto, Trading-Hero-LLM for stocks) for accuracy comparison.
304-#
305-# Historical note (2026-04-09, feat/fingpt-in-llmbatch):
306-# - v1: inline subprocess per call, cold-loading the GGUF every time (70-90s
307-#   GPU lock holds, broke the cycle budget)
308-# - v2: warm NDJSON daemon (scripts/fingpt_daemon.py) on GPU full offload
309-#   (OOM'd with llama-server also resident)
310:# - v3: warm NDJSON daemon on CPU (60-150s/cycle inference, forced
311-#   _FINGPT_REQUEST_TIMEOUT_S 60→180 and _TICKER_POOL_TIMEOUT 120→500 hotfix
312-#   bumps; worked but ugly)
313-# - v4 (current): fingpt runs in portfolio.llm_batch as Phase 3 of the
314-#   shared llama_server rotation on port 8787, with full -ngl 99 GPU
315-#   offload like ministral3 and qwen3. Retires ~250 LOC of daemon +
316-#   client code.
317-#
318-# Because fingpt now runs post-cycle in a batched phase, its results arrive
319:# AFTER get_sentiment() has already returned to the signal engine. The
320:# primary model and FinBERT shadow are still computed inline, but their A/B
321:# log write is DEFERRED: get_sentiment() stashes the primary + finbert shadow
322:# + the raw headlines + the enqueued fingpt sub_keys into
323-# _pending_ab_entries[ab_key] and returns. Once flush_llm_batch() completes
324:# in main.py, sentiment.flush_ab_log() walks the pending entries, merges the
325-# batched fingpt results into each, and writes the final A/B log rows.
326-#
327:# This preserves the EXACT schema of sentiment_ab_log.jsonl that downstream
328:# accuracy tracking consumes: one row per get_sentiment() call, with a
329:# shadow[] array containing fingpt per-headline + cumulative + finbert.
330-#
331:# Known open issue: sentiment_ab_log.jsonl shows fingpt returning constant
332:# "neutral, 0.7 confidence" for every real headline — see the
333-# project_fingpt_parser_defaulting_neutral memory. That is a parser / prompt
334:# bug in /mnt/q/models/fingpt_infer.py, NOT a problem with this migration.
335-# Scheduled as the immediate follow-up after this PR merges.
336-
337-_pending_ab_entries: dict[str, dict] = {}
338-_pending_ab_lock = threading.Lock()
339-
340-
341-def _stash_ab_context(
342-    ab_key: str,
343-    ticker: str,
344-    primary_result: dict,
345-    all_articles: list[dict],
346-    diss_mult: float,
347-) -> None:
348-    """Store the inline portion of an A/B entry until the batched fingpt
349:    results arrive in flush_ab_log(). Called from get_sentiment().
350-
351-    Thread-safe — multiple ThreadPoolExecutor workers call this concurrently.
352-
353:    2026-04-28: cryptobert_shadow slot added; CryptoBERT was demoted from
354:    crypto primary to shadow. See get_sentiment docstring for rationale.
355-    """
356-    with _pending_ab_lock:
357-        _pending_ab_entries[ab_key] = {
358-            "ticker": ticker,
359-            "primary_result": primary_result,
360:            "finbert_shadow": None,  # filled in below by get_sentiment
361:            "cryptobert_shadow": None,  # filled in below for crypto tickers (2026-04-28)
362-            "all_articles": all_articles,
363-            "diss_mult": diss_mult,
364:            "fingpt_headlines_raw": None,  # filled in by Phase 3
365-            "fingpt_cumulatives_raw": {},  # sub_key → raw dict, filled in by Phase 3
366-        }
367-
368-
369:def _stash_finbert_shadow(ab_key: str, finbert_shadow: dict | None) -> None:
370:    """Attach the inline FinBERT shadow result to a pending A/B entry."""
371-    with _pending_ab_lock:
372-        entry = _pending_ab_entries.get(ab_key)
373-        if entry is not None:
374:            entry["finbert_shadow"] = finbert_shadow
375-
376-
377:def _stash_cryptobert_shadow(ab_key: str, cryptobert_shadow: dict | None) -> None:
378:    """Attach the inline CryptoBERT shadow result to a pending A/B entry.
379-
380:    Added 2026-04-28 when CryptoBERT was demoted from crypto primary to
381:    shadow. Symmetric with _stash_finbert_shadow.
382-    """
383-    with _pending_ab_lock:
384-        entry = _pending_ab_entries.get(ab_key)
385-        if entry is not None:
386:            entry["cryptobert_shadow"] = cryptobert_shadow
387-
388-
389-def _stash_fingpt_result(ab_key: str, sub_key: str, result) -> None:
390-    """Called from portfolio.llm_batch._flush_fingpt_phase with the parsed
391-    fingpt result for one (ab_key, sub_key) tuple.
392-
393:    sub_key is either "headlines" (result is a list of per-headline dicts)
394-    or "cumul:<N>" (result is a single cumulative dict).
395-    """
396-    with _pending_ab_lock:
397-        entry = _pending_ab_entries.get(ab_key)
398-        if entry is None:
399:            # get_sentiment was never called for this key this cycle — can
400:            # happen if enqueue_fingpt ran but the parent get_sentiment
401-            # raised before _stash_ab_context. Drop silently.
402-            return
403:        if sub_key == "headlines":
404:            entry["fingpt_headlines_raw"] = result
405-        elif sub_key.startswith("cumul:"):
406-            entry["fingpt_cumulatives_raw"][sub_key] = result
407-
408-
409-def flush_ab_log() -> None:
410-    """Walk _pending_ab_entries, merge batched fingpt results into shadow
411-    arrays, write one JSONL row per entry, and clear the buffer.
412-
--
426-
427-    if not entries_snapshot:
428-        return
429-
430-    for ab_key, entry in entries_snapshot.items():
431-        try:
432-            shadow: list[dict] = []
433-
434:            # Fingpt per-headline → aggregate via _aggregate_sentiments the
435-            # same way the old inline path did. If the raw list is missing
436-            # (server returned nothing), skip the entry silently.
437:            fingpt_raw = entry.get("fingpt_headlines_raw")
438-            if fingpt_raw:
439-                # Filter out None entries (per-prompt failures).
440-                usable = [r for r in fingpt_raw if r is not None]
441-                if usable:
442-                    try:
443:                        fg_overall, fg_avg = _aggregate_sentiments(
444-                            usable,
445:                            headlines=entry["all_articles"],
446-                            dissemination_mult=entry.get("diss_mult", 1.0),
447-                        )
448-                        shadow.append({
449:                            "model": usable[0].get("model", "fingpt:finance-llama-8b"),
450:                            "sentiment": fg_overall,
451-                            "confidence": round(fg_avg[fg_overall], 4),
452:                            "avg_scores": {k: round(v, 4) for k, v in fg_avg.items()},
453-                        })
454-                    except Exception:
455-                        logger.debug(
456:                            "fingpt headlines aggregation failed for %s", ab_key,
457-                            exc_info=True,
458-                        )
459-
460-            # Fingpt cumulative clusters → one shadow entry per cluster.
461-            for _sub_key in sorted(entry.get("fingpt_cumulatives_raw", {})):
462-                cum = entry["fingpt_cumulatives_raw"][_sub_key]
463-                if cum is None:
464-                    continue
465-                shadow.append({
466:                    "model": cum.get("model", "fingpt:cumulative"),
467:                    "sentiment": cum.get("sentiment", "neutral"),
468-                    "confidence": cum.get("confidence", 0.0),
469:                    "headline_count": cum.get("headline_count", 0),
470-                })
471-
472:            # FinBERT shadow (already aggregated inline during get_sentiment).
473:            finbert = entry.get("finbert_shadow")
474:            if finbert is not None:
475:                shadow.append(finbert)
476-
477:            # CryptoBERT shadow (added 2026-04-28 — was the primary; demoted
478-            # to shadow due to 99.1% neutral output on press-wire input).
479-            # Crypto tickers only; entry stays None for stocks.
480:            cryptobert = entry.get("cryptobert_shadow")
481:            if cryptobert is not None:
482:                shadow.append(cryptobert)
483-
484-            if shadow:
485-                _log_ab_result(entry["ticker"], entry["primary_result"], shadow)
486-        except Exception:
487-            logger.debug("flush_ab_log: entry %s failed", ab_key, exc_info=True)
488-
489-
490:def _run_finbert(texts):
491:    """Run FinBERT sentiment inference.
492-
493:    2026-04-09 (fix/bert-inproc-gpu): routes through _run_model so FinBERT
494:    also benefits from the in-process GPU cache. _run_model's _INPROC_BERT_MAP
495:    knows that FINBERT_SCRIPT -> "FinBERT" and will hit bert_sentiment.predict
496-    first, falling back to the old subprocess path on any exception.
497-    """
498:    return _run_model(FINBERT_SCRIPT, texts)
499-
500-
501-# ---------------------------------------------------------------------------
502-# Headline clustering (Phase 3B)
503-# ---------------------------------------------------------------------------
504-
505:def _cluster_headlines(articles):
506:    """Group headlines by keyword overlap and time proximity.
507-
508:    Clusters enable cumulative sentiment analysis — many mildly negative
509:    headlines about the same topic together signal stronger negativity
510-    than scoring each independently.
511-
512-    Returns:
513-        List of clusters, each a list of article dicts
514-    """
515-    if not articles or len(articles) < 3:
516-        return [articles] if articles else []
517-
518:    from portfolio.news_keywords import score_headline
519-
520:    # Extract keywords per headline
521:    headline_keywords = []
522-    for a in articles:
523:        _, matched = score_headline(a.get("title", ""))
524-        # Also extract significant words (>4 chars, not stopwords)
525-        words = set()
526-        for word in a.get("title", "").lower().split():
527-            clean = word.strip(".,!?;:'\"()[]")
528-            if len(clean) > 4 and clean not in _STOPWORDS:
529-                words.add(clean)
530-        words.update(kw.lower() for kw in matched)
531:        headline_keywords.append(words)
532-
533-    # Simple greedy clustering by keyword overlap
534-    clusters = []
535-    assigned = set()
536-
537-    for i in range(len(articles)):
538-        if i in assigned:
539-            continue
540-        cluster = [articles[i]]
541-        assigned.add(i)
542:        kw_i = headline_keywords[i]
543-
544-        for j in range(i + 1, len(articles)):
545-            if j in assigned:
546-                continue
547:            kw_j = headline_keywords[j]
548-            overlap = len(kw_i & kw_j)
549-            # Cluster if they share 2+ keywords or 1 matched keyword
550-            if overlap >= 2 or (overlap >= 1 and kw_i & kw_j & _SIGNIFICANT_KEYWORDS):
551-                cluster.append(articles[j])
552-                assigned.add(j)
553-
554-        clusters.append(cluster)
555-
--
570-    "bitcoin", "ethereum", "crypto", "nvidia", "semiconductor",
571-}
572-
573-
574-# ---------------------------------------------------------------------------
575-# Aggregation
576-# ---------------------------------------------------------------------------
577-
578:# Decisiveness margins added 2026-04-28. See docs/PLAN_sentiment_2026_04_28.md
579-# for the full rationale; tl;dr: the old aggregator returned the top label by
580-# 0.001 vs second, so a 0.34/0.33/0.33 split labeled "positive". Now we require
581-# a real margin before committing to a non-neutral verdict, and we default to
582:# label-majority over score-averaging so a few decisive headlines are not
583-# drowned by many tepid-neutral peers.
584-_DECISIVE_MARGIN_AVG = 0.05      # avg-mode: top-vs-second margin in prob units
585:_DECISIVE_MARGIN_PER_HEADLINE = 0.10  # majority-mode: per-headline label margin
586-_DECISIVE_MARGIN_MAJORITY = 1e-9  # majority-mode: top-vs-second weight margin
587-                                  # (zero-tolerance — exact ties go neutral)
588-
589-
590:def _compute_weights(sentiments, headlines, dissemination_mult):
591:    """Return per-sentiment weights from keyword scoring + dissemination."""
592:    if headlines and len(headlines) == len(sentiments):
593:        from portfolio.news_keywords import score_headline
594-        weights = []
595:        for h in headlines:
596-            title = h.get("title", "") if isinstance(h, dict) else str(h)
597:            w, _ = score_headline(title)
598-            weights.append(w * dissemination_mult)
599-    else:
600:        weights = [dissemination_mult] * len(sentiments)
601-    return weights
602-
603-
604:def _aggregate_sentiments(sentiments, headlines=None, dissemination_mult=1.0,
605-                           *, mode="majority"):
606:    """Aggregate sentiment scores into a single (label, avg_dict) verdict.
607-
608-    mode="majority" (default, 2026-04-28): label-majority vote.
609:        Each headline gets its own decisive label (top score must beat second
610-        by >=_DECISIVE_MARGIN_PER_HEADLINE, else "neutral"), then a weighted
611-        majority over those labels picks the verdict. Exact ties resolve to
612:        neutral. The returned avg_dict is still the score-weighted-average
613-        (kept identical for backward-compat with consumers that read
614:        avg_scores like sentiment_avg_scores in signal_engine.py:2452).
615-
616:    mode="average": legacy probability-averaging. Returns the top-scored
617-        label IF its margin over the second exceeds _DECISIVE_MARGIN_AVG;
618-        otherwise downgrades to "neutral". The pure-max-without-margin
619:        behavior was the source of the W16-W17 sentiment regression and is
620-        no longer reachable.
621-
622:    When headlines are provided, score weights from news_keywords.score_headline()
623-    amplify high-impact keywords (tariff/war/crash 3x). dissemination_mult
624-    multiplies all weights when news is widely cross-referenced.
625-    """
626:    if not sentiments:
627-        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
628-
629:    weights = _compute_weights(sentiments, headlines, dissemination_mult)
630-    total_w = sum(weights)
631-    if total_w == 0:
632-        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
633-
634:    pos_sum = sum(s["scores"]["positive"] * w for s, w in zip(sentiments, weights))
635:    neg_sum = sum(s["scores"]["negative"] * w for s, w in zip(sentiments, weights))
636:    neu_sum = sum(s["scores"]["neutral"] * w for s, w in zip(sentiments, weights))
637-    avg = {
638-        "positive": pos_sum / total_w,
639-        "negative": neg_sum / total_w,
640-        "neutral":  neu_sum / total_w,
641-    }
642-
643-    if mode == "majority":
644:        verdict = _majority_label(sentiments, weights)
645-        return verdict, avg
646-
647:    # mode == "average" — legacy threshold-augmented score-averaging
648-    overall = max(avg, key=avg.get)
649:    sorted_scores = sorted(avg.values(), reverse=True)
650:    margin = sorted_scores[0] - sorted_scores[1]
651-    if margin < _DECISIVE_MARGIN_AVG and overall != "neutral":
652-        return "neutral", avg
653-    return overall, avg
654-
655-
656:def _majority_label(sentiments, weights):
657:    """Per-headline decisive label, then weighted majority vote.
658-
659:    Each headline classified as positive/negative/neutral with its own per-
660:    headline margin gate. Weights are summed per label; winner returned only
661-    if it beats the second by more than _DECISIVE_MARGIN_MAJORITY.
662-    """
663-    bucket = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
664:    for s, w in zip(sentiments, weights):
665:        scores = s["scores"]
666:        # decisive per-headline label: top must beat second by margin
667:        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
668:        top_label, top_score = ordered[0]
669:        second_score = ordered[1][1]
670:        if (top_score - second_score) < _DECISIVE_MARGIN_PER_HEADLINE:
671-            top_label = "neutral"
672-        bucket[top_label] += w
673-
674-    ordered_buckets = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
675-    winner_label, winner_w = ordered_buckets[0]
676-    runner_w = ordered_buckets[1][1]
677-    if (winner_w - runner_w) <= _DECISIVE_MARGIN_MAJORITY:
678-        return "neutral"
679-    return winner_label
680-
681-
682:def _filter_relevant_headlines(articles, ticker, *, fallback_n=3):
683:    """Drop wire-noise headlines before model inference.
684-
685:    Uses news_keywords.is_relevant_headline (keyword OR ticker-synonym match)
686-    plus a credible-source-with-long-title escape hatch (Reuters/Bloomberg/
687-    etc. + title >= 25 chars covers in-depth coverage that doesn't happen to
688-    mention the ticker by name).
689-
690:    Falls back to most-recent `fallback_n` if the filter would drop
691-    everything — better to have noisy signal than silent signal on slow
692-    news days.
693-    """
694-    if not articles:
695-        return []
696-
697:    from portfolio.news_keywords import is_credible_source, is_relevant_headline
698-
699-    kept = []
700-    for a in articles:
701-        title = a.get("title", "") if isinstance(a, dict) else str(a)
702:        if is_relevant_headline(title, ticker):
703-            kept.append(a)
704-            continue
705-        # Credible-source escape hatch: long titles from credible outlets are
706-        # almost always real coverage worth scoring.
707-        source = a.get("source", "") if isinstance(a, dict) else ""
708-        if is_credible_source(source) and len(title.strip()) >= 25:
709-            kept.append(a)
710-
711-    if kept:
712-        return kept
713-
714:    # All-irrelevant fallback: keep the most-recent N articles. Sort by the
715-    # `published` field (ISO timestamp string sorts chronologically), most
716-    # recent first. Articles without `published` sort last via empty-string
717-    # default.
718-    sorted_articles = sorted(
719-        articles,
720-        key=lambda a: a.get("published", "") if isinstance(a, dict) else "",
721-        reverse=True,
722-    )
723:    return sorted_articles[:fallback_n]
724-
725-
726-def _log_ab_result(ticker, primary_result, shadow_results):
727:    """Log A/B test results to sentiment_ab_log.jsonl for accuracy comparison."""
728-    try:
729-        entry = {
730-            "ts": datetime.now(UTC).isoformat(),
731-            "ticker": ticker,
732-            "primary": {
733:                "model": primary_result.get("model", "unknown"),
734:                "sentiment": primary_result.get("overall_sentiment", "unknown"),
735-                "confidence": primary_result.get("confidence", 0.0),
736-            },
737-            "shadow": shadow_results,
738-        }
739-        atomic_append_jsonl(AB_LOG_FILE, entry)
740-    except Exception:
741-        logger.debug("Failed to log A/B result", exc_info=True)
742-
743-
744-# ---------------------------------------------------------------------------
745-# Main entry point
746-# ---------------------------------------------------------------------------
747-
748:def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None,
749-                   *, cryptocompare_api_key=None) -> dict:
750:    """Get sentiment for a ticker using primary model + shadow A/B models.
751-
752:    2026-04-28 (fix/sentiment-relevance-and-aggregation): two changes here.
753:      1. Crypto primary model swapped CryptoBERT -> Trading-Hero-LLM.
754:         CryptoBERT was 99.1% neutral on 2,817 wire-feed samples (it was
755:         trained on crypto-twitter slang, not press-wire headlines).
756:         CryptoBERT now runs as a shadow for continuity of the 30d accuracy
757:         baseline. Trading-Hero is permabull on financial news but at least
758-         produces variance; the 47% directional accuracy gate still acts as
759-         a circuit breaker if it underperforms.
760:      2. Headlines pass through _filter_relevant_headlines() before model
761-         inference. Bare price-tickers ("Bitcoin: $67,123") and generic
762-         market-update boilerplate are dropped. Fallback keeps the most-
763-         recent N when the filter would drop everything.
764-
765:    Primary model (votes):  Trading-Hero-LLM (both crypto and stocks)
766:    Shadow models (logged): CryptoBERT (crypto only), FinGPT, FinBERT
767-
768:    Returns the primary model's result. Shadow results are logged to
769:    data/sentiment_ab_log.jsonl for accuracy tracking.
770-    """
771-    short = ticker.upper().replace("-USD", "")
772-    is_crypto = _is_crypto(short)
773-
774-    if is_crypto:
775:        articles = _fetch_crypto_headlines(
776-            short, cryptocompare_api_key=cryptocompare_api_key,
777-        )
778-    else:
779:        articles = _fetch_stock_headlines(short, newsapi_key=newsapi_key)
780:    # 2026-04-28: Trading-Hero-LLM is the primary across all asset classes.
781:    # See module/function docstring for the CryptoBERT demotion rationale.
782:    model_script = TRADING_HERO_SCRIPT
783:    model_name = "Trading-Hero-LLM"
784-
785-    social = social_posts or []
786-    raw_all = articles + social
787-    sources = {
788-        "news": len(articles),
789:        "reddit": sum(1 for p in social if "reddit" in p.get("source", "")),
790-    }
791-
792-    if not raw_all:
793-        return {
794:            "overall_sentiment": "unknown",
795-            "confidence": 0.0,
796-            "num_articles": 0,
797:            "model": model_name,
798-            "sources": sources,
799-            "details": [],
800-        }
801-
802-    # Drop wire-noise before inference. The filter has a most-recent-N
803:    # fallback so we never go silent on slow-news days.
804:    all_articles = _filter_relevant_headlines(raw_all, short)
805-    titles = [a["title"] for a in all_articles]
806-
807:    # Compute dissemination score for weight amplification
808-    diss_mult = 1.0
809-    try:
810:        from portfolio.news_keywords import dissemination_score
811:        diss_mult = dissemination_score(all_articles)
812-    except Exception:
813:        logger.debug("Dissemination score failed, using default 1.0", exc_info=True)
814-
815:    # --- Primary model (votes in consensus) ---
816:    sentiments = _run_model(model_script, titles)
817:    overall, avg = _aggregate_sentiments(sentiments, headlines=all_articles,
818-                                         dissemination_mult=diss_mult)
819-
820-    details = []
821:    for article, sent in zip(all_articles, sentiments):
822-        details.append(
823-            {
824-                "title": article["title"],
825-                "source": article["source"],
826-                "published": article["published"],
827:                "sentiment": sent.get("sentiment") or sent.get("label", "unknown"),
828-                "confidence": sent["confidence"],
829:                "scores": sent["scores"],
830-            }
831-        )
832-
833-    primary_result = {
834:        "overall_sentiment": overall,
835-        "confidence": round(avg[overall], 4),
836:        "num_articles": len(sentiments),
837:        "avg_scores": {k: round(v, 4) for k, v in avg.items()},
838:        "model": model_name,
839-        "sources": sources,
840-        "details": details,
841:        "dissemination_score": diss_mult,
842-    }
843-
844:    # --- Shadow models (A/B testing — logged only, don't affect consensus) ---
845-    #
846-    # 2026-04-09: The A/B log write used to happen inline at the bottom of
847-    # this function. It is now DEFERRED to flush_ab_log() which is called
848-    # post-cycle from main.py after flush_llm_batch() completes Phase 3
849:    # (fingpt sentiment). Rationale: fingpt used to run in a bespoke NDJSON
850-    # daemon (scripts/fingpt_daemon.py, now retired) blocking inside this
851-    # function; moving fingpt into portfolio.llm_batch's shared llama_server
852-    # rotation means the fingpt result does not arrive until AFTER
853:    # get_sentiment() has returned. Rather than duplicate the A/B log entry
854:    # or block on the batch, we stash the primary + finbert + context here
855-    # and let flush_ab_log() assemble the final row.
856-    #
857:    # The primary model's voting result is still computed and returned
858-    # SYNCHRONOUSLY — batching only affects the shadow log, not the vote.
859-    ab_key = f"{short}:{datetime.now(UTC).isoformat()}"
860-    _stash_ab_context(ab_key, short, primary_result, all_articles, diss_mult)
861-
862-    # Shadow: FinGPT — enqueue for post-cycle Phase 3 execution. Zero-cost
863-    # here; the actual inference runs via llama_server finance-llama-8b
864-    # rotation after the ticker pool completes.
865-    #
866:    # 2026-04-10 (perf/llama-swap-reduction): gated by is_llm_on_cycle. Unlike
867-    # ministral/qwen3 (which go through _cached_or_enqueue's should_enqueue_fn),
868-    # fingpt enqueues directly because it doesn't use the signal cache — it
869-    # only writes to the A/B shadow log. When off-cycle, skip the enqueue
870:    # entirely so the llama_server phase 3 skips the fingpt model swap +
871-    # inference cost. Fingpt is a shadow signal, so skipping 2 of 3 cycles
872-    # just reduces A/B sample density from every cycle to every 3rd cycle,
873-    # which is fine for long-running statistical comparison.
874-    try:
875-        from portfolio.llm_batch import enqueue_fingpt, is_llm_on_cycle
876-        if is_llm_on_cycle("fingpt"):
877-            enqueue_fingpt(
878:                ab_key, "headlines",
879:                {"mode": "headlines", "texts": titles, "ticker": short},
880-            )
881:            clusters = _cluster_headlines(all_articles)
882-            for idx, cluster in enumerate(clusters):
883-                if len(cluster) >= 3:
884-                    cluster_titles = [a["title"] for a in cluster]
885-                    enqueue_fingpt(
886-                        ab_key, f"cumul:{idx}",
887-                        {"mode": "cumulative", "texts": cluster_titles, "ticker": short},
888-                    )
889-    except Exception as e:
890-        logger.debug("FinGPT enqueue failed: %s", e)
891-
892:    # Shadow: FinBERT (CPU, fast) — still runs inline because it's cheap
893:    # and on CPU (no model swap cost) and we'd rather not add a fourth
894-    # phase to llm_batch for an already-shadow-of-shadow signal. Stash its
895-    # aggregated entry into the pending A/B buffer so flush_ab_log sees it.
896-    try:
897:        finbert_results = _run_finbert(titles)
898:        if finbert_results:
899:            fb_overall, fb_avg = _aggregate_sentiments(
900:                finbert_results, headlines=all_articles, dissemination_mult=diss_mult
901-            )
902:            _stash_finbert_shadow(ab_key, {
903:                "model": "FinBERT",
904:                "sentiment": fb_overall,
905-                "confidence": round(fb_avg[fb_overall], 4),
906:                "avg_scores": {k: round(v, 4) for k, v in fb_avg.items()},
907-            })
908-    except Exception as e:
909:        logger.debug("FinBERT shadow failed: %s", e)
910-
911:    # Shadow: CryptoBERT — demoted from primary 2026-04-28. Kept as shadow
912-    # for crypto tickers only so we (a) preserve the 30d accuracy baseline
913:    # for comparison and (b) still notice if the model ever recovers from
914-    # its 99.1% neutral-output collapse. Stashed into the pending A/B
915:    # buffer; picked up by flush_ab_log alongside FinGPT and FinBERT.
916-    if is_crypto:
917-        try:
918:            crypto_results = _run_model(CRYPTOBERT_SCRIPT, titles)
919-            if crypto_results:
920:                cb_overall, cb_avg = _aggregate_sentiments(
921:                    crypto_results, headlines=all_articles, dissemination_mult=diss_mult,
922-                )
923:                _stash_cryptobert_shadow(ab_key, {
924:                    "model": "CryptoBERT",
925:                    "sentiment": cb_overall,
926-                    "confidence": round(cb_avg[cb_overall], 4),
927:                    "avg_scores": {k: round(v, 4) for k, v in cb_avg.items()},
928-                })
929-        except Exception as e:
930:            logger.debug("CryptoBERT shadow failed: %s", e)
931-
932-    return primary_result
933-
934-
935:def get_crypto_sentiment(ticker="BTC") -> dict:
936:    return get_sentiment(ticker)
937-
938-
939-if __name__ == "__main__":
940-    for ticker in ["BTC", "ETH"]:
941-        print(f"\n{'='*60}")
942-        print(f"  Sentiment for {ticker}")
943-        print(f"{'='*60}")
944:        result = get_sentiment(ticker)
945-        print(
946:            f"Overall: {result['overall_sentiment']} "
947-            f"(confidence: {result['confidence']:.2%})"
948-        )
949:        print(f"Model: {result['model']}")
950-        print(f"Articles analyzed: {result['num_articles']}")
951:        print(f"Dissemination score: {result.get('dissemination_score', 1.0):.2f}")
952:        if result.get("avg_scores"):
953:            s = result["avg_scores"]
954-            print(
955:                f"Avg scores: pos={s['positive']:.3f} neg={s['negative']:.3f} neu={s['neutral']:.3f}"
956-            )
957:        print("\nTop headlines:")
958-        for d in result["details"][:5]:
959-            emoji = {"positive": "+", "negative": "-", "neutral": "~"}.get(
960:                d["sentiment"], "?"
961-            )
962-            print(
963:                f"  [{emoji}] {d['sentiment']:>8} ({d['confidence']:.2%}) {d['title']}"
964-            )
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\data_collector.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\api_utils.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\futures_data.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\metals_orderbook.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\news_keywords.py" in Q:\finance-analyzer
 succeeded in 961ms:
1:"""Binance FAPI futures data — open interest, long/short ratios, funding history.
2:
3:Fetches public endpoints for crypto tickers (BTC-USD, ETH-USD only).
4:Uses existing infrastructure: fetch_json, _cached, _binance_limiter.
5:"""
6:
7:import logging
8:import time
9:
10:from portfolio.api_utils import BINANCE_FAPI_BASE, BINANCE_FUTURES_DATA
11:from portfolio.http_retry import fetch_json
12:from portfolio.shared_state import _binance_limiter, _cached
13:
14:logger = logging.getLogger("portfolio.futures_data")
15:
16:SYMBOL_MAP = {
17:    "BTC-USD": "BTCUSDT",
18:    "ETH-USD": "ETHUSDT",
19:}
20:
21:# Cache TTLs
22:_OI_TTL = 300        # 5 min
23:_LS_TTL = 300        # 5 min
24:_FUNDING_TTL = 900   # 15 min
25:
26:
27:def _fetch_json(url, params=None, timeout=10):
28:    """Fetch JSON from Binance FAPI with rate limiting and retry."""
29:    _binance_limiter.wait()
30:    return fetch_json(url, params=params, timeout=timeout, label="binance_fapi")
31:
32:
33:def get_open_interest(ticker):
34:    """Current open interest for a crypto ticker.
35:
36:    Returns: {oi, oi_usdt, symbol, time} or None.
37:    """
38:    if ticker not in SYMBOL_MAP:
39:        return None
40:    symbol = SYMBOL_MAP[ticker]
41:
42:    def _fetch():
43:        data = _fetch_json(
44:            f"{BINANCE_FAPI_BASE}/openInterest",
45:            params={"symbol": symbol},
46:        )
47:        if data is None:
48:            return None
49:        return {
50:            "oi": float(data["openInterest"]),
51:            "symbol": data["symbol"],
52:            "time": data.get("time", int(time.time() * 1000)),
53:        }
54:
55:    return _cached(f"futures_oi_{ticker}", _OI_TTL, _fetch)
56:
57:
58:def get_open_interest_history(ticker, period="5m", limit=30):
59:    """Historical open interest snapshots.
60:
61:    Returns: list of {oi, oi_usdt, timestamp} or None.
62:    """
63:    if ticker not in SYMBOL_MAP:
64:        return None
65:    symbol = SYMBOL_MAP[ticker]
66:
67:    def _fetch():
68:        data = _fetch_json(
69:            f"{BINANCE_FUTURES_DATA}/openInterestHist",
70:            params={"symbol": symbol, "period": period, "limit": limit},
71:        )
72:        if not data:
73:            return None
74:        return [
75:            {
76:                "oi": float(d["sumOpenInterest"]),
77:                "oi_usdt": float(d["sumOpenInterestValue"]),
78:                "timestamp": d["timestamp"],
79:            }
80:            for d in data
81:        ]
82:
83:    return _cached(f"futures_oi_hist_{ticker}_{period}", _OI_TTL, _fetch)
84:
85:
86:def get_long_short_ratio(ticker, period="5m", limit=30):
87:    """Global long/short account ratio.
88:
89:    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
90:    """
91:    if ticker not in SYMBOL_MAP:
92:        return None
93:    symbol = SYMBOL_MAP[ticker]
94:
95:    def _fetch():
96:        data = _fetch_json(
97:            f"{BINANCE_FUTURES_DATA}/globalLongShortAccountRatio",
98:            params={"symbol": symbol, "period": period, "limit": limit},
99:        )
100:        if not data:
101:            return None
102:        return [
103:            {
104:                "longShortRatio": float(d["longShortRatio"]),
105:                "longAccount": float(d["longAccount"]),
106:                "shortAccount": float(d["shortAccount"]),
107:                "timestamp": d["timestamp"],
108:            }
109:            for d in data
110:        ]
111:
112:    return _cached(f"futures_ls_{ticker}_{period}", _LS_TTL, _fetch)
113:
114:
115:def get_top_trader_position_ratio(ticker, period="5m", limit=30):
116:    """Top trader long/short position ratio.
117:
118:    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
119:    """
120:    if ticker not in SYMBOL_MAP:
121:        return None
122:    symbol = SYMBOL_MAP[ticker]
123:
124:    def _fetch():
125:        data = _fetch_json(
126:            f"{BINANCE_FUTURES_DATA}/topLongShortPositionRatio",
127:            params={"symbol": symbol, "period": period, "limit": limit},
128:        )
129:        if not data:
130:            return None
131:        return [
132:            {
133:                "longShortRatio": float(d["longShortRatio"]),
134:                "longAccount": float(d["longAccount"]),
135:                "shortAccount": float(d["shortAccount"]),
136:                "timestamp": d["timestamp"],
137:            }
138:            for d in data
139:        ]
140:
141:    return _cached(f"futures_top_pos_{ticker}_{period}", _LS_TTL, _fetch)
142:
143:
144:def get_top_trader_account_ratio(ticker, period="5m", limit=30):
145:    """Top trader long/short account ratio.
146:
147:    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
148:    """
149:    if ticker not in SYMBOL_MAP:
150:        return None
151:    symbol = SYMBOL_MAP[ticker]
152:
153:    def _fetch():
154:        data = _fetch_json(
155:            f"{BINANCE_FUTURES_DATA}/topLongShortAccountRatio",
156:            params={"symbol": symbol, "period": period, "limit": limit},
157:        )
158:        if not data:
159:            return None
160:        return [
161:            {
162:                "longShortRatio": float(d["longShortRatio"]),
163:                "longAccount": float(d["longAccount"]),
164:                "shortAccount": float(d["shortAccount"]),
165:                "timestamp": d["timestamp"],
166:            }
167:            for d in data
168:        ]
169:
170:    return _cached(f"futures_top_acct_{ticker}_{period}", _LS_TTL, _fetch)
171:
172:
173:def get_funding_rate_history(ticker, limit=100):
174:    """Historical funding rates.
175:
176:    Returns: list of {fundingRate, fundingTime, symbol} or None.
177:    """
178:    if ticker not in SYMBOL_MAP:
179:        return None
180:    symbol = SYMBOL_MAP[ticker]
181:
182:    def _fetch():
183:        data = _fetch_json(
184:            f"{BINANCE_FAPI_BASE}/fundingRate",
185:            params={"symbol": symbol, "limit": limit},
186:        )
187:        if not data:
188:            return None
189:        return [
190:            {
191:                "fundingRate": float(d["fundingRate"]),
192:                "fundingTime": d["fundingTime"],
193:                "symbol": d.get("symbol", symbol),
194:            }
195:            for d in data
196:        ]
197:
198:    return _cached(f"futures_funding_hist_{ticker}", _FUNDING_TTL, _fetch)
199:
200:
201:def get_all_futures_data(ticker):
202:    """Fetch all futures data for a ticker. Each sub-key can be None on failure.
203:
204:    Returns: dict with keys: open_interest, oi_history, ls_ratio,
205:             top_position_ratio, top_account_ratio, funding_history.
206:    """
207:    if ticker not in SYMBOL_MAP:
208:        return None
209:
210:    return {
211:        "open_interest": get_open_interest(ticker),
212:        "oi_history": get_open_interest_history(ticker),
213:        "ls_ratio": get_long_short_ratio(ticker),
214:        "top_position_ratio": get_top_trader_position_ratio(ticker),
215:        "top_account_ratio": get_top_trader_account_ratio(ticker),
216:        "funding_history": get_funding_rate_history(ticker),
217:    }
218:
219:
220:if __name__ == "__main__":
221:    import json
222:    for t in ["BTC-USD", "ETH-USD"]:
223:        print(f"\n=== {t} ===")
224:        result = get_all_futures_data(t)
225:        if result is None:
226:            print("  Not a crypto ticker — skipped")
227:        else:
228:            for k, v in result.items():
229:                if v is None:
230:                    print(f"  {k}: None (fetch failed)")
231:                elif isinstance(v, list):
232:                    print(f"  {k}: {len(v)} entries, latest={v[-1] if v else 'empty'}")
233:                else:
234:                    print(f"  {k}: {json.dumps(v, indent=2)}")
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 979ms:
1:"""Data collection — Binance, Alpaca, yfinance kline fetchers + multi-timeframe collector."""
2:
3:import logging
4:import time
5:from concurrent.futures import ThreadPoolExecutor, as_completed
6:from datetime import UTC, datetime
7:
8:import pandas as pd
9:
10:import portfolio.shared_state as _ss
11:from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, BINANCE_FAPI_BASE, get_alpaca_headers
12:from portfolio.circuit_breaker import CircuitBreaker
13:from portfolio.http_retry import fetch_with_retry
14:from portfolio.indicators import compute_indicators, technical_signal
15:
16:logger = logging.getLogger("portfolio.data_collector")
17:
18:# --- Circuit breakers for each data source ---
19:
20:binance_spot_cb = CircuitBreaker("binance_spot", failure_threshold=5, recovery_timeout=60)
21:binance_fapi_cb = CircuitBreaker("binance_fapi", failure_threshold=5, recovery_timeout=60)
22:alpaca_cb = CircuitBreaker("alpaca", failure_threshold=5, recovery_timeout=60)
23:
24:# BUG-179: Timeout for parallel timeframe fetches (seconds)
25:_TF_POOL_TIMEOUT = 60
26:ALPACA_INTERVAL_MAP = {
27:    "15m": ("15Min", 5),
28:    "1h": ("1Hour", 10),
29:    "1d": ("1Day", 365),
30:    "1w": ("1Week", 730),
31:    "1M": ("1Month", 1825),
32:}
33:
34:# yfinance interval mapping: our interval → (yf_interval, yf_period)
35:_YF_INTERVAL_MAP = {
36:    "15m": ("15m", "5d"),       # yfinance max for intraday <=60d
37:    "1h": ("1h", "30d"),
38:    "1d": ("1d", "365d"),
39:    "1w": ("1wk", "730d"),
40:    "1M": ("1mo", "1825d"),
41:}
42:
43:# Multi-timeframe analysis — (label, binance_interval, num_candles, cache_ttl_seconds)
44:TIMEFRAMES = [
45:    ("Now", "15m", 100, 0),  # ~25h data, refresh every cycle
46:    ("12h", "1h", 100, 300),  # ~4d data, cache 5min
47:    ("2d", "4h", 100, 900),  # ~17d data, cache 15min
48:    ("7d", "1d", 100, 3600),  # ~100d data, cache 1hr
49:    ("1mo", "3d", 100, 14400),  # ~300d data, cache 4hr
50:    ("3mo", "1w", 100, 43200),  # ~2yr data, cache 12hr
51:    ("6mo", "1M", 48, 86400),  # ~4yr data, cache 24hr
52:]
53:
54:STOCK_TIMEFRAMES = [
55:    ("Now", "15m", 100, 0),
56:    ("12h", "1h", 100, 300),
57:    ("2d", "1h", 48, 900),
58:    ("7d", "1d", 30, 3600),
59:    ("1mo", "1d", 100, 3600),
60:    ("3mo", "1w", 100, 43200),
61:    ("6mo", "1M", 48, 86400),
62:]
63:
64:
65:# --- Binance API ---
66:
67:_BINANCE_KLINE_COLS = [
68:    "open_time", "open", "high", "low", "close", "volume",
69:    "close_time", "quote_vol", "trades", "taker_buy_vol",
70:    "taker_buy_quote_vol", "ignore",
71:]
72:
73:
74:def _binance_fetch(base_url, cb, label, symbol, interval="5m", limit=100):
75:    """Shared Binance kline fetcher for spot and FAPI endpoints."""
76:    if not cb.allow_request():
77:        logger.warning("Binance %s circuit OPEN — skipping %s", label, symbol)
78:        raise ConnectionError(f"Binance {label} circuit open for {symbol}")
79:    try:
80:        r = fetch_with_retry(
81:            f"{base_url}/klines",
82:            params={"symbol": symbol, "interval": interval, "limit": limit},
83:            timeout=10,
84:        )
85:        if r is None:
86:            raise ConnectionError(f"Binance {label} klines request failed for {symbol}")
87:        r.raise_for_status()
88:        data = r.json()
89:        if not data:
90:            raise ConnectionError(
91:                f"Binance {label} returned empty data for {symbol} {interval}"
92:            )
93:        df = pd.DataFrame(data, columns=_BINANCE_KLINE_COLS)
94:        for col in ["open", "high", "low", "close", "volume"]:
95:            df[col] = df[col].astype(float)
96:        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
97:        cb.record_success()
98:        return df
99:    except Exception:
100:        cb.record_failure()
101:        raise
102:
103:
104:def binance_klines(symbol, interval="5m", limit=100):
105:    return _binance_fetch(BINANCE_BASE, binance_spot_cb, "spot", symbol, interval, limit)
106:
107:
108:def binance_fapi_klines(symbol, interval="5m", limit=100):
109:    """Fetch klines from Binance Futures API (for metals like XAUUSDT, XAGUSDT)."""
110:    return _binance_fetch(BINANCE_FAPI_BASE, binance_fapi_cb, "FAPI", symbol, interval, limit)
111:
112:
113:# --- Alpaca API ---
114:
115:
116:def alpaca_klines(ticker, interval="1d", limit=100):
117:    if interval not in ALPACA_INTERVAL_MAP:
118:        raise ValueError(f"Unsupported Alpaca interval: {interval}")
119:    if not alpaca_cb.allow_request():
120:        logger.warning("Alpaca circuit OPEN — skipping %s", ticker)
121:        raise ConnectionError(f"Alpaca circuit open for {ticker}")
122:    try:
123:        alpaca_tf, lookback_days = ALPACA_INTERVAL_MAP[interval]
124:        end = datetime.now(UTC)
125:        start = end - pd.Timedelta(days=lookback_days)
126:        r = fetch_with_retry(
127:            f"{ALPACA_BASE}/stocks/{ticker}/bars",
128:            headers=get_alpaca_headers(),
129:            params={
130:                "timeframe": alpaca_tf,
131:                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
132:                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
133:                "feed": "iex",
134:                "adjustment": "split",
135:            },
136:            timeout=10,
137:        )
138:        if r is None:
139:            raise ConnectionError(f"Alpaca request failed for {ticker}")
140:        r.raise_for_status()
141:        bars = r.json().get("bars") or []
142:        if not bars:
143:            raise ValueError(f"No Alpaca data for {ticker} interval={interval}")
144:        df = pd.DataFrame(bars)
145:        df = df.rename(
146:            columns={
147:                "o": "open",
148:                "h": "high",
149:                "l": "low",
150:                "c": "close",
151:                "v": "volume",
152:                "t": "time",
153:            }
154:        )
155:        for col in ["open", "high", "low", "close", "volume"]:
156:            df[col] = df[col].astype(float)
157:        df["time"] = pd.to_datetime(df["time"])
158:        alpaca_cb.record_success()
159:        return df.tail(limit)
160:    except Exception:
161:        alpaca_cb.record_failure()
162:        raise
163:
164:
165:# --- yfinance API ---
166:
167:
168:def fetch_vix():
169:    """Fetch current VIX level via yfinance. Returns dict or None."""
170:    try:
171:        import yfinance as yf
172:
173:        vix = yf.Ticker("^VIX")
174:        hist = vix.history(period="5d")
175:        if hist is None or hist.empty:
176:            return None
177:        # Flatten MultiIndex columns if present
178:        if isinstance(hist.columns, pd.MultiIndex):
179:            hist.columns = hist.columns.get_level_values(0)
180:        last = hist.iloc[-1]
181:        prev = hist.iloc[-2] if len(hist) > 1 else last
182:        current = float(last["Close"])
183:        prev_close = float(prev["Close"])
184:        change_pct = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0
185:
186:        # VIX regime classification
187:        if current >= 30:
188:            regime_hint = "high-vol"
189:        elif current >= 20:
190:            regime_hint = "elevated"
191:        elif current >= 15:
192:            regime_hint = "normal"
193:        else:
194:            regime_hint = "complacent"
195:
196:        return {
197:            "value": round(current, 2),
198:            "prev_close": round(prev_close, 2),
199:            "change_pct": round(change_pct, 2),
200:            "regime_hint": regime_hint,
201:        }
202:    except Exception as e:
203:        logger.warning("VIX fetch failed: %s", e)
204:        return None
205:
206:
207:def yfinance_klines(ticker, interval="1d", limit=100):
208:    """Fetch candles via yfinance with extended-hours data (prepost=True).
209:
210:    Returns a DataFrame matching alpaca_klines() format:
211:    columns: open, high, low, close, volume, time
212:    """
213:    import yfinance as yf
214:
215:    from portfolio.tickers import YF_MAP
216:
217:    yf_ticker = YF_MAP.get(ticker, ticker)
218:    if interval not in _YF_INTERVAL_MAP:
219:        raise ValueError(f"Unsupported yfinance interval: {interval}")
220:    yf_interval, yf_period = _YF_INTERVAL_MAP[interval]
221:
222:    df = yf.download(
223:        yf_ticker,
224:        period=yf_period,
225:        interval=yf_interval,
226:        prepost=True,
227:        progress=False,
228:        auto_adjust=True,
229:    )
230:    if df is None or df.empty:
231:        raise ValueError(f"No yfinance data for {yf_ticker} interval={interval}")
232:
233:    # yfinance returns MultiIndex columns when downloading single ticker too
234:    # (e.g. ('Close', 'NVDA')); flatten them
235:    if isinstance(df.columns, pd.MultiIndex):
236:        df.columns = df.columns.get_level_values(0)
237:
238:    df = df.rename(columns={
239:        "Open": "open", "High": "high", "Low": "low",
240:        "Close": "close", "Volume": "volume",
241:    })
242:    for col in ["open", "high", "low", "close", "volume"]:
243:        if col in df.columns:
244:            df[col] = df[col].astype(float)
245:    df["time"] = df.index
246:    df = df.reset_index(drop=True)
247:    return df.tail(limit)
248:
249:
250:# --- Kline dispatcher ---
251:
252:
253:def _fetch_klines(source, interval, limit):
254:    if "binance_fapi" in source:
255:        _ss._binance_limiter.wait()
256:        return binance_fapi_klines(source["binance_fapi"], interval=interval, limit=limit)
257:    elif "binance" in source:
258:        _ss._binance_limiter.wait()
259:        return binance_klines(source["binance"], interval=interval, limit=limit)
260:    elif "alpaca" in source:
261:        ticker = source["alpaca"]
262:        if _ss._current_market_state in ("closed", "weekend", "holiday"):
263:            logger.debug("%s: using yfinance (market %s)", ticker, _ss._current_market_state)
264:            _ss._yfinance_limiter.wait()
265:            return yfinance_klines(ticker, interval=interval, limit=limit)
266:        _ss._alpaca_limiter.wait()
267:        return alpaca_klines(ticker, interval=interval, limit=limit)
268:    raise ValueError(f"Unknown source: {source}")
269:
270:
271:# --- Multi-timeframe collector ---
272:
273:
274:# yfinance is not thread-safe; serialize calls with a shared lock.
275:# H11/DC-R3-4: use the module-level lock from shared_state so all modules
276:# (fear_greed, golddigger/data_provider, data_collector) share one lock.
277:from portfolio.shared_state import yfinance_lock as _yfinance_lock
278:
279:
280:def _fetch_one_timeframe(source, source_key, label, interval, limit, ttl):
281:    """Fetch and process a single timeframe. Thread-safe."""
282:    cache_key = f"tf_{source_key}_{label}"
283:    if ttl > 0:
284:        with _ss._cache_lock:
285:            cached = _ss._tool_cache.get(cache_key)
286:            if cached and time.time() - cached["time"] < ttl:
287:                return (label, cached["data"])
288:    try:
289:        # yfinance is not thread-safe — serialize its calls
290:        if "alpaca" in source and _ss._current_market_state in ("closed", "weekend", "holiday"):
291:            with _yfinance_lock:
292:                df = _fetch_klines(source, interval, limit)
293:        else:
294:            df = _fetch_klines(source, interval, limit)
295:        ind = compute_indicators(df)
296:        if ind is None:
297:            logger.debug("%s/%s: insufficient data (%d rows), skipping",
298:                         source_key, label, len(df) if df is not None else 0)
299:            return None
300:        if label == "Now":
301:            action, conf = None, None
302:        else:
303:            action, conf = technical_signal(ind)
304:        entry = {"indicators": ind, "action": action, "confidence": conf}
305:        if label == "Now":
306:            entry["_df"] = df  # preserve raw DataFrame for enhanced signals
307:        if ttl > 0:
308:            with _ss._cache_lock:
309:                _ss._tool_cache[cache_key] = {"data": entry, "time": time.time()}
310:        return (label, entry)
311:    except Exception as e:
312:        return (label, {"error": str(e)})
313:
314:
315:def collect_timeframes(source):
316:    """Collect all timeframes for a source, fetching in parallel."""
317:    is_stock = "alpaca" in source
318:    tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
319:    source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")
320:
321:    # BUG-179: Submit all timeframe fetches with timeout to prevent hangs
322:    with ThreadPoolExecutor(max_workers=len(tfs), thread_name_prefix=f"tf_{source_key}") as pool:
323:        futures = {
324:            pool.submit(_fetch_one_timeframe, source, source_key, label, interval, limit, ttl): label
325:            for label, interval, limit, ttl in tfs
326:        }
327:        raw_results = []
328:        try:
329:            for future in as_completed(futures, timeout=_TF_POOL_TIMEOUT):
330:                result = future.result()
331:                if result is not None:
332:                    raw_results.append(result)
333:        except TimeoutError:
334:            stuck = [lbl for f, lbl in futures.items() if not f.done()]
335:            logger.error("BUG-179: Timeframe pool timeout for %s. Stuck: %s",
336:                         source_key, stuck)
337:            for f in futures:
338:                f.cancel()
339:
340:    # Maintain original timeframe order
341:    tf_order = {label: i for i, (label, _, _, _) in enumerate(tfs)}
342:    raw_results.sort(key=lambda x: tf_order.get(x[0], 999))
343:    return raw_results
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 996ms:
1:"""Shared API utilities for finance-analyzer."""
2:
3:import json
4:import pathlib
5:import threading
6:
7:BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
8:
9:# Canonical API endpoint URLs — import these instead of hardcoding
10:BINANCE_BASE = "https://api.binance.com/api/v3"
11:BINANCE_FAPI_BASE = "https://fapi.binance.com/fapi/v1"
12:BINANCE_FUTURES_DATA = "https://fapi.binance.com/futures/data"
13:ALPACA_BASE = "https://data.alpaca.markets/v2"
14:
15:# Config cache (thread-safe)
16:_config_lock = threading.Lock()
17:_config_cache = None
18:_config_mtime = 0.0
19:
20:
21:def load_config():
22:    """Load config.json with mtime-based caching."""
23:    global _config_cache, _config_mtime
24:    config_path = BASE_DIR / "config.json"
25:
26:    with _config_lock:
27:        try:
28:            mtime = config_path.stat().st_mtime
29:            if _config_cache is None or mtime != _config_mtime:
30:                with open(config_path, encoding="utf-8") as f:
31:                    _config_cache = json.load(f)
32:                _config_mtime = mtime
33:        except Exception:
34:            if _config_cache is None:
35:                raise
36:    return _config_cache
37:
38:
39:def get_alpaca_headers():
40:    """Get Alpaca API authentication headers."""
41:    config = load_config()
42:    alpaca = config.get("alpaca", {})
43:    return {
44:        "APCA-API-KEY-ID": alpaca.get("key", ""),
45:        "APCA-API-SECRET-KEY": alpaca.get("secret", ""),
46:    }
47:
48:
49:def get_telegram_config():
50:    """Get Telegram bot token and chat ID."""
51:    config = load_config()
52:    tg = config.get("telegram", {})
53:    return tg.get("token", ""), tg.get("chat_id", "")
54:
55:
56:def get_binance_config():
57:    """Get Binance API credentials."""
58:    config = load_config()
59:    ex = config.get("exchange", {})
60:    return ex.get("apiKey", ""), ex.get("secret", "")
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 989ms:
1:"""Binance FAPI order book depth and recent trades for metals.
2:
3:Fetches L2 snapshots and trade ticks for XAUUSDT / XAGUSDT.
4:Data feeds into microstructure feature computations (OFI, depth
5:imbalance, VPIN, spread metrics).
6:
7:Uses the same rate-limiting and caching patterns as futures_data.py.
8:"""
9:from __future__ import annotations
10:
11:import logging
12:import time
13:from functools import wraps
14:
15:from portfolio.api_utils import BINANCE_FAPI_BASE
16:from portfolio.http_retry import fetch_json
17:from portfolio.shared_state import _binance_limiter, _cached
18:
19:logger = logging.getLogger("portfolio.metals_orderbook")
20:
21:SYMBOL_MAP = {
22:    "XAU-USD": "XAUUSDT",
23:    "XAG-USD": "XAGUSDT",
24:    "BTC-USD": "BTCUSDT",
25:    "ETH-USD": "ETHUSDT",
26:}
27:
28:_DEPTH_TTL = 10
29:_TRADES_TTL = 10
30:
31:
32:def _fetch_fapi_json(url, params=None, timeout=10):
33:    """Fetch JSON from Binance FAPI with rate limiting and retry."""
34:    _binance_limiter.wait()
35:    return fetch_json(url, params=params, timeout=timeout, label="metals_orderbook")
36:
37:
38:def _nocache(func):
39:    """Mark a function so tests can call func.__wrapped__ to bypass _cached."""
40:    @wraps(func)
41:    def wrapper(*args, **kwargs):
42:        return func(*args, **kwargs)
43:    wrapper.__wrapped__ = func
44:    return wrapper
45:
46:
47:@_nocache
48:def get_orderbook_depth(ticker: str, limit: int = 20) -> dict | None:
49:    """Fetch order book depth snapshot from Binance FAPI.
50:
51:    Returns dict with bids, asks (as [[price, qty], ...] floats), best_bid, best_ask,
52:    mid_price, spread, spread_bps. None on failure.
53:    """
54:    if ticker not in SYMBOL_MAP:
55:        return None
56:    symbol = SYMBOL_MAP[ticker]
57:
58:    def _fetch():
59:        data = _fetch_fapi_json(
60:            f"{BINANCE_FAPI_BASE}/depth",
61:            params={"symbol": symbol, "limit": limit},
62:        )
63:        if data is None or "bids" not in data or "asks" not in data:
64:            return None
65:        bids = [[float(p), float(q)] for p, q in data["bids"]]
66:        asks = [[float(p), float(q)] for p, q in data["asks"]]
67:        if not bids or not asks:
68:            return None
69:        best_bid = bids[0][0]
70:        best_ask = asks[0][0]
71:        mid = (best_bid + best_ask) / 2.0
72:        spread = best_ask - best_bid
73:        return {
74:            "symbol": symbol,
75:            "bids": bids,
76:            "asks": asks,
77:            "best_bid": best_bid,
78:            "best_ask": best_ask,
79:            "mid_price": mid,
80:            "spread": spread,
81:            "spread_bps": (spread / mid) * 10000 if mid > 0 else 0.0,
82:            "bid_depth_total": sum(q for _, q in bids),
83:            "ask_depth_total": sum(q for _, q in asks),
84:            "ts": int(time.time() * 1000),
85:        }
86:
87:    return _cached(f"depth_{ticker}_{limit}", _DEPTH_TTL, _fetch)
88:
89:
90:@_nocache
91:def get_recent_trades(ticker: str, limit: int = 100) -> list[dict] | None:
92:    """Fetch recent trades from Binance FAPI.
93:
94:    Each trade includes a sign: +1 for buyer-initiated (taker buy),
95:    -1 for seller-initiated (taker sell). isBuyerMaker=True means the maker
96:    was the buyer, so the taker (aggressor) was the seller -> sign = -1.
97:    """
98:    if ticker not in SYMBOL_MAP:
99:        return None
100:    symbol = SYMBOL_MAP[ticker]
101:
102:    def _fetch():
103:        data = _fetch_fapi_json(
104:            f"{BINANCE_FAPI_BASE}/trades",
105:            params={"symbol": symbol, "limit": limit},
106:        )
107:        if not data:
108:            return None
109:        return [
110:            {
111:                "id": d["id"],
112:                "price": float(d["price"]),
113:                "qty": float(d["qty"]),
114:                "time": d["time"],
115:                "is_buyer_maker": d.get("isBuyerMaker", False),
116:                "sign": -1 if d.get("isBuyerMaker", False) else 1,
117:            }
118:            for d in data
119:        ]
120:
121:    return _cached(f"trades_{ticker}_{limit}", _TRADES_TTL, _fetch)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 992ms:
1:"""Shared keyword dictionaries and sector mappings for news/event signals.
2:
3:Provides:
4:- Tiered keyword lists with severity weights
5:- headline scoring function
6:- Sector-to-ticker and ticker-to-sector mappings
7:"""
8:
9:from __future__ import annotations
10:
11:import re
12:from datetime import UTC
13:
14:# ---------------------------------------------------------------------------
15:# Keyword tiers: word/phrase -> weight multiplier for sentiment scoring
16:# ---------------------------------------------------------------------------
17:
18:CRITICAL_KEYWORDS = {
19:    "war": 3.0,
20:    "crash": 3.0,
21:    "tariff": 3.0,
22:    "tariffs": 3.0,
23:    "sanctions": 3.0,
24:    "sanction": 3.0,
25:    "hack": 3.0,
26:    "hacked": 3.0,
27:    "default": 3.0,
28:    "bankruptcy": 3.0,
29:    "bankrupt": 3.0,
30:    "collapse": 3.0,
31:    "ban": 3.0,
32:    "banned": 3.0,
33:    "invasion": 3.0,
34:    "nuclear": 3.0,
35:}
36:
37:HIGH_KEYWORDS = {
38:    "rate hike": 2.0,
39:    "rate cut": 2.0,
40:    "cpi": 2.0,
41:    "inflation": 2.0,
42:    "recession": 2.0,
43:    "delisting": 2.0,
44:    "delisted": 2.0,
45:    "lawsuit": 2.0,
46:    "indictment": 2.0,
47:    "sec investigation": 2.0,
48:    "trade war": 2.0,
49:    "downgrade": 2.0,
50:    "debt ceiling": 2.0,
51:    "margin call": 2.0,
52:    "layoffs": 2.0,
53:    "fraud": 2.0,
54:    "subpoena": 2.0,
55:}
56:
57:MODERATE_KEYWORDS = {
58:    "etf approval": 1.5,
59:    "etf approved": 1.5,
60:    "regulation": 1.5,
61:    "regulatory": 1.5,
62:    "liquidation": 1.5,
63:    "liquidated": 1.5,
64:    "upgrade": 1.5,
65:    "earnings miss": 1.5,
66:    "earnings beat": 1.5,
67:    "guidance cut": 1.5,
68:    "guidance raise": 1.5,
69:    "buyback": 1.5,
70:    "stock split": 1.5,
71:    "merger": 1.5,
72:    "acquisition": 1.5,
73:    "ipo": 1.5,
74:}
75:
76:# Combined dict for fast lookup (longer phrases checked first)
77:ALL_KEYWORDS = {**MODERATE_KEYWORDS, **HIGH_KEYWORDS, **CRITICAL_KEYWORDS}
78:
79:# Pre-compile patterns sorted by length descending (match longer phrases first)
80:_KEYWORD_PATTERNS = [
81:    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), weight)
82:    for kw, weight in sorted(ALL_KEYWORDS.items(), key=lambda x: -len(x[0]))
83:]
84:
85:# ---------------------------------------------------------------------------
86:# Sector mappings
87:# ---------------------------------------------------------------------------
88:
89:SECTOR_MAP = {
90:    "semiconductor": {"NVDA", "AMD", "MU", "AVGO", "TSM", "SMCI"},
91:    "crypto": {"BTC-USD", "ETH-USD", "MSTR"},
92:    "defense": {"LMT"},
93:    "big_tech": {"GOOGL", "AMZN", "AAPL", "META"},
94:    "ai": {"NVDA", "AMD", "GOOGL", "META", "PLTR", "SOUN", "AVGO"},
95:    "software": {"PLTR"},
96:    "gaming": {"TTWO"},
97:    "infrastructure": {"VRT"},
98:    "metals": {"XAU-USD", "XAG-USD"},
99:}
100:
101:# Reverse map: ticker -> set of sectors
102:TICKER_SECTORS: dict[str, set[str]] = {}
103:for _sector, _tickers in SECTOR_MAP.items():
104:    for _t in _tickers:
105:        TICKER_SECTORS.setdefault(_t, set()).add(_sector)
106:
107:# Keywords that imply directional impact on specific sectors
108:KEYWORD_SECTOR_IMPACT = {
109:    "tariff": {"semiconductor": "SELL", "metals": "BUY"},
110:    "tariffs": {"semiconductor": "SELL", "metals": "BUY"},
111:    "trade war": {"semiconductor": "SELL", "metals": "BUY"},
112:    "sanctions": {"crypto": "SELL"},
113:    "sanction": {"crypto": "SELL"},
114:    "ban": {"crypto": "SELL"},
115:    "banned": {"crypto": "SELL"},
116:    "hack": {"crypto": "SELL"},
117:    "hacked": {"crypto": "SELL"},
118:    "rate hike": {"big_tech": "SELL", "crypto": "SELL", "metals": "SELL"},
119:    "rate cut": {"big_tech": "BUY", "crypto": "BUY", "metals": "BUY"},
120:    "recession": {"big_tech": "SELL", "defense": "BUY"},
121:    "inflation": {"metals": "BUY", "crypto": "BUY"},
122:    "etf approval": {"crypto": "BUY"},
123:    "etf approved": {"crypto": "BUY"},
124:    "regulation": {"crypto": "SELL"},
125:    "regulatory": {"crypto": "SELL"},
126:    "war": {"defense": "BUY", "metals": "BUY", "big_tech": "SELL"},
127:    "invasion": {"defense": "BUY", "metals": "BUY", "big_tech": "SELL"},
128:}
129:
130:# Credible financial news sources (weight 1.5x)
131:CREDIBLE_SOURCES = {
132:    "reuters", "bloomberg", "wsj", "wall street journal",
133:    "cnbc", "associated press", "ap", "financial times", "ft",
134:    "bbc", "new york times", "nyt", "the economist",
135:    "marketwatch", "barron's", "barrons",
136:}
137:
138:
139:def score_headline(title: str) -> tuple[float, list[str]]:
140:    """Score a headline by keyword severity.
141:
142:    Returns:
143:        (max_weight, matched_keywords) — max_weight is the highest keyword
144:        weight found (1.0 if no keywords match), matched_keywords lists all
145:        matched keyword strings.
146:    """
147:    if not title:
148:        return 1.0, []
149:
150:    matched = []
151:    max_weight = 1.0
152:
153:    for pattern, weight in _KEYWORD_PATTERNS:
154:        if pattern.search(title):
155:            matched.append(pattern.pattern.replace(r"\b", "").replace("\\", ""))
156:            if weight > max_weight:
157:                max_weight = weight
158:
159:    return max_weight, matched
160:
161:
162:def keyword_severity(title: str) -> str:
163:    """Classify headline severity: critical, high, moderate, or normal."""
164:    weight, _ = score_headline(title)
165:    if weight >= 3.0:
166:        return "critical"
167:    if weight >= 2.0:
168:        return "high"
169:    if weight >= 1.5:
170:        return "moderate"
171:    return "normal"
172:
173:
174:def is_credible_source(source: str) -> bool:
175:    """Check if a source name matches a credible financial news source."""
176:    if not source:
177:        return False
178:    lower = source.lower().strip()
179:    return any(cs in lower for cs in CREDIBLE_SOURCES)
180:
181:
182:def dissemination_score(articles: list[dict]) -> float:
183:    """Score how widely news has spread (FinGPT dissemination-aware pattern).
184:
185:    Factors:
186:    1. Unique source count — more sources = wider spread
187:    2. Source diversity — credible sources (Reuters, Bloomberg) weight more
188:    3. Time clustering — articles within 1h of each other = breaking news
189:
190:    Returns:
191:        Float multiplier (1.0 = normal, up to 3.0 for breaking news with wide coverage).
192:        Used to amplify headline weights in sentiment aggregation.
193:    """
194:    if not articles or len(articles) < 2:
195:        return 1.0
196:
197:    # Factor 1: Unique source count
198:    sources = set()
199:    for a in articles:
200:        src = a.get("source", "unknown").lower().strip()
201:        if src:
202:            sources.add(src)
203:    source_count = len(sources)
204:    # 1 source = 1.0, 3+ sources = 1.5, 5+ = 2.0
205:    source_factor = min(1.0 + (source_count - 1) * 0.25, 2.0)
206:
207:    # Factor 2: Source diversity — credible source presence
208:    credible_count = sum(1 for s in sources if any(cs in s for cs in CREDIBLE_SOURCES))
209:    diversity_factor = 1.0
210:    if credible_count >= 2:
211:        diversity_factor = 1.5
212:    elif credible_count >= 1:
213:        diversity_factor = 1.25
214:
215:    # Factor 3: Time clustering — articles within 1h of each other
216:    from datetime import datetime
217:    timestamps = []
218:    for a in articles:
219:        pub = a.get("published", "")
220:        if not pub:
221:            continue
222:        try:
223:            if isinstance(pub, (int, float)):
224:                ts = datetime.fromtimestamp(pub, tz=UTC)
225:            else:
226:                # Try ISO format
227:                pub_str = str(pub).replace("Z", "+00:00")
228:                ts = datetime.fromisoformat(pub_str)
229:            timestamps.append(ts.timestamp())
230:        except (ValueError, TypeError, OSError):
231:            continue
232:
233:    clustering_factor = 1.0
234:    if len(timestamps) >= 3:
235:        timestamps.sort()
236:        # Check if most articles appeared within a 1-hour window
237:        window = 3600  # 1 hour
238:        max_cluster = 1
239:        for i in range(len(timestamps)):
240:            cluster = sum(1 for t in timestamps if abs(t - timestamps[i]) <= window)
241:            max_cluster = max(max_cluster, cluster)
242:        # If 60%+ of articles are in a 1h cluster, it's breaking news
243:        cluster_ratio = max_cluster / len(timestamps)
244:        if cluster_ratio >= 0.6:
245:            clustering_factor = 1.5
246:
247:    # Combined score (multiplicative, capped at 3.0)
248:    score = source_factor * diversity_factor * clustering_factor
249:    return min(round(score, 2), 3.0)
250:
251:
252:def get_sector_impact(keyword: str, ticker: str) -> str | None:
253:    """Get the directional impact of a keyword on a specific ticker.
254:
255:    Returns "BUY", "SELL", or None if no sector-specific impact.
256:    """
257:    impacts = KEYWORD_SECTOR_IMPACT.get(keyword.lower(), {})
258:    ticker_secs = TICKER_SECTORS.get(ticker, set())
259:    for sector, direction in impacts.items():
260:        if sector in ticker_secs:
261:            return direction
262:    return None
263:
264:
265:# ---------------------------------------------------------------------------
266:# Headline relevance (added 2026-04-28 for sentiment regression fix)
267:# ---------------------------------------------------------------------------
268:#
269:# Background: shadow LLM accuracy investigation found that the sentiment
270:# pipeline was scoring every wire-feed headline returned by Yahoo/CryptoCompare,
271:# including bare price-tickers like "Bitcoin: $67,123" and generic
272:# "Markets mixed in afternoon trade" boilerplate. Models correctly labeled
273:# these neutral, but the neutral mass drowned out the few decisive headlines
274:# in the average. Sentiment regressed from 75.3% -> ~42% over W16-W17.
275:#
276:# A headline is "relevant" to a ticker if:
277:#   1. It triggers a keyword from score_headline (weight > 1.0), OR
278:#   2. It mentions the ticker symbol or a known synonym (Bitcoin/BTC,
279:#      Ethereum/ETH, gold/XAU, silver/XAG, or the stock symbol)
280:# Source-credibility lives in the wrapper in portfolio/sentiment.py, not here.
281:
282:_TICKER_SYNONYMS: dict[str, list[str]] = {
283:    "BTC": ["btc", "bitcoin", "bitcoins"],
284:    "ETH": ["eth", "ethereum", "ether"],
285:    "XAU": ["xau", "gold", "bullion"],
286:    "XAG": ["xag", "silver"],
287:    "MSTR": ["mstr", "microstrategy"],
288:    "NVDA": ["nvda", "nvidia"],
289:    "AMD": ["amd"],
290:    "GOOGL": ["googl", "google", "alphabet"],
291:    "AMZN": ["amzn", "amazon"],
292:    "AAPL": ["aapl", "apple"],
293:    "META": ["meta", "facebook", "instagram"],
294:    "AVGO": ["avgo", "broadcom"],
295:    "TSM": ["tsm", "tsmc"],
296:    "MU": ["mu", "micron"],
297:    "PLTR": ["pltr", "palantir"],
298:    "SMCI": ["smci"],
299:    "TTWO": ["ttwo", "rockstar"],
300:    "VRT": ["vrt", "vertiv"],
301:    "LMT": ["lmt", "lockheed"],
302:    "SOUN": ["soun", "soundhound"],
303:}
304:
305:
306:def _ticker_synonym_pattern(ticker: str) -> re.Pattern | None:
307:    short = ticker.upper().replace("-USD", "")
308:    syns = _TICKER_SYNONYMS.get(short)
309:    if not syns:
310:        if not short or not short.isalnum():
311:            return None
312:        return re.compile(r"\b" + re.escape(short) + r"\b", re.IGNORECASE)
313:    pattern = "|".join(re.escape(s) for s in syns)
314:    return re.compile(r"\b(" + pattern + r")\b", re.IGNORECASE)
315:
316:
317:# Memoize per-ticker patterns; tickers are a fixed small set so the cache
318:# never grows large. None values are cached too (use sentinel-via-membership).
319:_PATTERN_CACHE: dict[str, re.Pattern | None] = {}
320:
321:
322:def is_relevant_headline(title: str, ticker: str) -> bool:
323:    """Return True if the headline is plausibly relevant to the ticker.
324:
325:    Used by the sentiment pipeline to filter wire-noise before model inference.
326:    See the module-level comment block above for background.
327:
328:    Minimum content gate: even when a ticker synonym matches, the headline
329:    must have at least 3 word tokens AFTER stripping the synonym itself.
330:    This drops bare price-tickers like "Bitcoin: $67,123" (1 token after
331:    removing "Bitcoin") while keeping "Bitcoin treasury firm adds 500 BTC"
332:    (5 tokens after removing the synonyms).
333:    """
334:    if not title or not title.strip():
335:        return False
336:
337:    weight, _ = score_headline(title)
338:    if weight > 1.0:
339:        return True
340:
341:    short = ticker.upper().replace("-USD", "")
342:    if short not in _PATTERN_CACHE:
343:        _PATTERN_CACHE[short] = _ticker_synonym_pattern(short)
344:    pat = _PATTERN_CACHE[short]
345:    if pat is None or not pat.search(title):
346:        return False
347:
348:    # Synonym matched — guard against bare price-ticker noise. Strip the
349:    # synonym occurrences and count the remaining word tokens; need >=3 to
350:    # be considered real content.
351:    stripped = pat.sub(" ", title)
352:    tokens = [t for t in re.findall(r"\b[A-Za-z]{2,}\b", stripped) if t.lower() not in {"the", "and", "for", "from"}]
353:    return len(tokens) >= 3
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"talib|pandas_ta|warm|min_periods|rolling\\(|ewm\\(|dropna|iloc\\[-1\\]|RSI|MACD|ATR|BBANDS|EMA|SMA\" Q:\\finance-analyzer\\portfolio\\indicators.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\indicators.py" in Q:\finance-analyzer
 succeeded in 737ms:
23-    if len(df) < min_rows:
24-        logger.debug("compute_indicators: insufficient data (%d rows, need %d)", len(df), min_rows)
25-        return None
26-    close = df["close"].copy()
27-
28-    # BUG-87: Guard against NaN in close series
29:    if close.iloc[-1] != close.iloc[-1]:  # NaN check (NaN != NaN is True)
30-        logger.warning("compute_indicators: last close is NaN, returning None")
31-        return None
32-    if close.isna().all():
33-        logger.warning("compute_indicators: all close values are NaN, returning None")
34-        return None
35-    # Forward-fill interior NaN gaps to prevent downstream NaN propagation
--
37-        logger.debug("compute_indicators: forward-filling %d NaN close values", close.isna().sum())
38-        close = close.ffill().bfill()
39-        df = df.copy()
40-        df["close"] = close
41-
42-    # BUG-209: Guard against zero/negative prices (Binance maintenance, API glitches).
43:    # A single zero-price candle produces RSI=50, MACD=0, ATR=0 — poisoning consensus.
44-    bad_prices = (close <= 0).sum()
45-    if bad_prices > 0:
46-        logger.warning("compute_indicators: %d zero/negative close values detected, returning None", bad_prices)
47-        return None
48-
49:    # RSI(rsi_period)
50-    delta = close.diff()
51-    gain = delta.where(delta > 0, 0.0)
52-    loss = (-delta).where(delta < 0, 0.0)
53:    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
54:    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
55-    avg_loss_safe = avg_loss.replace(0, np.finfo(float).eps)
56-    rs = avg_gain / avg_loss_safe
57-    rsi = 100 - (100 / (1 + rs))
58-
59:    # MACD(macd_fast, macd_slow, macd_signal_period)
60:    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
61:    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
62-    macd = ema_fast - ema_slow
63:    macd_signal = macd.ewm(span=macd_signal_period, adjust=False).mean()
64-    macd_hist = macd - macd_signal
65-
66:    # EMA(9, 21)
67:    ema9 = close.ewm(span=9, adjust=False).mean()
68:    ema21 = close.ewm(span=21, adjust=False).mean()
69-
70-    # Bollinger Bands(20, 2)
71:    bb_mid = close.rolling(20).mean()
72:    bb_std = close.rolling(20).std()
73-    bb_upper = bb_mid + 2 * bb_std
74-    bb_lower = bb_mid - 2 * bb_std
75-
76:    # ATR(14) and ADX(14) — shared True Range calculation
77-    high = df["high"]
78-    low = df["low"]
79-    prev_close = close.shift(1)
80-    tr = pd.concat(
81-        [
82-            high - low,
83-            (high - prev_close).abs(),
84-            (low - prev_close).abs(),
85-        ],
86-        axis=1,
87-    ).max(axis=1)
88:    atr14 = tr.ewm(span=14, adjust=False).mean().iloc[-1]
89-
90-    # ADX(14): Average Directional Index for regime detection
91-    # +DM/-DM → smoothed → +DI/-DI → DX → ADX
92-    prev_high = high.shift(1)
93-    prev_low = low.shift(1)
94-    plus_dm = (high - prev_high).clip(lower=0)
95-    minus_dm = (prev_low - low).clip(lower=0)
96-    # Only keep the larger of +DM/-DM (the other becomes 0)
97-    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
98-    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
99-    # Smooth with EWM (Wilder's smoothing ≈ EWM with alpha=1/14)
100:    smoothed_tr = tr.ewm(span=14, adjust=False).mean()
101:    smoothed_plus_dm = plus_dm.ewm(span=14, adjust=False).mean()
102:    smoothed_minus_dm = minus_dm.ewm(span=14, adjust=False).mean()
103-    # Directional indicators
104-    eps = np.finfo(float).eps
105-    plus_di = 100 * smoothed_plus_dm / (smoothed_tr + eps)
106-    minus_di = 100 * smoothed_minus_dm / (smoothed_tr + eps)
107-    # DX and ADX
108-    di_sum = plus_di + minus_di + eps
109-    dx = 100 * (plus_di - minus_di).abs() / di_sum
110:    adx14 = dx.ewm(span=14, adjust=False).mean()
111-
112:    # RSI rolling percentiles for adaptive thresholds
113-    rsi_series = rsi
114:    rsi_p20 = rsi_series.rolling(100, min_periods=20).quantile(0.2).iloc[-1]
115:    rsi_p80 = rsi_series.rolling(100, min_periods=20).quantile(0.8).iloc[-1]
116-
117-    def _safe(val, default=0.0):
118-        """Return float(val) if finite, else default. Prevents NaN in JSON output."""
119-        v = float(val)
120-        if v != v or not np.isfinite(v):  # NaN or Inf
121-            return default
122-        return v
123-
124:    close_val = _safe(close.iloc[-1])
125:    bb_upper_val = _safe(bb_upper.iloc[-1], close_val)
126:    bb_lower_val = _safe(bb_lower.iloc[-1], close_val)
127-
128-    return {
129-        "close": close_val,
130:        "rsi": _safe(rsi.iloc[-1], 50.0),
131:        "macd_hist": _safe(macd_hist.iloc[-1]),
132-        "macd_hist_prev": _safe(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0,
133:        "ema9": _safe(ema9.iloc[-1], close_val),
134:        "ema21": _safe(ema21.iloc[-1], close_val),
135-        "bb_upper": bb_upper_val,
136-        "bb_lower": bb_lower_val,
137:        "bb_mid": _safe(bb_mid.iloc[-1], close_val),
138-        "price_vs_bb": (
139-            "below_lower"
140-            if close_val <= bb_lower_val
141-            else (
142-                "above_upper"
143-                if close_val >= bb_upper_val
144-                else "inside"
145-            )
146-        ),
147-        "atr": _safe(atr14),
148:        "atr_pct": _safe(atr14 / close.iloc[-1] * 100) if close_val != 0 else 0.0,
149:        "adx": _safe(adx14.iloc[-1], 0.0),
150-        "rsi_p20": _safe(rsi_p20, 30.0),
151-        "rsi_p80": _safe(rsi_p80, 70.0),
152-    }
153-
154-
155-def detect_regime(indicators, is_crypto=True):
--
180-
181-    close = indicators.get("close", 0)
182-    high_vol_threshold = 4.0 if is_crypto else 3.0
183-    if atr_pct > high_vol_threshold:
184-        result = "high-vol"
185-    elif adx < 20:
186:        # ADX < 20: no meaningful trend regardless of EMA gap
187-        result = "ranging"
188-    elif ema21 != 0 and abs(ema9 - ema21) / ema21 * 100 >= 0.5:
189-        if ema9 > ema21 and rsi > 45:
190-            result = "trending-up"
191-        elif ema9 < ema21 and rsi < 55:
192-            result = "trending-down"
193-        else:
194-            result = "ranging"
195-    else:
196-        result = "ranging"
197-
198:    # BUG-156: EMA crossover lags behind V-shaped recoveries.
199-    if (
200-        (result == "trending-down" and close > 0 and ema21 > 0 and close > ema21)
201-        or (result == "trending-up" and close > 0 and ema21 > 0 and close < ema21)
202-    ):
203-        result = "ranging"
204-
--
208-
209-
210-def technical_signal(ind):
211-    buy = 0
212-    sell = 0
213-    total = 0
214:    # RSI: BUY when < 30 (oversold), SELL when > 70 (overbought), else neutral
215-    if ind["rsi"] < 30:
216-        buy += 1
217-        total += 1
218-    elif ind["rsi"] > 70:
219-        sell += 1
220-        total += 1
221:    # MACD: histogram crossover (neg->pos = BUY, pos->neg = SELL)
222-    macd_hist = ind["macd_hist"]
223-    macd_hist_prev = ind.get("macd_hist_prev", 0.0)
224-    if macd_hist > 0 and macd_hist_prev <= 0:
225-        buy += 1
226-        total += 1
227-    elif macd_hist < 0 and macd_hist_prev >= 0:
228-        sell += 1
229-        total += 1
230:    # EMA: with deadband — only signal when gap > 0.5%
231-    ema9 = ind["ema9"]
232-    ema21 = ind["ema21"]
233-    ema_gap_pct = abs(ema9 - ema21) / ema21 * 100 if ema21 != 0 else 0
234-    if ema_gap_pct >= 0.5:
235-        if ema9 > ema21:
236-            buy += 1
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 745ms:
1:"""Technical indicator computation and regime detection."""
2:
3:import logging
4:
5:import numpy as np
6:import pandas as pd
7:
8:import portfolio.shared_state as _ss
9:
10:logger = logging.getLogger("portfolio.indicators")
11:
12:
13:def compute_indicators(df, horizon=None):
14:    if horizon == "3h":
15:        rsi_period = 7
16:        macd_fast, macd_slow, macd_signal_period = 8, 17, 9
17:        min_rows = macd_slow  # 17
18:    else:
19:        rsi_period = 14
20:        macd_fast, macd_slow, macd_signal_period = 12, 26, 9
21:        min_rows = macd_slow  # 26
22:
23:    if len(df) < min_rows:
24:        logger.debug("compute_indicators: insufficient data (%d rows, need %d)", len(df), min_rows)
25:        return None
26:    close = df["close"].copy()
27:
28:    # BUG-87: Guard against NaN in close series
29:    if close.iloc[-1] != close.iloc[-1]:  # NaN check (NaN != NaN is True)
30:        logger.warning("compute_indicators: last close is NaN, returning None")
31:        return None
32:    if close.isna().all():
33:        logger.warning("compute_indicators: all close values are NaN, returning None")
34:        return None
35:    # Forward-fill interior NaN gaps to prevent downstream NaN propagation
36:    if close.isna().any():
37:        logger.debug("compute_indicators: forward-filling %d NaN close values", close.isna().sum())
38:        close = close.ffill().bfill()
39:        df = df.copy()
40:        df["close"] = close
41:
42:    # BUG-209: Guard against zero/negative prices (Binance maintenance, API glitches).
43:    # A single zero-price candle produces RSI=50, MACD=0, ATR=0 — poisoning consensus.
44:    bad_prices = (close <= 0).sum()
45:    if bad_prices > 0:
46:        logger.warning("compute_indicators: %d zero/negative close values detected, returning None", bad_prices)
47:        return None
48:
49:    # RSI(rsi_period)
50:    delta = close.diff()
51:    gain = delta.where(delta > 0, 0.0)
52:    loss = (-delta).where(delta < 0, 0.0)
53:    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
54:    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
55:    avg_loss_safe = avg_loss.replace(0, np.finfo(float).eps)
56:    rs = avg_gain / avg_loss_safe
57:    rsi = 100 - (100 / (1 + rs))
58:
59:    # MACD(macd_fast, macd_slow, macd_signal_period)
60:    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
61:    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
62:    macd = ema_fast - ema_slow
63:    macd_signal = macd.ewm(span=macd_signal_period, adjust=False).mean()
64:    macd_hist = macd - macd_signal
65:
66:    # EMA(9, 21)
67:    ema9 = close.ewm(span=9, adjust=False).mean()
68:    ema21 = close.ewm(span=21, adjust=False).mean()
69:
70:    # Bollinger Bands(20, 2)
71:    bb_mid = close.rolling(20).mean()
72:    bb_std = close.rolling(20).std()
73:    bb_upper = bb_mid + 2 * bb_std
74:    bb_lower = bb_mid - 2 * bb_std
75:
76:    # ATR(14) and ADX(14) — shared True Range calculation
77:    high = df["high"]
78:    low = df["low"]
79:    prev_close = close.shift(1)
80:    tr = pd.concat(
81:        [
82:            high - low,
83:            (high - prev_close).abs(),
84:            (low - prev_close).abs(),
85:        ],
86:        axis=1,
87:    ).max(axis=1)
88:    atr14 = tr.ewm(span=14, adjust=False).mean().iloc[-1]
89:
90:    # ADX(14): Average Directional Index for regime detection
91:    # +DM/-DM → smoothed → +DI/-DI → DX → ADX
92:    prev_high = high.shift(1)
93:    prev_low = low.shift(1)
94:    plus_dm = (high - prev_high).clip(lower=0)
95:    minus_dm = (prev_low - low).clip(lower=0)
96:    # Only keep the larger of +DM/-DM (the other becomes 0)
97:    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
98:    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
99:    # Smooth with EWM (Wilder's smoothing ≈ EWM with alpha=1/14)
100:    smoothed_tr = tr.ewm(span=14, adjust=False).mean()
101:    smoothed_plus_dm = plus_dm.ewm(span=14, adjust=False).mean()
102:    smoothed_minus_dm = minus_dm.ewm(span=14, adjust=False).mean()
103:    # Directional indicators
104:    eps = np.finfo(float).eps
105:    plus_di = 100 * smoothed_plus_dm / (smoothed_tr + eps)
106:    minus_di = 100 * smoothed_minus_dm / (smoothed_tr + eps)
107:    # DX and ADX
108:    di_sum = plus_di + minus_di + eps
109:    dx = 100 * (plus_di - minus_di).abs() / di_sum
110:    adx14 = dx.ewm(span=14, adjust=False).mean()
111:
112:    # RSI rolling percentiles for adaptive thresholds
113:    rsi_series = rsi
114:    rsi_p20 = rsi_series.rolling(100, min_periods=20).quantile(0.2).iloc[-1]
115:    rsi_p80 = rsi_series.rolling(100, min_periods=20).quantile(0.8).iloc[-1]
116:
117:    def _safe(val, default=0.0):
118:        """Return float(val) if finite, else default. Prevents NaN in JSON output."""
119:        v = float(val)
120:        if v != v or not np.isfinite(v):  # NaN or Inf
121:            return default
122:        return v
123:
124:    close_val = _safe(close.iloc[-1])
125:    bb_upper_val = _safe(bb_upper.iloc[-1], close_val)
126:    bb_lower_val = _safe(bb_lower.iloc[-1], close_val)
127:
128:    return {
129:        "close": close_val,
130:        "rsi": _safe(rsi.iloc[-1], 50.0),
131:        "macd_hist": _safe(macd_hist.iloc[-1]),
132:        "macd_hist_prev": _safe(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0,
133:        "ema9": _safe(ema9.iloc[-1], close_val),
134:        "ema21": _safe(ema21.iloc[-1], close_val),
135:        "bb_upper": bb_upper_val,
136:        "bb_lower": bb_lower_val,
137:        "bb_mid": _safe(bb_mid.iloc[-1], close_val),
138:        "price_vs_bb": (
139:            "below_lower"
140:            if close_val <= bb_lower_val
141:            else (
142:                "above_upper"
143:                if close_val >= bb_upper_val
144:                else "inside"
145:            )
146:        ),
147:        "atr": _safe(atr14),
148:        "atr_pct": _safe(atr14 / close.iloc[-1] * 100) if close_val != 0 else 0.0,
149:        "adx": _safe(adx14.iloc[-1], 0.0),
150:        "rsi_p20": _safe(rsi_p20, 30.0),
151:        "rsi_p80": _safe(rsi_p80, 70.0),
152:    }
153:
154:
155:def detect_regime(indicators, is_crypto=True):
156:    # BUG-169: Access regime cache under lock — 8 threads from ThreadPoolExecutor
157:    # can call this concurrently. The check-then-clear pattern must be atomic.
158:    cache_key = (
159:        round(indicators.get("close", 0), 4),
160:        round(indicators.get("atr_pct", 0), 4),
161:        round(indicators.get("ema9", 0), 4),
162:        round(indicators.get("ema21", 0), 4),
163:        round(indicators.get("rsi", 50), 4),
164:        round(indicators.get("adx", 0), 4),
165:        is_crypto,
166:    )
167:    with _ss._regime_lock:
168:        if _ss._run_cycle_id != _ss._regime_cache_cycle:
169:            _ss._regime_cache = {}
170:            _ss._regime_cache_cycle = _ss._run_cycle_id
171:        if cache_key in _ss._regime_cache:
172:            return _ss._regime_cache[cache_key]
173:
174:    # Compute outside lock (pure function, no shared state)
175:    atr_pct = indicators.get("atr_pct", 0)
176:    ema9 = indicators.get("ema9", 0)
177:    ema21 = indicators.get("ema21", 0)
178:    rsi = indicators.get("rsi", 50)
179:    adx = indicators.get("adx", 0)
180:
181:    close = indicators.get("close", 0)
182:    high_vol_threshold = 4.0 if is_crypto else 3.0
183:    if atr_pct > high_vol_threshold:
184:        result = "high-vol"
185:    elif adx < 20:
186:        # ADX < 20: no meaningful trend regardless of EMA gap
187:        result = "ranging"
188:    elif ema21 != 0 and abs(ema9 - ema21) / ema21 * 100 >= 0.5:
189:        if ema9 > ema21 and rsi > 45:
190:            result = "trending-up"
191:        elif ema9 < ema21 and rsi < 55:
192:            result = "trending-down"
193:        else:
194:            result = "ranging"
195:    else:
196:        result = "ranging"
197:
198:    # BUG-156: EMA crossover lags behind V-shaped recoveries.
199:    if (
200:        (result == "trending-down" and close > 0 and ema21 > 0 and close > ema21)
201:        or (result == "trending-up" and close > 0 and ema21 > 0 and close < ema21)
202:    ):
203:        result = "ranging"
204:
205:    with _ss._regime_lock:
206:        _ss._regime_cache[cache_key] = result
207:    return result
208:
209:
210:def technical_signal(ind):
211:    buy = 0
212:    sell = 0
213:    total = 0
214:    # RSI: BUY when < 30 (oversold), SELL when > 70 (overbought), else neutral
215:    if ind["rsi"] < 30:
216:        buy += 1
217:        total += 1
218:    elif ind["rsi"] > 70:
219:        sell += 1
220:        total += 1
221:    # MACD: histogram crossover (neg->pos = BUY, pos->neg = SELL)
222:    macd_hist = ind["macd_hist"]
223:    macd_hist_prev = ind.get("macd_hist_prev", 0.0)
224:    if macd_hist > 0 and macd_hist_prev <= 0:
225:        buy += 1
226:        total += 1
227:    elif macd_hist < 0 and macd_hist_prev >= 0:
228:        sell += 1
229:        total += 1
230:    # EMA: with deadband — only signal when gap > 0.5%
231:    ema9 = ind["ema9"]
232:    ema21 = ind["ema21"]
233:    ema_gap_pct = abs(ema9 - ema21) / ema21 * 100 if ema21 != 0 else 0
234:    if ema_gap_pct >= 0.5:
235:        if ema9 > ema21:
236:            buy += 1
237:        else:
238:            sell += 1
239:        total += 1
240:    # BB: below lower = BUY, above upper = SELL
241:    if ind["price_vs_bb"] == "below_lower":
242:        buy += 1
243:        total += 1
244:    elif ind["price_vs_bb"] == "above_upper":
245:        sell += 1
246:        total += 1
247:    if total == 0:
248:        return "HOLD", 0.0
249:    if buy > sell:
250:        return "BUY", buy / total
251:    elif sell > buy:
252:        return "SELL", sell / total
253:    return "HOLD", 0.5
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"compute_indicators\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"horizon=\\\"3h\\\"|horizon='3h'|\\b3h\\b\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 870ms:
Q:\finance-analyzer\portfolio\data_collector.py:295:        ind = compute_indicators(df)
Q:\finance-analyzer\portfolio\indicators.py:13:def compute_indicators(df, horizon=None):
Q:\finance-analyzer\portfolio\main.py:493:                ind = compute_indicators(now_df)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:30:18.893006Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.9 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:horizon=\)
    ^
error: unclosed group

 exited 1 in 920ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:horizon=\)
    ^
error: unclosed group

codex
I’ve reached the point where the remaining value is in call-site validation, not more blind file reads. Next I’m checking who calls the calendar, indicator, and microstructure helpers so I can separate real production paths from theoretical edge cases.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"get_session_info\\(|remaining_session_minutes\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"persist_state\\(|load_persisted_state\\(|get_microstructure_state\\(|accumulate_snapshot\\(|get_recent_trades\\(|depth_imbalance\\(|trade_flow_imbalance\\(|compute_vpin\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"get_onchain_data\\(|interpret_onchain\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"get_sentiment\\(|get_crypto_sentiment\\(|get_reddit_posts\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 881ms:
Q:\finance-analyzer\portfolio\fin_snipe_manager.py:405:    session = get_session_info("warrant", underlying=snapshot.get("ticker"))
Q:\finance-analyzer\portfolio\risk_management.py:457:        remaining = remaining_session_minutes(inst_type)
Q:\finance-analyzer\portfolio\reporting.py:697:                sess = get_session_info("warrant", underlying=underlying)
Q:\finance-analyzer\portfolio\session_calendar.py:8:    info = get_session_info("warrant", underlying="XAG-USD")
Q:\finance-analyzer\portfolio\session_calendar.py:115:def get_session_info(instrument_type: str,
Q:\finance-analyzer\portfolio\session_calendar.py:193:        us_info = get_session_info("stock_us", now=now)
Q:\finance-analyzer\portfolio\session_calendar.py:207:def remaining_session_minutes(instrument_type: str = "warrant",
Q:\finance-analyzer\portfolio\session_calendar.py:210:    info = get_session_info(instrument_type, now=now)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 890ms:
Q:\finance-analyzer\portfolio\metals_orderbook.py:91:def get_recent_trades(ticker: str, limit: int = 100) -> list[dict] | None:
Q:\finance-analyzer\portfolio\microstructure.py:23:def depth_imbalance(depth: dict, levels: int | None = None) -> float:
Q:\finance-analyzer\portfolio\microstructure.py:41:def trade_flow_imbalance(trades: list[dict]) -> dict[str, float] | None:
Q:\finance-analyzer\portfolio\microstructure.py:63:def compute_vpin(trades: list[dict], n_buckets: int = 20) -> float | None:
Q:\finance-analyzer\portfolio\microstructure_state.py:55:def accumulate_snapshot(ticker: str, depth: dict) -> None:
Q:\finance-analyzer\portfolio\microstructure_state.py:175:def get_microstructure_state(ticker: str) -> dict:
Q:\finance-analyzer\portfolio\microstructure_state.py:205:def persist_state() -> None:
Q:\finance-analyzer\portfolio\microstructure_state.py:209:        ms = get_microstructure_state(ticker)
Q:\finance-analyzer\portfolio\microstructure_state.py:216:def load_persisted_state(ticker: str) -> dict | None:
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:51:    trades = get_recent_trades(ticker, limit=200)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:55:    di = depth_imbalance(depth)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:56:    tfi = trade_flow_imbalance(trades)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:57:    vpin = compute_vpin(trades, n_buckets=20)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:70:        ms_state = load_persisted_state(ticker)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 876ms:
Q:\finance-analyzer\portfolio\crypto_macro_data.py:330:        onchain = get_onchain_data()
Q:\finance-analyzer\portfolio\onchain_data.py:12:    data = get_onchain_data()  # returns dict or None
Q:\finance-analyzer\portfolio\onchain_data.py:13:    interp = interpret_onchain(data)  # returns interpretation dict
Q:\finance-analyzer\portfolio\onchain_data.py:244:def get_onchain_data():
Q:\finance-analyzer\portfolio\onchain_data.py:291:def interpret_onchain(data):
Q:\finance-analyzer\portfolio\reporting.py:397:        onchain = get_onchain_data()
Q:\finance-analyzer\portfolio\reporting.py:399:            interp = interpret_onchain(onchain)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 888ms:
Q:\finance-analyzer\portfolio\llm_batch.py:35:# get_sentiment() invocation so the results can be stitched back into one
Q:\finance-analyzer\portfolio\llm_batch.py:121:        ab_key: Shared key identifying the parent get_sentiment() call
Q:\finance-analyzer\portfolio\llm_batch.py:123:            the same get_sentiment() invocation share this key so their
Q:\finance-analyzer\portfolio\llm_batch.py:133:        # multiple cumulative clusters in the same get_sentiment() call.
Q:\finance-analyzer\portfolio\social_sentiment.py:91:def get_reddit_posts(ticker, limit=20):
Q:\finance-analyzer\portfolio\social_sentiment.py:133:        posts = get_reddit_posts(ticker)
Q:\finance-analyzer\portfolio\sentiment.py:319:# AFTER get_sentiment() has already returned to the signal engine. The
Q:\finance-analyzer\portfolio\sentiment.py:321:# log write is DEFERRED: get_sentiment() stashes the primary + finbert shadow
Q:\finance-analyzer\portfolio\sentiment.py:328:# accuracy tracking consumes: one row per get_sentiment() call, with a
Q:\finance-analyzer\portfolio\sentiment.py:349:    results arrive in flush_ab_log(). Called from get_sentiment().
Q:\finance-analyzer\portfolio\sentiment.py:748:def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None,
Q:\finance-analyzer\portfolio\sentiment.py:853:    # get_sentiment() has returned. Rather than duplicate the A/B log entry
Q:\finance-analyzer\portfolio\sentiment.py:935:def get_crypto_sentiment(ticker="BTC") -> dict:
Q:\finance-analyzer\portfolio\sentiment.py:936:    return get_sentiment(ticker)
Q:\finance-analyzer\portfolio\sentiment.py:944:        result = get_sentiment(ticker)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"get_recent_trades|depth_imbalance|trade_flow_imbalance|compute_vpin|load_persisted_state|ofi|spread_zscore|imbalance_ratio\" Q:\\finance-analyzer\\portfolio\\signals\\orderbook_flow.py" in Q:\finance-analyzer
 succeeded in 667ms:
33-_TRADE_THROUGH_THRESHOLD = 2
34-
35-
36-def _get_microstructure_context(ticker: str) -> dict | None:
37-    """Fetch live microstructure data for the given ticker."""
38-    try:
39:        from portfolio.metals_orderbook import get_orderbook_depth, get_recent_trades
40-        from portfolio.microstructure import (
41:            compute_vpin,
42:            depth_imbalance,
43-            detect_trade_throughs,
44:            trade_flow_imbalance,
45-        )
46-    except ImportError:
47-        logger.debug("Microstructure modules not available")
48-        return None
49-
50-    depth = get_orderbook_depth(ticker, limit=20)
51:    trades = get_recent_trades(ticker, limit=200)
52-    if depth is None or trades is None:
53-        return None
54-
55:    di = depth_imbalance(depth)
56:    tfi = trade_flow_imbalance(trades)
57:    vpin = compute_vpin(trades, n_buckets=20)
58-    tt = detect_trade_throughs(trades)
59-
60-    if tfi is None:
61-        return None
62-
63-    # Read accumulated OFI, OFI z-score, multiscale, and spread z-score
64:    ofi = 0.0
65:    ofi_zscore = 0.0
66-    sz = 0.0
67-    flow_acceleration = 0.0
68-    try:
69:        from portfolio.microstructure_state import load_persisted_state
70:        ms_state = load_persisted_state(ticker)
71-        if ms_state:
72:            ofi = ms_state.get("ofi", 0.0)
73:            ofi_zscore = ms_state.get("ofi_zscore", 0.0)
74:            sz = ms_state.get("spread_zscore", 0.0)
75-            flow_acceleration = ms_state.get("flow_acceleration", 0.0)
76-    except ImportError:
77-        pass
78-
79-    return {
80:        "depth_imbalance": di,
81:        "trade_imbalance_ratio": tfi["imbalance_ratio"],
82-        "vpin": vpin if vpin is not None else 0.0,
83:        "ofi": ofi,
84:        "ofi_zscore": ofi_zscore,
85:        "spread_zscore": sz,
86-        "spread_bps": depth.get("spread_bps", 0.0),
87-        "trade_throughs": tt,
88-        "flow_acceleration": flow_acceleration,
89-    }
90-
91-
--
114-        return empty
115-
116-    votes = []
117-    sub_signals = {}
118-
119-    # Sub 1: Depth Imbalance
120:    di = ctx["depth_imbalance"]
121-    if di > _DEPTH_IMBALANCE_THRESHOLD:
122:        sub_signals["depth_imbalance"] = "BUY"
123-    elif di < -_DEPTH_IMBALANCE_THRESHOLD:
124:        sub_signals["depth_imbalance"] = "SELL"
125-    else:
126:        sub_signals["depth_imbalance"] = "HOLD"
127:    votes.append(sub_signals["depth_imbalance"])
128-
129-    # Sub 2: Trade Flow Imbalance
130:    tir = ctx["trade_imbalance_ratio"]
131-    if tir > _TRADE_IMBALANCE_THRESHOLD:
132-        sub_signals["trade_flow"] = "BUY"
133-    elif tir < -_TRADE_IMBALANCE_THRESHOLD:
134-        sub_signals["trade_flow"] = "SELL"
135-    else:
136-        sub_signals["trade_flow"] = "HOLD"
--
147-            sub_signals["vpin"] = "HOLD"
148-    else:
149-        sub_signals["vpin"] = "HOLD"
150-    votes.append(sub_signals["vpin"])
151-
152-    # Sub 4: OFI — prefer z-score (asset-normalized), fallback to absolute
153:    ofi_z = ctx.get("ofi_zscore", 0.0)
154:    ofi = ctx["ofi"]
155:    if abs(ofi_z) > 0.01:  # z-score available (non-zero)
156:        if ofi_z > _OFI_ZSCORE_THRESHOLD:
157:            sub_signals["ofi"] = "BUY"
158:        elif ofi_z < -_OFI_ZSCORE_THRESHOLD:
159:            sub_signals["ofi"] = "SELL"
160-        else:
161:            sub_signals["ofi"] = "HOLD"
162-    else:  # cold start fallback
163:        if ofi > _OFI_THRESHOLD:
164:            sub_signals["ofi"] = "BUY"
165:        elif ofi < -_OFI_THRESHOLD:
166:            sub_signals["ofi"] = "SELL"
167-        else:
168:            sub_signals["ofi"] = "HOLD"
169:    votes.append(sub_signals["ofi"])
170-
171-    # Sub 5: Spread Health — always abstains (HOLD) because spread width
172-    # is non-directional.  The actual effect is the 0.3x confidence penalty
173:    # applied below when spread_zscore > threshold.
174:    sz = ctx.get("spread_zscore", 0.0)
175-    sub_signals["spread_health"] = "HOLD"
176-    votes.append(sub_signals["spread_health"])
177-
178-    # Sub 6: Trade Pressure — trade-through imbalance
179-    tt = ctx.get("trade_throughs") or {}
180-    buy_tt = tt.get("buy_throughs", 0)
--
198-
199-    return {
200-        "action": action,
201-        "confidence": round(confidence, 4),
202-        "sub_signals": sub_signals,
203-        "indicators": {
204:            "depth_imbalance": round(di, 4),
205:            "trade_imbalance_ratio": round(tir, 4),
206-            "vpin": round(vpin, 4),
207-            "high_toxicity": high_toxicity,
208:            "ofi": round(ofi, 4),
209:            "ofi_zscore": round(ctx.get("ofi_zscore", 0.0), 4),
210-            "flow_acceleration": round(ctx.get("flow_acceleration", 0.0), 4),
211:            "spread_zscore": round(sz, 4),
212-            "spread_bps": round(ctx.get("spread_bps", 0.0), 2),
213-            "buy_throughs": buy_tt,
214-            "sell_throughs": sell_tt,
215-            "through_volume": tt.get("through_volume", 0.0),
216-            "max_gap_bps": tt.get("max_gap_bps", 0.0),
217-        },
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"persist_state\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 778ms:
Q:\finance-analyzer\portfolio\microstructure_state.py:205:def persist_state() -> None:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"accumulate_snapshot\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 739ms:
Q:\finance-analyzer\portfolio\microstructure_state.py:55:def accumulate_snapshot(ticker: str, depth: dict) -> None:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"microstructure_state\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 820ms:
Q:\finance-analyzer\portfolio\microstructure_state.py:8:data/microstructure_state.json for cross-process access.
Q:\finance-analyzer\portfolio\microstructure_state.py:21:logger = logging.getLogger("portfolio.microstructure_state")
Q:\finance-analyzer\portfolio\microstructure_state.py:24:_STATE_FILE = _BASE_DIR / "data" / "microstructure_state.json"
Q:\finance-analyzer\portfolio\microstructure_state.py:95:    Called once per cycle from get_microstructure_state to avoid
Q:\finance-analyzer\portfolio\microstructure_state.py:175:def get_microstructure_state(ticker: str) -> dict:
Q:\finance-analyzer\portfolio\microstructure_state.py:209:        ms = get_microstructure_state(ticker)
Q:\finance-analyzer\portfolio\multi_agent_layer2.py:60:            "data/microstructure_state.json",
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:69:        from portfolio.microstructure_state import load_persisted_state
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"XAG-USD|SI=F|SILVER|XAGUSDT|XAU-USD|GC=F|GOLD|XAUUSDT|replace\\(\\\"-USD\\\"|replace\\('-USD'\" Q:\\finance-analyzer\\portfolio\\data_collector.py Q:\\finance-analyzer\\portfolio\\fear_greed.py Q:\\finance-analyzer\\portfolio\\sentiment.py Q:\\finance-analyzer\\portfolio\\bert_sentiment.py Q:\\finance-analyzer\\portfolio\\alpha_vantage.py Q:\\finance-analyzer\\portfolio\\futures_data.py Q:\\finance-analyzer\\portfolio\\funding_rate.py Q:\\finance-analyzer\\portfolio\\fx_rates.py Q:\\finance-analyzer\\portfolio\\onchain_data.py Q:\\finance-analyzer\\portfolio\\news_keywords.py Q:\\finance-analyzer\\portfolio\\social_sentiment.py Q:\\finance-analyzer\\portfolio\\crypto_macro_data.py Q:\\finance-analyzer\\portfolio\\crypto_scheduler.py Q:\\finance-analyzer\\portfolio\\earnings_calendar.py Q:\\finance-analyzer\\portfolio\\econ_dates.py Q:\\finance-analyzer\\portfolio\\fomc_dates.py Q:\\finance-analyzer\\portfolio\\seasonality.py Q:\\finance-analyzer\\portfolio\\seasonality_updater.py Q:\\finance-analyzer\\portfolio\\session_calendar.py Q:\\finance-analyzer\\portfolio\\price_source.py Q:\\finance-analyzer\\portfolio\\http_retry.py Q:\\finance-analyzer\\portfolio\\api_utils.py Q:\\finance-analyzer\\portfolio\\data_refresh.py Q:\\finance-analyzer\\portfolio\\forecast_signal.py Q:\\finance-analyzer\\portfolio\\indicators.py Q:\\finance-analyzer\\portfolio\\metals_orderbook.py Q:\\finance-analyzer\\portfolio\\microstructure.py Q:\\finance-analyzer\\portfolio\\microstructure_state.py Q:\\finance-analyzer\\portfolio\\metals_cross_assets.py Q:\\finance-analyzer\\portfolio\\tickers.py" in Q:\finance-analyzer
2026-05-11T15:31:06.045296Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: unrecognized flag -D

 exited 1 in 709ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: unrecognized flag -D

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -e \"XAG-USD\" -e \"SI=F\" -e \"SILVER\" -e \"XAGUSDT\" -e \"XAU-USD\" -e \"GC=F\" -e \"GOLD\" -e \"XAUUSDT\" -e \"replace\\(\\\"-USD\\\"\" Q:\\finance-analyzer\\portfolio\\data_collector.py Q:\\finance-analyzer\\portfolio\\fear_greed.py Q:\\finance-analyzer\\portfolio\\sentiment.py Q:\\finance-analyzer\\portfolio\\bert_sentiment.py Q:\\finance-analyzer\\portfolio\\alpha_vantage.py Q:\\finance-analyzer\\portfolio\\futures_data.py Q:\\finance-analyzer\\portfolio\\funding_rate.py Q:\\finance-analyzer\\portfolio\\fx_rates.py Q:\\finance-analyzer\\portfolio\\onchain_data.py Q:\\finance-analyzer\\portfolio\\news_keywords.py Q:\\finance-analyzer\\portfolio\\social_sentiment.py Q:\\finance-analyzer\\portfolio\\crypto_macro_data.py Q:\\finance-analyzer\\portfolio\\crypto_scheduler.py Q:\\finance-analyzer\\portfolio\\earnings_calendar.py Q:\\finance-analyzer\\portfolio\\econ_dates.py Q:\\finance-analyzer\\portfolio\\fomc_dates.py Q:\\finance-analyzer\\portfolio\\seasonality.py Q:\\finance-analyzer\\portfolio\\seasonality_updater.py Q:\\finance-analyzer\\portfolio\\session_calendar.py Q:\\finance-analyzer\\portfolio\\price_source.py Q:\\finance-analyzer\\portfolio\\http_retry.py Q:\\finance-analyzer\\portfolio\\api_utils.py Q:\\finance-analyzer\\portfolio\\data_refresh.py Q:\\finance-analyzer\\portfolio\\forecast_signal.py Q:\\finance-analyzer\\portfolio\\indicators.py Q:\\finance-analyzer\\portfolio\\metals_orderbook.py Q:\\finance-analyzer\\portfolio\\microstructure.py Q:\\finance-analyzer\\portfolio\\microstructure_state.py Q:\\finance-analyzer\\portfolio\\metals_cross_assets.py Q:\\finance-analyzer\\portfolio\\tickers.py" in Q:\finance-analyzer
2026-05-11T15:31:16.555568Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: unrecognized flag -D

 exited 1 in 668ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: unrecognized flag -D

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"replace\\(tzinfo=None\\)|fromisoformat\\(|now\\(UTC\\)|tzinfo|date_range|datetime\\.now|astimezone\\(\" Q:\\finance-analyzer\\portfolio\\forecast_signal.py Q:\\finance-analyzer\\portfolio\\crypto_scheduler.py Q:\\finance-analyzer\\portfolio\\econ_dates.py Q:\\finance-analyzer\\portfolio\\fomc_dates.py Q:\\finance-analyzer\\portfolio\\seasonality.py Q:\\finance-analyzer\\portfolio\\seasonality_updater.py" in Q:\finance-analyzer
 succeeded in 658ms:
Q:\finance-analyzer\portfolio\forecast_signal.py-190-    return results
Q:\finance-analyzer\portfolio\forecast_signal.py-191-
Q:\finance-analyzer\portfolio\forecast_signal.py-192-
Q:\finance-analyzer\portfolio\forecast_signal.py-193-def _forecast_chronos_v2(pipeline, ticker, prices, horizons=(1, 24)):
Q:\finance-analyzer\portfolio\forecast_signal.py-194-    """Chronos-2 DataFrame-based forecasting with quantile output."""
Q:\finance-analyzer\portfolio\forecast_signal.py-195-    n = len(prices)
Q:\finance-analyzer\portfolio\forecast_signal.py:196:    timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="h")
Q:\finance-analyzer\portfolio\forecast_signal.py-197-    context_df = pd.DataFrame({
Q:\finance-analyzer\portfolio\forecast_signal.py-198-        "timestamp": timestamps,
Q:\finance-analyzer\portfolio\forecast_signal.py-199-        "target": prices,
Q:\finance-analyzer\portfolio\forecast_signal.py-200-        "id": ticker or "default",
Q:\finance-analyzer\portfolio\forecast_signal.py-201-    })
Q:\finance-analyzer\portfolio\forecast_signal.py-202-
--
Q:\finance-analyzer\portfolio\forecast_signal.py-267-        from prophet import Prophet
Q:\finance-analyzer\portfolio\forecast_signal.py-268-        # Suppress Prophet's verbose stdout
Q:\finance-analyzer\portfolio\forecast_signal.py-269-        _logging.getLogger("prophet").setLevel(_logging.WARNING)
Q:\finance-analyzer\portfolio\forecast_signal.py-270-        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
Q:\finance-analyzer\portfolio\forecast_signal.py-271-
Q:\finance-analyzer\portfolio\forecast_signal.py-272-        # Build dataframe with hourly timestamps (tz-naive, Prophet requirement)
Q:\finance-analyzer\portfolio\forecast_signal.py:273:        now = datetime.now(UTC).replace(tzinfo=None)
Q:\finance-analyzer\portfolio\forecast_signal.py-274-        n = len(prices)
Q:\finance-analyzer\portfolio\forecast_signal.py:275:        ds = pd.date_range(end=now, periods=n, freq="h")
Q:\finance-analyzer\portfolio\forecast_signal.py-276-        df = pd.DataFrame({"ds": ds, "y": prices})
Q:\finance-analyzer\portfolio\forecast_signal.py-277-
Q:\finance-analyzer\portfolio\forecast_signal.py-278-        m = Prophet(
Q:\finance-analyzer\portfolio\forecast_signal.py-279-            daily_seasonality=True,
Q:\finance-analyzer\portfolio\forecast_signal.py-280-            weekly_seasonality=True,
Q:\finance-analyzer\portfolio\forecast_signal.py-281-            yearly_seasonality=False,
--
Q:\finance-analyzer\portfolio\forecast_signal.py-339-        summary = load_json(AGENT_SUMMARY_FILE)
Q:\finance-analyzer\portfolio\forecast_signal.py-340-        if summary is None:
Q:\finance-analyzer\portfolio\forecast_signal.py-341-            logger.error("Could not load tickers from agent_summary.json")
Q:\finance-analyzer\portfolio\forecast_signal.py-342-            return
Q:\finance-analyzer\portfolio\forecast_signal.py-343-        tickers = list(summary.get("signals", {}).keys())
Q:\finance-analyzer\portfolio\forecast_signal.py-344-
Q:\finance-analyzer\portfolio\forecast_signal.py:345:    ts = datetime.now(UTC).isoformat()
Q:\finance-analyzer\portfolio\forecast_signal.py-346-    results = []
Q:\finance-analyzer\portfolio\forecast_signal.py-347-
Q:\finance-analyzer\portfolio\forecast_signal.py-348-    for ticker in tickers:
Q:\finance-analyzer\portfolio\forecast_signal.py-349-        prices = _load_candles(ticker)
Q:\finance-analyzer\portfolio\forecast_signal.py-350-        if not prices or len(prices) < 50:
Q:\finance-analyzer\portfolio\forecast_signal.py-351-            logger.debug("Skipping %s: insufficient candle data (%d)", ticker, len(prices) if prices else 0)
--
Q:\finance-analyzer\portfolio\econ_dates.py-144-    """Return the next economic event on or after ref_date.
Q:\finance-analyzer\portfolio\econ_dates.py-145-
Q:\finance-analyzer\portfolio\econ_dates.py-146-    Returns dict with keys: date, type, impact, hours_until.
Q:\finance-analyzer\portfolio\econ_dates.py-147-    Returns None if no future events in the calendar.
Q:\finance-analyzer\portfolio\econ_dates.py-148-    """
Q:\finance-analyzer\portfolio\econ_dates.py-149-    if ref_date is None:
Q:\finance-analyzer\portfolio\econ_dates.py:150:        ref_date = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\econ_dates.py-151-
Q:\finance-analyzer\portfolio\econ_dates.py-152-    for evt in ECON_EVENTS:
Q:\finance-analyzer\portfolio\econ_dates.py-153-        if evt["date"] >= ref_date:
Q:\finance-analyzer\portfolio\econ_dates.py-154-            # Calculate hours until (approximate: assume 14:00 UTC release)
Q:\finance-analyzer\portfolio\econ_dates.py-155-            evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:156:                                      tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:157:            # M5: Use datetime.now(UTC) so past-today events show hours_until=0.
Q:\finance-analyzer\portfolio\econ_dates.py:158:            now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py-159-            delta = evt_dt - now
Q:\finance-analyzer\portfolio\econ_dates.py-160-            hours = max(0.0, delta.total_seconds() / 3600)
Q:\finance-analyzer\portfolio\econ_dates.py-161-            return {
Q:\finance-analyzer\portfolio\econ_dates.py-162-                "date": evt["date"],
Q:\finance-analyzer\portfolio\econ_dates.py-163-                "type": evt["type"],
Q:\finance-analyzer\portfolio\econ_dates.py-164-                "impact": evt["impact"],
--
Q:\finance-analyzer\portfolio\econ_dates.py-167-    return None
Q:\finance-analyzer\portfolio\econ_dates.py-168-
Q:\finance-analyzer\portfolio\econ_dates.py-169-
Q:\finance-analyzer\portfolio\econ_dates.py-170-def events_within_hours(hours: float, ref_date: date | None = None) -> list[dict]:
Q:\finance-analyzer\portfolio\econ_dates.py-171-    """Return all events within the given hours from ref_date."""
Q:\finance-analyzer\portfolio\econ_dates.py-172-    if ref_date is None:
Q:\finance-analyzer\portfolio\econ_dates.py:173:        ref_date = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\econ_dates.py-174-
Q:\finance-analyzer\portfolio\econ_dates.py-175-    results = []
Q:\finance-analyzer\portfolio\econ_dates.py:176:    now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py-177-    for evt in ECON_EVENTS:
Q:\finance-analyzer\portfolio\econ_dates.py-178-        if evt["date"] < ref_date:
Q:\finance-analyzer\portfolio\econ_dates.py-179-            continue
Q:\finance-analyzer\portfolio\econ_dates.py-180-        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:181:                                  tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py-182-        delta = evt_dt - now
Q:\finance-analyzer\portfolio\econ_dates.py-183-        hrs = delta.total_seconds() / 3600
Q:\finance-analyzer\portfolio\econ_dates.py-184-        if 0 <= hrs <= hours:
Q:\finance-analyzer\portfolio\econ_dates.py-185-            results.append({
Q:\finance-analyzer\portfolio\econ_dates.py-186-                "date": evt["date"],
Q:\finance-analyzer\portfolio\econ_dates.py-187-                "type": evt["type"],
--
Q:\finance-analyzer\portfolio\econ_dates.py-213-            wall-clock; production callers should leave this None.
Q:\finance-analyzer\portfolio\econ_dates.py-214-
Q:\finance-analyzer\portfolio\econ_dates.py-215-    Returns:
Q:\finance-analyzer\portfolio\econ_dates.py-216-        list[dict]: Events whose evt_dt is in [ref_time - hours, ref_time],
Q:\finance-analyzer\portfolio\econ_dates.py-217-            with keys date/type/impact/hours_since (positive number).
Q:\finance-analyzer\portfolio\econ_dates.py-218-    """
Q:\finance-analyzer\portfolio\econ_dates.py:219:    now = ref_time if ref_time is not None else datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py-220-    results = []
Q:\finance-analyzer\portfolio\econ_dates.py-221-    for evt in ECON_EVENTS:
Q:\finance-analyzer\portfolio\econ_dates.py-222-        if evt.get("impact") not in impact_filter:
Q:\finance-analyzer\portfolio\econ_dates.py-223-            continue
Q:\finance-analyzer\portfolio\econ_dates.py-224-        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
Q:\finance-analyzer\portfolio\econ_dates.py:225:                                  tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py-226-        delta_hrs = (now - evt_dt).total_seconds() / 3600
Q:\finance-analyzer\portfolio\econ_dates.py-227-        if 0 <= delta_hrs <= hours:
Q:\finance-analyzer\portfolio\econ_dates.py-228-            results.append({
Q:\finance-analyzer\portfolio\econ_dates.py-229-                "date": evt["date"],
Q:\finance-analyzer\portfolio\econ_dates.py-230-                "type": evt["type"],
Q:\finance-analyzer\portfolio\econ_dates.py-231-                "impact": evt["impact"],
--
Q:\finance-analyzer\portfolio\econ_dates.py-248-    envelope where technical signals (sentiment, momentum_factors,
Q:\finance-analyzer\portfolio\econ_dates.py-249-    structure, claude_fundamental) systematically misvote because price
Q:\finance-analyzer\portfolio\econ_dates.py-250-    is being driven by news, not pattern continuity.
Q:\finance-analyzer\portfolio\econ_dates.py-251-
Q:\finance-analyzer\portfolio\econ_dates.py-252-    Self-contained iteration over ``ECON_EVENTS`` so callers can pass an
Q:\finance-analyzer\portfolio\econ_dates.py-253-    explicit ``now`` for testability without monkey-patching
Q:\finance-analyzer\portfolio\econ_dates.py:254:    ``datetime.now``. ``events_within_hours`` reads wall clock directly
Q:\finance-analyzer\portfolio\econ_dates.py-255-    and is harder to stub.
Q:\finance-analyzer\portfolio\econ_dates.py-256-
Q:\finance-analyzer\portfolio\econ_dates.py-257-    Args:
Q:\finance-analyzer\portfolio\econ_dates.py:258:        now: Optional reference time (UTC). Defaults to ``datetime.now(UTC)``.
Q:\finance-analyzer\portfolio\econ_dates.py-259-        lookback_hours: Past window. ``0`` disables the backward check.
Q:\finance-analyzer\portfolio\econ_dates.py-260-        lookahead_hours: Future window. ``0`` disables the forward check.
Q:\finance-analyzer\portfolio\econ_dates.py-261-        impact_filter: Impact levels that count. Default ``("high",)``
Q:\finance-analyzer\portfolio\econ_dates.py-262-            so only FOMC/CPI/NFP trigger; GDP (medium) does not.
Q:\finance-analyzer\portfolio\econ_dates.py-263-
Q:\finance-analyzer\portfolio\econ_dates.py-264-    Returns:
Q:\finance-analyzer\portfolio\econ_dates.py-265-        True if any qualifying event lies in the combined window.
Q:\finance-analyzer\portfolio\econ_dates.py-266-    """
Q:\finance-analyzer\portfolio\econ_dates.py-267-    if now is None:
Q:\finance-analyzer\portfolio\econ_dates.py:268:        now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py-269-    for evt in ECON_EVENTS:
Q:\finance-analyzer\portfolio\econ_dates.py-270-        if evt.get("impact") not in impact_filter:
Q:\finance-analyzer\portfolio\econ_dates.py-271-            continue
Q:\finance-analyzer\portfolio\econ_dates.py-272-        evt_dt = datetime.combine(
Q:\finance-analyzer\portfolio\econ_dates.py:273:            evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC,
Q:\finance-analyzer\portfolio\econ_dates.py-274-        )
Q:\finance-analyzer\portfolio\econ_dates.py-275-        delta_hrs = (evt_dt - now).total_seconds() / 3600
Q:\finance-analyzer\portfolio\econ_dates.py-276-        # Forward window: evt is in the future, within lookahead_hours.
Q:\finance-analyzer\portfolio\econ_dates.py-277-        if 0 <= delta_hrs <= lookahead_hours:
Q:\finance-analyzer\portfolio\econ_dates.py-278-            return True
Q:\finance-analyzer\portfolio\econ_dates.py-279-        # Backward window: evt is in the past, within lookback_hours.
--
Q:\finance-analyzer\portfolio\crypto_scheduler.py-71-
Q:\finance-analyzer\portfolio\crypto_scheduler.py-72-    try:
Q:\finance-analyzer\portfolio\crypto_scheduler.py-73-        tz = zoneinfo.ZoneInfo(tz_name)
Q:\finance-analyzer\portfolio\crypto_scheduler.py-74-    except Exception:
Q:\finance-analyzer\portfolio\crypto_scheduler.py-75-        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py-76-
Q:\finance-analyzer\portfolio\crypto_scheduler.py:77:    now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\crypto_scheduler.py-78-    current_hour = now_local.hour
Q:\finance-analyzer\portfolio\crypto_scheduler.py-79-
Q:\finance-analyzer\portfolio\crypto_scheduler.py-80-    if current_hour not in hours:
Q:\finance-analyzer\portfolio\crypto_scheduler.py-81-        return False, 0
Q:\finance-analyzer\portfolio\crypto_scheduler.py-82-
Q:\finance-analyzer\portfolio\crypto_scheduler.py-83-    # Check if we already ran this hour
--
Q:\finance-analyzer\portfolio\crypto_scheduler.py-173-    # Compute ETH/BTC ratio
Q:\finance-analyzer\portfolio\crypto_scheduler.py-174-    eth_btc_ratio = round(eth_price / btc_price, 4) if btc_price > 0 else 0
Q:\finance-analyzer\portfolio\crypto_scheduler.py-175-
Q:\finance-analyzer\portfolio\crypto_scheduler.py-176-    # Get current local time
Q:\finance-analyzer\portfolio\crypto_scheduler.py-177-    try:
Q:\finance-analyzer\portfolio\crypto_scheduler.py-178-        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:179:        now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\crypto_scheduler.py-180-        time_str = now_local.strftime("%H:%M CET")
Q:\finance-analyzer\portfolio\crypto_scheduler.py-181-        date_str = now_local.strftime("%Y-%m-%d")
Q:\finance-analyzer\portfolio\crypto_scheduler.py-182-    except Exception:
Q:\finance-analyzer\portfolio\crypto_scheduler.py-183-        time_str = "?"
Q:\finance-analyzer\portfolio\crypto_scheduler.py-184-        date_str = "?"
Q:\finance-analyzer\portfolio\crypto_scheduler.py-185-
--
Q:\finance-analyzer\portfolio\crypto_scheduler.py-304-            lines.append(f"BTC levels: S ${swing_lo:,.0f} / VWAP ${vwap:,.0f} | R ${swing_hi:,.0f}")
Q:\finance-analyzer\portfolio\crypto_scheduler.py-305-
Q:\finance-analyzer\portfolio\crypto_scheduler.py-306-    msg = "\n".join(lines)
Q:\finance-analyzer\portfolio\crypto_scheduler.py-307-
Q:\finance-analyzer\portfolio\crypto_scheduler.py-308-    # Build log entry
Q:\finance-analyzer\portfolio\crypto_scheduler.py-309-    log_entry = {
Q:\finance-analyzer\portfolio\crypto_scheduler.py:310:        "ts": datetime.now().astimezone().isoformat(),
Q:\finance-analyzer\portfolio\crypto_scheduler.py-311-        "command": "crypto-scheduler",
Q:\finance-analyzer\portfolio\crypto_scheduler.py-312-        "tickers": ["BTC-USD", "ETH-USD", "MSTR"],
Q:\finance-analyzer\portfolio\crypto_scheduler.py-313-        "crypto_fear_greed": crypto_fg,
Q:\finance-analyzer\portfolio\crypto_scheduler.py-314-        "dxy": dxy_val if isinstance(dxy_val, (int, float)) else None,
Q:\finance-analyzer\portfolio\crypto_scheduler.py-315-        "fomc_days": fomc_days if isinstance(fomc_days, int) else None,
Q:\finance-analyzer\portfolio\crypto_scheduler.py-316-        "eth_btc_ratio": eth_btc_ratio,
--
Q:\finance-analyzer\portfolio\crypto_scheduler.py-370-        if log_entry:
Q:\finance-analyzer\portfolio\crypto_scheduler.py-371-            atomic_append_jsonl(LOG_FILE, log_entry)
Q:\finance-analyzer\portfolio\crypto_scheduler.py-372-
Q:\finance-analyzer\portfolio\crypto_scheduler.py-373-        # Update state
Q:\finance-analyzer\portfolio\crypto_scheduler.py-374-        try:
Q:\finance-analyzer\portfolio\crypto_scheduler.py-375-            tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:376:            now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\crypto_scheduler.py-377-            today_str = now_local.strftime("%Y-%m-%d")
Q:\finance-analyzer\portfolio\crypto_scheduler.py-378-        except Exception:
Q:\finance-analyzer\portfolio\crypto_scheduler.py-379-            today_str = ""
Q:\finance-analyzer\portfolio\crypto_scheduler.py-380-
Q:\finance-analyzer\portfolio\crypto_scheduler.py-381-        state = _get_state()
Q:\finance-analyzer\portfolio\crypto_scheduler.py-382-        state["last_report_time"] = time.time()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\data_refresh.py" in Q:\finance-analyzer
 succeeded in 696ms:
1:import time
2:from pathlib import Path
3:
4:import pandas as pd
5:
6:from portfolio.api_utils import BINANCE_BASE
7:from portfolio.http_retry import fetch_with_retry
8:
9:DATA_DIR = (
10:    Path(__file__).resolve().parent.parent
11:    / "user_data"
12:    / "data"
13:    / "binance"
14:    / "futures"
15:)
16:
17:PAIRS = {
18:    "BTC": "BTCUSDT",
19:    "ETH": "ETHUSDT",
20:}
21:
22:
23:def download_klines(symbol, interval="1h", days=365):
24:    all_data = []
25:    end_time = int(time.time() * 1000)
26:    ms_per_candle = {"1h": 3600000, "4h": 14400000, "1d": 86400000}[interval]
27:    start_time = end_time - (days * 86400000)
28:
29:    while start_time < end_time:
30:        r = fetch_with_retry(
31:            f"{BINANCE_BASE}/klines",
32:            params={
33:                "symbol": symbol,
34:                "interval": interval,
35:                "startTime": start_time,
36:                "limit": 1000,
37:            },
38:            timeout=30,
39:        )
40:        if r is None:
41:            break
42:        r.raise_for_status()
43:        batch = r.json()
44:        if not batch:
45:            break
46:        all_data.extend(batch)
47:        start_time = batch[-1][0] + ms_per_candle
48:        time.sleep(0.2)
49:
50:    df = pd.DataFrame(
51:        all_data,
52:        columns=[
53:            "open_time",
54:            "open",
55:            "high",
56:            "low",
57:            "close",
58:            "volume",
59:            "close_time",
60:            "quote_vol",
61:            "trades",
62:            "taker_base",
63:            "taker_quote",
64:            "ignore",
65:        ],
66:    )
67:    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
68:    for col in ["open", "high", "low", "close", "volume"]:
69:        df[col] = df[col].astype(float)
70:    df = (
71:        df.drop_duplicates(subset=["open_time"])
72:        .sort_values("date")
73:        .reset_index(drop=True)
74:    )
75:    return df
76:
77:
78:def refresh_all(days=365):
79:    DATA_DIR.mkdir(parents=True, exist_ok=True)
80:    for name, symbol in PAIRS.items():
81:        fname = f"{name}_USDT_USDT-1h-futures.feather"
82:        path = DATA_DIR / fname
83:        print(f"Downloading {symbol} 1h ({days}d)...", end=" ", flush=True)
84:        df = download_klines(symbol, interval="1h", days=days)
85:        df.to_feather(path)
86:        print(f"{len(df)} candles -> {path.name}")
87:
88:
89:if __name__ == "__main__":
90:    refresh_all()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\metals_cross_assets.py" in Q:\finance-analyzer
 succeeded in 745ms:
1:"""Cross-asset data for metals prediction.
2:
3:Fetches correlated markets that carry predictive information for
4:1-3h gold and silver moves:
5:    - Copper (HG=F): industrial demand proxy, correlated with silver
6:    - GVZ: CBOE Gold ETF Volatility Index (implied vol for gold)
7:    - SPY: S&P 500 ETF (risk-on/risk-off gauge)
8:    - Gold/Silver ratio: mean-reverting ratio, extreme readings signal
9:
10:All data fetched via yfinance with caching to avoid rate limits.
11:
12:2026-04-13: Added intraday (60m bar) fetchers next to the existing daily
13:ones after 4,916-sample measurement showed metals_cross_asset at 29.1%
14:on XAG 3h — root cause was 5-day lookbacks evaluated against 3h outcomes
15:(see docs/AVANZA_RESILIENCE_PLAN.md follow-up). Daily fetchers preserved
16:for longer-horizon callers; the metals_cross_asset signal switched to
17:intraday by default.
18:"""
19:from __future__ import annotations
20:
21:import logging
22:from functools import wraps
23:
24:import pandas as pd
25:
26:from portfolio.shared_state import _cached
27:
28:logger = logging.getLogger("portfolio.metals_cross_assets")
29:
30:_CROSS_TTL = 300
31:_GVZ_TTL = 600
32:# Intraday TTL is shorter — 60m bars refresh at the start of each hour,
33:# and we want to re-query shortly after the bar closes to pick up the new row.
34:_CROSS_INTRADAY_TTL = 180
35:
36:
37:def _yf_download(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
38:    """Fetch OHLCV OHLCV bars. Routed via ``portfolio.price_source``.
39:
40:    2026-04-14: no longer pinned to yfinance. The router dispatches
41:    commodity futures (HG=F, CL=F) to Binance FAPI for 7.7s-fresh data,
42:    stocks/ETFs (SPY, USO) to Alpaca, and falls back to yfinance only
43:    for tickers with no live alternative (^GVZ). Returns DataFrame with
44:    capitalized column names for backward compatibility with callers
45:    in this module that reference ``df["Close"]``.
46:    """
47:    # Rough period-to-limit mapping — price_source/Binance/Alpaca use row
48:    # limits while yfinance uses period strings. Slight over-fetch is OK.
49:    _limit_map = {
50:        "1d": 10, "5d": 120, "1mo": 30, "3mo": 90, "6mo": 180,
51:        "1y": 365, "2y": 730,
52:    }
53:    limit = _limit_map.get(period, 90)
54:
55:    try:
56:        from portfolio.price_source import fetch_klines
57:
58:        df = fetch_klines(ticker, interval=interval, limit=limit, period=period)
59:    except Exception as e:
60:        logger.warning("price_source fetch failed for %s: %s", ticker, e)
61:        return pd.DataFrame()
62:
63:    if df is None or df.empty:
64:        return pd.DataFrame()
65:
66:    # Callers in this module use capitalized column names (legacy
67:    # yfinance convention). Backends normalize to lowercase; re-capitalize
68:    # here so downstream getters ``df["Close"]`` keep working.
69:    rename = {
70:        "open": "Open", "high": "High", "low": "Low",
71:        "close": "Close", "volume": "Volume",
72:    }
73:    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
74:
75:
76:def _nocache(func):
77:    """Mark function so tests can bypass _cached via func.__wrapped__."""
78:    @wraps(func)
79:    def wrapper(*args, **kwargs):
80:        return func(*args, **kwargs)
81:    wrapper.__wrapped__ = func
82:    return wrapper
83:
84:
85:def _pct_change(series: pd.Series, periods: int) -> float:
86:    """Percentage change over N periods, returns NaN on insufficient data."""
87:    if len(series) < periods + 1:
88:        return float("nan")
89:    return float((series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100)
90:
91:
92:@_nocache
93:def get_copper_data() -> dict | None:
94:    """Copper futures (HG=F) price and momentum."""
95:    def _fetch():
96:        df = _yf_download("HG=F", period="3mo", interval="1d")
97:        if df.empty or "Close" not in df.columns:
98:            return None
99:        close = df["Close"].dropna()
100:        if len(close) < 20:
101:            return None
102:        return {
103:            "price": float(close.iloc[-1]),
104:            "change_1d_pct": _pct_change(close, 1),
105:            "change_5d_pct": _pct_change(close, 5),
106:            "sma20": float(close.rolling(20).mean().iloc[-1]),
107:            "vs_sma20_pct": float((close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1) * 100),
108:        }
109:    return _cached("cross_copper", _CROSS_TTL, _fetch)
110:
111:
112:@_nocache
113:def get_gvz() -> dict | None:
114:    """CBOE Gold ETF Volatility Index (^GVZ)."""
115:    def _fetch():
116:        df = _yf_download("^GVZ", period="3mo", interval="1d")
117:        if df.empty or "Close" not in df.columns:
118:            return None
119:        close = df["Close"].dropna()
120:        if len(close) < 10:
121:            return None
122:        level = float(close.iloc[-1])
123:        mean20 = float(close.rolling(20, min_periods=10).mean().iloc[-1])
124:        std20 = float(close.rolling(20, min_periods=10).std().iloc[-1])
125:        zscore = (level - mean20) / std20 if std20 > 0.01 else 0.0
126:        return {
127:            "level": level,
128:            "change_1d_pct": _pct_change(close, 1),
129:            "sma20": mean20,
130:            "zscore": zscore,
131:        }
132:    return _cached("cross_gvz", _GVZ_TTL, _fetch)
133:
134:
135:@_nocache
136:def get_gold_silver_ratio() -> dict | None:
137:    """Gold/Silver price ratio and deviation from mean."""
138:    def _fetch():
139:        gold_df = _yf_download("GC=F", period="6mo", interval="1d")
140:        silver_df = _yf_download("SI=F", period="6mo", interval="1d")
141:        if gold_df.empty or silver_df.empty:
142:            return None
143:        gold_close = gold_df["Close"].dropna()
144:        silver_close = silver_df["Close"].dropna()
145:        if len(gold_close) < 20 or len(silver_close) < 20:
146:            return None
147:        common = gold_close.index.intersection(silver_close.index)
148:        if len(common) < 20:
149:            return None
150:        g = gold_close.loc[common]
151:        s = silver_close.loc[common]
152:        ratio = g / s
153:        current = float(ratio.iloc[-1])
154:        sma20 = float(ratio.rolling(20).mean().iloc[-1])
155:        std20 = float(ratio.rolling(20).std().iloc[-1])
156:        zscore = (current - sma20) / std20 if std20 > 0.01 else 0.0
157:        return {
158:            "ratio": current,
159:            "sma20": sma20,
160:            "zscore": zscore,
161:            "change_5d_pct": _pct_change(ratio, 5),
162:        }
163:    return _cached("cross_gs_ratio", _CROSS_TTL, _fetch)
164:
165:
166:@_nocache
167:def get_oil_data() -> dict | None:
168:    """WTI Crude Oil futures (CL=F) price and momentum."""
169:    def _fetch():
170:        df = _yf_download("CL=F", period="3mo", interval="1d")
171:        if df.empty or "Close" not in df.columns:
172:            return None
173:        close = df["Close"].dropna()
174:        if len(close) < 10:
175:            return None
176:        return {
177:            "price": float(close.iloc[-1]),
178:            "change_1d_pct": _pct_change(close, 1),
179:            "change_5d_pct": _pct_change(close, 5),
180:        }
181:    return _cached("cross_oil", _CROSS_TTL, _fetch)
182:
183:
184:@_nocache
185:def get_spy_return() -> dict | None:
186:    """S&P 500 ETF (SPY) recent returns for risk-on/risk-off."""
187:    def _fetch():
188:        df = _yf_download("SPY", period="1mo", interval="1d")
189:        if df.empty or "Close" not in df.columns:
190:            return None
191:        close = df["Close"].dropna()
192:        if len(close) < 5:
193:            return None
194:        return {
195:            "price": float(close.iloc[-1]),
196:            "change_1d_pct": _pct_change(close, 1),
197:            "change_5d_pct": _pct_change(close, 5),
198:        }
199:    return _cached("cross_spy", _CROSS_TTL, _fetch)
200:
201:
202:def get_all_cross_asset_data() -> dict:
203:    """Fetch all cross-asset features in one call (daily bars)."""
204:    return {
205:        "copper": get_copper_data(),
206:        "gvz": get_gvz(),
207:        "gold_silver_ratio": get_gold_silver_ratio(),
208:        "spy": get_spy_return(),
209:        "oil": get_oil_data(),
210:    }
211:
212:
213:# --- Intraday variants (60m bars, for 1-3h prediction horizons) ---
214:#
215:# yfinance 60m interval supports up to 730 days of history. We use 5d
216:# period which yields ~35 hourly bars — enough for 3h change (3 bars) and
217:# intraday rolling stats. On weekends/holidays the last ~2 days of bars
218:# may be sparse; `_pct_change` returns NaN and signal votes HOLD.
219:
220:
221:@_nocache
222:def get_copper_intraday() -> dict | None:
223:    """Copper 60m bars. Exposes change_1h_pct + change_3h_pct."""
224:    def _fetch():
225:        df = _yf_download("HG=F", period="5d", interval="60m")
226:        if df.empty or "Close" not in df.columns:
227:            return None
228:        close = df["Close"].dropna()
229:        if len(close) < 4:
230:            return None
231:        return {
232:            "price": float(close.iloc[-1]),
233:            "change_1h_pct": _pct_change(close, 1),
234:            "change_3h_pct": _pct_change(close, 3),
235:        }
236:    return _cached("cross_copper_intraday", _CROSS_INTRADAY_TTL, _fetch)
237:
238:
239:@_nocache
240:def get_gold_silver_ratio_intraday() -> dict | None:
241:    """Gold/Silver ratio 60m bars. Exposes ratio_change_3h_pct."""
242:    def _fetch():
243:        gold_df = _yf_download("GC=F", period="5d", interval="60m")
244:        silver_df = _yf_download("SI=F", period="5d", interval="60m")
245:        if gold_df.empty or silver_df.empty:
246:            return None
247:        gold_close = gold_df["Close"].dropna()
248:        silver_close = silver_df["Close"].dropna()
249:        if len(gold_close) < 4 or len(silver_close) < 4:
250:            return None
251:        common = gold_close.index.intersection(silver_close.index)
252:        if len(common) < 4:
253:            return None
254:        ratio = gold_close.loc[common] / silver_close.loc[common]
255:        return {
256:            "ratio": float(ratio.iloc[-1]),
257:            "change_1h_pct": _pct_change(ratio, 1),
258:            "change_3h_pct": _pct_change(ratio, 3),
259:        }
260:    return _cached("cross_gs_ratio_intraday", _CROSS_INTRADAY_TTL, _fetch)
261:
262:
263:@_nocache
264:def get_oil_intraday() -> dict | None:
265:    """WTI crude 60m bars."""
266:    def _fetch():
267:        df = _yf_download("CL=F", period="5d", interval="60m")
268:        if df.empty or "Close" not in df.columns:
269:            return None
270:        close = df["Close"].dropna()
271:        if len(close) < 4:
272:            return None
273:        return {
274:            "price": float(close.iloc[-1]),
275:            "change_1h_pct": _pct_change(close, 1),
276:            "change_3h_pct": _pct_change(close, 3),
277:        }
278:    return _cached("cross_oil_intraday", _CROSS_INTRADAY_TTL, _fetch)
279:
280:
281:@_nocache
282:def get_spy_intraday() -> dict | None:
283:    """SPY 60m bars — captures intraday risk-on/risk-off."""
284:    def _fetch():
285:        df = _yf_download("SPY", period="5d", interval="60m")
286:        if df.empty or "Close" not in df.columns:
287:            return None
288:        close = df["Close"].dropna()
289:        if len(close) < 4:
290:            return None
291:        return {
292:            "price": float(close.iloc[-1]),
293:            "change_1h_pct": _pct_change(close, 1),
294:            "change_3h_pct": _pct_change(close, 3),
295:        }
296:    return _cached("cross_spy_intraday", _CROSS_INTRADAY_TTL, _fetch)
297:
298:
299:def get_all_cross_asset_intraday() -> dict:
300:    """Fetch all intraday (60m) cross-asset features in one call.
301:
302:    GVZ is intentionally absent — it's a daily-published index with no
303:    intraday bars. Callers should still read `get_gvz()` for GVZ context.
304:    """
305:    return {
306:        "copper": get_copper_intraday(),
307:        "gold_silver_ratio": get_gold_silver_ratio_intraday(),
308:        "spy": get_spy_intraday(),
309:        "oil": get_oil_intraday(),
310:    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"def _is_us_dst|US DST|dst\" Q:\\finance-analyzer\\portfolio\\market_timing.py" in Q:\finance-analyzer
 succeeded in 740ms:
23-
24-# States where US stock markets are NOT trading — use this tuple instead of
25-# hardcoding ("closed", "weekend") to avoid missing the "holiday" state.
26-MARKET_CLOSED_STATES = ("closed", "weekend", "holiday")
27-
28-
29:def _is_eu_dst(dt):
30-    """Check if a UTC datetime falls within EU Summer Time (CEST).
31-
32-    EU DST rule:
33-      Starts: last Sunday of March at 01:00 UTC
34-      Ends:   last Sunday of October at 01:00 UTC
35-
--
37-    """
38-    year = dt.year
39-
40-    # Last Sunday of March
41-    mar31 = date(year, 3, 31)
42-    last_sun_mar = 31 - (mar31.weekday() + 1) % 7
43:    eu_dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)
44-
45-    # Last Sunday of October
46-    oct31 = date(year, 10, 31)
47-    last_sun_oct = 31 - (oct31.weekday() + 1) % 7
48:    eu_dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)
49-
50:    return eu_dst_start <= dt < eu_dst_end
51-
52-
53-def _eu_market_open_hour_utc(dt):
54-    """Return the EU market open hour in UTC, adjusted for EU DST.
55-
56-    H47: London/Frankfurt open at 08:00 local time.
57-    CEST (summer, BST=UTC+1): 08:00 local = 07:00 UTC
58-    CET (winter, GMT=UTC+0): 08:00 local = 08:00 UTC
59-
60-    Previously hardcoded to 7 UTC year-round, which missed the winter hour.
61-    """
62:    if _is_eu_dst(dt):
63-        return 7
64-    return 8
65-
66-
67:def _is_us_dst(dt):
68-    """Check if a UTC datetime falls within US Eastern Daylight Time (EDT).
69-
70:    US DST rule (since 2007):
71-      Starts: second Sunday of March at 02:00 local (07:00 UTC)
72-      Ends:   first Sunday of November at 02:00 local (06:00 UTC)
73-
74-    Returns True during EDT (Mar-Nov), False during EST (Nov-Mar).
75-    """
76-    year = dt.year
77-
78-    # Second Sunday of March
79-    mar1_wd = date(year, 3, 1).weekday()  # 0=Mon..6=Sun
80-    first_sun_mar = 1 + (6 - mar1_wd) % 7
81-    second_sun_mar = first_sun_mar + 7
82:    dst_start = datetime(year, 3, second_sun_mar, 7, 0, tzinfo=UTC)
83-
84-    # First Sunday of November
85-    nov1_wd = date(year, 11, 1).weekday()
86-    first_sun_nov = 1 + (6 - nov1_wd) % 7
87:    dst_end = datetime(year, 11, first_sun_nov, 6, 0, tzinfo=UTC)
88-
89:    return dst_start <= dt < dst_end
90-
91-
92-def _market_close_hour_utc(dt):
93-    """Return the NYSE close hour in UTC, adjusted for DST.
94-
95-    NYSE closes at 16:00 ET.
96-    EDT (Mar-Nov): 16:00 ET = 20:00 UTC
97-    EST (Nov-Mar): 16:00 ET = 21:00 UTC
98-    """
99:    if _is_us_dst(dt):
100-        return 20
101-    return 21
102-
103-
104-# ---------------------------------------------------------------------------
105-# Holiday calendars
--
264-    """Return the NYSE open hour in UTC, adjusted for DST.
265-
266-    NYSE opens at 09:30 ET.
267-    EDT (Mar-Nov): 09:30 ET = 13:30 UTC -> hour 13
268-    EST (Nov-Mar): 09:30 ET = 14:30 UTC -> hour 14
269-    """
270:    if _is_us_dst(dt):
271-        return 13
272-    return 14
273-
274-
275-def is_us_stock_market_open(now=None, pre_market_buffer_min=0, post_market_buffer_min=0):
276-    """Check if US stock market (NYSE) is currently open.
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\seasonality.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\seasonality_updater.py" in Q:\finance-analyzer
 succeeded in 720ms:
1:"""Intraday seasonality detrending for metals and crypto.
2:
3:Computes average return and volatility profiles by hour-of-day from
4:historical data, then subtracts these patterns from current observations
5:to isolate non-seasonal signal content.
6:
7:Research basis: Smales & Yang (2015) — removing day-cycle (detrending)
8:sharpens short-term signals for gold and silver.
9:"""
10:from __future__ import annotations
11:
12:import logging
13:from pathlib import Path
14:
15:import pandas as pd
16:
17:from portfolio.file_utils import atomic_write_json, load_json
18:
19:logger = logging.getLogger("portfolio.seasonality")
20:
21:_BASE_DIR = Path(__file__).resolve().parent.parent
22:_STATE_FILE = _BASE_DIR / "data" / "seasonality_profiles.json"
23:_MIN_DAYS = 5  # minimum trading days to compute profiles
24:
25:
26:def compute_hourly_profile(klines_1h: pd.DataFrame) -> dict | None:
27:    """Compute average return and volatility by hour-of-day.
28:
29:    Args:
30:        klines_1h: DataFrame with 'close' column and DatetimeIndex (1h bars).
31:                   Needs at least _MIN_DAYS * 24 rows.
32:
33:    Returns:
34:        Dict keyed by hour (0-23), each with mean_return and mean_volatility.
35:        None if insufficient data.
36:    """
37:    if klines_1h is None or len(klines_1h) < _MIN_DAYS * 24:
38:        return None
39:
40:    df = klines_1h.copy()
41:    df["return"] = df["close"].pct_change()
42:    df["abs_return"] = df["return"].abs()
43:
44:    # Extract hour from index
45:    if hasattr(df.index, "hour"):
46:        df["hour"] = df.index.hour
47:    else:
48:        return None
49:
50:    # Group by hour and compute mean return + mean absolute return (vol proxy)
51:    grouped = df.groupby("hour").agg(
52:        mean_return=("return", "mean"),
53:        mean_volatility=("abs_return", "mean"),
54:        count=("return", "count"),
55:    )
56:
57:    profile = {}
58:    for hour in range(24):
59:        if hour in grouped.index:
60:            row = grouped.loc[hour]
61:            profile[str(hour)] = {
62:                "mean_return": float(row["mean_return"]),
63:                "mean_volatility": float(row["mean_volatility"]),
64:                "count": int(row["count"]),
65:            }
66:        else:
67:            profile[str(hour)] = {
68:                "mean_return": 0.0,
69:                "mean_volatility": 0.0,
70:                "count": 0,
71:            }
72:
73:    return profile
74:
75:
76:def detrend_return(raw_return: float, hour: int, profile: dict) -> float:
77:    """Remove seasonal component from a return observation.
78:
79:    Args:
80:        raw_return: The observed return (e.g. 0.002 for 0.2%).
81:        hour: Hour of day (0-23, UTC).
82:        profile: Hourly profile dict from compute_hourly_profile.
83:
84:    Returns:
85:        Detrended return: raw_return - mean_return_for_hour.
86:    """
87:    if profile is None:
88:        return raw_return
89:    entry = profile.get(str(hour))
90:    if entry is None:
91:        return raw_return
92:    return raw_return - entry["mean_return"]
93:
94:
95:def normalize_volatility(raw_vol: float, hour: int, profile: dict) -> float:
96:    """Normalize volatility by dividing by the seasonal average for this hour.
97:
98:    Args:
99:        raw_vol: Observed absolute return or volatility measure.
100:        hour: Hour of day (0-23, UTC).
101:        profile: Hourly profile dict from compute_hourly_profile.
102:
103:    Returns:
104:        Normalized volatility (1.0 = average for this hour).
105:        Returns raw_vol if profile unavailable.
106:    """
107:    if profile is None:
108:        return raw_vol
109:    entry = profile.get(str(hour))
110:    if entry is None or entry["mean_volatility"] < 1e-10:
111:        return raw_vol
112:    return raw_vol / entry["mean_volatility"]
113:
114:
115:def save_profiles(profiles: dict[str, dict]) -> None:
116:    """Persist ticker-keyed profiles to disk.
117:
118:    Args:
119:        profiles: Dict keyed by ticker, each value is an hourly profile.
120:    """
121:    atomic_write_json(_STATE_FILE, profiles)
122:
123:
124:def load_profiles() -> dict:
125:    """Load persisted profiles from disk."""
126:    return load_json(_STATE_FILE) or {}
127:
128:
129:def get_profile(ticker: str) -> dict | None:
130:    """Load the hourly profile for a specific ticker."""
131:    profiles = load_profiles()
132:    return profiles.get(ticker)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 718ms:
1:"""Compute and persist intraday seasonality profiles for metals.
2:
3:Fetches 1h klines from Binance FAPI and computes hour-of-day
4:return/volatility profiles. Called periodically to keep profiles fresh.
5:"""
6:from __future__ import annotations
7:
8:import logging
9:
10:from portfolio.metals_orderbook import SYMBOL_MAP
11:from portfolio.seasonality import compute_hourly_profile, load_profiles, save_profiles
12:
13:logger = logging.getLogger("portfolio.seasonality_updater")
14:
15:
16:def update_seasonality_profiles(tickers: list[str] | None = None) -> dict:
17:    """Fetch klines and recompute seasonality profiles for metals tickers.
18:
19:    Args:
20:        tickers: List of tickers to update. Defaults to XAG-USD, XAU-USD.
21:
22:    Returns:
23:        Dict of ticker -> profile (or None if failed).
24:    """
25:    if tickers is None:
26:        tickers = ["XAG-USD", "XAU-USD"]
27:
28:    profiles = load_profiles()
29:
30:    for ticker in tickers:
31:        try:
32:            df = _fetch_hourly_klines(ticker, limit=500)  # ~20 days
33:            if df is None or df.empty:
34:                logger.warning("No klines for %s, skipping", ticker)
35:                continue
36:            profile = compute_hourly_profile(df)
37:            if profile is not None:
38:                profiles[ticker] = profile
39:                logger.info("Updated seasonality profile for %s (%d hours)",
40:                           ticker, len(profile))
41:            else:
42:                logger.warning("Insufficient data for %s profile", ticker)
43:        except Exception as e:
44:            logger.warning("Seasonality update failed for %s: %s", ticker, e)
45:
46:    if profiles:
47:        save_profiles(profiles)
48:    return profiles
49:
50:
51:def _fetch_hourly_klines(ticker: str, limit: int = 500):
52:    """Fetch 1h klines from Binance FAPI for a metals ticker."""
53:    import pandas as pd
54:
55:    from portfolio.api_utils import BINANCE_FAPI_BASE
56:    from portfolio.http_retry import fetch_json
57:    from portfolio.shared_state import _binance_limiter
58:
59:    symbol = SYMBOL_MAP.get(ticker)
60:    if not symbol:
61:        return None
62:
63:    _binance_limiter.wait()
64:    data = fetch_json(
65:        f"{BINANCE_FAPI_BASE}/klines",
66:        params={"symbol": symbol, "interval": "1h", "limit": limit},
67:        timeout=15,
68:        label="seasonality_klines",
69:    )
70:    if not data:
71:        return None
72:
73:    rows = []
74:    for k in data:
75:        rows.append({
76:            "time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
77:            "open": float(k[1]),
78:            "high": float(k[2]),
79:            "low": float(k[3]),
80:            "close": float(k[4]),
81:            "volume": float(k[5]),
82:        })
83:    df = pd.DataFrame(rows)
84:    df = df.set_index("time")
85:    return df
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"load_persistent_cache\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 767ms:
Q:\finance-analyzer\portfolio\alpha_vantage.py:36:def load_persistent_cache():
Q:\finance-analyzer\portfolio\main.py:1171:        load_persistent_cache()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\fomc_dates.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^" Q:'"\\finance-analyzer\\portfolio\\econ_dates.py" in Q:\finance-analyzer
 succeeded in 962ms:
1:"""Shared FOMC meeting date constants.
2:
3:Single source of truth for all modules that need FOMC dates.
4:Each two-day meeting is listed as (start_date, announcement_date).
5:The announcement (rate decision) is always on day 2.
6:
7:Sources: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
8:"""
9:
10:from datetime import date
11:
12:# 2026 FOMC meeting dates (two-day meetings)
13:FOMC_DATES_2026 = [
14:    date(2026, 1, 28), date(2026, 1, 29),
15:    date(2026, 3, 17), date(2026, 3, 18),
16:    date(2026, 4, 28), date(2026, 4, 29),
17:    date(2026, 6, 16), date(2026, 6, 17),
18:    date(2026, 7, 28), date(2026, 7, 29),
19:    date(2026, 9, 15), date(2026, 9, 16),
20:    date(2026, 10, 27), date(2026, 10, 28),
21:    date(2026, 12, 8), date(2026, 12, 9),
22:]
23:
24:# 2027 FOMC meeting dates (two-day meetings)
25:FOMC_DATES_2027 = [
26:    date(2027, 1, 26), date(2027, 1, 27),
27:    date(2027, 3, 16), date(2027, 3, 17),
28:    date(2027, 4, 27), date(2027, 4, 28),
29:    date(2027, 6, 8), date(2027, 6, 9),
30:    date(2027, 7, 27), date(2027, 7, 28),
31:    date(2027, 9, 14), date(2027, 9, 15),
32:    date(2027, 10, 26), date(2027, 10, 27),
33:    date(2027, 12, 7), date(2027, 12, 8),
34:]
35:
36:# Combined list as ISO strings (for macro_context.py which uses string comparison)
37:FOMC_DATES_ISO = [d.isoformat() for d in FOMC_DATES_2026 + FOMC_DATES_2027]
38:
39:# Announcement dates only (day 2 of each meeting — the rate decision day)
40:FOMC_ANNOUNCEMENT_DATES = [
41:    date(2026, 1, 29),
42:    date(2026, 3, 18),
43:    date(2026, 4, 29),
44:    date(2026, 6, 17),
45:    date(2026, 7, 29),
46:    date(2026, 9, 16),
47:    date(2026, 10, 28),
48:    date(2026, 12, 9),
49:    date(2027, 1, 27),
50:    date(2027, 3, 17),
51:    date(2027, 4, 28),
52:    date(2027, 6, 9),
53:    date(2027, 7, 28),
54:    date(2027, 9, 15),
55:    date(2027, 10, 27),
56:    date(2027, 12, 8),
57:]
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 950ms:
1:"""Economic calendar dates for 2026-2027.
2:
3:Follows the same pattern as fomc_dates.py — hard-coded dates as a single
4:source of truth.  Used by the econ_calendar signal to gauge event proximity.
5:
6:Sources:
7:- CPI: Bureau of Labor Statistics release schedule
8:- NFP: Bureau of Labor Statistics (first Friday of each month)
9:- GDP: Bureau of Economic Analysis advance estimate schedule
10:- FOMC: imported from fomc_dates.py (not duplicated here)
11:"""
12:
13:from __future__ import annotations
14:
15:from datetime import UTC, date, datetime
16:
17:from portfolio.fomc_dates import FOMC_ANNOUNCEMENT_DATES
18:
19:# ---------------------------------------------------------------------------
20:# CPI release dates 2026 (8:30 AM ET, typically 2nd or 3rd week)
21:# ---------------------------------------------------------------------------
22:
23:CPI_DATES_2026 = [
24:    date(2026, 1, 14),   # Dec 2025 CPI
25:    date(2026, 2, 12),   # Jan 2026 CPI
26:    date(2026, 3, 11),   # Feb 2026 CPI
27:    date(2026, 4, 10),   # Mar 2026 CPI
28:    date(2026, 5, 13),   # Apr 2026 CPI
29:    date(2026, 6, 10),   # May 2026 CPI
30:    date(2026, 7, 14),   # Jun 2026 CPI
31:    date(2026, 8, 12),   # Jul 2026 CPI
32:    date(2026, 9, 11),   # Aug 2026 CPI
33:    date(2026, 10, 13),  # Sep 2026 CPI
34:    date(2026, 11, 12),  # Oct 2026 CPI
35:    date(2026, 12, 10),  # Nov 2026 CPI
36:]
37:
38:CPI_DATES_2027 = [
39:    date(2027, 1, 13),
40:    date(2027, 2, 10),
41:    date(2027, 3, 10),
42:    date(2027, 4, 13),
43:    date(2027, 5, 12),
44:    date(2027, 6, 10),
45:    date(2027, 7, 14),
46:    date(2027, 8, 11),
47:    date(2027, 9, 15),
48:    date(2027, 10, 13),
49:    date(2027, 11, 10),
50:    date(2027, 12, 10),
51:]
52:
53:# ---------------------------------------------------------------------------
54:# Non-Farm Payrolls (first Friday of each month, 8:30 AM ET)
55:# ---------------------------------------------------------------------------
56:
57:NFP_DATES_2026 = [
58:    date(2026, 1, 2),
59:    date(2026, 2, 6),
60:    date(2026, 3, 6),
61:    date(2026, 4, 2),   # BLS released April 2026 NFP on Apr 2 (Apr 3 = Good Friday, market closed)
62:    date(2026, 5, 1),
63:    date(2026, 6, 5),
64:    date(2026, 7, 2),
65:    date(2026, 8, 7),
66:    date(2026, 9, 4),
67:    date(2026, 10, 2),
68:    date(2026, 11, 6),
69:    date(2026, 12, 4),
70:]
71:
72:NFP_DATES_2027 = [
73:    date(2027, 1, 8),
74:    date(2027, 2, 5),
75:    date(2027, 3, 5),
76:    date(2027, 4, 2),
77:    date(2027, 5, 7),
78:    date(2027, 6, 4),
79:    date(2027, 7, 2),
80:    date(2027, 8, 6),
81:    date(2027, 9, 3),
82:    date(2027, 10, 1),
83:    date(2027, 11, 5),
84:    date(2027, 12, 3),
85:]
86:
87:# ---------------------------------------------------------------------------
88:# GDP advance estimate dates (quarterly, ~4th week of month after quarter end)
89:# ---------------------------------------------------------------------------
90:
91:GDP_DATES_2026 = [
92:    date(2026, 1, 29),   # Q4 2025 advance
93:    date(2026, 4, 29),   # Q1 2026 advance
94:    date(2026, 7, 30),   # Q2 2026 advance
95:    date(2026, 10, 29),  # Q3 2026 advance
96:]
97:
98:GDP_DATES_2027 = [
99:    date(2027, 1, 28),   # Q4 2026 advance
100:    date(2027, 4, 29),   # Q1 2027 advance
101:    date(2027, 7, 29),   # Q2 2027 advance
102:    date(2027, 10, 28),  # Q3 2027 advance
103:]
104:
105:# ---------------------------------------------------------------------------
106:# Unified event list
107:# ---------------------------------------------------------------------------
108:
109:# Impact levels: "high" (FOMC, CPI, NFP) or "medium" (GDP)
110:# Affected sectors: which sectors are most impacted
111:
112:EVENT_SECTOR_MAP = {
113:    "FOMC": {"crypto", "metals", "big_tech", "etf"},
114:    "CPI": {"crypto", "metals", "big_tech", "etf"},
115:    "NFP": {"etf", "big_tech"},
116:    "GDP": {"etf", "big_tech"},
117:}
118:
119:
120:def _build_events() -> list[dict]:
121:    """Build sorted list of all economic events."""
122:    events = []
123:
124:    for d in FOMC_ANNOUNCEMENT_DATES:
125:        events.append({"date": d, "type": "FOMC", "impact": "high"})
126:
127:    for d in CPI_DATES_2026 + CPI_DATES_2027:
128:        events.append({"date": d, "type": "CPI", "impact": "high"})
129:
130:    for d in NFP_DATES_2026 + NFP_DATES_2027:
131:        events.append({"date": d, "type": "NFP", "impact": "high"})
132:
133:    for d in GDP_DATES_2026 + GDP_DATES_2027:
134:        events.append({"date": d, "type": "GDP", "impact": "medium"})
135:
136:    events.sort(key=lambda e: e["date"])
137:    return events
138:
139:
140:ECON_EVENTS = _build_events()
141:
142:
143:def next_event(ref_date: date | None = None) -> dict | None:
144:    """Return the next economic event on or after ref_date.
145:
146:    Returns dict with keys: date, type, impact, hours_until.
147:    Returns None if no future events in the calendar.
148:    """
149:    if ref_date is None:
150:        ref_date = datetime.now(UTC).date()
151:
152:    for evt in ECON_EVENTS:
153:        if evt["date"] >= ref_date:
154:            # Calculate hours until (approximate: assume 14:00 UTC release)
155:            evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
156:                                      tzinfo=UTC)
157:            # M5: Use datetime.now(UTC) so past-today events show hours_until=0.
158:            now = datetime.now(UTC)
159:            delta = evt_dt - now
160:            hours = max(0.0, delta.total_seconds() / 3600)
161:            return {
162:                "date": evt["date"],
163:                "type": evt["type"],
164:                "impact": evt["impact"],
165:                "hours_until": round(hours, 1),
166:            }
167:    return None
168:
169:
170:def events_within_hours(hours: float, ref_date: date | None = None) -> list[dict]:
171:    """Return all events within the given hours from ref_date."""
172:    if ref_date is None:
173:        ref_date = datetime.now(UTC).date()
174:
175:    results = []
176:    now = datetime.now(UTC)
177:    for evt in ECON_EVENTS:
178:        if evt["date"] < ref_date:
179:            continue
180:        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
181:                                  tzinfo=UTC)
182:        delta = evt_dt - now
183:        hrs = delta.total_seconds() / 3600
184:        if 0 <= hrs <= hours:
185:            results.append({
186:                "date": evt["date"],
187:                "type": evt["type"],
188:                "impact": evt["impact"],
189:                "hours_until": round(hrs, 1),
190:            })
191:    return results
192:
193:
194:def recent_high_impact_events(hours: float, impact_filter=("high",),
195:                              ref_time: datetime | None = None) -> list[dict]:
196:    """Return high-impact events that occurred within the last `hours`.
197:
198:    BUG-178/W15-W16 follow-up (2026-04-16): events_within_hours() above
199:    only iterates FUTURE events (skips evt["date"] < ref_date). The
200:    accuracy degradation tracker needs the BACKWARD window too so it
201:    can blanket-suppress alerts during the post-event volatility hangover
202:    after FOMC/CPI/NFP — those releases routinely whipsaw signals for
203:    12-24 hours after the print, exactly when the lifetime-vs-recent
204:    delta would otherwise look like a degradation.
205:
206:    Args:
207:        hours: How far back to look. Typical: 24.0.
208:        impact_filter: Tuple of impact levels to include. Default is
209:            ("high",) — minor data releases shouldn't silence the
210:            tracker.
211:        ref_time: Optional reference time for the "now" anchor.
212:            Tests pass an explicit value so assertions don't depend on
213:            wall-clock; production callers should leave this None.
214:
215:    Returns:
216:        list[dict]: Events whose evt_dt is in [ref_time - hours, ref_time],
217:            with keys date/type/impact/hours_since (positive number).
218:    """
219:    now = ref_time if ref_time is not None else datetime.now(UTC)
220:    results = []
221:    for evt in ECON_EVENTS:
222:        if evt.get("impact") not in impact_filter:
223:            continue
224:        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
225:                                  tzinfo=UTC)
226:        delta_hrs = (now - evt_dt).total_seconds() / 3600
227:        if 0 <= delta_hrs <= hours:
228:            results.append({
229:                "date": evt["date"],
230:                "type": evt["type"],
231:                "impact": evt["impact"],
232:                "hours_since": round(delta_hrs, 1),
233:            })
234:    return results
235:
236:
237:def is_macro_window(
238:    now: datetime | None = None,
239:    lookback_hours: float = 24.0,
240:    lookahead_hours: float = 72.0,
241:    impact_filter: tuple[str, ...] = ("high",),
242:) -> bool:
243:    """True iff a macro event matching ``impact_filter`` falls within
244:    ``lookback_hours`` past or ``lookahead_hours`` future of ``now``.
245:
246:    Wider window than ``_is_econ_blackout`` (24h ± in accuracy_degradation)
247:    — captures the full pre-event risk-off + post-event volatility-hangover
248:    envelope where technical signals (sentiment, momentum_factors,
249:    structure, claude_fundamental) systematically misvote because price
250:    is being driven by news, not pattern continuity.
251:
252:    Self-contained iteration over ``ECON_EVENTS`` so callers can pass an
253:    explicit ``now`` for testability without monkey-patching
254:    ``datetime.now``. ``events_within_hours`` reads wall clock directly
255:    and is harder to stub.
256:
257:    Args:
258:        now: Optional reference time (UTC). Defaults to ``datetime.now(UTC)``.
259:        lookback_hours: Past window. ``0`` disables the backward check.
260:        lookahead_hours: Future window. ``0`` disables the forward check.
261:        impact_filter: Impact levels that count. Default ``("high",)``
262:            so only FOMC/CPI/NFP trigger; GDP (medium) does not.
263:
264:    Returns:
265:        True if any qualifying event lies in the combined window.
266:    """
267:    if now is None:
268:        now = datetime.now(UTC)
269:    for evt in ECON_EVENTS:
270:        if evt.get("impact") not in impact_filter:
271:            continue
272:        evt_dt = datetime.combine(
273:            evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC,
274:        )
275:        delta_hrs = (evt_dt - now).total_seconds() / 3600
276:        # Forward window: evt is in the future, within lookahead_hours.
277:        if 0 <= delta_hrs <= lookahead_hours:
278:            return True
279:        # Backward window: evt is in the past, within lookback_hours.
280:        if -lookback_hours <= delta_hrs < 0:
281:            return True
282:    return False
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"get_open_interest_history\\(|get_long_short_ratio\\(|get_top_trader_position_ratio\\(|get_top_trader_account_ratio\\(|get_funding_rate_history\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 657ms:
Q:\finance-analyzer\portfolio\futures_data.py:58:def get_open_interest_history(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:86:def get_long_short_ratio(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:115:def get_top_trader_position_ratio(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:144:def get_top_trader_account_ratio(ticker, period="5m", limit=30):
Q:\finance-analyzer\portfolio\futures_data.py:173:def get_funding_rate_history(ticker, limit=100):
Q:\finance-analyzer\portfolio\futures_data.py:212:        "oi_history": get_open_interest_history(ticker),
Q:\finance-analyzer\portfolio\futures_data.py:213:        "ls_ratio": get_long_short_ratio(ticker),
Q:\finance-analyzer\portfolio\futures_data.py:214:        "top_position_ratio": get_top_trader_position_ratio(ticker),
Q:\finance-analyzer\portfolio\futures_data.py:215:        "top_account_ratio": get_top_trader_account_ratio(ticker),
Q:\finance-analyzer\portfolio\futures_data.py:216:        "funding_history": get_funding_rate_history(ticker),
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"fetch_usd_sek\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 642ms:
Q:\finance-analyzer\portfolio\elongir\data_provider.py:89:        rate = fetch_usd_sek()
Q:\finance-analyzer\portfolio\fin_fish.py:311:        return fetch_usd_sek()
Q:\finance-analyzer\portfolio\fin_snipe_manager.py:433:        live_usdsek = fetch_usd_sek() or 10.85
Q:\finance-analyzer\portfolio\fx_rates.py:28:def fetch_usd_sek():
Q:\finance-analyzer\portfolio\golddigger\data_provider.py:228:        return fetch_usd_sek()
Q:\finance-analyzer\portfolio\iskbets.py:753:        fx_rate = fetch_usd_sek()
Q:\finance-analyzer\portfolio\iskbets.py:885:        fx_rate = fetch_usd_sek()
Q:\finance-analyzer\portfolio\main.py:449:    fx_rate = fetch_usd_sek()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"is_macro_window\\(|next_event\\(|events_within_hours\\(|recent_high_impact_events\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 743ms:
Q:\finance-analyzer\portfolio\accuracy_degradation.py:297:        forward = events_within_hours(ECON_BLACKOUT_HOURS)
Q:\finance-analyzer\portfolio\accuracy_degradation.py:300:        backward = recent_high_impact_events(
Q:\finance-analyzer\portfolio\econ_dates.py:143:def next_event(ref_date: date | None = None) -> dict | None:
Q:\finance-analyzer\portfolio\econ_dates.py:170:def events_within_hours(hours: float, ref_date: date | None = None) -> list[dict]:
Q:\finance-analyzer\portfolio\econ_dates.py:194:def recent_high_impact_events(hours: float, impact_filter=("high",),
Q:\finance-analyzer\portfolio\econ_dates.py:198:    BUG-178/W15-W16 follow-up (2026-04-16): events_within_hours() above
Q:\finance-analyzer\portfolio\econ_dates.py:237:def is_macro_window(
Q:\finance-analyzer\portfolio\golddigger\data_provider.py:518:    next_evt = next_event(now.date())
Q:\finance-analyzer\portfolio\signal_engine.py:2076:            active = bool(is_macro_window())
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:55:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:78:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:102:    nearby = events_within_hours(4, ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:127:    recent = recent_high_impact_events(24)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:136:        evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:141:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:156:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"social_posts|get_reddit_posts\\(|social_sentiment\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 723ms:
Q:\finance-analyzer\portfolio\sentiment.py:748:def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None,
Q:\finance-analyzer\portfolio\sentiment.py:785:    social = social_posts or []
Q:\finance-analyzer\portfolio\signal_engine.py:3163:    social_posts = []
Q:\finance-analyzer\portfolio\signal_engine.py:3167:            from portfolio.social_sentiment import get_reddit_posts
Q:\finance-analyzer\portfolio\signal_engine.py:3176:                social_posts.extend(reddit)
Q:\finance-analyzer\portfolio\signal_engine.py:3178:            logger.debug("Optional module %s not available", "social_sentiment")
Q:\finance-analyzer\portfolio\signal_engine.py:3199:                social_posts or None,
Q:\finance-analyzer\portfolio\social_sentiment.py:91:def get_reddit_posts(ticker, limit=20):
Q:\finance-analyzer\portfolio\social_sentiment.py:133:        posts = get_reddit_posts(ticker)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 5 \"newsapi_quota_ok|newsapi_track_call|newsapi_search_query|newsapi_ttl_for_ticker\" Q:\\finance-analyzer\\portfolio\\shared_state.py" in Q:\finance-analyzer
 succeeded in 783ms:
320-# Active monitoring: 08:00-22:00 CET = 07:00-21:00 UTC
321-_NEWSAPI_ACTIVE_START_UTC = 7
322-_NEWSAPI_ACTIVE_END_UTC = 21
323-
324-
325:def newsapi_quota_ok() -> bool:
326-    """Check if we still have NewsAPI quota today. Thread-safe."""
327-    global _newsapi_daily_count, _newsapi_daily_reset
328-    now = time.time()
329-    with _newsapi_lock:
330-        # Reset counter at midnight UTC
--
336-            _newsapi_daily_count = 0
337-            _newsapi_daily_reset = now
338-        return _newsapi_daily_count < _NEWSAPI_DAILY_BUDGET
339-
340-
341:def newsapi_track_call():
342-    """Increment NewsAPI daily counter. Call after each successful API request."""
343-    global _newsapi_daily_count
344-    with _newsapi_lock:
345-        _newsapi_daily_count += 1
346-        if _newsapi_daily_count == _NEWSAPI_DAILY_BUDGET:
347-            logger.warning("NewsAPI daily budget exhausted (%d/%d), falling back to Yahoo",
348-                          _newsapi_daily_count, _NEWSAPI_DAILY_BUDGET)
349-
350-
351:def newsapi_ttl_for_ticker(ticker: str):
352-    """Dynamic TTL based on ticker priority and time of day.
353-
354-    Returns TTL in seconds, or None to skip NewsAPI for this ticker.
355-    Tier 1 (metals): 20-min during active hours (08:00-22:00 CET).
356-    Other tickers: None (Yahoo-only, saves budget for metals).
--
369-            return 1200   # 20 min — metals
370-        return 10800      # 3h — secondary (MSTR etc.)
371-    return None  # off-hours: Yahoo-only
372-
373-
374:def newsapi_search_query(ticker: str) -> str:
375-    """Optimized search query for NewsAPI. Falls back to ticker symbol."""
376-    short = ticker.upper().replace("-USD", "")
377-    return _NEWSAPI_SEARCH_QUERIES.get(short, short)
378-
379-
--
384-SENTIMENT_TTL = 900      # 15 min
385-MINISTRAL_TTL = 900      # 15 min
386-ML_SIGNAL_TTL = 900      # 15 min
387-FUNDING_RATE_TTL = 900   # 15 min
388-VOLUME_TTL = 300         # 5 min
389:NEWSAPI_TTL = 1800       # 30 min fallback — overridden by newsapi_ttl_for_ticker()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"yfinance_klines\\(\" Q:\\finance-analyzer\\portfolio" in Q:\finance-analyzer
 succeeded in 682ms:
Q:\finance-analyzer\portfolio\data_collector.py:207:def yfinance_klines(ticker, interval="1d", limit=100):
Q:\finance-analyzer\portfolio\data_collector.py:265:            return yfinance_klines(ticker, interval=interval, limit=limit)
Q:\finance-analyzer\portfolio\golddigger\data_provider.py:129:            df = yfinance_klines(ticker, interval=interval, limit=max(lookback_bars, 2))
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"10m\" Q:\\finance-analyzer\\portfolio\\data_collector.py Q:\\finance-analyzer\\portfolio\\price_source.py Q:\\finance-analyzer\\portfolio\\metals_cross_assets.py Q:\\finance-analyzer\\portfolio\\seasonality_updater.py Q:\\finance-analyzer\\portfolio\\futures_data.py" in Q:\finance-analyzer
2026-05-11T15:35:36.725509Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 exited 1 in 651ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 4 \"get_sentiment\\(|ThreadPoolExecutor|submit\\(.*get_sentiment|sentiment\\)\" Q:\\finance-analyzer\\portfolio\\main.py Q:\\finance-analyzer\\portfolio\\signal_engine.py" in Q:\finance-analyzer
 succeeded in 632ms:
Q:\finance-analyzer\portfolio\main.py-471-
Q:\finance-analyzer\portfolio\main.py-472-    # --- Fully parallel: data collection + signal generation per ticker ---
Q:\finance-analyzer\portfolio\main.py-473-    # Each ticker: fetch timeframes, compute indicators, generate signals — all threaded.
Q:\finance-analyzer\portfolio\main.py-474-    # Rate limiters, cache locks, and GPU gate are already thread-safe.
Q:\finance-analyzer\portfolio\main.py:475:    from concurrent.futures import ThreadPoolExecutor, as_completed
Q:\finance-analyzer\portfolio\main.py-476-
Q:\finance-analyzer\portfolio\main.py-477-    active_items = [(name, source) for name, source in SYMBOLS.items() if name in active]
Q:\finance-analyzer\portfolio\main.py-478-
Q:\finance-analyzer\portfolio\main.py-479-    def _process_ticker(name, source):
--
Q:\finance-analyzer\portfolio\main.py-606-    # real bottleneck is identifiable without guessing.
Q:\finance-analyzer\portfolio\main.py-607-    _TICKER_POOL_TIMEOUT = 360
Q:\finance-analyzer\portfolio\main.py-608-    # OR-I-001: avoid context manager — __exit__ calls shutdown(wait=True)
Q:\finance-analyzer\portfolio\main.py-609-    # which blocks the loop when threads hang past the timeout.
Q:\finance-analyzer\portfolio\main.py:610:    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
Q:\finance-analyzer\portfolio\main.py-611-    futures = {
Q:\finance-analyzer\portfolio\main.py-612-        pool.submit(_process_ticker, name, source): name
Q:\finance-analyzer\portfolio\main.py-613-        for name, source in active_items
Q:\finance-analyzer\portfolio\main.py-614-    }
--
Q:\finance-analyzer\portfolio\signal_engine.py-34-# new DataFrame allocated at the same address as a freed one doesn't get a
Q:\finance-analyzer\portfolio\signal_engine.py-35-# stale hit (see _compute_adx for the C1 content-key rationale, 2026-05-10
Q:\finance-analyzer\portfolio\signal_engine.py-36-# fixed the annotation drift that read ``dict[int, …]``).
Q:\finance-analyzer\portfolio\signal_engine.py-37-_adx_cache: dict[tuple[int, int, float], float | None] = {}
Q:\finance-analyzer\portfolio\signal_engine.py:38:_adx_lock = threading.Lock()  # BUG-86: protect concurrent access from ThreadPoolExecutor
Q:\finance-analyzer\portfolio\signal_engine.py-39-_ADX_CACHE_MAX = 200  # prevent unbounded growth
Q:\finance-analyzer\portfolio\signal_engine.py-40-
Q:\finance-analyzer\portfolio\signal_engine.py-41-# BUG-178 diagnostics: per-ticker last-signal tracker.
Q:\finance-analyzer\portfolio\signal_engine.py-42-# Updated right before each enhanced signal's compute_fn() is called so that
--
Q:\finance-analyzer\portfolio\signal_engine.py-1003-
Q:\finance-analyzer\portfolio\signal_engine.py-1004-# Sentiment hysteresis — prevents rapid flip spam from ~50% confidence oscillation
Q:\finance-analyzer\portfolio\signal_engine.py-1005-_prev_sentiment: dict[str, str] = {}  # in-memory cache; seeded from sentiment_state.json on first call
Q:\finance-analyzer\portfolio\signal_engine.py-1006-_prev_sentiment_loaded = False
Q:\finance-analyzer\portfolio\signal_engine.py:1007:_sentiment_lock = threading.Lock()  # BUG-85: protect concurrent access from ThreadPoolExecutor
Q:\finance-analyzer\portfolio\signal_engine.py-1008-_sentiment_dirty = False  # Track whether in-memory state diverged from disk
Q:\finance-analyzer\portfolio\signal_engine.py-1009-
Q:\finance-analyzer\portfolio\signal_engine.py-1010-_SENTIMENT_STATE_FILE = DATA_DIR / "sentiment_state.json"
Q:\finance-analyzer\portfolio\signal_engine.py-1011-
--
Q:\finance-analyzer\portfolio\signal_engine.py-1055-    global _sentiment_dirty
Q:\finance-analyzer\portfolio\signal_engine.py-1056-    with _sentiment_lock:
Q:\finance-analyzer\portfolio\signal_engine.py-1057-        if not _sentiment_dirty:
Q:\finance-analyzer\portfolio\signal_engine.py-1058-            return
Q:\finance-analyzer\portfolio\signal_engine.py:1059:        snapshot = dict(_prev_sentiment)
Q:\finance-analyzer\portfolio\signal_engine.py-1060-    # Write outside the lock to avoid holding it during I/O
Q:\finance-analyzer\portfolio\signal_engine.py-1061-    try:
Q:\finance-analyzer\portfolio\signal_engine.py-1062-        from portfolio.file_utils import atomic_write_json
Q:\finance-analyzer\portfolio\signal_engine.py-1063-        atomic_write_json(_SENTIMENT_STATE_FILE, {"prev_sentiment": snapshot})
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"replace\\(tzinfo=None\\)|tzinfo is None|tzinfo=UTC|fromisoformat\\(|datetime\\.now\\(UTC\\)|datetime\\.now\\(|astimezone\\(|ZoneInfo|zoneinfo|naive|aware\" Q:\\finance-analyzer\\portfolio\\fear_greed.py Q:\\finance-analyzer\\portfolio\\sentiment.py Q:\\finance-analyzer\\portfolio\\crypto_scheduler.py Q:\\finance-analyzer\\portfolio\\earnings_calendar.py Q:\\finance-analyzer\\portfolio\\econ_dates.py Q:\\finance-analyzer\\portfolio\\session_calendar.py Q:\\finance-analyzer\\portfolio\\forecast_signal.py" in Q:\finance-analyzer
 succeeded in 684ms:
Q:\finance-analyzer\portfolio\forecast_signal.py:272:        # Build dataframe with hourly timestamps (tz-naive, Prophet requirement)
Q:\finance-analyzer\portfolio\forecast_signal.py:273:        now = datetime.now(UTC).replace(tzinfo=None)
Q:\finance-analyzer\portfolio\forecast_signal.py:345:    ts = datetime.now(UTC).isoformat()
Q:\finance-analyzer\portfolio\session_calendar.py:58:    mar31 = datetime(year, 3, 31, tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:60:    dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:63:    oct31 = datetime(year, 10, 31, tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:65:    dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:87:    if end.tzinfo is None:
Q:\finance-analyzer\portfolio\session_calendar.py:88:        end = end.replace(tzinfo=UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:129:        now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\session_calendar.py:130:    if now.tzinfo is None:
Q:\finance-analyzer\portfolio\session_calendar.py:131:        now = now.replace(tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:150:        ref_date = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\econ_dates.py:156:                                      tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:157:            # M5: Use datetime.now(UTC) so past-today events show hours_until=0.
Q:\finance-analyzer\portfolio\econ_dates.py:158:            now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:173:        ref_date = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\econ_dates.py:176:    now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:181:                                  tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:219:    now = ref_time if ref_time is not None else datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:225:                                  tzinfo=UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:258:        now: Optional reference time (UTC). Defaults to ``datetime.now(UTC)``.
Q:\finance-analyzer\portfolio\econ_dates.py:268:        now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\econ_dates.py:273:            evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC,
Q:\finance-analyzer\portfolio\earnings_calendar.py:71:        today = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\earnings_calendar.py:79:                ed = _date.fromisoformat(rd)
Q:\finance-analyzer\portfolio\earnings_calendar.py:120:                        ed = datetime.fromisoformat(earnings_date).date()
Q:\finance-analyzer\portfolio\earnings_calendar.py:125:                        today = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\crypto_scheduler.py:20:import zoneinfo
Q:\finance-analyzer\portfolio\crypto_scheduler.py:73:        tz = zoneinfo.ZoneInfo(tz_name)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:75:        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:77:    now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:178:        tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:179:        now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:310:        "ts": datetime.now().astimezone().isoformat(),
Q:\finance-analyzer\portfolio\crypto_scheduler.py:375:            tz = zoneinfo.ZoneInfo(DEFAULT_TZ)
Q:\finance-analyzer\portfolio\crypto_scheduler.py:376:            now_local = datetime.now(tz)
Q:\finance-analyzer\portfolio\fear_greed.py:45:    now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\fear_greed.py:169:        "timestamp": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\sentiment.py:153:                "published": pub or datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\sentiment.py:177:            "published": a.get("publishedAt", datetime.now(UTC).isoformat()),
Q:\finance-analyzer\portfolio\sentiment.py:730:            "ts": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\sentiment.py:859:    ab_key = f"{short}:{datetime.now(UTC).isoformat()}"
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I have the main defect set. I’m pulling exact line anchors for the quota/accounting paths now, then I’ll sort them by production impact rather than by how easy they were to notice.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 3 \"_daily_budget_used|_budget_reset_date|_check_budget|_save_persistent_cache|_cache_lock|daily budget|increment under lock\" Q:\\finance-analyzer\\portfolio\\alpha_vantage.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 3 \"bypass alpha_vantage|_alpha_vantage_limiter|EARNINGS|fetch_with_retry\\(|daily budget\" Q:\\finance-analyzer\\portfolio\\earnings_calendar.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 3 \"_fetch_newsapi_with_tracking|newsapi_track_call|only count against budget\" Q:\\finance-analyzer\\portfolio\\sentiment.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 3 \"fetch_json\\(|urlopen\\(|Request\\(|Reddit|keywords\" Q:\\finance-analyzer\\portfolio\\social_sentiment.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 3 \"_snapshot_buffers|_spread_buffers|_ofi_history|persist_state|load_persisted_state|get_ofi_zscore|_MIN_OFI_HISTORY_FOR_ZSCORE\" Q:\\finance-analyzer\\portfolio\\microstructure_state.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 2 '\"orderbook_flow\"' Q:\\finance-analyzer\\portfolio\\tickers.py" in Q:\finance-analyzer
 succeeded in 898ms:
21-GATE_DAYS = 2
22-
23-# Cache TTL: 24 hours (earnings dates are stable)
24:EARNINGS_CACHE_TTL = 86400
25-
26-# Per-ticker cache: {ticker: {"data": {...}, "time": epoch}}
27-_earnings_cache: dict[str, dict] = {}
--
29-
30-
31-def _fetch_earnings_alpha_vantage(ticker: str) -> dict | None:
32:    """Fetch next earnings date from Alpha Vantage EARNINGS endpoint.
33-
34-    Uses the already-configured AV API key and rate limiter.
35-    """
36-    try:
37-        from portfolio.api_utils import load_config
38-        from portfolio.http_retry import fetch_with_retry
39:        from portfolio.shared_state import _alpha_vantage_limiter
40-
41-        config = load_config()
42-        # C9/DC-R3-1: key lives under config["alpha_vantage"]["api_key"], not
--
45-        if not api_key:
46-            return None
47-
48:        _alpha_vantage_limiter.wait()
49:        # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
50-        # because there is no public increment function exported from that module.
51-        # Known limitation — earnings fetches consume 1 AV call each but are not
52-        # reflected in the budget tracker.  Each ticker only fetches once per 24h.
53:        r = fetch_with_retry(
54-            "https://www.alphavantage.co/query",
55-            params={
56:                "function": "EARNINGS",
57-                "symbol": ticker,
58-                "apikey": api_key,
59-            },
--
63-            return None
64-        data = r.json()
65-
66:        # AV EARNINGS returns quarterlyEarnings and annualEarnings
67-        quarterly = data.get("quarterlyEarnings", [])
68-        if not quarterly:
69-            return None
--
167-    now = time.time()
168-    with _earnings_lock:
169-        cached = _earnings_cache.get(ticker)
170:        if cached and now - cached["time"] < EARNINGS_CACHE_TTL:
171-            return cached["data"]
172-
173-    # Fetch fresh
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 926ms:
27-
28-# Module-level state
29-_cache = {}  # ticker -> normalized fundamentals dict
30:_cache_lock = threading.Lock()
31:_daily_budget_used = 0
32:_budget_reset_date = ""  # ISO date string for budget tracking
33-_cb = CircuitBreaker("alpha_vantage", failure_threshold=3, recovery_timeout=300)
34-
35-
--
41-        logger.info("No fundamentals cache found at %s", CACHE_FILE)
42-        return
43-    if isinstance(data, dict):
44:        with _cache_lock:
45-            _cache = data
46-        logger.info("Loaded fundamentals for %d tickers from cache", len(data))
47-
48-
49:def _save_persistent_cache():
50-    """Write current cache to disk atomically."""
51-    try:
52-        from portfolio.file_utils import atomic_write_json
53:        with _cache_lock:
54-            snapshot = dict(_cache)
55-        atomic_write_json(CACHE_FILE, snapshot)
56-    except Exception as e:
--
59-
60-def get_fundamentals(ticker):
61-    """Return cached fundamentals for a ticker, or None if not available."""
62:    with _cache_lock:
63-        return _cache.get(ticker)
64-
65-
66-def get_all_fundamentals():
67-    """Return all cached fundamentals (for reporting)."""
68:    with _cache_lock:
69-        return dict(_cache)
70-
71-
--
154-    return data
155-
156-
157:def _check_budget():
158:    """Check and reset daily budget counter. Returns current usage count.
159-
160:    BUG-108: Protected by _cache_lock for thread safety.
161-    """
162:    global _daily_budget_used, _budget_reset_date
163-    today = datetime.now(UTC).strftime("%Y-%m-%d")
164:    with _cache_lock:
165:        if _budget_reset_date != today:
166:            _daily_budget_used = 0
167:            _budget_reset_date = today
168:        return _daily_budget_used
169-
170-
171-def _is_stale(ticker, max_stale_days=5):
172-    """Check if cached data for ticker is older than max_stale_days."""
173:    with _cache_lock:
174-        entry = _cache.get(ticker)
175-    if not entry:
176-        return True
--
187-
188-def _cache_age_hours(ticker):
189-    """Return age of cached data in hours, or None if not cached."""
190:    with _cache_lock:
191-        entry = _cache.get(ticker)
192-    if not entry:
193-        return None
--
204-def refresh_fundamentals_batch(config):
205-    """Batch-refresh fundamentals for all stock tickers.
206-
207:    Respects rate limits (5/min) and daily budget (25/day).
208-    Returns number of successfully refreshed tickers.
209-    """
210:    global _daily_budget_used
211-
212-    av_config = config.get("alpha_vantage", {})
213-    if not av_config.get("enabled", True):
--
228-        logger.info("Alpha Vantage circuit breaker active (%s)", _cb.state.value)
229-        return 0
230-
231:    budget_used = _check_budget()
232-    if budget_used >= daily_budget:
233:        logger.info("Alpha Vantage daily budget exhausted (%d/%d)", budget_used, daily_budget)
234-        return 0
235-
236-    # Build refresh list: stock tickers not recently cached, not skipped
--
276-                    break
277-                continue
278-
279:            with _cache_lock:
280-                _cache[ticker] = normalized
281:                _daily_budget_used += 1  # BUG-108: increment under lock
282-            _cb.record_success()
283-            success_count += 1
284-            logger.info("Refreshed fundamentals for %s (PE=%.1f, sector=%s)",
--
293-                break
294-
295-    if success_count > 0:
296:        _save_persistent_cache()
297-        logger.info("Fundamentals refresh complete: %d/%d succeeded", success_count, len(tickers_to_refresh))
298-
299-    return success_count
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 903ms:
181-    ]
182-
183-
184:def _fetch_newsapi_with_tracking(ticker, api_key, limit=10, query=None):
185-    """Fetch from NewsAPI and track the call against daily quota.
186-
187:    H9/DC-R3-2: only count against budget when the fetch actually returned data
188-    (not on empty responses or errors), preventing spurious budget exhaustion.
189-    """
190:    from portfolio.shared_state import newsapi_track_call
191-    result = _fetch_newsapi_headlines(ticker, api_key, limit=limit, query=query)
192:    if result:  # only count against budget when we actually got data
193:        newsapi_track_call()
194-    return result
195-
196-
--
213-            cached_newsapi = _cached(
214-                f"newsapi_{ticker}",
215-                ttl,
216:                _fetch_newsapi_with_tracking,
217-                ticker,
218-                newsapi_key,
219-                limit,
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 913ms:
25-_MAX_SNAPSHOTS = 60  # ~30-60 min at 30-60s intervals
26-_MIN_SNAPSHOTS_FOR_OFI = 3
27-_MIN_SPREADS_FOR_ZSCORE = 10
28:_MIN_OFI_HISTORY_FOR_ZSCORE = 10
29-_MAX_OFI_HISTORY = 120  # ~2h of OFI readings for z-score normalization
30-
31-# Multi-scale OFI windows (snapshot counts)
--
37-# Protected by _buffer_lock — metals_loop fast-tick (10s) and main cycle
38-# (60s) can both call accumulate_snapshot / get_state concurrently.
39-_buffer_lock = threading.Lock()
40:_snapshot_buffers: dict[str, deque] = {}
41:_spread_buffers: dict[str, deque] = {}
42:_ofi_history: dict[str, deque] = {}  # rolling OFI values for z-score
43-
44-
45-def _ensure_buffer(ticker: str) -> None:
46-    """Initialize ring buffers for a ticker if needed."""
47:    if ticker not in _snapshot_buffers:
48:        _snapshot_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
49:    if ticker not in _spread_buffers:
50:        _spread_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
51:    if ticker not in _ofi_history:
52:        _ofi_history[ticker] = deque(maxlen=_MAX_OFI_HISTORY)
53-
54-
55-def accumulate_snapshot(ticker: str, depth: dict) -> None:
--
71-    }
72-    with _buffer_lock:
73-        _ensure_buffer(ticker)
74:        _snapshot_buffers[ticker].append(snapshot)
75:        _spread_buffers[ticker].append(depth["spread"])
76-
77-
78-def get_rolling_ofi(ticker: str) -> float:
--
83-    """
84-    with _buffer_lock:
85-        _ensure_buffer(ticker)
86:        snapshots = list(_snapshot_buffers[ticker])
87-    if len(snapshots) < _MIN_SNAPSHOTS_FOR_OFI:
88-        return 0.0
89-    return compute_ofi(snapshots)
--
97-    """
98-    with _buffer_lock:
99-        _ensure_buffer(ticker)
100:        _ofi_history[ticker].append(ofi_val)
101-
102-
103:def get_ofi_zscore(ticker: str, current_ofi: float | None = None) -> float:
104-    """Z-score of OFI relative to its own rolling distribution.
105-
106-    Scores ``current_ofi`` (or the latest recorded value) against the
--
111-    """
112-    with _buffer_lock:
113-        _ensure_buffer(ticker)
114:        history = list(_ofi_history[ticker])
115:    if len(history) < _MIN_OFI_HISTORY_FOR_ZSCORE:
116-        return 0.0
117-    import numpy as np
118-    arr = np.array(history, dtype=float)
--
132-    """
133-    with _buffer_lock:
134-        _ensure_buffer(ticker)
135:        snapshots = list(_snapshot_buffers[ticker])
136-    n = len(snapshots)
137-
138-    ofi_slow = compute_ofi(snapshots) if n >= _MIN_SNAPSHOTS_FOR_OFI else 0.0
--
166-    """
167-    with _buffer_lock:
168-        _ensure_buffer(ticker)
169:        spreads = list(_spread_buffers[ticker])
170-    if len(spreads) < _MIN_SPREADS_FOR_ZSCORE:
171-        return None
172-    return spread_zscore(spreads)
--
181-    # Compute z-score BEFORE appending current value to history —
182-    # scoring against a population that includes itself compresses
183-    # the z-score toward zero (code review finding CRITICAL-1).
184:    ofi_z = get_ofi_zscore(ticker, current_ofi=ofi)
185-    record_ofi(ticker, ofi)
186-    sz = get_spread_zscore(ticker)
187-    ms_ofi = get_multiscale_ofi(ticker)
188-    with _buffer_lock:
189-        _ensure_buffer(ticker)
190:        snap_count = len(_snapshot_buffers[ticker])
191:        spread_count = len(_spread_buffers[ticker])
192-    return {
193-        "ofi": ofi,
194-        "ofi_zscore": ofi_z,
--
202-    }
203-
204-
205:def persist_state() -> None:
206-    """Write current microstructure state to disk for cross-process access."""
207-    state = {}
208:    for ticker in _snapshot_buffers:
209-        ms = get_microstructure_state(ticker)
210-        ms["ts"] = int(time.time() * 1000)
211-        state[ticker] = ms
--
213-        atomic_write_json(_STATE_FILE, state)
214-
215-
216:def load_persisted_state(ticker: str) -> dict | None:
217-    """Read persisted microstructure state for a ticker.
218-
219-    Used by orderbook_flow signal when running in a different process
--
233-    """Return current snapshot buffer size for a ticker."""
234-    with _buffer_lock:
235-        _ensure_buffer(ticker)
236:        return len(_snapshot_buffers[ticker])
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 930ms:
124-    # 62.6% accuracy before disabling. Accuracy gate will auto-gate if BUY
125-    # signals degrade the composite.
126:    "orderbook_flow",   # 2026-04-11: 51.1% accuracy (360 sam), 93.3% activation rate,
127-                        # no recent data. Pure noise in every consensus decision.
128-                        # Re-evaluate after 2 weeks of accuracy data collection.
--
284-    "futures_flow",
285-    "crypto_macro",
286:    "orderbook_flow",
287-    "metals_cross_asset",
288-    "dxy_cross_asset",
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 931ms:
1:"""Social media sentiment — Reddit headline fetcher.
2-
3:Uses Reddit's public JSON API, no authentication needed.
4-"""
5-
6-import json
--
10-
11-USER_AGENT = "finance-analyzer/1.0 (portfolio intelligence bot)"
12-
13:# (subreddit, dedicated) — dedicated: keep all posts; general: filter by keywords
14-TICKER_SUBREDDITS = {
15-    "BTC": [("Bitcoin", True), ("CryptoCurrency", False)],
16-    "ETH": [("ethereum", True), ("CryptoCurrency", False)],
--
26-}
27-
28-
29:def _fetch_subreddit(sub, keywords, dedicated, per_sub):
30-    posts = []
31-    url = f"https://www.reddit.com/r/{sub}/hot.json?limit={per_sub + 5}&raw_json=1"
32:    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
33:    with urllib.request.urlopen(req, timeout=10) as resp:
34-        data = json.loads(resp.read())
35-    for child in data.get("data", {}).get("children", []):
36-        post = child.get("data", {})
37-        title = post.get("title", "").strip()
38-        if not title or post.get("stickied"):
39-            continue
40:        if not dedicated and not any(kw in title.lower() for kw in keywords):
41-            continue
42-        created = post.get("created_utc", 0)
43-        posts.append(
--
56-    return posts
57-
58-
59:def _search_subreddit(sub, keywords, limit=10):
60:    query = urllib.parse.quote(" OR ".join(keywords))
61-    url = (
62-        f"https://www.reddit.com/r/{sub}/search.json"
63-        f"?q={query}&sort=new&restrict_sr=on&limit={limit}&raw_json=1"
64-    )
65:    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
66:    with urllib.request.urlopen(req, timeout=10) as resp:
67-        data = json.loads(resp.read())
68-    posts = []
69-    for child in data.get("data", {}).get("children", []):
--
91-def get_reddit_posts(ticker, limit=20):
92-    short = ticker.upper().replace("-USD", "")
93-    subreddits = TICKER_SUBREDDITS.get(short, [])
94:    keywords = TICKER_KEYWORDS.get(short, [short.lower()])
95-    if not subreddits:
96-        return []
97-
--
101-
102-    for sub, dedicated in subreddits:
103-        try:
104:            fetched = _fetch_subreddit(sub, keywords, dedicated, per_sub)
105-            for p in fetched:
106-                if p["title"] not in seen:
107-                    seen.add(p["title"])
108-                    posts.append(p)
109-        except Exception as e:
110:            print(f"    [Reddit r/{sub}] error: {e}")
111-
112-    # Fallback: if keyword filtering yielded nothing, try search
113-    if not posts:
114-        for sub, _dedicated in subreddits[:1]:
115-            try:
116:                fetched = _search_subreddit(sub, keywords, limit=per_sub)
117-                for p in fetched:
118-                    if p["title"] not in seen:
119-                        seen.add(p["title"])
120-                        posts.append(p)
121-            except Exception as e:
122:                print(f"    [Reddit search r/{sub}] error: {e}")
123-
124-    posts.sort(key=lambda p: p.get("score", 0), reverse=True)
125-    return posts[:limit]
--
128-if __name__ == "__main__":
129-    for ticker in ["BTC", "ETH", "PLTR", "NVDA"]:
130-        print(f"\n{'='*60}")
131:        print(f"  Reddit posts for {ticker}")
132-        print(f"{'='*60}")
133-        posts = get_reddit_posts(ticker)
134-        print(f"  Found {len(posts)} posts")
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 4 \"_BINANCE_FAPI|_YFINANCE_LAST_RESORT|resolve_source|fetch_klines|Falling back to yfinance|All sources failed\" Q:\\finance-analyzer\\portfolio\\price_source.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 3 '\"GC=F\"|\"SI=F\"|\"CL=F\"|fetch_klines\\(' Q:\\finance-analyzer\\portfolio\\metals_cross_assets.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 3 \"14:00 UTC|8:30 AM ET|FOMC|datetime\\.combine\\(|lookback_hours|lookahead_hours\" Q:\\finance-analyzer\\portfolio\\econ_dates.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 3 \"cached_rate|_FX_STALE_THRESHOLD|hardcoded fallback|Using stale FX rate|rate outside sane bounds|hardcoded FX\" Q:\\finance-analyzer\\portfolio\\fx_rates.py" in Q:\finance-analyzer
 succeeded in 778ms:
17-Avanza market-guide              tick      1,259 ms
18-yfinance SI=F                   655.4 s      389 ms    ← 85× stale
19-
20-Every consumer of historical / intraday OHLCV should call
21::func:`fetch_klines` rather than importing yfinance directly. The
22-function raises :class:`SourceUnavailableError` if no live alternative
23-exists AND the yfinance fallback is not allowed for the requested
24-ticker.
25-"""
--
37-# obvious. The router is alias-aware: 'XAG-USD', 'SI=F', 'XAGUSDT' all
38-# resolve to the same Binance FAPI symbol.
39-# ---------------------------------------------------------------------------
40-
41:_BINANCE_FAPI = {
42-    # silver
43-    "XAG-USD": "XAGUSDT", "SI=F": "XAGUSDT", "XAGUSDT": "XAGUSDT",
44-    # gold
45-    "XAU-USD": "XAUUSDT", "GC=F": "XAUUSDT", "XAUUSDT": "XAUUSDT",
--
60-# Tickers for which yfinance is the only available free data source.
61-# Calls to these emit DEBUG (not WARNING) so legitimate use doesn't
62-# pollute the log; calls for non-allowed tickers emit WARNING so we
63-# can quantify residual leakage.
64:_YFINANCE_LAST_RESORT = frozenset({
65-    "HG=F",          # copper — no Binance perpetual
66-    "DX-Y.NYB",      # DXY pseudo-ticker (Alpha Vantage FX is paid intraday)
67-    "EURUSD=X",      # FX — Alpha Vantage paid intraday
68-    "^TNX",          # 10y treasury yield (CBOE; FRED has daily DGS10 fallback)
--
130-    """Yfinance fetcher. Allowed-list tickers emit DEBUG; everything else
131-    emits WARNING so we can quantify residual leakage."""
132-    import yfinance as yf
133-
134:    if ticker in _YFINANCE_LAST_RESORT:
135-        logger.debug(
136-            "price_source: yfinance for %s (interval=%s, period=%s) — allowed (no live alt)",
137-            ticker, interval, period,
138-        )
--
164-# Public API
165-# ---------------------------------------------------------------------------
166-
167-
168:def resolve_source(ticker: str) -> str:
169-    """Return the routing decision for ``ticker``: one of
170-    {"binance_fapi", "binance_spot", "alpaca", "yfinance"}.
171-
172-    Exported for diagnostics and tests.
173-    """
174:    if ticker in _BINANCE_FAPI:
175-        return "binance_fapi"
176-    if ticker in _BINANCE_SPOT:
177-        return "binance_spot"
178:    if ticker in _YFINANCE_LAST_RESORT:
179-        return "yfinance"
180-    # Default assumption: a bare uppercase symbol is a US stock/ETF → Alpaca.
181-    # Anything starting with '^' that isn't a CBOE vol index is an index we
182-    # don't have a mapping for — route to yfinance with warning.
--
184-        return "yfinance"
185-    return "alpaca"
186-
187-
188:def fetch_klines(
189-    ticker: str,
190-    interval: str = "1d",
191-    limit: int = 100,
192-    period: str | None = None,
--
208-
209-    Raises:
210-        SourceUnavailableError: If every applicable source failed.
211-    """
212:    source = resolve_source(ticker)
213-
214-    try:
215-        if source == "binance_fapi":
216:            return _fetch_binance_fapi(_BINANCE_FAPI[ticker], interval, limit)
217-        if source == "binance_spot":
218-            return _fetch_binance_spot(_BINANCE_SPOT[ticker], interval, limit)
219-        if source == "alpaca":
220-            return _fetch_alpaca(ticker, interval, limit)
--
227-        # so we can see how often the primary is flaking.
228-        if source in ("binance_fapi", "binance_spot", "alpaca"):
229-            logger.error(
230-                "price_source: primary source %s FAILED for %s (%r). "
231:                "Falling back to yfinance. Investigate the primary outage.",
232-                source, ticker, exc,
233-            )
234-            try:
235-                return _fetch_yfinance(ticker, interval, period=period, limit=limit)
236-            except Exception as exc2:
237-                raise SourceUnavailableError(
238:                    f"All sources failed for {ticker}: primary={source} "
239-                    f"({exc!r}), fallback=yfinance ({exc2!r})"
240-                ) from exc2
241-        raise SourceUnavailableError(
242-            f"yfinance failed for {ticker}: {exc!r}"
--
247-    """True if yfinance is an approved primary source for this ticker
248-    (CBOE vol indices + the explicit last-resort list).
249-
250-    Callers that want a direct yfinance path (for reasons beyond
251:    :func:`fetch_klines`'s OHLCV contract — e.g. yfinance's earnings
252-    calendar or fundamentals APIs) should check this first and log a
253-    WARNING if they still go to yfinance for a ticker outside this list.
254-    """
255:    return ticker in _YFINANCE_LAST_RESORT
256-
257-
258-__all__ = [
259:    "fetch_klines",
260:    "resolve_source",
261-    "is_yfinance_allowed",
262-    "SourceUnavailableError",
263-]
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 799ms:
55-    try:
56-        from portfolio.price_source import fetch_klines
57-
58:        df = fetch_klines(ticker, interval=interval, limit=limit, period=period)
59-    except Exception as e:
60-        logger.warning("price_source fetch failed for %s: %s", ticker, e)
61-        return pd.DataFrame()
--
136-def get_gold_silver_ratio() -> dict | None:
137-    """Gold/Silver price ratio and deviation from mean."""
138-    def _fetch():
139:        gold_df = _yf_download("GC=F", period="6mo", interval="1d")
140:        silver_df = _yf_download("SI=F", period="6mo", interval="1d")
141-        if gold_df.empty or silver_df.empty:
142-            return None
143-        gold_close = gold_df["Close"].dropna()
--
167-def get_oil_data() -> dict | None:
168-    """WTI Crude Oil futures (CL=F) price and momentum."""
169-    def _fetch():
170:        df = _yf_download("CL=F", period="3mo", interval="1d")
171-        if df.empty or "Close" not in df.columns:
172-            return None
173-        close = df["Close"].dropna()
--
240-def get_gold_silver_ratio_intraday() -> dict | None:
241-    """Gold/Silver ratio 60m bars. Exposes ratio_change_3h_pct."""
242-    def _fetch():
243:        gold_df = _yf_download("GC=F", period="5d", interval="60m")
244:        silver_df = _yf_download("SI=F", period="5d", interval="60m")
245-        if gold_df.empty or silver_df.empty:
246-            return None
247-        gold_close = gold_df["Close"].dropna()
--
264-def get_oil_intraday() -> dict | None:
265-    """WTI crude 60m bars."""
266-    def _fetch():
267:        df = _yf_download("CL=F", period="5d", interval="60m")
268-        if df.empty or "Close" not in df.columns:
269-            return None
270-        close = df["Close"].dropna()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 820ms:
7-- CPI: Bureau of Labor Statistics release schedule
8-- NFP: Bureau of Labor Statistics (first Friday of each month)
9-- GDP: Bureau of Economic Analysis advance estimate schedule
10:- FOMC: imported from fomc_dates.py (not duplicated here)
11-"""
12-
13-from __future__ import annotations
14-
15-from datetime import UTC, date, datetime
16-
17:from portfolio.fomc_dates import FOMC_ANNOUNCEMENT_DATES
18-
19-# ---------------------------------------------------------------------------
20:# CPI release dates 2026 (8:30 AM ET, typically 2nd or 3rd week)
21-# ---------------------------------------------------------------------------
22-
23-CPI_DATES_2026 = [
--
51-]
52-
53-# ---------------------------------------------------------------------------
54:# Non-Farm Payrolls (first Friday of each month, 8:30 AM ET)
55-# ---------------------------------------------------------------------------
56-
57-NFP_DATES_2026 = [
--
106-# Unified event list
107-# ---------------------------------------------------------------------------
108-
109:# Impact levels: "high" (FOMC, CPI, NFP) or "medium" (GDP)
110-# Affected sectors: which sectors are most impacted
111-
112-EVENT_SECTOR_MAP = {
113:    "FOMC": {"crypto", "metals", "big_tech", "etf"},
114-    "CPI": {"crypto", "metals", "big_tech", "etf"},
115-    "NFP": {"etf", "big_tech"},
116-    "GDP": {"etf", "big_tech"},
--
121-    """Build sorted list of all economic events."""
122-    events = []
123-
124:    for d in FOMC_ANNOUNCEMENT_DATES:
125:        events.append({"date": d, "type": "FOMC", "impact": "high"})
126-
127-    for d in CPI_DATES_2026 + CPI_DATES_2027:
128-        events.append({"date": d, "type": "CPI", "impact": "high"})
--
151-
152-    for evt in ECON_EVENTS:
153-        if evt["date"] >= ref_date:
154:            # Calculate hours until (approximate: assume 14:00 UTC release)
155:            evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
156-                                      tzinfo=UTC)
157-            # M5: Use datetime.now(UTC) so past-today events show hours_until=0.
158-            now = datetime.now(UTC)
--
177-    for evt in ECON_EVENTS:
178-        if evt["date"] < ref_date:
179-            continue
180:        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
181-                                  tzinfo=UTC)
182-        delta = evt_dt - now
183-        hrs = delta.total_seconds() / 3600
--
199-    only iterates FUTURE events (skips evt["date"] < ref_date). The
200-    accuracy degradation tracker needs the BACKWARD window too so it
201-    can blanket-suppress alerts during the post-event volatility hangover
202:    after FOMC/CPI/NFP — those releases routinely whipsaw signals for
203-    12-24 hours after the print, exactly when the lifetime-vs-recent
204-    delta would otherwise look like a degradation.
205-
--
221-    for evt in ECON_EVENTS:
222-        if evt.get("impact") not in impact_filter:
223-            continue
224:        evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
225-                                  tzinfo=UTC)
226-        delta_hrs = (now - evt_dt).total_seconds() / 3600
227-        if 0 <= delta_hrs <= hours:
--
236-
237-def is_macro_window(
238-    now: datetime | None = None,
239:    lookback_hours: float = 24.0,
240:    lookahead_hours: float = 72.0,
241-    impact_filter: tuple[str, ...] = ("high",),
242-) -> bool:
243-    """True iff a macro event matching ``impact_filter`` falls within
244:    ``lookback_hours`` past or ``lookahead_hours`` future of ``now``.
245-
246-    Wider window than ``_is_econ_blackout`` (24h ± in accuracy_degradation)
247-    — captures the full pre-event risk-off + post-event volatility-hangover
--
256-
257-    Args:
258-        now: Optional reference time (UTC). Defaults to ``datetime.now(UTC)``.
259:        lookback_hours: Past window. ``0`` disables the backward check.
260:        lookahead_hours: Future window. ``0`` disables the forward check.
261-        impact_filter: Impact levels that count. Default ``("high",)``
262:            so only FOMC/CPI/NFP trigger; GDP (medium) does not.
263-
264-    Returns:
265-        True if any qualifying event lies in the combined window.
--
269-    for evt in ECON_EVENTS:
270-        if evt.get("impact") not in impact_filter:
271-            continue
272:        evt_dt = datetime.combine(
273-            evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC,
274-        )
275-        delta_hrs = (evt_dt - now).total_seconds() / 3600
276:        # Forward window: evt is in the future, within lookahead_hours.
277:        if 0 <= delta_hrs <= lookahead_hours:
278-            return True
279:        # Backward window: evt is in the past, within lookback_hours.
280:        if -lookback_hours <= delta_hrs < 0:
281-            return True
282-    return False
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 808ms:
22-# BUG-215: Thread-safe FX cache. Accessed from 8-worker ThreadPoolExecutor.
23-_fx_lock = threading.Lock()
24-_fx_cache = {"rate": None, "time": 0}
25:_FX_STALE_THRESHOLD = 7200  # 2 hours — warn if FX rate hasn't been refreshed
26-
27-
28-def fetch_usd_sek():
29-    now = time.time()
30-    with _fx_lock:
31:        cached_rate = _fx_cache["rate"]
32-        cached_time = _fx_cache["time"]
33:    if cached_rate and now - cached_time < 900:  # 15 min cache
34:        return cached_rate
35-    try:
36-        r = fetch_with_retry(
37-            "https://api.frankfurter.app/latest",
--
55-        logger.warning("FX rate fetch failed: %s", e)
56-    # Fallback to stale cached value
57-    with _fx_lock:
58:        cached_rate = _fx_cache["rate"]
59-        cached_time = _fx_cache["time"]
60:    if cached_rate:
61-        age_secs = now - cached_time
62:        if age_secs > _FX_STALE_THRESHOLD:
63:            logger.warning("Using stale FX rate (%.1fh old)", age_secs / 3600)
64-            _fx_alert_telegram(age_secs)
65:        return cached_rate
66:    # Last resort: hardcoded fallback
67-    # BUG-117: Use ERROR level — hardcoded rate may be severely stale.
68-    # Portfolio valuations using this rate could be off by 10-15% if SEK has moved.
69:    logger.error("Using hardcoded FX fallback rate %.2f SEK — no cached or live rate available", FX_RATE_FALLBACK)
70-    _fx_alert_telegram(None)
71-    return FX_RATE_FALLBACK
72-
--
84-        if age_secs is not None:
85-            msg = f"_FX WARNING: USD/SEK rate is {age_secs / 3600:.1f}h stale. API may be down._"
86-        else:
87:            msg = f"_FX WARNING: Using hardcoded fallback rate {FX_RATE_FALLBACK} SEK. No live or cached rate available._"
88-        from portfolio.message_store import send_or_store
89-        send_or_store(msg, config, category="error")
90-    except Exception as e:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

  ✓ Inspect in-scope files and map data-external call paths
  ✓ Validate adversarial focus areas against concrete code and line numbers
  → Draft severity-ordered review with missing-test gaps
