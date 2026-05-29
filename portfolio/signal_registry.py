"""Signal registry — plugin system for modular signal management.

Enhanced signals register via register_enhanced(). signal_engine.py
discovers all signals from the registry instead of hardcoded lists.
"""
import importlib
import logging
import time
from collections.abc import Callable

logger = logging.getLogger("portfolio.signal_registry")

_ENHANCED_SIGNALS: dict[str, dict] = {}


def register_enhanced(name: str, module_path: str, func_name: str,
                      requires_macro: bool = False,
                      requires_context: bool = False,
                      max_confidence: float = 1.0):
    """Programmatically register an enhanced signal module."""
    if not (0.0 <= max_confidence <= 1.0):
        raise ValueError(f"max_confidence must be in [0, 1], got {max_confidence}")
    _ENHANCED_SIGNALS[name] = {
        "name": name,
        "type": "enhanced",
        "module_path": module_path,
        "func_name": func_name,
        "requires_macro": requires_macro,
        "requires_context": requires_context,
        "max_confidence": max_confidence,
        "func": None,  # lazy-loaded
    }


def get_enhanced_signals() -> dict[str, dict]:
    """Return all registered enhanced signals."""
    return dict(_ENHANCED_SIGNALS)


def get_signal_names() -> list:
    """Return all registered signal names."""
    return list(_ENHANCED_SIGNALS.keys())


_FAILED_IMPORT_SENTINEL = object()
_FAILED_IMPORT_COOLDOWN = 300  # retry broken imports after 5 min

def load_signal_func(entry: dict) -> Callable | None:
    """Lazy-load and cache the compute function for a signal.

    On import failure, caches the failure for _FAILED_IMPORT_COOLDOWN seconds
    so the warning is logged once, not 35× per cycle (5 tickers × 7 TFs).
    """
    cached = entry.get("func")
    if cached is not None and cached is not _FAILED_IMPORT_SENTINEL:
        return cached
    if cached is _FAILED_IMPORT_SENTINEL and time.monotonic() - entry.get("_fail_ts", 0) < _FAILED_IMPORT_COOLDOWN:
        return None
    try:
        # Suppressed false-positive: module_path comes from internal SignalSpec registry, not external input.
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
        mod = importlib.import_module(entry["module_path"])
        func = getattr(mod, entry["func_name"])
        entry["func"] = func
        entry.pop("_fail_ts", None)
        return func
    except Exception as e:
        logger.warning("Failed to load signal %s: %s", entry['name'], e)
        entry["func"] = _FAILED_IMPORT_SENTINEL
        entry["_fail_ts"] = time.monotonic()
        return None


# Register all enhanced signals (called at import time)
def _register_defaults():
    """Register the default set of enhanced signal modules."""
    defaults = [
        ("trend", "portfolio.signals.trend", "compute_trend_signal"),
        ("momentum", "portfolio.signals.momentum", "compute_momentum_signal"),
        ("volume_flow", "portfolio.signals.volume_flow", "compute_volume_flow_signal"),
        ("volatility_sig", "portfolio.signals.volatility", "compute_volatility_signal"),
        ("candlestick", "portfolio.signals.candlestick", "compute_candlestick_signal"),
        ("structure", "portfolio.signals.structure", "compute_structure_signal"),
        ("fibonacci", "portfolio.signals.fibonacci", "compute_fibonacci_signal"),
        ("smart_money", "portfolio.signals.smart_money", "compute_smart_money_signal"),
        ("oscillators", "portfolio.signals.oscillators", "compute_oscillator_signal"),
        ("heikin_ashi", "portfolio.signals.heikin_ashi", "compute_heikin_ashi_signal"),
        ("calendar", "portfolio.signals.calendar_seasonal", "compute_calendar_signal"),
    ]
    for name, mod_path, func_name in defaults:
        register_enhanced(name, mod_path, func_name)
    # mean_reversion and momentum_factors require context for seasonality detrending
    register_enhanced("mean_reversion", "portfolio.signals.mean_reversion",
                      "compute_mean_reversion_signal", requires_context=True)
    register_enhanced("momentum_factors", "portfolio.signals.momentum_factors",
                      "compute_momentum_factors_signal", requires_context=True)
    # macro_regime is special — requires_macro=True
    register_enhanced("macro_regime", "portfolio.signals.macro_regime",
                      "compute_macro_regime_signal", requires_macro=True)
    # news_event and econ_calendar require context (ticker, config); capped at 0.7
    register_enhanced("news_event", "portfolio.signals.news_event",
                      "compute_news_event_signal", requires_context=True, max_confidence=0.7)
    register_enhanced("econ_calendar", "portfolio.signals.econ_calendar",
                      "compute_econ_calendar_signal", requires_context=True, max_confidence=0.7)
    # forecast signal — Kronos + Chronos price direction prediction; capped at 0.7
    register_enhanced("forecast", "portfolio.signals.forecast",
                      "compute_forecast_signal", requires_context=True, max_confidence=0.7)
    # Claude fundamental — three-tier LLM cascade; capped at 0.7
    register_enhanced("claude_fundamental", "portfolio.signals.claude_fundamental",
                      "compute_claude_fundamental_signal", requires_context=True, max_confidence=0.7)
    # Futures flow — OI, LS ratios, funding history (crypto only); capped at 0.7
    register_enhanced("futures_flow", "portfolio.signals.futures_flow",
                      "compute_futures_flow_signal", requires_context=True, max_confidence=0.7)
    register_enhanced("crypto_derivatives_composite",
                      "portfolio.signals.crypto_derivatives_composite",
                      "compute_crypto_derivatives_composite_signal",
                      requires_context=True, max_confidence=0.7)
    # Crypto macro — options max pain, gold-BTC rotation, exchange reserves (crypto only); capped at 0.7
    register_enhanced("crypto_macro", "portfolio.signals.crypto_macro",
                      "compute_crypto_macro_signal", requires_context=True, max_confidence=0.7)
    # Orderbook flow — microstructure metrics (metals + crypto); capped at 0.7
    register_enhanced("orderbook_flow", "portfolio.signals.orderbook_flow",
                      "compute_orderbook_flow_signal", requires_context=True, max_confidence=0.7)
    # Metals cross-asset — copper, GVZ, G/S ratio, SPY, oil (metals only); capped at 0.7
    register_enhanced("metals_cross_asset", "portfolio.signals.metals_cross_asset",
                      "compute_metals_cross_asset_signal", requires_context=True, max_confidence=0.7)
    # DXY cross-asset — intraday USD index inverse correlation (metals only); capped at 0.8
    # 2026-04-13: added standalone signal to capture DXY R²~0.6 vs silver at
    # 1-3h horizon. Complements macro_regime's daily DXY sub-indicator.
    register_enhanced("dxy_cross_asset", "portfolio.signals.dxy_cross_asset",
                      "compute_dxy_cross_asset_signal", requires_context=True, max_confidence=0.8)
    # COT positioning — CFTC speculative/commercial positioning, contrarian (metals only); capped at 0.7
    register_enhanced("cot_positioning", "portfolio.signals.cot_positioning",
                      "compute_cot_positioning_signal", requires_context=True, max_confidence=0.7)
    # G/S Kalman z-score regime — Kalman-filtered gold/silver ratio MR (metals only); capped at 0.7
    register_enhanced("gs_kalman_zscore_regime", "portfolio.signals.gs_kalman_zscore_regime",
                      "compute_gs_kalman_zscore_regime_signal", requires_context=True, max_confidence=0.7)
    # Credit spread risk — HY OAS from FRED as cross-asset risk appetite gauge; capped at 0.7
    register_enhanced("credit_spread_risk", "portfolio.signals.credit_spread",
                      "compute_credit_spread_signal", requires_context=True, max_confidence=0.7)
    # Futures basis regime — mark-index spread, contango/backwardation detection; capped at 0.7
    register_enhanced("futures_basis", "portfolio.signals.futures_basis",
                      "compute_futures_basis_signal", requires_context=True, max_confidence=0.7)
    # Hurst regime detector — R/S analysis for trending/MR/random-walk classification
    register_enhanced("hurst_regime", "portfolio.signals.hurst_regime",
                      "compute_hurst_regime_signal", requires_context=True)
    # Shannon entropy — market noise/predictability filter; low entropy = trending
    register_enhanced("shannon_entropy", "portfolio.signals.shannon_entropy",
                      "compute_shannon_entropy_signal")
    # VIX term structure — contango/backwardation regime detection; capped at 0.7
    register_enhanced("vix_term_structure", "portfolio.signals.vix_term_structure",
                      "compute_vix_term_structure_signal", requires_context=True, max_confidence=0.7)
    # Gold real yield paradox — GYDI regime detector (metals only); capped at 0.7
    register_enhanced("gold_real_yield_paradox", "portfolio.signals.gold_real_yield_paradox",
                      "compute_gold_real_yield_paradox_signal", requires_context=True, max_confidence=0.7)
    # Cross-asset TSMOM — bond/equity momentum predicts target asset direction; capped at 0.7
    register_enhanced("cross_asset_tsmom", "portfolio.signals.cross_asset_tsmom",
                      "compute_cross_asset_tsmom_signal", requires_context=True, max_confidence=0.7)
    # Copper/gold ratio — intermarket regime indicator (all assets); capped at 0.7
    # 2026-04-17: cross-asset signal, inverts direction for metals (falling ratio = gold strength).
    # 94% recession prediction accuracy, 0.85 correlation with 10Y yields.
    register_enhanced("copper_gold_ratio", "portfolio.signals.copper_gold_ratio",
                      "compute_copper_gold_ratio_signal", requires_context=True, max_confidence=0.7)
    # Statistical Jump Model regime — jump detection + persistence penalty (all assets)
    # 2026-04-18: addresses failing macro_regime (46.6% at 1d, 30.3% metals).
    # Source: Shu, Yu, Mulvey 2024 (12 citations). Persistence penalty prevents whiplash.
    register_enhanced("statistical_jump_regime", "portfolio.signals.statistical_jump_regime",
                      "compute_statistical_jump_regime_signal")
    # Network momentum — cross-asset momentum spillover (all assets); capped at 0.7
    # 2026-04-19: Pu et al. 2023 (arXiv:2308.11294). Sharpe 1.511 across 64 futures.
    # Simplified: correlation-weighted peer momentum divergence instead of GNN.
    register_enhanced("network_momentum", "portfolio.signals.network_momentum",
                      "compute_network_momentum_signal", requires_context=True, max_confidence=0.7)
    # OVX metals spillover — oil implied volatility as metals predictor; capped at 0.7
    # 2026-04-20: OVX at extreme quantiles predicts precious metals returns via
    # contagion/inflation/dollar channels. Distinct from metals_cross_asset (oil PRICE).
    # Source: ScienceDirect OVX cross-asset quantile predictability papers.
    register_enhanced("ovx_metals_spillover", "portfolio.signals.ovx_metals_spillover",
                      "compute_ovx_metals_spillover_signal", requires_context=True, max_confidence=0.7)
    # Cross-asset equity trend spillover — SPY/QQQ TA predicts all assets; capped at 0.7
    # 2026-04-21: Fieberg et al. 2025, robust across 1.3M research designs.
    # Inverts for safe havens (XAU, XAG): bullish equities = risk-on = SELL metals.
    register_enhanced("xtrend_equity_spillover", "portfolio.signals.xtrend_equity_spillover",
                      "compute_xtrend_equity_spillover_signal", requires_context=True, max_confidence=0.7)
    # Complexity gap regime — RMT-based market structure/synchronization detector; capped at 0.7
    # 2026-04-22: Mukhia et al. 2026, arXiv:2604.19107. Complexity gap = norm_max_eigenvalue -
    # avg_pairwise_corr. Gap collapse = crisis synchronization. Inverts for safe havens.
    register_enhanced("complexity_gap_regime", "portfolio.signals.complexity_gap_regime",
                      "compute_complexity_gap_regime_signal", requires_context=True, max_confidence=0.7)
    # Realized skewness — 3rd moment directional signal (all assets)
    # 2026-04-23: Fernandez-Perez et al. 2018. Sharpe 0.79, 8.01% annual on
    # 27 commodity futures. Z-scored skewness + kurtosis confirmation.
    register_enhanced("realized_skewness", "portfolio.signals.realized_skewness",
                      "compute_realized_skewness_signal")
    # Mahalanobis turbulence — cross-asset regime detection via Mahalanobis distance
    # 2026-04-24: Kritzman & Li (2010). Sharpe 2.20 vs 1.0 B&H, max DD 6% vs 32%.
    # Measures statistical unusualness of multi-asset returns. Includes absorption ratio.
    register_enhanced("mahalanobis_turbulence", "portfolio.signals.mahalanobis_turbulence",
                      "compute_mahalanobis_turbulence_signal", requires_context=True, max_confidence=0.7)
    # Crypto eVRP — Expected Volatility Risk Premium (crypto only); capped at 0.7
    # 2026-04-25: Zarattini, Mele & Aziz (2025). eVRP = DVOL(30d) - RV(10d).
    # Options-derived signal uncorrelated with trend-following cluster.
    # Deribit public API, no auth. BTC + ETH only.
    register_enhanced("crypto_evrp", "portfolio.signals.crypto_evrp",
                      "compute_crypto_evrp_signal", requires_context=True, max_confidence=0.7)
    # Hash Ribbons BTC — miner capitulation detector (BTC-only); capped at 0.7
    # 2026-04-26: Charles Edwards / Capriole Investments. 89% win rate on 9 signals.
    # 30d/60d SMA hashrate crossover with price momentum confirmation.
    # blockchain.info API (free, no auth). Fires ~1/year but extreme conviction.
    register_enhanced("hash_ribbons", "portfolio.signals.hash_ribbons",
                      "compute_hash_ribbons_signal", requires_context=True, max_confidence=0.7)
    # Drift Regime Gate — positive-day fraction regime detector; capped at 0.7
    # 2026-04-28: arxiv:2511.12490 (2025). OOS Sharpe >13 on 20yr walk-forward.
    # Fraction of positive close-to-close days in 63-bar window detects drift regime.
    # Pure OHLCV, all 5 tickers. Directional via SMA distance.
    register_enhanced("drift_regime_gate", "portfolio.signals.drift_regime_gate",
                      "compute_drift_regime_gate_signal", max_confidence=0.7)
    # Vol ratio regime — GK/CC volatility ratio + VR test + ER as regime detector
    # 2026-04-29: Garman & Klass (1980), Lo & MacKinlay (1988), Kaufman ER.
    # Three orthogonal regime measures: GK/CC ratio, variance ratio, efficiency ratio.
    # Directional: mean-reversion in ranging, momentum in trending. All OHLCV, all assets.
    register_enhanced("vol_ratio_regime", "portfolio.signals.vol_ratio_regime",
                      "compute_vol_ratio_regime_signal", max_confidence=0.7)
    # Residual pair reversion — cointegration-based pairs trading (all assets)
    # 2026-04-30: Leung & Nguyen (2018), Amberdata crypto pairs (Sharpe 0.93).
    # Regime-neutral mean reversion on OLS residual: ETH~BTC, XAG~XAU, MSTR~BTC.
    # Specifically targets ETH-USD (weakest ticker) and XAG-USD (user focus).
    register_enhanced("residual_pair_reversion",
                      "portfolio.signals.residual_pair_reversion",
                      "compute_residual_pair_reversion_signal",
                      requires_context=True, max_confidence=0.7)
    # Williams VIX Fix — synthetic volatility bottom/capitulation detector (all assets)
    # 2026-05-01: Larry Williams (2007). Profit factor 2.0, 322 trades 1993-2024.
    # WVF = (highest_close_22 - low) / highest_close_22 * 100. Spikes above upper
    # Bollinger Band signal extreme fear = BUY. Directly targets system's BUY-side
    # accuracy weakness. Pure OHLCV, all 5 tickers.
    register_enhanced("williams_vix_fix",
                      "portfolio.signals.williams_vix_fix",
                      "compute_williams_vix_fix_signal",
                      max_confidence=0.7)
    # Treasury risk rotation — bond yield curve shape as cross-asset regime signal
    # 2026-05-07: Gayed (2014), SSRN 2431022. IEF vs TLT relative performance.
    # Steepening = risk-on, flattening = risk-off. Inverted for safe havens.
    # Only signal using bond market data — zero correlation with existing clusters.
    register_enhanced("treasury_risk_rotation",
                      "portfolio.signals.treasury_risk_rotation",
                      "compute_treasury_risk_rotation_signal",
                      requires_context=True, max_confidence=0.7)
    # Intraday seasonality gate — hour-of-day confidence multipliers (all assets)
    # 2026-05-08: Concretum Group 2025 (Sharpe 1.6 BTC Asia-open), ScienceDirect 2024
    # (33% annualized BTC 22:00-23:00 UTC), CME Group 2026 (60-70% gold daily range
    # in London-NY overlap). Zero correlation with existing signal clusters.
    register_enhanced("intraday_seasonality",
                      "portfolio.signals.intraday_seasonality",
                      "compute_intraday_seasonality_signal",
                      requires_context=True, max_confidence=0.7)
    # Cubic trend persistence — R(t+1) = b*phi + c*phi^3 (all assets)
    # 2026-05-09: arXiv:2501.16772 (2025). Universal across equities, bonds,
    # currencies, commodities. 330yr data. Weak trends persist, strong revert.
    register_enhanced("cubic_trend_persistence",
                      "portfolio.signals.cubic_trend_persistence",
                      "compute_cubic_trend_persistence_signal",
                      max_confidence=0.7)
    # VWAP Z-Score MR — volume-weighted mean reversion (all assets)
    # 2026-05-10: FMZ VWAP StdDev MR (77.78% win rate). Different from BB
    # (volume-weighted anchor vs SMA anchor). Three sub-signals: vwap_z,
    # vwap_slope, volume_confirm. Pure OHLCV, all assets.
    register_enhanced("vwap_zscore_mr",
                      "portfolio.signals.vwap_zscore_mr",
                      "compute_vwap_zscore_mr_signal",
                      max_confidence=0.85)
    # Gold overnight bias — LBMA fix session drift (metals only); capped at 0.7
    # 2026-05-11: Sprott Money 2024, LBMA data 1970-2024. 54-year edge:
    # overnight (PM->AM fix) = positive drift, London PM = negative drift.
    # Completely uncorrelated with all existing signal clusters.
    register_enhanced("gold_overnight_bias",
                      "portfolio.signals.gold_overnight_bias",
                      "compute_gold_overnight_bias_signal",
                      requires_context=True, max_confidence=0.7)
    # Metals VRP — GVZ (gold implied vol) minus realized vol as contrarian
    # 2026-05-12: BIS WP 619, CBOE GVZ. IV/RV ratio is established alpha source.
    # Metals-only. Uses FRED GVZCLS (same integration as credit_spread).
    register_enhanced("metals_vrp",
                      "portfolio.signals.metals_vrp",
                      "compute_metals_vrp_signal",
                      requires_context=True, max_confidence=0.7)
    # Breakeven inflation momentum — FRED T10YIE as inflation expectations proxy
    # 2026-05-13: ScienceDirect 2025, gold -0.82 corr with real yields.
    # Rising breakeven = rising inflation expectations = BUY metals/BTC.
    # Distinct from DFII10 real yield in metals_cross_asset/gold_real_yield_paradox.
    register_enhanced("breakeven_inflation_momentum",
                      "portfolio.signals.breakeven_inflation_momentum",
                      "compute_breakeven_inflation_momentum_signal",
                      requires_context=True, max_confidence=0.7)
    # Trend Slope Momentum — EMA-smoothed log-price slope z-score + 50d momentum
    # 2026-05-14: arxiv 2511.08571 (Forecast-to-Fill). Continuous probability,
    # not binary crossover. 60/40 trend/momentum blend.
    register_enhanced("trend_slope_momentum",
                      "portfolio.signals.trend_slope_momentum",
                      "compute_trend_slope_momentum_signal",
                      max_confidence=0.7)
    # TTM Squeeze — volatility compression (BB inside KC) + momentum breakout
    # 2026-05-16: John Carter; TrendSpider 2025 (68-72% filtered accuracy).
    # Fires rarely (only on squeeze release after 3+ bars). Cross-asset.
    # Different from volatility.py bb_squeeze: uses BB-inside-KC (not width)
    # plus linreg momentum histogram for direction prediction.
    register_enhanced("ttm_squeeze",
                      "portfolio.signals.ttm_squeeze",
                      "compute_ttm_squeeze_signal",
                      max_confidence=0.7)
    # 2026-05-15 LLM shadow-enrollment: scaffold registrations for three
    # on-disk LLMs that previously had no wrapper. Each compute_*_signal
    # returns HOLD/conf=0/feature_unavailable=True until real inference is
    # implemented. Registering now so the dispatch loop produces telemetry
    # (shadow_registry days-in-shadow, llm_probability_log abstention rows)
    # that proves the wiring is alive before inference work begins.
    #
    # max_confidence=0.7 matches the established cap on LLM voters
    # (ministral, qwen3, claude_fundamental, forecast).
    register_enhanced("finance_llama",
                      "portfolio.signals.finance_llama",
                      "compute_finance_llama_signal",
                      requires_context=True, max_confidence=0.7)
    register_enhanced("cryptotrader_lm",
                      "portfolio.signals.cryptotrader_lm",
                      "compute_cryptotrader_lm_signal",
                      requires_context=True, max_confidence=0.7)
    register_enhanced("meta_trader",
                      "portfolio.signals.meta_trader",
                      "compute_meta_trader_signal",
                      requires_context=True, max_confidence=0.7)
    register_enhanced("tsi_chop_mr",
                      "portfolio.signals.tsi_chop_mr",
                      "compute_tsi_chop_mr_signal",
                      max_confidence=0.7)
    # Amihud illiquidity regime — liquidity regime detector (all assets)
    # 2026-05-18: Amihud 2002 (11K citations). ILLIQ = |return|/dollar_volume.
    # Z-scored vs 60d rolling. High ILLIQ = thin market (fakeout risk).
    # Low ILLIQ = thick market (breakout conviction). First liquidity signal.
    register_enhanced("amihud_illiquidity_regime",
                      "portfolio.signals.amihud_illiquidity_regime",
                      "compute_amihud_illiquidity_regime_signal",
                      max_confidence=0.7)
    register_enhanced("absorption_ratio_regime",
                      "portfolio.signals.absorption_ratio_regime",
                      "compute_absorption_ratio_regime_signal",
                      requires_context=True, max_confidence=0.7)
    register_enhanced("connors_rsi2",
                      "portfolio.signals.connors_rsi2",
                      "compute_connors_rsi2_signal",
                      requires_context=True, max_confidence=0.75)
    register_enhanced("adx_regime_switch",
                      "portfolio.signals.adx_regime_switch",
                      "compute_adx_regime_switch_signal",
                      max_confidence=0.7)
    # Sentiment Extremity Gate — F&G intensity (not direction) as regime gate
    # 2026-05-20: Farzulla 2026 (arxiv:2602.07018). Extreme sentiment causes
    # adverse selection (wider spreads, worse fills). BUYs work better in
    # moderate F&G (30-70), not extreme fear. Counter-intuitive vs existing
    # fear_greed signal. Crypto-only (alt.me F&G is crypto-specific).
    register_enhanced("sentiment_extremity_gate",
                      "portfolio.signals.sentiment_extremity_gate",
                      "compute_sentiment_extremity_gate_signal",
                      requires_context=True, max_confidence=0.7)
    register_enhanced("choppiness_regime_gate",
                      "portfolio.signals.choppiness_regime_gate",
                      "compute_choppiness_regime_gate_signal",
                      max_confidence=0.7)
    # AutoTune Adaptive Cycle — Ehlers autocorrelation periodogram + bandpass
    # 2026-05-23: TASC May 2026. Detects dominant cycle period via rolling ACF,
    # tunes adaptive bandpass filter. ROC zero-crossing with correlation gate.
    # Cross-asset, OHLCV-only. Composite score 8.1/10.
    register_enhanced("autotune_adaptive_cycle",
                      "portfolio.signals.autotune_adaptive_cycle",
                      "compute_autotune_adaptive_cycle_signal",
                      max_confidence=0.7)
    # BOCPD Regime Switch — Bayesian Online Changepoint Detection (all assets)
    # 2026-05-24: Wood, Roberts, Zohren (2021), JFDS. Sharpe improvement ~33%.
    # Switches trend-following ↔ mean-reversion on detected regime breaks.
    register_enhanced("bocpd_regime_switch",
                      "portfolio.signals.bocpd_regime_switch",
                      "compute_bocpd_regime_switch_signal",
                      max_confidence=0.7)
    register_enhanced("btc_gold_correlation_regime",
                      "portfolio.signals.btc_gold_correlation_regime",
                      "compute_btc_gold_correlation_regime_signal",
                      requires_context=True,
                      max_confidence=0.7)
    register_enhanced("kalman_trend_momentum",
                      "portfolio.signals.kalman_trend_momentum",
                      "compute_kalman_trend_momentum_signal",
                      max_confidence=0.7)
    # Stablecoin Supply Ratio — SSR z-score + supply momentum + price divergence
    # 2026-05-27: CryptoQuant SSR, DefiLlama API (free, no auth). Crypto-only.
    # Low SSR = stablecoin buying power building. Orthogonal to all existing signals.
    register_enhanced("stablecoin_supply_ratio",
                      "portfolio.signals.stablecoin_supply_ratio",
                      "compute_stablecoin_supply_ratio_signal",
                      requires_context=True, max_confidence=0.7)
    # MSTR mNAV discount — valuation arbitrage (MSTR only); capped at 0.7
    # 2026-05-27: market_cap / (BTC_holdings × BTC_price). Discount <0.95x = BUY.
    # Post-ETF structural shift from 2-3x premiums to sub-1.0x.
    register_enhanced("mstr_mnav_discount",
                      "portfolio.signals.mstr_mnav_discount",
                      "compute_mstr_mnav_discount_signal",
                      requires_context=True, max_confidence=0.7)


_register_defaults()
