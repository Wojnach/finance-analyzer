"""Network momentum signal — cross-asset momentum spillover.

Academic basis: Pu et al. (2023), "Network Momentum across Asset Classes",
arXiv:2308.11294.  Sharpe 1.511, 22% annual return across 64 futures,
2000-2022 backtest.

Simplified implementation: instead of GNN graph learning, uses rolling
Pearson correlation to weight peer assets' momentum contribution.

Core insight: momentum propagates across correlated assets.  When peer
assets show positive momentum but the target asset lags, there is a
BUY opportunity (momentum catch-up).  When peers show negative momentum
but the target leads, SELL (expecting pull-down).

3 sub-indicators via majority vote:
    1. Own TSMOM           — target's own multi-scale momentum
    2. Network divergence  — peer momentum vs own momentum mismatch
    3. Correlation regime  — correlation strength (high = spillover likely)

Data: yfinance for peer asset daily closes (free, no API key).
Applicable to: all 5 tickers (BTC-USD, ETH-USD, XAU-USD, XAG-USD, MSTR).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.network_momentum")

MIN_ROWS = 65
_CACHE_TTL = 3600  # 1 hour

# Momentum lookback windows (trading days)
_MOM_WINDOWS = [5, 20, 60]

# Rolling correlation window
_CORR_WINDOW = 60

# Divergence thresholds — calibrated conservatively
_DIVERGENCE_THRESHOLD = 0.3  # std devs of network-vs-own momentum gap
_STRONG_DIVERGENCE = 0.8

# Peer tickers fetched via yfinance.  Mapped to our internal names
# so we can correlate with the target asset's OHLCV.
_YF_PEERS = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "XAU-USD": "GC=F",     # gold futures (24h coverage)
    "XAG-USD": "SI=F",     # silver futures (24h coverage)
    "SPY": "SPY",           # equity risk-on proxy
}
_YF_DOWNLOAD_TICKERS = ["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"]

# Map our internal ticker names to yfinance peer column names
_TICKER_TO_YF = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "XAU-USD": "GC=F",
    "XAG-USD": "SI=F",
    "MSTR": "SPY",  # MSTR correlates with risk-on (SPY) and BTC
}


def _fetch_peer_closes() -> pd.DataFrame | None:
    """Fetch ~4 months of daily closes for peer assets via yfinance.

    Returns DataFrame with columns = ticker names, index = dates.
    Cached for 1 hour.
    """
    def _do_fetch():
        try:
            import yfinance as yf

            tickers = list(_YF_DOWNLOAD_TICKERS)
            data = yf.download(
                tickers, period="4mo", progress=False, threads=True
            )
            if data is None or data.empty:
                return None

            close_col = "Close"
            if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
                close = data[close_col]
            else:
                close = data[[close_col]]
                close.columns = tickers[:1]

            # Drop rows with all NaN
            close = close.dropna(how="all")
            if len(close) < MIN_ROWS:
                return None

            return close
        except Exception as e:
            logger.warning("network_momentum yfinance fetch failed: %s", e)
            return None

    return _cached("network_momentum_peers", _CACHE_TTL, _do_fetch)


def _vol_scaled_return(series: pd.Series, window: int) -> float:
    """Compute volatility-scaled return over `window` periods.

    Following the paper: ret / realized_vol, where realized_vol is
    the std of daily returns over the same window.
    """
    if len(series) < window + 1:
        return 0.0

    prices = series.iloc[-(window + 1):]
    rets = prices.pct_change(fill_method=None).dropna()
    if len(rets) < window:
        return 0.0

    total_ret = float(prices.iloc[-1] / prices.iloc[0] - 1)
    vol = float(rets.std())
    if vol < 1e-10:
        return 0.0
    return total_ret / vol


def _compute_own_tsmom(close: pd.Series) -> tuple[str, float]:
    """Sub-indicator 1: target's own multi-scale momentum.

    Averages vol-scaled momentum across 5d, 20d, 60d windows.
    """
    moms = []
    for w in _MOM_WINDOWS:
        m = _vol_scaled_return(close, w)
        if np.isfinite(m):
            moms.append(m)

    if not moms:
        return "HOLD", 0.0

    avg_mom = float(np.mean(moms))

    if avg_mom > 0.3:
        return "BUY", min(abs(avg_mom) / 2.0, 1.0)
    if avg_mom < -0.3:
        return "SELL", min(abs(avg_mom) / 2.0, 1.0)
    return "HOLD", 0.0


def _compute_network_divergence(
    own_close: pd.Series,
    peer_closes: pd.DataFrame,
    ticker: str,
) -> tuple[str, float, dict]:
    """Sub-indicator 2: network momentum vs own momentum divergence.

    Computes correlation-weighted average of peers' momentum, then
    compares with own momentum.  A divergence signals potential
    catch-up or pull-down.
    """
    indicators = {}
    yf_col = _TICKER_TO_YF.get(ticker)

    # Compute own momentum (20d, vol-scaled)
    own_mom = _vol_scaled_return(own_close, 20)
    indicators["own_mom_20d"] = safe_float(own_mom)

    # Compute peer momentums and correlations
    peer_moms = {}
    peer_corrs = {}

    for peer_name in _YF_DOWNLOAD_TICKERS:
        if peer_name == yf_col:
            continue  # skip self
        if peer_name not in peer_closes.columns:
            continue

        peer_series = peer_closes[peer_name].dropna()
        if len(peer_series) < MIN_ROWS:
            continue

        # Peer momentum (20d, vol-scaled)
        pm = _vol_scaled_return(peer_series, 20)
        if not np.isfinite(pm):
            continue

        # Rolling correlation with own asset
        # Use the overlapping date range
        if yf_col and yf_col in peer_closes.columns:
            own_yf = peer_closes[yf_col].dropna()
            common_idx = own_yf.index.intersection(peer_series.index)
            if len(common_idx) >= _CORR_WINDOW:
                own_rets = own_yf.loc[common_idx].pct_change(fill_method=None).dropna()
                peer_rets = peer_series.loc[common_idx].pct_change(fill_method=None).dropna()
                common_rets = own_rets.index.intersection(peer_rets.index)
                if len(common_rets) >= _CORR_WINDOW:
                    corr = float(
                        own_rets.loc[common_rets].iloc[-_CORR_WINDOW:]
                        .corr(peer_rets.loc[common_rets].iloc[-_CORR_WINDOW:])
                    )
                    if np.isfinite(corr):
                        peer_moms[peer_name] = pm
                        peer_corrs[peer_name] = corr
        else:
            # Fallback: use target's own close series for correlation
            # Align by taking last N rows of both
            n = min(len(own_close), len(peer_series))
            if n >= _CORR_WINDOW:
                own_rets = own_close.iloc[-n:].pct_change(fill_method=None).dropna()
                peer_rets = peer_series.iloc[-n:].pct_change(fill_method=None).dropna()
                min_len = min(len(own_rets), len(peer_rets))
                if min_len >= _CORR_WINDOW:
                    corr = float(
                        own_rets.iloc[-_CORR_WINDOW:]
                        .reset_index(drop=True)
                        .corr(peer_rets.iloc[-_CORR_WINDOW:].reset_index(drop=True))
                    )
                    if np.isfinite(corr):
                        peer_moms[peer_name] = pm
                        peer_corrs[peer_name] = corr

    if not peer_moms:
        return "HOLD", 0.0, indicators

    # Compute correlation-weighted network momentum
    total_weight = sum(abs(c) for c in peer_corrs.values())
    if total_weight < 0.01:
        return "HOLD", 0.0, indicators

    network_mom = sum(
        peer_corrs[p] * peer_moms[p] for p in peer_moms
    ) / total_weight

    indicators["network_mom"] = safe_float(network_mom)
    indicators["avg_corr"] = safe_float(
        float(np.mean(list(peer_corrs.values())))
    )
    indicators["n_peers"] = len(peer_moms)

    for p in peer_moms:
        short_name = p.replace("-", "").replace("=", "").lower()
        indicators[f"peer_{short_name}_mom"] = safe_float(peer_moms[p])
        indicators[f"peer_{short_name}_corr"] = safe_float(peer_corrs[p])

    # Divergence: network momentum vs own momentum
    divergence = network_mom - own_mom
    indicators["divergence"] = safe_float(divergence)

    if divergence > _DIVERGENCE_THRESHOLD:
        # Peers are more bullish than us -> expect catch-up
        conf = min(abs(divergence) / (_STRONG_DIVERGENCE * 2), 1.0)
        return "BUY", conf, indicators
    if divergence < -_DIVERGENCE_THRESHOLD:
        # Peers are more bearish than us -> expect pull-down
        conf = min(abs(divergence) / (_STRONG_DIVERGENCE * 2), 1.0)
        return "SELL", conf, indicators

    return "HOLD", 0.0, indicators


def _compute_correlation_regime(
    peer_closes: pd.DataFrame, ticker: str
) -> tuple[str, float]:
    """Sub-indicator 3: correlation strength regime.

    When cross-asset correlations are high, spillover is more reliable.
    When correlations break down, network signals are noise.
    """
    yf_col = _TICKER_TO_YF.get(ticker)
    if not yf_col or yf_col not in peer_closes.columns:
        return "HOLD", 0.0

    own_series = peer_closes[yf_col].dropna()
    if len(own_series) < _CORR_WINDOW + 1:
        return "HOLD", 0.0

    own_rets = own_series.pct_change(fill_method=None).dropna().iloc[-_CORR_WINDOW:]

    correlations = []
    for peer in _YF_DOWNLOAD_TICKERS:
        if peer == yf_col or peer not in peer_closes.columns:
            continue
        peer_rets = peer_closes[peer].dropna().pct_change(fill_method=None).dropna()
        # Align indices
        common = own_rets.index.intersection(peer_rets.index)
        if len(common) >= 30:
            corr = float(own_rets.loc[common].corr(peer_rets.loc[common]))
            if np.isfinite(corr):
                correlations.append(abs(corr))

    if not correlations:
        return "HOLD", 0.0

    avg_abs_corr = float(np.mean(correlations))

    # High correlation regime: network signals are more reliable
    # This sub-signal amplifies the network divergence signal
    if avg_abs_corr > 0.5:
        return "BUY", avg_abs_corr  # direction determined by other sub-signals
    if avg_abs_corr < 0.2:
        return "HOLD", 0.0  # low correlation = network signals are noise

    return "HOLD", 0.0


def compute_network_momentum_signal(
    df: pd.DataFrame, context: dict | None = None
) -> dict[str, Any]:
    """Compute network momentum signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
            (minimum 65 rows)
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"]
    ticker = (context or {}).get("ticker", "")

    # Fetch peer asset data
    peer_closes = _fetch_peer_closes()

    # Sub-indicator 1: own multi-scale momentum
    own_tsmom, own_conf = _compute_own_tsmom(close)

    # Sub-indicator 2: network divergence (the core signal)
    if peer_closes is not None:
        net_div, net_conf, net_indicators = _compute_network_divergence(
            close, peer_closes, ticker
        )
    else:
        net_div, net_conf, net_indicators = "HOLD", 0.0, {}

    # Sub-indicator 3: correlation regime
    if peer_closes is not None:
        corr_regime, corr_conf = _compute_correlation_regime(
            peer_closes, ticker
        )
    else:
        corr_regime, corr_conf = "HOLD", 0.0

    # The correlation regime sub-signal inherits the network divergence
    # direction: it answers "is the network signal trustworthy?" not
    # "which direction?"
    if corr_regime == "BUY" and net_div != "HOLD":
        corr_regime = net_div  # same direction as divergence

    votes = [own_tsmom, net_div, corr_regime]
    action, confidence = majority_vote(votes, count_hold=False)

    # Compute aggregate momentum indicators
    mom_5d = _vol_scaled_return(close, 5)
    mom_20d = _vol_scaled_return(close, 20)
    mom_60d = _vol_scaled_return(close, 60)

    indicators = {
        "own_mom_5d": safe_float(mom_5d),
        "own_mom_20d": safe_float(mom_20d),
        "own_mom_60d": safe_float(mom_60d),
        "own_tsmom_conf": safe_float(own_conf),
        "net_div_conf": safe_float(net_conf),
        "corr_regime_conf": safe_float(corr_conf),
    }
    indicators.update(net_indicators)

    return {
        "action": action,
        "confidence": min(confidence, 0.7),
        "sub_signals": {
            "own_tsmom": own_tsmom,
            "network_divergence": net_div,
            "correlation_regime": corr_regime,
        },
        "indicators": indicators,
    }
