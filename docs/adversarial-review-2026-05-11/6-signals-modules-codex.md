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
session id: 019e17a7-3fab-7b81-9d4d-b20a19924215
--------
user
You are doing an ADVERSARIAL code review of the signals-modules subsystem (all 50 detector modules in portfolio/signals/) of a quantitative trading system at Q:\finance-analyzer. Sandbox: read-only.

In-scope files:
- portfolio/signals/*.py — the 50 detector modules. 33 active + 19 disabled-pending-validation.

Active detectors include: trend, momentum, volume_flow, volatility, candlestick, structure, fibonacci, smart_money, heikin_ashi, mean_reversion, calendar_seasonal, macro_regime, momentum_factors, news_event, econ_calendar, forecast, claude_fundamental, futures_flow, metals_cross_asset, dxy_cross_asset, cot_positioning, credit_spread.

Disabled-pending: ml_classifier_disabled (force-HOLD), oscillators, orderbook_flow, smart_money (disabled globally 2026-04-24), mahalanobis_turbulence, crypto_evrp, futures_basis, hurst_regime, shannon_entropy, vix_term_structure, gold_real_yield_paradox, cross_asset_tsmom, copper_gold_ratio, statistical_jump_regime, network_momentum, ovx_metals_spillover, xtrend_equity_spillover, complexity_gap_regime, realized_skewness.

Project rules:
- Every detector must return a verdict + confidence in a consistent shape (read portfolio/signal_utils.py to confirm).
- Force-HOLD via DISABLED_SIGNALS list in signal_engine.py — modules can still register but their vote is ignored.
- Each detector must be horizon-aware (some signals work at 3h but fail at 1d) — single global verdict is a bug.
- Live prices only; no precomputed signal data shall sneak in.
- Detectors must handle empty/short OHLCV gracefully (early-data startup).
- Look-ahead bias is the #1 silent failure: signals computed using future bars.

Adversarial focus per detector:
1. Look-ahead bias: indexing df.iloc[-1] but referencing future close; using talib funcs that need full-bar; using pandas .shift incorrectly.
2. NaN handling: ta libs leak NaN at warm-up; detector returns BUY/SELL when underlying value is NaN.
3. Division-by-zero / log of non-positive.
4. Hardcoded thresholds that don't match documented values (RSI<30, MACD cross, etc).
5. Tz-naive timestamps mixed with tz-aware.
6. Signature drift: missing or extra args vs registry expectation.
7. Detectors that silently swallow ImportError when an optional dep is missing — produce silent HOLD with no log.
8. Disabled detectors that still side-effect global state on import.
9. Inconsistent return for empty input (some return HOLD, others return None, others raise).
10. Module-level network calls / file reads at import time — bad for tests.
11. Per-ticker gating missing: signal works for crypto but is applied to metals.
12. Cache pollution: shared dicts not keyed by ticker.

Output format — strict markdown:
# Codex adversarial review: signals-modules
## Summary
(call out the worst 3-5 detectors by name)

## P0 — Blockers
- path:line — Why it bites: ... Fix: ...
## P1 — High
## P2 — Medium
## P3 — Low
## Tests missing

Be concrete. Cite file/line/detector. No false-positive padding. 50 files is a lot — don't pad with style nits.
codex
Reviewing `Q:\finance-analyzer\portfolio\signals` against the subsystem rules, starting with the shared contract in `signal_utils.py` and the engine/registry wiring before I walk detector-by-detector for concrete failures.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Name' in Q:\finance-analyzer\portfolio\signals
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content -Path Q:\\finance-analyzer\\signal_engine.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content -Path Q:\\finance-analyzer\\portfolio\\signal_utils.py" in Q:\finance-analyzer
 succeeded in 969ms:
__pycache__
__init__.py
calendar_seasonal.py
candlestick.py
claude_fundamental.py
complexity_gap_regime.py
copper_gold_ratio.py
cot_positioning.py
credit_spread.py
cross_asset_tsmom.py
crypto_evrp.py
crypto_macro.py
cubic_trend_persistence.py
drift_regime_gate.py
dxy_cross_asset.py
econ_calendar.py
fibonacci.py
forecast.py
futures_basis.py
futures_flow.py
gold_real_yield_paradox.py
hash_ribbons.py
heikin_ashi.py
hurst_regime.py
intraday_seasonality.py
macro_regime.py
mahalanobis_turbulence.py
mean_reversion.py
metals_cross_asset.py
momentum_factors.py
momentum.py
network_momentum.py
news_event.py
orderbook_flow.py
oscillators.py
ovx_metals_spillover.py
realized_skewness.py
residual_pair_reversion.py
shannon_entropy.py
smart_money.py
statistical_jump_regime.py
structure.py
treasury_risk_rotation.py
trend.py
vix_term_structure.py
vol_ratio_regime.py
volatility.py
volume_flow.py
vwap_zscore_mr.py
williams_vix_fix.py
xtrend_equity_spillover.py
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 969ms:
"""Shared signal utility functions.

Canonical implementations of common technical-analysis helpers used across
multiple signal modules.  Import from here instead of duplicating locally.

All functions operate on ``pd.Series`` inputs and return ``pd.Series``
(or ``float`` for ``safe_float``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Returns NaN where insufficient data (min_periods = period).
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average using pandas ewm.

    Uses ``adjust=False`` for recursive EMA (standard in TA).
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI.

    Uses ``clip()`` for separating gains/losses (numerically stable).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def safe_float(val) -> float:
    """Convert *val* to float, returning ``NaN`` for non-finite / missing values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    try:
        f = float(val)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA / SMMA).

    Equivalent to EMA with ``alpha = 1 / period``.
    """
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, min_periods=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average with linearly increasing weights.

    Weight_i = i + 1 for i in 0..period-1 (most recent bar has highest weight).
    """
    weights = np.arange(1, period + 1, dtype=float)

    def _apply_wma(x: np.ndarray) -> float:
        return np.dot(x, weights) / weights.sum()

    return series.rolling(window=period, min_periods=period).apply(
        _apply_wma, raw=True,
    )


def majority_vote(votes: list, count_hold: bool = False) -> tuple:
    """Compute majority vote from a list of BUY/SELL/HOLD strings.

    HOLD votes are treated as abstentions - they do not block a directional
    result.  Only BUY vs SELL counts determine direction.  This means a
    composite with 5 HOLDs and 2 BUYs produces BUY (not HOLD), because the
    HOLDs are neutral, not opposing votes.

    Args:
        votes: List of "BUY", "SELL", or "HOLD" strings
        count_hold: If False (default), confidence = winner / active_voters (BUY+SELL only).
                    If True, confidence = winner / total_votes (including HOLD).

    Returns:
        (action, confidence) tuple where action is "BUY", "SELL", or "HOLD"
    """
    buy = sum(1 for v in votes if v == "BUY")
    sell = sum(1 for v in votes if v == "SELL")
    hold = sum(1 for v in votes if v == "HOLD")

    active = buy + sell
    total = buy + sell + hold

    if active == 0:
        return "HOLD", 0.0

    denom = total if count_hold else active

    if buy > sell:
        return "BUY", round(buy / denom, 4) if denom > 0 else 0.0
    elif sell > buy:
        return "SELL", round(sell / denom, 4) if denom > 0 else 0.0
    else:
        # Tie between buy and sell - no clear direction.
        # HOLD confidence is always 0.0 - it's the absence of a signal, not a
        # directional vote.
        return "HOLD", 0.0


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of Change: ``100 * (current - n_periods_ago) / n_periods_ago``."""
    shifted = series.shift(period)
    return 100.0 * (series - shifted) / shifted.replace(0, np.nan)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:28:29.146539Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 1 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
[31;1mGet-Content: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mGet-Content -Path Q:\finance-analyzer\signal_engine.py[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mCannot find path 'Q:\finance-analyzer\signal_engine.py' because it does not exist.[0m

 exited 1 in 982ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
[31;1mGet-Content: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m [36;1mGet-Content -Path Q:\finance-analyzer\signal_engine.py[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mCannot find path 'Q:\finance-analyzer\signal_engine.py' because it does not exist.[0m

codex
The shared utility contract is simple primitives only, so the return-shape expectation is likely enforced in the engine or registry layer. I’m locating that wiring now and then scanning all detector entrypoints for signature and data-shape drift.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files Q:\\finance-analyzer | rg \"signal_engine\\.py"'$|signals.*registry|register_signal|DISABLED_SIGNALS|def detect_|def generate_|def get_signal"' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Name' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content -Path Q:\\finance-analyzer\\portfolio\\signals\\__init__.py" in Q:\finance-analyzer
 succeeded in 753ms:
.claude
.git
.hypothesis
.openclaw
.playwright-cli
.playwright-mcp
.pytest_cache
.pytest_tmp
.remember
.ruff_cache
.venv
.venv-train
.venv-unsloth
.worktree
.worktrees
dashboard
data
docs
llama.cpp
logs
memory
models
output
portfolio
scripts
tests
training
unsloth_compiled_cache
user_data
worktrees
_check_positions.py
_check_tasks.ps1
_check.py
_checkin.py
_diag.py
_diag2.py
_journal.py
_kill_all_loops.py
_kill_dupes.py
_l2_action.py
_l2_checkin.py
_parse_log.py
_run_analyze.py
_run.py
_send_tg.py
_start_loop.py
_test_claude_call.py
_test_integration.py
_test_watch_parse.py
_tmp_check.ps1
.git_commit_msg.txt
.gitattributes
.gitignore
.mcp.example.json
.mcp.json
0
AGENTS.md
CLAUDE.local.md
CLAUDE.md
config.example.json
config.json
HEARTBEAT.md
home_phone_full.png
IDENTITY.md
mypy.ini
nul
phone-assets-loaded.png
phone-assets.png
phone-avanza-live.png
phone-avanza.png
phone-bottomsheet.png
phone-dark.png
phone-decision-detail.png
phone-decisions.png
phone-health.png
phone-home-final.png
phone-home.png
phone-more.png
phone-prices.png
phone-refresh-clicked.png
phone-refresh-flash.png
phone-settings.png
phone-signals.png
portfolio_phone.png
pyproject.toml
README.md
requirements-dev.txt
requirements.txt
send_telegram.py
SESSION_PLAN.md
SOUL.md
start-loop.bat
test_digest.py
TODO.md
TOOLS.md
tradesv3.dryrun.sqlite
USER.md
WORKTREE_README.md
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 734ms:
"""Enhanced signal modules - grouped technical indicators.

Each module computes multiple sub-indicators and returns a composite vote.
"""
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 782ms:
Q:\finance-analyzer\portfolio\signal_engine.py
Q:\finance-analyzer\tests\test_signal_engine.py
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content -Path Q:\\finance-analyzer\\portfolio\\signal_engine.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"DISABLED_SIGNALS|detect_|generate_|return .*confidence|confidence\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
 succeeded in 672ms:
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:185:            confidence: float 0.0–1.0
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:192:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:210:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:212:    # Cap confidence at 0.7 (convention for new signals)
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:213:    confidence = min(confidence, 0.7)
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:217:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\futures_flow.py:3:Composite signal with 6 sub-indicators, majority vote, confidence capped at 0.7.
Q:\finance-analyzer\portfolio\signals\futures_flow.py:231:        dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\futures_flow.py:240:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\futures_flow.py:257:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\futures_flow.py:279:    action, confidence = majority_vote(list(sub.values()))
Q:\finance-analyzer\portfolio\signals\futures_flow.py:280:    confidence = min(confidence, _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\futures_flow.py:307:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:37:        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:46:            return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:53:            return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:95:        action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:99:            action, confidence = "HOLD", 0.0
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:101:        # Scale confidence by z-score magnitude (further from VWAP = higher conviction)
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:104:            confidence = confidence * (0.5 + 0.5 * z_magnitude)
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:105:            confidence = min(confidence, 0.85)
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:109:            "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:125:        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:227:    dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:231:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:334:    action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:338:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:272:        dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:274:    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:310:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:332:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:334:    # Cap confidence
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:335:    confidence = min(confidence, MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:339:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:15:magnitude up to a 0.5% cap (a ~5 sigma move saturates confidence).
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:32:# Above this 1h move magnitude, confidence saturates at 1.0. A 0.5% 1h
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:47:        dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:48:        HOLD with 0 confidence if ticker is not metals or DXY data is
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:52:        "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:78:        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:82:        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:85:        confidence = 0.0
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:89:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\fibonacci.py:5:with confidence score.
Q:\finance-analyzer\portfolio\signals\fibonacci.py:45:def _detect_trend(close: pd.Series, period: int = 20) -> str:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:387:        ``confidence``   : float 0.0-1.0 (fraction of active voters that
Q:\finance-analyzer\portfolio\signals\fibonacci.py:395:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\fibonacci.py:436:        trend = _detect_trend(close, period=20)
Q:\finance-analyzer\portfolio\signals\fibonacci.py:532:    action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\fibonacci.py:536:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:176:        dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:181:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:205:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:207:    confidence = min(confidence, _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:225:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\futures_basis.py:196:        dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\futures_basis.py:200:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\futures_basis.py:236:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\futures_basis.py:238:    # Cap confidence
Q:\finance-analyzer\portfolio\signals\futures_basis.py:239:    confidence = min(confidence, _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\futures_basis.py:249:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\trend.py:14:confidence when insufficient data is available.
Q:\finance-analyzer\portfolio\signals\trend.py:447:            'confidence': float (0.0 - 1.0),
Q:\finance-analyzer\portfolio\signals\trend.py:473:    - Returns HOLD with 0.0 confidence when data is insufficient.
Q:\finance-analyzer\portfolio\signals\trend.py:480:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\trend.py:558:    action, confidence = majority_vote(sub_signals_list)
Q:\finance-analyzer\portfolio\signals\trend.py:572:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:181:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:197:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:216:        "confidence": min(confidence, 0.7),
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:197:        dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:201:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:224:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:245:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:35:# Max confidence cap
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:189:        action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:193:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:259:    result["action"], result["confidence"] = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:261:    # Cap confidence
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:262:    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\forecast.py:493:# Regime-aware confidence discount — Chronos has a mean-reversion bias
Q:\finance-analyzer\portfolio\signals\forecast.py:495:_REGIME_DISCOUNT_TRENDING = 0.5   # Halve confidence in trending regimes
Q:\finance-analyzer\portfolio\signals\forecast.py:496:_REGIME_DISCOUNT_HIGH_VOL = 0.6   # Reduce confidence in high-vol regimes
Q:\finance-analyzer\portfolio\signals\forecast.py:626:    """Return confidence multiplier based on market regime.
Q:\finance-analyzer\portfolio\signals\forecast.py:629:    In trending markets, this is wrong, so we discount confidence.
Q:\finance-analyzer\portfolio\signals\forecast.py:650:    - Regime discount: trending markets → reduce confidence (mean-reversion bias)
Q:\finance-analyzer\portfolio\signals\forecast.py:652:    - Good accuracy: use raw vote, scale confidence by accuracy
Q:\finance-analyzer\portfolio\signals\forecast.py:658:    Returns (action, confidence, gating_info) where gating_info is a dict with
Q:\finance-analyzer\portfolio\signals\forecast.py:676:        "base_confidence": base_conf,
Q:\finance-analyzer\portfolio\signals\forecast.py:715:        # Good accuracy — use raw vote, scale confidence by accuracy
Q:\finance-analyzer\portfolio\signals\forecast.py:717:        # Apply regime discount (trending → lower confidence for mean-reversion bias)
Q:\finance-analyzer\portfolio\signals\forecast.py:746:        action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\forecast.py:750:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\forecast.py:856:            result["indicators"]["chronos_1h_conf"] = chronos["1h"].get("confidence", 0)
Q:\finance-analyzer\portfolio\signals\forecast.py:861:            result["indicators"]["chronos_24h_conf"] = chronos["24h"].get("confidence", 0)
Q:\finance-analyzer\portfolio\signals\forecast.py:891:            result["indicators"]["kronos_1h_conf"] = kr["1h"].get("confidence", 0)
Q:\finance-analyzer\portfolio\signals\forecast.py:900:            result["indicators"]["kronos_24h_conf"] = kr["24h"].get("confidence", 0)
Q:\finance-analyzer\portfolio\signals\forecast.py:924:    result["action"], result["confidence"], gating_info = _accuracy_weighted_vote(
Q:\finance-analyzer\portfolio\signals\forecast.py:936:    # Cap confidence (already capped inside _accuracy_weighted_vote, but belt-and-suspenders)
Q:\finance-analyzer\portfolio\signals\forecast.py:937:    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\forecast.py:956:                "confidence": result["confidence"],
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:17:At least 50 rows recommended; returns HOLD with 0.0 confidence when
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:496:            'confidence': 0.0 - 1.0,
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:522:    - Returns HOLD with 0.0 confidence when data is insufficient.
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:529:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:640:    action, confidence = majority_vote(sub_list)
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:644:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:158:        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:162:        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:170:        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:180:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:192:        "confidence": min(confidence, 0.7),
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:136:        dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:138:    _HOLD = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:230:    # Regime confidence: how clearly the indicators agree
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:247:    action, vote_confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:249:    # Scale confidence by regime clarity
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:250:    confidence = min(vote_confidence * regime_clarity, 0.7)
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:254:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\oscillators.py:4:BUY/SELL/HOLD signal with confidence score.
Q:\finance-analyzer\portfolio\signals\oscillators.py:447:        ``confidence``   : float 0.0-1.0 (proportion of sub-signals agreeing
Q:\finance-analyzer\portfolio\signals\oscillators.py:455:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\oscillators.py:582:    action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\oscillators.py:586:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:250:        dict with keys: action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:252:    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:300:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:320:        "confidence": min(confidence, 0.7),
Q:\finance-analyzer\portfolio\signals\credit_spread.py:260:        dict with keys: action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\credit_spread.py:263:        "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\credit_spread.py:306:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\credit_spread.py:318:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:138:    # Rising entropy = market becoming noisier = reduce confidence
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:176:    Entropy momentum adds confidence adjustment.
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:183:        dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:186:        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:192:        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:216:            "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:225:            "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:233:        base_confidence = 0.5
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:235:        # Stronger trend = higher confidence
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:237:            base_confidence += 0.15
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:239:            base_confidence += 0.08
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:243:            base_confidence += 0.10
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:246:            base_confidence -= 0.10
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:250:        base_confidence += entropy_depth * 0.3  # up to +0.195
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:252:        base_confidence = max(0.1, min(0.7, base_confidence))
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:260:            base_confidence = 0.0
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:271:            "confidence": round(base_confidence, 4),
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:276:    # Neutral entropy → use trend but with reduced confidence
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:278:        base_confidence = 0.3
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:281:            base_confidence += 0.10
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:283:            base_confidence -= 0.15
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:285:        base_confidence = max(0.1, min(0.5, base_confidence))
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:296:            "confidence": round(base_confidence, 4),
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:303:        "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:39:REGIME_DECAY = 10      # Bars before regime confidence starts decaying
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:42:def _detect_jumps(returns: pd.Series, vol: pd.Series,
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:151:            confidence: float 0.0-1.0
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:156:        "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:176:    jumps = _detect_jumps(log_returns, rolling_vol)
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:230:    action, raw_confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:233:    # Persistence-based confidence: longer regime = higher confidence
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:236:    # Regime decay: after REGIME_DECAY bars without a new jump, confidence decays
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:240:    confidence = raw_confidence * (0.5 + 0.3 * persistence_factor + 0.2 * jump_recency)
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:241:    confidence = min(max(confidence, 0.0), 1.0)
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:243:    # In high-vol regime, cap confidence
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:245:        confidence = min(confidence, 0.5)
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:254:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:234:        dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:238:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:282:        confidence = 0.7  # High conviction when all conditions align
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:286:        confidence = 0.0
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:290:        confidence = 0.0
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:299:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:43:            detect_trade_throughs,
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:58:    tt = detect_trade_throughs(trades)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:102:        "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:172:    # is non-directional.  The actual effect is the 0.3x confidence penalty
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:190:    action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:193:        confidence *= 0.3
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:201:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:51:    "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:339:{{"TICKER": {{"action": "BUY|SELL|HOLD", "confidence": 0.0-0.7}}, ...}}
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:342:- confidence max 0.7
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:343:- HOLD with 0.0 confidence if no strong fundamental view
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:545:    """Parse Haiku JSON → {ticker: {"action": ..., "confidence": ...}}."""
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:554:        conf = min(float(val.get("confidence", 0.0)), _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:557:            "confidence": round(conf, 4),
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:587:            "confidence": round(conf, 4),
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:616:        # Opus provides its own conviction — use it to scale confidence
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:618:        # Blend: majority_vote confidence weighted by Opus conviction
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:623:            "confidence": round(conf, 4),
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:654:        confidence = result.get("confidence", 0.0)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:657:            action == "HOLD" and confidence == 0.0 and not reasoning
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:664:            "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:863:                "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:902:        dict with action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:14:majority vote; confidence is the fraction of non-HOLD votes that agree with
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:15:the majority direction.  Maximum confidence is capped at 0.6 because
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:39:# Maximum confidence for any calendar signal
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:333:        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-0.6),
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:338:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:480:        result["confidence"] = 0.0
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:483:        result["confidence"] = min(round(buy_count / active_votes, 2), _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:486:        result["confidence"] = min(round(sell_count / active_votes, 2), _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:490:        result["confidence"] = 0.0
Q:\finance-analyzer\portfolio\signals\structure.py:10:majority vote; confidence is the fraction of non-HOLD votes that agree with
Q:\finance-analyzer\portfolio\signals\structure.py:189:        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-1.0),
Q:\finance-analyzer\portfolio\signals\structure.py:194:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\structure.py:260:    result["action"], result["confidence"] = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:4:BUY/SELL/HOLD signal with confidence score.
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:414:        ``confidence``   : float 0.0-1.0 (proportion of voting sub-signals
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:422:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:593:    action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:597:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:146:            confidence: float 0.0–1.0
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:153:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:168:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:170:    # Cap confidence at 0.7 (this is a regime-based signal, not pure price)
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:171:    confidence = min(confidence, 0.7)
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:175:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:249:            confidence: float 0.0-1.0
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:255:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:322:                "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:342:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:362:        confidence = 0.0
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:367:        confidence = min(0.7, 0.35 + 0.15 * (z_abs - 2.0))
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:368:        confidence = max(0.0, confidence)
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:372:            confidence = min(0.7, confidence * 1.15)
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:376:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\news_event.py:42:# Max confidence cap
Q:\finance-analyzer\portfolio\signals\news_event.py:512:        action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\news_event.py:516:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\news_event.py:605:    result["action"], result["confidence"] = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\news_event.py:607:    # Cap confidence
Q:\finance-analyzer\portfolio\signals\news_event.py:608:    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\candlestick.py:46:        ``confidence``        – 0.0 – 1.0
Q:\finance-analyzer\portfolio\signals\candlestick.py:52:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\candlestick.py:90:    trend = _detect_trend(tail)
Q:\finance-analyzer\portfolio\signals\candlestick.py:157:        confidence = 0.0
Q:\finance-analyzer\portfolio\signals\candlestick.py:160:        # (b) strength of the majority.  Max realistic confidence ~ 0.85
Q:\finance-analyzer\portfolio\signals\candlestick.py:163:        confidence = round(min(majority * 0.25, 1.0), 2)
Q:\finance-analyzer\portfolio\signals\candlestick.py:167:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\candlestick.py:177:def _detect_trend(df: pd.DataFrame) -> str:
Q:\finance-analyzer\portfolio\signals\volume_flow.py:245:        confidence : float    0.0 - 1.0 (proportion agreeing with majority)
Q:\finance-analyzer\portfolio\signals\volume_flow.py:251:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\volume_flow.py:349:        action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\volume_flow.py:353:            "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:180:        dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:183:        "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:246:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:254:    # Cap confidence at 0.7 for external data signals
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:255:    confidence = min(confidence, 0.7)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:259:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:14:is inherently slower-moving than price-based signals.  Max confidence
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:321:        Standard signal dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:325:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:403:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:405:    # Cap confidence at 0.7 (external data, weekly update cycle)
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:406:    confidence = min(confidence, 0.7)
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:410:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\macro_regime.py:12:majority vote; confidence is the fraction of non-HOLD votes that agree with
Q:\finance-analyzer\portfolio\signals\macro_regime.py:318:        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-1.0),
Q:\finance-analyzer\portfolio\signals\macro_regime.py:324:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\macro_regime.py:417:    result["action"], result["confidence"] = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:210:        dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:214:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:254:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:256:    # Boost confidence for strong z-scores
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:258:        confidence = min(confidence * 1.2, 0.7)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:262:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:4:a majority-vote composite BUY/SELL/HOLD signal with confidence score.
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:74:        confidence = min(abs(z) / 3.0, 1.0)
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:75:        return confidence, "BUY", indicators
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:77:        confidence = min(abs(z) / 3.0, 1.0)
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:78:        return confidence, "SELL", indicators
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:137:        confidence = min(kurt / 5.0, 0.8)
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:138:        return confidence, "BUY", indicators
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:141:        confidence = min(kurt / 5.0, 0.8)
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:142:        return confidence, "SELL", indicators
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:193:        dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:198:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:209:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:228:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:237:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\momentum.py:4:BUY/SELL/HOLD signal with confidence score.
Q:\finance-analyzer\portfolio\signals\momentum.py:301:        ``confidence``   : float 0.0-1.0 (proportion of sub-signals agreeing
Q:\finance-analyzer\portfolio\signals\momentum.py:309:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\momentum.py:432:    action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\momentum.py:436:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:4:signal confidence. Three sub-indicators:
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:10:During high-alpha hours, confidence is boosted. During low-alpha hours,
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:159:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:168:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:190:        confidence = 0.0
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:194:        confidence = min(base_conf * (combined_mult / 1.0), _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:197:        confidence = min(0.2 + trend_strength * 3.0, 0.4) * combined_mult
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:200:        confidence = 0.0
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:202:    confidence = round(min(max(confidence, 0.0), _MAX_CONFIDENCE), 4)
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:206:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\smart_money.py:85:def _detect_bos(
Q:\finance-analyzer\portfolio\signals\smart_money.py:129:def _detect_choch(
Q:\finance-analyzer\portfolio\signals\smart_money.py:185:def _detect_fvg(
Q:\finance-analyzer\portfolio\signals\smart_money.py:261:def _detect_liquidity_sweep(
Q:\finance-analyzer\portfolio\signals\smart_money.py:313:def _detect_supply_demand(
Q:\finance-analyzer\portfolio\signals\smart_money.py:413:        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-1.0),
Q:\finance-analyzer\portfolio\signals\smart_money.py:419:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\smart_money.py:480:            bos_vote, bos_ind = _detect_bos(
Q:\finance-analyzer\portfolio\signals\smart_money.py:490:            choch_vote, structure_label = _detect_choch(swing_highs, swing_lows)
Q:\finance-analyzer\portfolio\signals\smart_money.py:498:            fvg_vote, unfilled_count = _detect_fvg(
Q:\finance-analyzer\portfolio\signals\smart_money.py:508:            sweep_vote = _detect_liquidity_sweep(
Q:\finance-analyzer\portfolio\signals\smart_money.py:519:            sd_vote, in_demand, in_supply = _detect_supply_demand(
Q:\finance-analyzer\portfolio\signals\smart_money.py:551:        action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\smart_money.py:555:            "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\volatility.py:305:        confidence : float  -- 0.0-1.0 agreement ratio among sub-signals
Q:\finance-analyzer\portfolio\signals\volatility.py:411:    composite_action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\volatility.py:415:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\volatility.py:426:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:149:        dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:153:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:171:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:173:    confidence = min(confidence, _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:176:        confidence = min(max(confidence, 0.6), _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:189:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:255:        dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:259:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:304:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:306:    # Cap confidence at 0.7 (external/cross-asset data signal)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:307:    confidence = min(confidence, 0.7)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:309:    # Boost confidence for extreme turbulence
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:311:        confidence = min(confidence * 1.15, 0.7)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:315:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:58:def _detect_timeframe(df: pd.DataFrame) -> str:
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:79:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:90:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:100:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:109:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:117:    tf = _detect_timeframe(df)
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:149:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:151:    confidence = min(confidence * 0.7, 0.7)
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:155:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\network_momentum.py:318:        dict with keys: action, confidence, sub_signals, indicators
Q:\finance-analyzer\portfolio\signals\network_momentum.py:323:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\network_momentum.py:360:    action, confidence = majority_vote(votes, count_hold=False)
Q:\finance-analyzer\portfolio\signals\network_momentum.py:379:        "confidence": min(confidence, 0.7),
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:276:        "action": "HOLD", "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:437:    action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:441:        "confidence": round(confidence, 4),
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:4:BUY/SELL/HOLD signal with confidence score.
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:380:        ``confidence``   : float 0.0-1.0 (proportion of active voters agreeing
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:388:        "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:507:    action, confidence = majority_vote(votes)
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:511:        "confidence": confidence,
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:3:Composite signal with 5 sub-indicators, majority vote, confidence capped at 0.7.
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:205:        dict with action, confidence, sub_signals, indicators.
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:214:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:236:            "confidence": 0.0,
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:261:    action, confidence = majority_vote(list(sub.values()))
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:262:    confidence = min(confidence, _MAX_CONFIDENCE)
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:274:        "confidence": confidence,
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 989ms:
"""Signal generation engine - 32-signal voting system with weighted consensus."""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from portfolio.indicators import detect_regime
from portfolio.shared_state import (
    FEAR_GREED_TTL,
    FUNDING_RATE_TTL,
    MINISTRAL_TTL,
    ONCHAIN_TTL,
    SENTIMENT_TTL,
    VOLUME_TTL,
    _cached,
    _cached_or_enqueue,
)
from portfolio.signal_registry import get_enhanced_signals, load_signal_func
from portfolio.signal_utils import true_range
from portfolio.tickers import CRYPTO_SYMBOLS, DISABLED_SIGNALS, GPU_SIGNALS, METALS_SYMBOLS, SIGNAL_NAMES, STOCK_SYMBOLS

logger = logging.getLogger("portfolio.signal_engine")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOCAL_MODEL_ACCURACY_TTL = 1800

# ADX computation cache - keyed by (id(df), len(df), last_close) tuple so a
# new DataFrame allocated at the same address as a freed one doesn't get a
# stale hit (see _compute_adx for the C1 content-key rationale, 2026-05-10
# fixed the annotation drift that read ``dict[int, .]``).
_adx_cache: dict[tuple[int, int, float], float | None] = {}
_adx_lock = threading.Lock()  # BUG-86: protect concurrent access from ThreadPoolExecutor
_ADX_CACHE_MAX = 200  # prevent unbounded growth

# BUG-178 diagnostics: per-ticker last-signal tracker.
# Updated right before each enhanced signal's compute_fn() is called so that
# when the BUG-178 ticker pool timeout fires, main.py can ask which signal
# each stuck ticker was running. Surfaces silent hangs (signals that never
# complete and therefore never trip the [SLOW] >1s logger).
# Added 2026-04-10 after a 49-event BUG-178 audit traced silent hangs to the
# disabled-signals dispatch path. Cheap (single dict write per signal call).
_last_signal_per_ticker: dict[str, tuple[str, float]] = {}
_last_signal_lock = threading.Lock()


def _set_last_signal(ticker: str, sig_name: str) -> None:
    """Record the signal currently being computed for a ticker (BUG-178 diag)."""
    with _last_signal_lock:
        _last_signal_per_ticker[ticker] = (sig_name, time.monotonic())


def get_last_signal(ticker: str) -> tuple[str, float] | None:
    """Return (sig_name, elapsed_seconds) for the most recent signal start
    on this ticker, or None if no signal has been recorded.

    Used by main.py's BUG-178 timeout handler to identify which signal hung.
    """
    with _last_signal_lock:
        entry = _last_signal_per_ticker.get(ticker)
    if entry is None:
        return None
    sig_name, started = entry
    return sig_name, time.monotonic() - started


# BUG-178 phase log (added 2026-04-15): records per-ticker phase durations
# inside generate_signal()'s post-dispatch code. The __post_dispatch__ marker
# above was too coarse - it collapsed 7+ distinct post-dispatch operations
# (accuracy load, weighted consensus, penalties, linear factor, etc.) into
# a single "after dispatch" bucket, so slow cycles with elapsed_since_set
# ~170s out of a 180s pool timeout gave us zero signal about which phase
# was actually slow.
#
# Each phase records (phase_name, duration_seconds). main.py's slow-cycle
# diagnostic reads this log when the pool timeout fires so we can see the
# full phase breakdown retrospectively. Bounded per-ticker (replaced on
# each generate_signal call) so memory is constant.
_phase_log_per_ticker: dict[str, list[tuple[str, float]]] = {}
_phase_log_lock = threading.Lock()

_PHASE_WARN_THRESHOLD_S = 2.0

# Defensive bound on the number of distinct ticker keys kept. In production
# this is 5 (Tier-1 symbols); tests and probes may pass arbitrary names and
# slowly grow the dict. When we exceed the cap, prune the least-recently-
# used entries (reset cycles refresh them, so LRU by last-reset is fine).
# Prior callers were silently leaking one small list per unique ticker name.
_PHASE_LOG_MAX_TICKERS = 64


def _reset_phase_log(ticker: str) -> None:
    """Clear the phase log for a ticker at the start of generate_signal.

    Also enforces _PHASE_LOG_MAX_TICKERS by pruning older entries when the
    dict grows past the cap - cheap per-call O(n) but n is bounded and the
    prune happens at most once per generate_signal invocation.
    """
    if not ticker:
        return
    with _phase_log_lock:
        if len(_phase_log_per_ticker) >= _PHASE_LOG_MAX_TICKERS and ticker not in _phase_log_per_ticker:
            # Evict oldest half - we don't need true LRU, just bounded memory.
            # `iter(dict)` yields insertion order in CPython 3.7+; dropping
            # the first half gives us amortized O(1) per call.
            evict_count = len(_phase_log_per_ticker) // 2
            for old_key in list(_phase_log_per_ticker)[:evict_count]:
                del _phase_log_per_ticker[old_key]
        _phase_log_per_ticker[ticker] = []


def _record_phase(ticker: str, phase: str, start_mono: float) -> float:
    """Record a phase completion for a ticker. Returns the phase duration.

    Logs WARNING if duration > _PHASE_WARN_THRESHOLD_S so that slow
    individual phases (e.g., cold accuracy_stats load, lock contention)
    are visible in portfolio.log without waiting for a BUG-178 timeout.
    """
    if not ticker:
        return 0.0
    dur = time.monotonic() - start_mono
    with _phase_log_lock:
        _phase_log_per_ticker.setdefault(ticker, []).append((phase, dur))
    if dur > _PHASE_WARN_THRESHOLD_S:
        logger.warning("[SLOW-PHASE] %s/%s: %.1fs", ticker, phase, dur)
    return dur


def get_phase_log(ticker: str) -> list[tuple[str, float]]:
    """Return the phase breakdown for a ticker's last generate_signal call.

    Used by main.py's BUG-178 slow-cycle diagnostic to dump per-phase timing
    when the ticker pool times out. Returns an empty list if no log exists.
    """
    with _phase_log_lock:
        return list(_phase_log_per_ticker.get(ticker, []))

_LOCAL_MODEL_HOLD_THRESHOLD = 0.55
_LOCAL_MODEL_MIN_SAMPLES = 30
_LOCAL_MODEL_LOOKBACK_DAYS = 30

# 2026-05-11 - Stage 2 Batch 1: dead-zone HOLD-bias reduction.
# Investigation found EMA / BB / MACD core voters abstaining 88-94% on
# metals: _total_applicable=20, _voters=5. User rationale: HOLD is for
# managing open positions, not for entry decisions - entries should
# always pick a direction. So when these signals would otherwise HOLD
# (gap < 0.5% on EMA, price inside the BB band, |MACD hist| below
# threshold), we now emit a *weak* directional vote based on the
# secondary derivative (slope of EMA9 vs EMA21, distance from BB mid,
# slope of MACD histogram). The strong-vote paths are unchanged.
#
# The "conf" returned by the helpers is informational - it goes into
# extra_info so consumers can see the soft confidence. Downstream
# weighting (accuracy gate, regime mult, horizon mult, correlation
# group leader) still applies, so a soft vote on a force-HOLD signal
# is still force-HOLD'd. We are NOT bypassing the accuracy gate.
#
# Stays well below the typical strong-vote confidence (0.5-0.8) so
# the weight ranking continues to prefer real conviction signals.
EMA_DEAD_ZONE_SOFT_CONF = 0.20
BB_INSIDE_SOFT_CONF = 0.15
MACD_DEAD_ZONE_SOFT_CONF = 0.20
# 2026-05-11 - Stage 2 Batch 2: extend soft-directional pattern to two
# more high-HOLD enhanced voters. Candlestick abstained 87.6% on metals
# (HOLD when no recognised pattern); forecast abstained 87.0% (HOLD when
# Chronos low-conf or gated). Soft confs deliberately LOWER than Batch 1
# (0.12-0.15) because the secondary derivative we read (raw body
# direction / price+EMA slope) is even weaker than EMA9/EMA21 slope
# divergence - i.e. less of a "true regime hint", more of a tiebreaker.
# Strong-vote paths are unchanged; soft branches only fire when the
# enhanced compute_fn returned HOLD AND the secondary derivative has a
# clear direction. Mixed/ambiguous secondary signals keep HOLD.
CANDLESTICK_DEAD_ZONE_SOFT_CONF = 0.15
FORECAST_DEAD_ZONE_SOFT_CONF = 0.12
# How many of the last N candle bodies must agree (close vs open sign)
# before the candlestick soft branch emits a directional vote. 3/3 is a
# clean unanimity; 2/3 keeps HOLD because that's a coinflip+1, well
# within candle noise - the soft branch is not supposed to fight a
# 2-vs-1 split.
_CANDLE_BODY_LOOKBACK = 3
# How many bars the forecast soft branch reads to estimate the
# price+EMA21 slope alignment. 5 bars is the minimum that filters
# single-bar noise without lagging the regime - same rationale as
# _DEAD_ZONE_SLOPE_LOOKBACK but a touch longer because we want a
# slower, "actually still rising / falling" read, not a tick.
_FORECAST_SLOPE_LOOKBACK = 5
# How many bars to look back when measuring EMA slope and MACD slope
# inside the dead-zone helpers. 3 bars = enough to filter single-bar
# noise without lagging the regime.
_DEAD_ZONE_SLOPE_LOOKBACK = 3
# Slope tie-break tolerance. EMA slopes within `price * 1e-5` of each
# other count as "flat" and keep HOLD. MACD histogram slope within
# `1e-9` (absolute) of zero counts as flat.
_EMA_FLAT_EPS_REL = 1e-5
_MACD_FLAT_EPS = 1e-9
# 2026-05-11 (Codex review Fix A): MACD dead-zone helper must gate on
# *magnitude* of the current histogram before considering slope. Without
# this, any non-crossover bar with non-flat slope produces a soft vote -
# including bars where |hist| is large and the strong-vote path is the
# correct authority. The strong path handles crossovers (sign flip), so
# whenever |current_hist| is comfortably above zero but no crossover
# occurred, the right behaviour is HOLD and let the strong-path own it.
# 0.05 is a small absolute fraction picked to cover typical noisy zero
# crossings without bleeding into normal histogram amplitudes.
MACD_DEAD_ZONE_MAGNITUDE_THRESHOLD = 0.05


def _ema_dead_zone_vote(ind, df, lookback=_DEAD_ZONE_SLOPE_LOOKBACK):
    """Return (vote, conf) when EMA gap is in the dead zone (<0.5%).

    Compares the slope of EMA9 vs EMA21 over the last `lookback` bars.
    EMA9 rising faster than EMA21 -> soft BUY. EMA9 falling faster
    -> soft SELL. Slopes within `tiny_eps` of each other -> HOLD.

    Returns ("HOLD", 0.0) if df is missing/short or slopes truly flat.

    2026-05-11: introduced to convert dead-zone HOLDs into weak
    directional votes. See module-level rationale comment above.
    """
    if df is None or "close" not in df or len(df) < lookback + 21:
        return "HOLD", 0.0
    try:
        close = df["close"]
        ema9_series = close.ewm(span=9, adjust=False).mean()
        ema21_series = close.ewm(span=21, adjust=False).mean()
        # Slope = last - value-`lookback`-bars-ago. Linear; sign is what
        # matters, not magnitude.
        ema9_slope = float(ema9_series.iloc[-1] - ema9_series.iloc[-1 - lookback])
        ema21_slope = float(ema21_series.iloc[-1] - ema21_series.iloc[-1 - lookback])
    except (KeyError, IndexError, ValueError, TypeError):
        return "HOLD", 0.0
    price = float(ind.get("close", 0.0)) or float(close.iloc[-1])
    tiny_eps = abs(price) * _EMA_FLAT_EPS_REL
    if ema9_slope > ema21_slope + tiny_eps:
        return "BUY", EMA_DEAD_ZONE_SOFT_CONF
    if ema9_slope < ema21_slope - tiny_eps:
        return "SELL", EMA_DEAD_ZONE_SOFT_CONF
    return "HOLD", 0.0


def _bb_inside_band_vote(ind):
    """Return (vote, conf) when price is inside the Bollinger band.

    Uses normalized band position: (price - mid) / (upper - mid),
    clamped to [-1, +1]. Position > 0.6 -> soft SELL (near upper),
    < -0.6 -> soft BUY (near lower), else HOLD (mid-band).

    2026-05-11: introduced to convert dead-zone HOLDs into weak
    directional votes. See module-level rationale comment above.
    """
    try:
        price = float(ind["close"])
        mid = float(ind["bb_mid"])
        upper = float(ind["bb_upper"])
    except (KeyError, TypeError, ValueError):
        return "HOLD", 0.0
    half_width = upper - mid
    if half_width <= 0:
        return "HOLD", 0.0
    band_position = (price - mid) / half_width
    # Clamp to [-1, +1] so degenerate / wick data can't produce
    # arbitrarily large positions.
    band_position = max(-1.0, min(1.0, band_position))
    if band_position > 0.6:
        return "SELL", BB_INSIDE_SOFT_CONF
    if band_position < -0.6:
        return "BUY", BB_INSIDE_SOFT_CONF
    return "HOLD", 0.0


def _macd_dead_zone_vote(ind, df, lookback=_DEAD_ZONE_SLOPE_LOOKBACK):
    """Return (vote, conf) when |MACD hist| is in the dead zone.

    Measures histogram slope over the last `lookback` bars. Rising
    histogram -> soft BUY; falling -> soft SELL. Flat (within
    `_MACD_FLAT_EPS` absolute) -> HOLD.

    Recomputes the histogram from `df["close"]` because `ind` only
    carries the last two histogram values.

    2026-05-11: introduced to convert dead-zone HOLDs into weak
    directional votes. See module-level rationale comment above.

    2026-05-11 (Codex Fix A): added magnitude gate. The strong-vote
    path owns any bar where the histogram has meaningful amplitude;
    this helper must only fire when |current_hist| is actually in the
    dead zone (small absolute value). Without this gate, every non-
    crossover bar with non-flat slope produced a soft vote - even on
    large histogram magnitudes where the strong path's "no crossover"
    decision should win and the answer is genuinely HOLD.
    """
    if df is None or "close" not in df or len(df) < lookback + 26:
        return "HOLD", 0.0
    # 2026-05-11 Codex Fix A: prefer ind's current histogram (already
    # computed by the standard pipeline) for the magnitude gate. Falls
    # through to recomputing if ind didn't carry it.
    current_hist = None
    try:
        current_hist = float(ind.get("macd_hist", 0.0))
    except (TypeError, ValueError):
        current_hist = None
    try:
        close = df["close"]
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - macd_signal
        hist_now = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-1 - lookback])
    except (KeyError, IndexError, ValueError, TypeError):
        return "HOLD", 0.0
    # If ind lacked a usable current hist, fall back to the recomputed
    # value so the magnitude gate still applies.
    if current_hist is None:
        current_hist = hist_now
    # Magnitude gate: only fire in the actual dead zone. Outside the
    # dead zone the strong path is responsible; we return HOLD here
    # rather than emitting a soft vote that would overlap with (or
    # contradict) the strong-vote logic.
    if abs(current_hist) >= MACD_DEAD_ZONE_MAGNITUDE_THRESHOLD:
        return "HOLD", 0.0
    delta = hist_now - hist_prev
    if delta > _MACD_FLAT_EPS:
        return "BUY", MACD_DEAD_ZONE_SOFT_CONF
    if delta < -_MACD_FLAT_EPS:
        return "SELL", MACD_DEAD_ZONE_SOFT_CONF
    return "HOLD", 0.0


def _candlestick_dead_zone_vote(df, lookback=_CANDLE_BODY_LOOKBACK):
    """Return (vote, conf) when no candlestick pattern matched.

    Fallback secondary derivative: count bullish (close>open) vs bearish
    (close<open) bodies over the last `lookback` bars. Unanimous bullish
    -> soft BUY at CANDLESTICK_DEAD_ZONE_SOFT_CONF. Unanimous bearish ->
    soft SELL. Anything mixed (incl. doji-style equality) stays HOLD.

    2026-05-11 (Stage 2 Batch 2): introduced because the
    `compute_candlestick_signal` strong path abstains ~87% of the time
    when none of hammer / engulfing / doji / star matches. A 3-of-3
    body-direction agreement is a *weak* tie-breaker, not a pattern
    claim - hence the 0.15 soft conf and the 3/3 unanimity gate (no
    2/1 splits). df is the same OHLCV frame the strong path consumed.
    """
    if df is None or "close" not in df or "open" not in df:
        return "HOLD", 0.0
    try:
        if len(df) < lookback:
            return "HOLD", 0.0
        tail = df[["open", "close"]].iloc[-lookback:]
        closes = tail["close"].astype(float).to_numpy()
        opens = tail["open"].astype(float).to_numpy()
    except (KeyError, IndexError, ValueError, TypeError):
        return "HOLD", 0.0
    bull = int((closes > opens).sum())
    bear = int((closes < opens).sum())
    if bull == lookback:
        return "BUY", CANDLESTICK_DEAD_ZONE_SOFT_CONF
    if bear == lookback:
        return "SELL", CANDLESTICK_DEAD_ZONE_SOFT_CONF
    return "HOLD", 0.0


def _forecast_dead_zone_vote(df, forecast_indicators,
                             lookback=_FORECAST_SLOPE_LOOKBACK):
    """Return (vote, conf) when forecast voted HOLD but price+EMA align.

    Fallback secondary derivative: compare the slope of close prices to
    the slope of EMA21 over the last `lookback` bars. Both rising ->
    soft BUY; both falling -> soft SELL; mixed slopes -> HOLD.

    Critical guard: this branch ONLY fires when the forecast pipeline
    actually produced data. If `forecast_indicators` is empty / lacks
    Chronos output / has `models_disabled` or `error` keys, the soft
    branch is skipped entirely so we don't substitute slope-following
    for an unrun forecast. The user-visible contract: HOLD when the
    forecast model couldn't run, soft vote only when it ran but
    couldn't muster enough confidence.

    2026-05-11 (Stage 2 Batch 2): introduced because Chronos
    + accuracy gating force ~87% HOLD on metals (low confidence, vol
    gate, accuracy gate). When the forecast ran but didn't commit, a
    price+EMA21 slope-alignment read is a defensible weak tiebreaker.
    """
    if df is None or "close" not in df:
        return "HOLD", 0.0
    indicators = forecast_indicators or {}
    # Skip when the forecast pipeline didn't run at all. Any of these
    # markers indicates "no Chronos data" - we must not substitute.
    if indicators.get("models_disabled"):
        return "HOLD", 0.0
    if indicators.get("error"):
        return "HOLD", 0.0
    # Chronos is the live composite voter (Kronos is shadow per current
    # config). Require its 1h output to be present - that's the cheap
    # "did Chronos produce a forecast this cycle?" check.
    if indicators.get("chronos_1h_pct") is None:
        # Belt-and-suspenders: also accept chronos_ok flag if present.
        if not indicators.get("chronos_ok"):
            return "HOLD", 0.0
    try:
        if len(df) < lookback + 21:
            return "HOLD", 0.0
        close = df["close"].astype(float)
        ema21_series = close.ewm(span=21, adjust=False).mean()
        # Linear: last - value-`lookback`-bars-ago. Sign is what
        # matters, magnitude is informational.
        price_slope = float(close.iloc[-1] - close.iloc[-1 - lookback])
        ema21_slope = float(
            ema21_series.iloc[-1] - ema21_series.iloc[-1 - lookback]
        )
    except (KeyError, IndexError, ValueError, TypeError):
        return "HOLD", 0.0
    if price_slope > 0 and ema21_slope > 0:
        return "BUY", FORECAST_DEAD_ZONE_SOFT_CONF
    if price_slope < 0 and ema21_slope < 0:
        return "SELL", FORECAST_DEAD_ZONE_SOFT_CONF
    return "HOLD", 0.0


# Accuracy gate: signals with blended accuracy below this threshold are
# force-HOLD (treated like DISABLED_SIGNALS but dynamically). A signal at
# 44% is noise, not a reliable contrarian indicator - inverting it just
# produces different noise with whiplash as accuracy oscillates around 50%.
# 2026-04-11 (A-PR-batch-5): raised 0.45  0.47. The signal audit on
# 2026-04-10 found four signals sitting in the 45-47% band that the
# previous gate let through (volatility_sig 0.453, trend 0.454, etc.).
# Tightening the gate by 2pp removes ~4 coin-flip-adjacent signals from
# consensus while leaving the well-performing tier untouched.
ACCURACY_GATE_THRESHOLD = 0.47
ACCURACY_GATE_MIN_SAMPLES = 30  # need enough data before gating
# 2026-04-12: Tiered gate for high-confidence coin-flips. With 5000+ samples,
# a signal at 49.8% is coin-flip with p < 0.001 - no amount of waiting will
# fix it. Raising the gate to 50% for established signals removes structure
# (49.8%, 12K sam), heikin_ashi (49.6%, 23K sam) etc. while letting newer
# signals with <5000 samples prove themselves at the standard 47% threshold.
# 2026-04-16: raised high-sample min 5000 -> 10000. Investigation of W15/W16
# consensus collapse found the 5000 threshold catching signals during regime
# transitions where 5000 samples is too few to distinguish true coin-flip
# from transient degradation. 10000 samples reduces false-positive gating
# (e.g., a signal at 49.5% over 6000 samples may still have real edge in
# specific regimes that the aggregate accuracy hides).
_ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD = 0.50
_ACCURACY_GATE_HIGH_SAMPLE_MIN = 7000

# Directional accuracy gate: signals whose BUY or SELL accuracy is below this
# threshold get that direction force-HOLD'd while the other direction can still
# vote.  E.g., qwen3 BUY=30% (gated) but SELL=74.2% (votes normally).
# Uses the same min-samples threshold as the overall gate.
# 2026-04-10: raised from 0.35  0.40 to catch macro_regime BUY (38.9%),
# fibonacci SELL (35.9%), futures_flow both (36-37%).  Now with per-ticker
# directional data, this gate also works per-instrument.
_DIRECTIONAL_GATE_THRESHOLD = 0.40
_DIRECTIONAL_GATE_MIN_SAMPLES = 30

# Directional rescue (2026-04-28): when a signal fails the overall accuracy
# gate but its vote direction has strong accuracy, rescue it at reduced weight.
# E.g., heikin_ashi overall=42.6% (gated) but SELL=55.7%  rescue SELL vote.
# Only triggers when direction accuracy >= 55% with >= 30 samples, giving
# a 5pp safety margin above coin-flip.  Rescued signals get a 0.7x weight
# penalty so they contribute less than fully-passing signals.
_DIRECTIONAL_RESCUE_THRESHOLD = 0.55
_DIRECTIONAL_RESCUE_MIN_SAMPLES = 30
_DIRECTIONAL_RESCUE_WEIGHT_PENALTY = 0.70

# Adaptive recency blend: when recent accuracy diverges from all-time by more
# than this threshold, increase recent weight for faster regime adaptation.
# Normal: 70% recent + 30% all-time. Fast: 90% recent + 10% all-time.
# 2026-04-15: raised normal 0.700.75, fast 0.900.95 to better capture
# recent-regime signals like trend (40.3% alltime  61.6% recent).
# 2026-04-16: REVERTED to 0.70/0.90. The 0.75/0.95 tuning amplified noise
# during the W12-W13 crash -> W14-W16 recovery transition. A 7-day window
# with only 170 samples was dominating a 10K-sample all-time baseline,
# triggering gates on signals whose "bad recent" was just the crash tail
# rolling through the window. 0.70/0.90 gives regime adaptation while
# leaving enough all-time anchor to damp single-week noise.
_RECENCY_DIVERGENCE_THRESHOLD = 0.15  # 15% absolute divergence triggers fast blend
_RECENCY_WEIGHT_NORMAL = 0.70
_RECENCY_WEIGHT_FAST = 0.90
_RECENCY_MIN_SAMPLES = 30  # match ACCURACY_GATE_MIN_SAMPLES (was 50 default)

# Crisis regime: when multiple macro-external signals are simultaneously
# degraded (recent accuracy < 35% with 50+ samples), the market is in a
# regime that breaks fundamental assumptions (e.g., wartime, systemic crisis).
# In crisis mode, apply extra penalty to trend-following signals and boost
# mean-reversion/calendar signals.
_CRISIS_THRESHOLD = 0.35  # signal accuracy below this counts as "broken"
_CRISIS_MIN_BROKEN = 3  # need at least 3 broken macro signals for crisis flag
_CRISIS_TREND_PENALTY = 0.6  # 0.6x weight for trend signals in crisis
_CRISIS_MR_BOOST = 1.3  # 1.3x weight for mean-reversion in crisis

# Directional bias penalty: signals with extreme BUY or SELL bias (>85% of
# their non-HOLD votes in one direction) get penalized because their
# "accuracy" may just reflect market drift rather than genuine edge.
# E.g., calendar (100% BUY) in a ranging-up market looks accurate by luck.
_BIAS_THRESHOLD = 0.85  # >85% BUY or >85% SELL triggers penalty
_BIAS_PENALTY = 0.5  # 0.5x weight for high-bias signals (85-95%)
_BIAS_EXTREME_THRESHOLD = 0.95  # >95% triggers stronger penalty
_BIAS_EXTREME_PENALTY = 0.2  # 0.2x weight for extreme-bias signals (>95%)
_BIAS_MIN_ACTIVE = 30  # need enough active (non-HOLD) votes to judge bias

# IC-based weight multiplier (2026-04-18): adjusts signal weight based on
# Information Coefficient - the rank correlation between a signal's votes and
# actual return magnitude. A signal with 55% accuracy but IC=0.15 catches big
# moves; one with 58% accuracy but IC=0.00 is riding market drift.
_IC_ALPHA = 2.0         # IC sensitivity: IC=0.10  1.20x boost
_IC_MULT_FLOOR = 0.6    # never zero out a signal via IC alone
_IC_MULT_CAP = 1.5      # cap to prevent IC from dominating
_IC_MIN_SAMPLES = 100   # need reliable IC estimate
_IC_STABILITY_MIN = 0.10  # minimum |ICIR| to trust the IC value
_IC_ZERO_PENALTY = 0.85   # phantom performers (|IC|<0.01, 500+ samples) get 0.85x
_IC_ZERO_MIN_SAMPLES = 500  # sample floor for zero-IC penalty
_IC_DATA_TTL = 3600     # IC cache TTL (matches ic_computation.py)

# Signal persistence filter (2026-04-20): require signals to maintain their
# vote for MIN_PERSISTENCE_CYCLES consecutive cycles before counting in
# consensus. Eliminates documented "single-check MACD/RSI/volume improvements
# are noise" pattern. Raw votes are still recorded for accuracy tracking -
# only the consensus input is filtered.
#
# Design: in-memory dict tracks {ticker: {signal: {"vote": X, "cycles": N}}}.
# When a signal's vote matches its previous non-HOLD vote, cycles increments.
# When it flips direction or goes HOLDnon-HOLD for the first time, cycles=1.
# Only signals with cycles >= _PERSISTENCE_MIN_CYCLES get their vote passed
# to consensus; others are treated as HOLD for consensus purposes only.
_PERSISTENCE_MIN_CYCLES = 2        # require 2+ consecutive same-direction votes
_PERSISTENCE_ENABLED = True        # toggle for easy disable
_PERSISTENCE_MAX_TICKERS = 32      # bound on tracked tickers (prod=5, cap guards tests/probes)
_persistence_state: dict[str, dict[str, dict]] = {}  # {ticker: {signal: {"vote": str, "cycles": int}}}
_persistence_lock = threading.Lock()

# 2026-05-11: per-asset relaxation. Metals + crypto run intraday; one cycle
# of confirmation already costs a minute. Stocks keep 2 cycles because
# market-hours-only and short windows make whipsaw more expensive.
_PERSISTENCE_CYCLES_BY_ASSET = {
    "METALS": 1,
    "CRYPTO": 1,
    "STOCK": 2,
}
def _persistence_cycles_for(ticker: str | None) -> int:
    """Return how many same-direction cycles a signal must hold to count."""
    if ticker is None:
        return _PERSISTENCE_MIN_CYCLES
    # Local import-style references - these names are defined at module scope
    # (imported from portfolio.tickers at the top of this file).
    if ticker in METALS_SYMBOLS:
        return _PERSISTENCE_CYCLES_BY_ASSET["METALS"]
    if ticker in CRYPTO_SYMBOLS:
        return _PERSISTENCE_CYCLES_BY_ASSET["CRYPTO"]
    return _PERSISTENCE_CYCLES_BY_ASSET["STOCK"]

# Cross-ticker consensus cache: stores the most recent consensus action per
# ticker so synthetic cross-asset signals can reference other tickers' results.
# Stale reads (MSTR processing before BTC in the same cycle) are acceptable -
# the 60s loop ensures data is at most one cycle old.
_cross_ticker_consensus: dict[str, dict] = {}  # {ticker: {"action": str, "confidence": float}}
_cross_ticker_lock = threading.Lock()


def _apply_persistence_filter(votes: dict[str, str], ticker: str | None) -> dict[str, str]:
    """Filter votes to only include signals that persisted for MIN_PERSISTENCE_CYCLES.

    Returns a new dict with non-persistent signals forced to HOLD.
    The original votes dict is not modified (needed for accuracy tracking).

    Cold-start: on the first cycle for a ticker (no prior state), all signals
    pass through unfiltered. Filtering only activates once we have history.
    """
    if not _PERSISTENCE_ENABLED or not ticker:
        return votes

    # 2026-05-11: resolve the per-asset persistence threshold once per call.
    # _persistence_cycles_for() returns 1 for metals+crypto, 2 for stocks,
    # 2 fallback for ticker=None.
    min_cycles = _persistence_cycles_for(ticker)

    with _persistence_lock:
        # Cold start: if we have NO history for this ticker, seed state and
        # pass all votes through. The filter only applies from cycle 2 onward.
        if ticker not in _persistence_state:
            if len(_persistence_state) >= _PERSISTENCE_MAX_TICKERS:
                evict_count = len(_persistence_state) // 2
                for old_key in list(_persistence_state)[:evict_count]:
                    del _persistence_state[old_key]
            _persistence_state[ticker] = {
                sig: {"vote": vote, "cycles": min_cycles if vote != "HOLD" else 0}
                for sig, vote in votes.items()
            }
            return votes  # first cycle - trust all signals

        ticker_state = _persistence_state[ticker]
        filtered = {}
        for sig, vote in votes.items():
            prev = ticker_state.get(sig)

            if vote == "HOLD":
                # Signal went quiet - reset persistence
                ticker_state[sig] = {"vote": "HOLD", "cycles": 0}
                filtered[sig] = "HOLD"
            elif prev is None or prev["vote"] != vote:
                # New direction or first appearance - start counting
                ticker_state[sig] = {"vote": vote, "cycles": 1}
                # 2026-05-11: when min_cycles == 1 (metals/crypto), a freshly
                # appearing non-HOLD vote already meets the threshold and
                # should be passed through immediately. Stocks keep the
                # original 2-cycle confirmation requirement.
                if 1 >= min_cycles:
                    filtered[sig] = vote
                else:
                    filtered[sig] = "HOLD"
            else:
                # Same direction as previous cycle - increment
                prev["cycles"] += 1
                if prev["cycles"] >= min_cycles:
                    filtered[sig] = vote  # persistent - let it vote
                else:
                    filtered[sig] = "HOLD"  # still provisional

        return filtered


# Disabled signal per-ticker rescue (2026-04-18): signals in DISABLED_SIGNALS
# that have proven accuracy on specific tickers. These are re-enabled for
# compute+consensus on the listed ticker only. The standard accuracy gate
# (47%) still protects against degradation.
# Format: {(signal_name, ticker)} - if (sig, ticker) is in this set, the
# signal is computed and votes for that ticker despite being globally disabled.
# Evidence: data/disabled_signal_rescue_2026-04-18.json
_DISABLED_SIGNAL_OVERRIDES: frozenset[tuple[str, str]] = frozenset({
    # ml on ETH-USD: 55.1% at 3h (1206 samples). Globally disabled at 41.7%
    # because BTC-USD pulls it down to 26.4%. ETH-USD has genuine edge.
    ("ml", "ETH-USD"),
})

# Shadow-safe signals: disabled signals that are pure math (no network I/O)
# and safe to compute every cycle without impacting cycle time. Their votes
# are stored in _shadow_votes (not consensus votes) so outcome_tracker can
# track accuracy while they remain force-HOLD in consensus.
# Network-heavy disabled signals (futures_basis, vix_term_structure,
# gold_real_yield_paradox, cross_asset_tsmom, copper_gold_ratio,
# network_momentum, ovx_metals_spillover, xtrend_equity_spillover,
# orderbook_flow) are NOT shadow-safe - they do
# yfinance/FRED/Binance calls that would blow the 60s cycle budget.
_SHADOW_SAFE_SIGNALS = frozenset({
    "hurst_regime",
    "shannon_entropy",
    # "statistical_jump_regime" - RE-ENABLED 2026-04-29 (52.7% at 110 sam)
    "realized_skewness",
    "oscillators",
    # 2026-04-29: Added compute-only signals to accumulate accuracy data.
    # These use local OHLCV data only - no network calls.
    "complexity_gap_regime",
    "mahalanobis_turbulence",
    "crypto_evrp",
    "hash_ribbons",
    "fibonacci",  # newly disabled, shadow-track to confirm continued poor accuracy
    "calendar",  # 2026-05-09: disabled at 29.3%, shadow-track for recovery
})

# Per-ticker consensus gate: BUG-164.  Suppress all non-HOLD consensus for
# tickers where the system's overall consensus is historically harmful.
# AMD 24.8%, GOOGL 31.3%, META 34.2% - actively wrong.
_PER_TICKER_CONSENSUS_GATE = 0.38  # below 38% = force HOLD
_PER_TICKER_CONSENSUS_MIN_SAMPLES = 50

# Voter-count circuit breaker (2026-04-16, Batch 2 of accuracy gating reconfig).
# When cascaded gates would leave fewer than _MIN_ACTIVE_VOTERS_SOFT active voters
# for a ticker, progressively relax the accuracy gate by _GATE_RELAXATION_STEP
# until the voter floor is met or _GATE_RELAXATION_MAX is reached. Rationale:
# losing voter diversity is worse than letting a borderline signal vote, because
# the consensus is a weighted sum of possibly-correlated signals - 3 correlated
# voters aren't as informative as 5 independent ones.
#
# Expected impact: kicks in during regime transitions where the 47% gate is
# silencing several voters whose recent accuracy dipped to 45-47%. Keeps at
# least 5 voters active by relaxing the gate by up to 6pp (to 41% floor).
# Signals with directional or per-ticker gating are NOT un-gated by this -
# only the overall accuracy gate is relaxed.
_MIN_ACTIVE_VOTERS_SOFT = 5
_GATE_RELAXATION_STEP = 0.02  # relax by 2pp per step
_GATE_RELAXATION_MAX = 0.06   # cap at 6pp below base gate (0.47 -> 0.41)

# Per-ticker signal disable: force HOLD for specific signal+ticker combos
# where accuracy data shows the signal is actively harmful for that instrument.
#
# 2026-04-16 (Batch 4): horizon-specific per-ticker blacklists via
# _TICKER_DISABLED_BY_HORIZON. Structure:
#   {"3h": {ticker: frozenset(bad_signals_at_3h)},
#    "1d": {ticker: frozenset(bad_signals_at_1d)},
#    "_default": {ticker: frozenset(bad_at_ALL_horizons)}}
#
# Compute-time (signal dispatch loop): uses the _default list only. Signals
# compute once per ticker per cycle and their vote is reused across horizons,
# so disabling at compute time requires the signal to be bad at EVERY horizon.
# Horizon-specific entries do NOT skip compute (the vote still exists, but
# is force-HOLD'd per-horizon at consensus time).
#
# Consensus-time: when building consensus for horizon H, apply
# (_default[ticker] | _TICKER_DISABLED_BY_HORIZON[H][ticker]).
#
# Why this structure: the Apr 14 MSTR blacklist was built from 3h accuracy
# data but applied globally, causing W15/W16 consensus collapse (1d dropped
# to 21.9%). 5 of 7 blacklisted MSTR signals were 66-81% accurate at 1d.
# Batch 1 trimmed the list to 2 entries; Batch 4 (this) enables per-horizon
# entries so future audits can say "bad at 3h, fine at 1d" without global
# penalty.
_TICKER_DISABLED_BY_HORIZON: dict[str, dict[str, frozenset]] = {
    # Disabled at ALL horizons - bad everywhere, safe to skip even at compute.
    "_default": {
        # 2026-04-15 audit: per-ticker 3h accuracy gating, retained pending
        # per-horizon audit of 1d/3d/5d behaviors.
        # 2026-04-24 after-hours audit: added structure (metals), credit_spread_risk
        # and macro_regime (XAU), ema (crypto/metals), futures_flow (crypto).
        "ETH-USD": frozenset({"news_event", "qwen3", "smart_money",
                              "ema",           # 17.6% 1d (51 sam)
                              "futures_flow",  # 32.6% 1d (675 sam)
                              }),
        "BTC-USD": frozenset({"smart_money", "heikin_ashi",
                              "futures_flow",  # 39.7% 1d (511 sam)
                              }),
        "XAG-USD": frozenset({"ministral", "credit_spread_risk",
                              "metals_cross_asset", "smart_money",
                              "structure",     # 29.9% 1d (723 sam)
                              "ema",           # 14.7% 1d (34 sam)
                              "sentiment",     # 33.3% 1d (285 sam), 94% BUY-only
                              }),
        "XAU-USD": frozenset({"ministral", "metals_cross_asset",
                              "structure",           # 30.4% 1d (827 sam)
                              "credit_spread_risk",  # 35.4% 1d (413 sam), 38.8% 3h - bad everywhere
                              "macro_regime",        # 34.3% 1d (484 sam)
                              }),
        # 2026-04-16: trimmed from 7 to 2 (Batch 1). Full history in commit
        # fd504d4. Kept: bad at both 3h (33.2%) and 1d (47.8%).
        "MSTR": frozenset({"claude_fundamental", "credit_spread_risk",
                          "statistical_jump_regime",  # 27.0% 1d (74 sam)
                          "realized_skewness",        # 36.0% 1d (50 sam)
                          # 2026-05-10: crashed 40-58pp after never-sell policy broken May 5
                          "sentiment",          # 90.4% -> 39.2%
                          "volume_flow",        # 82.3% -> 33.7%
                          "heikin_ashi",        # 78.6% -> 35.0%
                          "momentum_factors",   # 88.7% -> 30.4%
                          }),
    },
    # 2026-04-16 after-hours audit: signals that PASS global gate (>0.47)
    # but FAIL per-ticker (<0.45 with >=50 samples).
    # Source: accuracy_by_ticker_signal_cached() cross-referenced with
    # global accuracy. Each entry justified by per-ticker accuracy data:
    #
    # 3h: BTC volatility_sig 43.3%/342, bb 44.8%/536;
    #     ETH credit_spread_risk 43.5%/186;
    #     XAU credit_spread_risk 38.8%/170;
    #     XAG forecast 40.3%/248, qwen3 44.8%/413;
    #     MSTR volume 35.6%/1400, volatility_sig 42.9%/319.
    # 1d: BTC news_event 40.8%/671, forecast 42.0%/300;
    #     XAU candlestick 43.3%/656;
    #     MSTR ema 41.4%/1405, bb 44.5%/245.
    "3h": {
        # 2026-04-30 audit: added sentiment (33.8% 3h_recent, 3629 sam, 94.9% BUY).
        # Also added bb for more tickers, forecast for BTC (38.3% 3h_recent).
        "BTC-USD": frozenset({"volatility_sig", "bb",
                              "sentiment",  # 33.8% 3h_recent (3629 sam), 94.9% BUY-only
                              }),
        "ETH-USD": frozenset({"credit_spread_risk",
                              "sentiment",  # 33.8% 3h_recent (3629 sam), 94.9% BUY-only
                              }),
        # credit_spread_risk promoted to _default (2026-04-24)
        "XAU-USD": frozenset({"sentiment",  # 33.8% 3h_recent (3629 sam)
                              }),
        "XAG-USD": frozenset({"forecast", "qwen3",
                              "sentiment",  # 33.8% 3h_recent (3629 sam)
                              }),
        "MSTR": frozenset({"volume", "volatility_sig",
                           "sentiment",  # 33.8% 3h_recent (3629 sam)
                           }),
    },
    "4h": {},
    "12h": {},
    "1d": {
        # 2026-04-24 audit: added econ_calendar (1.8% BTC/ETH), ema (BTC),
        # funding (ETH 12.5%), econ_calendar (XAG 29.5%).
        # 2026-04-30 audit: added signals with <40% 1d_recent accuracy (50+ sam).
        # Key finding: systemic BUY bias - calendar (100% BUY, 30.8%),
        # claude_fundamental (99.3% BUY, 34.2%), momentum_factors (32.7%),
        # volume_flow (35.8%), heikin_ashi (38.2%), crypto_macro (33.8%).
        # These signals are already auto-gated by the blended accuracy gate,
        # but per-horizon blacklists provide defense-in-depth.
        "BTC-USD": frozenset({"news_event", "forecast",
                              "econ_calendar",       # 1.8% 1d (113 sam)
                              "ema",                 # 23.8% 1d (42 sam)
                              "claude_fundamental",  # 34.2% 1d_recent (730 sam), 99.3% BUY-only
                              "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                              "momentum_factors",    # 32.7% 1d_recent (910 sam), 60.1% at 3h - horizon divergence
                              "volume_flow",         # 35.8% 1d_recent (924 sam)
                              "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                              "crypto_macro",        # 33.8% 1d_recent (476 sam)
                              "structure",           # 33.1% 1d_recent (758 sam)
                              }),
        "ETH-USD": frozenset({"econ_calendar",       # 1.8% 1d (113 sam)
                              "funding",             # 12.5% 1d (64 sam)
                              "claude_fundamental",  # 34.2% 1d_recent (730 sam), 99.3% BUY-only
                              "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                              "momentum_factors",    # 32.7% 1d_recent (910 sam)
                              "volume_flow",         # 35.8% 1d_recent (924 sam)
                              "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                              "crypto_macro",        # 33.8% 1d_recent (476 sam)
                              "structure",           # 33.1% 1d_recent (758 sam)
                              }),
        "XAU-USD": frozenset({"candlestick",
                              "claude_fundamental",  # 2026-04-27: metals have no earnings/guidance
                              "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                              "momentum_factors",    # 32.7% 1d_recent (910 sam)
                              "volume_flow",         # 35.8% 1d_recent (924 sam)
                              "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                              "smart_money",         # 34.2% 1d_recent (155 sam)
                              }),
        "XAG-USD": frozenset({"econ_calendar",       # 29.5% 1d (112 sam)
                              "claude_fundamental",  # 2026-04-27: metals have no earnings/guidance
                              "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                              "momentum_factors",    # 32.7% 1d_recent (910 sam)
                              "volume_flow",         # 35.8% 1d_recent (924 sam)
                              "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                              }),
        "MSTR": frozenset({"ema", "bb",
                           "calendar",            # 30.8% 1d_recent (712 sam), 100% BUY-only
                           "momentum_factors",    # 32.7% 1d_recent (910 sam)
                           "volume_flow",         # 35.8% 1d_recent (924 sam)
                           "heikin_ashi",         # 38.2% 1d_recent (709 sam)
                           "smart_money",         # 34.2% 1d_recent (155 sam)
                           "structure",           # 33.1% 1d_recent (758 sam)
                           "macro_regime",        # 40.3% 1d (1475 sam) - moved from _default to preserve 3h
                           }),
    },
    "3d": {
        # 2026-05-10: signals with <45% accuracy at 3d horizon (global).
        # ministral 37.2% (6214 sam), credit_spread_risk 38.6% (1545 sam),
        # ema 39.9% (16662 sam) - all actively harmful at this horizon.
        "BTC-USD": frozenset({"ministral", "credit_spread_risk", "ema"}),
        "ETH-USD": frozenset({"ministral", "credit_spread_risk", "ema"}),
        "XAG-USD": frozenset({"ministral", "credit_spread_risk", "ema"}),
        "XAU-USD": frozenset({"ministral", "credit_spread_risk", "ema"}),
        "MSTR": frozenset({"ministral", "credit_spread_risk", "ema"}),
    },
    "5d": {
        # 2026-05-10: signals with <45% accuracy at 5d horizon.
        # funding 32.1% (728), news_event 42.2% (8251), ema 42.3% (15596),
        # credit_spread_risk 43.1% (1455), heikin_ashi 44.1% (24761).
        "BTC-USD": frozenset({"funding", "news_event", "ema",
                              "credit_spread_risk", "heikin_ashi"}),
        "ETH-USD": frozenset({"funding", "news_event", "ema",
                              "credit_spread_risk", "heikin_ashi"}),
        "XAG-USD": frozenset({"news_event", "ema",
                              "credit_spread_risk", "heikin_ashi"}),
        "XAU-USD": frozenset({"news_event", "ema",
                              "credit_spread_risk", "heikin_ashi"}),
        "MSTR": frozenset({"news_event", "ema",
                           "credit_spread_risk", "heikin_ashi"}),
    },
    "10d": {},
}


# P2-H (2026-04-17): module-load validation of _TICKER_DISABLED_BY_HORIZON
# shape. Catches structural errors (missing _default, invalid horizon keys,
# non-frozenset values) at import time rather than silently at runtime.
_VALID_HORIZON_KEYS = frozenset({"_default", "3h", "4h", "12h", "1d", "3d", "5d", "10d"})
assert "_default" in _TICKER_DISABLED_BY_HORIZON, (
    "_TICKER_DISABLED_BY_HORIZON missing required '_default' key")
for _k, _inner in _TICKER_DISABLED_BY_HORIZON.items():
    assert _k in _VALID_HORIZON_KEYS, (
        f"_TICKER_DISABLED_BY_HORIZON has unknown horizon key {_k!r}; "
        f"valid keys: {sorted(_VALID_HORIZON_KEYS)}")
    assert isinstance(_inner, dict), (
        f"_TICKER_DISABLED_BY_HORIZON[{_k!r}] must be a dict")
    for _tk, _sigs in _inner.items():
        assert isinstance(_sigs, frozenset), (
            f"_TICKER_DISABLED_BY_HORIZON[{_k!r}][{_tk!r}] must be a "
            f"frozenset (got {type(_sigs).__name__})")
del _k, _inner, _tk, _sigs


def _get_horizon_disabled_signals(ticker: str | None, horizon: str | None) -> frozenset:
    """Return signals to force-HOLD for (ticker, horizon). Union of default + horizon-specific.

    P3-1 (2026-04-17): uses .get('_default', {}) defensively instead of []
    subscript. If _default is ever removed at runtime (shouldn't happen -
    module-load assertion prevents it - but defensive), we return an empty
    set rather than crash the hot consensus path.
    """
    if not ticker:
        return frozenset()
    default_map = _TICKER_DISABLED_BY_HORIZON.get("_default", {})
    default_set = default_map.get(ticker, frozenset())
    if not horizon:
        return default_set
    horizon_set = _TICKER_DISABLED_BY_HORIZON.get(horizon, {}).get(ticker, frozenset())
    return default_set | horizon_set


# Backward-compat alias: the compute-time (signal dispatch) gate. Equal to the
# _default list - the minimum set of signals that are bad at every horizon.
# Existing callers reference this name; keep it as a view of _default.
_TICKER_DISABLED_SIGNALS = _TICKER_DISABLED_BY_HORIZON["_default"]


# --- Macro-window regime overlay (2026-04-28) ---
#
# When a high-impact macro event (FOMC/CPI/NFP) is within ~24h past or
# ~72h future, technical/sentiment signals trained on price-pattern
# continuity systematically misvote because price is being driven by
# news. The 2026-04-28 audit found 19 of 21 flagged per-ticker
# degradations real (sentiment 60.9%42.7%, momentum_factors 44.5%30.4%,
# structure 41.7%30.0%, claude_fundamental 63.5%38.9%) - coincident
# with the densest macro week of 2026.
#
# This overlay reduces the influence of those signals during the
# macro window, then auto-reverts when the window passes. It composes
# multiplicatively with the existing regime/horizon weight chain.
MACRO_WINDOW_DOWNWEIGHT_SIGNALS = frozenset({
    "sentiment", "momentum_factors", "structure",
})
MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER = 0.5

# claude_fundamental has a known >75% BUY bias caught by its own
# tier-bias gate. During macro windows the bias dominates because the
# 30-120min LLM cascade (Haiku/Sonnet/Opus) lags real-time regime
# shifts. Force-HOLD instead of down-weighting (stricter than the
# others - its accuracy collapses the most).
MACRO_WINDOW_FORCE_HOLD_SIGNALS = frozenset({"claude_fundamental"})

# 5-minute cache. econ_dates is hardcoded so the underlying data
# doesn't change between cycles, but we still pay an iteration over
# ECON_EVENTS each call. Caching avoids that hit per signal per ticker
# per cycle.
_MACRO_WINDOW_CACHE_TTL_S = 300


# --- Signal (full 32-signal for "Now" timeframe) ---

MIN_VOTERS_CRYPTO = 3  # crypto has 30 signals (8 core + 22 enhanced; ml disabled) - need 3
MIN_VOTERS_STOCK = 3  # stocks have 24-26 signals (7 core + 17-19 enhanced, GPU-dependent) - need 3
MIN_VOTERS_METALS = 2  # 2026-05-11: metals run at noisier intraday horizon
                       # (1m-1h target) where the standard 3-voter floor
                       # almost never fires after persistence filter.
                       # Empirical: XAG sees 5 raw voters  2 post-persistence;
                       # MIN_VOTERS=3 produced 0 trades in 20 days.

# P2-F (2026-04-17 adversarial review): derived floors used by the
# circuit-breaker precondition. Placing here (after MIN_VOTERS_*) keeps the
# relationship explicit and prevents silent drift if the base MIN_VOTERS_*
# changes.
_MIN_VOTERS_BASE = max(MIN_VOTERS_CRYPTO, MIN_VOTERS_STOCK)
# Slate viability floor: the post-exclusion candidate count below which
# relaxation would produce a consensus thinner than any asset class's quorum.
_POST_EXCLUSION_MIN = _MIN_VOTERS_BASE
# Lone-signal escape floor: raised from 2 to _MIN_VOTERS_BASE (3) because a
# 2-voter relaxed consensus is still thinner than any asset class's outer
# quorum, so letting it emit trades was inconsistent with the system's
# design. Codex rounds 6-9 each flagged variants of this issue.
_LONE_SIGNAL_FLOOR = _MIN_VOTERS_BASE

# P2-G (2026-04-17): module-load assertions on constant relationships.
# These catch misconfigurations at import time rather than producing silent
# wrong behavior at runtime.
assert MIN_VOTERS_CRYPTO > 0 and MIN_VOTERS_STOCK > 0, (
    "MIN_VOTERS_* must be positive")
assert _POST_EXCLUSION_MIN <= _MIN_ACTIVE_VOTERS_SOFT, (
    f"_POST_EXCLUSION_MIN ({_POST_EXCLUSION_MIN}) must be <= "
    f"_MIN_ACTIVE_VOTERS_SOFT ({_MIN_ACTIVE_VOTERS_SOFT}); "
    f"otherwise the circuit breaker requires more candidates than it can "
    f"ever accept."
)
assert _GATE_RELAXATION_STEP > 0, (
    "_GATE_RELAXATION_STEP must be positive (else ZeroDivisionError in "
    "circuit-breaker step-count math).")
assert _GATE_RELAXATION_MAX > 0, "_GATE_RELAXATION_MAX must be positive."
assert (ACCURACY_GATE_THRESHOLD - _GATE_RELAXATION_MAX) > _DIRECTIONAL_GATE_THRESHOLD, (
    f"Relaxed overall accuracy gate "
    f"({ACCURACY_GATE_THRESHOLD - _GATE_RELAXATION_MAX:.2f}) must remain "
    f"above _DIRECTIONAL_GATE_THRESHOLD ({_DIRECTIONAL_GATE_THRESHOLD}); "
    f"otherwise directional gating becomes tighter than the relaxed "
    f"accuracy gate and the claim that the directional gate is NEVER "
    f"relaxed becomes meaningless."
)
# Step must divide max cleanly so iteration lands on the intended max.
_rel_ratio = _GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP
assert abs(_rel_ratio - round(_rel_ratio)) < 1e-9, (
    f"_GATE_RELAXATION_STEP ({_GATE_RELAXATION_STEP}) must divide "
    f"_GATE_RELAXATION_MAX ({_GATE_RELAXATION_MAX}) cleanly.")
del _rel_ratio

# Core signals that must have at least 1 active voter for non-HOLD consensus.
# Enhanced signals can strengthen/weaken but never create consensus alone.
CORE_SIGNAL_NAMES = frozenset({
    "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
    "volume", "ministral", "qwen3", "claude_fundamental",
})

# Sentiment hysteresis - prevents rapid flip spam from ~50% confidence oscillation
_prev_sentiment: dict[str, str] = {}  # in-memory cache; seeded from sentiment_state.json on first call
_prev_sentiment_loaded = False
_sentiment_lock = threading.Lock()  # BUG-85: protect concurrent access from ThreadPoolExecutor
_sentiment_dirty = False  # Track whether in-memory state diverged from disk

_SENTIMENT_STATE_FILE = DATA_DIR / "sentiment_state.json"


def _load_prev_sentiments():
    global _prev_sentiment, _prev_sentiment_loaded
    with _sentiment_lock:
        if _prev_sentiment_loaded:
            return
        try:
            from portfolio.file_utils import load_json as _load_json
            data = _load_json(str(_SENTIMENT_STATE_FILE), default=None)
            if data and isinstance(data, dict):
                _prev_sentiment = data.get("prev_sentiment", {})
            # Prune entries for removed tickers
            from portfolio.tickers import ALL_TICKERS
            removed = [k for k in _prev_sentiment if k not in ALL_TICKERS]
            for k in removed:
                del _prev_sentiment[k]
        except Exception:
            logger.warning("Failed to load prev sentiments", exc_info=True)
        _prev_sentiment_loaded = True


def _get_prev_sentiment(ticker):
    _load_prev_sentiments()
    with _sentiment_lock:
        return _prev_sentiment.get(ticker)


def _set_prev_sentiment(ticker, direction):
    """Set sentiment direction for a ticker (thread-safe, batched disk write)."""
    global _sentiment_dirty
    _load_prev_sentiments()
    with _sentiment_lock:
        _prev_sentiment[ticker] = direction
        _sentiment_dirty = True


def flush_sentiment_state():
    """Persist sentiment state to disk. Call once per cycle, not per-ticker.

    BUG-85 fix: batching prevents concurrent per-ticker writes that clobber each other.
    BUG-101 fix: dirty flag cleared only AFTER successful write, so a failed write
    will be retried on the next cycle instead of silently losing state.
    """
    global _sentiment_dirty
    with _sentiment_lock:
        if not _sentiment_dirty:
            return
        snapshot = dict(_prev_sentiment)
    # Write outside the lock to avoid holding it during I/O
    try:
        from portfolio.file_utils import atomic_write_json
        atomic_write_json(_SENTIMENT_STATE_FILE, {"prev_sentiment": snapshot})
        # BUG-101: Only clear dirty flag after successful write
        with _sentiment_lock:
            _sentiment_dirty = False
    except Exception:
        # Dirty flag remains True - next cycle will retry the write
        logger.warning("Failed to persist sentiment state (will retry next cycle)", exc_info=True)


REGIME_WEIGHTS = {
    "trending-up": {
        "ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7,
        # Enhanced: boost trend-following, dampen mean-reversion
        "trend": 1.4, "momentum_factors": 1.3, "heikin_ashi": 1.2,
        "structure": 1.2, "smart_money": 1.1,
        "mean_reversion": 0.6,  # fibonacci removed - disabled 2026-04-29
    },
    "trending-down": {
        "ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7,
        # Enhanced: same as trending-up (trend signals work both ways)
        "trend": 1.4, "momentum_factors": 1.3, "heikin_ashi": 1.2,
        "structure": 1.2, "smart_money": 1.1,
        "mean_reversion": 0.6,  # fibonacci removed - disabled 2026-04-29
    },
    "ranging": {
        "rsi": 1.5, "bb": 1.5, "ema": 0.5,
        # 2026-04-05 audit: macd 58.7% recent (crossover catches range turns)
        "macd": 1.3,
        # Enhanced: boost mean-reversion and level-based signals
        # 2026-04-05 audit: fibonacci 68.2% recent - boost to 1.8 (was 1.6)
        # mean_reversion 65.4% recent - boost to 1.7 (was 1.5)
        # ministral 68.0% recent (Apr 5) - was 1.4x boost but collapsed to 41.5%
        # recent (Apr 26 audit, 41 sam). Removed boost, added to regime gate.
        "mean_reversion": 1.7, "calendar": 1.2,  # fibonacci removed - disabled 2026-04-29
        # 2026-04-05 audit: momentum 58.9% in ranging (2196 samples) - untapped edge
        "momentum": 1.3,
        # 2026-04-04: BUG-161 - oscillators 34-39% per-ticker in ranging.
        # Was 1.2x (boosted), now 0.3x (heavily penalized).
        "oscillators": 0.3,
        # 2026-04-28 research: actual degradation 50-70% (BUY acc 13-25%),
        # previous 0.5-0.7x was insufficient. Lowered to 0.3x.
        "trend": 0.3, "momentum_factors": 0.3, "heikin_ashi": 0.3,
        "structure": 0.4, "fear_greed": 0.3,
    },
    "high-vol": {
        "bb": 1.5, "volume": 1.3, "ema": 0.5,
        # Enhanced: boost volatility-aware and smart money signals
        # 2026-03-31: mean_reversion works in high-vol too (war overshoot/reversion)
        "volatility_sig": 1.4, "smart_money": 1.3, "volume_flow": 1.2,
        "candlestick": 1.2, "mean_reversion": 1.2,
        "trend": 0.6, "calendar": 0.7,
    },
}

# Regime-gated signals: completely silenced (forced HOLD) in certain regimes
# because they produce negative alpha.  Horizon-aware since 2026-03-29:
# BUG-149: trend has 61.6% accuracy on 3h even in ranging - short-term trends
# exist within range-bound markets, so only gate on longer horizons.
# Structure: {regime: {horizon: frozenset(signals), ...}}
# "_default" key applies to horizons not explicitly listed.
REGIME_GATED_SIGNALS: dict[str, dict[str, frozenset[str]]] = {
    "ranging": {
        # 2026-04-02 audit: 13 signals below 45% on 1d_recent. Gate the worst.
        # trend 40.7%, momentum_factors 41.4%, ema 40.8%, heikin_ashi 42.0%,
        # structure 36.1%, fear_greed 25.9%, macro_regime 30.3%,
        # news_event 29.5%, volatility_sig 35.0%, forecast 36.1%,
        # candlestick 44.5%, smart_money 39.6%.
        # The dynamic 45% accuracy gate also catches these, but explicit
        # regime gating is clearer and doesn't depend on blending math.
        "_default": frozenset({
            "trend", "momentum_factors", "ema", "heikin_ashi", "structure",
            "fear_greed", "macro_regime",
            # 2026-04-02: added based on 1d_recent audit
            "news_event", "volatility_sig", "forecast", "smart_money",
            # 2026-04-04: BUG-161/163 - oscillators 34-39% per-ticker,
            # candlestick 44.5% recent (292 sam). Both noise in ranging.
            "oscillators", "candlestick",
            # 2026-04-09: funding 29.9% at 1d (536 sam) but 74.2% at 3h (535 sam).
            # Gate at 1d, let it vote at 3h/4h.
            "funding",
            # 2026-04-12: econ_calendar 34.2% in ranging (1911 sam). SELL-only signal
            # is actively harmful in range-bound markets. 62.8% overall is inflated by
            # unknown-regime (86.8%, 2562 sam) and trending-up (25.9%) dominance.
            "econ_calendar",
            # 2026-04-26: volume_flow collapsed to 40.8% recent (-10.0pp, 1310 sam).
            # credit_spread_risk collapsed to 39.0% recent (-15.2pp, 249 sam).
            # ministral collapsed to 41.5% recent (41 sam) from 58.4% all-time.
            # All three are noise in the current 141h+ ranging regime.
            "volume_flow", "credit_spread_risk", "ministral",
            # 2026-04-27: claude_fundamental 40.5% at 1d_recent (1178 sam),
            # 78-83% BUY bias. Was only gated at 3h/4h but also harms 12h/1d/3d/5d.
            "claude_fundamental",
            # 2026-04-27: sentiment 40.1% at 1d_recent (202 sam), 33.8% at 3h.
            # BUY-only bias. Was gated at 3h/4h in all regimes but still active
            # at _default in ranging where it pushes false BUY consensus.
            "sentiment",
        }),
        # 3h: news_event 58.5%, smart_money 53.1% - decent at short horizons.
        # volatility_sig 47.2%, forecast 47.2% - marginal, let accuracy gate
        # handle them dynamically at 3h.
        # funding 74.2% at 3h (535 sam) - NOT gated here.
        # 2026-04-11: sentiment added - 33.8% at 3h_recent (3629 sam). The 0.5x
        # horizon weight is insufficient; this signal actively harms 3h consensus.
        # 2026-04-25: claude_fundamental added - 0 accuracy samples at 3h so
        # the accuracy gate defaults to 0.5 and passes the 47% gate. At 1d the
        # fast-blended accuracy is ~40% (correctly gated), but at 3h it escapes.
        # Fundamentals operate on hours/days timescale, not 3h. Sonnet/Opus have
        # 78-83% BUY bias (500-entry audit), so the ungateed BUY vote at 3h is
        # pure noise. Gate at 3h/4h in ranging; let it vote at 12h/1d/3d/5d.
        "3h": frozenset({"fear_greed", "macro_regime", "sentiment", "claude_fundamental"}),
        "4h": frozenset({"fear_greed", "macro_regime", "sentiment", "claude_fundamental"}),
    },
    "trending-up": {
        # BUG-152: SELL-biased signals have 0-11% accuracy in trending-up.
        # Gating at 1d prevents false SELL consensus during breakouts.
        # trend ~0%, ema ~11%, volume_flow ~10%, macro_regime 11.1%, momentum_factors low
        # claude_fundamental 5.9% trending-up (34 samples) - BUG-154
        # 2026-04-09: funding gated at 1d (29.9%), active at 3h (74.2%)
        # 2026-04-13: fear_greed 25.9% at 1d (170 sam) - destructive in ALL regimes at 1d
        "_default": frozenset({
            "trend", "ema", "volume_flow", "macro_regime",
            "momentum_factors", "claude_fundamental",
            "funding", "fear_greed",
        }),
        # mean_reversion 3h_recent=45.5% - gate on short horizons
        # SELL-biased signals work short-term even in uptrends - do NOT gate at 3h
        # 2026-04-13: sentiment 33.8% at 3h (3629 sam) - destructive at 3h in ALL regimes
        "3h": frozenset({"mean_reversion", "sentiment"}),
        "4h": frozenset({"mean_reversion", "sentiment"}),
    },
    "trending-down": {
        # BUG-155: bb 21.7% in trending-down (false reversal signals)
        # BUG-154: claude_fundamental 30.4% in trending-down
        # BUG-156: volume_flow (0%), macro_regime (0%), ema (0%), trend (0%)
        # on MSTR/PLTR in trending-down. These are SELL-biased and catastrophically
        # wrong when the downtrend classification is stale or stocks are recovering.
        # BUG-165: smart_money 10.0% in trending-down (130 samples) - worst signal
        # 2026-04-09: funding gated at 1d, active at 3h
        # 2026-04-13: fear_greed 25.9% at 1d (170 sam) - destructive in ALL regimes at 1d
        "_default": frozenset({
            "bb", "claude_fundamental",
            "volume_flow", "macro_regime", "ema", "trend", "heikin_ashi",
            "smart_money",  # BUG-165: 10.0% accuracy in trending-down
            "funding", "fear_greed",
            # 2026-04-27: sentiment 40.1% at 1d_recent (202 sam), BUY-only bias.
            # Was only gated at 3h/4h; actively harmful at longer horizons too.
            "sentiment",
        }),
        # 3h: trend signals may still work short-term; keep mean_reversion gated
        # 2026-04-13: sentiment 33.8% at 3h (3629 sam) - destructive at 3h in ALL regimes
        "3h": frozenset({"mean_reversion", "bb", "claude_fundamental", "sentiment"}),
        "4h": frozenset({"mean_reversion", "bb", "claude_fundamental", "sentiment"}),
    },
    "high-vol": {
        # 2026-04-09: funding gated at 1d, active at 3h
        # 2026-04-13: fear_greed 25.9% at 1d (170 sam) - destructive in ALL regimes at 1d
        "_default": frozenset({"funding", "fear_greed"}),
        # 2026-04-13: sentiment 33.8% at 3h (3629 sam) - destructive at 3h in ALL regimes
        "3h": frozenset({"sentiment"}),
        "4h": frozenset({"sentiment"}),
    },
    "unknown": {
        "3h": frozenset({"sentiment"}),
        "4h": frozenset({"sentiment"}),
    },
}


def _get_regime_gated(regime: str, horizon: str | None = None) -> frozenset[str]:
    """Get the set of signals to gate for a regime+horizon combination.

    Intentional semantics: horizon-specific override REPLACES `_default`,
    NOT unions with it. This is by design (BUG-149, 2026-03-29):
    `_default` lists signals that are bad at long horizons (1d/3d/5d) in
    a regime, while a horizon override (3h/4h) is the FINER-grained list
    of what should still be gated at that intraday horizon. Example:
    `trend` has 40.7% accuracy at 1d ranging (gate via _default) but
    61.6% at 3h ranging (allow via no-mention in 3h override).

    2026-05-02 audit: 04-24 P0-1 / 04-29 SC-P1-1 / 05-01 P0 (carryover)
    framed this as a "union bug" by analogy to `_get_horizon_disabled_signals`.
    Re-reading the docstring at line 762-767 and the per-signal comments
    in REGIME_GATED_SIGNALS confirms the intent is replace-semantics.
    Verified: funding 74.2% at 3h (in _default for 1d but excluded from
    3h override on purpose), trend 61.6% at 3h (same pattern).
    Finding REJECTED - leaving behavior unchanged. No fix needed.
    """
    regime_dict = REGIME_GATED_SIGNALS.get(regime, {})
    if not regime_dict:
        return frozenset()
    if horizon and horizon in regime_dict:
        return regime_dict[horizon]
    return regime_dict.get("_default", frozenset())

# Horizon-specific signal weight multipliers.
# Signals with >15pp accuracy divergence between horizons get adjusted.
# Updated: 2026-04-27 accuracy audit (3h_recent vs 1d_recent).
HORIZON_SIGNAL_WEIGHTS: dict[str, dict[str, float]] = {
    "3h": {
        "news_event": 1.4,      # 70.0% at 3h_recent (1762 sam)
        "ema": 1.3,             # 62.9% at 3h (vs 48.6% at 1d)
        "ministral": 1.3,       # 62.6% at 3h (vs 42.4% at 1d) - boosted from 1.2
        "qwen3": 1.2,           # 61.8% at 3h
        "trend": 1.2,           # 61.6% at 3h (vs 37.7% at 1d)
        "volatility_sig": 1.2,  # 60.2% at 3h (304 sam)
        "momentum_factors": 1.2, # 60.1% at 3h (vs 35.4% at 1d)
        "momentum": 1.1,        # 56.1% at 3h (378 sam) - NEW 2026-04-27
        "heikin_ashi": 1.1,     # 55.0% at 3h (vs 42.7% at 1d) - NEW 2026-04-27
        "sentiment": 0.4,       # 33.8% at 3h - tightened from 0.5
        "forecast": 0.5,        # 38.3% at 3h
        "bb": 0.6,              # 41.7% at 3h (but 62.5% at 1d)
        "mean_reversion": 0.7,  # 45.5% at 3h (but 51.8% at 1d)
        "volume_flow": 0.7,     # 46.4% at 3h - NEW 2026-04-27
    },
    "4h": {
        "news_event": 1.4,
        "ema": 1.3,
        "ministral": 1.3,
        "qwen3": 1.2,
        "trend": 1.2,
        "volatility_sig": 1.2,
        "momentum_factors": 1.2,
        "momentum": 1.1,
        "heikin_ashi": 1.1,
        "sentiment": 0.4,
        "forecast": 0.5,
        "bb": 0.6,
        "mean_reversion": 0.7,
        "volume_flow": 0.7,
    },
    "1d": {
        "bb": 1.3,              # 62.5% at 1d_recent (120 sam) - boosted from 1.2
        "rsi": 1.1,             # 56.2% at 1d_recent (569 sam) - NEW 2026-04-27
        "credit_spread_risk": 1.1,  # 56.4% at 1d_recent (140 sam), SELL 77.9% - NEW 2026-04-27
        "volume": 1.1,          # 54.7% at 1d_recent (265 sam) - NEW 2026-04-27
        "macd": 1.1,            # 54.8% at 1d_recent (93 sam)
        "mean_reversion": 1.1,  # 51.8% at 1d_recent - reduced from 1.3
        "news_event": 1.4,      # 70.0% at 1d_recent (340 sam)! - was 0.5 (SELL-focused works now)
        "claude_fundamental": 0.5,  # 40.5% at 1d_recent (1178 sam) - NEW 2026-04-27 penalty
        "sentiment": 0.4,       # 40.1% at 1d_recent (202 sam) - NEW 2026-04-27
        "fear_greed": 0.4,      # 25.9% at 1d - still terrible
        "macro_regime": 0.5,    # 36.8% at 1d_recent
        "volatility_sig": 0.5,  # 45.5% at 1d_recent
        "structure": 0.5,       # 33.7% at 1d_recent - tightened from 0.6
        "forecast": 0.5,        # 44.6% at 1d_recent
        "ema": 0.5,             # 48.6% at 1d_recent - tightened from 0.6
        "trend": 0.5,           # 37.7% at 1d_recent - tightened from 0.6
        "heikin_ashi": 0.6,     # 42.7% at 1d_recent - tightened from 0.7
        "momentum_factors": 0.5, # 35.4% at 1d_recent - NEW 2026-04-27
        "volume_flow": 0.5,     # 40.0% at 1d_recent - NEW 2026-04-27
        "crypto_macro": 0.7,    # 46.9% at 1d_recent - NEW 2026-04-27
    },
}

# Activity rate cap: signals with activation rate above this threshold get
# an additional penalty to prevent a single high-activity signal from
# dominating consensus.  Targets volume_flow (83.1% activity, 49.2% accuracy).
_ACTIVITY_RATE_CAP = 0.70
_ACTIVITY_RATE_PENALTY = 0.5

# Dynamic horizon weight computation settings
_DYNAMIC_HORIZON_WEIGHT_TTL = 3600  # 1 hour cache
_DYNAMIC_HORIZON_MIN_SAMPLES = 50   # need enough data per signal per horizon
_DYNAMIC_HORIZON_CLAMP_LOW = 0.4    # minimum multiplier
_DYNAMIC_HORIZON_CLAMP_HIGH = 1.5   # maximum multiplier
_DYNAMIC_HORIZON_DEADBAND = 0.1     # ignore multipliers within ñ10% of 1.0

# Cross-horizon pairs: for a given horizon, which other horizons to compare against
_CROSS_HORIZON_PAIRS = {
    "3h": ["1d"],
    "4h": ["1d"],
    "1d": ["3h"],
}


def _compute_dynamic_horizon_weights(horizon: str) -> dict[str, float]:
    """Compute horizon-specific signal weight multipliers from accuracy cache.

    For each signal, computes the ratio of its accuracy on this horizon vs
    the comparison horizon(s). Signals that perform much better on this
    horizon get boosted; signals that perform much worse get penalized.

    Returns a dict of {signal_name: multiplier} for multipliers outside
    the deadband (i.e., > 1.1 or < 0.9). Falls back to static
    HORIZON_SIGNAL_WEIGHTS if accuracy cache is unavailable.
    """
    try:
        from portfolio.file_utils import load_json
        cache = load_json(DATA_DIR / "accuracy_cache.json")
        if not cache:
            return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})

        # Get recent accuracy for this horizon and comparison horizons
        this_key = f"{horizon}_recent"
        this_data = cache.get(this_key, {})
        if not this_data:
            return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})

        cross_horizons = _CROSS_HORIZON_PAIRS.get(horizon, [])
        if not cross_horizons:
            return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})

        # Gather comparison accuracies (true mean across comparison horizons)
        cross_sum: dict[str, float] = {}
        cross_count: dict[str, int] = {}
        for ch in cross_horizons:
            ch_key = f"{ch}_recent"
            ch_acc = cache.get(ch_key, {})
            for sig, stats in ch_acc.items():
                if stats.get("total", 0) >= _DYNAMIC_HORIZON_MIN_SAMPLES:
                    acc = stats.get("accuracy", 0.5)
                    cross_sum[sig] = cross_sum.get(sig, 0.0) + acc
                    cross_count[sig] = cross_count.get(sig, 0) + 1
        cross_data = {
            sig: cross_sum[sig] / cross_count[sig]
            for sig in cross_sum
            if cross_count.get(sig, 0) > 0
        }

        # Compute multipliers
        weights = {}
        for sig, stats in this_data.items():
            samples = stats.get("total", 0)
            if samples < _DYNAMIC_HORIZON_MIN_SAMPLES:
                continue
            this_acc = stats.get("accuracy", 0.5)
            cross_acc = cross_data.get(sig)
            if cross_acc is None or not (0.01 <= cross_acc <= 1.0):
                continue

            # Ratio of this-horizon accuracy to cross-horizon accuracy
            ratio = this_acc / cross_acc
            # Clamp
            ratio = max(_DYNAMIC_HORIZON_CLAMP_LOW, min(_DYNAMIC_HORIZON_CLAMP_HIGH, ratio))
            # Deadband: only include if meaningfully different from 1.0
            if abs(ratio - 1.0) > _DYNAMIC_HORIZON_DEADBAND:
                weights[sig] = round(ratio, 2)

        return weights if weights else HORIZON_SIGNAL_WEIGHTS.get(horizon, {})
    except Exception:
        logger.debug("Dynamic horizon weights unavailable, using static fallback", exc_info=True)
        return HORIZON_SIGNAL_WEIGHTS.get(horizon, {})


def _get_horizon_weights(horizon: str | None) -> dict[str, float]:
    """Get horizon-specific signal weight multipliers, preferring dynamic computation.

    Uses cached dynamic weights when available, falling back to static dict.
    """
    if not horizon:
        return {}
    cache_key = f"dynamic_horizon_weights_{horizon}"
    # Codex 2026-05-10: _cached returns None on dogpile/timeout/error
    # paths (shared_state.py:88, 109, 123, 126). The previous bare cast
    # silenced the type but left ``signal_name in horizon_mults`` to
    # crash at runtime when None leaked through. Coerce to {} here so
    # the contract - "horizon weights are always a dict" - holds at the
    # boundary where the lie used to live.
    weights = _cached(cache_key, _DYNAMIC_HORIZON_WEIGHT_TTL,
                      lambda: _compute_dynamic_horizon_weights(horizon))
    return cast(dict[str, float], weights) if weights else {}


# Signals that only apply to specific asset classes
_CRYPTO_ONLY_SIGNALS = {"futures_flow", "funding", "crypto_macro", "onchain"}
_METALS_ONLY_SIGNALS = {"metals_cross_asset"}
_NON_STOCK_SIGNALS = {"orderbook_flow"}  # metals + crypto only


def _compute_applicable_count(ticker: str, skip_gpu: bool = False) -> int:
    """Compute total applicable signals for a ticker dynamically.

    Accounts for disabled signals, per-asset-class restrictions,
    and GPU signals skipped outside market hours.
    """
    is_crypto = ticker in CRYPTO_SYMBOLS
    is_metal = ticker in METALS_SYMBOLS
    is_stock = ticker in STOCK_SYMBOLS
    count = 0
    for sig in SIGNAL_NAMES:
        if sig in DISABLED_SIGNALS and (sig, ticker) not in _DISABLED_SIGNAL_OVERRIDES:
            continue
        # Per-ticker blacklist: check _default horizon for signals bad at all horizons
        if sig in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
            continue
        # crypto-only signals (futures_flow, funding, crypto_macro)
        if sig in _CRYPTO_ONLY_SIGNALS and not is_crypto:
            continue
        # metals-only signals (metals_cross_asset)
        if sig in _METALS_ONLY_SIGNALS and not is_metal:
            continue
        # non-stock signals (orderbook_flow - metals + crypto only)
        if sig in _NON_STOCK_SIGNALS and is_stock:
            continue
        # ministral (CryptoTrader-LM) only runs for crypto
        if sig == "ministral" and not is_crypto:
            continue
        # GPU signals skipped for stocks outside market hours
        if skip_gpu and sig in GPU_SIGNALS:
            continue
        count += 1
    return count


_VALID_ACTIONS = frozenset({"BUY", "SELL", "HOLD"})


def _validate_signal_result(result, sig_name=None, max_confidence=1.0):
    """Normalize and validate a signal's return dict.

    Ensures action is a valid string, confidence is a finite float in [0, 1],
    and sub_signals is a dict. Returns a clean dict, always.
    """
    if not result or not isinstance(result, dict):
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}}

    action = result.get("action")
    if action not in _VALID_ACTIONS:
        if sig_name:
            logger.warning("Signal %s returned invalid action=%r, defaulting to HOLD", sig_name, action)
        action = "HOLD"

    conf = result.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    if not np.isfinite(conf):
        if sig_name:
            logger.warning("Signal %s returned non-finite confidence=%r, defaulting to 0.0", sig_name, conf)
        conf = 0.0
    conf = max(0.0, min(max_confidence, conf))

    sub_signals = result.get("sub_signals")
    if not isinstance(sub_signals, dict):
        sub_signals = {}

    return {
        "action": action,
        "confidence": conf,
        "sub_signals": sub_signals,
        "indicators": result.get("indicators") or {},
    }


# Dynamic correlation group computation TTL and thresholds
_DYNAMIC_CORR_TTL = 7200  # 2h cache for dynamic correlation groups
# 2026-04-18: changed from Pearson r > 0.7 to agreement rate > 0.85.
# Pearson on vote encoding (BUY=1, HOLD=0, SELL=-1) was diluted by 70-90%
# HOLD dominance - max observed r=0.538 (ematrend), making the 0.7
# threshold unreachable and dynamic groups always falling back to static.
# Agreement rate only counts pairs where at least one signal voted non-HOLD.
_DYNAMIC_CORR_THRESHOLD = 0.85  # agreement rate threshold for clustering
_DYNAMIC_CORR_MIN_SAMPLES = 30  # minimum signal log entries for reliable correlation
_DYNAMIC_CORR_MIN_PAIRS = 20    # minimum non-HOLD pairs to trust agreement rate


def _compute_agreement_rate(votes_a, votes_b):
    """Compute agreement rate between two signal vote lists.

    Only counts pairs where at least one signal voted non-HOLD.
    Returns (agreement_rate, n_pairs) where n_pairs is the count of
    non-HOLD pairs.
    """
    agree = 0
    total = 0
    for va, vb in zip(votes_a, votes_b):
        if va == 0 and vb == 0:
            continue  # both HOLD - skip
        total += 1
        if va == vb:
            agree += 1
    if total == 0:
        return 0.0, 0
    return agree / total, total


def _compute_dynamic_correlation_groups() -> dict[str, frozenset[str]]:
    """Compute signal correlation groups from recent signal_log data.

    Uses agreement rate (not Pearson correlation) on non-HOLD vote pairs.
    Signals that agree > 85% of the time on non-HOLD votes are clustered.

    Falls back to static CORRELATION_GROUPS if insufficient data.
    """
    try:
        from datetime import datetime, timedelta

        from portfolio.accuracy_stats import load_entries
        entries = load_entries()
        cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        recent = [e for e in entries if e.get("ts", "") >= cutoff]
        if len(recent) < _DYNAMIC_CORR_MIN_SAMPLES:
            return _STATIC_CORRELATION_GROUPS

        # Build signal vote matrix: each row is a (entry, ticker) pair
        vote_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        from portfolio.tickers import SIGNAL_NAMES as _SN
        active_signals = [s for s in _SN if s not in DISABLED_SIGNALS]

        rows = []
        for entry in recent:
            for _tk, tdata in entry.get("tickers", {}).items():
                signals = tdata.get("signals", {})
                row = {s: vote_map.get(signals.get(s, "HOLD"), 0) for s in active_signals}
                rows.append(row)

        if len(rows) < _DYNAMIC_CORR_MIN_SAMPLES:
            return _STATIC_CORRELATION_GROUPS

        df = pd.DataFrame(rows)
        # Drop signals that are always HOLD (no variance)
        df = df.loc[:, df.std() > 0.01]
        if df.shape[1] < 3:
            return _STATIC_CORRELATION_GROUPS

        # Compute pairwise agreement rate (non-HOLD pairs only)
        from collections import defaultdict
        signal_to_group: dict[str, int] = {}
        groups: dict[int, set[str]] = defaultdict(set)
        next_group = 0

        sig_list = list(df.columns)
        for i, s1 in enumerate(sig_list):
            for j in range(i + 1, len(sig_list)):
                s2 = sig_list[j]
                agree_rate, n_pairs = _compute_agreement_rate(
                    df[s1].values, df[s2].values,
                )
                if n_pairs < _DYNAMIC_CORR_MIN_PAIRS:
                    continue
                if agree_rate > _DYNAMIC_CORR_THRESHOLD:
                    g1 = signal_to_group.get(s1)
                    g2 = signal_to_group.get(s2)
                    if g1 is None and g2 is None:
                        gid = next_group
                        next_group += 1
                        groups[gid] = {s1, s2}
                        signal_to_group[s1] = gid
                        signal_to_group[s2] = gid
                    elif g1 is not None and g2 is None:
                        groups[g1].add(s2)
                        signal_to_group[s2] = g1
                    elif g1 is None and g2 is not None:
                        groups[g2].add(s1)
                        signal_to_group[s1] = g2
                    elif g1 != g2:
                        # Merge groups. By elimination of the prior elifs both
                        # g1 and g2 are non-None here, but mypy can't narrow
                        # via ``!=``; assert makes the invariant explicit.
                        assert g1 is not None and g2 is not None
                        merged = groups[g1] | groups[g2]
                        groups[g1] = merged
                        del groups[g2]
                        for s in merged:
                            signal_to_group[s] = g1

        # Convert to named frozensets (only groups with 2+ members)
        result = {}
        for gid, members in groups.items():
            if len(members) >= 2:
                name = f"dynamic_{gid}"
                result[name] = frozenset(members)

        return result if result else _STATIC_CORRELATION_GROUPS

    except Exception:
        logger.debug("Dynamic correlation groups unavailable, using static", exc_info=True)
        return _STATIC_CORRELATION_GROUPS


def _get_correlation_groups() -> dict[str, frozenset[str]]:
    """Get current correlation groups, preferring dynamic over static.

    Codex 2026-05-10: ``_cached`` can return None on dogpile/error; the
    sole caller already does ``... or _STATIC_CORRELATION_GROUPS`` so a
    None leak isn't a runtime crash here, but the bare cast lied about
    the actual return type. Treat empty/None as "no dynamic groups
    available" so the type matches reality.
    """
    groups = _cached("dynamic_corr_groups", _DYNAMIC_CORR_TTL,
                     _compute_dynamic_correlation_groups)
    return cast(dict[str, frozenset[str]], groups) if groups else {}


# Static correlation groups (fallback when dynamic computation unavailable).
# Updated 2026-04-08: empirical audit of 200 recent signal_log entries.
_STATIC_CORRELATION_GROUPS = {
    # 2026-04-14: Measured correlation analysis (300 snapshots, 1308 obs):
    # volatility_sig only weakly correlates with volume (r=0.38). Oscillators
    # moved to trend_direction (0.463 with heikin_ashi, 83.4% agreement).
    # Structure moved to trend_direction (0.608 with trend, 96.5% with macro_regime).
    # RES-2026-04-21: REMOVED volatility_cluster. r=0.38 is too weak for a
    # correlation group. volume (52.1% acc) was unfairly penalized by
    # volatility_sig (46.8% acc). Let both vote independently.
    # 2026-04-30: SPLIT trend_direction mega-cluster (was 9 members, 1.96x weight).
    # At 0.12x per follower, one directional signal family dominated consensus.
    # Split into 3 semantically distinct sub-clusters:
    #   pure_trend (MA-based): trend/ema/heikin_ashi - 85-90% agreement
    #   oscillator_trend: macd/momentum_factors/oscillators - oscillation methods
    #   structural_flow: volume_flow/macro_regime/structure - market structure
    # Each sub-cluster gets independent leader selection, preventing a single
    # broken trend signal from poisoning all 9 members' direction.
    # Previous: 1.0 + 8*0.12 = 1.96x total. Now: 3 * (1.0 + 2*0.20) = 4.20x
    # total, but with 3 independent leaders - better captures disagreement
    # between trend, momentum, and structural signals.
    # 2026-05-07: trend disabled (46.1% 1d, 17880 sam), heikin_ashi borderline (49.3%)
    # but ema is the sole surviving member. Kept as group in case heikin_ashi re-enabled.
    "pure_trend": frozenset({"ema", "heikin_ashi"}),
    # 2026-05-07: macd disabled (44.2% 1d). momentum_factors is the leader.
    "oscillator_trend": frozenset({"momentum_factors", "oscillators"}),
    "structural_flow": frozenset({"volume_flow", "macro_regime", "structure"}),
    # 2026-04-18: Expanded from 36 members. Research (2026-04-17 after-hours)
    # found calendarfear_greed 100% agreement (501 sam), fundingfear_greed
    # 100% (543 sam), news_eventecon_calendar 100% (714 sam). These orphaned
    # signals were voting with full weight despite being completely redundant.
    "macro_external": frozenset({
        "fear_greed", "sentiment", "news_event",
        "calendar", "econ_calendar", "funding",
    }),
    # 2026-04-04: BUG-162 - candlestick-fibonacci correlation 0.708 on BTC.
    # fibonacci disabled 2026-04-29 (43.6%, 17K sam). Group dissolved -
    # candlestick now votes unclustered at full weight.
    # "pattern_based" removed: single-member groups are invalid.
    # 2026-04-26: bb removed from all clusters - now unclustered (full 1.0x weight).
    # BB thrives in ranging (+15.2pp to 69.5% recent) independently of ema/macd
    # (which are regime-gated). Correlation with ema/macd is superficial (BB bands
    # track MA) but BB's edge is overbought/oversold detection. Putting it in
    # trend_direction (0.12x) destroyed its edge; standalone cluster is semantically
    # wrong (1 member). Unclustered = full weight, which matches its value.
    # 2026-04-08: rsi+bb agree 100%, bb+mean_reversion 100%, bb+momentum 98.8%.
    # 2026-04-25: Moved bb to trend_direction, 2026-04-26: removed from all clusters.
    "momentum_cluster": frozenset({"mean_reversion", "rsi", "momentum"}),
    # 2026-04-13: claude_fundamental + crypto_macro agree 92-100%.
    # structure removed (now in trend_direction where correlations are stronger).
    "fundamental_cluster": frozenset({"claude_fundamental", "crypto_macro"}),
    # 2026-04-19: Measured correlation (50 snapshots): credit_spread_risk
    # agrees 100% with macro_regime in ETH/XAU, 100% with news_event in BTC.
    # futures_flow agrees 100% with credit_spread_risk + macro_regime in ETH.
    # Both were orphaned and getting full 1.0x weight despite redundancy.
    # Grouped together rather than in trend_direction to avoid inflating that
    # mega-cluster further (already 9 members at 0.12x).
    # 2026-05-07: futures_flow disabled (38.3% 1d). credit_spread_risk (53.7%)
    # now unclustered - single-member groups are invalid per convention.
    # credit_spread_risk votes at full 1.0x weight independently.
}
# Public alias for backward compatibility (used by tests and reporting)
CORRELATION_GROUPS = _STATIC_CORRELATION_GROUPS
_CORRELATION_PENALTY = 0.3  # secondary signals in a group get 30% of normal weight
# Per-cluster overrides: momentum_cluster signals agree 88-100% of the time.
# 2026-04-25: momentum_cluster now 3 members (bb moved to trend_direction).
# At 0.15x: 1.0 + 2*0.15 = 1.30x effective weight (was 1.45x with 4 members).
_CLUSTER_CORRELATION_PENALTIES: dict[str, float] = {
    "momentum_cluster": 0.15,
    # 2026-04-30: split trend_direction (9 members, 0.12x) into 3 sub-clusters.
    # Each has 3 members at 0.20x: effective weight per cluster = 1.0 + 2*0.20 = 1.40x.
    # Previous total: 1.96x from 1 cluster. New total: 3 * 1.40x = 4.20x BUT with
    # 3 independent leaders - momentum/structural can disagree with pure trend.
    "pure_trend": 0.20,
    "oscillator_trend": 0.20,
    "structural_flow": 0.20,
    # 2026-04-18: macro_external expanded from 36 members. At 0.15x per follower:
    # effective weight = 1.0 + 5*0.15 = 1.75x. Previously 3 members at 0.3x gave
    # 1.0 + 2*0.3 = 1.6x. Slightly higher total accounts for 3 truly independent
    # information sources (sentiment, calendar/econ timing, macro FG) being merged
    # with their highly-correlated variants.
    "macro_external": 0.15,
}

# Meta-clusters: groups of correlation sub-clusters whose LEADERS frequently
# agree (100% cross-cluster leader agreement measured 2026-05-01, 20 snapshots).
# When all meta-cluster leaders vote the same direction, only the highest-accuracy
# leader retains full weight; others get the meta-cluster penalty.
# This prevents the trend mega-view from getting 3.0x effective leader weight
# when pure_trend, oscillator_trend, and structural_flow leaders vote identically.
# When leaders DISAGREE, no penalty is applied - that's informative diversity.
_META_CLUSTER_GROUPS: dict[str, list[str]] = {
    "trend_mega": ["pure_trend", "oscillator_trend", "structural_flow"],
}
_META_CLUSTER_PENALTY = 0.35  # 2nd/3rd agreeing leaders get 35% weight


def _safe_accuracy(value, default):
    """Coerce an accuracy value to a clean float, mapping None/NaN/inf to `default`.

    2026-04-17 (P1-C): the live consensus path previously crashed with
    TypeError when `accuracy_data[sig]` held explicit None (e.g., from a
    half-written cache), and with a silent fall-through-as-valid when it
    held NaN (every comparison with NaN is False). This helper normalizes.
    """
    import math
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _safe_sample_count(value):
    """Coerce a sample count to a non-negative int; None/NaN/negative -> 0."""
    import math
    if value is None:
        return 0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    if math.isnan(f) or math.isinf(f) or f < 0:
        return 0
    return int(f)


def _count_active_voters_at_gate(votes, accuracy_data, excluded, group_gated,
                                  base_gate, relaxation):
    """Count how many signals would pass gating at gate=(base_gate - relaxation).

    Counts only voters that survive the full gate cascade:
      1) excluded (top-N)
      2) group-gated (correlation leader below group-leader gate)
      3) accuracy gate at (base - relaxation), tiered for high-sample signals
      4) directional gate (unchanged by relaxation)

    Returns int - the number of signals still voting BUY/SELL.
    """
    gate_val = base_gate - relaxation
    # SC-P1-2 (2026-05-02 adversarial follow-ups): high-sample tier is NOT
    # relaxed. A signal with 10K+ samples at sub-50% accuracy has measurable
    # negative edge - circuit-breaker relaxation must not promote it back to
    # voting. Standard tier (under 10K samples) still relaxes so borderline
    # newer signals can be rescued during regime transitions. Must mirror
    # the same logic in `_weighted_consensus` (line ~2068).
    high_gate_val = _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD
    active = 0
    for signal_name, vote in votes.items():
        if vote == "HOLD":
            continue
        if signal_name in excluded:
            continue
        if signal_name in group_gated:
            continue
        stats = accuracy_data.get(signal_name) or {}
        # P1-C (2026-04-17 adversarial review): coerce None / NaN values to
        # safe defaults. The live path previously crashed with TypeError when
        # `accuracy_data[sig]` contained explicit None values (e.g., from a
        # half-written cache). Replay had `except TypeError` but live didn't.
        acc = _safe_accuracy(stats.get("accuracy"), default=0.5)
        samples = _safe_sample_count(stats.get("total"))
        effective_gate = gate_val
        if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN:
            effective_gate = max(gate_val, high_gate_val)
        if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < effective_gate:
            continue
        # Directional gate is not relaxed by the circuit breaker - those gates
        # catch signals that are actively wrong in one direction.
        if vote == "BUY":
            dir_acc = _safe_accuracy(stats.get("buy_accuracy"), default=acc)
            dir_n = _safe_sample_count(stats.get("total_buy"))
        else:
            dir_acc = _safe_accuracy(stats.get("sell_accuracy"), default=acc)
            dir_n = _safe_sample_count(stats.get("total_sell"))
        if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
            continue
        active += 1
    return active


def _normalize_regime(regime):
    """P2-D (2026-04-17): normalize regime strings to a canonical lowercase form.

    Protects against case/typo variants ("TRENDING-UP", " trending-up ",
    "trending_up") that would otherwise silently fall through to the
    strictest-quorum default. Returns None unchanged.
    """
    if regime is None:
        return None
    if not isinstance(regime, str):
        return regime  # Let downstream default handle non-strings.
    normalized = regime.strip().lower().replace("_", "-")
    # Common alias fixups.
    if normalized in ("trendingup", "trending"):
        normalized = "trending-up"
    elif normalized == "trendingdown":
        normalized = "trending-down"
    elif normalized in ("highvol", "high-volatility", "high_vol"):
        normalized = "high-vol"
    return normalized


def _dynamic_min_voters_for_regime(regime):
    """Regime-dependent final quorum. Single source of truth - called by both
    the circuit breaker and apply_confidence_penalties.

    This is the minimum voter count the OUTER consensus path requires before
    emitting a non-HOLD action. The circuit breaker uses it to size its
    recovery floor so relaxation is only engaged when it could reach the
    regime's actual quorum.

    2026-04-17 (P2-C/P2-D): de-duplicated. apply_confidence_penalties
    previously had an inline copy at line ~1623 that had to stay in lockstep
    manually - now it calls this helper. Also accepts case/typo-variant
    regime strings via _normalize_regime.
    """
    canonical = _normalize_regime(regime)
    if canonical in ("trending-up", "trending-down"):
        return 3
    if canonical == "high-vol":
        return 4
    return 5  # ranging, unknown, None


def _compute_gate_relaxation(votes, accuracy_data, excluded, group_gated, base_gate,
                              regime=None):
    """Compute circuit-breaker relaxation to preserve voter diversity.

    Progressively tests relaxation values 0, step, 2*step, ..., up to
    _GATE_RELAXATION_MAX. Returns the smallest relaxation that yields at
    least _MIN_ACTIVE_VOTERS_SOFT active voters.

    Decision tree:
      - baseline >= floor                    -> 0.0 (no relaxation needed)
      - best_possible <= baseline            -> 0.0 (relaxation doesn't help;
                                                either a low-signal scenario
                                                or a genuine regime break
                                                where remaining signals are
                                                below even the 41% relaxed
                                                gate - letting them vote
                                                would be wrong)
      - best_possible >= floor               -> smallest step that meets floor
      - baseline < best_possible < floor     -> _GATE_RELAXATION_MAX (partial
                                                recovery: a single
                                                irrecoverable outlier must
                                                not veto relaxation for the
                                                rest - Codex P2 fix)

    Uses `_count_active_voters_at_gate` which applies directional gating,
    so signals gated on BUY-accuracy=30% don't inflate the decision.

    Returns float - relaxation in absolute accuracy points (e.g., 0.02).
    """
    # Defensive: caller may pass None for either set (older paths or a future
    # refactor). Treat as empty to avoid `in None` TypeErrors in a hot path.
    excluded = excluded or set()
    group_gated = group_gated or set()

    # Three guards, in increasing strictness, all applied:
    #
    #   Guard A (raw vs regime quorum):
    #     Matches downstream's `apply_confidence_penalties` which checks
    #     `extra_info["_voters"]` against dynamic_min. `_voters` is raw
    #     non-HOLD count post-regime, pre top-N/group-gate, so this check
    #     must NOT subtract `excluded` or `group_gated`.
    #
    #   Guard B (post-exclusion slate viability):
    #     Downstream's raw `_voters` doesn't account for top-N or
    #     correlation-group exclusions. If the POST-exclusion slate is
    #     below MIN_VOTERS_BASE (3) - the floor across all asset classes -
    #     a relaxed consensus would be built from a too-thin slate even
    #     though downstream would accept the raw count. Codex round 9
    #     (2026-04-17) caught this with a 3-signal correlation cluster
    #     gated out, leaving only 2 voters to drive consensus.
    #
    #   Guard C (lone-signal escape):
    #     Even with a large post-exclusion slate, directional gating can
    #     leave a single accuracy-passing signal. `best_possible >= 2`
    #     catches this case.
    min_regime_quorum = _dynamic_min_voters_for_regime(regime)
    raw_candidates = sum(1 for v in votes.values() if v != "HOLD")
    if raw_candidates < min_regime_quorum:
        return 0.0

    # P2-F (2026-04-17): derived from MIN_VOTERS_CRYPTO/STOCK rather than
    # hardcoded. If the base quorum changes, this follows automatically.
    post_exclusion_candidates = sum(
        1 for sn, v in votes.items()
        if v != "HOLD" and sn not in excluded and sn not in group_gated
    )
    if post_exclusion_candidates < _POST_EXCLUSION_MIN:
        return 0.0

    baseline = _count_active_voters_at_gate(
        votes, accuracy_data, excluded, group_gated, base_gate, 0.0,
    )
    if baseline >= _MIN_ACTIVE_VOTERS_SOFT:
        return 0.0

    best_possible = _count_active_voters_at_gate(
        votes, accuracy_data, excluded, group_gated,
        base_gate, _GATE_RELAXATION_MAX,
    )

    # Lone-signal escape guard. Even when raw candidates meet the downstream
    # quorum, directional gating can leave a thin set of recoverable voters.
    # P2-A (2026-04-17): raised from 2 to MIN_VOTERS_BASE (3). A 2-voter
    # "consensus" is still exposure-worthy in trending markets where
    # dynamic_min=3 - but any relaxation that only recovers 2 voters from a
    # large slate is catching signals that the downstream quorum would
    # accept as a weak consensus. Require at least as many as the base
    # MIN_VOTERS_* to avoid creating "relaxed" sub-quorum consensuses.
    if best_possible < _LONE_SIGNAL_FLOOR:
        return 0.0

    # Regime break: relaxation recovers nothing beyond baseline. Keep the
    # strict gate so the event shows up in logs rather than silently opening
    # to sub-41% signals.
    if best_possible <= baseline:
        return 0.0

    # Integer steps up to and including max - use int steps to avoid float drift.
    n_steps = int(round(_GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP))
    for i in range(1, n_steps + 1):
        candidate_rel = round(i * _GATE_RELAXATION_STEP, 6)
        active = _count_active_voters_at_gate(
            votes, accuracy_data, excluded, group_gated, base_gate, candidate_rel,
        )
        if active >= _MIN_ACTIVE_VOTERS_SOFT:
            return candidate_rel
    # Partial-recovery case (Codex P2 fix): best_possible > baseline but
    # still < floor. A single irrecoverable outlier shouldn't veto recovery
    # of the recoverable majority - apply max relaxation to get as many
    # voters back as possible. Logs still carry the relaxation value so
    # operators can distinguish this from a clean relaxation-to-floor.
    return _GATE_RELAXATION_MAX


# ---------------------------------------------------------------------------
# IC-based weight multiplier (2026-04-18)
# ---------------------------------------------------------------------------

def _compute_ic_mult(ic: float, icir: float, samples: int) -> float:
    """Compute IC-based weight multiplier for a signal.

    Returns a multiplicative adjustment based on the signal's Information
    Coefficient:
    - IC > 0 with stable ICIR  boost (catches big moves)
    - IC ÷ 0 with many samples  slight penalty (phantom performer)
    - IC < 0 with stable ICIR  penalty (contrarian, accuracy gate handles)
    - Insufficient data or unstable  1.0 (no adjustment)

    Clamped to [_IC_MULT_FLOOR, _IC_MULT_CAP].
    """
    if samples < _IC_MIN_SAMPLES:
        return 1.0
    # Zero-IC penalty for phantom performers: signals with many samples but
    # no return-magnitude predictive power (e.g., calendar, econ_calendar).
    if abs(ic) < 0.01 and samples >= _IC_ZERO_MIN_SAMPLES:
        return _IC_ZERO_PENALTY
    if abs(icir) < _IC_STABILITY_MIN:
        return 1.0
    raw = 1.0 + _IC_ALPHA * ic
    return max(_IC_MULT_FLOOR, min(_IC_MULT_CAP, raw))


# IC data cache: reuse ic_computation.py infrastructure with in-memory TTL.
_ic_data_cache: dict = {}
_ic_data_lock = threading.Lock()


def _get_ic_data(horizon: str) -> dict | None:
    """Load IC data for the given horizon, computing if cache is stale.

    Returns the full cache dict {"global": {...}, "per_ticker": {...}}
    or None if IC data is unavailable.
    """
    now = time.time()
    with _ic_data_lock:
        cached = _ic_data_cache.get(horizon)
        if cached and now - cached.get("_loaded_at", 0) < _IC_DATA_TTL:
            return cast(dict[Any, Any], cached)

    try:
        from portfolio.ic_computation import compute_and_cache_ic, load_cached_ic
        cache = load_cached_ic(horizon)
        if cache is None:
            cache = compute_and_cache_ic(horizon)
        if cache:
            cache["_loaded_at"] = now
            with _ic_data_lock:
                _ic_data_cache[horizon] = cache
            return cast(dict[Any, Any], cache)
    except Exception:
        logger.debug("IC data unavailable for %s", horizon, exc_info=True)
    return None


_macro_window_cache: dict = {"value": False, "ts": 0.0}
_macro_window_cache_lock = threading.Lock()
_macro_window_last_state: dict = {"active": None}  # transition logger


def _is_macro_window_cached(now_ts: float | None = None) -> bool:
    """Return whether we're inside a macro event window, with TTL caching.

    The underlying ``portfolio.econ_dates.is_macro_window`` iterates
    ``ECON_EVENTS`` linearly. That's cheap, but called per signal per
    ticker per cycle becomes wasteful. Cache the result for
    ``_MACRO_WINDOW_CACHE_TTL_S`` (5 minutes by default) - events have
    hourly cadence at fastest, so 5min staleness is acceptable.

    Logs once per state transition so the operational log shows when
    we entered/exited a macro window without spamming every cycle.
    """
    import time as _time
    if now_ts is None:
        now_ts = _time.time()
    with _macro_window_cache_lock:
        if now_ts - _macro_window_cache["ts"] < _MACRO_WINDOW_CACHE_TTL_S:
            return bool(_macro_window_cache["value"])
        try:
            from portfolio.econ_dates import is_macro_window
            active = bool(is_macro_window())
        except Exception as e:
            logger.warning("macro window detection failed (treating as inactive): %s", e)
            active = False
        _macro_window_cache["value"] = active
        _macro_window_cache["ts"] = now_ts
        last = _macro_window_last_state["active"]
        if last is None:
            _macro_window_last_state["active"] = active
        elif last != active:
            logger.info(
                "macro_window state transition: %s -> %s",
                "ACTIVE" if last else "inactive",
                "ACTIVE" if active else "inactive",
            )
            _macro_window_last_state["active"] = active
        return active


def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None,
                        accuracy_gate=None, max_signals=None, horizon=None,
                        regime_gated_override=None, ticker=None,
                        soft_confidences=None):
    """Compute weighted consensus using accuracy, IC, regime, and activation frequency.

    Weight per signal = accuracy_weight * ic_mult * regime_mult * normalized_weight
                        * horizon_mult * activity_cap
    where normalized_weight = rarity_bonus * bias_penalty (from activation rates).
    Rare, balanced signals get more weight; noisy/biased signals get less.

    Signals below the accuracy gate (with sufficient samples) are force-skipped -
    they are noise, not useful contrarian indicators.

    Regime gating: signals in REGIME_GATED_SIGNALS for the current regime are
    forced to HOLD before vote processing - they produce negative alpha.

    Correlation deduplication: within defined correlation groups, only the
    highest-accuracy signal gets full weight. Others get 0.3x penalty.

    Horizon-specific weights: signals with divergent accuracy across horizons
    get boosted or penalized via HORIZON_SIGNAL_WEIGHTS.

    Activity rate cap: signals with >70% activation rate get 0.5x penalty
    to prevent a single high-activity signal from dominating consensus.

    Top-N gate: when max_signals is set, only the top max_signals non-HOLD
    signals (ranked by accuracy) participate in the consensus. This focuses
    the vote on the best performers and ignores marginal contributors.

    2026-05-11 (Codex Fix B) - soft-confidence dampening:
    The Stage 2 dead-zone helpers (EMA / BB / MACD) emit *weak* directional
    votes when the strong path would HOLD, and stash a small per-vote
    confidence (0.15-0.20) into extra_info under the keys
    "_soft_conf_ema" / "_soft_conf_bb" / "_soft_conf_macd". Without
    propagation, _weighted_consensus treated those soft votes as full-
    strength votes (just direction x accuracy weight), so an all-soft
    slate could produce full directional confidence - defeating the
    "weak weight" contract. We now scale each soft vote's contribution
    by its soft_conf, so e.g. 3 x 0.18 ÷ 0.54 < 1.0 (a single strong
    vote). Pass the soft_confidences dict to opt in; strong votes (no
    key present) keep their original weight x accuracy x regime mult.
    """
    soft_confidences = soft_confidences or {}
    gate = accuracy_gate if accuracy_gate is not None else ACCURACY_GATE_THRESHOLD
    buy_weight = 0.0
    sell_weight = 0.0
    gated_signals = []
    regime_mults = REGIME_WEIGHTS.get(regime, {})
    activation_rates = activation_rates or {}
    horizon_mults = _get_horizon_weights(horizon)

    # Codex round 10/11/12 (2026-04-17 follow-up): deep-sanitize accuracy_data
    # at function entry.
    #   Round 10: coerced non-dict container values to {}.
    #   Round 11: found dict values with poisoned numeric fields still
    #             crashed. Added per-field coercion.
    #   Round 12: coerce-with-0.5-default silently promoted partially-
    #             written cache rows ({"accuracy": null, "total": 200}) into
    #             mature 50% signals that cleared the min-samples gate.
    #             Now: if a numeric field is poisoned, DROP that field so
    #             downstream `.get(..., default)` falls back cleanly. A row
    #             whose overall accuracy is poisoned but total=200 becomes
    #             {"total": 200} - the gate sees no accuracy, the downstream
    #             code default to the safe fallback. The row no longer
    #             masquerades as a 50%-accurate mature signal.
    # Codex round 13 (2026-04-17): a poisoned accuracy must invalidate its
    # PAIRED sample count too. Otherwise `{"accuracy": None, "total": 200}`
    # becomes `{"total": 200}` which downstream still reads as a mature
    # 50% signal (accuracy defaults to 0.5, samples=200 clears the gate).
    # Drop-together semantics: overall acc poisoned -> drop (accuracy, total);
    # buy_accuracy poisoned -> drop (buy_accuracy, total_buy); likewise for
    # sell. Fields whose pair is clean but themselves clean pass through.
    import math as _math

    def _coerce_sample_count(val):
        """Return int >= 0, or None if val is missing/poisoned/invalid."""
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        if _math.isnan(f) or _math.isinf(f) or f < 0:
            return None
        return int(f)

    _PAIRED = (
        ("accuracy", "total"),
        ("buy_accuracy", "total_buy"),
        ("sell_accuracy", "total_sell"),
    )
    if accuracy_data:
        _sanitized: dict[Any, dict[str, Any]] = {}
        for _k, _v in accuracy_data.items():
            if not isinstance(_v, dict):
                _sanitized[_k] = {}
                continue
            _clean = dict(_v)  # start from a copy, then prune.
            for _acc_key, _cnt_key in _PAIRED:
                _acc_has = _acc_key in _clean
                _cnt_has = _cnt_key in _clean
                if _acc_has:
                    _clean_acc = _safe_accuracy(_clean.get(_acc_key), default=None)
                else:
                    _clean_acc = None
                if _cnt_has:
                    _clean_cnt = _coerce_sample_count(_clean.get(_cnt_key))
                else:
                    _clean_cnt = None
                # Decide whether to keep each field:
                #   Both clean      -> keep both.
                #   Only acc clean  -> keep acc; drop cnt (if it was present-and-poisoned).
                #   Only cnt clean  -> drop BOTH (count without trustworthy accuracy
                #                      must not promote the row to a mature signal).
                #   Neither clean   -> drop both.
                if _clean_acc is not None and _clean_cnt is not None:
                    _clean[_acc_key] = _clean_acc
                    _clean[_cnt_key] = _clean_cnt
                elif _clean_acc is not None and not _cnt_has:
                    # Accuracy present (clean), count field absent - keep acc.
                    _clean[_acc_key] = _clean_acc
                else:
                    # Poisoned accuracy OR poisoned count: drop both so the
                    # row doesn't masquerade as a mature signal. Downstream
                    # .get() calls then use their safe defaults.
                    _clean.pop(_acc_key, None)
                    _clean.pop(_cnt_key, None)
            _sanitized[_k] = _clean
        accuracy_data = _sanitized
    else:
        accuracy_data = {}

    # Regime gating: force-HOLD signals that produce negative alpha in this regime.
    # BUG-149: now horizon-aware - e.g., trend works at 3h in ranging (61.6%)
    # SC-I-001: when caller provides regime_gated_override (with BUG-158 per-ticker
    # exemptions already applied), use it instead of recomputing from scratch.
    regime_gated = regime_gated_override if regime_gated_override is not None else _get_regime_gated(regime, horizon)
    votes = {k: ("HOLD" if k in regime_gated else v) for k, v in votes.items()}

    # Horizon-specific per-ticker blacklist (2026-04-16, Batch 4). Extends the
    # compute-time _default blacklist with horizon-specific entries. Compute time
    # can't see horizon (one vote reused across 3h/4h/12h/1d/3d/5d/10d consensus),
    # so per-horizon gating must happen here.
    horizon_disabled = _get_horizon_disabled_signals(ticker, horizon)
    if horizon_disabled:
        votes = {k: ("HOLD" if k in horizon_disabled else v) for k, v in votes.items()}

    # Macro-window force-HOLD pre-pass (2026-04-28). When a high-impact
    # event is within ~24h past or ~72h future, force-HOLD the signals
    # whose lag/bias makes them dominantly wrong in news-driven regimes.
    # The downweight branch for the other macro-fragile signals lives in
    # the weight loop below so it composes with regime/horizon multipliers.
    macro_active = _is_macro_window_cached()
    if macro_active and MACRO_WINDOW_FORCE_HOLD_SIGNALS:
        votes = {
            k: ("HOLD" if k in MACRO_WINDOW_FORCE_HOLD_SIGNALS else v)
            for k, v in votes.items()
        }

    # Top-N gate: only let the top max_signals (by accuracy) participate.
    # Codex round 2 P2 (2026-04-28): rank with macro-adjusted accuracy so
    # downweighted signals lose Top-N slots to healthier peers during a
    # macro window. Without this, sentiment can keep its slot at full
    # raw accuracy and exclude a peer that would have voted more reliably.
    def _topn_accuracy_key(s: str) -> float:
        base = float(accuracy_data.get(s, {}).get("accuracy", 0.5))
        if macro_active and s in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
            base *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
        return base

    active_votes = {k: v for k, v in votes.items() if v != "HOLD"}
    if max_signals and len(active_votes) > max_signals:
        ranked = sorted(
            active_votes.keys(),
            key=_topn_accuracy_key,
            reverse=True,
        )
        excluded = set(ranked[max_signals:])
    else:
        excluded = set()

    # Pre-compute which signal is the "leader" (highest accuracy) in each
    # correlation group, considering only signals that are actively voting.
    # Prefer dynamic groups (from signal_log correlations) over static.
    active_non_hold = {s for s, v in votes.items() if v != "HOLD"}
    _active_corr_groups = _get_correlation_groups() or _STATIC_CORRELATION_GROUPS

    # Codex P2 (2026-04-28): apply the macro-window downweight to the
    # leader-selection key BEFORE picking the leader. Otherwise sentiment
    # (lifetime ~70% acc) stays leader of macro_external during a macro
    # window - and the 0.15x follower penalty pushes healthier peers
    # below sentiment's already-halved weight, making the overlay
    # actively reinforce the wrong signal.
    def _leader_accuracy_key(s: str) -> float:
        base = float(accuracy_data.get(s, {}).get("accuracy", 0.5))
        if macro_active and s in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
            base *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
        return base

    group_leaders = {}
    for group_name, group_sigs in _active_corr_groups.items():
        active_in_group = active_non_hold & group_sigs
        if len(active_in_group) <= 1:
            continue
        best_sig = max(active_in_group, key=_leader_accuracy_key)
        group_leaders[group_name] = best_sig

    # Correlation group leader gating: when the best signal in a group has
    # accuracy below threshold (with enough samples), gate the ENTIRE group.
    # Prevents the "least bad" broken signal from voting.
    # 2026-03-31: macro_external group (fear_greed 25.9%, sentiment 46.8%,
    # news_event 29.5%) - even the leader is near noise.
    # 2026-04-06: Lowered from 0.47  0.46 to catch borderline cases where
    # sentiment (blended ~46.4%) barely escapes as group leader.
    _GROUP_LEADER_GATE_THRESHOLD = 0.46
    group_gated_signals: set[str] = set()
    for group_name, group_sigs in _active_corr_groups.items():
        leader = group_leaders.get(group_name)
        if leader:
            leader_stats = accuracy_data.get(leader, {})
            leader_acc = leader_stats.get("accuracy", 0.5)
            leader_samples = leader_stats.get("total", 0)
            if leader_samples >= ACCURACY_GATE_MIN_SAMPLES and leader_acc < _GROUP_LEADER_GATE_THRESHOLD:
                group_gated_signals.update(group_sigs & active_non_hold)
                logger.debug(
                    "Correlation group %s gated: leader %s at %.1f%% < %.0f%% threshold",
                    group_name, leader, leader_acc * 100, _GROUP_LEADER_GATE_THRESHOLD * 100,
                )

    # Build a mapping of signal  correlation penalty (per-cluster override).
    # When a signal is in multiple groups, use the harshest (lowest) penalty.
    penalized_signals: dict[str, float] = {}
    for group_name, group_sigs in _active_corr_groups.items():
        leader = group_leaders.get(group_name)
        if leader:
            penalty = _CLUSTER_CORRELATION_PENALTIES.get(group_name, _CORRELATION_PENALTY)
            for s in group_sigs:
                if s != leader and s in active_non_hold:
                    penalized_signals[s] = min(penalized_signals.get(s, 1.0), penalty)

    # Meta-cluster deduplication (2026-05-01): when leaders from related
    # sub-clusters agree on direction, apply penalty to redundant leaders.
    # Prevents the trend mega-view from getting 3.0x effective leader weight
    # when pure_trend/oscillator_trend/structural_flow leaders vote identically.
    for meta_name, sub_clusters in _META_CLUSTER_GROUPS.items():
        meta_leaders: dict[str, str] = {}
        for sc_name in sub_clusters:
            leader = group_leaders.get(sc_name)
            if leader and leader in active_non_hold:
                meta_leaders[sc_name] = leader
        if len(meta_leaders) < 2:
            continue
        # Check if all leaders agree on direction
        leader_directions = {sc: votes.get(ldr, "HOLD")
                            for sc, ldr in meta_leaders.items()}
        active_dirs = set(leader_directions.values()) - {"HOLD"}
        if len(active_dirs) != 1:
            continue  # Leaders disagree - informative diversity, no penalty
        # All leaders agree: keep best-accuracy leader, penalize others
        best_sc = max(meta_leaders,
                      key=lambda sc: _leader_accuracy_key(meta_leaders[sc]))
        for sc_name, leader in meta_leaders.items():
            if sc_name != best_sc:
                current = penalized_signals.get(leader, 1.0)
                penalized_signals[leader] = min(current, _META_CLUSTER_PENALTY)
                logger.debug(
                    "Meta-cluster %s: %s leader %s agrees with %s leader %s "
                    "- penalized to %.0f%%",
                    meta_name, sc_name, leader, best_sc,
                    meta_leaders[best_sc], _META_CLUSTER_PENALTY * 100,
                )

    # Crisis mode detection: when multiple macro-external signals have degraded
    # accuracy, the market is in an abnormal regime (war, systemic crisis) where
    # trend-following breaks and mean-reversion becomes more reliable.
    #
    # 2026-04-19: Made crisis response conditional on trend signal performance.
    # When macro signals are broken but trend signals have >55% accuracy, the
    # crisis is in the macro indicators, not in the trend - penalizing trend
    # signals that are winning is actively harmful (observed: trend 61.6%,
    # EMA 62.9% being penalized 0.6x while crisis mode was active).
    _MACRO_CRISIS_SIGNALS = {"fear_greed", "macro_regime", "structure", "news_event", "sentiment"}
    broken_count = sum(
        1 for s in _MACRO_CRISIS_SIGNALS
        if accuracy_data.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
        and accuracy_data.get(s, {}).get("accuracy", 0.5) < _CRISIS_THRESHOLD
    )
    crisis_mode = broken_count >= _CRISIS_MIN_BROKEN

    _TREND_SIGNALS = {"ema", "trend", "heikin_ashi", "volume_flow"}
    _MR_SIGNALS = {"mean_reversion", "calendar"}

    # Check if trend signals are actually underperforming before penalizing.
    # If avg trend accuracy > 55%, trend is capturing edge despite macro chaos.
    _CRISIS_TREND_ACCURACY_FLOOR = 0.55
    crisis_penalize_trend = False
    if crisis_mode:
        trend_accs = [
            accuracy_data.get(s, {}).get("accuracy", 0.5)
            for s in _TREND_SIGNALS
            if accuracy_data.get(s, {}).get("total", 0) >= ACCURACY_GATE_MIN_SAMPLES
        ]
        avg_trend_acc = sum(trend_accs) / len(trend_accs) if trend_accs else 0.5
        crisis_penalize_trend = avg_trend_acc < _CRISIS_TREND_ACCURACY_FLOOR
        if crisis_penalize_trend:
            logger.info(
                "Crisis mode active (full): %d/%d macro signals broken, "
                "trend avg %.1f%% < %.0f%% floor - penalizing trend, boosting MR",
                broken_count, len(_MACRO_CRISIS_SIGNALS),
                avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
            )
        else:
            logger.info(
                "Crisis mode active (partial): %d/%d macro signals broken, but "
                "trend avg %.1f%% >= %.0f%% floor - NOT penalizing trend signals",
                broken_count, len(_MACRO_CRISIS_SIGNALS),
                avg_trend_acc * 100, _CRISIS_TREND_ACCURACY_FLOOR * 100,
            )

    # Voter-count circuit breaker (Batch 2 of 2026-04-16 accuracy gating reconfig).
    # Only the overall accuracy gate is relaxable - directional and correlation
    # gates still fire. Prevents regime-transition over-gating that silenced
    # ~8 voters in W15/W16.
    relaxation = _compute_gate_relaxation(
        votes=votes,
        accuracy_data=accuracy_data,
        excluded=excluded,
        group_gated=group_gated_signals,
        base_gate=gate,
        regime=regime,
    )
    if relaxation > 0:
        logger.debug(
            "Circuit breaker: relaxing accuracy gate by %.0fpp "
            "(base=%.2f -> effective=%.2f) to preserve voter diversity",
            relaxation * 100, gate, gate - relaxation,
        )

    # IC-based weight multiplier (2026-04-18): load IC data once per consensus
    # call. Returns {"global": {sig: {ic, icir, samples}}, "per_ticker": {...}}
    # or None if IC computation is unavailable.
    ic_cache = _get_ic_data(horizon) if horizon else None
    ic_global = ic_cache.get("global", {}) if ic_cache else {}
    ic_per_ticker = ic_cache.get("per_ticker", {}) if ic_cache else {}

    for signal_name, vote in votes.items():
        # P1-1 (2026-05-02 adversarial follow-ups): defensive - initialize
        # _rescued at the TOP of every iteration so a future contributor who
        # adds a third branch to the gate-check below cannot leak a stale
        # True from a prior iteration into line 2123 (`if _rescued: weight
        # *= _DIRECTIONAL_RESCUE_WEIGHT_PENALTY`). Today both arms of the
        # if/else at line 2072 set _rescued, so the bug doesn't manifest in
        # production - but the structural guarantee is now hardcoded.
        _rescued = False
        if vote == "HOLD":
            continue
        if signal_name in excluded:
            continue
        # Correlation group leader gating: entire group silenced
        if signal_name in group_gated_signals:
            gated_signals.append(signal_name)
            continue
        stats = accuracy_data.get(signal_name, {})
        acc = stats.get("accuracy", 0.5)
        samples = stats.get("total", 0)
        # Accuracy gate: skip signals that are below threshold with enough data.
        # Tiered: established signals (10000+ samples) use a tighter 50% gate;
        # newer signals use the standard 47% gate.
        # SC-P1-2 (2026-05-02 adversarial follow-ups): the high-sample tier
        # (10K+ samples, 0.50 gate) is NOT relaxed. A signal with 10K+ samples
        # at sub-50% accuracy has statistically demonstrated negative edge -
        # circuit-breaker relaxation must not let it back in. The standard
        # tier still relaxes uniformly so newer borderline signals can be
        # rescued during regime transitions.
        effective_gate = gate - relaxation
        if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN:
            effective_gate = max(
                gate - relaxation,
                _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD,
            )
        if samples >= ACCURACY_GATE_MIN_SAMPLES and acc < effective_gate:
            # Directional rescue: before gating, check if the vote direction
            # has strong enough accuracy to justify a reduced-weight vote.
            if vote == "BUY":
                rescue_acc = stats.get("buy_accuracy", 0.0)
                rescue_n = stats.get("total_buy", 0)
            else:
                rescue_acc = stats.get("sell_accuracy", 0.0)
                rescue_n = stats.get("total_sell", 0)
            if (rescue_n >= _DIRECTIONAL_RESCUE_MIN_SAMPLES
                    and rescue_acc >= _DIRECTIONAL_RESCUE_THRESHOLD):
                logger.debug(
                    "Directional rescue: %s overall=%.1f%% (gated) but "
                    "%s=%.1f%% (%d sam) - rescued at %.0f%% weight",
                    signal_name, acc * 100, vote,
                    rescue_acc * 100, rescue_n,
                    _DIRECTIONAL_RESCUE_WEIGHT_PENALTY * 100,
                )
                # Fall through to weighting with rescue penalty applied later
                _rescued = True
            else:
                gated_signals.append(signal_name)
                continue
        else:
            _rescued = False
        # Directional accuracy gate: gate individual BUY/SELL direction when
        # direction-specific accuracy is very poor, even if overall accuracy passes.
        # E.g., qwen3 overall=59.8% passes, but BUY=30.0%  gate BUY only.
        if vote == "BUY":
            dir_acc = stats.get("buy_accuracy", acc)
            dir_n = stats.get("total_buy", 0)
        else:
            dir_acc = stats.get("sell_accuracy", acc)
            dir_n = stats.get("total_sell", 0)
        if dir_n >= _DIRECTIONAL_GATE_MIN_SAMPLES and dir_acc < _DIRECTIONAL_GATE_THRESHOLD:
            gated_signals.append(f"{signal_name}_{vote}")
            continue
        # BUG-182: Use direction-specific accuracy as weight when available.
        # A signal with overall 60% accuracy may be 30% for BUY and 75% for SELL.
        # Using overall accuracy over-weights the weak direction.
        _DIR_WEIGHT_MIN_SAMPLES = 20
        if vote == "BUY" and stats.get("total_buy", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
            weight = stats.get("buy_accuracy", acc)  # BUG-185: .get() for cache safety
        elif vote == "SELL" and stats.get("total_sell", 0) >= _DIR_WEIGHT_MIN_SAMPLES:
            weight = stats.get("sell_accuracy", acc)  # BUG-185: .get() for cache safety
        elif samples >= 20:
            weight = acc
        else:
            weight = 0.5
        # Apply directional rescue penalty: rescued signals contribute at
        # reduced weight since their overall accuracy failed the gate.
        if _rescued:
            weight *= _DIRECTIONAL_RESCUE_WEIGHT_PENALTY
        # IC-based weight adjustment: boost signals with high return-magnitude
        # predictive power, penalize phantom performers with zero IC.
        if ic_global:
            # Prefer per-ticker IC when available with enough samples
            _ic_info = None
            if ticker and ic_per_ticker:
                _ic_info = ic_per_ticker.get(ticker, {}).get(signal_name)
                if _ic_info and _ic_info.get("samples", 0) < _IC_MIN_SAMPLES:
                    _ic_info = None  # fall back to global
            if _ic_info is None:
                _ic_info = ic_global.get(signal_name, {})
            _ic = _ic_info.get("ic", 0.0)
            _icir = _ic_info.get("icir", 0.0)
            _ic_n = _ic_info.get("samples", 0)
            ic_mult = _compute_ic_mult(_ic, _icir, _ic_n)
            weight *= ic_mult
        # Regime adjustment
        weight *= regime_mults.get(signal_name, 1.0)
        # Horizon-specific weight adjustment
        if signal_name in horizon_mults:
            weight *= horizon_mults[signal_name]
        # Macro-window downweight (2026-04-28). Composes with regime/
        # horizon multipliers - e.g., during a macro window in ranging
        # regime, sentiment hits 0.5 (macro) x 0.X (regime) x Y (horizon).
        # Only applies to MACRO_WINDOW_DOWNWEIGHT_SIGNALS - the
        # FORCE_HOLD signals were already mutated to HOLD above and won't
        # reach this branch.
        if macro_active and signal_name in MACRO_WINDOW_DOWNWEIGHT_SIGNALS:
            weight *= MACRO_WINDOW_DOWNWEIGHT_MULTIPLIER
        # Crisis mode adjustments: penalize trend signals (only if they're
        # underperforming), boost mean-reversion. See 2026-04-19 fix above.
        if crisis_mode:
            if signal_name in _TREND_SIGNALS and crisis_penalize_trend:
                weight *= _CRISIS_TREND_PENALTY
            elif signal_name in _MR_SIGNALS:
                weight *= _CRISIS_MR_BOOST
        # Activation frequency normalization (rarity * bias correction)
        act_data = activation_rates.get(signal_name, {})
        norm_weight = act_data.get("normalized_weight", 1.0)
        weight *= norm_weight
        # Activity rate cap: penalize signals with extremely high activation rates
        act_rate = act_data.get("activation_rate", 0.0)
        if act_rate > _ACTIVITY_RATE_CAP:
            weight *= _ACTIVITY_RATE_PENALTY
        # Correlation penalty: secondary signals in a group get reduced weight
        if signal_name in penalized_signals:
            weight *= penalized_signals[signal_name]
        # Directional bias penalty (2026-05-02 research): signals with extreme
        # BUY/SELL bias get penalized ONLY when voting in their bias direction.
        # Contrarian votes (rare, high-value) keep full weight.
        # E.g., calendar is 100% BUY - its BUY votes get 0.5x, but a rare
        # SELL (if it ever emits one) keeps 1.0x because that's genuinely
        # informative. Previous version penalized ALL votes equally.
        signal_bias = act_data.get("bias", 0.0)
        signal_samples = act_data.get("samples", 0)
        if signal_samples >= _BIAS_MIN_ACTIVE and signal_bias > _BIAS_THRESHOLD:
            buy_rate = act_data.get("buy_rate", 0.0)
            sell_rate = act_data.get("sell_rate", 0.0)
            bias_direction = "BUY" if buy_rate >= sell_rate else "SELL"
            if vote == bias_direction:
                penalty = _BIAS_EXTREME_PENALTY if signal_bias > _BIAS_EXTREME_THRESHOLD else _BIAS_PENALTY
                weight *= penalty
        # 2026-05-11 (Codex Fix B): apply soft-vote dampening LAST so it
        # composes with all upstream multipliers (accuracy, IC, regime,
        # horizon, macro, crisis, activity, correlation, bias). The
        # soft_conf is small (0.15-0.20) for dead-zone votes - a strong
        # vote has no soft_conf key and so this branch is skipped,
        # preserving the existing strong-vote weight contract.
        soft = soft_confidences.get(f"_soft_conf_{signal_name}")
        if soft is not None:
            try:
                weight *= float(soft)
            except (TypeError, ValueError):
                pass
        if vote == "BUY":
            buy_weight += weight
        elif vote == "SELL":
            sell_weight += weight
    if gated_signals:
        logger.debug("Accuracy-gated signals (<%s%%): %s", ACCURACY_GATE_THRESHOLD * 100, gated_signals)
    total_weight = buy_weight + sell_weight
    if total_weight == 0:
        return "HOLD", 0.0
    buy_conf = buy_weight / total_weight
    sell_conf = sell_weight / total_weight
    if buy_conf > sell_conf and buy_conf >= 0.5:
        return "BUY", round(buy_conf, 4)
    if sell_conf > buy_conf and sell_conf >= 0.5:
        return "SELL", round(sell_conf, 4)
    return "HOLD", round(max(buy_conf, sell_conf), 4)


def _confluence_score(votes, indicators):
    active = {k: v for k, v in votes.items() if v != "HOLD"}
    if not active:
        return 0.0
    buy_count = sum(1 for v in active.values() if v == "BUY")
    sell_count = sum(1 for v in active.values() if v == "SELL")
    majority = max(buy_count, sell_count)
    score = majority / len(active)
    if indicators.get("volume_action") in ("BUY", "SELL"):
        vol_dir = indicators.get("volume_action")
        majority_dir = "BUY" if buy_count >= sell_count else "SELL"
        if vol_dir == majority_dir:
            score += 0.1
    return min(round(score, 4), 1.0)


def _time_of_day_factor(horizon=None):
    hour = datetime.now(UTC).hour
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import time_of_day_scale_3h
        return time_of_day_scale_3h(hour)
    # Default 1d behavior
    if 2 <= hour <= 6:
        return 0.8
    return 1.0


def _load_local_model_accuracy(signal_name, horizon="1d", days=None, cache_ttl=None):
    """Load per-ticker accuracy for a local model signal."""
    lookback_days = days if days is not None else _LOCAL_MODEL_LOOKBACK_DAYS
    ttl = cache_ttl or _LOCAL_MODEL_ACCURACY_TTL
    cache_key = f"local_model_accuracy_{signal_name}_{horizon}_{lookback_days}"

    def _fetch():
        try:
            from portfolio.accuracy_stats import accuracy_by_signal_ticker

            return accuracy_by_signal_ticker(signal_name, horizon=horizon, days=lookback_days)
        except Exception:
            logger.warning("Failed to load %s accuracy", signal_name, exc_info=True)
            return {}

    return _cached(cache_key, ttl, _fetch)


def _build_llm_context(ticker, ind, timeframes, extra_info):
    """Build shared context dict for local LLM signals (Ministral, Qwen3)."""
    tf_summary = ""
    if timeframes:
        parts = []
        for label, entry in timeframes:
            if isinstance(entry, dict) and "action" in entry and entry["action"]:
                ti = entry.get("indicators", {})
                parts.append(f"{label}: {entry['action']} (RSI={ti.get('rsi', 0):.0f})")
        if parts:
            tf_summary = " | ".join(parts)

    ema_gap = (
        abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100
        if ind["ema21"] != 0
        else 0
    )

    return {
        "ticker": ticker.replace("-USD", ""),
        "price_usd": ind["close"],
        "rsi": round(ind["rsi"], 1),
        # 2026-04-10: bumped to 5 decimals - see portfolio/reporting.py:114
        # for the root-cause explanation (MACD-improving gate rounding).
        "macd_hist": round(ind["macd_hist"], 5),
        "ema_bullish": ind["ema9"] > ind["ema21"],
        "ema_gap_pct": round(ema_gap, 2),
        "bb_position": ind["price_vs_bb"],
        "fear_greed": extra_info.get("fear_greed", "N/A"),
        "fear_greed_class": extra_info.get("fear_greed_class", ""),
        "news_sentiment": extra_info.get("sentiment", "N/A"),
        "sentiment_confidence": extra_info.get("sentiment_conf", "N/A"),
        "volume_ratio": extra_info.get("volume_ratio", "N/A"),
        "funding_rate": extra_info.get("funding_action", "N/A"),
        "timeframe_summary": tf_summary,
        "headlines": "",
    }


def _gate_local_model_vote(signal_name, vote, ticker, config=None):
    """Apply accuracy-based abstention to local model votes."""
    # 2026-05-10 (codex re-review): explicit dict[str, Any] - initial
    # values mix str / None / int and later rounds add float (accuracy)
    # and int (samples). Without annotation mypy locks the value type
    # to the union of the literal initialisers and rejects every later
    # assignment.
    info: dict[str, Any] = {
        "gating": "raw",
        "accuracy": None,
        "samples": 0,
    }
    if vote == "HOLD" or not ticker:
        return vote, info

    cfg = ((config or {}).get("local_models", {}) or {}).get(signal_name, {})
    hold_threshold = cfg.get("hold_threshold", _LOCAL_MODEL_HOLD_THRESHOLD)
    min_samples = cfg.get("min_samples", _LOCAL_MODEL_MIN_SAMPLES)
    days = cfg.get("accuracy_days", _LOCAL_MODEL_LOOKBACK_DAYS)
    cache_ttl = cfg.get("accuracy_cache_ttl", _LOCAL_MODEL_ACCURACY_TTL)

    accuracy_data = _load_local_model_accuracy(
        signal_name, horizon=cfg.get("horizon", "1d"), days=days, cache_ttl=cache_ttl
    )
    ticker_stats = (accuracy_data or {}).get(ticker)
    if not ticker_stats or ticker_stats.get("samples", 0) < min_samples:
        info["gating"] = "insufficient_data"
        if ticker_stats:
            info["accuracy"] = round(ticker_stats.get("accuracy", 0.0), 3)
            info["samples"] = ticker_stats.get("samples", 0)
        return vote, info

    accuracy = float(ticker_stats.get("accuracy", 0.0))
    samples = int(ticker_stats.get("samples", 0))
    info["accuracy"] = round(accuracy, 3)
    info["samples"] = samples
    if accuracy < hold_threshold:
        info["gating"] = "held"
        return "HOLD", info

    return vote, info


def _compute_adx(df, period=14):
    """Compute ADX (Average Directional Index) from a DataFrame with high/low/close.

    Returns the latest ADX value, or None if insufficient data.
    Cached per DataFrame identity to avoid recomputation within a cycle.
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < period * 2:
        return None

    # C1: Content-based key prevents GC-reuse collisions when a new DataFrame
    # is allocated at the same address as a previously freed one.
    df_id = (id(df), len(df), float(df["close"].iloc[-1]) if len(df) > 0 else 0.0)
    with _adx_lock:
        if df_id in _adx_cache:
            return _adx_cache[df_id]

    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = true_range(high, low, close)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        alpha = 1.0 / period
        atr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        # Use clip(lower=1e-10) instead of replace(0, np.nan) to avoid NaN propagation
        atr_clipped = atr_smooth.clip(lower=1e-10)
        plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_clipped
        minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_clipped

        di_sum = (plus_di + minus_di).clip(lower=1e-10)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        val = adx.iloc[-1]
        result = float(val) if pd.notna(val) and np.isfinite(val) else None
        # BUG-86: Thread-safe cache write with eviction
        # BUG-180: LRU eviction - keep newest 50% instead of clearing all
        with _adx_lock:
            if len(_adx_cache) >= _ADX_CACHE_MAX:
                keys = list(_adx_cache.keys())
                for k in keys[:len(keys) // 2]:
                    del _adx_cache[k]
            _adx_cache[df_id] = result
        return result
    except Exception:
        logger.warning("ADX computation failed", exc_info=True)
        with _adx_lock:
            _adx_cache[df_id] = None
        return None


def apply_confidence_penalties(action, conf, regime, ind, extra_info, ticker, df, config):
    """Apply an 8-stage multiplicative confidence penalty cascade.

    Stages:
      1. Regime penalty - dampens confidence in choppy/volatile markets
      2. Volume/ADX gate - rejects low-conviction signals
      3. Trap detection - catches bull/bear traps (price vs volume divergence)
      4. Dynamic MIN_VOTERS - raises the bar in uncertain markets
      5. Unanimity penalty - over-agreement often means the move is priced in
      5b. Ensemble entropy - high 3-way disagreement caps confidence
      6. Per-ticker consensus - penalizes tickers where ensemble accuracy < 50%
      7. Calibration compression - compress overconfident predictions to honest levels

    Returns (action, conf, penalty_log) where penalty_log is a list of applied penalties.
    """
    cfg = (config or {}).get("confidence_penalties", {})
    if cfg.get("enabled") is False:
        return action, conf, []

    penalty_log = []

    # --- Stage 1: Regime penalties ---
    if regime == "ranging":
        conf *= 0.75
        penalty_log.append({"stage": "regime", "regime": "ranging", "mult": 0.75})
    elif regime == "high-vol":
        conf *= 0.80
        penalty_log.append({"stage": "regime", "regime": "high-vol", "mult": 0.80})
    elif regime in ("trending-up", "trending-down"):
        # Bonus only if action aligns with trend direction
        trending_buy = regime == "trending-up" and action == "BUY"
        trending_sell = regime == "trending-down" and action == "SELL"
        if trending_buy or trending_sell:
            conf *= 1.10
            penalty_log.append({"stage": "regime", "regime": regime, "aligned": True, "mult": 1.10})
    # BUG-90: Clamp after Stage 1 so inflated confidence doesn't bypass Stage 2 gates
    conf = min(1.0, conf)

    # --- Stage 2: Volume/ADX gate ---
    volume_ratio = extra_info.get("volume_ratio")
    adx = _compute_adx(df)
    extra_info["_adx"] = adx

    if volume_ratio is not None and action != "HOLD":
        if volume_ratio < 0.5:
            # Very low volume - force HOLD
            penalty_log.append({"stage": "volume_gate", "rvol": volume_ratio, "effect": "force_hold"})
            action = "HOLD"
            conf = 0.0
        elif volume_ratio < 0.8 and (adx is not None and adx < 20) and conf < 0.65:
            # Low volume + weak trend + marginal confidence - force HOLD
            penalty_log.append({
                "stage": "volume_adx_gate", "rvol": volume_ratio,
                "adx": round(adx, 1), "conf": round(conf, 4), "effect": "force_hold",
            })
            action = "HOLD"
            conf = 0.0
        elif volume_ratio > 1.5:
            # High volume - slight confidence boost
            conf *= 1.15
            penalty_log.append({"stage": "volume_boost", "rvol": volume_ratio, "mult": 1.15})
    # BUG-90: Clamp after Stage 2
    conf = min(1.0, conf)

    # --- Stage 3: Trap detection ---
    # NOTE: df must be the "Now" timeframe (15m candles, 100 bars ÷ 25h).
    # Last 5 bars = 75 minutes - appropriate for intraday trap detection.
    if action != "HOLD" and df is not None and isinstance(df, pd.DataFrame) and len(df) >= 5:
        try:
            recent_close = df["close"].iloc[-5:]
            recent_vol = df["volume"].iloc[-5:] if "volume" in df.columns else None
            price_up = recent_close.iloc[-1] > recent_close.iloc[0]
            price_down = recent_close.iloc[-1] < recent_close.iloc[0]

            if recent_vol is not None and len(recent_vol) >= 5:
                vol_declining = recent_vol.iloc[-1] < recent_vol.iloc[0] * 0.8

                if action == "BUY" and price_up and vol_declining:
                    conf *= 0.5
                    penalty_log.append({"stage": "trap", "type": "bull_trap", "mult": 0.5})
                elif action == "SELL" and price_down and vol_declining:
                    conf *= 0.5
                    penalty_log.append({"stage": "trap", "type": "bear_trap", "mult": 0.5})
        except Exception:
            logger.warning("Trap detection failed for %s", ticker, exc_info=True)
    # BUG-90: Clamp after Stage 3
    conf = min(1.0, conf)

    # --- Stage 4: Dynamic MIN_VOTERS ---
    # P2-C (2026-04-17): delegate to shared helper to avoid drift with the
    # circuit breaker's recovery-floor logic. Same semantic as before.
    # BUG-227: Use post-persistence voter count (not pre-filter) so the gate
    # reflects the actual participating voters after debounce filtering.
    active_voters = extra_info.get("_voters_post_filter",
                                    extra_info.get("_voters", 0))
    dynamic_min = _dynamic_min_voters_for_regime(regime)

    if action != "HOLD" and active_voters < dynamic_min:
        penalty_log.append({
            "stage": "dynamic_min_voters", "regime": regime,
            "required": dynamic_min, "actual": active_voters, "effect": "force_hold",
        })
        action = "HOLD"
        conf = 0.0

    # --- Stage 5: Unanimity penalty ---
    # When all signals agree, the move is often already priced in.
    # 90%+ confidence has 28-32% actual accuracy across all horizons.
    if action != "HOLD" and conf > 0.0:
        buy_count = extra_info.get("_buy_count", 0)
        sell_count = extra_info.get("_sell_count", 0)
        total_voters = buy_count + sell_count
        if total_voters > 0:
            agreement_ratio = max(buy_count, sell_count) / total_voters
            if agreement_ratio >= 0.9:  # 90%+ agreement
                conf *= 0.6
                penalty_log.append({"stage": "unanimity", "agreement": round(agreement_ratio, 3), "mult": 0.6})
            elif agreement_ratio >= 0.8:  # 80-90% agreement
                conf *= 0.75
                penalty_log.append({"stage": "unanimity", "agreement": round(agreement_ratio, 3), "mult": 0.75})

    # --- Stage 5b: Ensemble entropy guard ---
    # High Shannon entropy across BUY/SELL/HOLD means signals genuinely disagree -
    # the winning direction is near-random. Cap confidence.
    _ENTROPY_MIN_APPLICABLE = 5
    if action != "HOLD" and conf > 0.0:
        import math as _math_ent
        _ent_buy = extra_info.get("_buy_count", 0)
        _ent_sell = extra_info.get("_sell_count", 0)
        _ent_total = extra_info.get("_total_applicable", _ent_buy + _ent_sell)
        _ent_hold = max(0, _ent_total - _ent_buy - _ent_sell)
        if _ent_total >= _ENTROPY_MIN_APPLICABLE:
            _probs = [c / _ent_total for c in (_ent_buy, _ent_sell, _ent_hold) if c > 0]
            _entropy = -sum(p * _math_ent.log2(p) for p in _probs)
            _norm_entropy = _entropy / _math_ent.log2(3)
            extra_info["_ensemble_entropy"] = round(_norm_entropy, 4)
            if _norm_entropy >= 0.9:
                conf *= 0.6
                penalty_log.append({"stage": "ensemble_entropy", "entropy": round(_norm_entropy, 3), "mult": 0.6})
            elif _norm_entropy >= 0.8:
                conf *= 0.8
                penalty_log.append({"stage": "ensemble_entropy", "entropy": round(_norm_entropy, 3), "mult": 0.8})

    # --- Stage 6: Per-ticker consensus accuracy penalty ---
    # RES-2026-04-17: The consensus system has below-coinflip accuracy for some
    # tickers (ETH-USD 47.7% at 3h, MSTR 45.9%). When this happens, the ensemble
    # is net-negative - acting on its signals loses money. Apply a confidence
    # penalty proportional to how far below 52% the consensus accuracy is.
    # Don't force HOLD (too aggressive) - just reduce confidence.
    # RES-2026-04-21: Raised threshold 0.500.52 to catch coin-flip tickers
    # (XAG-USD 50.0%, XAU-USD 49.6% were getting zero penalty). Steepened
    # the curve and lowered floor (0.30.2) for truly broken instruments.
    _PTC_MIN_SAMPLES = 500
    _PTC_PENALTY_THRESHOLD = 0.52
    if action != "HOLD":
        ptc_acc = extra_info.get("_ptc_accuracy")
        ptc_samples = extra_info.get("_ptc_samples", 0)
        if ptc_acc is not None and ptc_samples >= _PTC_MIN_SAMPLES and ptc_acc < _PTC_PENALTY_THRESHOLD:
            # Scale penalty: 52% acc  0.6x, 50% acc  0.52x, 48% acc  0.44x, 40% acc  0.2x
            ptc_mult = max(0.2, 0.6 + (ptc_acc - _PTC_PENALTY_THRESHOLD) * 4.0)
            conf *= ptc_mult
            penalty_log.append({
                "stage": "per_ticker_consensus",
                "ticker": ticker,
                "ptc_accuracy": round(ptc_acc, 4),
                "ptc_samples": ptc_samples,
                "mult": round(ptc_mult, 4),
            })

    # --- Stage 7: Confidence calibration compression ---
    # RES-2026-04-18: Calibration analysis shows confidence is meaningless
    # above 60% - all bands (60-69%, 70-79%, 80-89%) have ~50% actual
    # accuracy. The system is massively overconfident. Compress high-confidence
    # predictions to honest levels while preserving relative ordering.
    # Formula: conf = 0.55 + (conf - 0.55) * 0.3  (for conf > 0.55)
    # Maps: 60%  56.5%, 70%  59.5%, 80%  62.5%, 90%  65.5%
    _CALIBRATION_THRESHOLD = 0.55
    _CALIBRATION_COMPRESSION = 0.3
    if action != "HOLD" and conf > _CALIBRATION_THRESHOLD:
        raw_conf = conf
        conf = _CALIBRATION_THRESHOLD + (conf - _CALIBRATION_THRESHOLD) * _CALIBRATION_COMPRESSION
        penalty_log.append({
            "stage": "calibration_compression",
            "raw_conf": round(raw_conf, 4),
            "compressed_conf": round(conf, 4),
        })

    # Clamp confidence to [0, 1]
    conf = max(0.0, min(1.0, conf))

    return action, conf, penalty_log


def generate_signal(ind, ticker=None, config=None, timeframes=None, df=None, horizon=None):
    # CRITICAL-2 guard (2026-04-17 adversarial review): empty/None ticker
    # slipped through scattered `if ticker:` checks in production before.
    # All real callers (main.py:486, agent_invocation, backtester) pass a
    # non-empty string; only tests/test_signal_engine.py:651 passes None
    # intentionally (BUG-178 regression guard for tracker pollution).
    # Warn on empty so future regressions surface rather than silently
    # degrading the signal pipeline.
    if not ticker:
        logger.warning(
            "generate_signal called with empty ticker=%r - "
            "tracker/phase updates will be skipped",
            ticker,
        )

    # 2026-05-10 (codex re-review): explicit annotations so the strict pilot
    # sees the heterogeneous shape we rely on. Without these mypy infers
    # ``dict[Any, bool]`` from the first truthy assignment and reports a
    # cascade of false-positive [assignment]/[arg-type] errors at every
    # subsequent extra_info["..."] = <non-bool> line (Codex flagged 30+).
    votes: dict[str, str] = {}
    shadow_votes: dict[str, str] = {}  # disabled signals tracked for accuracy
    extra_info: dict[str, Any] = {}

    # BUG-178 diagnostic phase marker (added 2026-04-10, diag/bug178-end-of-cycle-snapshot).
    # The per-ticker last-signal tracker is updated inside the enhanced-signal
    # dispatch loop, but slow cycles can also hang BEFORE the loop (sentiment,
    # fear_greed, news_event, _cached() macro fetches) or AFTER it (accuracy_stats
    # loading, weighted consensus, per-ticker gate). Writing `__pre_dispatch__`
    # here and `__post_dispatch__` after the loop gives the end-of-cycle slow
    # diagnostic in main.py three distinct phases to point at:
    #   - __pre_dispatch__   hang is in the pre-loop block
    #   - <signal name>      hang is in the dispatch loop at that signal
    #   - __post_dispatch__  hang is in accuracy/consensus code after the loop
    # The double-underscore prefix is reserved; no registered signal uses it.
    #
    # 2026-04-15 (bug178-instrumentation-timeout): also reset the per-phase log
    # so the post-dispatch code can record granular timing between __post_dispatch__
    # and the return. See _record_phase() and _reset_phase_log() above.
    if ticker:
        _set_last_signal(ticker, "__pre_dispatch__")
        _reset_phase_log(ticker)

    # Check if GPU-intensive signals should be skipped (stocks outside market hours)
    from portfolio.market_timing import should_skip_gpu
    skip_gpu = should_skip_gpu(ticker, config=config) if ticker else False
    if skip_gpu:
        extra_info["_gpu_signals_skipped"] = True

    # Compute regime early so F&G gating and other sections can use it
    regime = detect_regime(ind, is_crypto=ticker in CRYPTO_SYMBOLS)

    # RSI - only votes at extremes (adaptive thresholds from rolling percentiles)
    if horizon in ("3h", "4h"):
        # 3h: RSI(7) is more sensitive - use fixed 25/75 thresholds
        rsi_lower = 25
        rsi_upper = 75
    else:
        rsi_lower = ind.get("rsi_p20", 30)
        rsi_upper = ind.get("rsi_p80", 70)
        rsi_lower = max(rsi_lower, 15)
        rsi_upper = min(rsi_upper, 85)
    if ind["rsi"] < rsi_lower:
        votes["rsi"] = "BUY"
    elif ind["rsi"] > rsi_upper:
        votes["rsi"] = "SELL"
    else:
        votes["rsi"] = "HOLD"

    # MACD - strong vote on histogram crossover; soft directional vote in
    # the dead-zone (2026-05-11 Stage 2 Batch 1). The strong-vote path is
    # unchanged. When |hist| is small and there is no crossover, the
    # dead-zone helper inspects the histogram slope over the last few
    # bars and emits a weak BUY/SELL (conf=MACD_DEAD_ZONE_SOFT_CONF) when
    # there is directional drift, or HOLD when truly flat. Rationale:
    # entries should pick a direction; HOLD is for managing positions.
    if ind["macd_hist"] > 0 and ind["macd_hist_prev"] <= 0:
        votes["macd"] = "BUY"
    elif ind["macd_hist"] < 0 and ind["macd_hist_prev"] >= 0:
        votes["macd"] = "SELL"
    else:
        macd_vote, macd_soft_conf = _macd_dead_zone_vote(ind, df)
        votes["macd"] = macd_vote
        if macd_vote != "HOLD":
            extra_info["_soft_conf_macd"] = macd_soft_conf

    # EMA trend - strong vote when gap >= 0.5%; soft directional vote in
    # the dead-zone (2026-05-11 Stage 2 Batch 1). The strong-vote path is
    # unchanged. When the gap is small, compare EMA9 slope to EMA21
    # slope over the last few bars; faster EMA9 -> weak BUY, slower ->
    # weak SELL, flat -> HOLD. Rationale: entries should pick a
    # direction; HOLD is for managing positions.
    ema_gap_pct = (
        abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100 if ind["ema21"] != 0 else 0
    )
    if ema_gap_pct >= 0.5:
        votes["ema"] = "BUY" if ind["ema9"] > ind["ema21"] else "SELL"
    else:
        ema_vote, ema_soft_conf = _ema_dead_zone_vote(ind, df)
        votes["ema"] = ema_vote
        if ema_vote != "HOLD":
            extra_info["_soft_conf_ema"] = ema_soft_conf

    # Bollinger Bands - strong vote at band touches; soft directional
    # vote when price is inside the band but biased toward an edge
    # (2026-05-11 Stage 2 Batch 1). Normalized band position
    # (price - mid) / (upper - mid) clamped to [-1, +1]:
    # > 0.6 -> weak SELL, < -0.6 -> weak BUY, else HOLD. Rationale:
    # entries should pick a direction; HOLD is for managing positions.
    if ind["price_vs_bb"] == "below_lower":
        votes["bb"] = "BUY"
    elif ind["price_vs_bb"] == "above_upper":
        votes["bb"] = "SELL"
    else:
        bb_vote, bb_soft_conf = _bb_inside_band_vote(ind)
        votes["bb"] = bb_vote
        if bb_vote != "HOLD":
            extra_info["_soft_conf_bb"] = bb_soft_conf

    # --- Extended signals from tools (optional) ---

    # Fear & Greed Index (per-ticker: crypto->alternative.me, stocks->VIX)
    # Gated: F&G is contrarian (buy fear, sell greed) which fights trends.
    # Only allow F&G to vote in ranging/high-vol regimes where mean reversion works.
    # 2026-04-02: Added sustained fear duration gate - during prolonged extreme
    # fear (46+ consecutive days as of Apr 2), contrarian BUY signals are noise.
    # Historical data: buying at F&G <15 yields median +38.4% over 90 days,
    # but during sustained fear (2022), prices dropped another -40% after signal.
    votes["fear_greed"] = "HOLD"
    try:
        from portfolio.fear_greed import get_fear_greed, get_sustained_fear_days, update_fear_streak

        fg_key = f"fear_greed_{ticker}" if ticker else "fear_greed"
        fg = _cached(fg_key, FEAR_GREED_TTL, get_fear_greed, ticker)
        if fg:
            extra_info["fear_greed"] = fg["value"]
            extra_info["fear_greed_class"] = fg["classification"]
            # Read streak BEFORE updating - use previous cycle's state for voting
            fear_days = get_sustained_fear_days()
            # Update streak tracker (once per cycle, not per ticker)
            if ticker in ("BTC-USD", None):
                update_fear_streak(fg["value"])
            extra_info["fear_greed_streak_days"] = fear_days
            # Gate: suppress F&G votes in trending regimes
            if regime in ("trending-up", "trending-down"):
                extra_info["fear_greed_gated"] = regime
                votes["fear_greed"] = "HOLD"
            # Gate: sustained extreme fear - contrarian BUY is unreliable
            # during first 30 days of prolonged fear (whiplash risk).
            # After 30 days, allow BUY but at reduced confidence (handled
            # by the existing 0.3x ranging regime weight).
            elif fg["value"] <= 20 and fear_days > 30:
                votes["fear_greed"] = "BUY"
                extra_info["fear_greed_note"] = f"sustained_fear_{fear_days}d_allowing_contrarian"
            elif fg["value"] <= 20 and fear_days > 0:
                extra_info["fear_greed_gated"] = f"sustained_fear_{fear_days}d"
                votes["fear_greed"] = "HOLD"
            elif fg["value"] <= 20:
                votes["fear_greed"] = "BUY"
            elif fg["value"] >= 80:
                votes["fear_greed"] = "SELL"
    except ImportError:
        logger.debug("Optional module %s not available", "fear_greed")

    # Social media posts (Reddit) - fetched separately, merged into sentiment
    social_posts = []
    if ticker:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.social_sentiment import get_reddit_posts

            reddit = _cached(
                f"reddit_{short_ticker}",
                SENTIMENT_TTL,
                get_reddit_posts,
                short_ticker,
            )
            if reddit:
                social_posts.extend(reddit)
        except ImportError:
            logger.debug("Optional module %s not available", "social_sentiment")

    # Sentiment (crypto->CryptoBERT, stocks->Trading-Hero-LLM) - includes social posts
    # Hysteresis: flipping direction requires confidence > 0.55, same direction > 0.40
    votes["sentiment"] = "HOLD"
    if ticker:
        short_ticker = ticker.replace("-USD", "")
        try:
            from functools import partial

            from portfolio.sentiment import get_sentiment

            newsapi_key = (config or {}).get("newsapi_key", "")
            cc_api_key = (config or {}).get("cryptocompare_api_key", "") or None
            _sent_fn = partial(get_sentiment, cryptocompare_api_key=cc_api_key)
            sent = _cached(
                f"sentiment_{short_ticker}",
                SENTIMENT_TTL,
                _sent_fn,
                short_ticker,
                newsapi_key or None,
                social_posts or None,
            )
            if sent and sent.get("num_articles", 0) > 0:
                extra_info["sentiment"] = sent["overall_sentiment"]
                extra_info["sentiment_conf"] = sent["confidence"]
                extra_info["sentiment_model"] = sent.get("model", "unknown")
                if sent.get("sources"):
                    extra_info["sentiment_sources"] = sent["sources"]
                # 2026-04-21: carry avg_scores forward for the LLM probability
                # logger so it can write a rich P(positive)/P(negative)/P(neutral)
                # distribution instead of a confidence-split fallback.
                if sent.get("avg_scores"):
                    extra_info["sentiment_avg_scores"] = sent["avg_scores"]

                prev_sent_dir = _get_prev_sentiment(ticker)
                current_dir = sent["overall_sentiment"]
                if (
                    prev_sent_dir
                    and current_dir != prev_sent_dir
                    and current_dir != "neutral"
                ):
                    sent_threshold = 0.55
                else:
                    sent_threshold = 0.40

                if (
                    sent["overall_sentiment"] == "positive"
                    and sent["confidence"] > sent_threshold
                ):
                    votes["sentiment"] = "BUY"
                    _set_prev_sentiment(ticker, "positive")
                elif (
                    sent["overall_sentiment"] == "negative"
                    and sent["confidence"] > sent_threshold
                ):
                    votes["sentiment"] = "SELL"
                    _set_prev_sentiment(ticker, "negative")
        except ImportError:
            logger.debug("Optional module %s not available", "sentiment")

    # ML Classifier - disabled: 28.2% accuracy (1,027 samples, 1d horizon).
    # Worse than coin flip; actively harmful to consensus. Still tracked for
    # accuracy monitoring but never votes.
    votes["ml"] = "HOLD"

    # Funding Rate - 29.9% accuracy at 1d but 74.2% at 3h (535 samples).
    # Re-enabled 2026-04-09: horizon-gated via REGIME_GATED_SIGNALS (_default
    # gates it at 1d across all regimes; active only at 3h/4h horizons).
    # Crypto-only (BTC, ETH). The regime gate handles suppression at 1d.
    votes["funding"] = "HOLD"
    if ticker and ticker in CRYPTO_SYMBOLS:
        try:
            from portfolio.funding_rate import get_funding_rate

            fr = _cached(f"funding_{ticker}", FUNDING_RATE_TTL, get_funding_rate, ticker)
            if fr:
                extra_info["funding_rate"] = fr["rate_pct"]
                extra_info["funding_action"] = fr["action"]
                votes["funding"] = fr["action"]
        except ImportError:
            logger.debug("Optional module %s not available", "funding_rate")

    # On-Chain BTC Signal - MVRV Z-Score, SOPR, NUPL, Exchange Netflow.
    # BTC-only. Data from BGeometrics (12h cache, 15 req/day).
    # Added 2026-04-09: promotes existing on-chain data to a voting signal.
    votes["onchain"] = "HOLD"
    if ticker == "BTC-USD":
        try:
            from portfolio.onchain_data import get_onchain_data

            oc = _cached("onchain_btc_signal", ONCHAIN_TTL, get_onchain_data)
            if oc:
                sub_votes = []
                # MVRV Z-Score: <1 undervalued (BUY), >5 overheated (SELL)
                zscore = oc.get("mvrv_zscore")
                if zscore is not None:
                    if zscore < 1.0:
                        sub_votes.append("BUY")
                    elif zscore > 5.0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_mvrv_zscore"] = round(zscore, 2)
                # SOPR: <0.97 capitulation (BUY), >1.05 profit-taking (SELL)
                sopr = oc.get("sopr")
                if sopr is not None:
                    if sopr < 0.97:
                        sub_votes.append("BUY")
                    elif sopr > 1.05:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_sopr"] = round(sopr, 4)
                # NUPL: <0 capitulation (BUY), >0.75 euphoria (SELL)
                nupl = oc.get("nupl")
                if nupl is not None:
                    if nupl < 0:
                        sub_votes.append("BUY")
                    elif nupl > 0.75:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                # Exchange netflow: negative = accumulation (BUY)
                netflow = oc.get("netflow")
                if netflow is not None:
                    if netflow < 0:
                        sub_votes.append("BUY")
                    elif netflow > 0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                # Majority vote
                buy_count = sub_votes.count("BUY")
                sell_count = sub_votes.count("SELL")
                total = buy_count + sell_count
                if total >= 2:
                    if buy_count > sell_count:
                        votes["onchain"] = "BUY"
                    elif sell_count > buy_count:
                        votes["onchain"] = "SELL"
                    extra_info["onchain_sub_votes"] = f"{buy_count}B/{sell_count}S"
        except ImportError:
            logger.debug("Optional module %s not available", "onchain_data")

    # Volume Confirmation (spike + price direction = vote)
    votes["volume"] = "HOLD"
    if ticker:
        try:
            from portfolio.macro_context import get_volume_signal

            vs = _cached(f"volume_{ticker}", VOLUME_TTL, get_volume_signal, ticker)
            if vs:
                extra_info["volume_ratio"] = vs["ratio"]
                extra_info["volume_action"] = vs["action"]
                votes["volume"] = vs["action"]
        except ImportError:
            logger.debug("Optional module %s not available", "macro_context")

    # Ministral-3-8B LLM reasoning (all tickers - crypto, stocks, metals)
    # Upgraded from legacy Ministral-8B (44% accuracy) to Ministral-3-8B.
    # custom_lora fully disabled: 20.9% accuracy, 97% SELL bias (worse than random).
    # Uses batch queue: on cache miss, enqueues for post-cycle flush instead of
    # calling model inline (avoids model swap ping-pong between threads).
    votes["ministral"] = "HOLD"
    if ticker and not skip_gpu:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.llm_batch import enqueue_ministral, is_llm_on_cycle

            ctx = _build_llm_context(ticker, ind, timeframes, extra_info)
            # 2026-04-10 (perf/llama-swap-reduction): gate the enqueue with
            # rotation predicate. When ministral is off-cycle this cycle,
            # _cached_or_enqueue skips the enqueue and returns stale data.
            # max_stale_factor=5 gives 5 * 15 min = 75 min of stale tolerance,
            # comfortably covering the 3-cycle rotation period (~45-60 min
            # depending on cycle slippage) with slack.
            ms = _cached_or_enqueue(
                f"ministral_{short_ticker}",
                MINISTRAL_TTL,
                enqueue_ministral,
                ctx,
                should_enqueue_fn=lambda: is_llm_on_cycle("ministral"),
                max_stale_factor=5,
            )
            if ms:
                orig = ms.get("original") or ms
                raw_action = orig["action"]
                gated_action, gating = _gate_local_model_vote(
                    "ministral", raw_action, ticker, config=config
                )
                extra_info["ministral_raw_action"] = raw_action
                extra_info["ministral_action"] = gated_action
                extra_info["ministral_reasoning"] = orig.get("reasoning", "")
                extra_info["ministral_accuracy"] = gating.get("accuracy")
                extra_info["ministral_samples"] = gating.get("samples", 0)
                extra_info["ministral_gating"] = gating.get("gating", "raw")
                if orig.get("confidence") is not None:
                    extra_info["ministral_confidence"] = orig["confidence"]
                votes["ministral"] = gated_action

                # custom_lora fully disabled - not even stored in extra.
                # Shadow A/B data preserved in data/ab_test_log.jsonl.
        except ImportError:
            logger.debug("Optional module %s not available", "ministral_signal")

    # Qwen3-8B LLM reasoning (all tickers - crypto, stocks, metals)
    # General financial model providing ensemble diversification vs Ministral.
    # Config: config.json  local_models.qwen3 (hold_threshold, min_samples)
    # Uses batch queue: same pattern as Ministral above.
    votes["qwen3"] = "HOLD"
    qwen3_enabled = (config or {}).get("local_models", {}).get("qwen3", {}).get("enabled", True)
    if ticker and qwen3_enabled and not skip_gpu:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.llm_batch import enqueue_qwen3, is_llm_on_cycle

            ctx = _build_llm_context(ticker, ind, timeframes, extra_info)
            # Qwen3 gets asset_type for prompt diversification
            if ticker in CRYPTO_SYMBOLS:
                ctx["asset_type"] = "cryptocurrency"
            elif ticker in METALS_SYMBOLS:
                ctx["asset_type"] = "precious metal"
            else:
                ctx["asset_type"] = "stock"
            # 2026-04-10 (perf/llama-swap-reduction): rotation gate via
            # is_llm_on_cycle; see ministral block above for rationale.
            q3 = _cached_or_enqueue(
                f"qwen3_{short_ticker}",
                MINISTRAL_TTL,
                enqueue_qwen3,
                ctx,
                should_enqueue_fn=lambda: is_llm_on_cycle("qwen3"),
                max_stale_factor=5,
            )
            if q3:
                raw_action = q3.get("action", "HOLD")
                gated_action, gating = _gate_local_model_vote(
                    "qwen3", raw_action, ticker, config=config
                )
                extra_info["qwen3_raw_action"] = raw_action
                extra_info["qwen3_action"] = gated_action
                extra_info["qwen3_reasoning"] = q3.get("reasoning", "")
                extra_info["qwen3_accuracy"] = gating.get("accuracy")
                extra_info["qwen3_samples"] = gating.get("samples", 0)
                extra_info["qwen3_gating"] = gating.get("gating", "raw")
                if q3.get("confidence") is not None:
                    extra_info["qwen3_confidence"] = q3["confidence"]
                votes["qwen3"] = gated_action

        except ImportError:
            logger.debug("Optional module %s not available", "qwen3_signal")

    # --- Enhanced signal modules (composite indicators computed from raw OHLCV) ---
    # Loaded from signal_registry - no hardcoded list needed here.
    _enhanced_entries = get_enhanced_signals()

    if df is not None and isinstance(df, pd.DataFrame) and len(df) >= 26:
        # Fetch macro context once for any signal that requires it
        macro_data = None
        has_macro_signals = any(e.get("requires_macro") for e in _enhanced_entries.values())
        if has_macro_signals:
            try:
                from portfolio.macro_context import get_dxy, get_fed_calendar, get_treasury
                macro_data = {}
                dxy = _cached("dxy", 3600, get_dxy)
                if dxy:
                    macro_data["dxy"] = dxy
                treasury = _cached("treasury", 3600, get_treasury)
                if treasury:
                    macro_data["treasury"] = treasury
                fed = get_fed_calendar()
                if fed:
                    macro_data["fed"] = fed
            except Exception:
                logger.warning("Macro context fetch failed", exc_info=True)

        # Load seasonality profile for metals tickers (detrending)
        seasonality_profile = None
        if ticker in METALS_SYMBOLS:
            try:
                from portfolio.seasonality import get_profile
                seasonality_profile = get_profile(ticker)
            except Exception:
                logger.debug("Seasonality profile load failed for %s", ticker, exc_info=True)

        # Build context data once for signals that need it
        # BUG-144: Include regime so enhanced signals (forecast.py) can apply
        # regime-specific confidence discounts.
        context_data = {
            "ticker": ticker, "config": config or {}, "macro": macro_data,
            "regime": regime, "seasonality_profile": seasonality_profile,
        }

        _signal_failures = []
        for sig_name, entry in _enhanced_entries.items():
            # BUG-178 fix (2026-04-10): respect DISABLED_SIGNALS in the dispatch
            # loop. Previously this loop iterated *every* registered enhanced
            # signal regardless of disabled status, which meant the three
            # "registered but force-HOLD pending live validation" signals
            # (crypto_macro, cot_positioning, credit_spread_risk) were doing
            # network I/O on every cycle. Their late position in iteration
            # order matches the silent gap before all 49 BUG-178 events
            # observed 2026-04-09/10 (last [SLOW] log line is always the
            # signal *before* this set, then 150+s of silence). Skipping them
            # restores the documented behavior. Other DISABLED_SIGNALS-aware
            # call sites: count_active_signals():468, dynamic correlation:558,
            # accuracy_stats.py, ticker_accuracy.py, backtester.py, reporting.py.
            if sig_name in DISABLED_SIGNALS and (sig_name, ticker) not in _DISABLED_SIGNAL_OVERRIDES:
                # Shadow-safe signals: compute but don't let them vote.
                # Their predictions go into _shadow_votes for accuracy tracking.
                if sig_name in _SHADOW_SAFE_SIGNALS:
                    try:
                        _sig_t0 = time.monotonic()
                        compute_fn = load_signal_func(entry)
                        if compute_fn is not None:
                            if ticker:
                                _set_last_signal(ticker, f"shadow:{sig_name}")
                            if entry.get("requires_context"):
                                result = compute_fn(df, context=context_data)
                            elif entry.get("requires_macro"):
                                result = compute_fn(df, macro=macro_data or None)
                            else:
                                result = compute_fn(df)
                            _sig_dt = time.monotonic() - _sig_t0
                            if _sig_dt > 1.0:
                                logger.info("[SLOW-SHADOW] %s/%s: %.1fs", ticker, sig_name, _sig_dt)
                            max_conf = entry.get("max_confidence", 1.0)
                            validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
                            extra_info[f"{sig_name}_action"] = validated["action"]
                            extra_info[f"{sig_name}_confidence"] = validated["confidence"]
                            extra_info[f"shadow_{sig_name}"] = True
                            shadow_votes[sig_name] = validated["action"]
                    except Exception as e:
                        logger.debug("Shadow signal %s failed: %s", sig_name, e)
                votes[sig_name] = "HOLD"
                continue
            if sig_name in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
                votes[sig_name] = "HOLD"
                continue
            # Skip GPU-intensive enhanced signals for stocks outside market hours
            if skip_gpu and sig_name in GPU_SIGNALS:
                votes[sig_name] = "HOLD"
                continue
            try:
                _sig_t0 = time.monotonic()
                compute_fn = load_signal_func(entry)
                if compute_fn is None:
                    votes[sig_name] = "HOLD"
                    continue
                # BUG-178 diagnostic: track which signal each ticker is currently
                # running so main.py's pool-timeout handler can name the culprit.
                # Ticker guard added 2026-04-10 in the phase-marker diag commit
                # to prevent leaking a None-keyed entry when callers pass
                # ticker=None (legacy test harnesses and backtester paths).
                if ticker:
                    _set_last_signal(ticker, sig_name)
                if entry.get("requires_context"):
                    result = compute_fn(df, context=context_data)
                elif entry.get("requires_macro"):
                    result = compute_fn(df, macro=macro_data or None)
                else:
                    result = compute_fn(df)
                _sig_dt = time.monotonic() - _sig_t0
                if _sig_dt > 1.0:
                    logger.info("[SLOW] %s/%s: %.1fs", ticker, sig_name, _sig_dt)
                max_conf = entry.get("max_confidence", 1.0)
                validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
                extra_info[f"{sig_name}_action"] = validated["action"]
                extra_info[f"{sig_name}_confidence"] = validated["confidence"]
                extra_info[f"{sig_name}_sub_signals"] = validated["sub_signals"]
                if validated["indicators"]:
                    extra_info[f"{sig_name}_indicators"] = validated["indicators"]
                votes[sig_name] = validated["action"]
            except Exception as e:
                logger.warning("Signal %s failed: %s", sig_name, e)
                votes[sig_name] = "HOLD"
                _signal_failures.append(sig_name)
        if _signal_failures:
            extra_info["_signal_failures"] = _signal_failures
            if len(_signal_failures) > 3:
                logger.warning(
                    "%s: %d enhanced signals failed: %s",
                    ticker, len(_signal_failures), ", ".join(_signal_failures),
                )

        # Persist signal health (single batch write for all enhanced signals)
        try:
            from portfolio.health import update_signal_health_batch
            health_results = {
                sig_name: (sig_name not in _signal_failures)
                for sig_name in _enhanced_entries
            }
            update_signal_health_batch(health_results)
        except Exception:
            logger.debug("Signal health tracking failed", exc_info=True)

        # 2026-05-11 Stage 2 Batch 2: candlestick + forecast dead-zone
        # soft directional votes. Same pattern as Batch 1 (EMA / BB /
        # MACD) but on enhanced signals - the helper only fires when
        # the strong path returned HOLD AND a secondary derivative
        # (candle body direction / price+EMA slope alignment) points
        # cleanly one way. Strong votes are untouched. The soft conf is
        # written into extra_info so `_weighted_consensus` dampens the
        # vote's weight via the existing soft_confidences path.
        if "candlestick" in votes and votes["candlestick"] == "HOLD" \
                and "candlestick" not in DISABLED_SIGNALS \
                and "candlestick" not in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
            cs_vote, cs_soft_conf = _candlestick_dead_zone_vote(df)
            if cs_vote != "HOLD":
                votes["candlestick"] = cs_vote
                extra_info["_soft_conf_candlestick"] = cs_soft_conf
                extra_info["candlestick_action"] = cs_vote
        if "forecast" in votes and votes["forecast"] == "HOLD" \
                and "forecast" not in DISABLED_SIGNALS \
                and "forecast" not in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
            fc_indicators = extra_info.get("forecast_indicators") or {}
            fc_vote, fc_soft_conf = _forecast_dead_zone_vote(df, fc_indicators)
            if fc_vote != "HOLD":
                votes["forecast"] = fc_vote
                extra_info["_soft_conf_forecast"] = fc_soft_conf
                extra_info["forecast_action"] = fc_vote
    else:
        for sig_name in _enhanced_entries:
            votes[sig_name] = "HOLD"

    # MSTR BTC cross-asset proxy (2026-04-29): MSTR is a BTC treasury company
    # (818K BTC, 0.58 price correlation). Its per-ticker consensus accuracy is
    # 47.8% - worst of all Tier 1. Injecting BTC-USD's consensus as a synthetic
    # signal provides cross-asset information the signal system otherwise ignores.
    # The vote goes through all normal gates (accuracy, regime, persistence).
    if ticker == "MSTR":
        with _cross_ticker_lock:
            btc_cons = _cross_ticker_consensus.get("BTC-USD")
        if btc_cons is not None:
            btc_action = btc_cons.get("action", "HOLD")
            if btc_action in ("BUY", "SELL", "HOLD"):
                votes["btc_proxy"] = btc_action
                extra_info["btc_proxy_action"] = btc_action
                extra_info["btc_proxy_confidence"] = btc_cons.get("confidence", 0.0)
                extra_info["btc_proxy_source"] = "cross_ticker_cache"

    # BUG-178 diagnostic phase marker (added 2026-04-10, see docstring above
    # at the __pre_dispatch__ writer). We made it through the enhanced-signal
    # dispatch loop; any remaining slow code is in the post-dispatch accuracy /
    # consensus path. The end-of-cycle diagnostic in main.py uses this marker
    # to identify WHICH phase is slow on cycles > 120 s.
    if ticker:
        _set_last_signal(ticker, "__post_dispatch__")

    # BUG-178 phase timing (added 2026-04-15): _phase_start_*  track wall
    # time at the boundary between named post-dispatch phases so _record_phase()
    # can log per-phase duration. See module-level _phase_log_per_ticker and
    # main.py's slow-cycle diagnostic for how this log is consumed.
    _phase_start = time.monotonic()

    # 2026-04-21: Log per-vote probability distribution for every LLM-family
    # signal. Central hook before any gating logic rewrites votes. The
    # probability log is how we distinguish "confidently wrong" from
    # "uncertainly wrong" for calibration analysis - argmax accuracy alone
    # can't tell shadow models apart. Fire-and-forget: failures never abort
    # the signal cycle.
    try:
        from portfolio.llm_probability_log import (
            derive_probs_from_result,
            llm_signals,
            log_vote,
        )
        for sig_name in llm_signals():
            action = votes.get(sig_name)
            if not action or action not in ("BUY", "HOLD", "SELL"):
                continue
            if sig_name == "sentiment":
                conf = extra_info.get("sentiment_conf", 0.0)
                indicators = {
                    "avg_scores": extra_info.get("sentiment_avg_scores"),
                } if extra_info.get("sentiment_avg_scores") else None
            else:
                conf = extra_info.get(f"{sig_name}_confidence", 0.0)
                indicators = extra_info.get(f"{sig_name}_indicators")
            probs = derive_probs_from_result(
                sig_name, action, conf, indicators=indicators,
            )
            if probs is None:
                continue
            tier = None
            if sig_name == "claude_fundamental" and indicators:
                tier = indicators.get("tier")
            # Default horizon to "1d" when caller passes None so the log is
            # queryable by horizon without null-handling at every join site.
            log_vote(
                sig_name, ticker or "", probs,
                horizon=horizon or "1d", chosen=action, confidence=conf, tier=tier,
            )
    except Exception:
        logger.debug("llm probability logging failed", exc_info=True)

    # C10: Capture raw pre-gate votes BEFORE any gating rewrites them to HOLD.
    # This allows accuracy tracking for regime-gated signals, breaking the
    # dead-signal trap where gated signals can never accumulate accuracy data.
    raw_votes = dict(votes)
    # Merge shadow votes so outcome_tracker can track accuracy for disabled
    # signals that were shadow-computed (math-only, no network I/O).
    raw_votes.update(shadow_votes)
    if shadow_votes:
        extra_info["_shadow_votes"] = shadow_votes

    # 3h horizon: gate slow signals that are noise at short timeframes
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import is_slow_signal_3h
        for sig_name in list(votes.keys()):
            if is_slow_signal_3h(sig_name) and votes[sig_name] != "HOLD":
                votes[sig_name] = "HOLD"

    # BUG-143: Apply regime gating BEFORE computing buy/sell counts so that
    # all downstream code (core gate, min_voters, unanimity penalty) sees
    # post-gated counts.  _weighted_consensus also applies this internally
    # (idempotent - gating HOLDHOLD is a no-op).
    # BUG-149: now horizon-aware via _get_regime_gated()
    # BUG-158: Per-ticker exemption - if a signal has ò60% accuracy with ò50
    # samples on THIS ticker, exempt it from regime gating. fear_greed is 93.8%
    # on XAG-USD but globally gated in ranging - this recovers that alpha.
    regime_gated = _get_regime_gated(regime, horizon)
    _ticker_acc_data = {}
    try:
        from portfolio.accuracy_stats import accuracy_by_ticker_signal_cached
        acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
        _ticker_acc_data = (accuracy_by_ticker_signal_cached(acc_horizon) or {}).get(ticker, {})
    except Exception:
        logger.debug("Per-ticker accuracy unavailable for regime gating exemption", exc_info=True)
    _TICKER_EXEMPT_ACC = 0.60
    _TICKER_EXEMPT_MIN_SAMPLES = 50
    # RES-2026-04-21: Recent-accuracy override for regime gating. When a signal's
    # 7d recent accuracy is significantly above the gate threshold (>55%, 50+ samples),
    # exempt it from regime gating even if all-time data is bad. Prevents stale regime
    # gates from suppressing signals that have recovered in a new market regime.
    # Example: fibonacci went from 43% all-time to 68.2% recent - should not be gated.
    _RECENT_EXEMPT_ACC = 0.55
    _RECENT_EXEMPT_MIN_SAMPLES = 50
    _recent_acc_data: dict[str, Any] = {}
    try:
        from portfolio.accuracy_stats import get_or_compute_recent_accuracy
        # get_or_compute_recent_accuracy expects the base horizon, not the cache key
        base_hz = "3h" if horizon in ("3h", "4h") else "1d"
        _recent_acc_data = get_or_compute_recent_accuracy(base_hz) or {}
    except Exception:
        logger.debug("Recent accuracy unavailable for regime gating override", exc_info=True)

    regime_gated_effective = set(regime_gated)
    for sig_name in list(regime_gated_effective):
        # Per-ticker exemption (BUG-158)
        t_stats = _ticker_acc_data.get(sig_name, {})
        t_acc = t_stats.get("accuracy", 0)
        t_samples = t_stats.get("total", 0)
        if t_samples >= _TICKER_EXEMPT_MIN_SAMPLES and t_acc >= _TICKER_EXEMPT_ACC:
            regime_gated_effective.discard(sig_name)
            logger.debug(
                "BUG-158: %s exempt from %s regime gating for %s (%.1f%%, %d samples)",
                sig_name, regime, ticker, t_acc * 100, t_samples,
            )
            continue
        # Recent-accuracy override (RES-2026-04-21)
        r_stats = _recent_acc_data.get(sig_name, {})
        r_acc = r_stats.get("accuracy", 0)
        r_samples = r_stats.get("total", 0)
        if r_samples >= _RECENT_EXEMPT_MIN_SAMPLES and r_acc >= _RECENT_EXEMPT_ACC:
            regime_gated_effective.discard(sig_name)
            logger.debug(
                "RES-2026-04-21: %s exempt from %s regime gating - recent 7d "
                "accuracy %.1f%% (%d sam) overrides stale gate",
                sig_name, regime, r_acc * 100, r_samples,
            )
    for sig_name in regime_gated_effective:
        if sig_name in votes and votes[sig_name] != "HOLD":
            votes[sig_name] = "HOLD"

    # P1-B (2026-04-17 adversarial review): apply horizon-specific blacklist
    # HERE, before buy/sell counting, so `active_voters` reflects the post-
    # horizon-disable state. Previously this gating only happened inside
    # _weighted_consensus, leaving `extra_info["_voters"]` stale - Stage 4's
    # dynamic_min check could pass a 5-voter count even though only 2 voters
    # actually drove the consensus. _weighted_consensus still applies the
    # same gating internally (idempotent: HOLD->HOLD is a no-op) as defense
    # in depth for callers that bypass generate_signal.
    horizon_disabled_effective = _get_horizon_disabled_signals(ticker, horizon)
    for sig_name in horizon_disabled_effective:
        if sig_name in votes and votes[sig_name] != "HOLD":
            votes[sig_name] = "HOLD"

    # Codex round 2 P1 (2026-04-28): macro-window force-HOLD must mutate
    # `votes` BEFORE buy/sell/core_active are computed below - otherwise
    # those counts come from the pre-mutation state and the gate at
    # line ~3333 ("core_active == 0 ...") sees a stale 1 even when the
    # only core voter (e.g. claude_fundamental) was suppressed by macro.
    # Mirrors the existing regime_gate / horizon_disabled mutation
    # pattern above.
    macro_active_effective = _is_macro_window_cached()
    if macro_active_effective and MACRO_WINDOW_FORCE_HOLD_SIGNALS:
        for sig_name in MACRO_WINDOW_FORCE_HOLD_SIGNALS:
            if sig_name in votes and votes[sig_name] != "HOLD":
                votes[sig_name] = "HOLD"

    if ticker:
        _record_phase(ticker, "regime_gate", _phase_start)
        _phase_start = time.monotonic()

    # Derive buy/sell counts from named votes (post-gating)
    buy = sum(1 for v in votes.values() if v == "BUY")
    sell = sum(1 for v in votes.values() if v == "SELL")

    # Core signal gate: at least 1 core signal must be active for non-HOLD consensus.
    # Enhanced signals can strengthen/weaken a consensus but never create one alone.
    core_buy = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY")
    core_sell = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "SELL")
    core_active = core_buy + core_sell

    # Total applicable signals: computed dynamically from SIGNAL_NAMES
    # minus DISABLED_SIGNALS minus per-asset-class exclusions.
    total_applicable = _compute_applicable_count(ticker, skip_gpu=skip_gpu)

    active_voters = buy + sell
    if ticker in STOCK_SYMBOLS:
        min_voters = MIN_VOTERS_STOCK
    elif ticker in METALS_SYMBOLS:
        # 2026-05-11: metals lowered from MIN_VOTERS_STOCK(3) to
        # MIN_VOTERS_METALS(2). Intraday horizon + persistence filter
        # leaves only 2 voters in steady-state on XAG; the old 3-voter
        # floor produced 0 trades in 20 days.
        min_voters = MIN_VOTERS_METALS
    else:
        min_voters = MIN_VOTERS_CRYPTO

    # Core gate: if no core signal is active, force HOLD regardless of enhanced votes
    if core_active == 0 or active_voters < min_voters:
        action = "HOLD"
        conf = 0.0
    else:
        buy_conf = buy / active_voters
        sell_conf = sell / active_voters
        if buy_conf > sell_conf and buy_conf >= 0.5:
            action = "BUY"
            conf = buy_conf
        elif sell_conf > buy_conf and sell_conf >= 0.5:
            action = "SELL"
            conf = sell_conf
        else:
            action = "HOLD"
            conf = max(buy_conf, sell_conf)

    # Weighted consensus using accuracy data, regime, and activation frequency
    # (regime already computed early in the function for F&G gating)
    accuracy_data = {}
    activation_rates = {}
    # H3: Define acc_horizon before the try/except so the except block and
    # subsequent code can reference it even if the import fails.
    acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"
    try:
        from portfolio.accuracy_stats import (
            blend_accuracy_data,
            get_or_compute_accuracy,
            get_or_compute_per_ticker_accuracy,
            get_or_compute_recent_accuracy,
            load_cached_activation_rates,
        )

        # BUG-178 (2026-04-16): the get_or_compute_* helpers serialize the
        # cache-miss compute via _accuracy_compute_lock so 5 parallel ticker
        # threads don't each pay the 7s+ cost of loading 50,000 signal-log
        # entries when the 1h TTL expires (was 215s wall before the fix).
        # See accuracy_stats.py for the lock rationale.
        alltime = get_or_compute_accuracy(acc_horizon)
        recent = get_or_compute_recent_accuracy(acc_horizon, days=7)
        # BUG-164 lazy-populate per-ticker consensus accuracy - _ptc_key
        # convention preserved by get_or_compute_per_ticker_accuracy.
        _ptc_data = get_or_compute_per_ticker_accuracy(acc_horizon)
        # RES-2026-04-17: Pass per-ticker consensus accuracy into extra_info
        # so apply_confidence_penalties can penalize tickers where the consensus
        # system itself has below-coinflip accuracy (e.g. ETH-USD 47.7% at 3h,
        # MSTR 45.9%). The consensus is the aggregated output, not individual
        # signals - if it's below 50%, the ensemble is net-negative for this ticker.
        if _ptc_data and ticker and isinstance(_ptc_data, dict):
            _ticker_ptc = _ptc_data.get(ticker)
            if isinstance(_ticker_ptc, dict):
                extra_info["_ptc_accuracy"] = _ticker_ptc.get("accuracy")
                extra_info["_ptc_samples"] = _ticker_ptc.get("total", 0)

        # ARCH-23: Use shared blend function (replaces inline logic).
        accuracy_data = blend_accuracy_data(
            alltime, recent,
            divergence_threshold=_RECENCY_DIVERGENCE_THRESHOLD,
            normal_weight=_RECENCY_WEIGHT_NORMAL,
            fast_weight=_RECENCY_WEIGHT_FAST,
            min_recent_samples=_RECENCY_MIN_SAMPLES,
        )

        activation_rates = load_cached_activation_rates()
        _accuracy_failed = False
    except Exception:
        logger.error("Accuracy stats load failed", exc_info=True)
        # H3: Fail-closed: gate all signals (0% accuracy, 999 samples) rather than
        # leaving accuracy_data = {} which bypasses the accuracy gate entirely.
        accuracy_data = {sig: {"accuracy": 0.0, "total": 999} for sig in SIGNAL_NAMES}
        _accuracy_failed = True

    if ticker:
        _record_phase(ticker, "acc_load", _phase_start)
        _phase_start = time.monotonic()

    # Overlay regime-specific accuracy when available.
    # H3: Skip all overlays when primary load failed - they would silently restore
    # real accuracy values for cached signals, negating the fail-closed gate.
    if not _accuracy_failed:
        try:
            from portfolio.accuracy_stats import get_or_compute_regime_accuracy
            # BUG-134: Use acc_horizon (not hardcoded "1d") so regime accuracy
            # matches the prediction horizon (3h/4h/12h/1d).
            # 2026-05-04: switched from manual L2-only dance to L1+L2 wrapper.
            # The previous code re-read disk on every ticker call (~10-50ms
            # JSON parse) and on TTL miss all 5 ticker threads cold-computed
            # in parallel (~30s x 5). The wrapper adds an in-memory L1 with
            # the same dogpile-resistant pattern as signal_utility, dropping
            # 2nd-through-Nth ticker calls per cycle to <1ms. Empty dict on
            # failure preserves the pre-existing fall-through behavior.
            regime_acc = get_or_compute_regime_accuracy(acc_horizon)
            current_regime_data = regime_acc.get(regime, {})
            for sig_name, rdata in current_regime_data.items():
                if rdata.get("total", 0) >= 30:
                    accuracy_data[sig_name] = rdata
        except Exception:
            logger.debug("Regime-conditional accuracy unavailable", exc_info=True)

    # BUG-158: Override global accuracy with per-ticker accuracy for ALL signals.
    # Per-ticker variance is enormous: fear_greed is 93.8% on XAG-USD but 25.9%
    # globally. Using global accuracy throws away alpha on specific instruments.
    # H3: Skip when primary load failed to preserve fail-closed gate.
    _PER_TICKER_MIN_SAMPLES = 30
    if not _accuracy_failed and _ticker_acc_data:
        for sig_name, t_stats in _ticker_acc_data.items():
            if t_stats.get("total", 0) >= _PER_TICKER_MIN_SAMPLES:
                override = {
                    "accuracy": t_stats["accuracy"],
                    "total": t_stats["total"],
                    "correct": t_stats.get("correct", 0),
                    "pct": t_stats.get("pct", round(t_stats["accuracy"] * 100, 1)),
                }
                # Copy directional fields for per-ticker directional gating.
                # Without these, _weighted_consensus directional gate falls back
                # to overall per-ticker accuracy, missing direction-specific
                # weaknesses (e.g., ministral BUY 15% on XAG even if overall 20%).
                for field in ("correct_buy", "total_buy", "buy_accuracy",
                              "correct_sell", "total_sell", "sell_accuracy"):
                    if field in t_stats:
                        override[field] = t_stats[field]
                accuracy_data[sig_name] = override
    elif not _accuracy_failed:
        # Fallback: LLM-specific per-ticker data from extra_info
        for llm_sig in ("qwen3", "ministral"):
            per_ticker_acc = extra_info.get(f"{llm_sig}_accuracy")
            per_ticker_samples = extra_info.get(f"{llm_sig}_samples", 0)
            if per_ticker_acc is not None and per_ticker_samples >= 20:
                accuracy_data[llm_sig] = {
                    "accuracy": per_ticker_acc,
                    "total": per_ticker_samples,
                    "correct": int(per_ticker_acc * per_ticker_samples),
                    "pct": round(per_ticker_acc * 100, 1),
                }

    # Utility boost: scale accuracy weight by return-based utility score.
    # Utility boost and best-horizon overlay.
    # H3: Skip when primary load failed to preserve fail-closed gate.
    if not _accuracy_failed:
        try:
            from portfolio.accuracy_stats import signal_utility
            # BUG-135: Use acc_horizon (not hardcoded "1d") so utility boost
            # reflects the actual prediction horizon's return profile.
            utility_data = signal_utility(acc_horizon)
            for sig_name in list(accuracy_data.keys()):
                u = utility_data.get(sig_name, {})
                u_score = u.get("avg_return", 0.0)
                samples = u.get("samples", 0)
                if samples >= 30 and u_score > 0:
                    boost = min(1.0 + u_score, 1.5)
                    if sig_name in accuracy_data:
                        # BUG-136: Build a new dict instead of mutating in-place.
                        # The accuracy_data may be a reference to cached alltime data.
                        boosted_acc = min(accuracy_data[sig_name]["accuracy"] * boost, 0.95)
                        accuracy_data[sig_name] = {
                            **accuracy_data[sig_name],
                            "accuracy": boosted_acc,
                        }
        except Exception:
            logger.debug("Utility weighting unavailable", exc_info=True)

    if ticker:
        _record_phase(ticker, "utility_overlay", _phase_start)
        _phase_start = time.monotonic()

    # Multi-horizon: optionally use each signal's best horizon accuracy.
    # H3: Skip when primary load failed to preserve fail-closed gate.
    sig_cfg = (config or {}).get("signals", {})
    if not _accuracy_failed and sig_cfg.get("use_best_horizon", False):
        try:
            from portfolio.accuracy_stats import signal_best_horizon_accuracy
            best_hz = signal_best_horizon_accuracy(min_samples=50)
            for sig_name, bh_data in best_hz.items():
                if bh_data.get("total", 0) >= 30:
                    # Only override if best-horizon accuracy is meaningfully better
                    current = accuracy_data.get(sig_name, {}).get("accuracy", 0.5)
                    if bh_data["accuracy"] > current + 0.03:
                        accuracy_data[sig_name] = bh_data
        except Exception:
            logger.debug("Best-horizon accuracy unavailable", exc_info=True)
    accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
    max_signals = sig_cfg.get("max_active_signals")

    # Signal persistence filter: only let signals that maintained their vote
    # for 2+ consecutive cycles participate in consensus. Raw votes are kept
    # intact for accuracy tracking (signal_log records unfiltered votes).
    consensus_votes = _apply_persistence_filter(votes, ticker)
    # Track how many signals were filtered for debugging
    _filtered_count = sum(
        1 for s in votes
        if votes[s] != "HOLD" and consensus_votes.get(s) == "HOLD"
    )
    if _filtered_count > 0:
        extra_info["_persistence_filtered"] = _filtered_count

    # Macro-window force-HOLD has already been applied to `votes` above
    # (before buy/sell/core_active counting), so `consensus_votes` -
    # which derives from `votes` via persistence filter - already
    # carries the suppression. No additional mutation needed here.
    # `_weighted_consensus` runs its own pre-pass as defense-in-depth.

    # BUG-224: compute post-persistence voter count so downstream consumers
    # (accuracy tracking, Layer 2) see the actual participating voter count,
    # not the inflated pre-filter number.
    post_persistence_voters = sum(
        1 for v in consensus_votes.values() if v in ("BUY", "SELL")
    )

    weighted_action, weighted_conf = _weighted_consensus(
        consensus_votes, accuracy_data, regime, activation_rates,
        accuracy_gate=accuracy_gate,
        max_signals=max_signals,
        horizon=horizon,
        regime_gated_override=regime_gated_effective,
        ticker=ticker,
        # 2026-05-11 (Codex Fix B): pass extra_info so _weighted_consensus
        # can dampen ema/bb/macd soft votes by their _soft_conf_* values.
        soft_confidences=extra_info,
    )

    if ticker:
        _record_phase(ticker, "weighted_consensus", _phase_start)
        _phase_start = time.monotonic()

    # BUG-227: Apply core gate AND MIN_VOTERS gate to weighted consensus.
    # Use post_persistence_voters (not pre-filter active_voters) because the
    # persistence filter may have reduced voters below the threshold.
    if core_active == 0 or post_persistence_voters < min_voters:
        weighted_action = "HOLD"
        weighted_conf = 0.0

    # Confluence score
    confluence = _confluence_score(votes, extra_info)

    # Time-of-day confidence adjustment
    tod_factor = _time_of_day_factor(horizon=horizon)
    weighted_conf *= tod_factor

    # Store raw consensus in extra for debugging, then use weighted as primary
    extra_info["_raw_action"] = action
    extra_info["_raw_confidence"] = conf
    extra_info["_voters"] = active_voters  # pre-filter (compatibility)
    extra_info["_voters_post_filter"] = post_persistence_voters
    extra_info["_total_applicable"] = total_applicable
    extra_info["_buy_count"] = buy
    extra_info["_sell_count"] = sell
    extra_info["_core_buy"] = core_buy
    extra_info["_core_sell"] = core_sell
    extra_info["_core_active"] = core_active
    extra_info["_votes"] = votes
    extra_info["_raw_votes"] = raw_votes  # C10: pre-gate votes for accuracy recovery
    extra_info["_regime"] = regime
    if horizon:
        extra_info["_horizon"] = horizon
    extra_info["_weighted_action"] = weighted_action
    extra_info["_weighted_confidence"] = weighted_conf
    extra_info["_confluence_score"] = confluence

    # Primary action = weighted consensus (accounts for accuracy + bias penalties)
    action = weighted_action
    conf = weighted_conf

    # Apply confidence penalty cascade (regime, volume/ADX, trap, dynamic min_voters)
    action, conf, penalty_log = apply_confidence_penalties(
        action, conf, regime, ind, extra_info, ticker, df, config
    )
    if penalty_log:
        extra_info["_penalty_log"] = penalty_log

    if ticker:
        _record_phase(ticker, "penalties", _phase_start)
        _phase_start = time.monotonic()

    # --- Market health confidence penalty ---
    # Penalizes BUY signals when broad market is unhealthy (distribution days,
    # broken FTD, etc.).  Only affects BUY; SELL and HOLD pass through.
    try:
        from portfolio.market_health import get_confidence_penalty, get_market_health
        mh = get_market_health()
        mh_mult = get_confidence_penalty(action, mh)
        if mh_mult != 1.0:
            conf *= mh_mult
            extra_info.setdefault("_penalty_log", []).append(
                {"stage": "market_health", "zone": mh.get("zone") if mh else "unknown",
                 "score": mh.get("score") if mh else None, "mult": mh_mult}
            )
    except Exception:
        logger.debug("Market health penalty failed", exc_info=True)

    # --- Earnings proximity gate (stocks only) ---
    # Force HOLD if ticker has earnings within 2 calendar days.
    if ticker and action != "HOLD":
        try:
            from portfolio.earnings_calendar import should_gate_earnings
            if should_gate_earnings(ticker):
                extra_info.setdefault("_penalty_log", []).append(
                    {"stage": "earnings_gate", "ticker": ticker, "effect": "force_hold"}
                )
                extra_info["_earnings_gated"] = True
                action = "HOLD"
                conf = 0.0
        except Exception:
            logger.debug("Earnings gate failed for %s", ticker, exc_info=True)

    # --- Linear factor model score (supplementary, not overriding) ---
    # Provides a continuous predicted-return score from ridge regression
    # weights trained on historical signal+return data. Used to:
    # 1. Confirm or weaken consensus (agreement = boost, disagreement = dampen)
    # 2. Provide alternative ranking in agent_summary for Layer 2 decisions
    try:
        from portfolio.linear_factor import LinearFactorModel
        _lf_model = LinearFactorModel()
        if _lf_model.load():
            # Convert votes to numeric: BUY=+1, SELL=-1, HOLD=0
            _vote_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
            numeric_votes = {k: _vote_map.get(v, 0.0) for k, v in votes.items()}
            lf_score = _lf_model.predict(numeric_votes)
            lf_action, lf_conf = _lf_model.score_to_action(lf_score)
            extra_info["_linear_factor_score"] = round(lf_score, 6)
            extra_info["_linear_factor_action"] = lf_action
            extra_info["_linear_factor_confidence"] = lf_conf
            # Confirmation boost/dampen: if linear model agrees with consensus,
            # boost confidence by 10%. If it disagrees, dampen by 10%.
            if lf_action == action and action != "HOLD" and lf_conf > 0.3:
                conf *= 1.10
                extra_info.setdefault("_penalty_log", []).append(
                    {"stage": "linear_factor", "effect": "confirm_boost",
                     "lf_action": lf_action, "lf_score": round(lf_score, 6)})
            elif lf_action != "HOLD" and lf_action != action and action != "HOLD":
                conf *= 0.90
                extra_info.setdefault("_penalty_log", []).append(
                    {"stage": "linear_factor", "effect": "disagree_dampen",
                     "lf_action": lf_action, "lf_score": round(lf_score, 6)})
    except Exception:
        logger.debug("Linear factor model failed", exc_info=True)

    if ticker:
        _record_phase(ticker, "linear_factor", _phase_start)
        _phase_start = time.monotonic()

    # --- Per-ticker consensus accuracy gate ---
    # BUG-164: AMD 24.8%, GOOGL 31.3%, META 34.2% consensus accuracy.
    # The system is actively harmful on these tickers.  Gate consensus
    # when per-ticker historical accuracy is below threshold.
    if ticker and action != "HOLD":
        try:
            from portfolio.accuracy_stats import load_cached_accuracy

            # H1: Match the horizon-scoped key written above.
            _ptc_acc = load_cached_accuracy(f"per_ticker_consensus_{acc_horizon}")
            if _ptc_acc:
                _ptc_stats = _ptc_acc.get(ticker, {})
                _ptc_total = _ptc_stats.get("total", 0)
                _ptc_accuracy = _ptc_stats.get("accuracy", 0.5)
                if _ptc_total >= _PER_TICKER_CONSENSUS_MIN_SAMPLES and _ptc_accuracy < _PER_TICKER_CONSENSUS_GATE:
                    extra_info.setdefault("_penalty_log", []).append({
                        "stage": "per_ticker_consensus_gate",
                        "ticker": ticker,
                        "accuracy": round(_ptc_accuracy, 3),
                        "samples": _ptc_total,
                        "threshold": _PER_TICKER_CONSENSUS_GATE,
                        "effect": "force_hold",
                    })
                    extra_info["_ticker_consensus_gated"] = True
                    action = "HOLD"
                    conf = 0.0
        except Exception:
            logger.debug("Per-ticker consensus gate failed for %s", ticker, exc_info=True)

    # Global confidence cap - calibration data shows >80% confidence is
    # anti-correlated with accuracy at every horizon (70-80% bucket is the
    # best performing at 57-59% actual accuracy)
    conf = min(conf, 0.80)

    # 3h horizon: cap confidence to prevent overconfident short-term predictions
    if horizon in ("3h", "4h"):
        from portfolio.short_horizon import CONFIDENCE_CAP_3H
        conf = min(conf, CONFIDENCE_CAP_3H)

    if ticker:
        _record_phase(ticker, "consensus_gate", _phase_start)

    # Update cross-ticker consensus cache for synthetic cross-asset signals
    if ticker:
        with _cross_ticker_lock:
            _cross_ticker_consensus[ticker] = {"action": action, "confidence": conf}

    return action, conf, extra_info
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
The engine is returning `(action, conf, extra_info)` after reading detector result dicts, so I’m checking registry dispatch now to pin down the exact detector contract and catch modules whose signatures or return shapes drift from what the engine calls.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content -Path Q:\\finance-analyzer\\portfolio\\signal_registry.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"compute_.*signal|load_signal_func|get_enhanced_signals|inspect.signature|sub_signals|indicators|module\" Q:\\finance-analyzer\\portfolio\\signal_registry.py Q:\\finance-analyzer\\portfolio\\signal_engine.py" in Q:\finance-analyzer
 succeeded in 676ms:
Q:\finance-analyzer\portfolio\signal_registry.py:17:                    module_path: str = None, func_name: str = None,
Q:\finance-analyzer\portfolio\signal_registry.py:28:            "module_path": module_path or func.__module__,
Q:\finance-analyzer\portfolio\signal_registry.py:38:def register_enhanced(name: str, module_path: str, func_name: str,
Q:\finance-analyzer\portfolio\signal_registry.py:42:    """Programmatically register an enhanced signal module."""
Q:\finance-analyzer\portfolio\signal_registry.py:46:        "module_path": module_path,
Q:\finance-analyzer\portfolio\signal_registry.py:55:def get_enhanced_signals() -> dict[str, dict]:
Q:\finance-analyzer\portfolio\signal_registry.py:68:def load_signal_func(entry: dict) -> Callable | None:
Q:\finance-analyzer\portfolio\signal_registry.py:80:        mod = importlib.import_module(entry["module_path"])
Q:\finance-analyzer\portfolio\signal_registry.py:94:    """Register the default set of enhanced signal modules."""
Q:\finance-analyzer\portfolio\signal_registry.py:96:        ("trend", "portfolio.signals.trend", "compute_trend_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:97:        ("momentum", "portfolio.signals.momentum", "compute_momentum_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:98:        ("volume_flow", "portfolio.signals.volume_flow", "compute_volume_flow_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:99:        ("volatility_sig", "portfolio.signals.volatility", "compute_volatility_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:100:        ("candlestick", "portfolio.signals.candlestick", "compute_candlestick_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:101:        ("structure", "portfolio.signals.structure", "compute_structure_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:102:        ("fibonacci", "portfolio.signals.fibonacci", "compute_fibonacci_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:103:        ("smart_money", "portfolio.signals.smart_money", "compute_smart_money_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:104:        ("oscillators", "portfolio.signals.oscillators", "compute_oscillator_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:105:        ("heikin_ashi", "portfolio.signals.heikin_ashi", "compute_heikin_ashi_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:106:        ("calendar", "portfolio.signals.calendar_seasonal", "compute_calendar_signal"),
Q:\finance-analyzer\portfolio\signal_registry.py:112:                      "compute_mean_reversion_signal", requires_context=True)
Q:\finance-analyzer\portfolio\signal_registry.py:114:                      "compute_momentum_factors_signal", requires_context=True)
Q:\finance-analyzer\portfolio\signal_registry.py:117:                      "compute_macro_regime_signal", requires_macro=True)
Q:\finance-analyzer\portfolio\signal_registry.py:120:                      "compute_news_event_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:122:                      "compute_econ_calendar_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:125:                      "compute_forecast_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:128:                      "compute_claude_fundamental_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:131:                      "compute_futures_flow_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:134:                      "compute_crypto_macro_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:137:                      "compute_orderbook_flow_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:140:                      "compute_metals_cross_asset_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:145:                      "compute_dxy_cross_asset_signal", requires_context=True, max_confidence=0.8)
Q:\finance-analyzer\portfolio\signal_registry.py:148:                      "compute_cot_positioning_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:151:                      "compute_credit_spread_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:154:                      "compute_futures_basis_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:157:                      "compute_hurst_regime_signal", requires_context=True)
Q:\finance-analyzer\portfolio\signal_registry.py:160:                      "compute_shannon_entropy_signal")
Q:\finance-analyzer\portfolio\signal_registry.py:163:                      "compute_vix_term_structure_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:166:                      "compute_gold_real_yield_paradox_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:169:                      "compute_cross_asset_tsmom_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:174:                      "compute_copper_gold_ratio_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:179:                      "compute_statistical_jump_regime_signal")
Q:\finance-analyzer\portfolio\signal_registry.py:184:                      "compute_network_momentum_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:190:                      "compute_ovx_metals_spillover_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:195:                      "compute_xtrend_equity_spillover_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:200:                      "compute_complexity_gap_regime_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:205:                      "compute_realized_skewness_signal")
Q:\finance-analyzer\portfolio\signal_registry.py:210:                      "compute_mahalanobis_turbulence_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:216:                      "compute_crypto_evrp_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:222:                      "compute_hash_ribbons_signal", requires_context=True, max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:228:                      "compute_drift_regime_gate_signal", max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:234:                      "compute_vol_ratio_regime_signal", max_confidence=0.7)
Q:\finance-analyzer\portfolio\signal_registry.py:241:                      "compute_residual_pair_reversion_signal",
Q:\finance-analyzer\portfolio\signal_registry.py:250:                      "compute_williams_vix_fix_signal",
Q:\finance-analyzer\portfolio\signal_registry.py:258:                      "compute_treasury_risk_rotation_signal",
Q:\finance-analyzer\portfolio\signal_registry.py:266:                      "compute_intraday_seasonality_signal",
Q:\finance-analyzer\portfolio\signal_registry.py:273:                      "compute_cubic_trend_persistence_signal",
Q:\finance-analyzer\portfolio\signal_registry.py:281:                      "compute_vwap_zscore_mr_signal",
Q:\finance-analyzer\portfolio\signal_engine.py:13:from portfolio.indicators import detect_regime
Q:\finance-analyzer\portfolio\signal_engine.py:24:from portfolio.signal_registry import get_enhanced_signals, load_signal_func
Q:\finance-analyzer\portfolio\signal_engine.py:223:    directional votes. See module-level rationale comment above.
Q:\finance-analyzer\portfolio\signal_engine.py:254:    directional votes. See module-level rationale comment above.
Q:\finance-analyzer\portfolio\signal_engine.py:287:    directional votes. See module-level rationale comment above.
Q:\finance-analyzer\portfolio\signal_engine.py:345:    `compute_candlestick_signal` strong path abstains ~87% of the time
Q:\finance-analyzer\portfolio\signal_engine.py:370:def _forecast_dead_zone_vote(df, forecast_indicators,
Q:\finance-analyzer\portfolio\signal_engine.py:379:    actually produced data. If `forecast_indicators` is empty / lacks
Q:\finance-analyzer\portfolio\signal_engine.py:393:    indicators = forecast_indicators or {}
Q:\finance-analyzer\portfolio\signal_engine.py:396:    if indicators.get("models_disabled"):
Q:\finance-analyzer\portfolio\signal_engine.py:398:    if indicators.get("error"):
Q:\finance-analyzer\portfolio\signal_engine.py:403:    if indicators.get("chronos_1h_pct") is None:
Q:\finance-analyzer\portfolio\signal_engine.py:405:        if not indicators.get("chronos_ok"):
Q:\finance-analyzer\portfolio\signal_engine.py:550:    # Local import-style references — these names are defined at module scope
Q:\finance-analyzer\portfolio\signal_engine.py:868:# P2-H (2026-04-17): module-load validation of _TICKER_DISABLED_BY_HORIZON
Q:\finance-analyzer\portfolio\signal_engine.py:892:    module-load assertion prevents it — but defensive), we return an empty
Q:\finance-analyzer\portfolio\signal_engine.py:967:# P2-G (2026-04-17): module-load assertions on constant relationships.
Q:\finance-analyzer\portfolio\signal_engine.py:1474:    and sub_signals is a dict. Returns a clean dict, always.
Q:\finance-analyzer\portfolio\signal_engine.py:1477:        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}}
Q:\finance-analyzer\portfolio\signal_engine.py:1496:    sub_signals = result.get("sub_signals")
Q:\finance-analyzer\portfolio\signal_engine.py:1497:    if not isinstance(sub_signals, dict):
Q:\finance-analyzer\portfolio\signal_engine.py:1498:        sub_signals = {}
Q:\finance-analyzer\portfolio\signal_engine.py:1503:        "sub_signals": sub_signals,
Q:\finance-analyzer\portfolio\signal_engine.py:1504:        "indicators": result.get("indicators") or {},
Q:\finance-analyzer\portfolio\signal_engine.py:2107:    they are noise, not useful contrarian indicators.
Q:\finance-analyzer\portfolio\signal_engine.py:2374:    # crisis is in the macro indicators, not in the trend — penalizing trend
Q:\finance-analyzer\portfolio\signal_engine.py:2621:def _confluence_score(votes, indicators):
Q:\finance-analyzer\portfolio\signal_engine.py:2629:    if indicators.get("volume_action") in ("BUY", "SELL"):
Q:\finance-analyzer\portfolio\signal_engine.py:2630:        vol_dir = indicators.get("volume_action")
Q:\finance-analyzer\portfolio\signal_engine.py:2673:                ti = entry.get("indicators", {})
Q:\finance-analyzer\portfolio\signal_engine.py:3160:        logger.debug("Optional module %s not available", "fear_greed")
Q:\finance-analyzer\portfolio\signal_engine.py:3178:            logger.debug("Optional module %s not available", "social_sentiment")
Q:\finance-analyzer\portfolio\signal_engine.py:3237:            logger.debug("Optional module %s not available", "sentiment")
Q:\finance-analyzer\portfolio\signal_engine.py:3259:            logger.debug("Optional module %s not available", "funding_rate")
Q:\finance-analyzer\portfolio\signal_engine.py:3321:            logger.debug("Optional module %s not available", "onchain_data")
Q:\finance-analyzer\portfolio\signal_engine.py:3335:            logger.debug("Optional module %s not available", "macro_context")
Q:\finance-analyzer\portfolio\signal_engine.py:3382:            logger.debug("Optional module %s not available", "ministral_signal")
Q:\finance-analyzer\portfolio\signal_engine.py:3429:            logger.debug("Optional module %s not available", "qwen3_signal")
Q:\finance-analyzer\portfolio\signal_engine.py:3431:    # --- Enhanced signal modules (composite indicators computed from raw OHLCV) ---
Q:\finance-analyzer\portfolio\signal_engine.py:3433:    _enhanced_entries = get_enhanced_signals()
Q:\finance-analyzer\portfolio\signal_engine.py:3492:                        compute_fn = load_signal_func(entry)
Q:\finance-analyzer\portfolio\signal_engine.py:3524:                compute_fn = load_signal_func(entry)
Q:\finance-analyzer\portfolio\signal_engine.py:3548:                extra_info[f"{sig_name}_sub_signals"] = validated["sub_signals"]
Q:\finance-analyzer\portfolio\signal_engine.py:3549:                if validated["indicators"]:
Q:\finance-analyzer\portfolio\signal_engine.py:3550:                    extra_info[f"{sig_name}_indicators"] = validated["indicators"]
Q:\finance-analyzer\portfolio\signal_engine.py:3594:            fc_indicators = extra_info.get("forecast_indicators") or {}
Q:\finance-analyzer\portfolio\signal_engine.py:3595:            fc_vote, fc_soft_conf = _forecast_dead_zone_vote(df, fc_indicators)
Q:\finance-analyzer\portfolio\signal_engine.py:3630:    # can log per-phase duration. See module-level _phase_log_per_ticker and
Q:\finance-analyzer\portfolio\signal_engine.py:3652:                indicators = {
Q:\finance-analyzer\portfolio\signal_engine.py:3657:                indicators = extra_info.get(f"{sig_name}_indicators")
Q:\finance-analyzer\portfolio\signal_engine.py:3659:                sig_name, action, conf, indicators=indicators,
Q:\finance-analyzer\portfolio\signal_engine.py:3664:            if sig_name == "claude_fundamental" and indicators:
Q:\finance-analyzer\portfolio\signal_engine.py:3665:                tier = indicators.get("tier")
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 695ms:
"""Signal registry - plugin system for modular signal management.

Enhanced signals register via register_enhanced(). signal_engine.py
discovers all signals from the registry instead of hardcoded lists.
"""
import importlib
import logging
import time
from collections.abc import Callable

logger = logging.getLogger("portfolio.signal_registry")

_ENHANCED_SIGNALS: dict[str, dict] = {}


def register_signal(name: str, signal_type: str = "enhanced",
                    module_path: str = None, func_name: str = None,
                    requires_macro: bool = False):
    """Register a signal in the global registry.

    Can be used as a decorator on compute functions, or called directly
    to register signals programmatically.
    """
    def decorator(func):
        entry = {
            "name": name,
            "type": signal_type,
            "module_path": module_path or func.__module__,
            "func_name": func_name or func.__name__,
            "requires_macro": requires_macro,
            "func": func,
        }
        _ENHANCED_SIGNALS[name] = entry
        return func
    return decorator


def register_enhanced(name: str, module_path: str, func_name: str,
                      requires_macro: bool = False,
                      requires_context: bool = False,
                      max_confidence: float = 1.0):
    """Programmatically register an enhanced signal module."""
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
    so the warning is logged once, not 35x per cycle (5 tickers x 7 TFs).
    """
    cached = entry.get("func")
    if cached is not None and cached is not _FAILED_IMPORT_SENTINEL:
        return cached
    if cached is _FAILED_IMPORT_SENTINEL and time.monotonic() - entry.get("_fail_ts", 0) < _FAILED_IMPORT_COOLDOWN:
        return None
    try:
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
    # macro_regime is special - requires_macro=True
    register_enhanced("macro_regime", "portfolio.signals.macro_regime",
                      "compute_macro_regime_signal", requires_macro=True)
    # news_event and econ_calendar require context (ticker, config); capped at 0.7
    register_enhanced("news_event", "portfolio.signals.news_event",
                      "compute_news_event_signal", requires_context=True, max_confidence=0.7)
    register_enhanced("econ_calendar", "portfolio.signals.econ_calendar",
                      "compute_econ_calendar_signal", requires_context=True, max_confidence=0.7)
    # forecast signal - Kronos + Chronos price direction prediction; capped at 0.7
    register_enhanced("forecast", "portfolio.signals.forecast",
                      "compute_forecast_signal", requires_context=True, max_confidence=0.7)
    # Claude fundamental - three-tier LLM cascade; capped at 0.7
    register_enhanced("claude_fundamental", "portfolio.signals.claude_fundamental",
                      "compute_claude_fundamental_signal", requires_context=True, max_confidence=0.7)
    # Futures flow - OI, LS ratios, funding history (crypto only); capped at 0.7
    register_enhanced("futures_flow", "portfolio.signals.futures_flow",
                      "compute_futures_flow_signal", requires_context=True, max_confidence=0.7)
    # Crypto macro - options max pain, gold-BTC rotation, exchange reserves (crypto only); capped at 0.7
    register_enhanced("crypto_macro", "portfolio.signals.crypto_macro",
                      "compute_crypto_macro_signal", requires_context=True, max_confidence=0.7)
    # Orderbook flow - microstructure metrics (metals + crypto); capped at 0.7
    register_enhanced("orderbook_flow", "portfolio.signals.orderbook_flow",
                      "compute_orderbook_flow_signal", requires_context=True, max_confidence=0.7)
    # Metals cross-asset - copper, GVZ, G/S ratio, SPY, oil (metals only); capped at 0.7
    register_enhanced("metals_cross_asset", "portfolio.signals.metals_cross_asset",
                      "compute_metals_cross_asset_signal", requires_context=True, max_confidence=0.7)
    # DXY cross-asset - intraday USD index inverse correlation (metals only); capped at 0.8
    # 2026-04-13: added standalone signal to capture DXY Rı~0.6 vs silver at
    # 1-3h horizon. Complements macro_regime's daily DXY sub-indicator.
    register_enhanced("dxy_cross_asset", "portfolio.signals.dxy_cross_asset",
                      "compute_dxy_cross_asset_signal", requires_context=True, max_confidence=0.8)
    # COT positioning - CFTC speculative/commercial positioning, contrarian (metals only); capped at 0.7
    register_enhanced("cot_positioning", "portfolio.signals.cot_positioning",
                      "compute_cot_positioning_signal", requires_context=True, max_confidence=0.7)
    # Credit spread risk - HY OAS from FRED as cross-asset risk appetite gauge; capped at 0.7
    register_enhanced("credit_spread_risk", "portfolio.signals.credit_spread",
                      "compute_credit_spread_signal", requires_context=True, max_confidence=0.7)
    # Futures basis regime - mark-index spread, contango/backwardation detection; capped at 0.7
    register_enhanced("futures_basis", "portfolio.signals.futures_basis",
                      "compute_futures_basis_signal", requires_context=True, max_confidence=0.7)
    # Hurst regime detector - R/S analysis for trending/MR/random-walk classification
    register_enhanced("hurst_regime", "portfolio.signals.hurst_regime",
                      "compute_hurst_regime_signal", requires_context=True)
    # Shannon entropy - market noise/predictability filter; low entropy = trending
    register_enhanced("shannon_entropy", "portfolio.signals.shannon_entropy",
                      "compute_shannon_entropy_signal")
    # VIX term structure - contango/backwardation regime detection; capped at 0.7
    register_enhanced("vix_term_structure", "portfolio.signals.vix_term_structure",
                      "compute_vix_term_structure_signal", requires_context=True, max_confidence=0.7)
    # Gold real yield paradox - GYDI regime detector (metals only); capped at 0.7
    register_enhanced("gold_real_yield_paradox", "portfolio.signals.gold_real_yield_paradox",
                      "compute_gold_real_yield_paradox_signal", requires_context=True, max_confidence=0.7)
    # Cross-asset TSMOM - bond/equity momentum predicts target asset direction; capped at 0.7
    register_enhanced("cross_asset_tsmom", "portfolio.signals.cross_asset_tsmom",
                      "compute_cross_asset_tsmom_signal", requires_context=True, max_confidence=0.7)
    # Copper/gold ratio - intermarket regime indicator (all assets); capped at 0.7
    # 2026-04-17: cross-asset signal, inverts direction for metals (falling ratio = gold strength).
    # 94% recession prediction accuracy, 0.85 correlation with 10Y yields.
    register_enhanced("copper_gold_ratio", "portfolio.signals.copper_gold_ratio",
                      "compute_copper_gold_ratio_signal", requires_context=True, max_confidence=0.7)
    # Statistical Jump Model regime - jump detection + persistence penalty (all assets)
    # 2026-04-18: addresses failing macro_regime (46.6% at 1d, 30.3% metals).
    # Source: Shu, Yu, Mulvey 2024 (12 citations). Persistence penalty prevents whiplash.
    register_enhanced("statistical_jump_regime", "portfolio.signals.statistical_jump_regime",
                      "compute_statistical_jump_regime_signal")
    # Network momentum - cross-asset momentum spillover (all assets); capped at 0.7
    # 2026-04-19: Pu et al. 2023 (arXiv:2308.11294). Sharpe 1.511 across 64 futures.
    # Simplified: correlation-weighted peer momentum divergence instead of GNN.
    register_enhanced("network_momentum", "portfolio.signals.network_momentum",
                      "compute_network_momentum_signal", requires_context=True, max_confidence=0.7)
    # OVX metals spillover - oil implied volatility as metals predictor; capped at 0.7
    # 2026-04-20: OVX at extreme quantiles predicts precious metals returns via
    # contagion/inflation/dollar channels. Distinct from metals_cross_asset (oil PRICE).
    # Source: ScienceDirect OVX cross-asset quantile predictability papers.
    register_enhanced("ovx_metals_spillover", "portfolio.signals.ovx_metals_spillover",
                      "compute_ovx_metals_spillover_signal", requires_context=True, max_confidence=0.7)
    # Cross-asset equity trend spillover - SPY/QQQ TA predicts all assets; capped at 0.7
    # 2026-04-21: Fieberg et al. 2025, robust across 1.3M research designs.
    # Inverts for safe havens (XAU, XAG): bullish equities = risk-on = SELL metals.
    register_enhanced("xtrend_equity_spillover", "portfolio.signals.xtrend_equity_spillover",
                      "compute_xtrend_equity_spillover_signal", requires_context=True, max_confidence=0.7)
    # Complexity gap regime - RMT-based market structure/synchronization detector; capped at 0.7
    # 2026-04-22: Mukhia et al. 2026, arXiv:2604.19107. Complexity gap = norm_max_eigenvalue -
    # avg_pairwise_corr. Gap collapse = crisis synchronization. Inverts for safe havens.
    register_enhanced("complexity_gap_regime", "portfolio.signals.complexity_gap_regime",
                      "compute_complexity_gap_regime_signal", requires_context=True, max_confidence=0.7)
    # Realized skewness - 3rd moment directional signal (all assets)
    # 2026-04-23: Fernandez-Perez et al. 2018. Sharpe 0.79, 8.01% annual on
    # 27 commodity futures. Z-scored skewness + kurtosis confirmation.
    register_enhanced("realized_skewness", "portfolio.signals.realized_skewness",
                      "compute_realized_skewness_signal")
    # Mahalanobis turbulence - cross-asset regime detection via Mahalanobis distance
    # 2026-04-24: Kritzman & Li (2010). Sharpe 2.20 vs 1.0 B&H, max DD 6% vs 32%.
    # Measures statistical unusualness of multi-asset returns. Includes absorption ratio.
    register_enhanced("mahalanobis_turbulence", "portfolio.signals.mahalanobis_turbulence",
                      "compute_mahalanobis_turbulence_signal", requires_context=True, max_confidence=0.7)
    # Crypto eVRP - Expected Volatility Risk Premium (crypto only); capped at 0.7
    # 2026-04-25: Zarattini, Mele & Aziz (2025). eVRP = DVOL(30d) - RV(10d).
    # Options-derived signal uncorrelated with trend-following cluster.
    # Deribit public API, no auth. BTC + ETH only.
    register_enhanced("crypto_evrp", "portfolio.signals.crypto_evrp",
                      "compute_crypto_evrp_signal", requires_context=True, max_confidence=0.7)
    # Hash Ribbons BTC - miner capitulation detector (BTC-only); capped at 0.7
    # 2026-04-26: Charles Edwards / Capriole Investments. 89% win rate on 9 signals.
    # 30d/60d SMA hashrate crossover with price momentum confirmation.
    # blockchain.info API (free, no auth). Fires ~1/year but extreme conviction.
    register_enhanced("hash_ribbons", "portfolio.signals.hash_ribbons",
                      "compute_hash_ribbons_signal", requires_context=True, max_confidence=0.7)
    # Drift Regime Gate - positive-day fraction regime detector; capped at 0.7
    # 2026-04-28: arxiv:2511.12490 (2025). OOS Sharpe >13 on 20yr walk-forward.
    # Fraction of positive close-to-close days in 63-bar window detects drift regime.
    # Pure OHLCV, all 5 tickers. Directional via SMA distance.
    register_enhanced("drift_regime_gate", "portfolio.signals.drift_regime_gate",
                      "compute_drift_regime_gate_signal", max_confidence=0.7)
    # Vol ratio regime - GK/CC volatility ratio + VR test + ER as regime detector
    # 2026-04-29: Garman & Klass (1980), Lo & MacKinlay (1988), Kaufman ER.
    # Three orthogonal regime measures: GK/CC ratio, variance ratio, efficiency ratio.
    # Directional: mean-reversion in ranging, momentum in trending. All OHLCV, all assets.
    register_enhanced("vol_ratio_regime", "portfolio.signals.vol_ratio_regime",
                      "compute_vol_ratio_regime_signal", max_confidence=0.7)
    # Residual pair reversion - cointegration-based pairs trading (all assets)
    # 2026-04-30: Leung & Nguyen (2018), Amberdata crypto pairs (Sharpe 0.93).
    # Regime-neutral mean reversion on OLS residual: ETH~BTC, XAG~XAU, MSTR~BTC.
    # Specifically targets ETH-USD (weakest ticker) and XAG-USD (user focus).
    register_enhanced("residual_pair_reversion",
                      "portfolio.signals.residual_pair_reversion",
                      "compute_residual_pair_reversion_signal",
                      requires_context=True, max_confidence=0.7)
    # Williams VIX Fix - synthetic volatility bottom/capitulation detector (all assets)
    # 2026-05-01: Larry Williams (2007). Profit factor 2.0, 322 trades 1993-2024.
    # WVF = (highest_close_22 - low) / highest_close_22 * 100. Spikes above upper
    # Bollinger Band signal extreme fear = BUY. Directly targets system's BUY-side
    # accuracy weakness. Pure OHLCV, all 5 tickers.
    register_enhanced("williams_vix_fix",
                      "portfolio.signals.williams_vix_fix",
                      "compute_williams_vix_fix_signal",
                      max_confidence=0.7)
    # Treasury risk rotation - bond yield curve shape as cross-asset regime signal
    # 2026-05-07: Gayed (2014), SSRN 2431022. IEF vs TLT relative performance.
    # Steepening = risk-on, flattening = risk-off. Inverted for safe havens.
    # Only signal using bond market data - zero correlation with existing clusters.
    register_enhanced("treasury_risk_rotation",
                      "portfolio.signals.treasury_risk_rotation",
                      "compute_treasury_risk_rotation_signal",
                      requires_context=True, max_confidence=0.7)
    # Intraday seasonality gate - hour-of-day confidence multipliers (all assets)
    # 2026-05-08: Concretum Group 2025 (Sharpe 1.6 BTC Asia-open), ScienceDirect 2024
    # (33% annualized BTC 22:00-23:00 UTC), CME Group 2026 (60-70% gold daily range
    # in London-NY overlap). Zero correlation with existing signal clusters.
    register_enhanced("intraday_seasonality",
                      "portfolio.signals.intraday_seasonality",
                      "compute_intraday_seasonality_signal",
                      requires_context=True, max_confidence=0.7)
    # Cubic trend persistence - R(t+1) = b*phi + c*phi^3 (all assets)
    # 2026-05-09: arXiv:2501.16772 (2025). Universal across equities, bonds,
    # currencies, commodities. 330yr data. Weak trends persist, strong revert.
    register_enhanced("cubic_trend_persistence",
                      "portfolio.signals.cubic_trend_persistence",
                      "compute_cubic_trend_persistence_signal",
                      max_confidence=0.7)
    # VWAP Z-Score MR - volume-weighted mean reversion (all assets)
    # 2026-05-10: FMZ VWAP StdDev MR (77.78% win rate). Different from BB
    # (volume-weighted anchor vs SMA anchor). Three sub-signals: vwap_z,
    # vwap_slope, volume_confirm. Pure OHLCV, all assets.
    register_enhanced("vwap_zscore_mr",
                      "portfolio.signals.vwap_zscore_mr",
                      "compute_vwap_zscore_mr_signal",
                      max_confidence=0.85)


_register_defaults()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:28:52.503625Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'; (Get-Content "'$p | Select-Object -Skip 1460 -First 60)'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'; (Get-Content "'$p | Select-Object -Skip 1460 -First 60)' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'; (Get-Content "'$p | Select-Object -Skip 3470 -First 100)' in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"requires_context|requires_macro|validate.*signal|_validate\" Q:\\finance-analyzer\\portfolio\\signal_engine.py" in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'; (Get-Content "'$p | Select-Object -Skip 1460 -First 60)'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'; (Get-Content "'$p | Select-Object -Skip 3470 -First 100)'` rejected: blocked by policy
2026-05-11T15:28:52.503977Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'Q:\\finance-analyzer\\portfolio\\signal_engine.py'; (Get-Content "'$p | Select-Object -Skip 3470 -First 100)'` rejected: blocked by policy
 succeeded in 650ms:
1470:def _validate_signal_result(result, sig_name=None, max_confidence=1.0):
1471:    """Normalize and validate a signal's return dict.
3438:        has_macro_signals = any(e.get("requires_macro") for e in _enhanced_entries.values())
3496:                            if entry.get("requires_context"):
3498:                            elif entry.get("requires_macro"):
3506:                            validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
3535:                if entry.get("requires_context"):
3537:                elif entry.get("requires_macro"):
3545:                validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
3548:                extra_info[f"{sig_name}_sub_signals"] = validated["sub_signals"]
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I have the registry contract now. Next pass is the engine’s validator and dispatch paths, because that determines whether a detector bug turns into a hard exception, a silent HOLD, or a malformed vote.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signal_engine.py | Select-Object -Skip 1469 -First 45" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signal_engine.py | Select-Object -Skip 3488 -First 70" in Q:\finance-analyzer
 succeeded in 759ms:
                if sig_name in _SHADOW_SAFE_SIGNALS:
                    try:
                        _sig_t0 = time.monotonic()
                        compute_fn = load_signal_func(entry)
                        if compute_fn is not None:
                            if ticker:
                                _set_last_signal(ticker, f"shadow:{sig_name}")
                            if entry.get("requires_context"):
                                result = compute_fn(df, context=context_data)
                            elif entry.get("requires_macro"):
                                result = compute_fn(df, macro=macro_data or None)
                            else:
                                result = compute_fn(df)
                            _sig_dt = time.monotonic() - _sig_t0
                            if _sig_dt > 1.0:
                                logger.info("[SLOW-SHADOW] %s/%s: %.1fs", ticker, sig_name, _sig_dt)
                            max_conf = entry.get("max_confidence", 1.0)
                            validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
                            extra_info[f"{sig_name}_action"] = validated["action"]
                            extra_info[f"{sig_name}_confidence"] = validated["confidence"]
                            extra_info[f"shadow_{sig_name}"] = True
                            shadow_votes[sig_name] = validated["action"]
                    except Exception as e:
                        logger.debug("Shadow signal %s failed: %s", sig_name, e)
                votes[sig_name] = "HOLD"
                continue
            if sig_name in _TICKER_DISABLED_SIGNALS.get(ticker, ()):
                votes[sig_name] = "HOLD"
                continue
            # Skip GPU-intensive enhanced signals for stocks outside market hours
            if skip_gpu and sig_name in GPU_SIGNALS:
                votes[sig_name] = "HOLD"
                continue
            try:
                _sig_t0 = time.monotonic()
                compute_fn = load_signal_func(entry)
                if compute_fn is None:
                    votes[sig_name] = "HOLD"
                    continue
                # BUG-178 diagnostic: track which signal each ticker is currently
                # running so main.py's pool-timeout handler can name the culprit.
                # Ticker guard added 2026-04-10 in the phase-marker diag commit
                # to prevent leaking a None-keyed entry when callers pass
                # ticker=None (legacy test harnesses and backtester paths).
                if ticker:
                    _set_last_signal(ticker, sig_name)
                if entry.get("requires_context"):
                    result = compute_fn(df, context=context_data)
                elif entry.get("requires_macro"):
                    result = compute_fn(df, macro=macro_data or None)
                else:
                    result = compute_fn(df)
                _sig_dt = time.monotonic() - _sig_t0
                if _sig_dt > 1.0:
                    logger.info("[SLOW] %s/%s: %.1fs", ticker, sig_name, _sig_dt)
                max_conf = entry.get("max_confidence", 1.0)
                validated = _validate_signal_result(result, sig_name=sig_name, max_confidence=max_conf)
                extra_info[f"{sig_name}_action"] = validated["action"]
                extra_info[f"{sig_name}_confidence"] = validated["confidence"]
                extra_info[f"{sig_name}_sub_signals"] = validated["sub_signals"]
                if validated["indicators"]:
                    extra_info[f"{sig_name}_indicators"] = validated["indicators"]
                votes[sig_name] = validated["action"]
            except Exception as e:
                logger.warning("Signal %s failed: %s", sig_name, e)
                votes[sig_name] = "HOLD"
                _signal_failures.append(sig_name)
        if _signal_failures:
            extra_info["_signal_failures"] = _signal_failures
            if len(_signal_failures) > 3:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 785ms:
def _validate_signal_result(result, sig_name=None, max_confidence=1.0):
    """Normalize and validate a signal's return dict.

    Ensures action is a valid string, confidence is a finite float in [0, 1],
    and sub_signals is a dict. Returns a clean dict, always.
    """
    if not result or not isinstance(result, dict):
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}}

    action = result.get("action")
    if action not in _VALID_ACTIONS:
        if sig_name:
            logger.warning("Signal %s returned invalid action=%r, defaulting to HOLD", sig_name, action)
        action = "HOLD"

    conf = result.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    if not np.isfinite(conf):
        if sig_name:
            logger.warning("Signal %s returned non-finite confidence=%r, defaulting to 0.0", sig_name, conf)
        conf = 0.0
    conf = max(0.0, min(max_confidence, conf))

    sub_signals = result.get("sub_signals")
    if not isinstance(sub_signals, dict):
        sub_signals = {}

    return {
        "action": action,
        "confidence": conf,
        "sub_signals": sub_signals,
        "indicators": result.get("indicators") or {},
    }


# Dynamic correlation group computation TTL and thresholds
_DYNAMIC_CORR_TTL = 7200  # 2h cache for dynamic correlation groups
# 2026-04-18: changed from Pearson r > 0.7 to agreement rate > 0.85.
# Pearson on vote encoding (BUY=1, HOLD=0, SELL=-1) was diluted by 70-90%
# HOLD dominance - max observed r=0.538 (ematrend), making the 0.7
# threshold unreachable and dynamic groups always falling back to static.
# Agreement rate only counts pairs where at least one signal voted non-HOLD.
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’m switching to an inventory pass over all detector entrypoints. That gives me a hard list of signatures, globals, and import-time behavior before I drill into the modules with the highest risk patterns.
2026-05-11T15:29:21.224364Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast, pathlib, json
base = pathlib.Path(r'Q:\\finance-analyzer\\portfolio\\signals')
rows=[]
for path in sorted(base.glob('*.py')):
    src = path.read_text(encoding='utf-8')
    mod = ast.parse(src)
    funcs=[]
    top_calls=[]
    top_mutable=[]
    imports=[]
    for node in mod.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names=[a.name for a in node.names]
            else:
                names=[(node.module or '') + ':' + a.name for a in node.names]
            imports.extend(names)
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith('compute_'):
                args=[a.arg for a in node.args.args]
                if node.args.vararg: args.append('*'+node.args.vararg.arg)
                if node.args.kwarg: args.append('**'+node.args.kwarg.arg)
                funcs.append((node.name, args, node.lineno))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                    names=[]
                    if isinstance(t, ast.Name): names=[t.id]
                    elif isinstance(t, ast.Tuple):
                        names=[e.id for e in t.elts if isinstance(e, ast.Name)]
                    for name in names:
                        top_mutable.append((name, type(node.value).__name__, node.lineno))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.value, (ast.Dict, ast.List, ast.Set)) and isinstance(node.target, ast.Name):
                top_mutable.append((node.target.id, type(node.value).__name__, node.lineno))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn=node.value.func
            if isinstance(fn, ast.Name):
                top_calls.append((fn.id, node.lineno))
            elif isinstance(fn, ast.Attribute):
                top_calls.append((fn.attr, node.lineno))
    rows.append({
        'file': path.name,
        'compute_funcs': funcs,
        'top_calls': top_calls[:10],
        'top_mutable': top_mutable,
        'imports': imports,
    })
print(json.dumps(rows, indent=2))
'@ | python -"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast, pathlib, json
base = pathlib.Path(r'Q:\\finance-analyzer\\portfolio\\signals')
rows=[]
for path in sorted(base.glob('*.py')):
    src = path.read_text(encoding='utf-8')
    mod = ast.parse(src)
    funcs=[]
    top_calls=[]
    top_mutable=[]
    imports=[]
    for node in mod.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names=[a.name for a in node.names]
            else:
                names=[(node.module or '') + ':' + a.name for a in node.names]
            imports.extend(names)
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith('compute_'):
                args=[a.arg for a in node.args.args]
                if node.args.vararg: args.append('*'+node.args.vararg.arg)
                if node.args.kwarg: args.append('**'+node.args.kwarg.arg)
                funcs.append((node.name, args, node.lineno))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                    names=[]
                    if isinstance(t, ast.Name): names=[t.id]
                    elif isinstance(t, ast.Tuple):
                        names=[e.id for e in t.elts if isinstance(e, ast.Name)]
                    for name in names:
                        top_mutable.append((name, type(node.value).__name__, node.lineno))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.value, (ast.Dict, ast.List, ast.Set)) and isinstance(node.target, ast.Name):
                top_mutable.append((node.target.id, type(node.value).__name__, node.lineno))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn=node.value.func
            if isinstance(fn, ast.Name):
                top_calls.append((fn.id, node.lineno))
            elif isinstance(fn, ast.Attribute):
                top_calls.append((fn.attr, node.lineno))
    rows.append({
        'file': path.name,
        'compute_funcs': funcs,
        'top_calls': top_calls[:10],
        'top_mutable': top_mutable,
        'imports': imports,
    })
print(json.dumps(rows, indent=2))
'@ | python -" in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import ast, pathlib, json
base = pathlib.Path(r'Q:\\finance-analyzer\\portfolio\\signals')
rows=[]
for path in sorted(base.glob('*.py')):
    src = path.read_text(encoding='utf-8')
    mod = ast.parse(src)
    funcs=[]
    top_calls=[]
    top_mutable=[]
    imports=[]
    for node in mod.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names=[a.name for a in node.names]
            else:
                names=[(node.module or '') + ':' + a.name for a in node.names]
            imports.extend(names)
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith('compute_'):
                args=[a.arg for a in node.args.args]
                if node.args.vararg: args.append('*'+node.args.vararg.arg)
                if node.args.kwarg: args.append('**'+node.args.kwarg.arg)
                funcs.append((node.name, args, node.lineno))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                    names=[]
                    if isinstance(t, ast.Name): names=[t.id]
                    elif isinstance(t, ast.Tuple):
                        names=[e.id for e in t.elts if isinstance(e, ast.Name)]
                    for name in names:
                        top_mutable.append((name, type(node.value).__name__, node.lineno))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.value, (ast.Dict, ast.List, ast.Set)) and isinstance(node.target, ast.Name):
                top_mutable.append((node.target.id, type(node.value).__name__, node.lineno))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn=node.value.func
            if isinstance(fn, ast.Name):
                top_calls.append((fn.id, node.lineno))
            elif isinstance(fn, ast.Attribute):
                top_calls.append((fn.attr, node.lineno))
    rows.append({
        'file': path.name,
        'compute_funcs': funcs,
        'top_calls': top_calls[:10],
        'top_mutable': top_mutable,
        'imports': imports,
    })
print(json.dumps(rows, indent=2))
'@ | python -"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^def compute_.*signal" Q:'"\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
 succeeded in 703ms:
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:240:def compute_gold_real_yield_paradox_signal(
Q:\finance-analyzer\portfolio\signals\fibonacci.py:373:def compute_fibonacci_signal(df: pd.DataFrame) -> dict:
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:224:def compute_hash_ribbons_signal(
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:37:def compute_dxy_cross_asset_signal(
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:176:def compute_econ_calendar_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\forecast.py:733:def compute_forecast_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:309:def compute_cot_positioning_signal(
Q:\finance-analyzer\portfolio\signals\futures_flow.py:223:def compute_futures_flow_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\futures_basis.py:184:def compute_futures_basis_signal(df=None, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:894:def compute_claude_fundamental_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\candlestick.py:33:def compute_candlestick_signal(df: pd.DataFrame) -> dict:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:319:def compute_calendar_signal(df: pd.DataFrame) -> dict:
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:200:def compute_complexity_gap_regime_signal(
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:264:def compute_crypto_evrp_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:174:def compute_cross_asset_tsmom_signal(
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:169:def compute_copper_gold_ratio_signal(
Q:\finance-analyzer\portfolio\signals\credit_spread.py:250:def compute_credit_spread_signal(
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:197:def compute_crypto_macro_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:73:def compute_cubic_trend_persistence_signal(
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:365:def compute_momentum_factors_signal(df: pd.DataFrame, context: dict | None = None) -> dict:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:398:def compute_mean_reversion_signal(df: pd.DataFrame, context: dict | None = None) -> dict:
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:134:def compute_drift_regime_gate_signal(
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:266:def compute_metals_cross_asset_signal(
Q:\finance-analyzer\portfolio\signals\macro_regime.py:301:def compute_macro_regime_signal(df: pd.DataFrame, macro: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\news_event.py:499:def compute_news_event_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:481:def compute_heikin_ashi_signal(df: pd.DataFrame) -> dict:
Q:\finance-analyzer\portfolio\signals\momentum.py:288:def compute_momentum_signal(df: pd.DataFrame) -> dict:
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:152:def compute_intraday_seasonality_signal(
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:92:def compute_orderbook_flow_signal(
Q:\finance-analyzer\portfolio\signals\network_momentum.py:307:def compute_network_momentum_signal(
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:212:def compute_hurst_regime_signal(df: pd.DataFrame,
Q:\finance-analyzer\portfolio\signals\oscillators.py:434:def compute_oscillator_signal(df: pd.DataFrame) -> dict:
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:245:def compute_mahalanobis_turbulence_signal(
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:166:def compute_ovx_metals_spillover_signal(
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:183:def compute_realized_skewness_signal(
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:237:def compute_residual_pair_reversion_signal(
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:171:def compute_shannon_entropy_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\smart_money.py:401:def compute_smart_money_signal(df: pd.DataFrame) -> dict[str, Any]:
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:138:def compute_statistical_jump_regime_signal(
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:153:def compute_treasury_risk_rotation_signal(
Q:\finance-analyzer\portfolio\signals\structure.py:177:def compute_structure_signal(df: pd.DataFrame) -> dict:
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:141:def compute_vix_term_structure_signal(df=None, context=None, **kwargs) -> dict:
Q:\finance-analyzer\portfolio\signals\trend.py:432:def compute_trend_signal(df: pd.DataFrame) -> dict:
Q:\finance-analyzer\portfolio\signals\volume_flow.py:232:def compute_volume_flow_signal(df: pd.DataFrame) -> dict[str, Any]:
Q:\finance-analyzer\portfolio\signals\volatility.py:292:def compute_volatility_signal(df: pd.DataFrame) -> dict[str, Any]:
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:34:def compute_vwap_zscore_mr_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:128:def compute_vol_ratio_regime_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:187:def compute_xtrend_equity_spillover_signal(
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:173:def compute_williams_vix_fix_signal(
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"shift\\(-|pct_change\\(-|diff\\(-|rolling\\(["'^'"\\n]*center\\s*=\\s*True|\\.iloc\\[-1\\].*shift|future|lead|next_\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"ImportError|ModuleNotFoundError|except Exception|try:\\s*"'$|import .*talib|from talib|import requests|httpx|urllib|yfinance|fredapi|open'"\\(|Path\\(|read_csv|read_json|read_parquet\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"tz_localize|tz_convert|datetime\\.now|datetime\\.utcnow|Timestamp\\.now|pd\\.Timestamp|timezone|UTC|tzinfo\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
 succeeded in 962ms:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:19:from datetime import UTC
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:190:                today = datetime.now(UTC).date()
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:308:    date_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:353:    date_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:413:    date_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:651:    ts = _dt.datetime.now(_dt.UTC).isoformat()
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:912:    # Market hours gate — only refresh during EU+US hours (07:00-21:00 UTC weekdays).
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:384:        if isinstance(last_time, (pd.Timestamp, datetime)):
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:389:            last_date = pd.Timestamp(last_time).date()
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:20:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:43:        if isinstance(last_time, pd.Timestamp):
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:44:            return last_time.to_pydatetime().replace(tzinfo=UTC)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:45:    return datetime.now(UTC)
Q:\finance-analyzer\portfolio\signals\forecast.py:24:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\signals\forecast.py:224:            "ts": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\signals\forecast.py:949:                "ts": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:99:        dates = [pd.Timestamp.fromtimestamp(v["x"], tz="UTC").normalize() for v in values]
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:6:  1. Hour Alpha        — UTC hour maps to empirical return multiplier per asset class
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:15:  - BTC 22:00-23:00 UTC: 33% annualized (ScienceDirect 2024)
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:18:  - Crypto tea-time peak 16:00-17:00 UTC (Springer RQFA 2024)
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:80:    """Extract UTC hour and day-of-week from DataFrame's last timestamp."""
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:85:                if last_ts.tzinfo is not None:
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:86:                    utc_ts = last_ts.astimezone(datetime.timezone.utc)
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:91:    now = datetime.datetime.now(datetime.timezone.utc)
Q:\finance-analyzer\portfolio\signals\news_event.py:18:from datetime import UTC, datetime
Q:\finance-analyzer\portfolio\signals\news_event.py:92:            "timestamp": datetime.now(UTC).isoformat(),
Q:\finance-analyzer\portfolio\signals\volume_flow.py:64:    RESET at session boundaries (00:00 UTC).
Q:\finance-analyzer\portfolio\signals\volume_flow.py:69:    each day. A pure session VWAP must reset every UTC midnight so the
Q:\finance-analyzer\portfolio\signals\volume_flow.py:73:    boundary is 00:00 UTC. US stocks have a separate convention (13:30 UTC
Q:\finance-analyzer\portfolio\signals\volume_flow.py:88:        # tz-aware: normalize to UTC so the session boundary is consistent.
Q:\finance-analyzer\portfolio\signals\volume_flow.py:89:        # tz-naive: assume UTC (or at least daily-aligned) — extract date directly.
Q:\finance-analyzer\portfolio\signals\volume_flow.py:91:            session_id = idx.tz_convert("UTC").date
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1001ms:
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:17:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:19:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:27:Data: copper futures (HG=F) and gold futures (GC=F) via yfinance/price_source.
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:29:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:69:    """Fetch copper and gold futures daily data, return ratio Series."""
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:176:            externally from copper/gold futures).
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:23:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:26:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:14:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:21:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:12:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:140:        parts.append(f"FOMC: {fed.get('next_date', '?')} ({fed.get('days_until', '?')}d)")
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:187:                next_date = dates[0] if isinstance(dates, list) else dates
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:188:                if hasattr(next_date, "date"):
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:189:                    next_date = next_date.date()
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:191:                days_until = (next_date - today).days
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:193:                    "date": str(next_date),
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:458:- Cross-asset reasoning: if BTC breaks down, how does ETH follow? If semis rally, which names lead?
Q:\finance-analyzer\portfolio\signals\news_event.py:14:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\news_event.py:285:                # missed expectations" — the cut is the lead item).
Q:\finance-analyzer\portfolio\signals\futures_basis.py:3:Computes the perpetual futures basis (mark price vs index price spread) from
Q:\finance-analyzer\portfolio\signals\futures_basis.py:21:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\futures_basis.py:33:logger = logging.getLogger("portfolio.signals.futures_basis")
Q:\finance-analyzer\portfolio\signals\futures_basis.py:184:def compute_futures_basis_signal(df=None, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\futures_basis.py:185:    """Compute futures basis regime signal.
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:19:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:222:    next_day = last_date + timedelta(days=1)
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:223:    is_pre_holiday = (next_day.month, next_day.day) in _US_HOLIDAYS
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:21:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:317:    # Shift forward (the Alligator projects lines into the future)
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:15:    XAG-USD  -> GC=F  (silver follows gold futures, 24h coverage)
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:22:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\credit_spread.py:20:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\forecast.py:14:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\forecast.py:22:from concurrent.futures import ThreadPoolExecutor
Q:\finance-analyzer\portfolio\signals\forecast.py:23:from concurrent.futures import TimeoutError as FuturesTimeout
Q:\finance-analyzer\portfolio\signals\forecast.py:416:            future = executor.submit(forecast_chronos, "", prices, horizons=horizons)
Q:\finance-analyzer\portfolio\signals\forecast.py:418:                result = future.result(timeout=_timeout)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:27:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:22:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:17:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:27:    next_event,
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:55:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:56:    indicators = {"next_event": None, "hours_until": None}
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:61:    indicators["next_event"] = f"{evt['type']} {evt['date'].isoformat()}"
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:78:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:136:        evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:141:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:144:        indicators["next_event_hours"] = evt["hours_until"]
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:156:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:214:        if prox_ind.get("next_event") is None:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:215:            logger.warning("econ_calendar: no future events found — dates may need updating")
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:16:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:108:    Based on the thesis that gold leads BTC by ~3 months.
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:127:        return "SELL", indicators  # gold still leading — rotation not started
Q:\finance-analyzer\portfolio\signals\fibonacci.py:18:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\oscillators.py:20:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\futures_flow.py:17:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\futures_flow.py:27:logger = logging.getLogger("portfolio.signals.futures_flow")
Q:\finance-analyzer\portfolio\signals\futures_flow.py:223:def compute_futures_flow_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\futures_flow.py:224:    """Compute the futures flow composite signal.
Q:\finance-analyzer\portfolio\signals\futures_flow.py:245:    # Fetch all futures data
Q:\finance-analyzer\portfolio\signals\futures_flow.py:246:    from portfolio.futures_data import get_all_futures_data
Q:\finance-analyzer\portfolio\signals\futures_flow.py:247:    futures = _cached(
Q:\finance-analyzer\portfolio\signals\futures_flow.py:248:        f"futures_flow_data_{ticker}",
Q:\finance-analyzer\portfolio\signals\futures_flow.py:250:        get_all_futures_data,
Q:\finance-analyzer\portfolio\signals\futures_flow.py:254:    if futures is None:
Q:\finance-analyzer\portfolio\signals\futures_flow.py:262:    oi_history = futures.get("oi_history")
Q:\finance-analyzer\portfolio\signals\futures_flow.py:263:    ls_ratio = futures.get("ls_ratio")
Q:\finance-analyzer\portfolio\signals\futures_flow.py:264:    top_position = futures.get("top_position_ratio")
Q:\finance-analyzer\portfolio\signals\futures_flow.py:265:    funding_hist = futures.get("funding_history")
Q:\finance-analyzer\portfolio\signals\futures_flow.py:266:    current_oi = futures.get("open_interest")
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:14:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\network_momentum.py:4:arXiv:2308.11294.  Sharpe 1.511, 22% annual return across 64 futures,
Q:\finance-analyzer\portfolio\signals\network_momentum.py:13:but the target leads, SELL (expecting pull-down).
Q:\finance-analyzer\portfolio\signals\network_momentum.py:23:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\network_momentum.py:54:    "XAU-USD": "GC=F",     # gold futures (24h coverage)
Q:\finance-analyzer\portfolio\signals\network_momentum.py:55:    "XAG-USD": "SI=F",     # silver futures (24h coverage)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:15:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:22:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:19:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:14:      futures. Low-skew assets outperform high-skew assets.
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:16:      skewness risk and future crypto returns.
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:21:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\macro_regime.py:23:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:14:This signal uses oil IMPLIED VOLATILITY (expectation of future moves).
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:27:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:20:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:226:    # is fragile to future refactors, so make it explicit: any negative
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:29:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:65:    "XAG-USD": "GC=F",      # gold futures (yfinance symbol)
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:66:    "XAU-USD": "SI=F",      # silver futures (inverse pair)
Q:\finance-analyzer\portfolio\signals\shannon_entropy.py:22:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:30:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:29:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\momentum.py:20:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\trend.py:17:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:21:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\structure.py:14:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:21:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:23:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:21:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:15:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\volatility.py:13:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\volume_flow.py:16:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\smart_money.py:15:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:30:from __future__ import annotations
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:21:from __future__ import annotations
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 978ms:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:27:Data: copper futures (HG=F) and gold futures (GC=F) via yfinance/price_source.
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:52:    # Flatten MultiIndex columns (yfinance quirk: ("Close", "HG=F"))
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:75:    try:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:80:    except Exception:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:81:        try:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:82:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:87:        except Exception as exc:
Q:\finance-analyzer\portfolio\signals\candlestick.py:76:    try:
Q:\finance-analyzer\portfolio\signals\candlestick.py:187:    try:
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:24:Data: 5-asset daily closes via yfinance (free, no API key).
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:56:# Peer tickers for yfinance download
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:67:        try:
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:68:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:76:            # Extract Close prices — handle MultiIndex (yfinance multi-ticker)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:88:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:131:        try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:379:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:390:    except Exception:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:396:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:398:    except Exception:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:402:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:404:    except Exception:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:408:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:410:    except Exception:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:414:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:416:    except Exception:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:420:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:422:    except Exception:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:426:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:428:    except Exception:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:432:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:434:    except Exception:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:438:    try:
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:440:    except Exception:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:28:DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:153:    """Fetch next earnings dates from yfinance for all stock tickers.
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:164:    try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:166:    except ImportError:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:171:        try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:172:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:197:        except Exception:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:296:    try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:299:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:518:        try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:526:        try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:535:        try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:670:        try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:672:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:763:    try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:766:    except Exception:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:809:    try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:812:    except Exception:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:888:    try:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:890:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:20:Data: yfinance for TLT/SPY/GC=F/BTC-USD (free, no API key). Cached 1 hour.
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:55:    """Fetch 12-month daily closes for TLT, SPY, GLD, BTC-USD via yfinance.
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:61:        try:
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:62:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:100:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:101:            logger.warning("cross_asset_tsmom yfinance fetch failed: %s", e)
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:114:    try:
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:28:# The previous code used relative `Path("data")` / `data/...` which silently
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:33:_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:93:    try:
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:94:        import requests
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:123:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:132:    try:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:73:    try:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:75:    except ImportError:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:76:        import requests
Q:\finance-analyzer\portfolio\signals\credit_spread.py:84:    try:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:107:                try:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:119:    except Exception:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:284:        try:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:287:        except Exception:
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:75:    try:
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:110:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:125:    try:
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:174:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:61:    try:
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:63:    except ImportError:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:211:    try:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:216:    except Exception:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:220:    try:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:222:    except Exception:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:226:    try:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:228:    except Exception:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:232:    try:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:234:    except Exception:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:238:    try:
Q:\finance-analyzer\portfolio\signals\econ_calendar.py:240:    except Exception:
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:38:    try:
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:46:    except ImportError:
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:68:    try:
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:76:    except ImportError:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:464:        try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:488:        except Exception:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:500:    try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:504:    except Exception:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:510:    try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:514:    except Exception:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:520:    try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:524:    except Exception:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:530:    try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:534:    except Exception:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:540:    try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:547:    except Exception:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:554:    try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:562:    except Exception:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:568:    try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:573:    except Exception:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:580:    try:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:585:    except Exception:
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:27:Data: 5-asset daily closes via yfinance (free, no API key).
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:63:# Peer tickers for yfinance download (proxy for our 5 assets)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:74:        try:
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:75:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:83:            # Extract Close prices — handle MultiIndex (yfinance multi-ticker)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:95:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:157:        try:
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:167:        try:
Q:\finance-analyzer\portfolio\signals\news_event.py:56:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:97:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:114:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:131:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:148:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:169:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:446:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:449:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:547:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:549:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:552:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:554:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:557:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:559:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:562:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:564:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:567:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:569:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:572:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:574:    except Exception:
Q:\finance-analyzer\portfolio\signals\news_event.py:578:    try:
Q:\finance-analyzer\portfolio\signals\news_event.py:580:    except Exception:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:129:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:159:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:203:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:234:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:364:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:366:    except Exception:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:370:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:372:    except Exception:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:376:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:378:    except Exception:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:382:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:384:    except Exception:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:388:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:390:    except Exception:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:394:    try:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:396:    except Exception:
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:82:        try:
Q:\finance-analyzer\portfolio\signals\intraday_seasonality.py:89:        except Exception:
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:282:        try:
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:287:        except Exception:
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:299:        try:
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:304:        except Exception:
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:323:    try:
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:327:    except Exception:
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:567:    try:
Q:\finance-analyzer\portfolio\signals\network_momentum.py:20:Data: yfinance for peer asset daily closes (free, no API key).
Q:\finance-analyzer\portfolio\signals\network_momentum.py:49:# Peer tickers fetched via yfinance.  Mapped to our internal names
Q:\finance-analyzer\portfolio\signals\network_momentum.py:60:# Map our internal ticker names to yfinance peer column names
Q:\finance-analyzer\portfolio\signals\network_momentum.py:71:    """Fetch ~4 months of daily closes for peer assets via yfinance.
Q:\finance-analyzer\portfolio\signals\network_momentum.py:77:        try:
Q:\finance-analyzer\portfolio\signals\network_momentum.py:78:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\network_momentum.py:100:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\network_momentum.py:101:            logger.warning("network_momentum yfinance fetch failed: %s", e)
Q:\finance-analyzer\portfolio\signals\oscillators.py:505:    try:
Q:\finance-analyzer\portfolio\signals\oscillators.py:509:    except Exception:
Q:\finance-analyzer\portfolio\signals\oscillators.py:514:    try:
Q:\finance-analyzer\portfolio\signals\oscillators.py:518:    except Exception:
Q:\finance-analyzer\portfolio\signals\oscillators.py:523:    try:
Q:\finance-analyzer\portfolio\signals\oscillators.py:528:    except Exception:
Q:\finance-analyzer\portfolio\signals\oscillators.py:534:    try:
Q:\finance-analyzer\portfolio\signals\oscillators.py:538:    except Exception:
Q:\finance-analyzer\portfolio\signals\oscillators.py:543:    try:
Q:\finance-analyzer\portfolio\signals\oscillators.py:548:    except Exception:
Q:\finance-analyzer\portfolio\signals\oscillators.py:554:    try:
Q:\finance-analyzer\portfolio\signals\oscillators.py:558:    except Exception:
Q:\finance-analyzer\portfolio\signals\oscillators.py:563:    try:
Q:\finance-analyzer\portfolio\signals\oscillators.py:567:    except Exception:
Q:\finance-analyzer\portfolio\signals\oscillators.py:572:    try:
Q:\finance-analyzer\portfolio\signals\oscillators.py:576:    except Exception:
Q:\finance-analyzer\portfolio\signals\forecast.py:87:    try:
Q:\finance-analyzer\portfolio\signals\forecast.py:90:            str(Path(__file__).resolve().parent.parent.parent / "config.json"),
Q:\finance-analyzer\portfolio\signals\forecast.py:100:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:114:_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
Q:\finance-analyzer\portfolio\signals\forecast.py:153:        try:
Q:\finance-analyzer\portfolio\signals\forecast.py:161:        try:
Q:\finance-analyzer\portfolio\signals\forecast.py:172:            try:
Q:\finance-analyzer\portfolio\signals\forecast.py:183:def _kronos_circuit_open() -> bool:
Q:\finance-analyzer\portfolio\signals\forecast.py:195:def _chronos_circuit_open() -> bool:
Q:\finance-analyzer\portfolio\signals\forecast.py:222:    try:
Q:\finance-analyzer\portfolio\signals\forecast.py:233:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:279:    try:
Q:\finance-analyzer\portfolio\signals\forecast.py:307:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:317:    if _kronos_circuit_open():
Q:\finance-analyzer\portfolio\signals\forecast.py:320:    try:
Q:\finance-analyzer\portfolio\signals\forecast.py:326:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:336:    try:
Q:\finance-analyzer\portfolio\signals\forecast.py:338:        try:
Q:\finance-analyzer\portfolio\signals\forecast.py:340:            cfg = load_json(str(Path(__file__).resolve().parent.parent.parent / "config.json"), {})
Q:\finance-analyzer\portfolio\signals\forecast.py:342:        except Exception:
Q:\finance-analyzer\portfolio\signals\forecast.py:386:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:397:    if _chronos_circuit_open():
Q:\finance-analyzer\portfolio\signals\forecast.py:411:    try:
Q:\finance-analyzer\portfolio\signals\forecast.py:417:            try:
Q:\finance-analyzer\portfolio\signals\forecast.py:433:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:526:    try:
Q:\finance-analyzer\portfolio\signals\forecast.py:529:    except ImportError:
Q:\finance-analyzer\portfolio\signals\forecast.py:541:        try:
Q:\finance-analyzer\portfolio\signals\forecast.py:544:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:557:        try:
Q:\finance-analyzer\portfolio\signals\forecast.py:568:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:772:        try:
Q:\finance-analyzer\portfolio\signals\forecast.py:775:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:810:            try:
Q:\finance-analyzer\portfolio\signals\forecast.py:825:            except Exception as e:
Q:\finance-analyzer\portfolio\signals\forecast.py:832:    result["indicators"]["kronos_circuit_open"] = _kronos_circuit_open()
Q:\finance-analyzer\portfolio\signals\forecast.py:833:    result["indicators"]["chronos_circuit_open"] = _chronos_circuit_open()
Q:\finance-analyzer\portfolio\signals\forecast.py:942:    try:
Q:\finance-analyzer\portfolio\signals\forecast.py:972:    except Exception:
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:62:# Pair mapping: target ticker → yfinance ticker for the driver asset
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:65:    "XAG-USD": "GC=F",      # gold futures (yfinance symbol)
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:73:    """Fetch ~1 year of daily closes for the driver asset via yfinance.
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:78:        try:
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:79:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:90:        except Exception as exc:
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:24:Data: yfinance ^OVX (CBOE Oil Volatility Index), free, daily.
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:59:    """Fetch OVX historical data via yfinance."""
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:60:    try:
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:61:        import yfinance as yf
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:85:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:43:    try:
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:124:    except Exception:
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:67:    try:
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:69:    except ImportError:
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:71:        import requests
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:82:    try:
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:85:    except Exception:
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:98:    try:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:435:    try:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:437:    except Exception:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:442:    try:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:444:    except Exception:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:471:    try:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:477:    except Exception:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:482:    try:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:488:    except Exception:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:493:    try:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:499:    except Exception:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:504:    try:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:510:    except Exception:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:518:    try:
Q:\finance-analyzer\portfolio\signals\fibonacci.py:524:    except Exception:
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:19:Data: yfinance ^VIX and ^VIX3M (free, no API key).
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:44:    """Fetch VIX and VIX3M closing prices via yfinance."""
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:45:    try:
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:46:        import yfinance as yf
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:88:    except Exception as e:
Q:\finance-analyzer\portfolio\signals\volatility.py:349:    try:
Q:\finance-analyzer\portfolio\signals\volatility.py:353:    except Exception as exc:
Q:\finance-analyzer\portfolio\signals\volatility.py:358:    try:
Q:\finance-analyzer\portfolio\signals\volatility.py:360:    except Exception as exc:
Q:\finance-analyzer\portfolio\signals\volatility.py:364:    try:
Q:\finance-analyzer\portfolio\signals\volatility.py:368:    except Exception as exc:
Q:\finance-analyzer\portfolio\signals\volatility.py:373:    try:
Q:\finance-analyzer\portfolio\signals\volatility.py:377:    except Exception as exc:
Q:\finance-analyzer\portfolio\signals\volatility.py:382:    try:
Q:\finance-analyzer\portfolio\signals\volatility.py:386:    except Exception as exc:
Q:\finance-analyzer\portfolio\signals\volatility.py:391:    try:
Q:\finance-analyzer\portfolio\signals\volatility.py:395:    except Exception as exc:
Q:\finance-analyzer\portfolio\signals\volatility.py:400:    try:
Q:\finance-analyzer\portfolio\signals\volatility.py:404:    except Exception as exc:
Q:\finance-analyzer\portfolio\signals\momentum.py:356:    try:
Q:\finance-analyzer\portfolio\signals\momentum.py:358:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum.py:362:    try:
Q:\finance-analyzer\portfolio\signals\momentum.py:367:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum.py:373:    try:
Q:\finance-analyzer\portfolio\signals\momentum.py:377:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum.py:382:    try:
Q:\finance-analyzer\portfolio\signals\momentum.py:386:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum.py:391:    try:
Q:\finance-analyzer\portfolio\signals\momentum.py:395:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum.py:400:    try:
Q:\finance-analyzer\portfolio\signals\momentum.py:404:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum.py:409:    try:
Q:\finance-analyzer\portfolio\signals\momentum.py:414:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum.py:420:    try:
Q:\finance-analyzer\portfolio\signals\momentum.py:425:    except Exception:
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:17:Data: yfinance IEF + TLT (free, no API key). Cached 1 hour.
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:48:    """Fetch IEF and TLT daily closes via yfinance.  Cached 1 hour."""
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:51:        try:
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:52:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:73:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:74:            logger.warning("treasury_risk_rotation yfinance fetch failed: %s", e)
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:18:Data: yfinance for SPY and QQQ (free, no API key).  Cached 1 hour via
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:50:    """Fetch SPY and QQQ daily OHLCV via yfinance (90 days).
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:56:        try:
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:57:            import yfinance as yf
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:67:                try:
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:105:        except Exception as e:
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:106:            logger.warning("xtrend_equity_spillover: yfinance fetch failed: %s", e)
Q:\finance-analyzer\portfolio\signals\structure.py:223:    try:
Q:\finance-analyzer\portfolio\signals\structure.py:225:    except Exception:
Q:\finance-analyzer\portfolio\signals\structure.py:229:    try:
Q:\finance-analyzer\portfolio\signals\structure.py:231:    except Exception:
Q:\finance-analyzer\portfolio\signals\structure.py:235:    try:
Q:\finance-analyzer\portfolio\signals\structure.py:237:    except Exception:
Q:\finance-analyzer\portfolio\signals\structure.py:241:    try:
Q:\finance-analyzer\portfolio\signals\structure.py:243:    except Exception:
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:71:    try:
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:73:    except ImportError:
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:77:    try:
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:101:            try:
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:114:    except Exception:
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:263:        try:
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:268:        except Exception:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:22:yfinance hiccup, etc.). GVZ stays daily — it's a daily-published index.
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:121:    try:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:123:    except ImportError:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:127:    try:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:145:                try:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:160:    except Exception:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:192:    try:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:199:    except ImportError:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:346:    try:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:360:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:432:    try:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:436:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:442:    try:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:446:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:452:    try:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:456:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:462:    try:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:466:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:472:    try:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:476:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:482:    try:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:487:    except Exception:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:494:    try:
Q:\finance-analyzer\portfolio\signals\momentum_factors.py:499:    except Exception:
Q:\finance-analyzer\portfolio\signals\volume_flow.py:301:    try:
Q:\finance-analyzer\portfolio\signals\volume_flow.py:358:    except Exception:
Q:\finance-analyzer\portfolio\signals\smart_money.py:467:    try:
Q:\finance-analyzer\portfolio\signals\smart_money.py:479:        try:
Q:\finance-analyzer\portfolio\signals\smart_money.py:483:        except Exception:
Q:\finance-analyzer\portfolio\signals\smart_money.py:489:        try:
Q:\finance-analyzer\portfolio\signals\smart_money.py:491:        except Exception:
Q:\finance-analyzer\portfolio\signals\smart_money.py:497:        try:
Q:\finance-analyzer\portfolio\signals\smart_money.py:501:        except Exception:
Q:\finance-analyzer\portfolio\signals\smart_money.py:507:        try:
Q:\finance-analyzer\portfolio\signals\smart_money.py:513:        except Exception:
Q:\finance-analyzer\portfolio\signals\smart_money.py:518:        try:
Q:\finance-analyzer\portfolio\signals\smart_money.py:525:        except Exception:
Q:\finance-analyzer\portfolio\signals\smart_money.py:560:    except Exception:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"shift\\(-|pct_change\\(-|diff\\(-|\\.shift\\s*\\(\\s*-|\\.rolling\\(["'^'"\\)]*center\\s*=\\s*True|\\.rolling\\(["'^'"\\n]*,\\s*center\\s*=\\s*True\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"iloc\\[-1\\s*[-+]|iloc\\[-2\\]|iloc\\[-3\\]|tail\\(|last\\(|shift\\(1\\)\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
2026-05-11T15:29:50.816684Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 1 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 exited 1 in 983ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 978ms:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:193:    prev_close = float(close.iloc[-2])
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:188:    log_ret = np.log(close / close.shift(1))
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:216:    rv_hist = np.log(rv_series / rv_series.shift(1)).rolling(RV_WINDOW).std() * np.sqrt(365) * 100
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:104:    returns = np.log(closes / closes.shift(1)).dropna()
Q:\finance-analyzer\portfolio\signals\momentum.py:103:    k_prev = k.iloc[-2] if len(k) >= 2 else np.nan
Q:\finance-analyzer\portfolio\signals\momentum.py:104:    d_prev = d.iloc[-2] if len(d) >= 2 else np.nan
Q:\finance-analyzer\portfolio\signals\momentum.py:207:    prev = roc.iloc[-2]
Q:\finance-analyzer\portfolio\signals\momentum.py:242:    ppo_prev = ppo_line.iloc[-2] if len(ppo_line) >= 2 else np.nan
Q:\finance-analyzer\portfolio\signals\momentum.py:243:    sig_prev = signal_line.iloc[-2] if len(signal_line) >= 2 else np.nan
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:115:    returns = np.log(closes / closes.shift(1)).dropna()
Q:\finance-analyzer\portfolio\signals\oscillators.py:58:    prev = ao.dropna().iloc[-2]
Q:\finance-analyzer\portfolio\signals\oscillators.py:143:    vm_plus = (high - low.shift(1)).abs()
Q:\finance-analyzer\portfolio\signals\oscillators.py:144:    vm_minus = (low - high.shift(1)).abs()
Q:\finance-analyzer\portfolio\signals\oscillators.py:246:    kst_prev = kst_line.dropna().iloc[-2]
Q:\finance-analyzer\portfolio\signals\oscillators.py:247:    sig_prev = signal_line.dropna().iloc[-2]
Q:\finance-analyzer\portfolio\signals\oscillators.py:317:        prev = float(stc_clean.iloc[-2])
Q:\finance-analyzer\portfolio\signals\oscillators.py:346:    trix_line = 100.0 * (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, np.nan)
Q:\finance-analyzer\portfolio\signals\oscillators.py:362:    prev = float(trix_clean.iloc[-2])
Q:\finance-analyzer\portfolio\signals\oscillators.py:376:            sig_prev = float(sig_clean.iloc[-2])
Q:\finance-analyzer\portfolio\signals\oscillators.py:417:    prev = float(cc_clean.iloc[-2])
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:101:    prior_close = float(df["close"].iloc[-2])
Q:\finance-analyzer\portfolio\signals\calendar_seasonal.py:102:    prior_open = float(df["open"].iloc[-2])
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:150:    old = ratio.iloc[-1 - periods]
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:160:    copper_base = combined["copper"].iloc[-1 - periods]
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:161:    gold_base = combined["gold"].iloc[-1 - periods]
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:765:        entries = load_jsonl_tail(_CF_LOG, max_entries=400)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:811:        entries = load_jsonl_tail(_CF_LOG, max_entries=500)
Q:\finance-analyzer\portfolio\signals\cubic_trend_persistence.py:85:    log_ret = np.log(close / close.shift(1)).dropna()
Q:\finance-analyzer\portfolio\signals\fibonacci.py:457:    high_prev = float(high.iloc[-2])
Q:\finance-analyzer\portfolio\signals\fibonacci.py:458:    low_prev = float(low.iloc[-2])
Q:\finance-analyzer\portfolio\signals\fibonacci.py:459:    close_prev = float(close.iloc[-2])
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:131:    prev_fast = sma30.iloc[-2]
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:132:    prev_slow = sma60.iloc[-2]
Q:\finance-analyzer\portfolio\signals\structure.py:158:    previous = hist.iloc[-2]
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:244:    prev = ha.iloc[-2]
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:368:    ema_prev = ema13.iloc[-2]
Q:\finance-analyzer\portfolio\signals\heikin_ashi.py:370:    hist_prev = macd_hist.iloc[-2]
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:290:    target_ret = np.log(aligned["target"] / aligned["target"].shift(1)).dropna()
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:291:    driver_ret = np.log(aligned["driver"] / aligned["driver"].shift(1)).dropna()
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:171:    log_returns = np.log(close / close.shift(1)).dropna()
Q:\finance-analyzer\portfolio\signals\volatility.py:61:        prev_width = bb_width.iloc[-2]
Q:\finance-analyzer\portfolio\signals\volatility.py:116:        price_change = close.iloc[-1] - close.iloc[-2] if len(close) >= 2 else 0.0
Q:\finance-analyzer\portfolio\signals\volatility.py:159:    log_returns = np.log(close / close.shift(1))
Q:\finance-analyzer\portfolio\signals\volatility.py:209:    has_prev = len(dc_upper) >= 2 and not np.isnan(dc_upper.iloc[-2])
Q:\finance-analyzer\portfolio\signals\volatility.py:214:            if price > dc_upper.iloc[-2]:
Q:\finance-analyzer\portfolio\signals\volatility.py:221:            if price < dc_lower.iloc[-2]:
Q:\finance-analyzer\portfolio\signals\trend.py:56:    prev_above = sma50.iloc[-2] > sma200.iloc[-2]
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:53:    cc_var = np.log(df["close"] / df["close"].shift(1)) ** 2
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:71:    log_ret_1 = np.log(close / close.shift(1))
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:29:56.458696Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python -c \"print('ok')\""` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python -c \"print('ok')\"" in Q:\finance-analyzer
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "python -c \"print('ok')\""` rejected: blocked by policy
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 120ms:
codex
The shell policy is tight enough that ad-hoc Python helpers are blocked, so I’m using the Node REPL for the bulk scan and then validating only the modules that surface concrete risk patterns.
exec
js_repl in Q:\fa-adv-2026-05-11
2026-05-11T15:30:06.802140Z ERROR codex_core::tools::router: error=Identifier 'fs' has already been declared
 exited 1 in 1ms:
Identifier 'fs' has already been declared
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 1ms:
50 [
  '__init__.py',
  'calendar_seasonal.py',
  'candlestick.py',
  'claude_fundamental.py',
  'complexity_gap_regime.py',
  'copper_gold_ratio.py',
  'cot_positioning.py',
  'credit_spread.py',
  'cross_asset_tsmom.py',
  'crypto_evrp.py'
]
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 50ms:
registered 38
[]
[
  {
    file: 'calendar_seasonal.py',
    match: 'compute_calendar_signal(df: pd.DataFrame)'
  },
  {
    file: 'candlestick.py',
    match: 'compute_candlestick_signal(df: pd.DataFrame)'
  },
  {
    file: 'claude_fundamental.py',
    match: 'compute_claude_fundamental_signal(df: pd.DataFrame, context: dict = None)'
  },
  {
    file: 'complexity_gap_regime.py',
    match: 'compute_complexity_gap_regime_signal(df: pd.DataFrame, context: dict | None = None,)'
  },
  {
    file: 'copper_gold_ratio.py',
    match: 'compute_copper_gold_ratio_signal(df: pd.DataFrame, context: dict | None = None,)'
  },
  {
    file: 'cot_positioning.py',
    match: 'compute_cot_positioning_signal(df: Any, context: dict | None = None, **kwargs,)'
  },
  {
    file: 'credit_spread.py',
    match: 'compute_credit_spread_signal(df: Any, context: dict | None = None, **kwargs,)'
  },
  {
    file: 'cross_asset_tsmom.py',
    match: 'compute_cross_asset_tsmom_signal(df: pd.DataFrame, context: dict | None = None)'
  }
]
exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 16ms:
[
  {
    file: 'shannon_entropy.py',
    sig: 'shannon_entropy',
    requires_context: false,
    requires_macro: false,
    params: [ 'df: pd.DataFrame', 'context: dict = None' ]
  },
  {
    file: 'statistical_jump_regime.py',
    sig: 'statistical_jump_regime',
    requires_context: false,
    requires_macro: false,
    params: [ 'df: pd.DataFrame', 'context: dict = None' ]
  },
  {
    file: 'realized_skewness.py',
    sig: 'realized_skewness',
    requires_context: false,
    requires_macro: false,
    params: [ 'df: pd.DataFrame', 'context: dict = None' ]
  },
  {
    file: 'drift_regime_gate.py',
    sig: 'drift_regime_gate',
    requires_context: false,
    requires_macro: false,
    params: [ 'df: pd.DataFrame', 'context: dict = None' ]
  },
  {
    file: 'vol_ratio_regime.py',
    sig: 'vol_ratio_regime',
    requires_context: false,
    requires_macro: false,
    params: [ 'df: pd.DataFrame', 'context: dict = None' ]
  },
  {
    file: 'williams_vix_fix.py',
    sig: 'williams_vix_fix',
    requires_context: false,
    requires_macro: false,
    params: [ 'df: pd.DataFrame', 'context: dict = None' ]
  },
  {
    file: 'cubic_trend_persistence.py',
    sig: 'cubic_trend_persistence',
    requires_context: false,
    requires_macro: false,
    params: [ 'df: pd.DataFrame', 'context: dict = None' ]
  },
  {
    file: 'vwap_zscore_mr.py',
    sig: 'vwap_zscore_mr',
    requires_context: false,
    requires_macro: false,
    params: [ 'df: pd.DataFrame', 'context: dict = None' ]
  }
]
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"context_data\\s*=|horizon\\b.*context|\\\"horizon\\\"|timeframe|ticker\" Q:\\finance-analyzer\\portfolio\\signal_engine.py" in Q:\finance-analyzer
2026-05-11T15:30:40.274920Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:context_data\s*=|horizon\b.*context|\)
    ^
error: unclosed group

 exited 1 in 772ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:context_data\s*=|horizon\b.*context|\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"context_data\\s*=\" Q:\\finance-analyzer\\portfolio\\signal_engine.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"\\\"horizon\\\"|\\\"timeframe\\\"|\\\"ticker\\\"\" Q:\\finance-analyzer\\portfolio\\signal_engine.py" in Q:\finance-analyzer
 succeeded in 761ms:
3467:        context_data = {
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:30:44.519386Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:\)
    ^
error: unclosed group

 exited 1 in 769ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signal_engine.py | Select-Object -Skip 3460 -First 20" in Q:\finance-analyzer
 succeeded in 750ms:
            except Exception:
                logger.debug("Seasonality profile load failed for %s", ticker, exc_info=True)

        # Build context data once for signals that need it
        # BUG-144: Include regime so enhanced signals (forecast.py) can apply
        # regime-specific confidence discounts.
        context_data = {
            "ticker": ticker, "config": config or {}, "macro": macro_data,
            "regime": regime, "seasonality_profile": seasonality_profile,
        }

        _signal_failures = []
        for sig_name, entry in _enhanced_entries.items():
            # BUG-178 fix (2026-04-10): respect DISABLED_SIGNALS in the dispatch
            # loop. Previously this loop iterated *every* registered enhanced
            # signal regardless of disabled status, which meant the three
            # "registered but force-HOLD pending live validation" signals
            # (crypto_macro, cot_positioning, credit_spread_risk) were doing
            # network I/O on every cycle. Their late position in iteration
            # order matches the silent gap before all 49 BUG-178 events
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
js_repl in Q:\fa-adv-2026-05-11
 succeeded in 14ms:
[
  {
    file: 'claude_fundamental.py',
    line: 3,
    text: 'Tier 1 (Haiku):  Quick directional pulse, every ~1 min'
  },
  {
    file: 'claude_fundamental.py',
    line: 4,
    text: 'Tier 2 (Sonnet): Full 5-dimension fundamental analysis, every ~10 min'
  },
  {
    file: 'claude_fundamental.py',
    line: 5,
    text: 'Tier 3 (Opus):   Deep conviction with cross-asset reasoning, every ~30 min'
  },
  {
    file: 'claude_fundamental.py',
    line: 7,
    text: 'Highest-tier fresh analysis wins (Opus > Sonnet > Haiku).'
  },
  {
    file: 'claude_fundamental.py',
    line: 331,
    text: 'Use the real fundamental data provided below (P/E, revenue growth, analyst targets)'
  },
  {
    file: 'claude_fundamental.py',
    line: 338,
    text: 'Respond with ONLY a JSON object (no markdown, no explanation):'
  },
  {
    file: 'claude_fundamental.py',
    line: 344,
    text: '- BUY only if fundamentals are clearly positive (strong earnings, moat, tailwinds)'
  },
  {
    file: 'claude_fundamental.py',
    line: 345,
    text: '- SELL only if fundamentals are clearly negative (deteriorating margins, headwinds)'
  },
  {
    file: 'claude_fundamental.py',
    line: 387,
    text: 'Respond with ONLY a JSON object (no markdown, no explanation):'
  },
  {
    file: 'claude_fundamental.py',
    line: 468,
    text: 'Respond with ONLY a JSON object (no markdown, no explanation):'
  },
  {
    file: 'complexity_gap_regime.py',
    line: 3,
    text: 'Academic basis: Mukhia, Ansari, Nurujjaman (2026), "Structural Dynamics of'
  },
  {
    file: 'complexity_gap_regime.py',
    line: 24,
    text: 'Data: 5-asset daily closes via yfinance (free, no API key).'
  },
  {
    file: 'copper_gold_ratio.py',
    line: 5,
    text: '(construction, electronics, EVs), while gold is the quintessential'
  },
  {
    file: 'copper_gold_ratio.py',
    line: 25,
    text: 'and at least 60 rows of data (for 50-period z-score + lookback).'
  },
  {
    file: 'copper_gold_ratio.py',
    line: 27,
    text: 'Data: copper futures (HG=F) and gold futures (GC=F) via yfinance/price_source.'
  },
  {
    file: 'cot_positioning.py',
    line: 11,
    text: 'deep context files (metals_precompute.py) with CFTC API fallback.'
  },
  {
    file: 'cot_positioning.py',
    line: 13,
    text: 'COT reports are published weekly (Friday for Tuesday data), so this signal'
  },
  {
    file: 'credit_spread.py',
    line: 3,
    text: 'Signal #35.  Uses ICE BofA High Yield Option-Adjusted Spread (HY OAS)'
  },
  {
    file: 'credit_spread.py',
    line: 12,
    text: '- Gold/Silver (safe haven):  risk-off → BUY, risk-on → SELL'
  },
  {
    file: 'credit_spread.py',
    line: 13,
    text: '- Crypto/Stocks (risk-on):   risk-off → SELL, risk-on → BUY'
  },
  {
    file: 'credit_spread.py',
    line: 15,
    text: 'Data: FRED API series BAMLH0A0HYM2 (daily, free with API key).'
  },
  {
    file: 'credit_spread.py',
    line: 18,
    text: 'Requires context dict with keys: ticker, config (for FRED API key).'
  },
  {
    file: 'cross_asset_tsmom.py',
    line: 4,
    text: 'Academic basis: Pitkajarvi, Suominen, Vaittinen (2020), JFE — cross-asset'
  },
  {
    file: 'cross_asset_tsmom.py',
    line: 20,
    text: 'Data: yfinance for TLT/SPY/GC=F/BTC-USD (free, no API key). Cached 1 hour.'
  },
  {
    file: 'crypto_evrp.py',
    line: 3,
    text: 'Measures the gap between implied volatility (Deribit DVOL 30-day) and'
  },
  {
    file: 'crypto_evrp.py',
    line: 4,
    text: 'realized volatility (10-day rolling from OHLCV). The VRP is persistently'
  },
  {
    file: 'crypto_evrp.py',
    line: 12,
    text: 'When eVRP is very high (>10), implied vol far exceeds realized vol — the'
  },
  {
    file: 'crypto_evrp.py',
    line: 15,
    text: 'price action (vol compression = calm = uptrend).'
  },
  {
    file: 'crypto_evrp.py',
    line: 17,
    text: 'When eVRP is very negative (<-10), realized vol exceeds implied — the'
  },
  {
    file: 'crypto_evrp.py',
    line: 21,
    text: 'Data: Deribit public REST API (free, no authentication).'
  },
  {
    file: 'crypto_evrp.py',
    line: 22,
    text: 'Applicable: BTC-USD, ETH-USD only (crypto assets with Deribit DVOL).'
  },
  {
    file: 'crypto_evrp.py',
    line: 24,
    text: 'Source: Zarattini, Mele & Aziz (2025) "The Volatility Edge";'
  },
  {
    file: 'crypto_macro.py',
    line: 4,
    text: 'Only applicable to crypto tickers (BTC-USD, ETH-USD). Non-crypto -> immediate HOLD.'
  },
  {
    file: 'cubic_trend_persistence.py',
    line: 3,
    text: 'Uses the cubic polynomial trend model from Bouchaud et al. (arXiv:2501.16772).'
  },
  {
    file: 'cubic_trend_persistence.py',
    line: 4,
    text: 'R(t+1) = b*phi(t) + c*phi(t)^3, where phi is exponentially-weighted normalized'
  },
  {
    file: 'cubic_trend_persistence.py',
    line: 5,
    text: 'trend strength. Weak trends persist (b>0), strong trends revert (c<0).'
  },
  {
    file: 'drift_regime_gate.py',
    line: 20,
    text: 'and at least 65 rows of data (63-bar lookback + 2 for diff/shift).'
  },
  {
    file: 'dxy_cross_asset.py',
    line: 3,
    text: 'Why this exists: DXY (US Dollar Index) has R² ~0.6 inverse correlation'
  },
  {
    file: 'dxy_cross_asset.py',
    line: 10,
    text: 'Metals-only (XAU-USD, XAG-USD). Crypto + equities get HOLD.'
  },
  {
    file: 'dxy_cross_asset.py',
    line: 15,
    text: 'magnitude up to a 0.5% cap (a ~5 sigma move saturates confidence).'
  },
  {
    file: 'forecast.py',
    line: 3,
    text: 'Combines Kronos (K-line foundation model), Chronos (Amazon), and Prophet'
  },
  {
    file: 'forecast.py',
    line: 11,
    text: 'Registered as enhanced signal #28 with weight=0 (shadow mode) initially.'
  },
  { file: 'forecast.py', line: 104, text: '_init_kronos_enabled()' },
  {
    file: 'futures_basis.py',
    line: 3,
    text: 'Computes the perpetual futures basis (mark price vs index price spread) from'
  },
  {
    file: 'futures_basis.py',
    line: 13,
    text: 'Applicable to all Binance FAPI tickers (BTC-USD, ETH-USD, XAU-USD, XAG-USD).'
  },
  {
    file: 'futures_basis.py',
    line: 15,
    text: 'Research basis: CF Benchmarks 2025 (Sharpe 1.52 sentiment-gated basis strategy);'
  },
  {
    file: 'futures_basis.py',
    line: 16,
    text: 'CoinDesk backwardation-bottom analysis (Nov 2022, Mar 2023, Aug 2023 bottoms).'
  },
  {
    file: 'futures_flow.py',
    line: 4,
    text: 'Only applicable to crypto tickers (BTC-USD, ETH-USD). Non-crypto → immediate HOLD.'
  },
  {
    file: 'gold_real_yield_paradox.py',
    line: 3,
    text: 'Detects when gold rises WITH real yields (historically inverse). Three'
  },
  {
    file: 'gold_real_yield_paradox.py',
    line: 9,
    text: 'Applies to XAU-USD (primary) and XAG-USD (secondary, via gold proxy).'
  },
  {
    file: 'gold_real_yield_paradox.py',
    line: 10,
    text: 'Data: FRED DFII10 (10Y TIPS real yield), cached 4 hours.'
  },
  {
    file: 'hash_ribbons.py',
    line: 5,
    text: 'are capitulating (unprofitable, shutting down).  When the 30-day crosses back'
  },
  {
    file: 'hash_ribbons.py',
    line: 7,
    text: 'bottom signal (9 signals since 2011, Capriole Investments).'
  },
  {
    file: 'hash_ribbons.py',
    line: 14,
    text: 'This is a BUY-only signal (outputs BUY or HOLD, never SELL).'
  },
  {
    file: 'hash_ribbons.py',
    line: 17,
    text: 'Data: blockchain.info free API (no auth, no rate limit).'
  },
  {
    file: 'hash_ribbons.py',
    line: 20,
    text: 'Source: Charles Edwards, Capriole Investments. Endorsed by VanEck (Dec 2025).'
  },
  {
    file: 'hurst_regime.py',
    line: 3,
    text: 'Computes the rolling Hurst exponent via Rescaled Range (R/S) analysis to'
  },
  {
    file: 'hurst_regime.py',
    line: 18,
    text: 'and at least 160 rows of data (for 150-bar R/S window + buffer).'
  },
  {
    file: 'macro_regime.py',
    line: 20,
    text: 'they work on short timeframes (100x 15m candles) as well as long ones.'
  },
  {
    file: 'mahalanobis_turbulence.py',
    line: 3,
    text: 'Academic basis: Kritzman & Li (2010), "Skulls, Financial Turbulence, and'
  },
  {
    file: 'mahalanobis_turbulence.py',
    line: 4,
    text: 'Risk Management". Also: Kritzman, Li, Page, Rigobon (2010), "Principal'
  },
  {
    file: 'mahalanobis_turbulence.py',
    line: 14,
    text: '1.0 buy-and-hold, max drawdown 6% vs 32% (SPY/SHY, 2000-2022).'
  },
  {
    file: 'mahalanobis_turbulence.py',
    line: 25,
    text: 'Requires a pandas DataFrame with at least 65 rows (for rolling covariance).'
  },
  {
    file: 'mahalanobis_turbulence.py',
    line: 27,
    text: 'Data: 5-asset daily closes via yfinance (free, no API key).'
  },
  {
    file: 'mean_reversion.py',
    line: 17,
    text: 'and at least 20 rows of data (for Bollinger Band calculation).'
  },
  {
    file: 'metals_cross_asset.py',
    line: 14,
    text: 'Gold and silver interpret some signals differently (e.g. G/S ratio).'
  },
  {
    file: 'metals_cross_asset.py',
    line: 17,
    text: 'XAG 3h (178 BUY / 1 SELL bias over 179 samples). Root cause was using'
  },
  {
    file: 'metals_cross_asset.py',
    line: 25,
    text: 'EPU improves gold RMSE by ~18% (Baker/Bloom/Davis 2016). TIPS real yield'
  },
  {
    file: 'network_momentum.py',
    line: 3,
    text: 'Academic basis: Pu et al. (2023), "Network Momentum across Asset Classes",'
  },
  {
    file: 'network_momentum.py',
    line: 12,
    text: 'BUY opportunity (momentum catch-up).  When peers show negative momentum'
  },
  {
    file: 'network_momentum.py',
    line: 13,
    text: 'but the target leads, SELL (expecting pull-down).'
  },
  {
    file: 'network_momentum.py',
    line: 20,
    text: 'Data: yfinance for peer asset daily closes (free, no API key).'
  },
  {
    file: 'network_momentum.py',
    line: 21,
    text: 'Applicable to: all 5 tickers (BTC-USD, ETH-USD, XAU-USD, XAG-USD, MSTR).'
  },
  {
    file: 'orderbook_flow.py',
    line: 11,
    text: 'Applicable to metals (XAU-USD, XAG-USD) and crypto (BTC-USD, ETH-USD).'
  },
  {
    file: 'ovx_metals_spillover.py',
    line: 3,
    text: 'Crude Oil Volatility Index (OVX) at extreme quantiles predicts precious'
  },
  {
    file: 'ovx_metals_spillover.py',
    line: 10,
    text: 'predicts NEGATIVE metals returns (stress sells everything including gold).'
  },
  {
    file: 'ovx_metals_spillover.py',
    line: 14,
    text: 'This signal uses oil IMPLIED VOLATILITY (expectation of future moves).'
  },
  {
    file: 'ovx_metals_spillover.py',
    line: 16,
    text: 'Metals-only (XAU-USD, XAG-USD). Returns HOLD for other tickers.'
  },
  {
    file: 'ovx_metals_spillover.py',
    line: 24,
    text: 'Data: yfinance ^OVX (CBOE Oil Volatility Index), free, daily.'
  },
  {
    file: 'realized_skewness.py',
    line: 19,
    text: 'and at least 60 rows of data (for z-score normalization).'
  },
  {
    file: 'residual_pair_reversion.py',
    line: 27,
    text: 'and at least 200 rows of data (180-bar OLS window + 20 for z-scoring).'
  },
  {
    file: 'shannon_entropy.py',
    line: 5,
    text: 'noisy/random regime (HOLD — no edge). This is fundamentally different from'
  },
  {
    file: 'shannon_entropy.py',
    line: 6,
    text: 'volatility: a market can be volatile but trending (low entropy) or calm but'
  },
  {
    file: 'shannon_entropy.py',
    line: 7,
    text: 'choppy (high entropy).'
  },
  {
    file: 'shannon_entropy.py',
    line: 15,
    text: 'Source: Richard Shu (2025), Shannon Entropy breakout filter (29.6% Sharpe improvement).'
  },
  {
    file: 'statistical_jump_regime.py',
    line: 3,
    text: 'Detects market regime (bull/bear/neutral) using statistical jump detection'
  },
  {
    file: 'statistical_jump_regime.py',
    line: 7,
    text: '(12 citations). Simplified implementation using threshold-based jump detection'
  },
  {
    file: 'statistical_jump_regime.py',
    line: 18,
    text: 'and at least 50 rows of data (for rolling volatility + SMA baseline).'
  },
  {
    file: 'treasury_risk_rotation.py',
    line: 3,
    text: 'Uses relative performance of IEF (7-10Y Treasury) vs TLT (20Y+ Treasury)'
  },
  {
    file: 'treasury_risk_rotation.py',
    line: 4,
    text: 'as a cross-asset risk regime detector.  Steepening curve (TLT outperforms)'
  },
  {
    file: 'treasury_risk_rotation.py',
    line: 5,
    text: 'signals risk-on; flattening/inverting (IEF outperforms) signals risk-off.'
  },
  {
    file: 'treasury_risk_rotation.py',
    line: 14,
    text: '- Risk-on assets (BTC, ETH, MSTR): steepening = BUY, flattening = SELL'
  },
  {
    file: 'treasury_risk_rotation.py',
    line: 15,
    text: '- Safe havens (XAU, XAG):          steepening = SELL, flattening = BUY'
  },
  {
    file: 'treasury_risk_rotation.py',
    line: 17,
    text: 'Data: yfinance IEF + TLT (free, no API key). Cached 1 hour.'
  },
  {
    file: 'treasury_risk_rotation.py',
    line: 18,
    text: 'Source: Gayed (2014). An Intermarket Approach to Tactical Risk Rotation.'
  },
  {
    file: 'vix_term_structure.py',
    line: 4,
    text: '(VIX > VIX3M) occurs <20% of the time and signals market stress.'
  },
  {
    file: 'vix_term_structure.py',
    line: 5,
    text: 'Deep contango (ratio < 0.85) signals complacency — contrarian BUY'
  },
  {
    file: 'vix_term_structure.py',
    line: 9,
    text: 'GLD 55%/61% at 1d/3d). Weaker on BTC/equities (~40-50%). Applied'
  },
  {
    file: 'vix_term_structure.py',
    line: 19,
    text: 'Data: yfinance ^VIX and ^VIX3M (free, no API key).'
  },
  {
    file: 'vol_ratio_regime.py',
    line: 18,
    text: 'Directional logic (per regime):'
  },
  ... 10 more items
]
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "''^[A-Z_a-z][A-Z_a-z0-9]*'"\\s*[:=]["'^'"\\n]*\\{\\}|"'^[A-Z_a-z][A-Z_a-z0-9]*'"\\s*=\\s*dict\\(\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
 succeeded in 689ms:
Q:\finance-analyzer\portfolio\signals\forecast.py:132:_last_prediction_ts: dict[str, float] = {}  # ticker -> monotonic timestamp
Q:\finance-analyzer\portfolio\signals\credit_spread.py:53:_oas_cache: dict = {}
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:43:_CACHE: dict = {}
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:51:_hash_cache: dict[str, Any] = {}
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:148:_earnings_cache = {"data": {}, "ts": 0}
Q:\finance-analyzer\portfolio\signals\gold_real_yield_paradox.py:39:_yield_cache: dict = {}
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:51:_DVOL_CACHE: dict[str, tuple[float, float]] = {}  # currency -> (timestamp, dvol)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:53:_DVOL_HISTORY_CACHE: dict[str, tuple[float, pd.Series]] = {}
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:85:_epu_cache: dict = {}
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:86:_tips_cache: dict = {}
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"return None|raise [A-Za-z_]+\\(|raise [A-Za-z_]+"'$" Q:'"\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"return \\{|return result|return empty|return _HOLD|return HOLD|return \\(\\\"HOLD\\\"|return \\{\\\"action\\\": \\\"HOLD\\\"\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
2026-05-11T15:31:15.305790Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:return \{|return result|return empty|return _HOLD|return HOLD|return \(\)
    ^
error: unclosed group

 exited 1 in 736ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:return \{|return result|return empty|return _HOLD|return HOLD|return \(\)
    ^
error: unclosed group

 succeeded in 745ms:
Q:\finance-analyzer\portfolio\signals\forecast.py:145:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:149:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:180:    return None
Q:\finance-analyzer\portfolio\signals\forecast.py:294:            return None
Q:\finance-analyzer\portfolio\signals\forecast.py:310:    return None
Q:\finance-analyzer\portfolio\signals\forecast.py:316:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:318:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:324:                return None
Q:\finance-analyzer\portfolio\signals\forecast.py:331:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:367:            return None
Q:\finance-analyzer\portfolio\signals\forecast.py:371:            return None
Q:\finance-analyzer\portfolio\signals\forecast.py:379:            return None
Q:\finance-analyzer\portfolio\signals\forecast.py:383:            return None
Q:\finance-analyzer\portfolio\signals\forecast.py:391:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:398:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:403:            return None
Q:\finance-analyzer\portfolio\signals\forecast.py:424:                return None
Q:\finance-analyzer\portfolio\signals\forecast.py:438:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:507:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:511:        return None
Q:\finance-analyzer\portfolio\signals\forecast.py:520:        return None
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:81:                return None
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:92:                return None
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:97:            return None
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:112:        return None
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:120:            return None
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:183:        return None
Q:\finance-analyzer\portfolio\signals\network_momentum.py:85:                return None
Q:\finance-analyzer\portfolio\signals\network_momentum.py:97:                return None
Q:\finance-analyzer\portfolio\signals\network_momentum.py:102:            return None
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:48:        return None
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:53:        return None
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:61:        return None
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:85:                return None
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:92:            return None
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:201:        return None
Q:\finance-analyzer\portfolio\signals\futures_basis.py:64:        return None
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:31:# loaders would return None, the signal would silently fall back to API
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:65:        return None
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:71:        return None
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:131:        return None
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:135:        return None
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:145:        return None
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:67:                return None
Q:\finance-analyzer\portfolio\signals\cross_asset_tsmom.py:102:            return None
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:95:            return None
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:101:            return None
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:112:        return None
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:163:            return None
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:176:        return None
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:509:        raise RuntimeError(f"claude_gate returned exit_code={exit_code}")
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:883:    return None
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:65:            return None
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:72:            return None
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:79:            return None
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:87:        return None
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:74:                return None
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:85:                return None
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:90:            return None
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:101:        return None
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:106:        return None
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:152:        return None
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:50:        return None
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:58:        return None
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:65:    return None
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:89:            return None
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:92:        return None
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:94:        return None
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:101:        return None
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:110:        return None
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:58:                return None
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:63:                return None
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:70:                return None
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:75:            return None
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:52:            return None
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:64:            return None
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:68:            return None
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:78:            return None
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:90:        return None
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:63:                return None
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:107:            return None
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
Two look-ahead candidates stand out from the scan: `smart_money` because swing/structure logic often cheats with future confirmation, and `heikin_ashi` because it explicitly mentions forward-projected Alligator lines. I’m reading those closely first.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\smart_money.py | Select-Object -First 260" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\heikin_ashi.py | Select-Object -First 430" in Q:\finance-analyzer
 succeeded in 745ms:
"""Composite Smart Money Concepts (SMC) and market structure signal.

Combines five sub-indicators into a single BUY/SELL/HOLD vote via majority
voting:
    1. Break of Structure (BOS) - swing high/low breakouts
    2. Change of Character (CHoCH) - trend reversal detection
    3. Fair Value Gap (FVG) - unfilled 3-candle gaps being revisited
    4. Liquidity Sweep / Stop Hunt - wick-based fake-out reversals
    5. Supply and Demand Zones - institutional order flow zones

Requires a pandas DataFrame with columns: open, high, low, close, volume.
At least 50 rows recommended; returns HOLD on insufficient data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

logger = logging.getLogger(__name__)

MIN_ROWS = 50
_SWING_LOOKBACK = 3          # bars on each side for swing detection
_FVG_SCAN_BARS = 20          # how far back to scan for unfilled FVGs
_LIQUIDITY_SWEEP_PCT = 0.005 # wick must exceed extreme by >0.5%
_STRONG_BODY_MULT = 2.0      # body > 2x avg body = strong candle
_SUPPLY_DEMAND_LOOKBACK = 30 # bars to scan for S/D zones
_ZONE_PROXIMITY_PCT = 0.005  # within 0.5% of zone boundary counts as "in zone"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body(open_s: pd.Series, close_s: pd.Series) -> pd.Series:
    """Absolute candle body size."""
    return (close_s - open_s).abs()


def _find_swing_highs(highs: np.ndarray, lookback: int = _SWING_LOOKBACK) -> list[tuple[int, float]]:
    """Find swing highs: bar whose high > high of `lookback` bars on each side.

    Returns list of (index, high_value) tuples sorted by index ascending.
    """
    swings: list[tuple[int, float]] = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing = False
                break
        if is_swing:
            swings.append((i, float(highs[i])))
    return swings


def _find_swing_lows(lows: np.ndarray, lookback: int = _SWING_LOOKBACK) -> list[tuple[int, float]]:
    """Find swing lows: bar whose low < low of `lookback` bars on each side.

    Returns list of (index, low_value) tuples sorted by index ascending.
    """
    swings: list[tuple[int, float]] = []
    n = len(lows)
    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing = False
                break
        if is_swing:
            swings.append((i, float(lows[i])))
    return swings


# ---------------------------------------------------------------------------
# Sub-indicator 1: Break of Structure (BOS)
# ---------------------------------------------------------------------------

def _detect_bos(
    highs: np.ndarray,
    lows: np.ndarray,
    close: np.ndarray,
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
) -> tuple[str, dict]:
    """Detect Break of Structure on the most recent bar.

    Bullish BOS: current close breaks above the most recent swing high.
    Bearish BOS: current close breaks below the most recent swing low.

    Returns (vote, indicators_dict).
    """
    indicators: dict = {"last_swing_high": np.nan, "last_swing_low": np.nan}

    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return "HOLD", indicators

    last_sh_idx, last_sh_val = swing_highs[-1]
    last_sl_idx, last_sl_val = swing_lows[-1]
    indicators["last_swing_high"] = last_sh_val
    indicators["last_swing_low"] = last_sl_val

    current_close = float(close[-1])

    # Only count as BOS if the swing was detected *before* the current bar
    # (swing detection requires lookback bars on the right, so it is always
    # at least _SWING_LOOKBACK bars before the end).
    bullish_bos = current_close > last_sh_val
    bearish_bos = current_close < last_sl_val

    if bullish_bos and not bearish_bos:
        return "BUY", indicators
    if bearish_bos and not bullish_bos:
        return "SELL", indicators
    # Both or neither
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Sub-indicator 2: Change of Character (CHoCH)
# ---------------------------------------------------------------------------

def _detect_choch(
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
) -> tuple[str, str]:
    """Detect Change of Character from swing sequence.

    Bullish CHoCH: bearish structure (lower highs + lower lows) makes a
        higher low followed by a higher high.
    Bearish CHoCH: bullish structure (higher highs + higher lows) makes a
        lower high followed by a lower low.

    Returns (vote, structure_label).
    """
    # Need at least 3 swing highs and 3 swing lows to assess structure change
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "HOLD", "neutral"

    sh_vals = [v for _, v in swing_highs]
    sl_vals = [v for _, v in swing_lows]

    # Assess the last 3 swings for pattern
    # Previous structure (using swings [-3] and [-2])
    prev_hh = sh_vals[-2] > sh_vals[-3]  # was making higher highs
    prev_hl = sl_vals[-2] > sl_vals[-3]  # was making higher lows
    prev_lh = sh_vals[-2] < sh_vals[-3]  # was making lower highs
    prev_ll = sl_vals[-2] < sl_vals[-3]  # was making lower lows

    # Current structure change (using swings [-2] and [-1])
    curr_hh = sh_vals[-1] > sh_vals[-2]
    curr_hl = sl_vals[-1] > sl_vals[-2]
    curr_lh = sh_vals[-1] < sh_vals[-2]
    curr_ll = sl_vals[-1] < sl_vals[-2]

    # Bearish-to-bullish CHoCH: was making lower highs/lows, now higher
    # low AND higher high
    if (prev_lh or prev_ll) and curr_hl and curr_hh:
        return "BUY", "bullish"

    # Bullish-to-bearish CHoCH: was making higher highs/lows, now lower
    # high AND lower low
    if (prev_hh or prev_hl) and curr_lh and curr_ll:
        return "SELL", "bearish"

    # Determine current structure label without a change
    if curr_hh and curr_hl:
        return "HOLD", "bullish"
    if curr_lh and curr_ll:
        return "HOLD", "bearish"

    return "HOLD", "neutral"


# ---------------------------------------------------------------------------
# Sub-indicator 3: Fair Value Gap (FVG)
# ---------------------------------------------------------------------------

def _detect_fvg(
    highs: np.ndarray,
    lows: np.ndarray,
    close: np.ndarray,
    scan_bars: int = _FVG_SCAN_BARS,
) -> tuple[str, int]:
    """Detect Fair Value Gaps and check if current price is filling one.

    Bullish FVG (gap up): candle[i+2].low > candle[i].high
        -> price dropping back into this gap = BUY (filling from above)
    Bearish FVG (gap down): candle[i+2].high < candle[i].low
        -> price rising back into this gap = SELL (filling from below)

    Returns (vote, unfilled_fvg_count).
    """
    n = len(highs)
    current_close = float(close[-1])
    unfilled_bullish: list[tuple[float, float]] = []  # (gap_low, gap_high)
    unfilled_bearish: list[tuple[float, float]] = []

    start = max(0, n - scan_bars - 2)

    for i in range(start, n - 2):
        candle1_high = float(highs[i])
        candle1_low = float(lows[i])
        candle3_low = float(lows[i + 2])
        candle3_high = float(highs[i + 2])

        # Bullish FVG: gap up - candle 3 low is above candle 1 high
        if candle3_low > candle1_high:
            gap_low = candle1_high
            gap_high = candle3_low
            # Check if gap has been filled by any subsequent bar
            filled = False
            for j in range(i + 3, n):
                if float(lows[j]) <= gap_low:
                    filled = True
                    break
            if not filled:
                unfilled_bullish.append((gap_low, gap_high))

        # Bearish FVG: gap down - candle 3 high is below candle 1 low
        if candle3_high < candle1_low:
            gap_high = candle1_low
            gap_low = candle3_high
            filled = False
            for j in range(i + 3, n):
                if float(highs[j]) >= gap_high:
                    filled = True
                    break
            if not filled:
                unfilled_bearish.append((gap_low, gap_high))

    total_unfilled = len(unfilled_bullish) + len(unfilled_bearish)

    # Check if current price is filling any unfilled FVG
    filling_bullish = any(
        gap_low <= current_close <= gap_high
        for gap_low, gap_high in unfilled_bullish
    )
    filling_bearish = any(
        gap_low <= current_close <= gap_high
        for gap_low, gap_high in unfilled_bearish
    )

    if filling_bullish and not filling_bearish:
        return "BUY", total_unfilled
    if filling_bearish and not filling_bullish:
        return "SELL", total_unfilled
    return "HOLD", total_unfilled


# ---------------------------------------------------------------------------
# Sub-indicator 4: Liquidity Sweep / Stop Hunt
# ---------------------------------------------------------------------------

[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 741ms:
"""Composite Heikin-Ashi and advanced trend signal module.

Computes 7 sub-indicators using Heikin-Ashi candles, Hull Moving Average,
Williams Alligator, Elder Impulse System, and TTM Squeeze, then returns a
composite BUY/SELL/HOLD vote via majority voting.

Sub-indicators:
    1. Heikin-Ashi Trend       - 3 consecutive strong HA candles (no wick)
    2. Heikin-Ashi Doji        - HA doji reversal after streak
    3. Heikin-Ashi Color Change - HA candle color transition
    4. Hull Moving Average Cross(9, 21)
    5. Alligator Indicator     - SMMA(13/8/5) with forward shifts
    6. Elder Impulse System    - EMA(13) + MACD histogram direction
    7. TTM Squeeze             - Bollinger inside Keltner + momentum

Requires a DataFrame with columns: open, high, low, close, volume.
At least 50 rows recommended; returns HOLD with 0.0 confidence when
insufficient data is available.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from portfolio.signal_utils import (
    ema,
    majority_vote,
    rma,
    safe_float,
    sma,
    true_range,
    wma,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_ROWS = 50          # minimum rows for meaningful output
_MIN_ROWS_BASIC = 10    # absolute minimum to attempt any calculation
_NUM_SUB_SIGNALS = 7
_DOJI_BODY_PCT = 0.10   # body < 10% of range = doji
_HA_STREAK_LEN = 3      # consecutive candles for strong trend


def _hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average.

    HMA(n) = WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )
    """
    half_period = max(int(round(period / 2)), 1)
    sqrt_period = max(int(round(math.sqrt(period))), 1)

    wma_half = wma(series, half_period)
    wma_full = wma(series, period)

    raw = 2.0 * wma_half - wma_full
    return wma(raw, sqrt_period)


# ---------------------------------------------------------------------------
# Heikin-Ashi candle computation
# ---------------------------------------------------------------------------

def _compute_ha_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi OHLC from regular OHLC data.

    HA_Close = (O + H + L + C) / 4
    HA_Open[0] = (O[0] + C[0]) / 2
    HA_Open[i] = (HA_Open[i-1] + HA_Close[i-1]) / 2
    HA_High = max(H, HA_Open, HA_Close)
    HA_Low  = min(L, HA_Open, HA_Close)
    """
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(o)

    ha_close = (o + h + lo + c) / 4.0

    ha_open = np.empty(n, dtype=float)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))

    return pd.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Sub-signal 1: Heikin-Ashi Trend (strong candles with no opposing wicks)
# ---------------------------------------------------------------------------

def _ha_trend_signal(ha: pd.DataFrame) -> tuple[str, str, int]:
    """Check last N HA candles for strong trend (no opposing wicks).

    Strong BUY:  last 3 candles all green AND no lower wicks
                 (ha_low == ha_open for each candle).
    Strong SELL: last 3 candles all red AND no upper wicks
                 (ha_high == ha_open for each candle).

    Returns (signal, ha_color, ha_streak).
    """
    n = len(ha)
    if n < _HA_STREAK_LEN:
        last = ha.iloc[-1]
        color = "green" if last["ha_close"] > last["ha_open"] else "red"
        return "HOLD", color, 0

    # Determine color and streak of the last candle
    last = ha.iloc[-1]
    is_green_last = last["ha_close"] > last["ha_open"]
    color = "green" if is_green_last else "red"

    # Count consecutive same-color candles from the end
    streak = 0
    for i in range(n - 1, -1, -1):
        row = ha.iloc[i]
        row_green = row["ha_close"] > row["ha_open"]
        if row_green == is_green_last:
            streak += 1
        else:
            break

    # Check the last _HA_STREAK_LEN candles for the strong pattern
    tail = ha.iloc[-_HA_STREAK_LEN:]

    # Use a small tolerance for wick comparison (floating-point precision)
    tol = 1e-10

    all_green_no_lower_wick = True
    all_red_no_upper_wick = True

    for i in range(len(tail)):
        row = tail.iloc[i]
        is_green = row["ha_close"] > row["ha_open"]
        is_red = row["ha_close"] < row["ha_open"]
        ha_range = row["ha_high"] - row["ha_low"]

        if ha_range < tol:
            # Negligible range candle -- not a strong trend candle
            all_green_no_lower_wick = False
            all_red_no_upper_wick = False
            break

        # Green candle: no lower wick means ha_low == ha_open
        if not is_green or abs(row["ha_low"] - row["ha_open"]) > tol * ha_range + tol:
            all_green_no_lower_wick = False

        # Red candle: no upper wick means ha_high == ha_open
        if not is_red or abs(row["ha_high"] - row["ha_open"]) > tol * ha_range + tol:
            all_red_no_upper_wick = False

    if all_green_no_lower_wick:
        return "BUY", color, streak
    if all_red_no_upper_wick:
        return "SELL", color, streak
    return "HOLD", color, streak


# ---------------------------------------------------------------------------
# Sub-signal 2: Heikin-Ashi Doji (reversal after streak)
# ---------------------------------------------------------------------------

def _ha_doji_signal(ha: pd.DataFrame) -> str:
    """Detect HA doji candle and infer reversal direction from prior streak.

    Doji: body < 10% of range on the last candle.
    After green streak -> doji = potential SELL reversal.
    After red streak   -> doji = potential BUY reversal.
    """
    n = len(ha)
    if n < 2:
        return "HOLD"

    last = ha.iloc[-1]
    ha_range = last["ha_high"] - last["ha_low"]
    if ha_range <= 0:
        return "HOLD"

    body = abs(last["ha_close"] - last["ha_open"])
    body_pct = body / ha_range

    if body_pct >= _DOJI_BODY_PCT:
        return "HOLD"  # Not a doji

    # Doji detected -- determine prior streak direction
    # Look at the candle before the doji to determine the streak
    prev_streak_color = None
    streak_count = 0
    for i in range(n - 2, -1, -1):
        row = ha.iloc[i]
        is_green = row["ha_close"] > row["ha_open"]
        if prev_streak_color is None:
            prev_streak_color = "green" if is_green else "red"
            streak_count = 1
        elif (is_green and prev_streak_color == "green") or \
             (not is_green and prev_streak_color == "red"):
            streak_count += 1
        else:
            break

    # Need at least 2 consecutive candles in one direction before the doji
    if streak_count < 2:
        return "HOLD"

    if prev_streak_color == "green":
        return "SELL"  # Doji after green streak = potential top reversal
    elif prev_streak_color == "red":
        return "BUY"   # Doji after red streak = potential bottom reversal

    return "HOLD"


# ---------------------------------------------------------------------------
# Sub-signal 3: Heikin-Ashi Color Change
# ---------------------------------------------------------------------------

def _ha_color_change_signal(ha: pd.DataFrame) -> str:
    """Detect HA color transition on the most recent bar.

    Red -> Green = BUY.
    Green -> Red = SELL.
    Same color   = HOLD.
    """
    n = len(ha)
    if n < 2:
        return "HOLD"

    prev = ha.iloc[-2]
    curr = ha.iloc[-1]

    prev_green = prev["ha_close"] > prev["ha_open"]
    curr_green = curr["ha_close"] > curr["ha_open"]

    # Handle flat candles (close == open) as continuation
    if abs(prev["ha_close"] - prev["ha_open"]) < 1e-12:
        return "HOLD"
    if abs(curr["ha_close"] - curr["ha_open"]) < 1e-12:
        return "HOLD"

    if not prev_green and curr_green:
        return "BUY"   # Red to green
    if prev_green and not curr_green:
        return "SELL"  # Green to red

    return "HOLD"


# ---------------------------------------------------------------------------
# Sub-signal 4: Hull Moving Average Cross (9, 21)
# ---------------------------------------------------------------------------

def _hull_ma_signal(close: pd.Series) -> tuple[str, float, float]:
    """HMA(9) crossing HMA(21).

    HMA(9) > HMA(21) = BUY (fast above slow).
    HMA(9) < HMA(21) = SELL (fast below slow).

    Returns (signal, hull_fast, hull_slow).
    """
    hma_fast = _hma(close, 9)
    hma_slow = _hma(close, 21)

    fast_val = hma_fast.iloc[-1]
    slow_val = hma_slow.iloc[-1]

    if pd.isna(fast_val) or pd.isna(slow_val):
        return "HOLD", float("nan"), float("nan")

    fast_f = float(fast_val)
    slow_f = float(slow_val)

    if fast_f > slow_f:
        return "BUY", fast_f, slow_f
    elif fast_f < slow_f:
        return "SELL", fast_f, slow_f

    return "HOLD", fast_f, slow_f


# ---------------------------------------------------------------------------
# Sub-signal 5: Alligator Indicator
# ---------------------------------------------------------------------------

def _alligator_signal(close: pd.Series) -> tuple[str, float, float, float]:
    """Williams Alligator.

    Jaw   = SMMA(13) shifted forward 8 bars.
    Teeth = SMMA(8) shifted forward 5 bars.
    Lips  = SMMA(5) shifted forward 3 bars.

    Lips > Teeth > Jaw = BUY  (awakening upward).
    Lips < Teeth < Jaw = SELL (awakening downward).
    Otherwise          = HOLD (sleeping / intertwined).

    Returns (signal, lips, teeth, jaw).
    """
    jaw_raw = rma(close, 13)
    teeth_raw = rma(close, 8)
    lips_raw = rma(close, 5)

    # Shift forward (the Alligator projects lines into the future)
    jaw = jaw_raw.shift(8)
    teeth = teeth_raw.shift(5)
    lips = lips_raw.shift(3)

    jaw_val = jaw.iloc[-1]
    teeth_val = teeth.iloc[-1]
    lips_val = lips.iloc[-1]

    if pd.isna(jaw_val) or pd.isna(teeth_val) or pd.isna(lips_val):
        return "HOLD", float("nan"), float("nan"), float("nan")

    jaw_f = float(jaw_val)
    teeth_f = float(teeth_val)
    lips_f = float(lips_val)

    if lips_f > teeth_f > jaw_f:
        return "BUY", lips_f, teeth_f, jaw_f
    elif lips_f < teeth_f < jaw_f:
        return "SELL", lips_f, teeth_f, jaw_f

    return "HOLD", lips_f, teeth_f, jaw_f


# ---------------------------------------------------------------------------
# Sub-signal 6: Elder Impulse System
# ---------------------------------------------------------------------------

def _elder_impulse_signal(close: pd.Series) -> tuple[str, str]:
    """Elder Impulse System.

    Combines EMA(13) direction with MACD histogram direction:
      - EMA rising  + MACD-hist rising  = green (BUY)
      - EMA falling + MACD-hist falling = red   (SELL)
      - Mixed                           = blue  (HOLD)

    MACD uses standard (12, 26, 9) parameters.

    Returns (signal, elder_color).
    """
    ema13 = ema(close, 13)

    # MACD: fast EMA(12) - slow EMA(26)
    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    if len(ema13.dropna()) < 2 or len(macd_hist.dropna()) < 2:
        return "HOLD", "blue"

    ema_curr = ema13.iloc[-1]
    ema_prev = ema13.iloc[-2]
    hist_curr = macd_hist.iloc[-1]
    hist_prev = macd_hist.iloc[-2]

    if pd.isna(ema_curr) or pd.isna(ema_prev) or \
       pd.isna(hist_curr) or pd.isna(hist_prev):
        return "HOLD", "blue"

    ema_rising = float(ema_curr) > float(ema_prev)
    ema_falling = float(ema_curr) < float(ema_prev)
    hist_rising = float(hist_curr) > float(hist_prev)
    hist_falling = float(hist_curr) < float(hist_prev)

    if ema_rising and hist_rising:
        return "BUY", "green"
    if ema_falling and hist_falling:
        return "SELL", "red"

    return "HOLD", "blue"


# ---------------------------------------------------------------------------
# Sub-signal 7: TTM Squeeze
# ---------------------------------------------------------------------------

def _ttm_squeeze_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> tuple[str, bool, float]:
    """TTM Squeeze.

    Squeeze ON  = Bollinger Bands fully inside Keltner Channels.
    Squeeze OFF = Bollinger Bands expand outside Keltner Channels.

    When squeeze releases:
      - Momentum (close - midline of Donchian/linear regression) positive = BUY
      - Momentum negative = SELL
    While squeeze is ON = HOLD.

    Momentum is approximated as the difference between the close and the
    midpoint of the highest-high and lowest-low over the same period (a
    simple Donchian midline), which closely matches TTM Squeeze momentum.

    Returns (signal, squeeze_on, momentum).
    """
    # Bollinger Bands
    bb_mid = sma(close, bb_period)
    bb_std = close.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # Keltner Channels (using ATR)
    tr = true_range(high, low, close)
    atr = ema(tr, kc_period)
    kc_mid = ema(close, kc_period)
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"polyfit|linregress|OLS|LinearRegression|ewm\\(|quantile\\(|percentile|zscore|rolling\\(.*mean\\(|expanding\\(|cum\" Q:\\finance-analyzer\\portfolio\\signals -g \"*.py\"" in Q:\finance-analyzer
 succeeded in 700ms:
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:56:# Absorption ratio threshold (percentile)
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:229:    """Vote based on absorption ratio percentile."""
Q:\finance-analyzer\portfolio\signals\mahalanobis_turbulence.py:326:            "ar_percentile": safe_float(
Q:\finance-analyzer\portfolio\signals\futures_flow.py:236:    from portfolio.tickers import CRYPTO_SYMBOLS
Q:\finance-analyzer\portfolio\signals\futures_flow.py:237:    if ticker not in CRYPTO_SYMBOLS:
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:151:        trend in ("strong_accumulation",)
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:153:        or (trend in ("accumulation",) and consecutive_neg >= 3)
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:210:    from portfolio.tickers import CRYPTO_SYMBOLS
Q:\finance-analyzer\portfolio\signals\crypto_macro.py:211:    if ticker not in CRYPTO_SYMBOLS:
Q:\finance-analyzer\portfolio\signals\macro_regime.py:50:    return series.rolling(window=period, min_periods=period).mean()
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:165:        from portfolio.tickers import STOCK_SYMBOLS
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:170:    for ticker in STOCK_SYMBOLS:
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:65:        cumdev = np.cumsum(chunk - mean_c)
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:66:        r = cumdev.max() - cumdev.min()
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:79:    log(R/S) = H * log(n) + c  via OLS to get the Hurst exponent H.
Q:\finance-analyzer\portfolio\signals\hurst_regime.py:97:    # OLS: H = slope of log(R/S) vs log(n)
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:166:def _compute_zscore(values: list[float], lookback: int = 252) -> float:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:260:    result["gvz_zscore"] = gvz["zscore"] if gvz else 0.0
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:261:    result["gs_ratio_zscore"] = gs_daily["zscore"] if gs_daily else 0.0
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:331:    gvz = ctx["gvz_zscore"]
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:341:    gsr = ctx["gs_ratio_zscore"]
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:409:    epu_zscore = _compute_zscore(epu_values) if epu_values else 0.0
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:410:    if epu_zscore > _EPU_ZSCORE_HIGH:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:412:    elif epu_zscore < _EPU_ZSCORE_LOW:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:445:            "gvz_zscore": round(gvz, 2),
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:446:            "gs_ratio_zscore": round(gsr, 2),
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:450:            "epu_zscore": round(epu_zscore, 2),
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:5:    1. COT Index: 156-week percentile of speculative net positioning (contrarian)
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:38:# COT Index thresholds (0-100 percentile scale)
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:139:    """Compute COT Index as percentile of current net positioning.
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:144:    if len(nc_net_history) < 10:  # Need minimum history for meaningful percentile
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:159:    """Sub-indicator 1: COT Index percentile (contrarian).
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:232:    indicators = {"mm_net": None, "mm_zscore": None}
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:257:    zscore = (mm_net - mean) / std
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:258:    indicators["mm_zscore"] = round(float(zscore), 2)
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:260:    if zscore > _MM_ZSCORE_HIGH:
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:262:    elif zscore < _MM_ZSCORE_LOW:
Q:\finance-analyzer\portfolio\signals\cot_positioning.py:374:    # Sub 1: COT Index (contrarian percentile)
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:347:    Returns (half_life, zscore, signal).
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:360:    # OLS regression: y = theta * x + c
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:381:    zscore = float((prices.iloc[-1] - mean_price) / std_price)
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:386:        return float(half_life), zscore, "HOLD"
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:388:    if zscore < -1.5:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:389:        return float(half_life), zscore, "BUY"
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:390:    if zscore > 1.5:
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:391:        return float(half_life), zscore, "SELL"
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:393:    return float(half_life), zscore, "HOLD"
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:444:            "zscore": float("nan"),
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:584:        indicators["zscore"] = safe_float(zs_val)
Q:\finance-analyzer\portfolio\signals\mean_reversion.py:589:        indicators["zscore"] = float("nan")
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:63:    # Read accumulated OFI, OFI z-score, multiscale, and spread z-score
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:65:    ofi_zscore = 0.0
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:73:            ofi_zscore = ms_state.get("ofi_zscore", 0.0)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:74:            sz = ms_state.get("spread_zscore", 0.0)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:84:        "ofi_zscore": ofi_zscore,
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:85:        "spread_zscore": sz,
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:153:    ofi_z = ctx.get("ofi_zscore", 0.0)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:173:    # applied below when spread_zscore > threshold.
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:174:    sz = ctx.get("spread_zscore", 0.0)
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:209:            "ofi_zscore": round(ctx.get("ofi_zscore", 0.0), 4),
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:211:            "spread_zscore": round(sz, 4),
Q:\finance-analyzer\portfolio\signals\candlestick.py:188:        slope = np.polyfit(x, closes, 1)[0]
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:43:PCTILE_WINDOW = 90  # Rolling window for percentile ranking
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:44:PCTILE_BUY = 10  # Below 10th percentile → BUY
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:45:PCTILE_SELL = 90  # Above 90th percentile → SELL
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:116:    """Fetch DVOL history for percentile/momentum computation.
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:204:def _evrp_percentile_signal(dvol_history: pd.Series | None,
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:207:    """Sub-signal 2: eVRP percentile rank in recent history.
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:209:    Returns (signal, percentile_value).
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:225:    # If lengths don't align, use simple percentile of current eVRP vs recent DVOL-RV
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:230:    # Use just the DVOL percentile as proxy (DVOL carries most of the signal)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:318:    # Step 4: Fetch DVOL history for percentile and momentum
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:324:    # Sub-signal 2: eVRP percentile
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:325:    pctile_vote, pctile_val = _evrp_percentile_signal(dvol_history, evrp, close)
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:342:            "evrp_percentile": pctile_vote,
Q:\finance-analyzer\portfolio\signals\crypto_evrp.py:349:            "evrp_percentile": safe_float(pctile_val),
Q:\finance-analyzer\portfolio\signals\momentum.py:93:    k = raw_k.rolling(window=smooth_k).mean()
Q:\finance-analyzer\portfolio\signals\momentum.py:94:    d = k.rolling(window=d_period).mean()
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:161:def _gap_zscore_vote(gap_z: float, is_safe_haven: bool) -> str:
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:242:    gap_mean = gap_df["gap"].rolling(60, min_periods=20).mean().iloc[-1]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:249:    zscore_vote = _gap_zscore_vote(gap_z, is_safe_haven)
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:253:    votes = [zscore_vote, slope_vote, corr_vote]
Q:\finance-analyzer\portfolio\signals\complexity_gap_regime.py:264:            "gap_zscore": zscore_vote,
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:117:def _ratio_zscore(ratio: pd.Series, window: int = 50) -> float:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:121:    rolling_mean = ratio.rolling(window).mean()
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:135:        sma_short = ratio.rolling(20).mean().iloc[-1]
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:136:        sma_long = ratio.rolling(50).mean().iloc[-1]
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:138:        sma_short = ratio.rolling(50).mean().iloc[-1]
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:139:        sma_long = ratio.rolling(200).mean().iloc[-1]
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:205:    zscore = _ratio_zscore(ratio, window=50)
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:206:    if zscore < -2.0:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:207:        zscore_vote = "SELL"  # Strong risk-off
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:208:    elif zscore < -1.5:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:209:        zscore_vote = "SELL"  # Risk-off
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:210:    elif zscore > 2.0:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:211:        zscore_vote = "BUY"   # Strong expansion
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:212:    elif zscore > 1.5:
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:213:        zscore_vote = "BUY"   # Expansion
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:215:        zscore_vote = "HOLD"
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:245:    votes = [zscore_vote, trend_vote, momentum_vote, spread_vote]
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:261:            "ratio_zscore": zscore_vote,
Q:\finance-analyzer\portfolio\signals\copper_gold_ratio.py:268:            "ratio_zscore": round(zscore, 4),
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:126:    sma30 = hashrate.rolling(HASH_FAST).mean()
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:127:    sma60 = hashrate.rolling(HASH_SLOW).mean()
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:195:    sma30 = hashrate.rolling(HASH_FAST).mean()
Q:\finance-analyzer\portfolio\signals\hash_ribbons.py:196:    sma60 = hashrate.rolling(HASH_SLOW).mean()
Q:\finance-analyzer\portfolio\signals\forecast.py:257:    from portfolio.tickers import SYMBOLS
Q:\finance-analyzer\portfolio\signals\forecast.py:259:    source_info = SYMBOLS.get(ticker, {})
Q:\finance-analyzer\portfolio\signals\forecast.py:527:        from portfolio.tickers import CRYPTO_SYMBOLS
Q:\finance-analyzer\portfolio\signals\forecast.py:528:        return ticker in CRYPTO_SYMBOLS
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:37:VOL_LOW_PCTILE = 25    # Below this percentile = low-vol regime
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:38:VOL_HIGH_PCTILE = 75   # Above this percentile = high-vol regime
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:54:    """Classify volatility regime using rolling percentile rank.
Q:\finance-analyzer\portfolio\signals\statistical_jump_regime.py:58:    # Use available history up to `window` bars for percentile rank
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:110:def _ratio_zscore(ratio_series: list[float]) -> tuple[float, str]:
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:167:    z_val, sub3 = _ratio_zscore(ratio_series)
Q:\finance-analyzer\portfolio\signals\vix_term_structure.py:193:            "ratio_zscore": sub3,
Q:\finance-analyzer\portfolio\signals\credit_spread.py:148:def _oas_zscore_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:151:        return "HOLD", {"oas_zscore": 0.0}
Q:\finance-analyzer\portfolio\signals\credit_spread.py:161:        return "HOLD", {"oas_zscore": 0.0}
Q:\finance-analyzer\portfolio\signals\credit_spread.py:163:    zscore = (current - mean) / std
Q:\finance-analyzer\portfolio\signals\credit_spread.py:166:    if zscore > _ZSCORE_RISK_OFF:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:169:    elif zscore < _ZSCORE_RISK_ON:
Q:\finance-analyzer\portfolio\signals\credit_spread.py:173:    return action, {"oas_zscore": round(zscore, 3), "oas_current": round(current, 2),
Q:\finance-analyzer\portfolio\signals\credit_spread.py:300:    zscore_action, zscore_ind = _oas_zscore_signal(values, safe_haven)
Q:\finance-analyzer\portfolio\signals\credit_spread.py:305:    votes = [zscore_action, mom_action, accel_action, crisis_action]
Q:\finance-analyzer\portfolio\signals\credit_spread.py:310:    indicators.update(zscore_ind)
Q:\finance-analyzer\portfolio\signals\credit_spread.py:320:            "oas_zscore": zscore_action,
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:49:def _sub_skew_zscore(returns: pd.Series) -> tuple[float, str, dict]:
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:215:    conf1, vote1, ind1 = _sub_skew_zscore(returns)
Q:\finance-analyzer\portfolio\signals\realized_skewness.py:221:        "skew_zscore": vote1,
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:106:def _sub_slope_zscore(spread_series: pd.Series) -> tuple[float, str]:
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:176:    zscore_val, v_zscore = _sub_slope_zscore(spread_series)
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:179:    votes = [v_direction, v_momentum, v_zscore, v_persistence]
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:196:            "slope_zscore": v_zscore,
Q:\finance-analyzer\portfolio\signals\treasury_risk_rotation.py:201:            "zscore": safe_float(zscore_val),
Q:\finance-analyzer\portfolio\signals\drift_regime_gate.py:116:    atr_val = tr.rolling(window=atr_period, min_periods=atr_period).mean().iloc[-1]
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:4:rolling OLS, then z-scores the residual to detect mean-reversion
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:16:    1. Residual Z-Score   — z-scored OLS residual (primary)
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:27:and at least 200 rows of data (180-bar OLS window + 20 for z-scoring).
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:41:MIN_ROWS = 200  # 180-bar OLS window + 20 for z-scoring
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:45:# OLS rolling regression window
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:46:_OLS_WINDOW = 180
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:99:    """Compute rolling OLS beta and residual.
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:162:    """Z-score of the OLS residual.
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:170:    mean = residual.rolling(window=z_lookback, min_periods=z_lookback).mean()
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:301:    # Rolling OLS regression
Q:\finance-analyzer\portfolio\signals\residual_pair_reversion.py:302:    beta, residual = _rolling_ols_beta(target_ret, driver_ret, _OLS_WINDOW)
Q:\finance-analyzer\portfolio\signals\vwap_zscore_mr.py:34:def compute_vwap_zscore_mr_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\structure.py:37:    ema_fast = close.ewm(span=fast, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\structure.py:38:    ema_slow = close.ewm(span=slow, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\structure.py:40:    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:74:def _wvf_percentile(wvf: pd.Series, pct_lookback: int = 50,
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:78:    Returns (percentile_value, signal).
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:113:    # Check if WVF is elevated (above 75th percentile of recent range)
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:205:    pct_val, pct_vote = _wvf_percentile(wvf)
Q:\finance-analyzer\portfolio\signals\williams_vix_fix.py:220:            "wvf_percentile": pct_vote,
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:19:    1. ovx_level     - percentile rank over 252d (>80th=SELL, <20th=BUY)
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:21:    3. ovx_zscore    - z-score over 20d mean (>1.5=SELL, <-1.5=BUY)
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:42:_LOOKBACK_PCTILE = 252  # 1 year for percentile rank
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:90:def _percentile_rank(series: list[float], lookback: int) -> float:
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:91:    """Compute percentile rank of latest value over lookback window."""
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:129:def _ovx_zscore_signal(series: list[float]) -> tuple[float, str]:
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:198:    pctile = _percentile_rank(series, _LOOKBACK_PCTILE)
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:201:    z_val, sub3 = _ovx_zscore_signal(series)
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:212:        "ovx_zscore": sub3,
Q:\finance-analyzer\portfolio\signals\ovx_metals_spillover.py:220:        "ovx_zscore": safe_float(z_val),
Q:\finance-analyzer\portfolio\signals\volatility.py:36:    return tr.ewm(span=period, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:55:    gk_sma = gk_var.rolling(window=window, min_periods=window).mean()
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:56:    cc_sma = cc_var.rolling(window=window, min_periods=window).mean()
Q:\finance-analyzer\portfolio\signals\vol_ratio_regime.py:157:    gk_cc_mean = gk_cc_series.rolling(60, min_periods=20).mean()
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:87:                ema_50 = close_series.ewm(span=_EMA_TREND, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:117:    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:118:    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:127:    ema_fast = close.ewm(span=_MACD_FAST, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:128:    ema_slow = close.ewm(span=_MACD_SLOW, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\xtrend_equity_spillover.py:130:    signal_line = macd_line.ewm(span=_MACD_SIGNAL, adjust=False).mean()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:7:    3. Accumulation/Distribution Line vs its 20-period SMA
Q:\finance-analyzer\portfolio\signals\volume_flow.py:56:    obv = (direction * volume).cumsum()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:57:    obv_sma = obv.rolling(window=20, min_periods=20).mean()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:63:    """Session VWAP: cumulative(volume * typical_price) / cumulative(volume),
Q:\finance-analyzer\portfolio\signals\volume_flow.py:67:    cumulated over the entire dataframe with no session reset, which made
Q:\finance-analyzer\portfolio\signals\volume_flow.py:77:    Falls back to the lifetime-cumulative behavior when the index is not
Q:\finance-analyzer\portfolio\signals\volume_flow.py:84:    # Build a per-bar session id from the index date. groupby+cumsum gives
Q:\finance-analyzer\portfolio\signals\volume_flow.py:85:    # us cumulative-within-session sums in a single vectorized pass.
Q:\finance-analyzer\portfolio\signals\volume_flow.py:95:        cum_vol = volume.groupby(session_id).cumsum()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:96:        cum_vp = vp.groupby(session_id).cumsum()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:98:        # Non-datetime index — fall back to the lifetime cumulative behavior.
Q:\finance-analyzer\portfolio\signals\volume_flow.py:99:        cum_vol = volume.cumsum()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:100:        cum_vp = vp.cumsum()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:103:    vwap = cum_vp / cum_vol.replace(0, np.nan)
Q:\finance-analyzer\portfolio\signals\volume_flow.py:109:    """Accumulation/Distribution Line and its 20-period SMA."""
Q:\finance-analyzer\portfolio\signals\volume_flow.py:114:    ad = (clv * volume).cumsum()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:115:    ad_sma = ad.rolling(window=20, min_periods=20).mean()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:155:    avg_gain = gain.rolling(window=period, min_periods=period).mean()
Q:\finance-analyzer\portfolio\signals\volume_flow.py:156:    avg_loss = loss.rolling(window=period, min_periods=period).mean()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\orderbook_flow.py | Select-Object -First 120" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\dxy_cross_asset.py | Select-Object -First 120" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\credit_spread.py | Select-Object -First 140" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\gold_real_yield_paradox.py | Select-Object -First 130" in Q:\finance-analyzer
 succeeded in 1040ms:
"""Orderbook flow signal - microstructure-based short-term prediction.

Signal #31.  Combines 6 microstructure sub-indicators via majority vote:
    1. Depth Imbalance: ln(V_bid) - ln(V_ask) direction
    2. Trade Flow Imbalance: signed volume ratio direction
    3. VPIN Toxicity: high VPIN confirms directional flow
    4. OFI Direction: order flow imbalance trend
    5. Spread Health: abnormally wide spread  caution
    6. Trade Pressure: trade-through imbalance (buy vs sell throughs)

Applicable to metals (XAU-USD, XAG-USD) and crypto (BTC-USD, ETH-USD).
Requires context: calls metals_orderbook + microstructure modules.
Returns HOLD for stock tickers or when data unavailable.
"""
from __future__ import annotations

import logging
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.orderbook_flow")

_APPLICABLE_TICKERS = {"XAU-USD", "XAG-USD", "BTC-USD", "ETH-USD"}

_DEPTH_IMBALANCE_THRESHOLD = 0.5
_TRADE_IMBALANCE_THRESHOLD = 0.3
_VPIN_HIGH = 0.6
_VPIN_TOXICITY = 0.7  # threshold for flagging vol expansion risk
_OFI_ZSCORE_THRESHOLD = 1.5  # z-score threshold (was absolute 5.0)
_OFI_THRESHOLD = 5.0  # fallback when z-score unavailable
_SPREAD_ZSCORE_DANGER = 2.0
_TRADE_THROUGH_THRESHOLD = 2


def _get_microstructure_context(ticker: str) -> dict | None:
    """Fetch live microstructure data for the given ticker."""
    try:
        from portfolio.metals_orderbook import get_orderbook_depth, get_recent_trades
        from portfolio.microstructure import (
            compute_vpin,
            depth_imbalance,
            detect_trade_throughs,
            trade_flow_imbalance,
        )
    except ImportError:
        logger.debug("Microstructure modules not available")
        return None

    depth = get_orderbook_depth(ticker, limit=20)
    trades = get_recent_trades(ticker, limit=200)
    if depth is None or trades is None:
        return None

    di = depth_imbalance(depth)
    tfi = trade_flow_imbalance(trades)
    vpin = compute_vpin(trades, n_buckets=20)
    tt = detect_trade_throughs(trades)

    if tfi is None:
        return None

    # Read accumulated OFI, OFI z-score, multiscale, and spread z-score
    ofi = 0.0
    ofi_zscore = 0.0
    sz = 0.0
    flow_acceleration = 0.0
    try:
        from portfolio.microstructure_state import load_persisted_state
        ms_state = load_persisted_state(ticker)
        if ms_state:
            ofi = ms_state.get("ofi", 0.0)
            ofi_zscore = ms_state.get("ofi_zscore", 0.0)
            sz = ms_state.get("spread_zscore", 0.0)
            flow_acceleration = ms_state.get("flow_acceleration", 0.0)
    except ImportError:
        pass

    return {
        "depth_imbalance": di,
        "trade_imbalance_ratio": tfi["imbalance_ratio"],
        "vpin": vpin if vpin is not None else 0.0,
        "ofi": ofi,
        "ofi_zscore": ofi_zscore,
        "spread_zscore": sz,
        "spread_bps": depth.get("spread_bps", 0.0),
        "trade_throughs": tt,
        "flow_acceleration": flow_acceleration,
    }


def compute_orderbook_flow_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute orderbook flow composite signal.

    Args:
        df: OHLCV DataFrame (unused - microstructure data fetched separately).
        context: dict with keys {ticker, config, macro, regime}.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))

    if ticker not in _APPLICABLE_TICKERS:
        return empty

    ctx = _get_microstructure_context(ticker)
    if ctx is None:
        return empty

    votes = []
    sub_signals = {}

    # Sub 1: Depth Imbalance
    di = ctx["depth_imbalance"]
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1042ms:
"""DXY cross-asset signal - inverse correlation with metals on intraday.

Why this exists: DXY (US Dollar Index) has Rı ~0.6 inverse correlation
with silver/gold on 1-3h horizons - arguably the single strongest
short-term directional driver for metals. The existing macro_regime
signal has DXY as 1 of 6 sub-indicators at daily resolution, diluting
its vote. This standalone signal captures DXY directly at 60m resolution
so intraday dollar moves translate to a real BUY/SELL vote for metals.

Metals-only (XAU-USD, XAG-USD). Crypto + equities get HOLD.

Direction: DXY up  USD strong  metals weak  SELL. DXY down  BUY.
Threshold: 0.15% 1h move. DXY typical 1h range is 0.05-0.10% - a 0.15%
move is a clearly directional hour. Confidence scales linearly with move
magnitude up to a 0.5% cap (a ~5 sigma move saturates confidence).

Complements rather than replaces macro_regime's daily DXY vote. Keep both.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("portfolio.signals.dxy_cross_asset")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}

# 1h DXY move threshold. Typical 1h DXY range is ~0.05-0.10%; 0.15% is a
# ~2-sigma move. Below this, signal votes HOLD.
_DXY_1H_THRESHOLD_PCT = 0.15

# Above this 1h move magnitude, confidence saturates at 1.0. A 0.5% 1h
# move in DXY is exceptional - think major macro data release.
_DXY_1H_CONFIDENCE_CAP_PCT = 0.5


def compute_dxy_cross_asset_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute DXY-direction vote for a metals ticker.

    Args:
        df: OHLCV DataFrame (unused - DXY data fetched separately).
        context: dict with ``ticker`` key. Other keys ignored.

    Returns:
        dict with action, confidence, sub_signals, indicators.
        HOLD with 0 confidence if ticker is not metals or DXY data is
        unavailable.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))
    if ticker not in _METALS_TICKERS:
        return empty

    try:
        from portfolio.macro_context import get_dxy_intraday
    except ImportError:
        logger.debug("macro_context.get_dxy_intraday not available")
        return empty

    data = get_dxy_intraday()
    if data is None:
        return empty

    change_1h = data.get("change_1h_pct")
    if change_1h is None:
        return empty

    if change_1h < -_DXY_1H_THRESHOLD_PCT:
        # Weak USD  strong metals  BUY
        action = "BUY"
        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
    elif change_1h > _DXY_1H_THRESHOLD_PCT:
        # Strong USD  weak metals  SELL
        action = "SELL"
        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
    else:
        action = "HOLD"
        confidence = 0.0

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {"dxy_1h": action},
        "indicators": {
            "dxy_value": data.get("value"),
            "dxy_change_1h_pct": round(change_1h, 3),
            "dxy_change_3h_pct": (
                round(data["change_3h_pct"], 3)
                if data.get("change_3h_pct") is not None else None
            ),
            "source": data.get("source", "unknown"),
        },
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1045ms:
"""Gold-Yield Divergence Index (GYDI) signal - macro regime detector.

Detects when gold rises WITH real yields (historically inverse). Three
sub-indicators via majority vote:
    1. Paradox Spread:  gold 30d return > 0 AND real yield 30d change > 0
    2. Correlation Break: 30d gold-yield correlation deviates from 3yr baseline
    3. Momentum Split:  gold trend vs yield trend divergence/convergence

Applies to XAU-USD (primary) and XAG-USD (secondary, via gold proxy).
Data: FRED DFII10 (10Y TIPS real yield), cached 4 hours.

Source: AHA Signals GYDI tracker; Valadkhani 2024 MSI-VAR.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

logger = logging.getLogger(__name__)

MIN_ROWS = 50
_FRED_SERIES = "DFII10"
_FRED_TIMEOUT = 15
_CACHE_TTL = 4 * 3600
_HISTORY_LIMIT = 800

_APPLICABLE_TICKERS = frozenset({
    "XAU-USD", "XAUUSD", "XAU/USD",
    "XAG-USD", "XAGUSD", "XAG/USD",
})

_yield_cache: dict = {}
_yield_cache_lock = threading.Lock()


def _get_fred_key(context: dict | None) -> str:
    if not context:
        return ""
    cfg = context.get("config")
    if not cfg:
        return ""
    if isinstance(cfg, dict):
        return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
    return getattr(cfg, "fred_api_key", "") or getattr(
        getattr(cfg, "golddigger", None), "fred_api_key", ""
    ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""


def _fetch_real_yield(fred_api_key: str) -> list[float] | None:
    """Fetch 10Y TIPS real yield from FRED. Returns list newest-first."""
    now = time.time()
    with _yield_cache_lock:
        if (
            _yield_cache.get("key") == fred_api_key
            and _yield_cache.get("data")
            and now - _yield_cache.get("time", 0) < _CACHE_TTL
        ):
            return _yield_cache["data"]

    if not fred_api_key:
        logger.debug("No FRED API key for real yield fetch")
        return _yield_cache.get("data")

    try:
        from portfolio.http_retry import fetch_with_retry
    except ImportError:
        logger.warning("http_retry not available")
        return _yield_cache.get("data")

    try:
        resp = fetch_with_retry(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": _FRED_SERIES,
                "api_key": fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": _HISTORY_LIMIT,
            },
            timeout=_FRED_TIMEOUT,
        )
        if hasattr(resp, "json"):
            data = resp.json()
        else:
            import json
            data = json.loads(resp.text if hasattr(resp, "text") else resp)

        observations = data.get("observations", [])
        values = []
        for obs in observations:
            val = obs.get("value", ".")
            if val == ".":
                continue
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue

        if values:
            with _yield_cache_lock:
                _yield_cache["key"] = fred_api_key
                _yield_cache["data"] = values
                _yield_cache["time"] = now
            logger.debug("Real yield fetched: %d obs, current=%.3f", len(values), values[0])
            return values

    except Exception:
        logger.warning("FRED real yield fetch failed", exc_info=True)

    return _yield_cache.get("data")


def _paradox_spread(gold_returns_30d: float, yield_change_30d: float) -> tuple[str, dict]:
    """Sub-indicator 1: gold and yield both rising (paradox regime = BUY)."""
    both_positive = gold_returns_30d > 0 and yield_change_30d > 0

    if both_positive:
        magnitude = (
            min(abs(gold_returns_30d) / 0.10, 1.0) * 50
            + min(abs(yield_change_30d) / 0.75, 1.0) * 50
        )
        action = "BUY"
    elif gold_returns_30d < 0 and yield_change_30d < 0:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1093ms:
"""Credit spread risk appetite signal - cross-asset macro indicator.

Signal #35.  Uses ICE BofA High Yield Option-Adjusted Spread (HY OAS)
from FRED API as a measure of institutional credit risk appetite.
Combines 4 sub-indicators via majority vote:
    1. OAS Level Z-Score:    where current spread sits vs 252d history
    2. OAS 5d Momentum:      rate of spread widening/tightening
    3. OAS Momentum Accel:   acceleration of momentum (early warning)
    4. Crisis Level Gate:    hard threshold at 500bp (every bear market)

Direction depends on asset class:
- Gold/Silver (safe haven):  risk-off  BUY, risk-on  SELL
- Crypto/Stocks (risk-on):   risk-off  SELL, risk-on  BUY

Data: FRED API series BAMLH0A0HYM2 (daily, free with API key).
Cached for 4 hours since data updates daily.

Requires context dict with keys: ticker, config (for FRED API key).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from portfolio.file_utils import load_json
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.credit_spread")

# ---------------------------------------------------------------------------
# Asset-class classification for directional interpretation
# ---------------------------------------------------------------------------
_SAFE_HAVEN = {"XAU-USD", "XAG-USD"}
_RISK_ASSETS = {"BTC-USD", "ETH-USD", "MSTR"}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_ZSCORE_RISK_OFF = 1.5      # OAS z-score above this = risk-off
_ZSCORE_RISK_ON = -1.0      # OAS z-score below this = risk-on (complacent)
_MOM_5D_THRESHOLD = 0.20    # 20bp 5d change = meaningful momentum
_ACCEL_THRESHOLD = 0.15     # 15bp acceleration = early warning
_CRISIS_LEVEL = 5.0         # 500bp = crisis threshold
_HISTORY_LIMIT = 300        # FRED observations to fetch (covers ~252 trading days)
_FRED_SERIES = "BAMLH0A0HYM2"
_FRED_TIMEOUT = 15
_CACHE_TTL = 4 * 3600       # 4 hours

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_oas_cache: dict = {}


def _fetch_hy_oas(fred_api_key: str) -> list[float] | None:
    """Fetch HY OAS history from FRED.  Returns list of floats (newest first).

    Cached for 4 hours since data is daily.
    """
    now = time.time()
    if (
        _oas_cache.get("key") == fred_api_key
        and _oas_cache.get("data")
        and now - _oas_cache.get("time", 0) < _CACHE_TTL
    ):
        return _oas_cache["data"]

    if not fred_api_key:
        logger.debug("No FRED API key - cannot fetch HY OAS")
        return _oas_cache.get("data")

    try:
        from portfolio.http_retry import fetch_with_retry
    except ImportError:
        import requests

        class _Shim:
            @staticmethod
            def __call__(url, **kwargs):
                return requests.get(url, **kwargs)
        fetch_with_retry = _Shim()

    try:
        resp = fetch_with_retry(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": _FRED_SERIES,
                "api_key": fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": _HISTORY_LIMIT,
            },
            timeout=_FRED_TIMEOUT,
        )
        if hasattr(resp, "json"):
            data = resp.json()
        else:
            import json
            data = json.loads(resp)

        observations = data.get("observations", [])
        values = []
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue

        if values:
            _oas_cache["key"] = fred_api_key
            _oas_cache["data"] = values
            _oas_cache["time"] = now
            logger.debug("HY OAS fetched: %d observations, current=%.2f", len(values), values[0])
            return values

    except Exception:
        logger.warning("FRED HY OAS fetch failed", exc_info=True)

    return _oas_cache.get("data")


def _get_fred_key(context: dict | None) -> str:
    """Extract FRED API key from context -> config."""
    if not context:
        return ""
    cfg = context.get("config")
    if not cfg:
        return ""
    if isinstance(cfg, dict):
        return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
    return getattr(cfg, "fred_api_key", "") or getattr(
        getattr(cfg, "golddigger", None), "fred_api_key", ""
    ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""


def _is_safe_haven(ticker: str) -> bool:
    """Return True if ticker is a safe-haven asset (gold/silver)."""
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\forecast.py | Select-Object -First 170" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\forecast.py | Select-Object -Skip 520 -First 70" in Q:\finance-analyzer
 succeeded in 829ms:
    return atr / current


def _is_crypto_ticker(ticker: str) -> bool:
    """Check if ticker is crypto (BTC-USD, ETH-USD)."""
    try:
        from portfolio.tickers import CRYPTO_SYMBOLS
        return ticker in CRYPTO_SYMBOLS
    except ImportError:
        return ticker in {"BTC-USD", "ETH-USD"}


def _load_forecast_accuracy(cache_ttl=None):
    """Load per-ticker forecast accuracy, cached via _cached().

    Returns dict: {ticker: {accuracy, samples}} or empty dict on error.
    """
    ttl = cache_ttl or _ACCURACY_CACHE_TTL

    def _fetch():
        try:
            from portfolio.forecast_accuracy import get_all_ticker_accuracies
            return get_all_ticker_accuracies(horizon="24h", days=14)
        except Exception as e:
            logger.debug("Failed to load forecast accuracy: %s", e)
            return {}

    return _cached("forecast_ticker_accuracy", ttl, _fetch)


def _load_forecast_subsignal_accuracy(cache_ttl=None, days=30):
    """Load raw sub-signal accuracy for 1h and 24h forecast votes."""
    ttl = cache_ttl or _ACCURACY_CACHE_TTL
    cache_key = f"forecast_subsignal_accuracy_{days}"

    def _fetch():
        try:
            from portfolio.forecast_accuracy import compute_forecast_accuracy

            return {
                "1h": compute_forecast_accuracy(
                    horizon="1h", days=days, use_raw_sub_signals=True
                ),
                "24h": compute_forecast_accuracy(
                    horizon="24h", days=days, use_raw_sub_signals=True
                ),
            }
        except Exception as e:
            logger.debug("Failed to load forecast sub-signal accuracy: %s", e)
            return {}

    return _cached(cache_key, ttl, _fetch)


def _gate_subsignal_votes_by_accuracy(sub_signals, ticker, config_forecast=None):
    """Gate individual forecast sub-signals using raw historical accuracy."""
    cfg = config_forecast or {}
    hold_threshold = cfg.get("subsignal_hold_threshold", cfg.get("hold_threshold", _HOLD_THRESHOLD))
    min_samples = cfg.get("subsignal_min_samples", cfg.get("min_samples", _MIN_SAMPLES))
    lookback_days = cfg.get("subsignal_accuracy_days", 30)
    cache_ttl = cfg.get("subsignal_accuracy_cache_ttl", _ACCURACY_CACHE_TTL)

    gated = dict(sub_signals)
    info = {}
    if not ticker:
        return gated, info

    accuracy_matrix = _load_forecast_subsignal_accuracy(cache_ttl=cache_ttl, days=lookback_days)
    for sub_name, vote in sub_signals.items():
        if vote == "HOLD":
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 836ms:
"""Forecast signal - price direction prediction from time-series models.

Combines Kronos (K-line foundation model), Chronos (Amazon), and Prophet
into a majority-vote composite signal with four sub-signals:
  1. kronos_1h  - Kronos 1-hour prediction
  2. kronos_24h - Kronos 24-hour prediction
  3. chronos_1h - Chronos 1-hour prediction
  4. chronos_24h - Chronos 24-hour prediction

Reuses candle loading from portfolio.forecast_signal. Confidence capped at 0.7.
Registered as enhanced signal #28 with weight=0 (shadow mode) initially.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from portfolio.file_utils import atomic_append_jsonl
from portfolio.gpu_gate import gpu_gate
from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.forecast")

# Cache TTL - forecasts don't change fast
_FORECAST_TTL = 300  # 5 minutes

# Confidence cap (same as news_event, econ_calendar)
_MAX_CONFIDENCE = 0.7

# Default Chronos timeout (seconds) - reduced from 120 to avoid long hangs
_CHRONOS_TIMEOUT = 60

# Default Kronos subprocess timeout (seconds) - reduced from 90; fails fast
_KRONOS_TIMEOUT = 30

# Forecast models master switch. Set to True to disable all model calls (early-return HOLD).
# Circuit breakers remain as secondary protection - auto-trip on failure, 5min TTL.
_FORECAST_MODELS_DISABLED = False

# Kronos inference - UN-RETIRED 2026-04-21 afternoon. The morning's retire
# was premature. The real problem was not Kronos per se but the shadow-mode
# implementation: when _KRONOS_SHADOW=True, forecast.py forced Kronos's
# composite-vote sub-signal to HOLD (see lines ~811/820) while still counting
# those HOLD votes inside _health_weighted_vote's majority tally - so when
# Chronos voted BUY/BUY/BUY and Kronos (silently HOLD) also "voted" H/H/H,
# the result was a 3-3 tie that broke to HOLD and polluted Chronos's verdict.
#
# The correct structure (this session):
#   - Kronos subprocess runs normally, real prediction captured in
#     raw_sub_signals["kronos_*"] for accuracy backfill.
#   - In shadow mode the Kronos sub-signal is EXCLUDED FROM THE COMPOSITE
#     VOTE POOL entirely (see _health_weighted_vote below) - it does not
#     contribute HOLD weights, does not dilute Chronos.
#   - Subprocess reliability (59 % success  90 % target) remains a separate
#     work stream. The shadow-logging path tolerates failure gracefully;
#     un-retirement does not require the reliability fix to land first.
#
# Config re-read here so operators can flip kronos_enabled=true|"shadow"|false
# without a code edit, matching the pre-retire behavior.
_KRONOS_ENABLED = False
_KRONOS_SHADOW = False


def _init_kronos_enabled():
    """Read kronos_enabled from config.json at import time.

    Values:
      - `true`   active voter (raw prediction contributes to composite vote)
      - `"shadow"`  inference runs, raw prediction is logged, but the
        sub-signal is EXCLUDED from the composite vote pool (see
        _health_weighted_vote). Zero composite pollution.
      - `false` (default)  subprocess skipped entirely.
    """
    global _KRONOS_ENABLED, _KRONOS_SHADOW
    try:
        from portfolio.file_utils import load_json as _load_json
        _cfg = _load_json(
            str(Path(__file__).resolve().parent.parent.parent / "config.json"),
            {},
        )
        val = _cfg.get("forecast", {}).get("kronos_enabled", False)
        if val == "shadow":
            _KRONOS_ENABLED = True
            _KRONOS_SHADOW = True
        else:
            _KRONOS_ENABLED = bool(val)
            _KRONOS_SHADOW = False
    except Exception as e:
        logger.debug("Kronos init from config: %s", e)


_init_kronos_enabled()

if platform.system() == "Windows":
    _KRONOS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
    _KRONOS_SCRIPT = r"Q:\models\kronos_infer.py"
else:
    _KRONOS_PYTHON = "/home/deck/models/.venv/bin/python"
    _KRONOS_SCRIPT = "/home/deck/models/kronos_infer.py"

# Prediction log
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PREDICTIONS_FILE = _DATA_DIR / "forecast_predictions.jsonl"
_HEALTH_FILE = _DATA_DIR / "forecast_health.jsonl"

# Circuit breaker - after first failure, skip remaining tickers in this loop cycle.
# Prevents 27 x 6s GPU timeouts when CUDA is broken.
_CIRCUIT_BREAKER_TTL = 30  # 30 seconds before retry
_kronos_tripped_until = 0.0  # monotonic timestamp when breaker resets
_chronos_tripped_until = 0.0

# BUG-102: Lock protects circuit breaker state and dedup cache from ThreadPoolExecutor races.
# The read-check-write pattern in _log_health() is not atomic without a lock.
_forecast_lock = threading.Lock()

# Prediction dedup - track last logged timestamp per ticker to avoid
# logging cached replays. Key: ticker, value: ISO-8601 timestamp.
_PREDICTION_DEDUP_TTL = 60  # seconds - don't re-log within this window
_PREDICTION_DEDUP_EVICT_AGE = 600  # BUG-106: evict entries older than 10 minutes
_last_prediction_ts: dict[str, float] = {}  # ticker -> monotonic timestamp


def _extract_json_from_stdout(stdout: str | None) -> dict | None:
    """Extract JSON from potentially contaminated subprocess stdout.

    HuggingFace's from_pretrained() prints to stdout during model loading,
    which contaminates the subprocess output before the JSON result.
    This function handles that by finding the first '{' and parsing from there.

    Returns parsed dict on success, None on failure.
    """
    if not stdout:
        return None

    text = stdout.strip()
    if not text:
        return None

    # Fast path: stdout starts with '{' - clean JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Slow path: find first '{' and try parsing from there
    brace_idx = text.find("{")
    if brace_idx > 0:
        try:
            parsed = json.loads(text[brace_idx:])
            logger.debug("JSON extracted via brace-offset fallback (offset=%d, len=%d)", brace_idx, len(text))
            return parsed
        except json.JSONDecodeError:
            pass

    # Last resort: scan lines in reverse for a JSON line
    for line in reversed(text.splitlines()):
        line = line.strip()
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\vix_term_structure.py | Select-Object -First 220" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\cross_asset_tsmom.py | Select-Object -First 240" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\treasury_risk_rotation.py | Select-Object -First 220" in Q:\finance-analyzer
 succeeded in 865ms:
"""Cross-asset time-series momentum signal.

Uses cross-asset return momentum to predict target asset direction.
Academic basis: Pitkajarvi, Suominen, Vaittinen (2020), JFE - cross-asset
filtered TSMOM yields 45% higher Sharpe than standard TSMOM.

4 sub-indicators via majority vote:
    1. Own TSMOM (252d)       - target asset's 12-month momentum
    2. Cross-pair momentum    - paired asset's 3-month momentum
    3. Bond momentum (TLT)   - 3-month treasury bond returns
    4. Equity momentum (SPY) - 3-month equity market returns

Cross-asset pairs:
    XAU-USD  -> TLT   (gold follows bonds in risk-off)
    XAG-USD  -> GC=F  (silver follows gold futures, 24h coverage)
    BTC-USD  -> SPY   (crypto correlates with risk-on)
    ETH-USD  -> BTC   (ETH follows BTC)
    MSTR     -> BTC   (MSTR is leveraged BTC)

Data: yfinance for TLT/SPY/GC=F/BTC-USD (free, no API key). Cached 1 hour.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.cross_asset_tsmom")

MIN_ROWS = 60
_CACHE_TTL = 3600

_TSMOM_LOOKBACK = 252
_CROSS_PAIR_LOOKBACK = 63
_BOND_LOOKBACK = 63
_EQUITY_LOOKBACK = 63

_CROSS_PAIRS = {
    "XAU-USD": "TLT",
    "XAG-USD": "GC=F",
    "BTC-USD": "SPY",
    "ETH-USD": "BTC-USD",
    "MSTR": "BTC-USD",
}

_YF_TICKERS = ["TLT", "SPY", "GC=F", "BTC-USD"]


def _fetch_yf_returns() -> dict[str, dict] | None:
    """Fetch 12-month daily closes for TLT, SPY, GLD, BTC-USD via yfinance.

    Returns dict mapping ticker -> {"ret_63d": float, "ret_252d": float}
    or None on failure.  Cached for 1 hour.
    """
    def _do_fetch():
        try:
            import yfinance as yf

            tickers = list(_YF_TICKERS)
            data = yf.download(tickers, period="13mo", progress=False, threads=True)
            if data is None or data.empty:
                return None

            close_col = "Close"
            if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
                close = data[close_col]
            else:
                close = data[[close_col]]
                close.columns = tickers[:1]

            result = {}
            for t in tickers:
                col = t
                if col not in close.columns:
                    continue
                series = close[col].dropna()
                if len(series) < _CROSS_PAIR_LOOKBACK + 1:
                    continue

                cur = float(series.iloc[-1])
                idx_63 = max(0, len(series) - _CROSS_PAIR_LOOKBACK - 1)
                idx_252 = max(0, len(series) - _TSMOM_LOOKBACK - 1)
                prev_63 = float(series.iloc[idx_63])
                prev_252 = float(series.iloc[idx_252])

                ret_63d = (cur / prev_63 - 1) if prev_63 > 0 else 0.0
                ret_252d = (cur / prev_252 - 1) if prev_252 > 0 else 0.0

                result[t] = {"ret_63d": ret_63d, "ret_252d": ret_252d}

            missing = set(_YF_TICKERS) - set(result)
            if missing:
                logger.warning("cross_asset_tsmom: missing tickers %s", missing)
            return result if result else None
        except Exception as e:
            logger.warning("cross_asset_tsmom yfinance fetch failed: %s", e)
            return None

    return _cached("cross_asset_tsmom_yf", _CACHE_TTL, _do_fetch)


def _compute_own_tsmom(close: pd.Series) -> str:
    """Sub-indicator 1: target asset's own 252d momentum."""
    n = len(close)
    lookback = min(_TSMOM_LOOKBACK, n - 1)
    if lookback < 20:
        return "HOLD"

    try:
        cur = float(close.iloc[-1])
        prev = float(close.iloc[-lookback - 1])
    except (TypeError, ValueError):
        return "HOLD"
    if prev <= 0 or not np.isfinite(cur) or not np.isfinite(prev):
        return "HOLD"

    ret = cur / prev - 1
    if ret > 0.005:
        return "BUY"
    if ret < -0.005:
        return "SELL"
    return "HOLD"


def _compute_cross_pair(ticker: str, yf_data: dict) -> str:
    """Sub-indicator 2: cross-asset pair's 63d momentum."""
    pair = _CROSS_PAIRS.get(ticker)
    if not pair or not yf_data:
        return "HOLD"

    pair_data = yf_data.get(pair)
    if not pair_data:
        return "HOLD"

    ret = pair_data["ret_63d"]
    if ret > 0.005:
        return "BUY"
    if ret < -0.005:
        return "SELL"
    return "HOLD"


def _compute_bond_momentum(yf_data: dict) -> str:
    """Sub-indicator 3: TLT 63d return as macro risk appetite gauge."""
    if not yf_data or "TLT" not in yf_data:
        return "HOLD"

    ret = yf_data["TLT"]["ret_63d"]
    if ret > 0.005:
        return "BUY"
    if ret < -0.005:
        return "SELL"
    return "HOLD"


def _compute_equity_momentum(yf_data: dict) -> str:
    """Sub-indicator 4: SPY 63d return as risk-on/risk-off gauge."""
    if not yf_data or "SPY" not in yf_data:
        return "HOLD"

    ret = yf_data["SPY"]["ret_63d"]
    if ret > 0.005:
        return "BUY"
    if ret < -0.005:
        return "SELL"
    return "HOLD"


def compute_cross_asset_tsmom_signal(
    df: pd.DataFrame, context: dict | None = None
) -> dict[str, Any]:
    """Compute cross-asset time-series momentum signal."""
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"]
    ticker = (context or {}).get("ticker", "")

    yf_data = _fetch_yf_returns()

    own_tsmom = _compute_own_tsmom(close)
    cross_pair = _compute_cross_pair(ticker, yf_data)
    bond_mom = _compute_bond_momentum(yf_data)
    equity_mom = _compute_equity_momentum(yf_data)

    votes = [own_tsmom, cross_pair, bond_mom, equity_mom]
    action, confidence = majority_vote(votes, count_hold=False)

    n = len(close)
    cur = float(close.iloc[-1])
    idx_252 = max(0, n - _TSMOM_LOOKBACK - 1)
    idx_63 = max(0, n - _CROSS_PAIR_LOOKBACK - 1)
    base_252 = float(close.iloc[idx_252])
    base_63 = float(close.iloc[idx_63])
    own_ret_252 = (cur / base_252 - 1) if base_252 > 0 else 0.0
    own_ret_63 = (cur / base_63 - 1) if base_63 > 0 else 0.0

    def _yf_ret(t):
        return yf_data[t]["ret_63d"] if yf_data and t in yf_data else None

    pair_ticker = _CROSS_PAIRS.get(ticker, "none")
    pair_ret = _yf_ret(pair_ticker) if pair_ticker != "none" else None

    return {
        "action": action,
        "confidence": min(confidence, 0.7),
        "sub_signals": {
            "own_tsmom_252d": own_tsmom,
            "cross_pair_63d": cross_pair,
            "bond_momentum": bond_mom,
            "equity_momentum": equity_mom,
        },
        "indicators": {
            "own_ret_252d": safe_float(own_ret_252),
            "own_ret_63d": safe_float(own_ret_63),
            "tlt_ret_63d": safe_float(_yf_ret("TLT")),
            "spy_ret_63d": safe_float(_yf_ret("SPY")),
            "gld_ret_63d": safe_float(_yf_ret("GLD")),
            "btc_ret_63d": safe_float(_yf_ret("BTC-USD")),
            "cross_pair_ticker": pair_ticker,
            "cross_pair_ret_63d": safe_float(pair_ret),
        },
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 879ms:
"""VIX term structure signal - contango/backwardation regime detection.

Uses VIX/VIX3M ratio to detect risk regime shifts. Backwardation
(VIX > VIX3M) occurs <20% of the time and signals market stress.
Deep contango (ratio < 0.85) signals complacency - contrarian BUY
on recovery.

Backtested strongest on metals (SLV 55%/64%/73% at 1d/3d/5d,
GLD 55%/61% at 1d/3d). Weaker on BTC/equities (~40-50%). Applied
to all assets; per-ticker accuracy gating auto-disables for assets
where it underperforms.

Sub-indicators:
    1. Backwardation flag     - ratio >= 1.0 = stress
    2. Contango depth         - how far below 1.0 (deep = complacent)
    3. VIX ratio z-score      - any deviation from 20d mean
    4. Ratio slope (5d)       - rate of change signals transitions

Data: yfinance ^VIX and ^VIX3M (free, no API key).
"""
from __future__ import annotations

import logging

import numpy as np

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.vix_term_structure")

_MAX_CONFIDENCE = 0.7

_BACKWARDATION_THRESHOLD = 1.0
_STRONG_BACKWARDATION = 1.05
_DEEP_CONTANGO = 0.85
_Z_THRESHOLD = 0.0  # any deviation votes; backtested: z=0.0 >> z=1.0
_SLOPE_THRESHOLD_PCT = 2.0
_LOOKBACK = 20
_CACHE_TTL = 900


def _fetch_vix_data() -> dict | None:
    """Fetch VIX and VIX3M closing prices via yfinance."""
    try:
        import yfinance as yf

        vix = yf.download("^VIX", period="2mo", progress=False)
        vix3m = yf.download("^VIX3M", period="2mo", progress=False)

        if vix is None or vix3m is None or len(vix) < _LOOKBACK or len(vix3m) < _LOOKBACK:
            return None

        close_col = "Close"
        if hasattr(vix.columns, "levels") and len(vix.columns.levels) > 1:
            vix = vix.droplevel(level=1, axis=1)
        if hasattr(vix3m.columns, "levels") and len(vix3m.columns.levels) > 1:
            vix3m = vix3m.droplevel(level=1, axis=1)

        vix_close = vix[close_col].dropna()
        vix3m_close = vix3m[close_col].dropna()

        if len(vix_close) < _LOOKBACK or len(vix3m_close) < _LOOKBACK:
            return None

        idx = vix_close.index.intersection(vix3m_close.index)
        if len(idx) < _LOOKBACK:
            return None

        vix_vals = vix_close.loc[idx].values.astype(float)
        vix3m_vals = vix3m_close.loc[idx].values.astype(float)

        mask = (vix3m_vals > 0) & np.isfinite(vix_vals) & np.isfinite(vix3m_vals)
        vix_vals = vix_vals[mask]
        vix3m_vals = vix3m_vals[mask]

        if len(vix_vals) < _LOOKBACK:
            return None

        ratio = vix_vals / vix3m_vals

        return {
            "vix_current": float(vix_vals[-1]),
            "vix3m_current": float(vix3m_vals[-1]),
            "ratio_current": float(ratio[-1]),
            "ratio_series": ratio.tolist(),
        }
    except Exception as e:
        logger.debug("VIX term structure fetch failed: %s", e)
        return None


def _backwardation_flag(ratio: float) -> str:
    if ratio >= _BACKWARDATION_THRESHOLD:
        return "SELL"
    if ratio < _DEEP_CONTANGO:
        return "BUY"
    return "HOLD"


def _contango_depth(ratio: float) -> str:
    depth = 1.0 - ratio
    if depth > 0.10:
        return "BUY"
    if depth < 0.0:
        return "SELL"
    return "HOLD"


def _ratio_zscore(ratio_series: list[float]) -> tuple[float, str]:
    arr = np.array(ratio_series[-_LOOKBACK:])
    if len(arr) < _LOOKBACK:
        return 0.0, "HOLD"
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-8:
        return 0.0, "HOLD"
    z = (arr[-1] - mean) / std
    if z > _Z_THRESHOLD:
        return float(z), "SELL"
    if z < -_Z_THRESHOLD:
        return float(z), "BUY"
    return float(z), "HOLD"


def _ratio_slope_5d(ratio_series: list[float]) -> tuple[float, str]:
    if len(ratio_series) < 6:
        return 0.0, "HOLD"
    current = ratio_series[-1]
    prev = ratio_series[-6]
    if prev == 0:
        return 0.0, "HOLD"
    slope_pct = (current - prev) / prev * 100
    if slope_pct > _SLOPE_THRESHOLD_PCT:
        return float(slope_pct), "SELL"
    if slope_pct < -_SLOPE_THRESHOLD_PCT:
        return float(slope_pct), "BUY"
    return float(slope_pct), "HOLD"


def compute_vix_term_structure_signal(df=None, context=None, **kwargs) -> dict:
    """Compute VIX term structure signal for any Tier-1 instrument.

    Args:
        df: OHLCV DataFrame (unused - VIX data fetched separately).
        context: Optional dict with ``ticker`` key.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    data = _cached("vix_term_structure", _CACHE_TTL, _fetch_vix_data)
    if data is None:
        return empty

    ratio = data["ratio_current"]
    ratio_series = data["ratio_series"]

    sub1 = _backwardation_flag(ratio)
    sub2 = _contango_depth(ratio)
    z_val, sub3 = _ratio_zscore(ratio_series)
    slope_val, sub4 = _ratio_slope_5d(ratio_series)

    votes = [sub1, sub2, sub3, sub4]
    action, confidence = majority_vote(votes, count_hold=False)

    confidence = min(confidence, _MAX_CONFIDENCE)

    if ratio >= _STRONG_BACKWARDATION:
        confidence = min(max(confidence, 0.6), _MAX_CONFIDENCE)

    indicators = {
        "vix": safe_float(data["vix_current"]),
        "vix3m": safe_float(data["vix3m_current"]),
        "ratio": safe_float(ratio),
        "z_score": safe_float(z_val),
        "slope_5d_pct": safe_float(slope_val),
        "in_backwardation": ratio >= _BACKWARDATION_THRESHOLD,
    }

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "backwardation_flag": sub1,
            "contango_depth": sub2,
            "ratio_zscore": sub3,
            "ratio_slope_5d": sub4,
        },
        "indicators": indicators,
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 873ms:
"""Treasury yield curve risk rotation signal.

Uses relative performance of IEF (7-10Y Treasury) vs TLT (20Y+ Treasury)
as a cross-asset risk regime detector.  Steepening curve (TLT outperforms)
signals risk-on; flattening/inverting (IEF outperforms) signals risk-off.

4 sub-indicators via majority vote:
    1. Curve Slope Direction   - 65d relative return sign
    2. Slope Momentum          - 21d rate of change of the spread
    3. Slope Z-Score           - 252d z-score of current spread
    4. Regime Persistence      - days since last regime flip

Direction depends on asset class:
- Risk-on assets (BTC, ETH, MSTR): steepening = BUY, flattening = SELL
- Safe havens (XAU, XAG):          steepening = SELL, flattening = BUY

Data: yfinance IEF + TLT (free, no API key). Cached 1 hour.
Source: Gayed (2014). An Intermarket Approach to Tactical Risk Rotation.
        SSRN 2431022.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.treasury_risk_rotation")

MIN_ROWS = 30
_CACHE_TTL = 3600

_SLOPE_LOOKBACK = 65
_MOM_LOOKBACK = 21
_ZSCORE_LOOKBACK = 252
_SPREAD_THRESHOLD = 0.01
_ZSCORE_THRESHOLD = 1.0
_MOM_THRESHOLD = 0.005
_PERSISTENCE_DAYS = 5

_SAFE_HAVENS = frozenset({"XAU-USD", "XAG-USD"})


def _fetch_treasury_data() -> dict | None:
    """Fetch IEF and TLT daily closes via yfinance.  Cached 1 hour."""

    def _do_fetch():
        try:
            import yfinance as yf

            data = yf.download(
                ["IEF", "TLT"], period="14mo", progress=False, threads=True,
            )
            if data is None or data.empty:
                return None

            close = data["Close"]
            if "IEF" not in close.columns or "TLT" not in close.columns:
                logger.warning("treasury_risk_rotation: missing IEF or TLT column")
                return None

            ief = close["IEF"].dropna()
            tlt = close["TLT"].dropna()

            if len(ief) < _SLOPE_LOOKBACK + 1 or len(tlt) < _SLOPE_LOOKBACK + 1:
                logger.warning("treasury_risk_rotation: insufficient data rows")
                return None

            return {"ief": ief, "tlt": tlt}
        except Exception as e:
            logger.warning("treasury_risk_rotation yfinance fetch failed: %s", e)
            return None

    return _cached("treasury_risk_rotation_yf", _CACHE_TTL, _do_fetch)


def _compute_spread_series(ief: pd.Series, tlt: pd.Series) -> pd.Series:
    """Compute rolling spread = TLT_return - IEF_return for each lookback window."""
    ief_ret = ief.pct_change(_SLOPE_LOOKBACK)
    tlt_ret = tlt.pct_change(_SLOPE_LOOKBACK)
    return tlt_ret - ief_ret


def _sub_slope_direction(spread_current: float) -> str:
    if spread_current > _SPREAD_THRESHOLD:
        return "BUY"
    if spread_current < -_SPREAD_THRESHOLD:
        return "SELL"
    return "HOLD"


def _sub_slope_momentum(spread_series: pd.Series) -> str:
    if len(spread_series) < _MOM_LOOKBACK + 1:
        return "HOLD"
    mom = float(spread_series.iloc[-1]) - float(spread_series.iloc[-_MOM_LOOKBACK - 1])
    if mom > _MOM_THRESHOLD:
        return "BUY"
    if mom < -_MOM_THRESHOLD:
        return "SELL"
    return "HOLD"


def _sub_slope_zscore(spread_series: pd.Series) -> tuple[float, str]:
    n = min(_ZSCORE_LOOKBACK, len(spread_series))
    if n < 30:
        return 0.0, "HOLD"
    window = spread_series.iloc[-n:]
    mean = float(window.mean())
    std = float(window.std())
    if std < 1e-9:
        return 0.0, "HOLD"
    z = (float(spread_series.iloc[-1]) - mean) / std
    if z > _ZSCORE_THRESHOLD:
        return z, "BUY"
    if z < -_ZSCORE_THRESHOLD:
        return z, "SELL"
    return z, "HOLD"


def _sub_regime_persistence(spread_series: pd.Series) -> str:
    """Count consecutive days the spread has stayed on the same side."""
    if len(spread_series) < 2:
        return "HOLD"
    last = float(spread_series.iloc[-1])
    if last == 0.0:
        return "HOLD"
    current_sign = 1 if last > 0 else -1
    days = 0
    for i in range(len(spread_series) - 1, -1, -1):
        val = float(spread_series.iloc[i])
        if val == 0.0:
            break
        s = 1 if val > 0 else -1
        if s != current_sign:
            break
        days += 1
    if days >= _PERSISTENCE_DAYS:
        return "BUY" if current_sign > 0 else "SELL"
    return "HOLD"


def _invert(action: str) -> str:
    if action == "BUY":
        return "SELL"
    if action == "SELL":
        return "BUY"
    return "HOLD"


def compute_treasury_risk_rotation_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict:
    """Compute treasury risk rotation signal."""
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    treasury = _fetch_treasury_data()
    if treasury is None:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    ief = treasury["ief"]
    tlt = treasury["tlt"]
    spread_series = _compute_spread_series(ief, tlt)
    spread_series = spread_series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(spread_series) < 30:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    spread_current = float(spread_series.iloc[-1])

    v_direction = _sub_slope_direction(spread_current)
    v_momentum = _sub_slope_momentum(spread_series)
    zscore_val, v_zscore = _sub_slope_zscore(spread_series)
    v_persistence = _sub_regime_persistence(spread_series)

    votes = [v_direction, v_momentum, v_zscore, v_persistence]
    action, confidence = majority_vote(votes, count_hold=False)

    ticker = (context or {}).get("ticker", "")
    is_safe_haven = ticker in _SAFE_HAVENS
    if is_safe_haven:
        action = _invert(action)

    ief_ret_65 = float(ief.iloc[-1] / ief.iloc[-min(_SLOPE_LOOKBACK, len(ief) - 1) - 1] - 1)
    tlt_ret_65 = float(tlt.iloc[-1] / tlt.iloc[-min(_SLOPE_LOOKBACK, len(tlt) - 1) - 1] - 1)

    return {
        "action": action,
        "confidence": min(confidence, 0.7),
        "sub_signals": {
            "slope_direction": v_direction,
            "slope_momentum": v_momentum,
            "slope_zscore": v_zscore,
            "regime_persistence": v_persistence,
        },
        "indicators": {
            "spread_65d": safe_float(spread_current),
            "zscore": safe_float(zscore_val),
            "ief_ret_65d": safe_float(ief_ret_65),
            "tlt_ret_65d": safe_float(tlt_ret_65),
            "is_safe_haven": is_safe_haven,
        },
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
A second high-risk cluster is the “fixed-horizon” detectors. Anything hardcoding `1h/24h`, `1h DXY`, or daily macro returns without reading horizon will leak the same vote into every engine horizon. I’m confirming that in `forecast`, `credit_spread`, `cot_positioning`, and `metals_cross_asset`.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\forecast.py | Select-Object -Skip 720 -First 250" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\cot_positioning.py | Select-Object -First 450" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\metals_cross_asset.py | Select-Object -First 520" in Q:\finance-analyzer
 succeeded in 753ms:
        return base_action, min(scaled_conf, _MAX_CONFIDENCE), gating_info


def _direction_to_action(direction: str) -> str:
    """Convert direction string to action."""
    if direction in ("up", "BUY"):
        return "BUY"
    if direction in ("down", "SELL"):
        return "SELL"
    return "HOLD"


def compute_forecast_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the composite forecast signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (used as fallback if candle fetch fails).
    context : dict, optional
        Dict with keys: ticker, config, macro.

    Returns
    -------
    dict
        action, confidence, sub_signals, indicators
    """
    result = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "kronos_1h": "HOLD",
            "kronos_24h": "HOLD",
            "chronos_1h": "HOLD",
            "chronos_24h": "HOLD",
        },
        "indicators": {},
    }

    # Bulletproof early return - skip ALL work when models are disabled
    if _FORECAST_MODELS_DISABLED:
        result["indicators"]["models_disabled"] = True
        return result

    ticker = (context or {}).get("ticker", "")
    if not ticker:
        return result

    # Apply Chronos model config if specified
    chronos_model = (context or {}).get("config", {}).get("forecast", {}).get("chronos_model")
    if chronos_model:
        try:
            from portfolio.forecast_signal import set_chronos_model
            set_chronos_model(chronos_model)
        except Exception as e:
            logger.debug("Chronos model config override failed: %s", e)

    config_forecast = (context or {}).get("config", {}).get("forecast", {})

    # Load candles (1h for Chronos, optionally 5m for Kronos)
    cache_key = f"forecast_candles_{ticker}"
    candles = _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv, ticker)

    # Load 5m candles for Kronos if configured (more granular context)
    kronos_interval = config_forecast.get("kronos_interval", "1h")
    if kronos_interval != "1h" and _KRONOS_ENABLED:
        kronos_periods = config_forecast.get("kronos_periods", 500)
        kronos_cache_key = f"forecast_candles_{ticker}_{kronos_interval}"
        kronos_candles = _cached(kronos_cache_key, _FORECAST_TTL,
                                  _load_candles_ohlcv, ticker, kronos_periods,
                                  kronos_interval)
    else:
        kronos_candles = None

    if not candles or len(candles) < 50:
        # Fallback to df close prices if available
        if df is not None and len(df) >= 50 and "close" in df.columns:
            close_prices = df["close"].values.tolist()
        else:
            result["indicators"]["error"] = "insufficient_candle_data"
            return result
    else:
        close_prices = [c["close"] for c in candles]

    # If Kronos-specific candle fetch failed but df has full OHLCV data, build candle dicts
    # from the DataFrame so Kronos still gets richer data than just close prices.
    if kronos_candles is None and df is not None and len(df) >= 50:
        ohlcv_cols = {"open", "high", "low", "close", "volume"}
        if ohlcv_cols.issubset(df.columns):
            try:
                kronos_candles = [
                    {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                    for _, row in df.iterrows()
                ]
                logger.debug(
                    "Kronos candle fallback from df for %s (%d candles)", ticker, len(kronos_candles)
                )
                result["indicators"]["kronos_candles_source"] = "df_fallback"
            except Exception as e:
                logger.debug("Kronos df candle fallback failed for %s: %s", ticker, e)
                kronos_candles = None

    current_price = close_prices[-1]
    result["indicators"]["current_price"] = current_price
    result["indicators"]["candle_count"] = len(close_prices)
    result["indicators"]["kronos_circuit_open"] = _kronos_circuit_open()
    result["indicators"]["chronos_circuit_open"] = _chronos_circuit_open()

    # 2026-05-03 ordering: Chronos BEFORE Kronos. Chronos is in-process and
    # fast (~50ms warm, ~1.7s cold model load). Kronos is a subprocess that
    # holds the GPU file-lock for the duration of model load + inference,
    # which on cold-start was ~210s - long enough that 4 concurrent ticker
    # threads all timed out their Chronos gate (120s) waiting behind one
    # Kronos. Running Chronos first lets all 4 tickers pipeline through the
    # GPU in seconds; Kronos (shadow-only by current config) then runs and
    # whichever threads can't grab the gate within 90s skip silently - fine
    # because the live consensus uses Chronos, not Kronos.
    #
    # Run Chronos (skip entirely if circuit breaker is open)
    t0 = time.time()
    chronos_key = f"chronos_forecast_{ticker}"
    chronos = _cached(chronos_key, _FORECAST_TTL, _run_chronos, close_prices, (1, 24), ticker)
    chronos_ms = round((time.time() - t0) * 1000)
    result["indicators"]["chronos_time_ms"] = chronos_ms

    if chronos:
        if "1h" in chronos:
            result["sub_signals"]["chronos_1h"] = chronos["1h"].get("action", "HOLD")
            result["indicators"]["chronos_1h_pct"] = chronos["1h"].get("pct_move", 0)
            result["indicators"]["chronos_1h_conf"] = chronos["1h"].get("confidence", 0)

        if "24h" in chronos:
            result["sub_signals"]["chronos_24h"] = chronos["24h"].get("action", "HOLD")
            result["indicators"]["chronos_24h_pct"] = chronos["24h"].get("pct_move", 0)
            result["indicators"]["chronos_24h_conf"] = chronos["24h"].get("confidence", 0)

    # Run Kronos - use 5m candles if available, otherwise 1h
    t0 = time.time()
    kronos_key = f"kronos_forecast_{ticker}"
    kronos_input = kronos_candles if kronos_candles and len(kronos_candles) >= 50 else (candles or [])
    kronos = _cached(kronos_key, _FORECAST_TTL, _run_kronos, kronos_input, (1, 24), ticker)
    if kronos_candles and len(kronos_candles) >= 50:
        result["indicators"]["kronos_interval"] = kronos_interval
    kronos_ms = round((time.time() - t0) * 1000)
    result["indicators"]["kronos_time_ms"] = kronos_ms

    if kronos and kronos.get("results"):
        kr = kronos["results"]
        result["indicators"]["kronos_method"] = kronos.get("method", "unknown")
        result["indicators"]["kronos_shadow"] = _KRONOS_SHADOW

        if "1h" in kr:
            k1h_action = _direction_to_action(kr["1h"].get("direction", "neutral"))
            # 2026-04-21 shadow fix: always record the REAL prediction in
            # sub_signals. The vote-pool filter in _health_weighted_vote
            # excludes the sub-signal when _KRONOS_SHADOW is True, so the
            # composite is never polluted. This change means the
            # forecast_predictions.jsonl log captures Kronos's true verdict
            # (not a forced HOLD), which unblocks meaningful accuracy
            # backfill. Pre-fix: `"HOLD" if _KRONOS_SHADOW else k1h_action`
            # - left zero statistical signal in 3668 logged predictions.
            result["sub_signals"]["kronos_1h"] = k1h_action
            result["indicators"]["kronos_1h_raw"] = k1h_action
            result["indicators"]["kronos_1h_pct"] = kr["1h"].get("pct_move", 0)
            result["indicators"]["kronos_1h_conf"] = kr["1h"].get("confidence", 0)
            result["indicators"]["kronos_1h_range_pct"] = kr["1h"].get("predicted_range_pct", 0)
            result["indicators"]["kronos_1h_range_skew"] = kr["1h"].get("range_skew", 0)

        if "24h" in kr:
            k24h_action = _direction_to_action(kr["24h"].get("direction", "neutral"))
            result["sub_signals"]["kronos_24h"] = k24h_action
            result["indicators"]["kronos_24h_raw"] = k24h_action
            result["indicators"]["kronos_24h_pct"] = kr["24h"].get("pct_move", 0)
            result["indicators"]["kronos_24h_conf"] = kr["24h"].get("confidence", 0)
            result["indicators"]["kronos_24h_range_pct"] = kr["24h"].get("predicted_range_pct", 0)
            result["indicators"]["kronos_24h_range_skew"] = kr["24h"].get("range_skew", 0)
            result["indicators"]["kronos_24h_predicted_high"] = kr["24h"].get("predicted_high", 0)
            result["indicators"]["kronos_24h_predicted_low"] = kr["24h"].get("predicted_low", 0)

    raw_sub_signals = dict(result["sub_signals"])
    gated_sub_signals, subsignal_gating = _gate_subsignal_votes_by_accuracy(
        raw_sub_signals, ticker, config_forecast=config_forecast
    )
    result["sub_signals"] = gated_sub_signals
    result["indicators"]["forecast_subsignal_gating"] = subsignal_gating

    # Accuracy-weighted vote - per-ticker accuracy gating + health exclusion
    kronos_ok = kronos is not None and bool(kronos.get("results"))
    chronos_ok = chronos is not None
    result["indicators"]["kronos_ok"] = kronos_ok
    result["indicators"]["chronos_ok"] = chronos_ok

    # Compute ATR% for volatility gate
    atr_pct = _compute_atr_pct(close_prices)
    result["indicators"]["forecast_atr_pct"] = round(atr_pct, 4) if atr_pct else None

    regime = (context or {}).get("regime", "")
    result["action"], result["confidence"], gating_info = _accuracy_weighted_vote(
        result["sub_signals"], kronos_ok, chronos_ok,
        ticker=ticker, config_forecast=config_forecast,
        atr_pct=atr_pct, regime=regime,
    )

    # Store gating metadata in indicators
    result["indicators"]["forecast_accuracy"] = gating_info.get("forecast_accuracy")
    result["indicators"]["forecast_samples"] = gating_info.get("forecast_samples", 0)
    result["indicators"]["forecast_gating"] = gating_info.get("forecast_gating", "raw")
    result["indicators"]["forecast_inverted"] = gating_info.get("forecast_inverted", False)

    # Cap confidence (already capped inside _accuracy_weighted_vote, but belt-and-suspenders)
    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)

    # Log prediction for accuracy tracking (with dedup)
    # BUG-102: Lock protects _last_prediction_ts from concurrent ThreadPoolExecutor access
    # BUG-106: Evict stale entries to prevent unbounded dict growth
    try:
        now_mono = time.monotonic()
        with _forecast_lock:
            last_ts = _last_prediction_ts.get(ticker, 0.0)
            should_log = now_mono - last_ts >= _PREDICTION_DEDUP_TTL
        if should_log:
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "ticker": ticker,
                "current_price": current_price,
                "sub_signals": result["sub_signals"],
                "raw_sub_signals": raw_sub_signals,
                "subsignal_gating": subsignal_gating,
                "action": result["action"],
                "confidence": result["confidence"],
                "per_ticker_accuracy": gating_info.get("forecast_accuracy"),
                "gating_action": gating_info.get("forecast_gating", "raw"),
            }
            if kronos and kronos.get("results"):
                entry["kronos"] = kronos["results"]
            if chronos:
                entry["chronos"] = chronos
            atomic_append_jsonl(_PREDICTIONS_FILE, entry)
            with _forecast_lock:
                _last_prediction_ts[ticker] = now_mono
                # BUG-106: Evict stale entries older than 10 minutes
                stale = [k for k, v in _last_prediction_ts.items()
                         if now_mono - v > _PREDICTION_DEDUP_EVICT_AGE]
                for k in stale:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 764ms:
"""Cross-asset signal for metals -- correlated market indicators.

Signal #32.  Combines 8 cross-asset sub-indicators via majority vote:
    1. Copper Momentum: copper up -> industrial demand -> silver bullish
    2. GVZ (Gold VIX): high implied vol signals breakout/reversal
    3. Gold/Silver Ratio: mean-reversion signal (high = silver cheap)
    4. G/S Ratio Velocity: rate of change - falling = silver outperforming
    5. SPY Momentum: risk-on/risk-off gauge
    6. Oil Momentum: inflation expectations proxy
    7. EPU (Economic Policy Uncertainty): high uncertainty -> safe haven BUY
    8. TIPS Real Yield direction: falling real yields -> BUY metals

Applicable to XAU-USD and XAG-USD only.
Gold and silver interpret some signals differently (e.g. G/S ratio).

2026-04-13: Horizon realignment after live measurement showed 29.1% on
XAG 3h (178 BUY / 1 SELL bias over 179 samples). Root cause was using
5-day / 1-day lookbacks against 3-hour outcomes - lagged features with
no intraday resolution. Fix: switch primary data to intraday (60m bars
via `get_all_cross_asset_intraday`) and tighten thresholds proportionally.
Daily data retained as fallback when intraday fetch fails (weekend,
yfinance hiccup, etc.). GVZ stays daily - it's a daily-published index.

2026-04-26: Added EPU + TIPS real yield from FRED API as sub-signals #7-8.
EPU improves gold RMSE by ~18% (Baker/Bloom/Davis 2016). TIPS real yield
direction captures opportunity-cost channel (metals pay no yield - when
real yields fall, holding metals becomes relatively more attractive).
Both are daily-cadence indicators like GVZ.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.metals_cross_asset")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}

# 3h-calibrated thresholds. Rationale:
#   Copper: daily 5d threshold was 1.5% (~0.3%/day). 3h typical range ~0.2%
#     - threshold 0.4% captures a clearly directional 3h move.
#   SPY: daily 1d threshold was 0.8%. Intraday 3h range typically 0.2-0.5%
#     - threshold 0.25% catches risk-on/off shifts at 1-3h scale.
#   Oil: daily 5d threshold was 2.0% (~0.4%/day). 3h typical range 0.3-0.8%
#     - threshold 0.5% captures meaningful oil moves.
#   G/S ratio velocity: daily 5d was 2.0%. 3h ratio moves typically 0.3-0.7%
#     - threshold 0.5% captures silver-vs-gold divergence in the last 3h.
_COPPER_MOVE_INTRADAY_PCT = 0.4
_SPY_MOVE_INTRADAY_PCT = 0.25
_OIL_MOVE_INTRADAY_PCT = 0.5
_GS_VELOCITY_INTRADAY_PCT = 0.5

# Daily thresholds retained for the fallback path (intraday fetch failure).
_COPPER_MOVE_DAILY_PCT = 1.5
_SPY_MOVE_DAILY_PCT = 0.8
_OIL_MOVE_DAILY_PCT = 2.0
_GS_VELOCITY_DAILY_PCT = 2.0

_GVZ_ZSCORE_HIGH = 1.5
_GVZ_ZSCORE_LOW = -1.0
_GS_RATIO_ZSCORE = 1.5

# --- FRED-sourced macro indicators (daily, 4h cache) ---
# EPU: Economic Policy Uncertainty (Baker/Bloom/Davis daily news index).
# High uncertainty  flight to safety  BUY metals.
_EPU_ZSCORE_HIGH = 1.5
_EPU_ZSCORE_LOW = -1.0
_EPU_FRED_SERIES = "USEPUINDXD"

# TIPS real yield (10Y TIPS, FRED series DFII10).
# Direction matters: falling real yields  lower opportunity cost  BUY metals.
# Threshold: 10bp change in 5d-vs-5d moving avg is meaningful.
_TIPS_CHANGE_THRESHOLD = 0.10
_TIPS_FRED_SERIES = "DFII10"

_FRED_TIMEOUT = 15
_FRED_CACHE_TTL = 4 * 3600
_FRED_HISTORY_LIMIT = 300

# Module-level FRED caches (separate dict per series)
_epu_cache: dict = {}
_tips_cache: dict = {}
_fred_cache_lock = threading.Lock()


def _get_fred_key(context: dict | None) -> str:
    """Extract FRED API key from context  config."""
    if not context:
        return ""
    cfg = context.get("config")
    if not cfg:
        return ""
    if isinstance(cfg, dict):
        return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
    return getattr(cfg, "fred_api_key", "") or getattr(
        getattr(cfg, "golddigger", None), "fred_api_key", ""
    ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""


def _fetch_fred_values(
    series_id: str, fred_api_key: str, cache: dict,
) -> list[float] | None:
    """Fetch a FRED series.  Returns list of floats (newest first), cached 4h."""
    now = time.time()
    with _fred_cache_lock:
        if (
            cache.get("key") == fred_api_key
            and cache.get("data")
            and now - cache.get("time", 0) < _FRED_CACHE_TTL
        ):
            return cache["data"]

    if not fred_api_key:
        logger.debug("No FRED API key - cannot fetch %s", series_id)
        return cache.get("data")

    try:
        from portfolio.http_retry import fetch_with_retry
    except ImportError:
        logger.debug("http_retry not available for FRED fetch")
        return cache.get("data")

    try:
        resp = fetch_with_retry(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": _FRED_HISTORY_LIMIT,
            },
            timeout=_FRED_TIMEOUT,
        )
        data = resp.json() if hasattr(resp, "json") else __import__("json").loads(resp)
        observations = data.get("observations", [])
        values = []
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue

        if values:
            with _fred_cache_lock:
                cache["key"] = fred_api_key
                cache["data"] = values
                cache["time"] = now
            logger.debug(
                "FRED %s fetched: %d values, latest=%.2f",
                series_id, len(values), values[0],
            )
            return values
    except Exception:
        logger.warning("FRED %s fetch failed", series_id, exc_info=True)

    return cache.get("data")


def _compute_zscore(values: list[float], lookback: int = 252) -> float:
    """Z-score of most recent value vs lookback history."""
    if len(values) < 20:
        return 0.0
    n = min(lookback, len(values))
    history = values[:n]
    mean = sum(history) / len(history)
    variance = sum((v - mean) ** 2 for v in history) / len(history)
    std = variance ** 0.5
    if std < 1e-10:
        return 0.0
    return (values[0] - mean) / std


def _get_cross_asset_context(ticker: str) -> dict | None:
    """Fetch cross-asset data. Prefer intraday (60m); fall back to daily.

    Returns a dict with per-source payloads AND a ``_using_intraday`` flag
    so the caller can pick the right thresholds. Returns None only if the
    ``metals_cross_assets`` module is not importable (module-level failure).

    Daily G/S ratio is ALWAYS fetched - its z-score is a stable 20-day
    measure that's informative at every horizon, and pre-fetching it here
    ensures the intraday path never makes an extra conditional call mid-
    routing.
    """
    try:
        from portfolio.metals_cross_assets import (
            get_all_cross_asset_data,
            get_all_cross_asset_intraday,
            get_gold_silver_ratio,
            get_gvz,
        )
    except ImportError:
        logger.debug("metals_cross_assets module not available")
        return None

    # Daily anchors - always fetched, cached with their own TTL elsewhere:
    #   GVZ: no intraday source exists (CBOE daily index).
    #   G/S daily: z-score is a stable 20-day measure used on every path.
    gvz = get_gvz()
    gs_daily = get_gold_silver_ratio()

    intraday = get_all_cross_asset_intraday()
    intraday_ok = sum(
        1 for key in ("copper", "gold_silver_ratio", "spy", "oil")
        if intraday.get(key) is not None
    )
    # Require at least 3 of 4 sources to consider intraday healthy;
    # one-off API hiccups shouldn't downgrade the whole signal to stale
    # daily data.
    use_intraday = intraday_ok >= 3

    result: dict = {"_using_intraday": use_intraday}

    if use_intraday:
        # When exactly 3 of 4 intraday sources are healthy, the missing
        # one silently contributes 0  HOLD. Log at WARNING so operators
        # can see a degraded source rather than a quiet vote loss.
        degraded = [
            key for key in ("copper", "gold_silver_ratio", "spy", "oil")
            if intraday.get(key) is None
        ]
        if degraded:
            logger.warning(
                "metals_cross_asset: intraday source(s) unavailable %s - "
                "sub-signals for these will vote HOLD this cycle",
                degraded,
            )

        copper = intraday["copper"]
        gs = intraday["gold_silver_ratio"]
        spy = intraday["spy"]
        oil = intraday["oil"]
        result["copper_change_pct"] = copper["change_3h_pct"] if copper else 0.0
        result["gs_velocity_pct"] = gs["change_3h_pct"] if gs else 0.0
        result["spy_change_pct"] = spy["change_3h_pct"] if spy else 0.0
        result["oil_change_pct"] = oil["change_3h_pct"] if oil else 0.0
    else:
        logger.debug(
            "metals_cross_asset: intraday health=%d/4, falling back to daily",
            intraday_ok,
        )
        daily = get_all_cross_asset_data()
        copper = daily.get("copper")
        # Daily G/S ratio already pre-fetched above - reuse for both the
        # velocity field (5d change) and the z-score.
        spy = daily.get("spy")
        oil = daily.get("oil")
        result["copper_change_pct"] = copper["change_5d_pct"] if copper else 0.0
        result["gs_velocity_pct"] = gs_daily["change_5d_pct"] if gs_daily else 0.0
        result["spy_change_pct"] = spy["change_1d_pct"] if spy else 0.0
        result["oil_change_pct"] = oil["change_5d_pct"] if oil else 0.0

    result["gvz_zscore"] = gvz["zscore"] if gvz else 0.0
    result["gs_ratio_zscore"] = gs_daily["zscore"] if gs_daily else 0.0

    return result


def compute_metals_cross_asset_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute cross-asset composite signal for metals.

    Args:
        df: OHLCV DataFrame (unused - cross-asset data fetched separately).
        context: dict with keys {ticker, config, macro, regime}.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))
    macro = context.get("macro")

    if ticker not in _METALS_TICKERS:
        return empty

    ctx = _get_cross_asset_context(ticker)
    if ctx is None:
        return empty

    using_intraday = ctx.get("_using_intraday", False)

    # Select threshold set based on data cadence
    if using_intraday:
        copper_thr = _COPPER_MOVE_INTRADAY_PCT
        spy_thr = _SPY_MOVE_INTRADAY_PCT
        oil_thr = _OIL_MOVE_INTRADAY_PCT
        gs_vel_thr = _GS_VELOCITY_INTRADAY_PCT
    else:
        copper_thr = _COPPER_MOVE_DAILY_PCT
        spy_thr = _SPY_MOVE_DAILY_PCT
        oil_thr = _OIL_MOVE_DAILY_PCT
        gs_vel_thr = _GS_VELOCITY_DAILY_PCT

    # Oil fallback: if still zero, try macro_data
    if ctx["oil_change_pct"] == 0.0 and macro and isinstance(macro, dict):
        oil_ctx = macro.get("oil", {})
        if isinstance(oil_ctx, dict):
            # Use 1d when available on intraday path (macro dict is daily),
            # else 5d. We take whatever's there rather than zero.
            fallback = oil_ctx.get("change_1d_pct") or oil_ctx.get("change_5d_pct") or 0.0
            ctx["oil_change_pct"] = fallback

    is_silver = ticker == "XAG-USD"
    votes = []
    sub_signals = {}

    # Sub 1: Copper Momentum
    cu = ctx["copper_change_pct"]
    if cu > copper_thr:
        sub_signals["copper"] = "BUY"
    elif cu < -copper_thr:
        sub_signals["copper"] = "SELL"
    else:
        sub_signals["copper"] = "HOLD"
    votes.append(sub_signals["copper"])

    # Sub 2: GVZ (Gold Volatility Index) - still daily
    # High GVZ = fear/uncertainty  safe haven demand (BUY gold, SELL silver)
    # Low GVZ = complacency  no safe haven premium (SELL gold, BUY silver)
    gvz = ctx["gvz_zscore"]
    if gvz > _GVZ_ZSCORE_HIGH:
        sub_signals["gvz"] = "BUY" if not is_silver else "SELL"
    elif gvz < _GVZ_ZSCORE_LOW:
        sub_signals["gvz"] = "SELL" if not is_silver else "BUY"
    else:
        sub_signals["gvz"] = "HOLD"
    votes.append(sub_signals["gvz"])

    # Sub 3: Gold/Silver Ratio - z-score is daily-stable, OK to keep
    gsr = ctx["gs_ratio_zscore"]
    if is_silver:
        # High G/S ratio = silver undervalued relative to gold -> BUY silver
        if gsr > _GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "BUY"
        elif gsr < -_GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "SELL"
        else:
            sub_signals["gs_ratio"] = "HOLD"
    else:
        # For gold: high G/S ratio = gold already outperforming -> HOLD
        # Low G/S ratio = gold underperforming -> mean-reversion BUY
        if gsr > _GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "HOLD"
        elif gsr < -_GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "BUY"
        else:
            sub_signals["gs_ratio"] = "HOLD"
    votes.append(sub_signals["gs_ratio"])

    # Sub 4: G/S Ratio Velocity - now intraday when available
    # Falling G/S ratio = silver outperforming gold = bullish silver
    # Rising G/S ratio = gold outperforming silver = bearish silver
    gs_vel = ctx["gs_velocity_pct"]
    if is_silver:
        if gs_vel < -gs_vel_thr:
            sub_signals["gs_velocity"] = "BUY"   # Silver gaining vs gold
        elif gs_vel > gs_vel_thr:
            sub_signals["gs_velocity"] = "SELL"   # Silver losing vs gold
        else:
            sub_signals["gs_velocity"] = "HOLD"
    else:
        # For gold: rising G/S = gold outperforming -> BUY gold
        if gs_vel > gs_vel_thr:
            sub_signals["gs_velocity"] = "BUY"
        elif gs_vel < -gs_vel_thr:
            sub_signals["gs_velocity"] = "SELL"
        else:
            sub_signals["gs_velocity"] = "HOLD"
    votes.append(sub_signals["gs_velocity"])

    # Sub 5: SPY Momentum (risk-on/risk-off) - now intraday when available
    spy = ctx["spy_change_pct"]
    if spy > spy_thr:
        # Risk-on: silver benefits (industrial), gold neutral
        sub_signals["spy_risk"] = "BUY" if is_silver else "HOLD"
    elif spy < -spy_thr:
        # Risk-off: gold benefits (safe haven), silver hurt
        sub_signals["spy_risk"] = "BUY" if not is_silver else "SELL"
    else:
        sub_signals["spy_risk"] = "HOLD"
    votes.append(sub_signals["spy_risk"])

    # Sub 6: Oil Momentum (inflation expectations) - now intraday when available
    oil = ctx["oil_change_pct"]
    if oil > oil_thr:
        sub_signals["oil"] = "BUY"
    elif oil < -oil_thr:
        sub_signals["oil"] = "SELL"
    else:
        sub_signals["oil"] = "HOLD"
    votes.append(sub_signals["oil"])

    # Sub 7: EPU (Economic Policy Uncertainty) - daily FRED
    # High uncertainty  safe-haven demand  BUY both gold and silver
    # Low uncertainty  risk-on  less safe-haven premium  SELL
    fred_key = _get_fred_key(context)
    epu_values = _fetch_fred_values(_EPU_FRED_SERIES, fred_key, _epu_cache)
    epu_zscore = _compute_zscore(epu_values) if epu_values else 0.0
    if epu_zscore > _EPU_ZSCORE_HIGH:
        sub_signals["epu"] = "BUY"
    elif epu_zscore < _EPU_ZSCORE_LOW:
        sub_signals["epu"] = "SELL"
    else:
        sub_signals["epu"] = "HOLD"
    votes.append(sub_signals["epu"])

    # Sub 8: TIPS Real Yield direction - daily FRED (DFII10)
    # Falling real yields  lower opportunity cost of holding metals  BUY
    # Rising real yields  higher opportunity cost  SELL
    tips_values = _fetch_fred_values(_TIPS_FRED_SERIES, fred_key, _tips_cache)
    if tips_values and len(tips_values) >= 10:
        tips_recent = sum(tips_values[:5]) / 5
        tips_older = sum(tips_values[5:10]) / 5
        tips_change = tips_recent - tips_older
    else:
        tips_change = 0.0

    if tips_change < -_TIPS_CHANGE_THRESHOLD:
        sub_signals["tips_yield"] = "BUY"
    elif tips_change > _TIPS_CHANGE_THRESHOLD:
        sub_signals["tips_yield"] = "SELL"
    else:
        sub_signals["tips_yield"] = "HOLD"
    votes.append(sub_signals["tips_yield"])

    action, confidence = majority_vote(votes)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": {
            "copper_change": round(cu, 3),
            "gvz_zscore": round(gvz, 2),
            "gs_ratio_zscore": round(gsr, 2),
            "gs_velocity": round(gs_vel, 3),
            "spy_change": round(spy, 3),
            "oil_change": round(oil, 3),
            "epu_zscore": round(epu_zscore, 2),
            "tips_change": round(tips_change, 3),
            "using_intraday": using_intraday,
        },
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 779ms:
"""CFTC Commitment of Traders (COT) positioning signal for precious metals.

Signal #33.  Contrarian positioning indicator using CFTC COT reports.
Combines 4 sub-indicators via majority vote:
    1. COT Index: 156-week percentile of speculative net positioning (contrarian)
    2. Commercial Hedger Change: week-over-week smart money direction
    3. Managed Money Intensity: speculator crowding z-score
    4. Real Yield Direction: falling yields = bullish gold (with regime gate)

Applicable to XAU-USD and XAG-USD only.  Data sourced from precomputed
deep context files (metals_precompute.py) with CFTC API fallback.

COT reports are published weekly (Friday for Tuesday data), so this signal
is inherently slower-moving than price-based signals.  Max confidence
capped at 0.7 to reflect data staleness.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.cot_positioning")

# SM-P1-4 (2026-05-02 adversarial follow-ups): absolute path resolution.
# The previous code used relative `Path("data")` / `data/...` which silently
# broke when the scheduled task CWD differed from the repo root (e.g.
# PF-DataLoop launched from C:\Windows). The deep context and COT history
# loaders would return None, the signal would silently fall back to API
# fetching every cycle. Mirrors the c5b78210 ic_computation fix.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}
_COMMODITY_MAP = {"XAU-USD": "gold", "XAG-USD": "silver"}

# COT Index thresholds (0-100 percentile scale)
_COT_EXTREME_HIGH = 80  # Overbought (contrarian SELL)
_COT_EXTREME_LOW = 20   # Oversold (contrarian BUY)

# Commercial hedger change threshold (contracts)
_COMM_CHANGE_THRESHOLD = 5000

# Managed money z-score threshold
_MM_ZSCORE_HIGH = 1.5
_MM_ZSCORE_LOW = -1.5

# CFTC API settings for historical lookback
_COT_HISTORY_WEEKS = 156  # 3 years
_CFTC_LEGACY_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_CFTC_TIMEOUT = 15


def _load_deep_context(ticker: str) -> dict | None:
    """Load precomputed deep context for the given metal.

    SM-P1-4 (2026-05-02): use absolute _DATA_DIR - was relative `f"data/..."`
    which silently broke when the loop's CWD wasn't the repo root.
    """
    from portfolio.file_utils import load_json

    metal = _COMMODITY_MAP.get(ticker)
    if not metal:
        return None

    path = str(_DATA_DIR / f"{metal}_deep_context.json")
    ctx = load_json(path, default=None)
    if not ctx or not isinstance(ctx, dict):
        logger.debug("Deep context not available: %s", path)
        return None
    return ctx


def _load_cot_history(metal: str) -> list[dict]:
    """Load COT history from the local JSONL file.

    SM-P1-4 (2026-05-02): use absolute _DATA_DIR - was relative
    `"data/cot_history.jsonl"`.
    """
    from portfolio.file_utils import load_jsonl

    entries = load_jsonl(str(_DATA_DIR / "cot_history.jsonl"))
    return [e for e in entries if e.get("metal") == metal]


def _fetch_cot_historical(commodity_name: str) -> list[dict]:
    """Fetch historical COT data from CFTC SOCRATA API.

    Returns up to 156 weeks of net positioning data for z-score computation.
    commodity_name should be 'GOLD' or 'SILVER'.
    """
    try:
        import requests

        url = (
            f"{_CFTC_LEGACY_URL}"
            f"?$where=commodity_name='{commodity_name}'"
            f"&$order=report_date_as_yyyy_mm_dd DESC"
            f"&$limit={_COT_HISTORY_WEEKS}"
        )
        resp = requests.get(url, timeout=_CFTC_TIMEOUT)
        resp.raise_for_status()
        rows = resp.json()

        result = []
        for r in rows:
            nc_long = _int_safe(r.get("noncomm_positions_long_all"))
            nc_short = _int_safe(r.get("noncomm_positions_short_all"))
            comm_long = _int_safe(r.get("comm_positions_long_all"))
            comm_short = _int_safe(r.get("comm_positions_short_all"))

            if nc_long is not None and nc_short is not None:
                result.append({
                    "date": r.get("report_date_as_yyyy_mm_dd", ""),
                    "nc_net": nc_long - nc_short,
                    "comm_net": (comm_long - comm_short)
                    if comm_long is not None and comm_short is not None
                    else None,
                    "oi": _int_safe(r.get("open_interest_all")),
                })
        return result
    except Exception as e:
        logger.warning("CFTC historical fetch failed: %s", e)
        return []


def _int_safe(val) -> int | None:
    """Safely convert to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _compute_cot_index(nc_net_history: list[int]) -> float | None:
    """Compute COT Index as percentile of current net positioning.

    Formula: (Current - Min_156w) / (Max_156w - Min_156w) * 100
    Returns 0-100 scale, or None if insufficient data.
    """
    if len(nc_net_history) < 10:  # Need minimum history for meaningful percentile
        return None

    current = nc_net_history[0]  # Most recent
    hist_min = min(nc_net_history)
    hist_max = max(nc_net_history)
    hist_range = hist_max - hist_min

    if hist_range == 0:
        return 50.0  # No variation

    return round((current - hist_min) / hist_range * 100, 1)


def _sub_cot_index(cot_data: dict, historical: list[dict]) -> tuple[str, float, dict]:
    """Sub-indicator 1: COT Index percentile (contrarian).

    >80 = speculators extremely long = contrarian SELL.
    <20 = speculators extremely short = contrarian BUY.
    """
    indicators = {"cot_index": None, "nc_net": None}

    nc_net = cot_data.get("noncomm_net")
    if nc_net is None:
        return "HOLD", 0.0, indicators

    indicators["nc_net"] = nc_net

    # Build history of nc_net values
    nc_net_history = [nc_net]
    for h in historical:
        val = h.get("nc_net")
        if val is not None:
            nc_net_history.append(val)

    cot_index = _compute_cot_index(nc_net_history)
    if cot_index is None:
        return "HOLD", 0.0, indicators

    indicators["cot_index"] = cot_index

    if cot_index > _COT_EXTREME_HIGH:
        # Speculators extremely long - contrarian SELL
        intensity = min((cot_index - _COT_EXTREME_HIGH) / 20.0, 1.0)
        return "SELL", round(0.4 + 0.3 * intensity, 2), indicators
    elif cot_index < _COT_EXTREME_LOW:
        # Speculators extremely short - contrarian BUY
        intensity = min((_COT_EXTREME_LOW - cot_index) / 20.0, 1.0)
        return "BUY", round(0.4 + 0.3 * intensity, 2), indicators

    return "HOLD", 0.0, indicators


def _sub_commercial_change(cot_data: dict) -> tuple[str, dict]:
    """Sub-indicator 2: Commercial hedger net change (smart money).

    Commercial traders are hedgers with actual exposure - their positioning
    reflects fundamental supply/demand knowledge.  Increasing net long
    (less short) = bullish signal.
    """
    indicators = {"comm_net": None, "comm_net_change": None}

    comm_net = cot_data.get("comm_net")
    if comm_net is None:
        return "HOLD", indicators

    indicators["comm_net"] = comm_net

    # Use the WoW change from metals_precompute if available
    change = cot_data.get("noncomm_net_change")
    if change is not None:
        # If non-commercial net is INCREASING, commercials are getting more short
        # (they're the counterparty).  Speculators adding longs = bearish contrarian.
        indicators["comm_net_change"] = -change  # Commercial change is inverse
        if change > _COMM_CHANGE_THRESHOLD:
            return "SELL", indicators  # Specs adding longs aggressively
        elif change < -_COMM_CHANGE_THRESHOLD:
            return "BUY", indicators   # Specs liquidating longs

    return "HOLD", indicators


def _sub_managed_money(cot_data: dict, historical: list[dict]) -> tuple[str, dict]:
    """Sub-indicator 3: Managed money (hedge fund) sentiment.

    Managed money net long/short intensity relative to recent history.
    Extreme long = contrarian SELL.  Extreme short = contrarian BUY.
    """
    indicators = {"mm_net": None, "mm_zscore": None}

    mm_net = cot_data.get("managed_money_net")
    if mm_net is None:
        return "HOLD", indicators

    indicators["mm_net"] = mm_net

    # Compute z-score against local history
    mm_history = [mm_net]
    for e in historical:
        val = e.get("mm_net")
        if val is not None:
            mm_history.append(val)

    if len(mm_history) < 5:
        return "HOLD", indicators

    import numpy as np

    mean = np.mean(mm_history)
    std = np.std(mm_history)
    if std < 1:
        return "HOLD", indicators

    zscore = (mm_net - mean) / std
    indicators["mm_zscore"] = round(float(zscore), 2)

    if zscore > _MM_ZSCORE_HIGH:
        return "SELL", indicators  # Managed money extremely long - contrarian SELL
    elif zscore < _MM_ZSCORE_LOW:
        return "BUY", indicators   # Managed money extremely short - contrarian BUY

    return "HOLD", indicators


def _sub_real_yield(deep_ctx: dict, ticker: str) -> tuple[str, dict]:
    """Sub-indicator 4: Real yield direction (gold-specific).

    Falling real yields = lower opportunity cost for gold = BUY.
    Rising real yields = higher opportunity cost = SELL.

    CAVEAT: The gold-real yield inverse correlation has weakened since 2022.
    This sub-indicator carries lower weight and is gated by direction clarity.
    """
    indicators = {"real_yield": None, "real_yield_direction": None}

    # Try to get FRED data from deep context
    fred = None
    if isinstance(deep_ctx, dict):
        # Check in refresh_data section
        refresh = deep_ctx.get("refresh_data", {})
        if isinstance(refresh, dict):
            fred = refresh.get("fred")
        # Also check top-level
        if fred is None:
            fred = deep_ctx.get("fred")

    if not isinstance(fred, dict):
        return "HOLD", indicators

    real_yield = fred.get("real_yield")
    direction = fred.get("real_yield_direction")

    if real_yield is not None:
        indicators["real_yield"] = real_yield
    if direction is not None:
        indicators["real_yield_direction"] = direction

    if direction == "falling":
        return "BUY", indicators  # Falling yields = bullish gold
    elif direction == "rising":
        return "SELL", indicators  # Rising yields = bearish gold

    return "HOLD", indicators


def compute_cot_positioning_signal(
    df: Any,
    context: dict | None = None,
    **kwargs,
) -> dict:
    """Compute COT positioning composite signal for precious metals.

    Args:
        df: Price DataFrame (used minimally - this is a fundamental signal).
        context: Must contain 'ticker' key for asset identification.

    Returns:
        Standard signal dict with action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    # Extract ticker from context or kwargs
    ticker = ""
    if context and isinstance(context, dict):
        ticker = context.get("ticker", "")
    ticker = kwargs.get("ticker", ticker)

    if ticker not in _METALS_TICKERS:
        return empty

    metal = _COMMODITY_MAP[ticker]

    # Load precomputed deep context
    deep_ctx = _load_deep_context(ticker)

    # Extract COT data from deep context.
    # metals_precompute stores COT at external_research.cot_positioning.live
    cot_data = None
    if deep_ctx:
        ext = deep_ctx.get("external_research", {})
        if isinstance(ext, dict):
            cot_section = ext.get("cot_positioning", {})
            if isinstance(cot_section, dict):
                cot_data = cot_section.get("live")

    if not cot_data or not isinstance(cot_data, dict):
        logger.debug("No COT data available for %s", ticker)
        return empty

    # Load local COT history for z-score computation
    local_history = _load_cot_history(metal)

    # If local history is thin (<20 entries), fetch from CFTC API
    historical_data = []
    if len(local_history) < 20:
        cftc_name = "GOLD" if metal == "gold" else "SILVER"
        historical_data = _fetch_cot_historical(cftc_name)
    else:
        historical_data = local_history

    # Compute sub-indicators
    votes = []
    sub_signals = {}
    all_indicators = {}

    # Sub 1: COT Index (contrarian percentile)
    cot_vote, cot_conf, cot_ind = _sub_cot_index(cot_data, historical_data)
    sub_signals["cot_index"] = cot_vote
    all_indicators.update(cot_ind)
    votes.append(cot_vote)

    # Sub 2: Commercial hedger change (smart money)
    comm_vote, comm_ind = _sub_commercial_change(cot_data)
    sub_signals["commercial_change"] = comm_vote
    all_indicators.update(comm_ind)
    votes.append(comm_vote)

    # Sub 3: Managed money intensity
    mm_vote, mm_ind = _sub_managed_money(cot_data, historical_data)
    sub_signals["managed_money"] = mm_vote
    all_indicators.update(mm_ind)
    votes.append(mm_vote)

    # Sub 4: Real yield direction (gold-specific, lower weight)
    ry_vote, ry_ind = _sub_real_yield(deep_ctx, ticker)
    sub_signals["real_yield"] = ry_vote
    all_indicators.update(ry_ind)
    votes.append(ry_vote)

    # Add report date to indicators
    all_indicators["report_date"] = cot_data.get("report_date")
    all_indicators["ticker"] = ticker

    # Majority vote
    action, confidence = majority_vote(votes, count_hold=False)

    # Cap confidence at 0.7 (external data, weekly update cycle)
    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": all_indicators,
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\futures_flow.py | Select-Object -First 340" in Q:\finance-analyzer
 succeeded in 721ms:
"""Futures flow signal - OI, long/short ratios, funding rate history.

Composite signal with 6 sub-indicators, majority vote, confidence capped at 0.7.
Only applicable to crypto tickers (BTC-USD, ETH-USD). Non-crypto  immediate HOLD.

Sub-indicators:
  1. oi_trend       - Rising OI + price direction  new longs/shorts
  2. oi_divergence  - OI/price divergence  de-risking or hidden buildup
  3. ls_extreme     - Contrarian: crowd overleveraged one side
  4. top_vs_crowd   - Top traders disagree with crowd  follow smart money
  5. funding_trend  - Funding rate extremes  contrarian
  6. oi_acceleration - Second derivative of OI  momentum confirmation

The ``context`` parameter is a dict with keys: ticker, config, macro.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.futures_flow")

# Confidence cap (same as news_event, econ_calendar)
_MAX_CONFIDENCE = 0.7

# Thresholds
_LS_EXTREME_HIGH = 2.0   # crowd overleveraged long
_LS_EXTREME_LOW = 0.7    # crowd overleveraged short
_FUNDING_HIGH = 0.0005   # 0.05% - contrarian SELL
_FUNDING_LOW = -0.0003   # -0.03% - contrarian BUY
_TOP_DIVERGE_THRESHOLD = 0.3  # top trader LS differs from crowd by this much

# Minimum data points needed for trend calculations
_MIN_HISTORY = 5


def _oi_trend(oi_history, df):
    """Sub-1: OI trend + price direction.

    Rising OI + price up = BUY (new longs entering)
    Rising OI + price down = SELL (new shorts entering)
    Falling OI = HOLD (deleveraging, no directional signal)
    """
    if not oi_history or len(oi_history) < _MIN_HISTORY:
        return "HOLD"

    recent_oi = [d.get("oi", 0) or 0 for d in oi_history[-_MIN_HISTORY:]]  # H16
    oi_change = (recent_oi[-1] - recent_oi[0]) / recent_oi[0] if recent_oi[0] else 0

    if oi_change <= 0.005:  # OI not meaningfully rising (< 0.5%)
        return "HOLD"

    # Price direction from OHLCV dataframe
    if df is not None and len(df) >= _MIN_HISTORY:
        price_start = float(df["close"].iloc[-_MIN_HISTORY])
        price_end = float(df["close"].iloc[-1])
        # H35: NaN is truthy in Python; explicit guard required.
        # SM-P1-2 (2026-05-02 adversarial follow-ups): the original code
        # used `if price_start and price_end > price_start` which Python
        # parses as `(price_start) and (price_end > price_start)` - correct
        # but easy to misread (could look like `(price_start and price_end)
        # > price_start`). Replace truthiness check with explicit `> 0` so
        # the intent (no division-by-zero, positive prices only) is obvious.
        if not math.isnan(price_start) and not math.isnan(price_end) and price_start > 0:
            if price_end > price_start:
                return "BUY"   # rising OI + rising price = new longs
            if price_end < price_start:
                return "SELL"  # rising OI + falling price = new shorts

    return "HOLD"


def _oi_divergence(oi_history, df):
    """Sub-2: OI/price divergence.

    Price up but OI falling = bearish divergence (SELL) - rally on thin leverage
    Price down but OI falling = de-risking/capitulation (BUY) - sellers exhausting
    """
    if not oi_history or len(oi_history) < _MIN_HISTORY:
        return "HOLD"

    recent_oi = [d.get("oi", 0) or 0 for d in oi_history[-_MIN_HISTORY:]]  # H16
    oi_change = (recent_oi[-1] - recent_oi[0]) / recent_oi[0] if recent_oi[0] else 0

    if abs(oi_change) < 0.005:  # OI flat - no divergence
        return "HOLD"

    if df is not None and len(df) >= _MIN_HISTORY:
        price_start = float(df["close"].iloc[-_MIN_HISTORY])
        price_end = float(df["close"].iloc[-1])
        # H35: Explicit NaN guard.
        if not math.isnan(price_start) and not math.isnan(price_end) and price_start:
            price_change = (price_end - price_start) / price_start

            if price_change > 0.005 and oi_change < -0.005:
                return "SELL"  # price up, OI down - bearish divergence
            if price_change < -0.005 and oi_change < -0.005:
                return "BUY"   # price down, OI down - de-risking, capitulation buy

    return "HOLD"


def _ls_extreme(ls_ratio):
    """Sub-3: Long/short ratio extreme - contrarian.

    LS > 2.0 = crowd overleveraged long  contrarian SELL
    LS < 0.7 = crowd overleveraged short  contrarian BUY
    """
    if not ls_ratio:
        return "HOLD"

    latest = ls_ratio[-1]["longShortRatio"]
    if latest > _LS_EXTREME_HIGH:
        return "SELL"
    elif latest < _LS_EXTREME_LOW:
        return "BUY"
    return "HOLD"


def _top_vs_crowd(top_position_ratio, ls_ratio):
    """Sub-4: Top trader vs crowd divergence - follow top traders.

    When top traders' positioning significantly differs from crowd,
    follow the top traders (they tend to be right).
    """
    if not top_position_ratio or not ls_ratio:
        return "HOLD"

    top_ls = top_position_ratio[-1]["longShortRatio"]
    crowd_ls = ls_ratio[-1]["longShortRatio"]

    diff = top_ls - crowd_ls
    if abs(diff) < _TOP_DIVERGE_THRESHOLD:
        return "HOLD"

    # Top traders more long than crowd  BUY
    if diff > _TOP_DIVERGE_THRESHOLD:
        return "BUY"
    # Top traders more short than crowd  SELL
    if diff < -_TOP_DIVERGE_THRESHOLD:
        return "SELL"

    return "HOLD"


def _funding_trend(funding_history):
    """Sub-5: Funding rate trend - contrarian.

    Rising funding > 0.05% = overleveraged longs  contrarian SELL
    Deeply negative < -0.03% = overleveraged shorts  contrarian BUY
    Also checks 3-period trend for strengthening signal.
    """
    if not funding_history or len(funding_history) < 3:
        return "HOLD"

    latest_rate = funding_history[-1]["fundingRate"]

    # 3-period average for trend
    recent_rates = [d["fundingRate"] for d in funding_history[-3:]]
    avg_rate = sum(recent_rates) / len(recent_rates)

    # Use average to smooth noise
    if avg_rate > _FUNDING_HIGH:
        return "SELL"
    elif avg_rate < _FUNDING_LOW:
        return "BUY"

    # Check if latest is extreme even if average isn't
    if latest_rate > _FUNDING_HIGH * 1.5:
        return "SELL"
    elif latest_rate < _FUNDING_LOW * 1.5:
        return "BUY"

    return "HOLD"


def _oi_acceleration(oi_history, df):
    """Sub-6: OI acceleration - second derivative of OI.

    Accelerating OI growth + aligned price = strong momentum confirmation.
    Decelerating OI growth = momentum fading.
    """
    if not oi_history or len(oi_history) < _MIN_HISTORY * 2:
        return "HOLD"

    oi_values = [d.get("oi", 0) or 0 for d in oi_history]  # H16

    # First derivative: rate of change
    mid = len(oi_values) // 2
    first_half_change = (oi_values[mid] - oi_values[0]) / oi_values[0] if oi_values[0] else 0
    second_half_change = (oi_values[-1] - oi_values[mid]) / oi_values[mid] if oi_values[mid] else 0

    # Second derivative: acceleration
    acceleration = second_half_change - first_half_change

    if abs(acceleration) < 0.005:  # not meaningful
        return "HOLD"

    # Check price alignment
    if df is not None and len(df) >= _MIN_HISTORY:
        price_start = float(df["close"].iloc[-_MIN_HISTORY])
        price_end = float(df["close"].iloc[-1])
        # H35: Explicit NaN guard.
        if math.isnan(price_start) or math.isnan(price_end):
            return "HOLD"
        price_up = price_end > price_start

        if acceleration > 0.005:  # OI accelerating
            return "BUY" if price_up else "SELL"
        elif acceleration < -0.005:  # OI decelerating - momentum fading
            # Fading momentum  contrarian lean
            return "SELL" if price_up else "BUY"

    return "HOLD"


def compute_futures_flow_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the futures flow composite signal.

    Args:
        df: OHLCV DataFrame for the ticker.
        context: dict with keys {ticker, config, macro}.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    ticker = context.get("ticker", "") if context else ""

    # Non-crypto  immediate HOLD
    from portfolio.tickers import CRYPTO_SYMBOLS
    if ticker not in CRYPTO_SYMBOLS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    # Fetch all futures data
    from portfolio.futures_data import get_all_futures_data
    futures = _cached(
        f"futures_flow_data_{ticker}",
        300,
        get_all_futures_data,
        ticker,
    )

    if futures is None:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {"error": "fetch_failed"},
        }

    oi_history = futures.get("oi_history")
    ls_ratio = futures.get("ls_ratio")
    top_position = futures.get("top_position_ratio")
    funding_hist = futures.get("funding_history")
    current_oi = futures.get("open_interest")

    # Compute sub-signals
    sub = {
        "oi_trend": _oi_trend(oi_history, df),
        "oi_divergence": _oi_divergence(oi_history, df),
        "ls_extreme": _ls_extreme(ls_ratio),
        "top_vs_crowd": _top_vs_crowd(top_position, ls_ratio),
        "funding_trend": _funding_trend(funding_hist),
        "oi_acceleration": _oi_acceleration(oi_history, df),
    }

    # Majority vote - pass list of vote strings, not the dict
    action, confidence = majority_vote(list(sub.values()))
    confidence = min(confidence, _MAX_CONFIDENCE)

    # Build indicators dict for reporting
    indicators = {}
    if current_oi:
        indicators["open_interest"] = current_oi.get("oi")
    if oi_history and len(oi_history) >= 2:
        oi_vals = [d["oi"] for d in oi_history]
        indicators["oi_change_pct"] = round(
            (oi_vals[-1] - oi_vals[0]) / oi_vals[0] * 100 if oi_vals[0] else 0, 2
        )
    if ls_ratio:
        latest_ls = ls_ratio[-1]
        indicators["ls_ratio"] = round(latest_ls["longShortRatio"], 3)
        indicators["long_pct"] = round(latest_ls["longAccount"] * 100, 1)
        indicators["short_pct"] = round(latest_ls["shortAccount"] * 100, 1)
    if top_position:
        indicators["top_trader_ls"] = round(top_position[-1]["longShortRatio"], 3)
    if funding_hist and len(funding_hist) >= 1:
        indicators["funding_rate"] = funding_hist[-1]["fundingRate"]
        indicators["funding_rate_pct"] = round(funding_hist[-1]["fundingRate"] * 100, 4)
        if len(funding_hist) >= 3:
            recent = [d["fundingRate"] for d in funding_hist[-3:]]
            indicators["funding_3period_avg"] = round(sum(recent) / len(recent) * 100, 4)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub,
        "indicators": indicators,
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\econ_calendar.py | Select-Object -First 280" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\calendar_seasonal.py | Select-Object -First 520" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\intraday_seasonality.py | Select-Object -First 240" in Q:\finance-analyzer
 succeeded in 861ms:
"""Economic calendar signal - event proximity, risk-off, and post-event relief.

Combines five sub-indicators into a majority-vote composite:
  1. event_proximity    - hours until next event; <4h risk-off, <24h cautious
  2. event_type         - classify event for informational purposes
  3. pre_event_risk     - binary risk-off within 4h of high-impact event
  4. sector_exposure    - is this ticker's sector affected by the upcoming event?
  5. post_event_relief  - BUY after high-impact event passes (uncertainty premium removed)

BUG-218 fix: added post_event_relief sub-signal so the composite can emit BUY,
not just SELL/HOLD. Previously structurally SELL-only.

The ``context`` parameter is a dict with keys: ticker, config, macro.
Uses economic calendar dates from econ_dates.py and FOMC dates from fomc_dates.py.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from portfolio.econ_dates import (
    EVENT_SECTOR_MAP,
    events_within_hours,
    next_event,
    recent_high_impact_events,
)
from portfolio.news_keywords import TICKER_SECTORS
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.econ_calendar")

# Max confidence cap
_MAX_CONFIDENCE = 0.7


def _get_current_date(df: pd.DataFrame) -> datetime:
    """Extract current timestamp from df or use now()."""
    if df is not None and "time" in df.columns and len(df) > 0:
        last_time = df["time"].iloc[-1]
        if isinstance(last_time, pd.Timestamp):
            return last_time.to_pydatetime().replace(tzinfo=UTC)
    return datetime.now(UTC)


def _event_proximity(ref_date) -> tuple[str, dict]:
    """Hours until next event  risk-off signal.

    <4h = SELL (risk-off before high-vol event)
    4-24h = cautious SELL (only for high-impact events)
    >24h = HOLD
    """
    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    indicators = {"next_event": None, "hours_until": None}

    if evt is None:
        return "HOLD", indicators

    indicators["next_event"] = f"{evt['type']} {evt['date'].isoformat()}"
    indicators["hours_until"] = evt["hours_until"]

    if evt["hours_until"] <= 4:
        return "SELL", indicators
    if evt["hours_until"] <= 24 and evt["impact"] == "high":
        return "SELL", indicators
    return "HOLD", indicators


def _event_type_info(ref_date) -> tuple[str, dict]:
    """Classify upcoming event type (informational, doesn't vote strongly).

    FOMC and CPI = high uncertainty  lean SELL.
    NFP = moderate  HOLD.
    GDP and others = low  HOLD.
    """
    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    indicators = {"event_type": None, "event_impact": None}

    if evt is None:
        return "HOLD", indicators

    indicators["event_type"] = evt["type"]
    indicators["event_impact"] = evt["impact"]

    # Only vote if event is within 48h
    if evt["hours_until"] > 48:
        return "HOLD", indicators

    if evt["type"] in ("FOMC", "CPI") and evt["hours_until"] <= 24:
        return "SELL", indicators
    return "HOLD", indicators


def _pre_event_risk(ref_date) -> tuple[str, dict]:
    """Binary risk-off within 4h of any high-impact event.

    This is the strongest sub-signal: if ANY high-impact event is within
    4 hours, vote SELL unconditionally.
    """
    nearby = events_within_hours(4, ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    high_impact = [e for e in nearby if e["impact"] == "high"]

    indicators = {"events_within_4h": len(nearby),
                  "high_impact_within_4h": len(high_impact)}

    if high_impact:
        indicators["nearest_event"] = f"{high_impact[0]['type']} in {high_impact[0]['hours_until']}h"
        return "SELL", indicators
    return "HOLD", indicators


def _post_event_relief(ref_date) -> tuple[str, dict]:
    """Post-event relief BUY signal.

    After a high-impact event has just passed (4-24h ago), the uncertainty
    premium dissipates and markets tend to rally. Also, when the next event
    is >72h away, reduced macro uncertainty is mildly bullish.

    BUG-218: This sub-signal adds BUY capability that was previously missing,
    making the econ_calendar signal structurally balanced (not SELL-only).
    """
    indicators = {"post_event_relief": False, "event_free_window": False}

    # Check for recently passed high-impact events (4-24h ago = relief window)
    recent = recent_high_impact_events(24)
    relief_events = [e for e in recent if e.get("hours_since", 0) >= 4]

    if relief_events:
        indicators["post_event_relief"] = True
        indicators["relief_event"] = (
            f"{relief_events[0]['type']} passed {relief_events[0]['hours_since']:.0f}h ago"
        )
        # Check that no new event is imminent (would negate the relief)
        evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
        if evt is None or evt["hours_until"] > 24:
            return "BUY", indicators

    # Event-free calm window: next event >72h away
    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    if evt is not None and evt["hours_until"] > 72:
        indicators["event_free_window"] = True
        indicators["next_event_hours"] = evt["hours_until"]
        return "BUY", indicators

    return "HOLD", indicators


def _sector_exposure(ref_date, ticker: str) -> tuple[str, dict]:
    """Check if this ticker's sector is affected by upcoming events.

    Maps event types to affected sectors via EVENT_SECTOR_MAP,
    then checks if the ticker belongs to any affected sector.
    """
    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
    indicators = {"ticker_sectors": list(TICKER_SECTORS.get(ticker, set())),
                  "event_affects_sector": False}

    if evt is None or evt["hours_until"] > 48:
        return "HOLD", indicators

    affected_sectors = EVENT_SECTOR_MAP.get(evt["type"], set())
    ticker_secs = TICKER_SECTORS.get(ticker, set())
    overlap = affected_sectors & ticker_secs

    indicators["affected_sectors"] = list(affected_sectors)
    indicators["overlap_sectors"] = list(overlap)
    indicators["event_affects_sector"] = bool(overlap)

    if overlap and evt["hours_until"] <= 24:
        return "SELL", indicators
    return "HOLD", indicators


def compute_econ_calendar_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the composite economic calendar signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (used to extract current timestamp).
    context : dict, optional
        Dict with keys: ticker, config, macro.

    Returns
    -------
    dict
        action, confidence, sub_signals, indicators
    """
    result = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "event_proximity": "HOLD",
            "event_type": "HOLD",
            "pre_event_risk": "HOLD",
            "sector_exposure": "HOLD",
            "post_event_relief": "HOLD",
        },
        "indicators": {},
    }

    ticker = ""
    if context:
        ticker = context.get("ticker", "")

    ref_date = _get_current_date(df)

    # Compute each sub-signal
    try:
        prox_action, prox_ind = _event_proximity(ref_date)
        # BUG-118: Warn when all economic dates have passed (data staleness)
        if prox_ind.get("next_event") is None:
            logger.warning("econ_calendar: no future events found - dates may need updating")
    except Exception:
        logger.exception("event_proximity sub-signal failed")
        prox_action, prox_ind = "HOLD", {}

    try:
        type_action, type_ind = _event_type_info(ref_date)
    except Exception:
        logger.exception("event_type sub-signal failed")
        type_action, type_ind = "HOLD", {}

    try:
        risk_action, risk_ind = _pre_event_risk(ref_date)
    except Exception:
        logger.exception("pre_event_risk sub-signal failed")
        risk_action, risk_ind = "HOLD", {}

    try:
        sec_action, sec_ind = _sector_exposure(ref_date, ticker)
    except Exception:
        logger.exception("sector_exposure sub-signal failed")
        sec_action, sec_ind = "HOLD", {}

    try:
        relief_action, relief_ind = _post_event_relief(ref_date)
    except Exception:
        logger.exception("post_event_relief sub-signal failed")
        relief_action, relief_ind = "HOLD", {}

    # Populate result
    result["sub_signals"]["event_proximity"] = prox_action
    result["sub_signals"]["event_type"] = type_action
    result["sub_signals"]["pre_event_risk"] = risk_action
    result["sub_signals"]["sector_exposure"] = sec_action
    result["sub_signals"]["post_event_relief"] = relief_action

    result["indicators"].update({f"proximity_{k}": v for k, v in prox_ind.items()})
    result["indicators"].update({f"type_{k}": v for k, v in type_ind.items()})
    result["indicators"].update({f"risk_{k}": v for k, v in risk_ind.items()})
    result["indicators"].update({f"exposure_{k}": v for k, v in sec_ind.items()})
    result["indicators"].update({f"relief_{k}": v for k, v in relief_ind.items()})

    # Majority vote (5 sub-signals: 3 SELL-capable + 1 BUY-capable + 1 either)
    votes = [prox_action, type_action, risk_action, sec_action, relief_action]
    result["action"], result["confidence"] = majority_vote(votes)

    # Cap confidence
    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)

    return result
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 872ms:
"""Intraday seasonality gate signal.

Exploits empirical hour-of-day and day-of-week return patterns to modulate
signal confidence. Three sub-indicators:

  1. Hour Alpha        - UTC hour maps to empirical return multiplier per asset class
  2. Day-of-Week Bias  - Monday crypto boost, Wednesday FOMC caution
  3. Trend Context     - EMA(9)/EMA(21) crossover for directional context

During high-alpha hours, confidence is boosted. During low-alpha hours,
signal outputs HOLD to suppress noise. Direction comes from trend context,
not from time alone.

Academic backing:
  - BTC 22:00-23:00 UTC: 33% annualized (ScienceDirect 2024)
  - Monday Asia Open: Sharpe 1.6 (Concretum Group 2025)
  - Gold London-NY overlap: 60-70% of daily range (CME Group 2026)
  - Crypto tea-time peak 16:00-17:00 UTC (Springer RQFA 2024)

Requires DataFrame with OHLCV and at least 21 rows.
"""
from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

from portfolio.signal_utils import ema, safe_float

logger = logging.getLogger(__name__)

MIN_ROWS = 21
_MAX_CONFIDENCE = 0.7

# Hour multipliers: >1.0 = high-alpha, <1.0 = low-alpha, ~0.0 = suppress
# Based on empirical intraday return patterns from academic literature.

_CRYPTO_HOUR_MULT = {
    0: 1.2, 1: 1.1, 2: 0.9, 3: 0.9, 4: 0.8, 5: 0.8,
    6: 0.9, 7: 0.9, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0,
    12: 0.9, 13: 0.9, 14: 0.7, 15: 0.7, 16: 0.8, 17: 0.7,
    18: 0.7, 19: 0.7, 20: 0.8, 21: 1.3, 22: 1.4, 23: 1.3,
}

_METALS_HOUR_MULT = {
    0: 0.7, 1: 0.7, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5,
    6: 0.6, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.0,
    12: 1.1, 13: 1.3, 14: 1.3, 15: 1.3, 16: 1.2, 17: 1.1,
    18: 0.9, 19: 0.8, 20: 0.7, 21: 0.6, 22: 0.6, 23: 0.6,
}

_STOCKS_HOUR_MULT = {
    0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3, 5: 0.3,
    6: 0.3, 7: 0.3, 8: 0.3, 9: 0.5, 10: 0.6, 11: 0.7,
    12: 0.8, 13: 0.9, 14: 1.3, 15: 1.2, 16: 1.0, 17: 0.7,
    18: 0.8, 19: 1.2, 20: 1.3, 21: 0.5, 22: 0.4, 23: 0.3,
}

_ASSET_HOUR_MAP = {
    "crypto": _CRYPTO_HOUR_MULT,
    "metals": _METALS_HOUR_MULT,
    "stocks": _STOCKS_HOUR_MULT,
}

# Day-of-week multipliers (0=Monday..6=Sunday)
_CRYPTO_DOW_MULT = {0: 1.15, 1: 1.0, 2: 0.9, 3: 1.0, 4: 1.0, 5: 0.95, 6: 1.1}
_METALS_DOW_MULT = {0: 1.0, 1: 1.05, 2: 0.9, 3: 1.0, 4: 0.95, 5: 0.5, 6: 0.5}
_STOCKS_DOW_MULT = {0: 0.95, 1: 1.05, 2: 0.9, 3: 1.0, 4: 1.05, 5: 0.3, 6: 0.3}

_ASSET_DOW_MAP = {
    "crypto": _CRYPTO_DOW_MULT,
    "metals": _METALS_DOW_MULT,
    "stocks": _STOCKS_DOW_MULT,
}


def _get_utc_hour_and_dow(df: pd.DataFrame) -> tuple[int, int]:
    """Extract UTC hour and day-of-week from DataFrame's last timestamp."""
    if hasattr(df.index, "hour"):
        try:
            last_ts = df.index[-1]
            if hasattr(last_ts, "hour"):
                if last_ts.tzinfo is not None:
                    utc_ts = last_ts.astimezone(datetime.timezone.utc)
                    return utc_ts.hour, utc_ts.weekday()
                return last_ts.hour, last_ts.weekday()
        except Exception:
            pass
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.hour, now.weekday()


def _classify_asset(context: dict | None) -> str:
    """Determine asset class from context."""
    if not context:
        return "crypto"
    ac = context.get("asset_class", "")
    if ac in ("crypto", "metals", "stocks"):
        return ac
    ticker = context.get("ticker", "")
    if "XAU" in ticker or "XAG" in ticker:
        return "metals"
    if "MSTR" in ticker:
        return "stocks"
    return "crypto"


def _hour_alpha_vote(hour: int, asset_class: str) -> tuple[str, float]:
    """Return (vote, raw_multiplier) based on hour and asset class."""
    mult_map = _ASSET_HOUR_MAP.get(asset_class, _CRYPTO_HOUR_MULT)
    mult = mult_map.get(hour, 1.0)
    if mult >= 1.2:
        return "BUY", mult
    if mult <= 0.5:
        return "HOLD", mult
    return "HOLD", mult


def _dow_vote(dow: int, asset_class: str) -> tuple[str, float]:
    """Return (vote, multiplier) based on day of week."""
    dow_map = _ASSET_DOW_MAP.get(asset_class, _CRYPTO_DOW_MULT)
    mult = dow_map.get(dow, 1.0)
    if mult >= 1.1:
        return "BUY", mult
    if mult <= 0.5:
        return "HOLD", mult
    return "HOLD", mult


def _trend_direction(close: pd.Series) -> tuple[str, float]:
    """EMA(9)/EMA(21) crossover for direction."""
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    if ema9 is None or ema21 is None:
        return "HOLD", 0.0
    last_fast = safe_float(ema9.iloc[-1])
    last_slow = safe_float(ema21.iloc[-1])
    if last_fast is None or last_slow is None:
        return "HOLD", 0.0
    if np.isnan(last_fast) or np.isnan(last_slow) or last_slow == 0:
        return "HOLD", 0.0
    pct_diff = (last_fast - last_slow) / last_slow
    if pct_diff > 0.002:
        return "BUY", abs(pct_diff)
    if pct_diff < -0.002:
        return "SELL", abs(pct_diff)
    return "HOLD", abs(pct_diff)


def compute_intraday_seasonality_signal(
    df: pd.DataFrame, context: dict | None = None
) -> dict:
    """Compute intraday seasonality gate signal."""
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"].dropna()
    if len(close) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    asset_class = _classify_asset(context)
    utc_hour, dow = _get_utc_hour_and_dow(df)

    # Sub-signal 1: Hour alpha
    hour_vote, hour_mult = _hour_alpha_vote(utc_hour, asset_class)

    # Sub-signal 2: Day-of-week
    dow_vote_str, dow_mult = _dow_vote(dow, asset_class)

    # Sub-signal 3: Trend direction
    trend_vote, trend_strength = _trend_direction(close)

    # Combine: hour/dow determine WHEN to act, trend determines direction
    combined_mult = hour_mult * dow_mult

    if combined_mult < 0.6:
        action = "HOLD"
        confidence = 0.0
    elif combined_mult >= 1.1 and trend_vote != "HOLD":
        action = trend_vote
        base_conf = min(0.3 + trend_strength * 5.0, 0.6)
        confidence = min(base_conf * (combined_mult / 1.0), _MAX_CONFIDENCE)
    elif trend_vote != "HOLD":
        action = trend_vote
        confidence = min(0.2 + trend_strength * 3.0, 0.4) * combined_mult
    else:
        action = "HOLD"
        confidence = 0.0

    confidence = round(min(max(confidence, 0.0), _MAX_CONFIDENCE), 4)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "hour_alpha": hour_vote,
            "day_of_week": dow_vote_str,
            "trend_context": trend_vote,
        },
        "indicators": {
            "utc_hour": utc_hour,
            "day_of_week": dow,
            "hour_multiplier": round(hour_mult, 3),
            "dow_multiplier": round(dow_mult, 3),
            "combined_multiplier": round(combined_mult, 3),
            "trend_strength": round(trend_strength, 6),
            "asset_class": asset_class,
        },
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 902ms:
"""Calendar-based and seasonal trading signals.

Combines eight sub-indicators into a majority-vote composite:
  1. Day-of-Week Effect       (Monday=SELL, Friday=BUY)
  2. Turnaround Tuesday       (Tuesday reversal after red Monday)
  3. Month-End Effect          (Last 3 calendar days bullish)
  4. Sell in May / Halloween   (May-Oct=SELL, Nov-Apr=BUY)
  5. January Effect            (Jan=BUY, Dec=SELL)
  6. Pre-Holiday Effect        (Day before multi-day gap=BUY)
  7. FOMC Drift                (2 days before FOMC=BUY, day of/after=HOLD)
  8. Santa Claus Rally         (Last 5 trading days Dec + first 2 Jan=BUY)

Each sub-indicator votes BUY / SELL / HOLD.  The composite action is the
majority vote; confidence is the fraction of non-HOLD votes that agree with
the majority direction.  Maximum confidence is capped at 0.6 because
calendar signals are inherently weak.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta

import pandas as pd

logger = logging.getLogger("portfolio.signals.calendar_seasonal")

# ---------------------------------------------------------------------------
# FOMC meeting dates - imported from shared constant
# ---------------------------------------------------------------------------
from portfolio.fomc_dates import (
    FOMC_ANNOUNCEMENT_DATES as _FOMC_ANNOUNCEMENT_DATES,
)

# H14: Cache max() at module level - was recomputed on every call (420 calls/cycle).
_FOMC_MAX_DATE = max(_FOMC_ANNOUNCEMENT_DATES) if _FOMC_ANNOUNCEMENT_DATES else None

# Maximum confidence for any calendar signal
_MAX_CONFIDENCE = 0.6

# Minimum rows needed for Turnaround Tuesday check
_MIN_BARS = 2


# ---------------------------------------------------------------------------
# Sub-signal functions
# ---------------------------------------------------------------------------

def _day_of_week_effect(last_date: date) -> tuple[str, dict]:
    """Monday historically bearish for equities (SELL), Friday bullish (BUY).

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    dow = last_date.weekday()  # 0=Mon, 4=Fri
    indicators = {"day_of_week": dow, "day_name": last_date.strftime("%A")}

    if dow == 0:  # Monday
        return "SELL", indicators
    if dow == 4:  # Friday
        return "BUY", indicators
    return "HOLD", indicators


def _turnaround_tuesday(df: pd.DataFrame, last_date: date) -> tuple[str, dict]:
    """Tuesday reversal after a red Monday.

    If the last bar is Tuesday AND the prior bar closed below its open
    (red candle on Monday), signal BUY.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with at least 2 rows.
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    indicators = {"is_tuesday": False, "prior_bar_red": False}

    if last_date.weekday() != 1:  # Not Tuesday
        return "HOLD", indicators

    indicators["is_tuesday"] = True

    if len(df) < 2:
        return "HOLD", indicators

    prior_close = float(df["close"].iloc[-2])
    prior_open = float(df["open"].iloc[-2])
    prior_red = prior_close < prior_open
    indicators["prior_bar_red"] = prior_red

    if prior_red:
        return "BUY", indicators
    return "HOLD", indicators


def _month_end_effect(last_date: date) -> tuple[str, dict]:
    """Last 3 calendar days of the month tend to be bullish.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
    days_remaining = days_in_month - last_date.day
    is_month_end = days_remaining < 3  # last 3 calendar days (day 29, 30, 31 of a 31-day month)
    indicators = {
        "is_month_end": is_month_end,
        "days_remaining_in_month": days_remaining,
    }

    if is_month_end:
        return "BUY", indicators
    return "HOLD", indicators


def _sell_in_may(last_date: date) -> tuple[str, dict]:
    """Sell in May and go away / Halloween indicator.

    May through October = historically weaker (SELL bias).
    Only historically *strong* months vote BUY: Nov, Dec, Jan, Apr.
    Transitional months (Feb, Mar) = HOLD - they are not consistently
    strong enough to justify a standing BUY vote.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    month = last_date.month
    is_weak_period = 5 <= month <= 10
    is_strong_month = month in (1, 4, 11, 12)  # Jan, Apr, Nov, Dec
    indicators = {"month": month, "is_weak_period": is_weak_period}

    if is_weak_period:
        return "HOLD", indicators
    if is_strong_month:
        return "BUY", indicators
    return "HOLD", indicators


def _january_effect(last_date: date) -> tuple[str, dict]:
    """January historically bullish for small caps; December tax-loss selling.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    month = last_date.month
    indicators = {"month": month}

    if month == 1:
        return "BUY", indicators
    if month == 12:
        return "SELL", indicators
    return "HOLD", indicators


def _pre_holiday_effect(last_date: date) -> tuple[str, dict]:
    """Trading day before a market holiday tends to be bullish.

    Checks for days preceding US market holidays (approximate).
    Regular Fridays are NOT counted here - that is already handled by
    ``_day_of_week_effect``.  Only true pre-holiday sessions (the
    trading day before a market closure beyond the normal weekend) vote.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    # Major US market holidays (month, day) - approximate, does not handle
    # observed-date shifts (e.g., July 4 on Saturday  Friday off).
    _US_HOLIDAYS = [
        (1, 1),    # New Year's Day
        (1, 20),   # MLK Day (approx - 3rd Monday)
        (2, 17),   # Presidents' Day (approx - 3rd Monday)
        (5, 26),   # Memorial Day (approx - last Monday)
        (6, 19),   # Juneteenth
        (7, 4),    # Independence Day
        (9, 1),    # Labor Day (approx - 1st Monday)
        (11, 27),  # Thanksgiving (approx - 4th Thursday)
        (12, 25),  # Christmas
    ]

    next_day = last_date + timedelta(days=1)
    is_pre_holiday = (next_day.month, next_day.day) in _US_HOLIDAYS
    indicators = {"is_pre_holiday": is_pre_holiday, "day_of_week": last_date.weekday()}

    if is_pre_holiday:
        return "BUY", indicators
    return "HOLD", indicators


def _fomc_drift(last_date: date) -> tuple[str, dict]:
    """Pre-FOMC announcement drift.

    The 24 hours before a scheduled FOMC announcement tend to drift upward.
    BUY if within 2 days before an announcement date.
    HOLD on the day of the announcement or the day after (volatility).

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    indicators = {
        "is_pre_fomc": False,
        "is_fomc_day": False,
        "is_post_fomc": False,
        "days_to_fomc": None,
    }

    # BUG-118: Warn when all FOMC dates are in the past (data staleness)
    if _FOMC_MAX_DATE is not None and last_date > _FOMC_MAX_DATE:  # H14: cached
        logger.warning("calendar_seasonal: all FOMC dates are in the past - fomc_dates.py needs updating")

    for fomc_date in _FOMC_ANNOUNCEMENT_DATES:
        delta = (fomc_date - last_date).days

        if delta == 0:
            # Day of FOMC announcement
            indicators["is_fomc_day"] = True
            indicators["days_to_fomc"] = 0
            return "HOLD", indicators

        if delta == -1:
            # Day after FOMC announcement
            indicators["is_post_fomc"] = True
            indicators["days_to_fomc"] = -1
            return "HOLD", indicators

        if 1 <= delta <= 2:
            # 1-2 days before FOMC announcement
            indicators["is_pre_fomc"] = True
            indicators["days_to_fomc"] = delta
            return "BUY", indicators

    return "HOLD", indicators


def _santa_claus_rally(last_date: date) -> tuple[str, dict]:
    """Santa Claus Rally: last 5 trading days of Dec + first 2 of Jan.

    Approximate by checking calendar dates: Dec 25-31 and Jan 1-3.
    The actual trading days vary by year, but this is a reasonable proxy.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    month = last_date.month
    day = last_date.day

    # Last 5 trading days of Dec: approx Dec 24-31 (some are holidays/weekends)
    is_late_dec = month == 12 and day >= 24
    # First 2 trading days of Jan: approx Jan 1-3
    is_early_jan = month == 1 and day <= 3

    is_santa_rally = is_late_dec or is_early_jan
    indicators = {"is_santa_rally": is_santa_rally, "month": month, "day": day}

    if is_santa_rally:
        return "BUY", indicators
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def compute_calendar_signal(df: pd.DataFrame) -> dict:
    """Compute the composite calendar/seasonal signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data with columns ``open``, ``high``, ``low``,
        ``close``, ``volume``, ``time``.  The ``time`` column is used
        for date extraction.  At least 2 rows required for Turnaround
        Tuesday check.

    Returns
    -------
    dict
        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-0.6),
        ``sub_signals`` dict, and ``indicators`` dict.
    """
    result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "day_of_week": "HOLD",
            "turnaround_tuesday": "HOLD",
            "month_end": "HOLD",
            "sell_in_may": "HOLD",
            "january_effect": "HOLD",
            "pre_holiday": "HOLD",
            "fomc_drift": "HOLD",
            "santa_claus_rally": "HOLD",
        },
        "indicators": {
            "day_of_week": None,
            "day_name": None,
            "is_tuesday": False,
            "prior_bar_red": False,
            "is_month_end": False,
            "days_remaining_in_month": None,
            "month": None,
            "is_weak_period": None,
            "is_pre_holiday": False,
            "is_pre_fomc": False,
            "is_fomc_day": False,
            "is_post_fomc": False,
            "days_to_fomc": None,
            "is_santa_rally": False,
        },
    }

    # ---- Validate input ----
    if df is None or not isinstance(df, pd.DataFrame):
        return result

    required_cols = {"open", "high", "low", "close", "volume", "time"}
    if not required_cols.issubset(set(df.columns)):
        return result

    if len(df) < _MIN_BARS:
        return result

    # ---- Extract date from last bar ----
    try:
        time_col = df["time"]
        if not pd.api.types.is_datetime64_any_dtype(time_col):
            time_col = pd.to_datetime(time_col)
        last_time = time_col.iloc[-1]
        if isinstance(last_time, (pd.Timestamp, datetime)):
            last_date = last_time.date()
        elif isinstance(last_time, date):
            last_date = last_time
        else:
            last_date = pd.Timestamp(last_time).date()
    except Exception:
        logger.debug("calendar: failed to extract last_date from DataFrame", exc_info=True)
        return result

    # ---- Compute each sub-signal ----
    # BUG-104: All exception handlers now log failures instead of silently swallowing.
    try:
        dow_action, dow_ind = _day_of_week_effect(last_date)
    except Exception:
        logger.debug("calendar: day_of_week_effect failed", exc_info=True)
        dow_action, dow_ind = "HOLD", {}

    try:
        tt_action, tt_ind = _turnaround_tuesday(df, last_date)
    except Exception:
        logger.debug("calendar: turnaround_tuesday failed", exc_info=True)
        tt_action, tt_ind = "HOLD", {}

    try:
        me_action, me_ind = _month_end_effect(last_date)
    except Exception:
        logger.debug("calendar: month_end_effect failed", exc_info=True)
        me_action, me_ind = "HOLD", {}

    try:
        sim_action, sim_ind = _sell_in_may(last_date)
    except Exception:
        logger.debug("calendar: sell_in_may failed", exc_info=True)
        sim_action, sim_ind = "HOLD", {}

    try:
        jan_action, jan_ind = _january_effect(last_date)
    except Exception:
        logger.debug("calendar: january_effect failed", exc_info=True)
        jan_action, jan_ind = "HOLD", {}

    try:
        ph_action, ph_ind = _pre_holiday_effect(last_date)
    except Exception:
        logger.debug("calendar: pre_holiday_effect failed", exc_info=True)
        ph_action, ph_ind = "HOLD", {}

    try:
        fomc_action, fomc_ind = _fomc_drift(last_date)
    except Exception:
        logger.debug("calendar: fomc_drift failed", exc_info=True)
        fomc_action, fomc_ind = "HOLD", {}

    try:
        santa_action, santa_ind = _santa_claus_rally(last_date)
    except Exception:
        logger.debug("calendar: santa_claus_rally failed", exc_info=True)
        santa_action, santa_ind = "HOLD", {}

    # ---- Populate sub-signals and indicators ----
    result["sub_signals"]["day_of_week"] = dow_action
    result["sub_signals"]["turnaround_tuesday"] = tt_action
    result["sub_signals"]["month_end"] = me_action
    result["sub_signals"]["sell_in_may"] = sim_action
    result["sub_signals"]["january_effect"] = jan_action
    result["sub_signals"]["pre_holiday"] = ph_action
    result["sub_signals"]["fomc_drift"] = fomc_action
    result["sub_signals"]["santa_claus_rally"] = santa_action

    result["indicators"].update(dow_ind)
    result["indicators"].update(tt_ind)
    result["indicators"].update(me_ind)
    result["indicators"].update(sim_ind)
    result["indicators"].update(jan_ind)
    result["indicators"].update(ph_ind)
    result["indicators"].update(fomc_ind)
    result["indicators"].update(santa_ind)

    # ---- Majority vote with quorum ----
    # Require at least 2 active votes in the winning direction to emit a signal.
    # Without this quorum, a single BUY sub-signal (e.g., just month_end or
    # pre_holiday) beats 0 SELL and creates a permanent BUY bias - most
    # calendar sub-signals are structurally BUY-only by design.
    _MIN_WINNING_VOTES = 2

    votes = [
        dow_action, tt_action, me_action, sim_action,
        jan_action, ph_action, fomc_action, santa_action,
    ]
    buy_count = votes.count("BUY")
    sell_count = votes.count("SELL")
    active_votes = buy_count + sell_count  # non-HOLD votes

    if active_votes == 0:
        result["action"] = "HOLD"
        result["confidence"] = 0.0
    elif buy_count > sell_count and buy_count >= _MIN_WINNING_VOTES:
        result["action"] = "BUY"
        result["confidence"] = min(round(buy_count / active_votes, 2), _MAX_CONFIDENCE)
    elif sell_count > buy_count and sell_count >= _MIN_WINNING_VOTES:
        result["action"] = "SELL"
        result["confidence"] = min(round(sell_count / active_votes, 2), _MAX_CONFIDENCE)
    else:
        # Quorum not met or tied
        result["action"] = "HOLD"
        result["confidence"] = 0.0

    return result
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"horizon|timeframe|1h|24h|3h|1d|ticker|compute_claude_fundamental_signal|compute_news_event_signal\" Q:\\finance-analyzer\\portfolio\\signals\\claude_fundamental.py Q:\\finance-analyzer\\portfolio\\signals\\news_event.py" in Q:\finance-analyzer
 succeeded in 725ms:
Q:\finance-analyzer\portfolio\signals\news_event.py:10:The ``context`` parameter is a dict with keys: ticker, config, macro.
Q:\finance-analyzer\portfolio\signals\news_event.py:52:def _persist_headlines(ticker: str, headlines: list[dict]) -> None:
Q:\finance-analyzer\portfolio\signals\news_event.py:93:            "ticker": ticker,
Q:\finance-analyzer\portfolio\signals\news_event.py:98:        logger.debug("Failed to persist headlines for %s", ticker, exc_info=True)
Q:\finance-analyzer\portfolio\signals\news_event.py:101:def _fetch_headlines(ticker: str, config: dict) -> list[dict]:
Q:\finance-analyzer\portfolio\signals\news_event.py:102:    """Fetch headlines for a ticker + sector peers.
Q:\finance-analyzer\portfolio\signals\news_event.py:104:    Fetches ticker-specific headlines, then merges sector-wide headlines
Q:\finance-analyzer\portfolio\signals\news_event.py:108:    if not ticker:
Q:\finance-analyzer\portfolio\signals\news_event.py:110:    short = ticker.upper().replace("-USD", "")
Q:\finance-analyzer\portfolio\signals\news_event.py:132:        logger.debug("Failed to fetch headlines for %s", ticker, exc_info=True)
Q:\finance-analyzer\portfolio\signals\news_event.py:134:    # Merge sector-peer headlines: fetch from a representative ticker per sector
Q:\finance-analyzer\portfolio\signals\news_event.py:151:        ticker_secs = TICKER_SECTORS.get(ticker, set())
Q:\finance-analyzer\portfolio\signals\news_event.py:152:        for sec in ticker_secs:
Q:\finance-analyzer\portfolio\signals\news_event.py:170:        logger.debug("Failed to fetch sector headlines for %s", ticker, exc_info=True)
Q:\finance-analyzer\portfolio\signals\news_event.py:347:def _sector_impact_vote(headlines: list[dict], ticker: str) -> tuple[str, dict]:
Q:\finance-analyzer\portfolio\signals\news_event.py:348:    """Keyword + sector mapping for ticker-specific directional vote.
Q:\finance-analyzer\portfolio\signals\news_event.py:351:    directional impact on this ticker's sector.
Q:\finance-analyzer\portfolio\signals\news_event.py:361:            impact = get_sector_impact(kw, ticker)
Q:\finance-analyzer\portfolio\signals\news_event.py:416:def _thesis_alignment_vote(headlines: list[dict], ticker: str, config: dict) -> tuple[str, dict]:
Q:\finance-analyzer\portfolio\signals\news_event.py:425:        Recent headlines for the ticker.
Q:\finance-analyzer\portfolio\signals\news_event.py:426:    ticker : str
Q:\finance-analyzer\portfolio\signals\news_event.py:427:        Instrument ticker.
Q:\finance-analyzer\portfolio\signals\news_event.py:445:    # Get active beliefs for this ticker
Q:\finance-analyzer\portfolio\signals\news_event.py:448:        beliefs = get_active_beliefs(ticker=ticker)
Q:\finance-analyzer\portfolio\signals\news_event.py:499:def compute_news_event_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\news_event.py:507:        Dict with keys: ticker, config, macro.
Q:\finance-analyzer\portfolio\signals\news_event.py:532:    ticker = context.get("ticker", "")
Q:\finance-analyzer\portfolio\signals\news_event.py:535:    if not ticker:
Q:\finance-analyzer\portfolio\signals\news_event.py:539:    headlines = _fetch_headlines(ticker, config)
Q:\finance-analyzer\portfolio\signals\news_event.py:544:    _persist_headlines(ticker, headlines)
Q:\finance-analyzer\portfolio\signals\news_event.py:568:        sec_action, sec_ind = _sector_impact_vote(headlines, ticker)
Q:\finance-analyzer\portfolio\signals\news_event.py:579:        thesis_action, thesis_ind = _thesis_alignment_vote(headlines, ticker, config)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:97:def _build_ticker_grid(summary):
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:98:    """Build compact ticker grid from agent_summary_compact data."""
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:100:    tickers = summary.get("signals", summary.get("tickers", {}))
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:101:    if not tickers:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:102:        logger.warning("claude_fundamental: empty ticker grid — summary has keys %s "
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:103:                        "but no 'signals' or 'tickers' data. Signal will return HOLD.",
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:106:    for ticker, data in tickers.items():
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:117:            f"{ticker}: ${price} RSI={rsi_val} MACD={macd_val} "
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:153:    """Fetch next earnings dates from yfinance for all stock tickers.
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:156:    Returns dict of {ticker: {date, eps_estimate, days_until}}.
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:165:        from portfolio.tickers import STOCK_SYMBOLS
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:170:    for ticker in STOCK_SYMBOLS:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:173:            t = yf.Ticker(ticker)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:192:                result[ticker] = {
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:198:            logger.debug("Earnings fetch failed for ticker", exc_info=True)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:206:def _build_fundamentals_block(ticker, fundamentals, tier="haiku", earnings=None):
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:207:    """Build a fundamentals data string for a ticker based on tier detail level.
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:210:        ticker: Stock ticker symbol
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:217:    fund = fundamentals.get(ticker) if fundamentals else None
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:221:    earn = (earnings or {}).get(ticker, {})
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:225:        parts = [ticker + ":"]
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:243:    lines = [f"  {ticker} Fundamentals:"]
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:310:    ticker_grid = _build_ticker_grid(summary)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:312:    # Add one-liner fundamentals for each ticker
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:316:    tickers = summary.get("signals", summary.get("tickers", {}))
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:317:    for ticker in tickers:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:318:        line = _build_fundamentals_block(ticker, fundamentals, tier="haiku", earnings=earnings)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:327:For each ticker below, give a quick fundamental directional read.
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:336:{ticker_grid}
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:355:    ticker_grid = _build_ticker_grid(summary)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:357:    # Add detailed fundamentals per ticker
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:361:    tickers = summary.get("signals", summary.get("tickers", {}))
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:362:    for ticker in tickers:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:363:        block = _build_fundamentals_block(ticker, fundamentals, tier="sonnet", earnings=earnings)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:372:For each ticker, evaluate 5 fundamental dimensions:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:385:{ticker_grid}
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:415:    ticker_grid = _build_ticker_grid(summary)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:435:    tickers = summary.get("signals", summary.get("tickers", {}))
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:436:    for ticker in tickers:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:437:        block = _build_fundamentals_block(ticker, fundamentals, tier="opus", earnings=earnings)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:450:For each ticker, evaluate 5 fundamental dimensions:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:466:{ticker_grid}
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:545:    """Parse Haiku JSON → {ticker: {"action": ..., "confidence": ...}}."""
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:548:    for ticker, val in data.items():
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:555:        results[ticker] = {
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:565:    """Parse Sonnet JSON → {ticker: {full signal result with sub_signals}}."""
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:568:    for ticker, val in data.items():
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:585:        results[ticker] = {
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:598:    """Parse Opus JSON → {ticker: {full signal result + contrarian_flag}}."""
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:601:    for ticker, val in data.items():
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:621:        results[ticker] = {
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:652:    for ticker, result in results.items():
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:662:            "ticker": ticker,
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:715:    logger.info("Claude fundamental %s refreshed: %d tickers", tier, len(results))
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:721:def _bias_rate_from_entries(entries: list, tier: str, ticker: str | None = None) -> tuple[float, int, str | None]:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:723:    scoped to one ticker) from a journal entry list.
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:733:        and (ticker is None or e.get("ticker") == ticker)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:735:    # Only check most recent 30 votes for the (tier, ticker?) slice
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:747:    """Detect BUY or SELL bias from recent journal entries (global, all tickers).
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:782:def _is_tier_biased_for_ticker(tier: str, ticker: str) -> bool:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:783:    """Per-ticker bias detection — catches one-sided per-ticker patterns
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:790:    cascade kept picking Opus's per-ticker-biased BUY, which lost as BTC
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:794:    detector firing on every active ticker (BTC/ETH/MSTR/XAG/XAU all
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:802:    vs 200) because per-(tier, ticker) volume is roughly 1/Nth of the per-tier
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:803:    volume across N tickers — too small a tail can leave a ticker with 0
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:813:        logger.debug("CF per-ticker bias detection: log load failed", exc_info=True)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:816:    bias_rate, n, label = _bias_rate_from_entries(entries, tier, ticker=ticker)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:822:            tier, ticker, bias_rate * 100, label, n,
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:828:def _get_best_result(ticker):
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:829:    """Cascade: Opus > Sonnet > Haiku. Return best available result for ticker.
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:832:    (or SELL) bias across all tickers, its non-HOLD vote is treated as HOLD
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:836:    2026-04-28: Added per-ticker bias detection AND fixed a fallback bug
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:849:        result = _cache[tier]["results"].get(ticker)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:854:            _is_tier_biased(tier) or _is_tier_biased_for_ticker(tier, ticker)
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:858:                tier, action, ticker,
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:894:def compute_claude_fundamental_signal(df: pd.DataFrame, context: dict = None) -> dict:
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:895:    """Signal entry point — called once per ticker by signal_engine.
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:899:        context: dict with keys: ticker, config, macro
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:936:    # Cascade lookup for this ticker
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:937:    ticker = context.get("ticker", "")
Q:\finance-analyzer\portfolio\signals\claude_fundamental.py:938:    result = _get_best_result(ticker)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\credit_spread.py | Select-Object -Skip 240 -First 120" in Q:\finance-analyzer
 succeeded in 781ms:
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_crisis": crisis, "oas_level": round(current, 2)}


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_credit_spread_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute credit spread risk appetite signal.

    Args:
        df: OHLCV DataFrame (unused - credit spread data fetched from FRED).
        context: dict with keys {ticker, config, asset_class, regime}.

    Returns:
        dict with keys: action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))

    if not ticker:
        return empty

    # All tickers can use this signal
    if ticker not in _SAFE_HAVEN and ticker not in _RISK_ASSETS:
        return empty

    safe_haven = _is_safe_haven(ticker)

    # Get FRED API key from config
    fred_key = _get_fred_key(context)

    # Also try loading from config.json directly as fallback
    if not fred_key:
        try:
            cfg = load_json("config.json", default={}) or {}
            fred_key = cfg.get("golddigger", {}).get("fred_api_key", "") or ""
        except Exception:
            logger.debug("config.json fallback read failed", exc_info=True)

    if not fred_key:
        logger.debug("No FRED API key available for credit spread signal")
        return empty

    # Fetch HY OAS data
    values = _fetch_hy_oas(fred_key)
    if not values or len(values) < 20:
        return empty

    # Compute sub-indicators
    zscore_action, zscore_ind = _oas_zscore_signal(values, safe_haven)
    mom_action, mom_ind = _oas_momentum_signal(values, safe_haven)
    accel_action, accel_ind = _oas_acceleration_signal(values, safe_haven)
    crisis_action, crisis_ind = _crisis_level_signal(values, safe_haven)

    votes = [zscore_action, mom_action, accel_action, crisis_action]
    action, confidence = majority_vote(votes, count_hold=False)

    # Merge all indicators
    indicators = {}
    indicators.update(zscore_ind)
    indicators.update(mom_ind)
    indicators.update(accel_ind)
    indicators.update(crisis_ind)
    indicators["safe_haven_mode"] = safe_haven

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "oas_zscore": zscore_action,
            "oas_momentum": mom_action,
            "oas_acceleration": accel_action,
            "crisis_level": crisis_action,
        },
        "indicators": indicators,
    }
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I have enough context to pin line-accurate findings now. I’m collecting exact anchors for the fixed-horizon modules and the timezone/import-time issues so the final review stays concrete.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"compute_forecast_signal|kronos_1h|chronos_1h|chronos_24h|_load_candles_ohlcv|_run_chronos|_run_kronos|_init_kronos_enabled\\(|config.json\" Q:\\finance-analyzer\\portfolio\\signals\\forecast.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_DXY_1H_THRESHOLD_PCT|change_1h_pct|change_3h_pct|compute_dxy_cross_asset_signal|1-3h horizons|1h DXY\" Q:\\finance-analyzer\\portfolio\\signals\\dxy_cross_asset.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"Horizon realignment|change_3h_pct|change_5d_pct|using_intraday|compute_metals_cross_asset_signal|daily-cadence|threshold set based on data cadence|EPU|TIPS\" Q:\\finance-analyzer\\portfolio\\signals\\metals_cross_asset.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_get_current_date|replace\\(tzinfo=UTC\\)|next_event|events_within_hours|recent_high_impact_events|compute_econ_calendar_signal\" Q:\\finance-analyzer\\portfolio\\signals\\econ_calendar.py" in Q:\finance-analyzer
 succeeded in 769ms:
4:with silver/gold on 1-3h horizons — arguably the single strongest
28:# 1h DXY move threshold. Typical 1h DXY range is ~0.05-0.10%; 0.15% is a
30:_DXY_1H_THRESHOLD_PCT = 0.15
37:def compute_dxy_cross_asset_signal(
71:    change_1h = data.get("change_1h_pct")
75:    if change_1h < -_DXY_1H_THRESHOLD_PCT:
79:    elif change_1h > _DXY_1H_THRESHOLD_PCT:
93:            "dxy_change_1h_pct": round(change_1h, 3),
94:            "dxy_change_3h_pct": (
95:                round(data["change_3h_pct"], 3)
96:                if data.get("change_3h_pct") is not None else None
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 778ms:
10:    7. EPU (Economic Policy Uncertainty): high uncertainty -> safe haven BUY
11:    8. TIPS Real Yield direction: falling real yields -> BUY metals
16:2026-04-13: Horizon realignment after live measurement showed 29.1% on
24:2026-04-26: Added EPU + TIPS real yield from FRED API as sub-signals #7-8.
25:EPU improves gold RMSE by ~18% (Baker/Bloom/Davis 2016). TIPS real yield
28:Both are daily-cadence indicators like GVZ.
68:# EPU: Economic Policy Uncertainty (Baker/Bloom/Davis daily news index).
70:_EPU_ZSCORE_HIGH = 1.5
71:_EPU_ZSCORE_LOW = -1.0
72:_EPU_FRED_SERIES = "USEPUINDXD"
74:# TIPS real yield (10Y TIPS, FRED series DFII10).
77:_TIPS_CHANGE_THRESHOLD = 0.10
78:_TIPS_FRED_SERIES = "DFII10"
183:    Returns a dict with per-source payloads AND a ``_using_intraday`` flag
219:    result: dict = {"_using_intraday": use_intraday}
240:        result["copper_change_pct"] = copper["change_3h_pct"] if copper else 0.0
241:        result["gs_velocity_pct"] = gs["change_3h_pct"] if gs else 0.0
242:        result["spy_change_pct"] = spy["change_3h_pct"] if spy else 0.0
243:        result["oil_change_pct"] = oil["change_3h_pct"] if oil else 0.0
255:        result["copper_change_pct"] = copper["change_5d_pct"] if copper else 0.0
256:        result["gs_velocity_pct"] = gs_daily["change_5d_pct"] if gs_daily else 0.0
258:        result["oil_change_pct"] = oil["change_5d_pct"] if oil else 0.0
266:def compute_metals_cross_asset_signal(
291:    using_intraday = ctx.get("_using_intraday", False)
293:    # Select threshold set based on data cadence
294:    if using_intraday:
311:            fallback = oil_ctx.get("change_1d_pct") or oil_ctx.get("change_5d_pct") or 0.0
404:    # Sub 7: EPU (Economic Policy Uncertainty) — daily FRED
408:    epu_values = _fetch_fred_values(_EPU_FRED_SERIES, fred_key, _epu_cache)
410:    if epu_zscore > _EPU_ZSCORE_HIGH:
412:    elif epu_zscore < _EPU_ZSCORE_LOW:
418:    # Sub 8: TIPS Real Yield direction — daily FRED (DFII10)
421:    tips_values = _fetch_fred_values(_TIPS_FRED_SERIES, fred_key, _tips_cache)
429:    if tips_change < -_TIPS_CHANGE_THRESHOLD:
431:    elif tips_change > _TIPS_CHANGE_THRESHOLD:
452:            "using_intraday": using_intraday,
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 787ms:
26:    events_within_hours,
27:    next_event,
28:    recent_high_impact_events,
39:def _get_current_date(df: pd.DataFrame) -> datetime:
44:            return last_time.to_pydatetime().replace(tzinfo=UTC)
55:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
56:    indicators = {"next_event": None, "hours_until": None}
61:    indicators["next_event"] = f"{evt['type']} {evt['date'].isoformat()}"
78:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
102:    nearby = events_within_hours(4, ref_date.date() if isinstance(ref_date, datetime) else ref_date)
127:    recent = recent_high_impact_events(24)
136:        evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
141:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
144:        indicators["next_event_hours"] = evt["hours_until"]
156:    evt = next_event(ref_date.date() if isinstance(ref_date, datetime) else ref_date)
176:def compute_econ_calendar_signal(df: pd.DataFrame, context: dict = None) -> dict:
208:    ref_date = _get_current_date(df)
214:        if prox_ind.get("next_event") is None:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 836ms:
5:  1. kronos_1h  — Kronos 1-hour prediction
7:  3. chronos_1h — Chronos 1-hour prediction
8:  4. chronos_24h — Chronos 24-hour prediction
76:def _init_kronos_enabled():
77:    """Read kronos_enabled from config.json at import time.
90:            str(Path(__file__).resolve().parent.parent.parent / "config.json"),
104:_init_kronos_enabled()
248:def _load_candles_ohlcv(ticker: str, periods: int = 168,
313:def _run_kronos(candles: list[dict], horizons: tuple = (1, 24), _ticker: str = "") -> dict | None:
325:            return _run_kronos_inner(candles, horizons, _ticker, t0)
334:def _run_kronos_inner(candles, horizons, _ticker, t0):
340:            cfg = load_json(str(Path(__file__).resolve().parent.parent.parent / "config.json"), {})
394:def _run_chronos(prices: list[float], horizons: tuple = (1, 24), _ticker: str = "",
404:        return _run_chronos_inner(prices, horizons, _ticker, timeout)
407:def _run_chronos_inner(prices, horizons, _ticker, timeout):
465:        alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
466:        alive_votes.append(sub_signals.get("kronos_1h", "HOLD"))
470:        alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
471:        alive_votes.append(sub_signals.get("chronos_1h", "HOLD"))
472:        alive_votes.append(sub_signals.get("chronos_24h", "HOLD"))
733:def compute_forecast_signal(df: pd.DataFrame, context: dict = None) -> dict:
752:            "kronos_1h": "HOLD",
754:            "chronos_1h": "HOLD",
755:            "chronos_24h": "HOLD",
782:    candles = _cached(cache_key, _FORECAST_TTL, _load_candles_ohlcv, ticker)
790:                                  _load_candles_ohlcv, ticker, kronos_periods,
848:    chronos = _cached(chronos_key, _FORECAST_TTL, _run_chronos, close_prices, (1, 24), ticker)
854:            result["sub_signals"]["chronos_1h"] = chronos["1h"].get("action", "HOLD")
855:            result["indicators"]["chronos_1h_pct"] = chronos["1h"].get("pct_move", 0)
856:            result["indicators"]["chronos_1h_conf"] = chronos["1h"].get("confidence", 0)
859:            result["sub_signals"]["chronos_24h"] = chronos["24h"].get("action", "HOLD")
860:            result["indicators"]["chronos_24h_pct"] = chronos["24h"].get("pct_move", 0)
861:            result["indicators"]["chronos_24h_conf"] = chronos["24h"].get("confidence", 0)
867:    kronos = _cached(kronos_key, _FORECAST_TTL, _run_kronos, kronos_input, (1, 24), ticker)
888:            result["sub_signals"]["kronos_1h"] = k1h_action
889:            result["indicators"]["kronos_1h_raw"] = k1h_action
890:            result["indicators"]["kronos_1h_pct"] = kr["1h"].get("pct_move", 0)
891:            result["indicators"]["kronos_1h_conf"] = kr["1h"].get("confidence", 0)
892:            result["indicators"]["kronos_1h_range_pct"] = kr["1h"].get("predicted_range_pct", 0)
893:            result["indicators"]["kronos_1h_range_skew"] = kr["1h"].get("range_skew", 0)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"precomputed|deep context files|_load_deep_context|gold_deep_context|cot_history|real_yield_direction|compute_cot_positioning_signal|cot_positioning\\.live|report_date\" Q:\\finance-analyzer\\portfolio\\signals\\cot_positioning.py" in Q:\finance-analyzer
 succeeded in 682ms:
10:Applicable to XAU-USD and XAG-USD only.  Data sourced from precomputed
11:deep context files (metals_precompute.py) with CFTC API fallback.
55:def _load_deep_context(ticker: str) -> dict | None:
56:    """Load precomputed deep context for the given metal.
75:def _load_cot_history(metal: str) -> list[dict]:
79:    `"data/cot_history.jsonl"`.
83:    entries = load_jsonl(str(_DATA_DIR / "cot_history.jsonl"))
99:            f"&$order=report_date_as_yyyy_mm_dd DESC"
115:                    "date": r.get("report_date_as_yyyy_mm_dd", ""),
277:    indicators = {"real_yield": None, "real_yield_direction": None}
294:    direction = fred.get("real_yield_direction")
299:        indicators["real_yield_direction"] = direction
309:def compute_cot_positioning_signal(
341:    # Load precomputed deep context
342:    deep_ctx = _load_deep_context(ticker)
345:    # metals_precompute stores COT at external_research.cot_positioning.live
359:    local_history = _load_cot_history(metal)
399:    all_indicators["report_date"] = cot_data.get("report_date")
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"except ImportError|logger\\.debug\\(\" Q:\\finance-analyzer\\portfolio\\signals\\orderbook_flow.py Q:\\finance-analyzer\\portfolio\\signals\\dxy_cross_asset.py Q:\\finance-analyzer\\portfolio\\signals\\metals_cross_asset.py" in Q:\finance-analyzer
 succeeded in 709ms:
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:63:    except ImportError:
Q:\finance-analyzer\portfolio\signals\dxy_cross_asset.py:64:        logger.debug("macro_context.get_dxy_intraday not available")
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:118:        logger.debug("No FRED API key — cannot fetch %s", series_id)
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:123:    except ImportError:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:124:        logger.debug("http_retry not available for FRED fetch")
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:155:            logger.debug(
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:199:    except ImportError:
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:200:        logger.debug("metals_cross_assets module not available")
Q:\finance-analyzer\portfolio\signals\metals_cross_asset.py:245:        logger.debug(
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:46:    except ImportError:
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:47:        logger.debug("Microstructure modules not available")
Q:\finance-analyzer\portfolio\signals\orderbook_flow.py:76:    except ImportError:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\orderbook_flow.py | Select-Object -Skip 60 -First 25" in Q:\finance-analyzer
 succeeded in 692ms:
        return None

    # Read accumulated OFI, OFI z-score, multiscale, and spread z-score
    ofi = 0.0
    ofi_zscore = 0.0
    sz = 0.0
    flow_acceleration = 0.0
    try:
        from portfolio.microstructure_state import load_persisted_state
        ms_state = load_persisted_state(ticker)
        if ms_state:
            ofi = ms_state.get("ofi", 0.0)
            ofi_zscore = ms_state.get("ofi_zscore", 0.0)
            sz = ms_state.get("spread_zscore", 0.0)
            flow_acceleration = ms_state.get("flow_acceleration", 0.0)
    except ImportError:
        pass

    return {
        "depth_imbalance": di,
        "trade_imbalance_ratio": tfi["imbalance_ratio"],
        "vpin": vpin if vpin is not None else 0.0,
        "ofi": ofi,
        "ofi_zscore": ofi_zscore,
        "spread_zscore": sz,
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"get_all_ticker_accuracies\\(horizon=\\\"24h\\\"|compute_forecast_accuracy\\(|horizon=\\\"1h\\\"|horizon=\\\"24h\\\"\" Q:\\finance-analyzer\\portfolio\\signals\\forecast.py" in Q:\finance-analyzer
2026-05-11T15:34:32.131683Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:get_all_ticker_accuracies\(horizon=\)
    ^
error: unclosed group

 exited 1 in 703ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:get_all_ticker_accuracies\(horizon=\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"get_all_ticker_accuracies|compute_forecast_accuracy\" Q:\\finance-analyzer\\portfolio\\signals\\forecast.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"horizon=\\\"24h\\\"|horizon=\\\"1h\\\"\" Q:\\finance-analyzer\\portfolio\\signals\\forecast.py" in Q:\finance-analyzer
 succeeded in 759ms:
542:            from portfolio.forecast_accuracy import get_all_ticker_accuracies
543:            return get_all_ticker_accuracies(horizon="24h", days=14)
558:            from portfolio.forecast_accuracy import compute_forecast_accuracy
561:                "1h": compute_forecast_accuracy(
564:                "24h": compute_forecast_accuracy(
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 exited 1 in 735ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:horizon=\)
    ^
error: unclosed group

2026-05-11T15:34:37.433185Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:horizon=\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\forecast.py | Select-Object -Skip 536 -First 35" in Q:\finance-analyzer
 succeeded in 685ms:
    """
    ttl = cache_ttl or _ACCURACY_CACHE_TTL

    def _fetch():
        try:
            from portfolio.forecast_accuracy import get_all_ticker_accuracies
            return get_all_ticker_accuracies(horizon="24h", days=14)
        except Exception as e:
            logger.debug("Failed to load forecast accuracy: %s", e)
            return {}

    return _cached("forecast_ticker_accuracy", ttl, _fetch)


def _load_forecast_subsignal_accuracy(cache_ttl=None, days=30):
    """Load raw sub-signal accuracy for 1h and 24h forecast votes."""
    ttl = cache_ttl or _ACCURACY_CACHE_TTL
    cache_key = f"forecast_subsignal_accuracy_{days}"

    def _fetch():
        try:
            from portfolio.forecast_accuracy import compute_forecast_accuracy

            return {
                "1h": compute_forecast_accuracy(
                    horizon="1h", days=days, use_raw_sub_signals=True
                ),
                "24h": compute_forecast_accuracy(
                    horizon="24h", days=days, use_raw_sub_signals=True
                ),
            }
        except Exception as e:
            logger.debug("Failed to load forecast sub-signal accuracy: %s", e)
            return {}

[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"Sell in May|May through October|is_weak_period|return \\\"HOLD\\\", indicators|return \\\"BUY\\\", indicators\" Q:\\finance-analyzer\\portfolio\\signals\\calendar_seasonal.py" in Q:\finance-analyzer
2026-05-11T15:34:53.343995Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:Sell in May|May through October|is_weak_period|return \)
    ^
error: unclosed group

 exited 1 in 674ms:
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:Sell in May|May through October|is_weak_period|return \)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"Sell in May|May through October|is_weak_period\" Q:\\finance-analyzer\\portfolio\\signals\\calendar_seasonal.py" in Q:\finance-analyzer
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\calendar_seasonal.py | Select-Object -Skip 120 -First 30" in Q:\finance-analyzer
 succeeded in 819ms:
    tuple[str, dict]
        Vote and indicators.
    """
    days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
    days_remaining = days_in_month - last_date.day
    is_month_end = days_remaining < 3  # last 3 calendar days (day 29, 30, 31 of a 31-day month)
    indicators = {
        "is_month_end": is_month_end,
        "days_remaining_in_month": days_remaining,
    }

    if is_month_end:
        return "BUY", indicators
    return "HOLD", indicators


def _sell_in_may(last_date: date) -> tuple[str, dict]:
    """Sell in May and go away / Halloween indicator.

    May through October = historically weaker (SELL bias).
    Only historically *strong* months vote BUY: Nov, Dec, Jan, Apr.
    Transitional months (Feb, Mar) = HOLD - they are not consistently
    strong enough to justify a standing BUY vote.

    Parameters
    ----------
    last_date : date
        Date of the last bar.

    Returns
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 820ms:
7:  4. Sell in May / Halloween   (May-Oct=SELL, Nov-Apr=BUY)
138:    """Sell in May and go away / Halloween indicator.
140:    May through October = historically weaker (SELL bias).
156:    is_weak_period = 5 <= month <= 10
158:    indicators = {"month": month, "is_weak_period": is_weak_period}
160:    if is_weak_period:
357:            "is_weak_period": None,
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\calendar_seasonal.py | Select-Object -Skip 150 -First 18" in Q:\finance-analyzer
 succeeded in 820ms:
    -------
    tuple[str, dict]
        Vote and indicators.
    """
    month = last_date.month
    is_weak_period = 5 <= month <= 10
    is_strong_month = month in (1, 4, 11, 12)  # Jan, Apr, Nov, Dec
    indicators = {"month": month, "is_weak_period": is_weak_period}

    if is_weak_period:
        return "HOLD", indicators
    if is_strong_month:
        return "BUY", indicators
    return "HOLD", indicators


def _january_effect(last_date: date) -> tuple[str, dict]:
    """January historically bullish for small caps; December tax-loss selling.
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content Q:\\finance-analyzer\\portfolio\\signals\\forecast.py | Select-Object -Skip 240 -First 40" in Q:\finance-analyzer
 succeeded in 769ms:
                _kronos_tripped_until = 0.0
                logger.info("Kronos circuit breaker RESET on successful %s", ticker)
            elif model == "chronos" and _chronos_tripped_until > 0:
                _chronos_tripped_until = 0.0
                logger.info("Chronos circuit breaker RESET on successful %s", ticker)


def _load_candles_ohlcv(ticker: str, periods: int = 168,
                        interval: str = "1h") -> list[dict] | None:
    """Load recent OHLCV candles as list of dicts.

    Args:
        ticker: Instrument ticker (e.g., "BTC-USD")
        periods: Number of candles to fetch
        interval: Candle interval ("1h", "5m", "15m", etc.)
    """
    from portfolio.tickers import SYMBOLS

    source_info = SYMBOLS.get(ticker, {})

    # Determine the data source - needed to apply source-specific interval constraints
    if "binance" in source_info:
        source = "binance"
    elif "binance_fapi" in source_info:
        source = "binance_fapi"
    elif "alpaca" in source_info:
        source = "alpaca"
    else:
        source = None

    # Alpaca minimum supported interval is 15m - fall back if configured interval is smaller.
    # alpaca_klines() does its own mapping; pass the raw internal interval directly.
    if source == "alpaca" and interval in ("1m", "3m", "5m"):
        logger.debug(
            "Alpaca does not support %s interval for %s - falling back to 15m", interval, ticker
        )
        interval = "15m"

    try:
        if source == "binance":
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"load_persisted_state|ofi_zscore|spread_zscore|except ImportError\" Q:\\finance-analyzer\\portfolio\\signals\\orderbook_flow.py" in Q:\finance-analyzer
 succeeded in 666ms:
46:    except ImportError:
65:    ofi_zscore = 0.0
69:        from portfolio.microstructure_state import load_persisted_state
70:        ms_state = load_persisted_state(ticker)
73:            ofi_zscore = ms_state.get("ofi_zscore", 0.0)
74:            sz = ms_state.get("spread_zscore", 0.0)
76:    except ImportError:
84:        "ofi_zscore": ofi_zscore,
85:        "spread_zscore": sz,
153:    ofi_z = ctx.get("ofi_zscore", 0.0)
173:    # applied below when spread_zscore > threshold.
174:    sz = ctx.get("spread_zscore", 0.0)
209:            "ofi_zscore": round(ctx.get("ofi_zscore", 0.0), 4),
211:            "spread_zscore": round(sz, 4),
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"Applied to all assets|compute_vix_term_structure_signal|context: Optional dict|data = _cached\\(|action, confidence = majority_vote|ticker\" Q:\\finance-analyzer\\portfolio\\signals\\vix_term_structure.py" in Q:\finance-analyzer
 succeeded in 630ms:
10:to all assets; per-ticker accuracy gating auto-disables for assets
141:def compute_vix_term_structure_signal(df=None, context=None, **kwargs) -> dict:
146:        context: Optional dict with ``ticker`` key.
158:    data = _cached("vix_term_structure", _CACHE_TTL, _fetch_vix_data)
171:    action, confidence = majority_vote(votes, count_hold=False)
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
# Codex adversarial review: signals-modules
## Summary
Worst offenders are `forecast`, `metals_cross_asset`, `dxy_cross_asset`, `econ_calendar`, and `cot_positioning`.

I did not find a clean future-bar read like `shift(-1)` or a centered rolling window in the active set. The bigger failures are fixed-horizon detectors being reused as a single global vote, one timezone-conversion bug in event gating, and one active detector consuming precomputed context/signal data.

## P0 — Blockers
- `portfolio/signals/forecast.py:248,543,558,733,848,867` — Why it bites: this detector is not horizon-aware at all. It loads candles with default `interval="1h"`, only requests model horizons `(1, 24)`, and even loads accuracy stats hardcoded to `24h` plus sub-signal stats only for `1h`/`24h`. The same composite is therefore reused for `3h`, `4h`, `12h`, and `1d`, which is exactly the “single global verdict” bug you called out. Fix: pass the requested horizon into the module and make candle interval, model horizon set, and accuracy cache keys depend on it; otherwise force `HOLD` outside explicitly supported horizons.
- `portfolio/signals/dxy_cross_asset.py:4,30,37,71,75,79` — Why it bites: the module is explicitly documented as a `1-3h` detector, then hardcodes a `1h` DXY move threshold and votes only from `change_1h_pct`. Because it never reads horizon, that 1-hour macro shock is reused unchanged in longer-horizon composites. Fix: accept horizon/context and gate to short horizons only, or compute separate `3h`/`1d` thresholds from matching DXY windows.
- `portfolio/signals/metals_cross_asset.py:16,240,255,266,291,404,418` — Why it bites: this module was “realigned” for `3h`, uses `change_3h_pct` when intraday data is healthy, falls back to `1d/5d` changes otherwise, and mixes those with daily EPU/TIPS votes. It still never reads requested horizon, so one mixed-cadence verdict is reused across every engine horizon. Fix: make horizon an explicit input and build separate feature sets/thresholds per horizon; do not let “data availability” decide horizon semantics.

## P1 — High
- `portfolio/signals/econ_calendar.py:39-44` — Why it bites: `_get_current_date()` does `last_time.to_pydatetime().replace(tzinfo=UTC)`. For any tz-aware non-UTC timestamp, that relabels the wall clock as UTC instead of converting it, shifting `hours_until` and every event gate by the timezone offset. Fix: use `astimezone(UTC)` / `tz_convert("UTC")` for aware timestamps, and only assign UTC to naive values if that is an explicit policy.
- `portfolio/signals/cot_positioning.py:10-11,55-83,294,341-359` — Why it bites: this active detector reads `*_deep_context.json` and `cot_history.jsonl`, then votes off precomputed fields like `external_research.cot_positioning.live` and `real_yield_direction`. That violates the “no precomputed signal data” rule and gives you stale BUY/SELL votes whenever those snapshots lag. Fix: fetch raw CFTC/FRED data here, compute the directional fields locally, and add an age/staleness gate before emitting anything non-HOLD.
- `portfolio/signals/forecast.py:76-104` — Why it bites: `_init_kronos_enabled()` reads `config.json` at import time and mutates module globals. That is import-time filesystem I/O, makes tests/order-of-import matter, and freezes `_KRONOS_ENABLED` until process restart. Fix: move the config read into call-time initialization or an explicit reload path.

## P2 — Medium
- `portfolio/signals/calendar_seasonal.py:7,140,156,160` — Why it bites: the module header and `_sell_in_may()` docstring both say “May-Oct = SELL”, but the implementation returns `HOLD` for the entire weak period. That is a live logic/doc mismatch, not a style nit. Fix: either restore the SELL vote or update the documented strategy and sub-signal name so backtest/live expectations match.

## P3 — Low
- `portfolio/signals/orderbook_flow.py:65-76` — Why it bites: if `portfolio.microstructure_state` is missing, the module silently zeros `ofi`, `ofi_zscore`, `spread_zscore`, and `flow_acceleration` and keeps voting. That masks a missing dependency as neutral market structure instead of unavailable data. Fix: log it and abstain the affected sub-signals.
- `portfolio/signals/vix_term_structure.py:10,141,146,158` — Why it bites: this disabled-pending detector accepts `context` but never reads ticker or horizon, so it emits the same VIX vote for every asset even though the doc already admits it underperforms on BTC/equities. Fix: restrict applicability or key thresholds by ticker/horizon before re-enabling.

## Tests missing
- Add horizon-regression tests for `forecast`, `dxy_cross_asset`, and `metals_cross_asset`: same ticker on `3h` vs `1d` must not blindly reuse the same feature windows/vote.
- Add a tz-aware timestamp test for `econ_calendar` using a non-UTC `pd.Timestamp`; assert `hours_until` is identical after proper UTC conversion.
- Add a freshness/staleness test for `cot_positioning` that forces old `deep_context` / `cot_history` snapshots and expects `HOLD`.
- Add an import-safety test for `forecast` that imports the module in an environment without repo-root `config.json`; import should not perform filesystem I/O or freeze runtime config state.
- Add missing-dependency tests for `orderbook_flow` and other optional-dep modules so absence produces an explicit warning and deterministic abstention, not silent zero-filled features.
tokens used
182,618
# Codex adversarial review: signals-modules
## Summary
Worst offenders are `forecast`, `metals_cross_asset`, `dxy_cross_asset`, `econ_calendar`, and `cot_positioning`.

I did not find a clean future-bar read like `shift(-1)` or a centered rolling window in the active set. The bigger failures are fixed-horizon detectors being reused as a single global vote, one timezone-conversion bug in event gating, and one active detector consuming precomputed context/signal data.

## P0 — Blockers
- `portfolio/signals/forecast.py:248,543,558,733,848,867` — Why it bites: this detector is not horizon-aware at all. It loads candles with default `interval="1h"`, only requests model horizons `(1, 24)`, and even loads accuracy stats hardcoded to `24h` plus sub-signal stats only for `1h`/`24h`. The same composite is therefore reused for `3h`, `4h`, `12h`, and `1d`, which is exactly the “single global verdict” bug you called out. Fix: pass the requested horizon into the module and make candle interval, model horizon set, and accuracy cache keys depend on it; otherwise force `HOLD` outside explicitly supported horizons.
- `portfolio/signals/dxy_cross_asset.py:4,30,37,71,75,79` — Why it bites: the module is explicitly documented as a `1-3h` detector, then hardcodes a `1h` DXY move threshold and votes only from `change_1h_pct`. Because it never reads horizon, that 1-hour macro shock is reused unchanged in longer-horizon composites. Fix: accept horizon/context and gate to short horizons only, or compute separate `3h`/`1d` thresholds from matching DXY windows.
- `portfolio/signals/metals_cross_asset.py:16,240,255,266,291,404,418` — Why it bites: this module was “realigned” for `3h`, uses `change_3h_pct` when intraday data is healthy, falls back to `1d/5d` changes otherwise, and mixes those with daily EPU/TIPS votes. It still never reads requested horizon, so one mixed-cadence verdict is reused across every engine horizon. Fix: make horizon an explicit input and build separate feature sets/thresholds per horizon; do not let “data availability” decide horizon semantics.

## P1 — High
- `portfolio/signals/econ_calendar.py:39-44` — Why it bites: `_get_current_date()` does `last_time.to_pydatetime().replace(tzinfo=UTC)`. For any tz-aware non-UTC timestamp, that relabels the wall clock as UTC instead of converting it, shifting `hours_until` and every event gate by the timezone offset. Fix: use `astimezone(UTC)` / `tz_convert("UTC")` for aware timestamps, and only assign UTC to naive values if that is an explicit policy.
- `portfolio/signals/cot_positioning.py:10-11,55-83,294,341-359` — Why it bites: this active detector reads `*_deep_context.json` and `cot_history.jsonl`, then votes off precomputed fields like `external_research.cot_positioning.live` and `real_yield_direction`. That violates the “no precomputed signal data” rule and gives you stale BUY/SELL votes whenever those snapshots lag. Fix: fetch raw CFTC/FRED data here, compute the directional fields locally, and add an age/staleness gate before emitting anything non-HOLD.
- `portfolio/signals/forecast.py:76-104` — Why it bites: `_init_kronos_enabled()` reads `config.json` at import time and mutates module globals. That is import-time filesystem I/O, makes tests/order-of-import matter, and freezes `_KRONOS_ENABLED` until process restart. Fix: move the config read into call-time initialization or an explicit reload path.

## P2 — Medium
- `portfolio/signals/calendar_seasonal.py:7,140,156,160` — Why it bites: the module header and `_sell_in_may()` docstring both say “May-Oct = SELL”, but the implementation returns `HOLD` for the entire weak period. That is a live logic/doc mismatch, not a style nit. Fix: either restore the SELL vote or update the documented strategy and sub-signal name so backtest/live expectations match.

## P3 — Low
- `portfolio/signals/orderbook_flow.py:65-76` — Why it bites: if `portfolio.microstructure_state` is missing, the module silently zeros `ofi`, `ofi_zscore`, `spread_zscore`, and `flow_acceleration` and keeps voting. That masks a missing dependency as neutral market structure instead of unavailable data. Fix: log it and abstain the affected sub-signals.
- `portfolio/signals/vix_term_structure.py:10,141,146,158` — Why it bites: this disabled-pending detector accepts `context` but never reads ticker or horizon, so it emits the same VIX vote for every asset even though the doc already admits it underperforms on BTC/equities. Fix: restrict applicability or key thresholds by ticker/horizon before re-enabling.

## Tests missing
- Add horizon-regression tests for `forecast`, `dxy_cross_asset`, and `metals_cross_asset`: same ticker on `3h` vs `1d` must not blindly reuse the same feature windows/vote.
- Add a tz-aware timestamp test for `econ_calendar` using a non-UTC `pd.Timestamp`; assert `hours_until` is identical after proper UTC conversion.
- Add a freshness/staleness test for `cot_positioning` that forces old `deep_context` / `cot_history` snapshots and expects `HOLD`.
- Add an import-safety test for `forecast` that imports the module in an environment without repo-root `config.json`; import should not perform filesystem I/O or freeze runtime config state.
- Add missing-dependency tests for `orderbook_flow` and other optional-dep modules so absence produces an explicit warning and deterministic abstention, not silent zero-filled features.
