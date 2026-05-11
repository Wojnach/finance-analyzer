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
session id: 019e17a7-2eb1-7391-bd78-a350c841cbbd
--------
user
You are doing an ADVERSARIAL code review of the portfolio-risk subsystem of a quantitative trading system at Q:\finance-analyzer. Sandbox: read-only.

In-scope files (read these and ONLY these):
- portfolio/portfolio_mgr.py
- portfolio/portfolio_validator.py
- portfolio/trade_guards.py
- portfolio/trade_validation.py
- portfolio/trade_risk_classifier.py
- portfolio/risk_management.py
- portfolio/monte_carlo.py
- portfolio/monte_carlo_risk.py
- portfolio/equity_curve.py
- portfolio/exit_optimizer.py
- portfolio/kelly_sizing.py
- portfolio/kelly_metals.py
- portfolio/exposure_coach.py
- portfolio/warrant_portfolio.py
- portfolio/cost_model.py
- portfolio/instrument_profile.py
- portfolio/stats.py
- portfolio/strategies/  (recurse)

Project rules:
- Patient and Bold portfolios start at 500K SEK each. Atomic I/O via file_utils only.
- Stop-loss API must use /_api/trading/stoploss/new, NOT the regular order endpoint (Mar 3 incident: regular order endpoint causes instant fill).
- User accepts 10-20% knockout risk on 5x certs; only de-risk at 50%+ (per memory/feedback_risk_tolerance.md).
- NEVER stop-loss within 3% of MINI warrant barrier (per memory/feedback_mini_stoploss.md).
- Min order size 1000 SEK (per memory/feedback_min_order_size_1000_sek.md).
- Drawdown circuit breaker / ATR stops / concentration limits live here.
- Live prices first — never use cached/precomputed for risk math (per memory/feedback_live_prices_first.md).

Adversarial focus:
1. Position-sizing bugs: Kelly half/quarter not honored, fixed budget overrides volatility scaling, missing min-order check.
2. Stop-loss math: distance to barrier on MINI warrants not verified, ATR-based stops with wrong horizon.
3. Cost model: spread/fee missing, FX conversion missing for SEK/USD positions, double-counting fees.
4. Equity curve: round-trip P&L wrong on partial fills, NaN propagation from missing closes.
5. Monte Carlo / VaR: look-ahead bias, sample size guard missing, IID assumption on correlated assets.
6. Concurrency: portfolio state race when buy and sell run in parallel, atomic write but non-atomic read-modify-write pair.
7. Drawdown circuit breaker stuck-closed or stuck-open; reset path missing.
8. Trade guards / cooldowns: bypassable through reload, monotonic clock not used.
9. Warrant leverage/strike math: barrier knockout not detected, gearing mis-computed.
10. Risk classifier: silent fallback to permissive class on missing field.

Output format — strict markdown:
# Codex adversarial review: portfolio-risk
## Summary
## P0 — Blockers
- path:line — Why it bites: ... Fix: ...
## P1 — High
## P2 — Medium
## P3 — Low
## Tests missing

Be concrete. Cite line numbers. No false-positive padding.
codex
Reviewing the in-scope `portfolio` risk files under `Q:\finance-analyzer` and checking each against the failure modes you listed. I’ll read only those paths, extract line-anchored evidence, then rank real findings by severity with concrete fixes.
  → Enumerate in-scope files under Q:\finance-analyzer\portfolio and strategies
  • Read risk, sizing, stop-loss, cost, equity, Monte Carlo, and guard logic for concrete bugs
  • Draft adversarial review with severity-ranked findings and missing tests
codex
I’m enumerating the exact review surface first so the rest of the pass stays inside the allowed files. After that I’ll inspect the state-management and sizing paths before moving to stop-loss and Monte Carlo code.
2026-05-11T15:28:35.686679Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files Q:\\finance-analyzer\\portfolio\\strategies; @(
'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py',
'Q:\\finance-analyzer\\portfolio\\portfolio_validator.py',
'Q:\\finance-analyzer\\portfolio\\trade_guards.py',
'Q:\\finance-analyzer\\portfolio\\trade_validation.py',
'Q:\\finance-analyzer\\portfolio\\trade_risk_classifier.py',
'Q:\\finance-analyzer\\portfolio\\risk_management.py',
'Q:\\finance-analyzer\\portfolio\\monte_carlo.py',
'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py',
'Q:\\finance-analyzer\\portfolio\\equity_curve.py',
'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py',
'Q:\\finance-analyzer\\portfolio\\kelly_sizing.py',
'Q:\\finance-analyzer\\portfolio\\kelly_metals.py',
'Q:\\finance-analyzer\\portfolio\\exposure_coach.py',
'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py',
'Q:\\finance-analyzer\\portfolio\\cost_model.py',
'Q:\\finance-analyzer\\portfolio\\instrument_profile.py',
'Q:\\finance-analyzer\\portfolio\\stats.py'
) | ForEach-Object { if (Test-Path "'$_) { $_ } }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files Q:\\finance-analyzer\\portfolio\\strategies; @(
'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py',
'Q:\\finance-analyzer\\portfolio\\portfolio_validator.py',
'Q:\\finance-analyzer\\portfolio\\trade_guards.py',
'Q:\\finance-analyzer\\portfolio\\trade_validation.py',
'Q:\\finance-analyzer\\portfolio\\trade_risk_classifier.py',
'Q:\\finance-analyzer\\portfolio\\risk_management.py',
'Q:\\finance-analyzer\\portfolio\\monte_carlo.py',
'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py',
'Q:\\finance-analyzer\\portfolio\\equity_curve.py',
'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py',
'Q:\\finance-analyzer\\portfolio\\kelly_sizing.py',
'Q:\\finance-analyzer\\portfolio\\kelly_metals.py',
'Q:\\finance-analyzer\\portfolio\\exposure_coach.py',
'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py',
'Q:\\finance-analyzer\\portfolio\\cost_model.py',
'Q:\\finance-analyzer\\portfolio\\instrument_profile.py',
'Q:\\finance-analyzer\\portfolio\\stats.py'
) | ForEach-Object { if (Test-Path "'$_) { $_ } }' in Q:\fa-adv-2026-05-11
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files Q:\\finance-analyzer\\portfolio\\strategies; @(
'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py',
'Q:\\finance-analyzer\\portfolio\\portfolio_validator.py',
'Q:\\finance-analyzer\\portfolio\\trade_guards.py',
'Q:\\finance-analyzer\\portfolio\\trade_validation.py',
'Q:\\finance-analyzer\\portfolio\\trade_risk_classifier.py',
'Q:\\finance-analyzer\\portfolio\\risk_management.py',
'Q:\\finance-analyzer\\portfolio\\monte_carlo.py',
'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py',
'Q:\\finance-analyzer\\portfolio\\equity_curve.py',
'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py',
'Q:\\finance-analyzer\\portfolio\\kelly_sizing.py',
'Q:\\finance-analyzer\\portfolio\\kelly_metals.py',
'Q:\\finance-analyzer\\portfolio\\exposure_coach.py',
'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py',
'Q:\\finance-analyzer\\portfolio\\cost_model.py',
'Q:\\finance-analyzer\\portfolio\\instrument_profile.py',
'Q:\\finance-analyzer\\portfolio\\stats.py'
) | ForEach-Object { if (Test-Path "'$_) { $_ } }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files Q:\\finance-analyzer\\portfolio\\strategies" in Q:\fa-adv-2026-05-11
 succeeded in 641ms:
Q:\finance-analyzer\portfolio\strategies\orchestrator.py
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py
Q:\finance-analyzer\portfolio\strategies\elongir_strategy.py
Q:\finance-analyzer\portfolio\strategies\base.py
[31;1mMicrosoft.PowerShell_profile.ps1: [31;1mCannot dot-source this command because it was defined in a different language mode. To invoke this command without importing its contents, omit the '.' operator.[0m
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I have the strategy subfiles. Next I’m getting file sizes and then reading the code in batches so I can focus on real defects instead of keyword-only guesses.
2026-05-11T15:28:58.242967Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py').Count"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py').Count" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\risk_management.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py').Count" in Q:\fa-adv-2026-05-11
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py').Count"` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\risk_management.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py').Count"` rejected: blocked by policy
2026-05-11T15:28:58.243919Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\risk_management.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py').Count"` rejected: blocked by policy
2026-05-11T15:28:58.270488Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py').Count"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py').Count" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\equity_curve.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\equity_curve.py').Count" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py').Count" in Q:\fa-adv-2026-05-11
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py').Count"` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\equity_curve.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\equity_curve.py').Count"` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py').Count"` rejected: blocked by policy
2026-05-11T15:28:58.271032Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\equity_curve.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\equity_curve.py').Count"` rejected: blocked by policy
2026-05-11T15:28:58.271704Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "(Get-Item 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py').FullName; (Get-Content 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py').Count"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \".*\" Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py" in Q:\fa-adv-2026-05-11
 succeeded in 630ms:
1:"""Portfolio state management — load, save, atomic writes, value calculation."""
2:
3:import logging
4:import math
5:import shutil
6:import threading
7:from datetime import UTC, datetime
8:from pathlib import Path
9:
10:logger = logging.getLogger("portfolio.portfolio_mgr")
11:
12:from portfolio.file_utils import atomic_write_json as _atomic_write_json
13:from portfolio.file_utils import load_json
14:
15:BASE_DIR = Path(__file__).resolve().parent.parent
16:DATA_DIR = BASE_DIR / "data"
17:STATE_FILE = DATA_DIR / "portfolio_state.json"
18:BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"
19:INITIAL_CASH_SEK = 500_000
20:
21:_DEFAULT_STATE = {
22:    "cash_sek": INITIAL_CASH_SEK,
23:    "holdings": {},
24:    "transactions": [],
25:    "initial_value_sek": INITIAL_CASH_SEK,
26:}
27:
28:# C8: Per-file locks for concurrency safety
29:_state_locks: dict[str, threading.Lock] = {}
30:_locks_lock = threading.Lock()
31:
32:_MAX_BACKUPS = 3  # Rolling backup count
33:
34:
35:def _get_lock(path: Path) -> threading.Lock:
36:    """Get or create a per-file lock for concurrency safety (C8)."""
37:    key = str(path)
38:    with _locks_lock:
39:        if key not in _state_locks:
40:            _state_locks[key] = threading.Lock()
41:        return _state_locks[key]
42:
43:
44:def _rotate_backups(path: Path):
45:    """C7: Create rolling .bak backups before overwriting state.
46:
47:    Maintains up to _MAX_BACKUPS copies: path.bak, path.bak2, path.bak3.
48:    Only backs up if the file exists and has content.
49:    """
50:    if not path.exists() or path.stat().st_size == 0:
51:        return
52:    try:
53:        # Rotate existing backups: .bak2 → .bak3, .bak → .bak2
54:        for i in range(_MAX_BACKUPS, 1, -1):
55:            src = path.with_suffix(f".json.bak{i - 1}" if i > 2 else ".json.bak")
56:            dst = path.with_suffix(f".json.bak{i}")
57:            if src.exists():
58:                shutil.copy2(str(src), str(dst))
59:        # Current file → .bak
60:        shutil.copy2(str(path), str(path.with_suffix(".json.bak")))
61:    except OSError as e:
62:        logger.warning("Failed to rotate backups for %s: %s", path.name, e)
63:
64:
65:def _validated_state(loaded):
66:    """Merge loaded state with defaults to ensure all required keys exist."""
67:    if not loaded or not isinstance(loaded, dict):
68:        return {**_DEFAULT_STATE, "start_date": datetime.now(UTC).isoformat()}
69:    result = {**_DEFAULT_STATE, **loaded}
70:    # Ensure types are correct for critical fields
71:    if not isinstance(result.get("holdings"), dict):
72:        result["holdings"] = {}
73:    if not isinstance(result.get("transactions"), list):
74:        result["transactions"] = []
75:    return result
76:
77:
78:def _load_state_from(path: Path):
79:    """Load portfolio state from a specific file.
80:
81:    C7: On corruption, logs CRITICAL and attempts recovery from backups.
82:    Returns validated defaults only if file AND all backups are missing/corrupt.
83:    """
84:    loaded = load_json(str(path), default=None)
85:    if loaded is not None:
86:        return _validated_state(loaded)
87:
88:    # File is missing or corrupt — check if the file exists (corruption vs missing)
89:    if path.exists():
90:        logger.critical(
91:            "CORRUPT portfolio state file: %s — attempting backup recovery", path.name
92:        )
93:        # Try backups in order
94:        for i in range(1, _MAX_BACKUPS + 1):
95:            bak = path.with_suffix(f".json.bak{i}" if i > 1 else ".json.bak")
96:            if bak.exists():
97:                loaded = load_json(str(bak), default=None)
98:                if loaded is not None:
99:                    logger.warning("Recovered %s from backup %s", path.name, bak.name)
100:                    return _validated_state(loaded)
101:        logger.critical(
102:            "ALL backups corrupt/missing for %s — returning fresh defaults", path.name
103:        )
104:
105:    return {**_DEFAULT_STATE, "start_date": datetime.now(UTC).isoformat()}
106:
107:
108:def _save_state_to(path: Path, state):
109:    """Save state with rolling backup (C7) and lock (C8)."""
110:    lock = _get_lock(path)
111:    with lock:
112:        _rotate_backups(path)
113:        _atomic_write_json(path, state)
114:
115:
116:def load_state():
117:    """Load Patient portfolio state. Returns validated defaults if missing or corrupt."""
118:    return _load_state_from(STATE_FILE)
119:
120:
121:def save_state(state):
122:    """Save Patient portfolio state with backup rotation (C7)."""
123:    _save_state_to(STATE_FILE, state)
124:
125:
126:def load_bold_state():
127:    """Load Bold portfolio state. Returns validated defaults if missing or corrupt."""
128:    return _load_state_from(BOLD_STATE_FILE)
129:
130:
131:def save_bold_state(state):
132:    """Save Bold portfolio state with backup rotation (C7)."""
133:    _save_state_to(BOLD_STATE_FILE, state)
134:
135:
136:def update_state(mutate_fn, bold=False):
137:    """Atomic read-modify-write for portfolio state (C8).
138:
139:    Holds a lock for the entire read-modify-write cycle to prevent
140:    concurrent callers from overwriting each other's mutations.
141:
142:    Args:
143:        mutate_fn: Callable that receives the current state dict and mutates it.
144:            The function should modify the dict in-place and optionally return it.
145:        bold: If True, operates on the Bold portfolio instead of Patient.
146:
147:    Returns:
148:        The updated state dict.
149:    """
150:    path = BOLD_STATE_FILE if bold else STATE_FILE
151:    lock = _get_lock(path)
152:    with lock:
153:        state = _load_state_from(path)
154:        result = mutate_fn(state)
155:        if result is not None:
156:            state = result
157:        _rotate_backups(path)
158:        _atomic_write_json(path, state)
159:    return state
160:
161:
162:def portfolio_value(state, prices_usd, fx_rate):
163:    if not isinstance(fx_rate, (int, float)) or not math.isfinite(fx_rate) or fx_rate <= 0:
164:        logger.warning("portfolio_value: invalid fx_rate=%r, returning cash only", fx_rate)
165:        return state.get("cash_sek", 0)
166:    total = state.get("cash_sek", 0)
167:    for ticker, h in state.get("holdings", {}).items():
168:        try:
169:            shares = h.get("shares", 0)
170:            price = prices_usd.get(ticker)
171:            if shares > 0 and price is not None and price > 0:
172:                total += shares * price * fx_rate
173:            elif shares > 0 and (price is None or price <= 0):
174:                logger.warning(
175:                    "portfolio_value: invalid price for %s: %r (shares=%s)",
176:                    ticker, price, shares,
177:                )
178:        except (TypeError, ValueError, AttributeError) as e:
179:            logger.warning("portfolio_value: error calculating %s: %s", ticker, e)
180:    return total
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve confirmed the portfolio manager file already, including the `update_state` path. Now I’m scanning the rest of the in-scope files for the exact risk surfaces: sizing, stop-loss, cost/FX, equity accounting, Monte Carlo, guards, and classifier fallbacks.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n -S \"kelly|half|quarter|min[_ -]?order|stop[_ -]?loss|barrier|ATR|atr|spread|fee|fx|partial|NaN|nan|monte|VaR|var|drawdown|circuit|cooldown|monotonic|time\\(|perf_counter|fallback|default|classif|live price|cached|precomputed|atomic|lock|thread\" Q:\\finance-analyzer\\portfolio\\portfolio_validator.py Q:\\finance-analyzer\\portfolio\\trade_guards.py Q:\\finance-analyzer\\portfolio\\trade_validation.py Q:\\finance-analyzer\\portfolio\\trade_risk_classifier.py Q:\\finance-analyzer\\portfolio\\risk_management.py Q:\\finance-analyzer\\portfolio\\monte_carlo.py Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py Q:\\finance-analyzer\\portfolio\\equity_curve.py Q:\\finance-analyzer\\portfolio\\exit_optimizer.py Q:\\finance-analyzer\\portfolio\\kelly_sizing.py Q:\\finance-analyzer\\portfolio\\kelly_metals.py Q:\\finance-analyzer\\portfolio\\exposure_coach.py Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py Q:\\finance-analyzer\\portfolio\\cost_model.py Q:\\finance-analyzer\\portfolio\\instrument_profile.py Q:\\finance-analyzer\\portfolio\\stats.py Q:\\finance-analyzer\\portfolio\\strategies" in Q:\fa-adv-2026-05-11
 succeeded in 734ms:
Q:\finance-analyzer\portfolio\trade_risk_classifier.py:17:logger = logging.getLogger("portfolio.trade_risk_classifier")
Q:\finance-analyzer\portfolio\trade_risk_classifier.py:29:def classify_trade_risk(
Q:\finance-analyzer\portfolio\stats.py:1:from collections import Counter, defaultdict
Q:\finance-analyzer\portfolio\stats.py:18:    by_date = defaultdict(list)
Q:\finance-analyzer\portfolio\stats.py:22:        day = dt.strftime("%Y-%m-%d")
Q:\finance-analyzer\portfolio\stats.py:55:    by_date = defaultdict(list)
Q:\finance-analyzer\portfolio\stats.py:59:        day = dt.strftime("%Y-%m-%d")
Q:\finance-analyzer\portfolio\instrument_profile.py:138:        "binance_symbol": "XAGUSDT",
Q:\finance-analyzer\portfolio\instrument_profile.py:173:        "binance_symbol": "XAUUSDT",
Q:\finance-analyzer\portfolio\risk_management.py:4:- Maximum drawdown circuit breaker
Q:\finance-analyzer\portfolio\risk_management.py:5:- ATR-based trailing stop-loss tracking
Q:\finance-analyzer\portfolio\risk_management.py:16:import threading
Q:\finance-analyzer\portfolio\risk_management.py:18:from portfolio.file_utils import atomic_append_jsonl, load_json
Q:\finance-analyzer\portfolio\risk_management.py:19:from portfolio.fx_rates import FX_RATE_FALLBACK, FX_RATE_MAX, FX_RATE_MIN
Q:\finance-analyzer\portfolio\risk_management.py:31:# ThreadPoolExecutor invokes check_drawdown() concurrently across the
Q:\finance-analyzer\portfolio\risk_management.py:33:# _streaming_max. Without a lock, two threads could last-writer-wins on
Q:\finance-analyzer\portfolio\risk_management.py:34:# _peak_cache[cache_key] = {...}, occasionally losing a cached offset and
Q:\finance-analyzer\portfolio\risk_management.py:38:# update sequence in _streaming_max — both reading the cached offset and
Q:\finance-analyzer\portfolio\risk_management.py:39:# writing back the new offset have to be atomic relative to other workers.
Q:\finance-analyzer\portfolio\risk_management.py:40:_peak_cache_lock = threading.Lock()
Q:\finance-analyzer\portfolio\risk_management.py:51:    _peak_cache_lock to avoid the 8-worker ThreadPoolExecutor losing cached
Q:\finance-analyzer\portfolio\risk_management.py:52:    offsets to last-writer-wins races (see lock comment above).
Q:\finance-analyzer\portfolio\risk_management.py:67:    # Snapshot cache under the lock — keeps the (read offset, decide
Q:\finance-analyzer\portfolio\risk_management.py:70:    with _peak_cache_lock:
Q:\finance-analyzer\portfolio\risk_management.py:71:        cached = _peak_cache.get(cache_key)
Q:\finance-analyzer\portfolio\risk_management.py:72:        if cached is not None:
Q:\finance-analyzer\portfolio\risk_management.py:73:            if file_size >= cached["offset"]:
Q:\finance-analyzer\portfolio\risk_management.py:74:                start_offset = cached["offset"]
Q:\finance-analyzer\portfolio\risk_management.py:75:                peak = cached["peak"]
Q:\finance-analyzer\portfolio\risk_management.py:102:        with _peak_cache_lock:
Q:\finance-analyzer\portfolio\risk_management.py:103:            cached_after = _peak_cache.get(cache_key)
Q:\finance-analyzer\portfolio\risk_management.py:104:        if cached_after is not None:
Q:\finance-analyzer\portfolio\risk_management.py:105:            return float(cached_after["peak"])
Q:\finance-analyzer\portfolio\risk_management.py:108:    with _peak_cache_lock:
Q:\finance-analyzer\portfolio\risk_management.py:116:# Adversarial review 05-01 P1-15 (2026-05-02): persistent fallback for fx_rate.
Q:\finance-analyzer\portfolio\risk_management.py:117:# FX constants imported from portfolio.fx_rates (single source of truth).
Q:\finance-analyzer\portfolio\risk_management.py:118:_FX_CACHE_FILENAME = "fx_rate_cache.json"
Q:\finance-analyzer\portfolio\risk_management.py:121:def _resolve_fx_rate(agent_summary: dict) -> float:
Q:\finance-analyzer\portfolio\risk_management.py:122:    """Return USD→SEK rate, preferring (1) summary, (2) cached, (3) hardcoded.
Q:\finance-analyzer\portfolio\risk_management.py:125:    ``fx_rate = agent_summary.get("fx_rate", 1.0)``. When agent_summary was
Q:\finance-analyzer\portfolio\risk_management.py:127:    rotation, fx_rates.py crashed mid-fetch — the 1.0 default understated
Q:\finance-analyzer\portfolio\risk_management.py:128:    SEK valuations by ~10x and could trigger a false drawdown breach
Q:\finance-analyzer\portfolio\risk_management.py:129:    (a 5_435_000 SEK position valued at fx_rate=1.0 looks like 95%
Q:\finance-analyzer\portfolio\risk_management.py:130:    drawdown from the 500_000 SEK initial).
Q:\finance-analyzer\portfolio\risk_management.py:133:      1. ``agent_summary["fx_rate"]`` if it's a finite number in [7, 15].
Q:\finance-analyzer\portfolio\risk_management.py:134:         (Same sanity band as portfolio/fx_rates.py:42.) Successful values
Q:\finance-analyzer\portfolio\risk_management.py:135:         update the disk cache for future fallbacks.
Q:\finance-analyzer\portfolio\risk_management.py:136:      2. Cached rate from ``DATA_DIR/fx_rate_cache.json`` if present and
Q:\finance-analyzer\portfolio\risk_management.py:140:         portfolio/fx_rates.py:66 so both modules disagree-by-zero on the
Q:\finance-analyzer\portfolio\risk_management.py:144:    pattern ``agent_summary.get("fx_rate", 1.0)`` continues to defer
Q:\finance-analyzer\portfolio\risk_management.py:147:    raw = agent_summary.get("fx_rate") if isinstance(agent_summary, dict) else None
Q:\finance-analyzer\portfolio\risk_management.py:153:        # Cache the good rate for future fallback paths.
Q:\finance-analyzer\portfolio\risk_management.py:155:            from portfolio.file_utils import atomic_write_json
Q:\finance-analyzer\portfolio\risk_management.py:156:            atomic_write_json(DATA_DIR / _FX_CACHE_FILENAME, {
Q:\finance-analyzer\portfolio\risk_management.py:161:            logger.debug("fx cache persist failed: %s", e)
Q:\finance-analyzer\portfolio\risk_management.py:165:    cached = load_json(DATA_DIR / _FX_CACHE_FILENAME, default=None)
Q:\finance-analyzer\portfolio\risk_management.py:166:    if isinstance(cached, dict):
Q:\finance-analyzer\portfolio\risk_management.py:167:        # 2026-05-10 (codex re-review): cached.get("rate") is Any|None;
Q:\finance-analyzer\portfolio\risk_management.py:171:        rate_raw = cached.get("rate")
Q:\finance-analyzer\portfolio\risk_management.py:174:                cached_rate = float(rate_raw)
Q:\finance-analyzer\portfolio\risk_management.py:175:                if FX_RATE_MIN <= cached_rate <= FX_RATE_MAX:
Q:\finance-analyzer\portfolio\risk_management.py:176:                    return cached_rate
Q:\finance-analyzer\portfolio\risk_management.py:181:        "fx_rate fallback to hardcoded %.2f — agent_summary missing/invalid "
Q:\finance-analyzer\portfolio\risk_management.py:193:    current_price_sek is derived from agent_summary prices * fx_rate.
Q:\finance-analyzer\portfolio\risk_management.py:197:    # P1-15 (2026-05-02): use cached fallback chain instead of raw .get(..., 1.0).
Q:\finance-analyzer\portfolio\risk_management.py:198:    fx_rate = _resolve_fx_rate(agent_summary)
Q:\finance-analyzer\portfolio\risk_management.py:209:            holdings_value += shares * price_usd * fx_rate
Q:\finance-analyzer\portfolio\risk_management.py:211:            # Fallback: use avg_cost_usd from holdings if no live price
Q:\finance-analyzer\portfolio\risk_management.py:213:            holdings_value += shares * avg_cost * fx_rate
Q:\finance-analyzer\portfolio\risk_management.py:218:def check_drawdown(portfolio_path: str, max_drawdown_pct: float = 20.0,
Q:\finance-analyzer\portfolio\risk_management.py:220:    """Check if portfolio has exceeded maximum drawdown threshold.
Q:\finance-analyzer\portfolio\risk_management.py:227:        max_drawdown_pct: Maximum allowed drawdown percentage (default 20%).
Q:\finance-analyzer\portfolio\risk_management.py:228:        agent_summary_path: Path to agent_summary.json for live prices.
Q:\finance-analyzer\portfolio\risk_management.py:233:            - breached: bool -- True if drawdown exceeds threshold
Q:\finance-analyzer\portfolio\risk_management.py:234:            - current_drawdown_pct: float -- current drawdown from peak (positive number)
Q:\finance-analyzer\portfolio\risk_management.py:239:    portfolio = load_json(portfolio_path, default={})
Q:\finance-analyzer\portfolio\risk_management.py:249:        summary = load_json(agent_summary_path, default={})
Q:\finance-analyzer\portfolio\risk_management.py:255:            # feed is stale while holdings are underwater, the
Q:\finance-analyzer\portfolio\risk_management.py:256:            # drawdown reading will look tiny and the circuit
Q:\finance-analyzer\portfolio\risk_management.py:258:            # WARNING so dashboards/oncall can see "feed stale"
Q:\finance-analyzer\portfolio\risk_management.py:266:                "check_drawdown: agent_summary empty at %s — falling "
Q:\finance-analyzer\portfolio\risk_management.py:268:                "Drawdown circuit breaker reading may be optimistic.",
Q:\finance-analyzer\portfolio\risk_management.py:281:    # any rally older than that fell off the back and the drawdown circuit
Q:\finance-analyzer\portfolio\risk_management.py:289:    # Guard against NaN/Inf in peak_value or current_value — corrupted
Q:\finance-analyzer\portfolio\risk_management.py:290:    # history or failed computation. NaN silently passes all comparison
Q:\finance-analyzer\portfolio\risk_management.py:291:    # checks (NaN > 50.0 is False), bypassing the circuit breaker.
Q:\finance-analyzer\portfolio\risk_management.py:294:            "check_drawdown: non-finite value detected (peak=%.2f, current=%.2f) "
Q:\finance-analyzer\portfolio\risk_management.py:295:            "— treating as 100%% drawdown (fail-safe).",
Q:\finance-analyzer\portfolio\risk_management.py:300:            "current_drawdown_pct": 100.0,
Q:\finance-analyzer\portfolio\risk_management.py:306:    # Calculate drawdown
Q:\finance-analyzer\portfolio\risk_management.py:308:        current_drawdown_pct = ((peak_value - current_value) / peak_value) * 100
Q:\finance-analyzer\portfolio\risk_management.py:310:        current_drawdown_pct = 0.0
Q:\finance-analyzer\portfolio\risk_management.py:313:        "breached": current_drawdown_pct > max_drawdown_pct,
Q:\finance-analyzer\portfolio\risk_management.py:314:        "current_drawdown_pct": round(current_drawdown_pct, 4),
Q:\finance-analyzer\portfolio\risk_management.py:322:    """Compute ATR-based stop-loss levels for all positions.
Q:\finance-analyzer\portfolio\risk_management.py:324:    For each holding with shares > 0, calculates stop-loss levels based on
Q:\finance-analyzer\portfolio\risk_management.py:325:    2x ATR (Average True Range) from the entry price.
Q:\finance-analyzer\portfolio\risk_management.py:336:            - atr_pct: float (from agent_summary)
Q:\finance-analyzer\portfolio\risk_management.py:337:            - stop_price_usd: float (entry_price * (1 - 2 * atr_pct/100))
Q:\finance-analyzer\portfolio\risk_management.py:354:        # Get current price and ATR from agent_summary
Q:\finance-analyzer\portfolio\risk_management.py:360:                "atr_pct": None,
Q:\finance-analyzer\portfolio\risk_management.py:371:        atr_pct = sig.get("atr_pct", 0)
Q:\finance-analyzer\portfolio\risk_management.py:373:        # 2x ATR stop-loss — cap ATR at 15% to prevent meaninglessly wide stops for warrants
Q:\finance-analyzer\portfolio\risk_management.py:374:        atr_pct = min(atr_pct, 15.0)
Q:\finance-analyzer\portfolio\risk_management.py:375:        stop_price = entry_price * (1 - 2 * atr_pct / 100)
Q:\finance-analyzer\portfolio\risk_management.py:389:            "atr_pct": round(atr_pct, 4),
Q:\finance-analyzer\portfolio\risk_management.py:400:    """Compute Monte Carlo probabilistic stop-loss quality for held positions.
Q:\finance-analyzer\portfolio\risk_management.py:403:    the probability of hitting the ATR stop level, plus the expected time to
Q:\finance-analyzer\portfolio\risk_management.py:412:            - stop_price_usd: ATR-based stop level
Q:\finance-analyzer\portfolio\risk_management.py:415:            - knockout_prob: For warrants, P(reaching financing level)
Q:\finance-analyzer\portfolio\risk_management.py:444:        atr_pct = sig.get("atr_pct", 0)
Q:\finance-analyzer\portfolio\risk_management.py:445:        if current_price <= 0 or atr_pct <= 0:
Q:\finance-analyzer\portfolio\risk_management.py:461:        # Estimate volatility from ATR
Q:\finance-analyzer\portfolio\risk_management.py:463:        vol = max(atr_pct / 100.0 * math.sqrt(252.0 / 14), 0.05)
Q:\finance-analyzer\portfolio\risk_management.py:465:        # ATR stop level
Q:\finance-analyzer\portfolio\risk_management.py:466:        stop_price = entry_price * (1 - 2 * atr_pct / 100)
Q:\finance-analyzer\portfolio\risk_management.py:590:    patient = load_json(patient_path, default={})
Q:\finance-analyzer\portfolio\risk_management.py:591:    bold = load_json(bold_path, default={})
Q:\finance-analyzer\portfolio\risk_management.py:592:    # P1-15 (2026-05-02): the {"fx_rate": 1.0} default here was load-bearing
Q:\finance-analyzer\portfolio\risk_management.py:593:    # for the false-circuit-breaker bug. Using a missing-fx_rate default lets
Q:\finance-analyzer\portfolio\risk_management.py:594:    # _resolve_fx_rate inside _compute_portfolio_value walk the cache chain
Q:\finance-analyzer\portfolio\risk_management.py:596:    summary = load_json(agent_summary_path, default={"signals": {}})
Q:\finance-analyzer\portfolio\risk_management.py:620:        "fx_rate": _resolve_fx_rate(summary),  # P1-15 (2026-05-02)
Q:\finance-analyzer\portfolio\risk_management.py:625:    atomic_append_jsonl(history_path, entry)
Q:\finance-analyzer\portfolio\risk_management.py:631:    Examines all transactions and accumulated fees to produce a cost report.
Q:\finance-analyzer\portfolio\risk_management.py:638:            - total_fees_sek: float -- accumulated fees
Q:\finance-analyzer\portfolio\risk_management.py:639:            - fees_as_pct_of_initial: float -- total_fees / initial_value * 100
Q:\finance-analyzer\portfolio\risk_management.py:640:            - avg_fee_per_trade: float -- average fee per transaction
Q:\finance-analyzer\portfolio\risk_management.py:642:            - fees_as_pct_of_pnl: float | None -- total_fees / abs(pnl) * 100
Q:\finance-analyzer\portfolio\risk_management.py:653:    # total_fees_sek from portfolio state
Q:\finance-analyzer\portfolio\risk_management.py:654:    total_fees_from_state = portfolio.get("total_fees_sek", 0) or 0
Q:\finance-analyzer\portfolio\risk_management.py:656:    # Also compute fees from transaction records (fee_sek field)
Q:\finance-analyzer\portfolio\risk_management.py:657:    computed_fees = 0.0
Q:\finance-analyzer\portfolio\risk_management.py:664:        fee = tx.get("fee_sek", 0) or 0
Q:\finance-analyzer\portfolio\risk_management.py:665:        computed_fees += fee
Q:\finance-analyzer\portfolio\risk_management.py:676:    # Use the larger of state fees vs computed fees (handles missing fee_sek fields)
Q:\finance-analyzer\portfolio\risk_management.py:677:    total_fees = max(total_fees_from_state, computed_fees)
Q:\finance-analyzer\portfolio\risk_management.py:680:    avg_fee = total_fees / total_trades if total_trades > 0 else 0
Q:\finance-analyzer\portfolio\risk_management.py:682:    fees_as_pct_initial = (total_fees / initial_value) * 100 if initial_value > 0 else 0
Q:\finance-analyzer\portfolio\risk_management.py:686:    # value requires live prices). A full PnL needs _compute_portfolio_value.
Q:\finance-analyzer\portfolio\risk_management.py:702:        fees_as_pct_pnl = (total_fees / abs(pnl)) * 100
Q:\finance-analyzer\portfolio\risk_management.py:704:        fees_as_pct_pnl = None
Q:\finance-analyzer\portfolio\risk_management.py:707:        "total_fees_sek": round(total_fees, 2),
Q:\finance-analyzer\portfolio\risk_management.py:708:        "fees_as_pct_of_initial": round(fees_as_pct_initial, 4),
Q:\finance-analyzer\portfolio\risk_management.py:709:        "avg_fee_per_trade": round(avg_fee, 2),
Q:\finance-analyzer\portfolio\risk_management.py:711:        "fees_as_pct_of_pnl": round(fees_as_pct_pnl, 4) if fees_as_pct_pnl is not None else None,
Q:\finance-analyzer\portfolio\risk_management.py:752:    # P1-15 (2026-05-02): use cached fallback chain instead of raw .get(..., 1.0).
Q:\finance-analyzer\portfolio\risk_management.py:753:    fx_rate = _resolve_fx_rate(agent_summary)
Q:\finance-analyzer\portfolio\risk_management.py:763:        total_value += shares * price * fx_rate
Q:\finance-analyzer\portfolio\risk_management.py:776:    existing_value = existing_shares * existing_price * fx_rate
Q:\finance-analyzer\portfolio\risk_management.py:873:def check_atr_stop_proximity(ticker, action, portfolio, agent_summary):
Q:\finance-analyzer\portfolio\risk_management.py:874:    """Check if current price is within 1x ATR of the computed stop level.
Q:\finance-analyzer\portfolio\risk_management.py:876:    This flags positions that are dangerously close to their stop-loss.
Q:\finance-analyzer\portfolio\risk_management.py:896:    atr_pct = sig.get("atr_pct", 0)
Q:\finance-analyzer\portfolio\risk_management.py:898:    if current_price <= 0 or atr_pct <= 0:
Q:\finance-analyzer\portfolio\risk_management.py:901:    # 2x ATR stop level
Q:\finance-analyzer\portfolio\risk_management.py:902:    stop_price = entry_price * (1 - 2 * atr_pct / 100)
Q:\finance-analyzer\portfolio\risk_management.py:903:    # Distance from current price to stop (in ATR units)
Q:\finance-analyzer\portfolio\risk_management.py:904:    atr_value = current_price * atr_pct / 100
Q:\finance-analyzer\portfolio\risk_management.py:905:    if atr_value <= 0:
Q:\finance-analyzer\portfolio\risk_management.py:909:    distance_in_atr = distance_to_stop / atr_value
Q:\finance-analyzer\portfolio\risk_management.py:911:    if distance_in_atr < 1.0:
Q:\finance-analyzer\portfolio\risk_management.py:913:            "flag": "atr_stop_proximity",
Q:\finance-analyzer\portfolio\risk_management.py:918:            "distance_atr": round(distance_in_atr, 2),
Q:\finance-analyzer\portfolio\risk_management.py:920:                f"{ticker}: price ${current_price:.2f} is {distance_in_atr:.1f}x ATR "
Q:\finance-analyzer\portfolio\risk_management.py:921:                f"from stop ${stop_price:.2f} (danger zone < 1.0x ATR)"
Q:\finance-analyzer\portfolio\risk_management.py:951:            # Still check ATR proximity for held positions
Q:\finance-analyzer\portfolio\risk_management.py:953:                flag = check_atr_stop_proximity(ticker, "CHECK", pf, agent_summary)
Q:\finance-analyzer\portfolio\risk_management.py:970:            # ATR stop proximity
Q:\finance-analyzer\portfolio\risk_management.py:971:            flag = check_atr_stop_proximity(ticker, action, pf, agent_summary)
Q:\finance-analyzer\portfolio\risk_management.py:986:            by_flag.setdefault(f["flag"], []).append(f)
Q:\finance-analyzer\portfolio\trade_guards.py:4:1. Per-ticker cooldown: No re-trade on same ticker within N minutes.
Q:\finance-analyzer\portfolio\trade_guards.py:5:2. Consecutive-loss escalation: After losses, increase cooldown multiplier.
Q:\finance-analyzer\portfolio\trade_guards.py:12:import threading
Q:\finance-analyzer\portfolio\trade_guards.py:16:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\trade_guards.py:29:LOSS_ESCALATION = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8}  # consecutive_losses -> cooldown multiplier
Q:\finance-analyzer\portfolio\trade_guards.py:32:_state_lock = threading.Lock()
Q:\finance-analyzer\portfolio\trade_guards.py:37:    return load_json(str(STATE_FILE), default={
Q:\finance-analyzer\portfolio\trade_guards.py:47:    atomic_write_json(STATE_FILE, state)
Q:\finance-analyzer\portfolio\trade_guards.py:66:        pf = load_json(str(DATA_DIR / pf_name), default={})
Q:\finance-analyzer\portfolio\trade_guards.py:72:def _get_cooldown_multiplier(consecutive_losses, last_loss_ts_str=None):
Q:\finance-analyzer\portfolio\trade_guards.py:73:    """Get cooldown multiplier based on consecutive loss count with time decay.
Q:\finance-analyzer\portfolio\trade_guards.py:116:            - severity: "warning" or "block"
Q:\finance-analyzer\portfolio\trade_guards.py:126:    with _state_lock:
Q:\finance-analyzer\portfolio\trade_guards.py:130:    # --- Guard 1: Per-ticker cooldown ---
Q:\finance-analyzer\portfolio\trade_guards.py:131:    base_cooldown = cfg.get("ticker_cooldown_minutes", DEFAULT_TICKER_COOLDOWN_MINUTES)
Q:\finance-analyzer\portfolio\trade_guards.py:134:    multiplier = _get_cooldown_multiplier(consecutive, last_loss_ts)
Q:\finance-analyzer\portfolio\trade_guards.py:135:    effective_cooldown = base_cooldown * multiplier
Q:\finance-analyzer\portfolio\trade_guards.py:147:            if elapsed < effective_cooldown:
Q:\finance-analyzer\portfolio\trade_guards.py:148:                remaining = effective_cooldown - elapsed
Q:\finance-analyzer\portfolio\trade_guards.py:150:                    "guard": "ticker_cooldown",
Q:\finance-analyzer\portfolio\trade_guards.py:151:                    "severity": "block",
Q:\finance-analyzer\portfolio\trade_guards.py:154:                        f"Cooldown: {effective_cooldown:.0f}m (base {base_cooldown}m × {multiplier}x). "
Q:\finance-analyzer\portfolio\trade_guards.py:161:                        "cooldown_min": effective_cooldown,
Q:\finance-analyzer\portfolio\trade_guards.py:171:        base_mult = _get_cooldown_multiplier(consecutive, None)
Q:\finance-analyzer\portfolio\trade_guards.py:218:                "severity": "block",
Q:\finance-analyzer\portfolio\trade_guards.py:240:    Call this after executing a trade to update cooldowns and loss streaks.
Q:\finance-analyzer\portfolio\trade_guards.py:264:    with _state_lock:
Q:\finance-analyzer\portfolio\trade_guards.py:351:            by_guard.setdefault(guard, []).append(w)
Q:\finance-analyzer\portfolio\trade_guards.py:361:    with _state_lock:
Q:\finance-analyzer\portfolio\trade_guards.py:376:def should_block_trade(guard_result):
Q:\finance-analyzer\portfolio\trade_guards.py:377:    """Check if any guard warning has 'block' severity.
Q:\finance-analyzer\portfolio\trade_guards.py:385:        True if any warning has severity="block", False otherwise.
Q:\finance-analyzer\portfolio\trade_guards.py:388:    return any(w.get("severity") == "block" for w in warnings)
Q:\finance-analyzer\portfolio\trade_validation.py:3:Validates trade parameters (price, volume, spread, cash, position size) before
Q:\finance-analyzer\portfolio\trade_validation.py:19:    warnings: list = field(default_factory=list)
Q:\finance-analyzer\portfolio\trade_validation.py:30:    max_spread_pct: float = 2.0,
Q:\finance-analyzer\portfolio\trade_validation.py:32:    min_order_sek: float = 500.0,
Q:\finance-analyzer\portfolio\trade_validation.py:42:    5. Bid/ask spread width
Q:\finance-analyzer\portfolio\trade_validation.py:60:    if order_value < min_order_sek:
Q:\finance-analyzer\portfolio\trade_validation.py:63:            f"Order value {order_value:.0f} SEK below minimum {min_order_sek:.0f} SEK",
Q:\finance-analyzer\portfolio\trade_validation.py:83:    # --- 5. Bid/ask spread check ---
Q:\finance-analyzer\portfolio\trade_validation.py:85:        spread_pct = ((ask - bid) / bid) * 100
Q:\finance-analyzer\portfolio\trade_validation.py:86:        if spread_pct > max_spread_pct:
Q:\finance-analyzer\portfolio\trade_validation.py:89:                f"Spread too wide: {spread_pct:.2f}% (max {max_spread_pct:.1f}%)",
Q:\finance-analyzer\portfolio\trade_validation.py:91:        if spread_pct > max_spread_pct * 0.7:
Q:\finance-analyzer\portfolio\trade_validation.py:92:            warnings.append(f"Spread warning: {spread_pct:.2f}% approaching limit")
Q:\finance-analyzer\portfolio\kelly_sizing.py:21:def kelly_fraction(win_prob, avg_win_pct, avg_loss_pct):
Q:\finance-analyzer\portfolio\kelly_sizing.py:49:    kelly = (win_prob * b - q) / b
Q:\finance-analyzer\portfolio\kelly_sizing.py:52:    return max(0.0, min(1.0, kelly))
Q:\finance-analyzer\portfolio\kelly_sizing.py:69:    from collections import defaultdict
Q:\finance-analyzer\portfolio\kelly_sizing.py:70:    buys_by_ticker = defaultdict(list)
Q:\finance-analyzer\portfolio\kelly_sizing.py:71:    sells_by_ticker = defaultdict(list)
Q:\finance-analyzer\portfolio\kelly_sizing.py:134:        ticker: Ticker to get accuracy for (used for weighted_confidence fallback).
Q:\finance-analyzer\portfolio\kelly_sizing.py:165:    (produced by `accuracy_stats.accuracy_by_ticker_signal_cached()` upstream).
Q:\finance-analyzer\portfolio\kelly_sizing.py:168:    `signal_accuracy_1d.signals` block. Without this, a signal that's 70%
Q:\finance-analyzer\portfolio\kelly_sizing.py:170:    system-wide aggregate, distorting Kelly sizing on both. The fallback
Q:\finance-analyzer\portfolio\kelly_sizing.py:172:    don't yet emit the per-ticker block.
Q:\finance-analyzer\portfolio\kelly_sizing.py:185:    # P1-11: per-ticker per-signal accuracy block (preferred when present).
Q:\finance-analyzer\portfolio\kelly_sizing.py:186:    per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
Q:\finance-analyzer\portfolio\kelly_sizing.py:187:    per_ticker_for_ticker = per_ticker_block.get(ticker, {}) or {}
Q:\finance-analyzer\portfolio\kelly_sizing.py:239:        portfolio_path: Path to portfolio state JSON. If None, uses default
Q:\finance-analyzer\portfolio\kelly_sizing.py:242:        strategy: "patient" or "bold" - determines default portfolio and max alloc.
Q:\finance-analyzer\portfolio\kelly_sizing.py:246:            kelly_pct: Full Kelly fraction (0-1),
Q:\finance-analyzer\portfolio\kelly_sizing.py:247:            half_kelly_pct: Half Kelly (more conservative),
Q:\finance-analyzer\portfolio\kelly_sizing.py:248:            quarter_kelly_pct: Quarter Kelly (most conservative),
Q:\finance-analyzer\portfolio\kelly_sizing.py:249:            recommended_sek: Recommended trade size in SEK (using half Kelly),
Q:\finance-analyzer\portfolio\kelly_sizing.py:260:    portfolio = load_json(portfolio_path, default={})
Q:\finance-analyzer\portfolio\kelly_sizing.py:266:        agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
Q:\finance-analyzer\portfolio\kelly_sizing.py:276:    # per-ticker block is supplied in agent_summary. Source string
Q:\finance-analyzer\portfolio\kelly_sizing.py:278:    per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
Q:\finance-analyzer\portfolio\kelly_sizing.py:279:    has_per_ticker = bool(per_ticker_block.get(ticker))
Q:\finance-analyzer\portfolio\kelly_sizing.py:284:        source = f"weighted signal accuracy for {ticker} (system-wide fallback)"
Q:\finance-analyzer\portfolio\kelly_sizing.py:302:        # Use ATR from agent summary if available
Q:\finance-analyzer\portfolio\kelly_sizing.py:305:        atr_pct = ticker_data.get("atr_pct", 1.5) if isinstance(ticker_data, dict) else 1.5
Q:\finance-analyzer\portfolio\kelly_sizing.py:307:        # Assume avg win = 1.5x ATR, avg loss = 1x ATR (realistic risk/reward)
Q:\finance-analyzer\portfolio\kelly_sizing.py:308:        avg_win = atr_pct * 1.5
Q:\finance-analyzer\portfolio\kelly_sizing.py:309:        avg_loss = atr_pct * 1.0
Q:\finance-analyzer\portfolio\kelly_sizing.py:310:        source += f" + ATR-based estimates (win={avg_win:.1f}%, loss={avg_loss:.1f}%)"
Q:\finance-analyzer\portfolio\kelly_sizing.py:313:    full_kelly = kelly_fraction(win_prob, avg_win, avg_loss)
Q:\finance-analyzer\portfolio\kelly_sizing.py:314:    half_kelly = full_kelly / 2.0
Q:\finance-analyzer\portfolio\kelly_sizing.py:315:    quarter_kelly = full_kelly / 4.0
Q:\finance-analyzer\portfolio\kelly_sizing.py:322:    # Recommended size = half Kelly * cash * exposure_ceiling, capped at max allocation
Q:\finance-analyzer\portfolio\kelly_sizing.py:323:    rec_sek = min(half_kelly * cash_sek * exposure_ceiling, max_alloc)
Q:\finance-analyzer\portfolio\kelly_sizing.py:330:        "kelly_pct": round(full_kelly, 4),
Q:\finance-analyzer\portfolio\kelly_sizing.py:331:        "half_kelly_pct": round(half_kelly, 4),
Q:\finance-analyzer\portfolio\kelly_sizing.py:332:        "quarter_kelly_pct": round(quarter_kelly, 4),
Q:\finance-analyzer\portfolio\kelly_sizing.py:350:    agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
Q:\finance-analyzer\portfolio\kelly_sizing.py:372:            f"{ticker:<10} {rec['kelly_pct']*100:>6.1f}% {rec['half_kelly_pct']*100:>7.1f}% "
Q:\finance-analyzer\portfolio\cost_model.py:1:"""Cost model for trade execution — fees, spread, and slippage.
Q:\finance-analyzer\portfolio\cost_model.py:22:        courtage_bps: Brokerage fee in basis points (e.g., 6.9 = 0.069%).
Q:\finance-analyzer\portfolio\cost_model.py:23:        min_fee_sek: Minimum fee in SEK (courtage floor).
Q:\finance-analyzer\portfolio\cost_model.py:24:        spread_bps: Half-spread cost in basis points. For a sell, you cross
Q:\finance-analyzer\portfolio\cost_model.py:25:            from mid to bid, losing half the spread.
Q:\finance-analyzer\portfolio\cost_model.py:31:    min_fee_sek: float = 0.0
Q:\finance-analyzer\portfolio\cost_model.py:32:    spread_bps: float = 0.0
Q:\finance-analyzer\portfolio\cost_model.py:34:    label: str = "default"
Q:\finance-analyzer\portfolio\cost_model.py:44:        courtage = max(trade_value_sek * self.courtage_bps / 10_000, self.min_fee_sek)
Q:\finance-analyzer\portfolio\cost_model.py:45:        spread = trade_value_sek * self.spread_bps / 10_000
Q:\finance-analyzer\portfolio\cost_model.py:47:        return courtage + spread + slippage
Q:\finance-analyzer\portfolio\cost_model.py:50:        """Total cost as a percentage of trade value (excluding min fee)."""
Q:\finance-analyzer\portfolio\cost_model.py:51:        return (self.courtage_bps + self.spread_bps + self.slippage_bps) / 100.0
Q:\finance-analyzer\portfolio\cost_model.py:62:# Avanza warrants/certificates: 0 courtage on many, spread is the real cost.
Q:\finance-analyzer\portfolio\cost_model.py:63:# Typical MINI silver spread: 0.6-1.0% (30-50 bps half-spread).
Q:\finance-analyzer\portfolio\cost_model.py:66:    min_fee_sek=0.0,
Q:\finance-analyzer\portfolio\cost_model.py:67:    spread_bps=40.0,    # 0.40% half-spread (conservative)
Q:\finance-analyzer\portfolio\cost_model.py:75:    min_fee_sek=1.0,
Q:\finance-analyzer\portfolio\cost_model.py:76:    spread_bps=5.0,      # 0.05% half-spread (liquid US stocks)
Q:\finance-analyzer\portfolio\cost_model.py:81:# Crypto (Binance-equivalent fees, used for simulated portfolio)
Q:\finance-analyzer\portfolio\cost_model.py:83:    courtage_bps=5.0,     # 0.05% taker fee
Q:\finance-analyzer\portfolio\cost_model.py:84:    min_fee_sek=0.0,
Q:\finance-analyzer\portfolio\cost_model.py:85:    spread_bps=5.0,       # 0.05% half-spread
Q:\finance-analyzer\portfolio\cost_model.py:90:# Elongir silver bot (specific spread from config)
Q:\finance-analyzer\portfolio\cost_model.py:93:    min_fee_sek=0.0,
Q:\finance-analyzer\portfolio\cost_model.py:94:    spread_bps=40.0,      # 0.40% half-spread
Q:\finance-analyzer\portfolio\kelly_metals.py:9:    from portfolio.kelly_metals import recommended_metals_size
Q:\finance-analyzer\portfolio\kelly_metals.py:19:    # rec["kelly_pct"]      -> full Kelly fraction (0-1)
Q:\finance-analyzer\portfolio\kelly_metals.py:20:    # rec["half_kelly_pct"] -> half Kelly (recommended)
Q:\finance-analyzer\portfolio\kelly_metals.py:30:from portfolio.kelly_sizing import kelly_fraction
Q:\finance-analyzer\portfolio\kelly_metals.py:51:    cache = load_json(str(ACCURACY_CACHE), default={})
Q:\finance-analyzer\portfolio\kelly_metals.py:195:        source_parts.append(f"default ({win_rate:.1%})")
Q:\finance-analyzer\portfolio\kelly_metals.py:205:        source_parts.append(f"default W={avg_win:.2f}% L={avg_loss:.2f}%")
Q:\finance-analyzer\portfolio\kelly_metals.py:208:    full_kelly = kelly_fraction(win_rate, avg_win, avg_loss)
Q:\finance-analyzer\portfolio\kelly_metals.py:209:    half_kelly = full_kelly / 2.0
Q:\finance-analyzer\portfolio\kelly_metals.py:214:    # Position fraction = half_kelly / (avg_loss * leverage / 100)
Q:\finance-analyzer\portfolio\kelly_metals.py:217:        position_fraction = half_kelly / cert_loss_frac
Q:\finance-analyzer\portfolio\kelly_metals.py:239:    # --- Step 8: Expected growth rate (daily log-growth at half-Kelly) ---
Q:\finance-analyzer\portfolio\kelly_metals.py:258:        "kelly_pct": round(full_kelly, 4),
Q:\finance-analyzer\portfolio\kelly_metals.py:259:        "half_kelly_pct": round(half_kelly, 4),
Q:\finance-analyzer\portfolio\kelly_metals.py:271:        # Provenance
Q:\finance-analyzer\portfolio\kelly_metals.py:276:def format_kelly_line(rec: dict) -> str:
Q:\finance-analyzer\portfolio\kelly_metals.py:278:    k = rec["half_kelly_pct"] * 100
Q:\finance-analyzer\portfolio\exposure_coach.py:34:# Bias classification
Q:\finance-analyzer\portfolio\exposure_coach.py:62:            "rationale": "No market health data available — using default exposure",
Q:\finance-analyzer\portfolio\exposure_coach.py:88:    # Block new entries only in danger zone with bearish regime
Q:\finance-analyzer\portfolio\portfolio_validator.py:4:holdings, fees, and transaction records are all internally consistent.
Q:\finance-analyzer\portfolio\portfolio_validator.py:8:from collections import defaultdict
Q:\finance-analyzer\portfolio\portfolio_validator.py:17:    all invariants hold. Returns a list of error messages -- an empty
Q:\finance-analyzer\portfolio\portfolio_validator.py:25:    5. Fee reconciliation: total_fees_sek = sum(all fee_sek in transactions)
Q:\finance-analyzer\portfolio\portfolio_validator.py:32:            - cash_sek, holdings, transactions, initial_value_sek, total_fees_sek
Q:\finance-analyzer\portfolio\portfolio_validator.py:39:    # --- Extract fields with defaults ---
Q:\finance-analyzer\portfolio\portfolio_validator.py:44:    total_fees_sek = portfolio.get("total_fees_sek")
Q:\finance-analyzer\portfolio\portfolio_validator.py:50:    if total_fees_sek is None:
Q:\finance-analyzer\portfolio\portfolio_validator.py:51:        errors.append("Missing or null field: total_fees_sek (should be 0 if no fees)")
Q:\finance-analyzer\portfolio\portfolio_validator.py:52:        total_fees_sek = 0
Q:\finance-analyzer\portfolio\portfolio_validator.py:71:    # Note: BUY total_sek = full allocation (including fee)
Q:\finance-analyzer\portfolio\portfolio_validator.py:72:    # SELL total_sek = net proceeds (after fee deducted)
Q:\finance-analyzer\portfolio\portfolio_validator.py:94:    ticker_bought = defaultdict(float)
Q:\finance-analyzer\portfolio\portfolio_validator.py:95:    ticker_sold = defaultdict(float)
Q:\finance-analyzer\portfolio\portfolio_validator.py:116:        # Compare (with tolerance for floating point and rounding from repeated partial sells)
Q:\finance-analyzer\portfolio\portfolio_validator.py:125:            # Tolerance for small remainders from repeated partial sells (e.g.,
Q:\finance-analyzer\portfolio\portfolio_validator.py:148:    computed_fees = 0.0
Q:\finance-analyzer\portfolio\portfolio_validator.py:149:    tx_with_fees = 0
Q:\finance-analyzer\portfolio\portfolio_validator.py:150:    tx_without_fees = 0
Q:\finance-analyzer\portfolio\portfolio_validator.py:152:        fee = tx.get("fee_sek")
Q:\finance-analyzer\portfolio\portfolio_validator.py:153:        if fee is not None:
Q:\finance-analyzer\portfolio\portfolio_validator.py:154:            computed_fees += fee
Q:\finance-analyzer\portfolio\portfolio_validator.py:155:            tx_with_fees += 1
Q:\finance-analyzer\portfolio\portfolio_validator.py:157:            tx_without_fees += 1
Q:\finance-analyzer\portfolio\portfolio_validator.py:159:    if tx_without_fees > 0 and tx_with_fees > 0:
Q:\finance-analyzer\portfolio\portfolio_validator.py:161:            f"Inconsistent fee tracking: {tx_with_fees} transactions have fee_sek, "
Q:\finance-analyzer\portfolio\portfolio_validator.py:162:            f"{tx_without_fees} do not"
Q:\finance-analyzer\portfolio\portfolio_validator.py:165:    # Only compare fees if transactions have fee_sek fields
Q:\finance-analyzer\portfolio\portfolio_validator.py:166:    if tx_with_fees > 0:
Q:\finance-analyzer\portfolio\portfolio_validator.py:167:        fee_diff = abs(computed_fees - total_fees_sek)
Q:\finance-analyzer\portfolio\portfolio_validator.py:168:        if fee_diff > 0.01:  # 0.01 SEK tolerance
Q:\finance-analyzer\portfolio\portfolio_validator.py:170:                f"Fee reconciliation failed: sum of transaction fees = {computed_fees:.2f} SEK, "
Q:\finance-analyzer\portfolio\portfolio_validator.py:171:                f"total_fees_sek = {total_fees_sek:.2f} SEK (diff: {fee_diff:.2f})"
Q:\finance-analyzer\portfolio\portfolio_validator.py:173:    elif len(transactions) > 0 and total_fees_sek == 0:
Q:\finance-analyzer\portfolio\portfolio_validator.py:174:        # Transactions exist but no fee tracking at all -- warn
Q:\finance-analyzer\portfolio\portfolio_validator.py:176:            "No fee_sek fields in any transaction and total_fees_sek is 0, "
Q:\finance-analyzer\portfolio\portfolio_validator.py:261:    # from portfolio_mgr.atomic_write_json: a partial write could be read
Q:\finance-analyzer\portfolio\portfolio_validator.py:264:    # decode failures and uses the same atomic-rename window as the writer.
Q:\finance-analyzer\portfolio\equity_curve.py:10:from collections import defaultdict
Q:\finance-analyzer\portfolio\equity_curve.py:36:        - fx_rate: float
Q:\finance-analyzer\portfolio\equity_curve.py:122:            - max_drawdown_pct: float -- maximum peak-to-trough drawdown
Q:\finance-analyzer\portfolio\equity_curve.py:129:            - days_in_drawdown: int -- number of days below previous peak
Q:\finance-analyzer\portfolio\equity_curve.py:139:        "max_drawdown_pct": 0.0,
Q:\finance-analyzer\portfolio\equity_curve.py:146:        "days_in_drawdown": 0,
Q:\finance-analyzer\portfolio\equity_curve.py:193:    # --- Maximum drawdown ---
Q:\finance-analyzer\portfolio\equity_curve.py:196:    # Group by date for drawdown day counting
Q:\finance-analyzer\portfolio\equity_curve.py:212:    result["max_drawdown_pct"] = round(max_dd, 4)
Q:\finance-analyzer\portfolio\equity_curve.py:213:    result["days_in_drawdown"] = len(date_was_below)
Q:\finance-analyzer\portfolio\equity_curve.py:230:            variance = sum((r - mean_ret) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
Q:\finance-analyzer\portfolio\equity_curve.py:231:            daily_vol = math.sqrt(variance)
Q:\finance-analyzer\portfolio\equity_curve.py:280:        "drawdown_diff_pct": round(patient["max_drawdown_pct"] - bold["max_drawdown_pct"], 4),
Q:\finance-analyzer\portfolio\equity_curve.py:281:        "lower_drawdown": "patient" if patient["max_drawdown_pct"] < bold["max_drawdown_pct"] else "bold",
Q:\finance-analyzer\portfolio\equity_curve.py:334:            - fee_sek: float (total fees for this round trip)
Q:\finance-analyzer\portfolio\equity_curve.py:338:    buy_queues = defaultdict(list)
Q:\finance-analyzer\portfolio\equity_curve.py:344:            fee = tx.get("fee_sek", 0) or 0
Q:\finance-analyzer\portfolio\equity_curve.py:352:                    "fee_sek": fee,
Q:\finance-analyzer\portfolio\equity_curve.py:363:        sell_fee = tx.get("fee_sek", 0) or 0
Q:\finance-analyzer\portfolio\equity_curve.py:392:            # Proportional fees — use original buy quantity as denominator
Q:\finance-analyzer\portfolio\equity_curve.py:393:            # to avoid inflating fee allocation on subsequent partial matches
Q:\finance-analyzer\portfolio\equity_curve.py:396:            # P0-6 (2026-05-02): pnl_sek is now NET of buy+sell fees so
Q:\finance-analyzer\portfolio\equity_curve.py:399:            # only) — used for streaks and expectancy. `fee_sek` field still
Q:\finance-analyzer\portfolio\equity_curve.py:400:            # reports total fees so consumers can reconstruct gross if needed.
Q:\finance-analyzer\portfolio\equity_curve.py:403:            buy_fee_share = (buy["fee_sek"] * matched / buy["original_shares"]) if buy["original_shares"] > 0 else 0
Q:\finance-analyzer\portfolio\equity_curve.py:404:            sell_fee_share = (sell_fee * matched / sell_shares) if sell_shares > 0 else 0
Q:\finance-analyzer\portfolio\equity_curve.py:405:            pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share
Q:\finance-analyzer\portfolio\equity_curve.py:417:                "fee_sek": round(buy_fee_share + sell_fee_share, 2),
Q:\finance-analyzer\portfolio\equity_curve.py:445:            - calmar_ratio: annualized_return / max_drawdown (None if insufficient data)
Q:\finance-analyzer\portfolio\equity_curve.py:534:    # Calmar ratio (annualized return / max drawdown)
Q:\finance-analyzer\portfolio\equity_curve.py:580:            print(f"  Max drawdown:     {m['max_drawdown_pct']:.2f}%")
Q:\finance-analyzer\portfolio\equity_curve.py:588:            print(f"  Days in drawdown: {m['days_in_drawdown']}")
Q:\finance-analyzer\portfolio\equity_curve.py:598:        print(f"  Lower drawdown:   {c['lower_drawdown']} (by {abs(c['drawdown_diff_pct']):.2f}%)")
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:1:"""Portfolio VaR with t-copula correlated simulation.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:3:Computes Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), and
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:4:joint drawdown probabilities for multi-position portfolios using a Student-t
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:12:    from portfolio.monte_carlo_risk import compute_portfolio_var
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:13:    result = compute_portfolio_var(portfolio_state, agent_summary)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:15:    #   "var_95_usd": -1234.56,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:16:    #   "cvar_95_usd": -1567.89,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:17:    #   "var_99_usd": -2345.67,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:30:from portfolio.fx_rates import FX_RATE_FALLBACK
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:31:from portfolio.monte_carlo import (
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:34:    volatility_from_atr,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:37:logger = logging.getLogger("portfolio.monte_carlo_risk")
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:44:# Correlation matrix estimation
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:47:def estimate_correlation_matrix(
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:51:    """Estimate correlation matrix from historical return series.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:53:    For tickers with insufficient data, defaults to zero correlation
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:54:    (independent assumption) which is conservative for VaR.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:58:        tickers: Ordered list of tickers (defines matrix row/column order).
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:61:        Correlation matrix, shape (n, n), guaranteed symmetric and PSD.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:69:    # Build return matrix for tickers with data
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:88:def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:89:    """Project matrix to nearest positive semi-definite correlation matrix.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:94:    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:99:    # Rescale to correlation matrix (diagonal = 1)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:110:# Known correlation pairs (fallback when no historical data)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:128:def build_correlation_matrix(
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:132:    """Build correlation matrix using historical data with prior fallback.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:142:        Correlation matrix (n x n).
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:155:            return estimate_correlation_matrix(historical_returns, tickers)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:176:    then computes portfolio-level P&L, VaR, and CVaR.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:183:        correlation_matrix: Correlation matrix for position tickers.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:193:        correlation_matrix: np.ndarray,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:204:        # If we filtered tickers, extract the sub-matrix
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:205:        if self._n_assets > 0 and correlation_matrix.size > 0:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:208:            self._corr = correlation_matrix[np.ix_(indices, indices)]
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:224:        1. Cholesky decompose correlation matrix: L = chol(R)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:228:        5. Scale to t: T = W * sqrt(df / S) → multivariate t
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:238:        rng = np.random.default_rng(self.seed)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:246:                "Cholesky decomposition failed for %d-asset correlation matrix — "
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:261:        # Step 5: Scale to multivariate t
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:274:        # variance by ~sqrt(df/(df-2)) ≈ sqrt(2) at df=4. The correct
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:329:    def var(self, confidence: float = 0.95) -> float:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:332:        VaR is the loss at the (1-confidence) percentile of the P&L distribution.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:339:            VaR in USD (negative = loss).
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:347:    def cvar(self, confidence: float = 0.95) -> float:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:348:        """Compute Conditional VaR (Expected Shortfall) at given confidence.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:350:        CVaR is the average loss in the worst (1-confidence) scenarios.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:351:        Always worse than VaR (further from zero).
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:357:            CVaR in USD (negative = loss).
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:362:        var_threshold = self.var(confidence)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:363:        tail = self._pnl[self._pnl <= var_threshold]
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:365:            return var_threshold
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:368:    def drawdown_probability(self, threshold_pct: float) -> float:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:392:# Convenience: compute portfolio VaR from system data structures
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:395:def compute_portfolio_var(
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:402:    """Compute portfolio VaR from portfolio state and agent summary.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:409:        agent_summary: Agent summary with signals, fx_rate.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:415:        Dict with VaR metrics in both USD and SEK.
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:419:    fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:436:        atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:437:        vol = volatility_from_atr(atr_pct)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:440:        from portfolio.monte_carlo import _get_directional_probability
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:454:            "var_95_usd": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:455:            "var_99_usd": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:456:            "cvar_95_usd": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:457:            "cvar_99_usd": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:458:            "var_95_sek": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:459:            "var_99_sek": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:460:            "cvar_95_sek": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:461:            "cvar_99_sek": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:465:            "drawdown_1pct_prob": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:466:            "drawdown_5pct_prob": 0.0,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:469:    # Build correlation matrix
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:470:    corr = build_correlation_matrix(tickers, historical_returns)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:475:        correlation_matrix=corr,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:485:    var95 = sim.var(0.95)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:486:    var99 = sim.var(0.99)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:487:    cvar95 = sim.cvar(0.95)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:488:    cvar99 = sim.cvar(0.99)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:491:        "var_95_usd": round(var95, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:492:        "var_99_usd": round(var99, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:493:        "cvar_95_usd": round(cvar95, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:494:        "cvar_99_usd": round(cvar99, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:495:        "var_95_sek": round(var95 * fx_rate, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:496:        "var_99_sek": round(var99 * fx_rate, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:497:        "cvar_95_sek": round(cvar95 * fx_rate, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:498:        "cvar_99_sek": round(cvar99 * fx_rate, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:500:        "total_exposure_sek": round(total_exposure * fx_rate, 2),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:502:        "drawdown_1pct_prob": round(sim.drawdown_probability(1.0), 3),
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:503:        "drawdown_5pct_prob": round(sim.drawdown_probability(5.0), 3),
Q:\finance-analyzer\portfolio\warrant_portfolio.py:11:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\warrant_portfolio.py:48:    atomic_write_json(WARRANT_STATE_FILE, state)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:52:def warrant_pnl(holding, current_underlying_usd, fx_rate):
Q:\finance-analyzer\portfolio\warrant_portfolio.py:66:        fx_rate: current USD/SEK exchange rate.
Q:\finance-analyzer\portfolio\warrant_portfolio.py:80:    if not holding or not current_underlying_usd or not fx_rate:
Q:\finance-analyzer\portfolio\warrant_portfolio.py:116:def get_warrant_summary(prices_usd, fx_rate):
Q:\finance-analyzer\portfolio\warrant_portfolio.py:121:        fx_rate: current USD/SEK exchange rate.
Q:\finance-analyzer\portfolio\warrant_portfolio.py:157:        pnl = warrant_pnl(holding, current_price, fx_rate)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:231:                # spot price, causing the metals stop-loss reference (read
Q:\finance-analyzer\portfolio\warrant_portfolio.py:233:                # trip earlier than the VWAP entry implied — partial-add
Q:\finance-analyzer\portfolio\warrant_portfolio.py:234:                # positions would hard-stop on small underlying drawdowns.
Q:\finance-analyzer\portfolio\monte_carlo.py:4:with antithetic variates for variance reduction. Converts directional
Q:\finance-analyzer\portfolio\monte_carlo.py:5:signal probabilities into price quantile bands, stop-loss probabilities,
Q:\finance-analyzer\portfolio\monte_carlo.py:11:    from portfolio.monte_carlo import simulate_ticker
Q:\finance-analyzer\portfolio\monte_carlo.py:27:logger = logging.getLogger("portfolio.monte_carlo")
Q:\finance-analyzer\portfolio\monte_carlo.py:30:DEFAULT_N_PATHS = 10_000   # 5K pairs with antithetic variates
Q:\finance-analyzer\portfolio\monte_carlo.py:34:# Per-asset-class ATR fallbacks when actual ATR is missing from signals.
Q:\finance-analyzer\portfolio\monte_carlo.py:36:_ATR_DEFAULT_BY_CLASS = {
Q:\finance-analyzer\portfolio\monte_carlo.py:37:    "crypto": 3.5,  # BTC 3-4% typical daily ATR
Q:\finance-analyzer\portfolio\monte_carlo.py:47:def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
Q:\finance-analyzer\portfolio\monte_carlo.py:48:    """Convert ATR% (14-period) to annualized volatility.
Q:\finance-analyzer\portfolio\monte_carlo.py:50:    ATR measures average true range over `period` candles. For hourly candles
Q:\finance-analyzer\portfolio\monte_carlo.py:51:    (our primary timeframe), we annualize: vol = atr_frac * sqrt(trading_periods/period).
Q:\finance-analyzer\portfolio\monte_carlo.py:56:        atr_pct: ATR as percentage of price (e.g., 3.5 means 3.5%).
Q:\finance-analyzer\portfolio\monte_carlo.py:57:        period: ATR lookback period (default 14).
Q:\finance-analyzer\portfolio\monte_carlo.py:62:    atr_frac = atr_pct / 100.0
Q:\finance-analyzer\portfolio\monte_carlo.py:64:    vol = atr_frac * annual_factor
Q:\finance-analyzer\portfolio\monte_carlo.py:100:def _atr_default_for_ticker(ticker: str) -> float:
Q:\finance-analyzer\portfolio\monte_carlo.py:101:    """Return the per-asset-class ATR fallback for a ticker."""
Q:\finance-analyzer\portfolio\monte_carlo.py:104:        return _ATR_DEFAULT_BY_CLASS["crypto"]
Q:\finance-analyzer\portfolio\monte_carlo.py:106:        return _ATR_DEFAULT_BY_CLASS["metals"]
Q:\finance-analyzer\portfolio\monte_carlo.py:107:    return _ATR_DEFAULT_BY_CLASS["stocks"]
Q:\finance-analyzer\portfolio\monte_carlo.py:115:    """Geometric Brownian Motion price path simulator with antithetic variates.
Q:\finance-analyzer\portfolio\monte_carlo.py:118:    Uses antithetic variates for 50-75% variance reduction at zero extra cost.
Q:\finance-analyzer\portfolio\monte_carlo.py:141:        """Generate terminal prices via GBM with antithetic variates.
Q:\finance-analyzer\portfolio\monte_carlo.py:145:        Antithetic variates: for each random draw Z, we also compute the
Q:\finance-analyzer\portfolio\monte_carlo.py:147:        estimates, reducing variance of the mean estimator.
Q:\finance-analyzer\portfolio\monte_carlo.py:152:        rng = np.random.default_rng(self.seed)
Q:\finance-analyzer\portfolio\monte_carlo.py:158:        # Draw half the paths, use antithetic for the other half
Q:\finance-analyzer\portfolio\monte_carlo.py:159:        n_half = self.n_paths // 2
Q:\finance-analyzer\portfolio\monte_carlo.py:160:        Z = rng.standard_normal(n_half)
Q:\finance-analyzer\portfolio\monte_carlo.py:193:            percentiles: List of percentiles to compute (default: [5, 25, 50, 75, 95]).
Q:\finance-analyzer\portfolio\monte_carlo.py:208:        Useful for stop-loss probability: "what's the chance price hits my stop?"
Q:\finance-analyzer\portfolio\monte_carlo.py:272:    Extracts price, ATR volatility, and directional probability from the
Q:\finance-analyzer\portfolio\monte_carlo.py:279:        horizons: List of horizon days (default: [1, 3]).
Q:\finance-analyzer\portfolio\monte_carlo.py:299:    atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct") or _atr_default_for_ticker(ticker)
Q:\finance-analyzer\portfolio\monte_carlo.py:301:    # Get volatility from ATR
Q:\finance-analyzer\portfolio\monte_carlo.py:302:    vol = volatility_from_atr(atr_pct)
Q:\finance-analyzer\portfolio\monte_carlo.py:309:    # Compute ATR-based stop level (2x ATR below entry/current)
Q:\finance-analyzer\portfolio\monte_carlo.py:310:    stop_price = price * (1 - 2 * atr_pct / 100)
Q:\finance-analyzer\portfolio\monte_carlo.py:333:    result["atr_pct"] = round(atr_pct, 2)
Q:\finance-analyzer\portfolio\exit_optimizer.py:53:    atr_pct: float | None = None             # ATR% for vol estimation
Q:\finance-analyzer\portfolio\exit_optimizer.py:72:    financing_level: float | None = None  # MINI future financing level (USD)
Q:\finance-analyzer\portfolio\exit_optimizer.py:86:        ev_sek: Expected value = fill_prob × pnl + (1-fill_prob) × fallback.
Q:\finance-analyzer\portfolio\exit_optimizer.py:116:        provenance: Audit trail (model version, parameters, data sources).
Q:\finance-analyzer\portfolio\exit_optimizer.py:124:    session_max_distribution: dict[str, float] = field(default_factory=dict)
Q:\finance-analyzer\portfolio\exit_optimizer.py:125:    session_min_distribution: dict[str, float] = field(default_factory=dict)
Q:\finance-analyzer\portfolio\exit_optimizer.py:127:    provenance: dict[str, Any] = field(default_factory=dict)
Q:\finance-analyzer\portfolio\exit_optimizer.py:181:    if market.atr_pct and market.atr_pct > 0:
Q:\finance-analyzer\portfolio\exit_optimizer.py:182:        # Convert ATR% (14-period) to annualized vol
Q:\finance-analyzer\portfolio\exit_optimizer.py:183:        atr_frac = market.atr_pct / 100.0
Q:\finance-analyzer\portfolio\exit_optimizer.py:184:        return max(atr_frac * math.sqrt(252.0 / 14), _MIN_VOLATILITY)
Q:\finance-analyzer\portfolio\exit_optimizer.py:199:    Uses antithetic variates for variance reduction (~50% lower variance).
Q:\finance-analyzer\portfolio\exit_optimizer.py:214:    rng = np.random.default_rng(seed)
Q:\finance-analyzer\portfolio\exit_optimizer.py:224:    # Antithetic variates: generate half, mirror the other half
Q:\finance-analyzer\portfolio\exit_optimizer.py:225:    n_half = n_paths // 2
Q:\finance-analyzer\portfolio\exit_optimizer.py:226:    Z = rng.standard_normal((n_half, n_steps))
Q:\finance-analyzer\portfolio\exit_optimizer.py:312:        warrant_value = (underlying - financing_level) × usdsek
Q:\finance-analyzer\portfolio\exit_optimizer.py:318:    fx = market.usdsek
Q:\finance-analyzer\portfolio\exit_optimizer.py:320:    if position.instrument_type == "warrant" and position.financing_level is not None:
Q:\finance-analyzer\portfolio\exit_optimizer.py:321:        # MINI future: warrant price = (underlying - financing_level) × fx
Q:\finance-analyzer\portfolio\exit_optimizer.py:322:        exit_warrant_sek = (exit_price_usd - position.financing_level) * fx
Q:\finance-analyzer\portfolio\exit_optimizer.py:327:        # Leveraged product without explicit financing level
Q:\finance-analyzer\portfolio\exit_optimizer.py:336:        exit_value = position.qty * exit_price_usd * fx
Q:\finance-analyzer\portfolio\exit_optimizer.py:337:        entry_value = position.qty * position.entry_underlying_usd * fx
Q:\finance-analyzer\portfolio\exit_optimizer.py:373:    if position.financing_level and position.financing_level > 0:
Q:\finance-analyzer\portfolio\exit_optimizer.py:374:        distance_pct = (market.price - position.financing_level) / market.price * 100
Q:\finance-analyzer\portfolio\exit_optimizer.py:396:    if session_min is not None and position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:397:        stop_buffer = position.financing_level * 1.03  # 3% above financing
Q:\finance-analyzer\portfolio\exit_optimizer.py:419:    - Knock-out danger (< 3% from financing level)
Q:\finance-analyzer\portfolio\exit_optimizer.py:431:    if position.financing_level and position.financing_level > 0:
Q:\finance-analyzer\portfolio\exit_optimizer.py:432:        distance_pct = (market.price - position.financing_level) / market.price * 100
Q:\finance-analyzer\portfolio\exit_optimizer.py:434:            logger.warning("RISK OVERRIDE: Knock-out danger (%.1f%% from barrier), "
Q:\finance-analyzer\portfolio\exit_optimizer.py:445:    if session_min is not None and position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:446:        stop_buffer = position.financing_level * 1.03
Q:\finance-analyzer\portfolio\exit_optimizer.py:533:            provenance={"reason": "session_ended"},
Q:\finance-analyzer\portfolio\exit_optimizer.py:562:    fallback_pnl = _compute_pnl_sek(position, median_terminal, market, costs)
Q:\finance-analyzer\portfolio\exit_optimizer.py:582:        # Expected value: fill_prob × conditional_pnl + (1-fill_prob) × fallback
Q:\finance-analyzer\portfolio\exit_optimizer.py:583:        ev = fill_prob * pnl + (1 - fill_prob) * fallback_pnl
Q:\finance-analyzer\portfolio\exit_optimizer.py:628:        pnl_sek=round(fallback_pnl, 2),
Q:\finance-analyzer\portfolio\exit_optimizer.py:630:        pnl_pct=round(_pnl_pct(fallback_pnl, position), 2),
Q:\finance-analyzer\portfolio\exit_optimizer.py:643:    elif position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:645:        stop_buffer = position.financing_level * 1.03
Q:\finance-analyzer\portfolio\exit_optimizer.py:663:        provenance={
Q:\finance-analyzer\portfolio\exit_optimizer.py:686:    financing_level: float | None = None,
Q:\finance-analyzer\portfolio\exit_optimizer.py:701:        financing_level: For MINI futures, the knock-out level.
Q:\finance-analyzer\portfolio\exit_optimizer.py:718:    atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct")
Q:\finance-analyzer\portfolio\exit_optimizer.py:719:    fx_rate = agent_summary.get("fx_rate", 10.85)
Q:\finance-analyzer\portfolio\exit_optimizer.py:725:        atr_pct=atr_pct,
Q:\finance-analyzer\portfolio\exit_optimizer.py:726:        usdsek=fx_rate,
Q:\finance-analyzer\portfolio\exit_optimizer.py:749:        financing_level=financing_level,
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:1:"""Strategy orchestrator — daemon thread managing multiple strategy plugins."""
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:5:import threading
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:18:    """Manages strategy plugins in a daemon thread.
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:21:    strategy do not affect others. The thread stops cleanly on stop().
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:36:        self._thread: threading.Thread | None = None
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:40:        """Start the orchestrator daemon thread."""
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:41:        if self._thread is not None and self._thread.is_alive():
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:44:        self._thread = threading.Thread(
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:49:        self._thread.start()
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:56:        """Signal the thread to stop and wait for it."""
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:58:        if self._thread is not None:
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:59:            self._thread.join(timeout=5.0)
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:60:            self._thread = None
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:76:            now = time.monotonic()
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:91:                    self._last_tick[name] = time.monotonic()
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:3:Builds MarketSnapshot from SharedData + lightweight Binance fetch.
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:13:from portfolio.file_utils import atomic_write_json, load_json
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:34:    - Gold spot: fetched from Binance FAPI at each tick (5s freshness)
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:74:        self._shared = shared  # store ref for trade queue lock
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:75:        gold = fetch_gold_price(self._cfg.binance_gold_symbol)
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:91:        # Fetch own FX rate (cached in fx_rates module, ~daily refresh)
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:92:        fx = fetch_usdsek()
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:93:        if fx is None or fx <= 0:
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:94:            fx = shared.fx_rate if shared.fx_rate > 0 else 10.5
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:101:                yfinance_ticker=self._cfg.rates_proxy_ticker,
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:114:                block_types=self._cfg.event_risk_block_types,
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:123:            usdsek=fx,
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:133:            fx_fetch_ts=now,
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:140:                snap.cert_spread_pct = (snap.cert_ask - snap.cert_bid) / snap.cert_bid
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:147:        Uses shared.trade_queue_lock to prevent race with metals loop's
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:148:        process_trade_queue() running on the main thread.
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:150:        lock = shared.trade_queue_lock if shared else None
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:151:        if lock:
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:152:            lock.acquire()
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:154:            self._enqueue_trade_locked(action)
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:156:            if lock:
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:157:                lock.release()
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:159:    def _enqueue_trade_locked(self, action: dict) -> None:
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:160:        """Enqueue while holding the trade queue lock."""
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:161:        queue = load_json(self._trade_queue_file, default=None)
Q:\finance-analyzer\portfolio\strategies\golddigger_strategy.py:183:        atomic_write_json(self._trade_queue_file, queue)
Q:\finance-analyzer\portfolio\strategies\elongir_strategy.py:24:    - Klines (1m/5m/15m): fetched from Binance FAPI at each tick
Q:\finance-analyzer\portfolio\strategies\elongir_strategy.py:75:        # Fetch own FX rate (cached in fx_rates module)
Q:\finance-analyzer\portfolio\strategies\elongir_strategy.py:76:        fx = fetch_usdsek()
Q:\finance-analyzer\portfolio\strategies\elongir_strategy.py:77:        if fx is None or fx <= 0:
Q:\finance-analyzer\portfolio\strategies\elongir_strategy.py:78:            fx = shared.fx_rate if shared.fx_rate > 0 else 10.5
Q:\finance-analyzer\portfolio\strategies\elongir_strategy.py:81:            fx_rate=fx,
Q:\finance-analyzer\portfolio\strategies\base.py:5:from threading import Lock
Q:\finance-analyzer\portfolio\strategies\base.py:12:    Updated by metals loop main thread, read by orchestrator thread.
Q:\finance-analyzer\portfolio\strategies\base.py:16:    underlying_prices: dict[str, float] = field(default_factory=dict)
Q:\finance-analyzer\portfolio\strategies\base.py:17:    fx_rate: float = 0.0
Q:\finance-analyzer\portfolio\strategies\base.py:18:    cert_prices: dict[str, dict] = field(default_factory=dict)
Q:\finance-analyzer\portfolio\strategies\base.py:20:    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
Q:\finance-analyzer\portfolio\strategies\base.py:21:    trade_queue_lock: Lock = field(default_factory=Lock)
Q:\finance-analyzer\portfolio\strategies\base.py:48:        Must not call Playwright or block for more than a few seconds.
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\trade_risk_classifier.py' | Select-Object -First 200" in Q:\fa-adv-2026-05-11
 succeeded in 649ms:
"""Trade Risk Classifier - scores proposed trades as LOW / MEDIUM / HIGH risk.

Scoring system (0-11 total points):
  Position size risk   (0-3): >20% = 3, >10% = 2, >5% = 1
  Regime risk          (0-3): trending-up=0, trending-down=1, ranging=2, high-vol=3, capitulation=3
  Counter-trend risk   (0-2): BUY in trending-down or SELL in trending-up = 2
  Weak consensus       (0-2): <60% = 2, <70% = 1
  Low confidence       (0-1): <50% = 1
  Concentration risk   (0-2): total exposure >40% = 2, >25% = 1

Thresholds: 0-3 = LOW, 4-6 = MEDIUM, 7+ = HIGH.
HOLD is always LOW (score 0, no factors).
"""

import logging

logger = logging.getLogger("portfolio.trade_risk_classifier")

# Regime risk mapping
_REGIME_SCORES = {
    "trending-up": 0,
    "trending-down": 1,
    "ranging": 2,
    "high-vol": 3,
    "capitulation": 3,
}


def classify_trade_risk(
    action: str,
    confidence: float,
    position_pct: float,
    regime: str,
    consensus_ratio: float,
    existing_exposure_pct: float = 0.0,
) -> dict:
    """Classify a proposed trade into LOW / MEDIUM / HIGH risk.

    Parameters
    ----------
    action : str
        Trade action: "BUY", "SELL", or "HOLD".
    confidence : float
        Signal confidence (0.0-1.0).
    position_pct : float
        Proposed position size as percentage of portfolio (0-100).
    regime : str
        Current market regime (trending-up, trending-down, ranging, high-vol, capitulation).
    consensus_ratio : float
        Signal consensus ratio (0.0-1.0). Fraction of voters that agree.
    existing_exposure_pct : float
        Current total portfolio exposure percentage (0-100).

    Returns
    -------
    dict
        {"level": "LOW"/"MEDIUM"/"HIGH", "score": int, "factors": list[str]}
    """
    action_upper = action.upper()

    # HOLD is always LOW risk
    if action_upper == "HOLD":
        return {"level": "LOW", "score": 0, "factors": []}

    score = 0
    factors = []

    # 1. Position size risk (0-3)
    if position_pct > 20:
        score += 3
        factors.append(f"large position ({position_pct:.1f}% > 20%)")
    elif position_pct > 10:
        score += 2
        factors.append(f"medium position ({position_pct:.1f}% > 10%)")
    elif position_pct > 5:
        score += 1
        factors.append(f"notable position ({position_pct:.1f}% > 5%)")

    # 2. Regime risk (0-3)
    regime_lower = regime.lower()
    regime_score = _REGIME_SCORES.get(regime_lower, 0)
    if regime_score > 0:
        score += regime_score
        factors.append(f"regime={regime_lower} (+{regime_score})")

    # 3. Counter-trend risk (0-2)
    if (action_upper == "BUY" and regime_lower == "trending-down") or \
       (action_upper == "SELL" and regime_lower == "trending-up"):
        score += 2
        factors.append(f"counter-trend {action_upper} in {regime_lower}")

    # 4. Weak consensus (0-2)
    if consensus_ratio < 0.60:
        score += 2
        factors.append(f"weak consensus ({consensus_ratio:.0%} < 60%)")
    elif consensus_ratio < 0.70:
        score += 1
        factors.append(f"moderate consensus ({consensus_ratio:.0%} < 70%)")

    # 5. Low confidence (0-1)
    if confidence < 0.50:
        score += 1
        factors.append(f"low confidence ({confidence:.0%} < 50%)")

    # 6. Concentration risk (0-2)
    if existing_exposure_pct > 40:
        score += 2
        factors.append(f"high concentration ({existing_exposure_pct:.1f}% > 40%)")
    elif existing_exposure_pct > 25:
        score += 1
        factors.append(f"moderate concentration ({existing_exposure_pct:.1f}% > 25%)")

    # Determine level
    if score <= 3:
        level = "LOW"
    elif score <= 6:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {"level": level, "score": score, "factors": factors}
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\trade_guards.py'" in Q:\fa-adv-2026-05-11
 succeeded in 666ms:
"""Trade guards - overtrading prevention for the trading agent.

Three guards:
1. Per-ticker cooldown: No re-trade on same ticker within N minutes.
2. Consecutive-loss escalation: After losses, increase cooldown multiplier.
3. Position rate limit: Max N new positions per time window.

State is persisted to data/trade_guard_state.json.
"""

import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.trade_guards")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "trade_guard_state.json"

# Defaults
DEFAULT_TICKER_COOLDOWN_MINUTES = 30
DEFAULT_BOLD_POSITION_LIMIT = 1       # max new positions per window
DEFAULT_BOLD_POSITION_WINDOW_H = 4    # hours
DEFAULT_PATIENT_POSITION_LIMIT = 1
DEFAULT_PATIENT_POSITION_WINDOW_H = 8
LOSS_ESCALATION = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8}  # consecutive_losses -> cooldown multiplier
LOSS_DECAY_HOURS = 24  # halve escalation multiplier every N hours without a trade

_state_lock = threading.Lock()


def _load_state():
    """Load trade guard state from disk."""
    return load_json(str(STATE_FILE), default={
        "ticker_trades": {},
        "consecutive_losses": {"patient": 0, "bold": 0},
        "last_loss_ts": {"patient": None, "bold": None},
        "new_position_timestamps": {"patient": [], "bold": []},
    })


def _save_state(state):
    """Persist trade guard state to disk."""
    atomic_write_json(STATE_FILE, state)


def _portfolios_have_transactions():
    """Return True if any portfolio file has at least one recorded transaction.

    Used by the C4 sanity check to distinguish "no trades happened yet"
    (quiet startup state) from "trades happened but weren't recorded"
    (broken wiring - real bug).

    2026-04-22 follow-up: include warrants portfolio - CLAUDE.md lists it as
    an independent strategy state file, and warrants-only activity would
    have left C4 silent forever.
    """
    for pf_name in (
        "portfolio_state.json",
        "portfolio_state_bold.json",
        "portfolio_state_warrants.json",
    ):
        pf = load_json(str(DATA_DIR / pf_name), default={})
        if pf and pf.get("transactions"):
            return True
    return False


def _get_cooldown_multiplier(consecutive_losses, last_loss_ts_str=None):
    """Get cooldown multiplier based on consecutive loss count with time decay.

    After LOSS_DECAY_HOURS without a new trade, the multiplier halves
    repeatedly (geometric decay). E.g. 8x  4x after 24h  2x after 48h  1x.
    """
    if consecutive_losses >= 4:
        base = LOSS_ESCALATION[4]
    else:
        base = LOSS_ESCALATION.get(consecutive_losses, 1)

    if base <= 1 or not last_loss_ts_str:
        return base

    # Apply time-based decay
    try:
        last_loss = datetime.fromisoformat(
            last_loss_ts_str.replace("Z", "+00:00")
        )
        if last_loss.tzinfo is None:
            last_loss = last_loss.replace(tzinfo=UTC)
        elapsed_hours = (datetime.now(UTC) - last_loss).total_seconds() / 3600
        if elapsed_hours > LOSS_DECAY_HOURS:
            halvings = int(elapsed_hours // LOSS_DECAY_HOURS)
            base = max(1, base >> halvings)  # bit-shift right = halve
    except (ValueError, TypeError, OverflowError):
        pass

    return base


def check_overtrading_guards(ticker, action, strategy, portfolio, config=None):
    """Check all trade guards for a proposed trade.

    Args:
        ticker: Instrument ticker (e.g., "BTC-USD").
        action: "BUY" or "SELL".
        strategy: "patient" or "bold".
        portfolio: Portfolio state dict.
        config: Optional config dict with trade_guards settings.

    Returns:
        list of warning dicts, each with:
            - guard: str (guard name)
            - severity: "warning" or "block"
            - message: str
            - details: dict (guard-specific data)
        Empty list means all guards pass.
    """
    cfg = (config or {}).get("trade_guards", {})
    if cfg.get("enabled") is False:
        return []

    warnings = []
    with _state_lock:
        state = _load_state()
    now = datetime.now(UTC)

    # --- Guard 1: Per-ticker cooldown ---
    base_cooldown = cfg.get("ticker_cooldown_minutes", DEFAULT_TICKER_COOLDOWN_MINUTES)
    consecutive = state.get("consecutive_losses", {}).get(strategy, 0)
    last_loss_ts = state.get("last_loss_ts", {}).get(strategy)
    multiplier = _get_cooldown_multiplier(consecutive, last_loss_ts)
    effective_cooldown = base_cooldown * multiplier

    key = f"{strategy}:{ticker}"
    ticker_trades = state.get("ticker_trades", {})
    last_trade_str = ticker_trades.get(key)
    if last_trade_str:
        try:
            last_trade = datetime.fromisoformat(last_trade_str)
            # M8: ensure aware datetime before comparison with aware now
            if last_trade.tzinfo is None:
                last_trade = last_trade.replace(tzinfo=UTC)
            elapsed = (now - last_trade).total_seconds() / 60
            if elapsed < effective_cooldown:
                remaining = effective_cooldown - elapsed
                warnings.append({
                    "guard": "ticker_cooldown",
                    "severity": "block",
                    "message": (
                        f"{ticker} traded {elapsed:.0f}m ago by {strategy}. "
                        f"Cooldown: {effective_cooldown:.0f}m (base {base_cooldown}m x {multiplier}x). "
                        f"{remaining:.0f}m remaining."
                    ),
                    "details": {
                        "ticker": ticker,
                        "strategy": strategy,
                        "elapsed_min": round(elapsed, 1),
                        "cooldown_min": effective_cooldown,
                        "multiplier": multiplier,
                        "remaining_min": round(remaining, 1),
                    },
                })
        except (ValueError, TypeError):
            pass

    # --- Guard 2: Consecutive-loss escalation (informational) ---
    if consecutive >= 2:
        base_mult = _get_cooldown_multiplier(consecutive, None)
        warnings.append({
            "guard": "consecutive_losses",
            "severity": "warning",
            "message": (
                f"{strategy}: {consecutive} consecutive losses. "
                f"Cooldown multiplier: {multiplier}x"
                f"{f' (decayed from {base_mult}x)' if multiplier < base_mult else ''}."
            ),
            "details": {
                "strategy": strategy,
                "consecutive_losses": consecutive,
                "multiplier": multiplier,
                "base_multiplier": base_mult,
                "decayed": multiplier < base_mult,
            },
        })

    # --- Guard 3: Position rate limit (BUY only) ---
    if action == "BUY":
        is_bold = strategy == "bold"
        limit = cfg.get(
            f"{'bold' if is_bold else 'patient'}_position_limit",
            DEFAULT_BOLD_POSITION_LIMIT if is_bold else DEFAULT_PATIENT_POSITION_LIMIT,
        )
        window_h = cfg.get(
            f"{'bold' if is_bold else 'patient'}_position_window_h",
            DEFAULT_BOLD_POSITION_WINDOW_H if is_bold else DEFAULT_PATIENT_POSITION_WINDOW_H,
        )
        cutoff = now - timedelta(hours=window_h)

        timestamps = state.get("new_position_timestamps", {}).get(strategy, [])
        recent = []
        for ts_str in timestamps:
            try:
                ts = datetime.fromisoformat(ts_str)
                # M8: ensure aware datetime before comparison with aware cutoff
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= cutoff:
                    recent.append(ts)
            except (ValueError, TypeError):
                continue

        if len(recent) >= limit:
            warnings.append({
                "guard": "position_rate_limit",
                "severity": "block",
                "message": (
                    f"{strategy}: {len(recent)} new position(s) in last {window_h}h "
                    f"(limit: {limit})."
                ),
                "details": {
                    "strategy": strategy,
                    "recent_count": len(recent),
                    "limit": limit,
                    "window_hours": window_h,
                },
            })

    return warnings


_wiring_confirmed = False  # process-scoped flag - positive proof for C4


def record_trade(ticker, direction, strategy, pnl_pct=None, config=None):
    """Record a completed trade for guard tracking.

    Call this after executing a trade to update cooldowns and loss streaks.

    Args:
        ticker: Instrument ticker.
        direction: "BUY" or "SELL".
        strategy: "patient" or "bold".
        pnl_pct: Realized P&L percentage (for SELL trades). None for BUY.
        config: Optional config dict.
    """
    # 2026-04-22 follow-up: positive-proof wiring check. The previous C4
    # warning was *reactive* - it could only tell you after a trade had
    # already slipped through unguarded. Log INFO once per process the first
    # time this function fires, so operators get explicit confirmation the
    # BUG-219/PR-R4-4 wiring is alive rather than having to infer it from
    # absence-of-warnings.
    global _wiring_confirmed
    if not _wiring_confirmed:
        logger.info(
            "C4: record_trade() wiring confirmed - first call this process "
            "(ticker=%s direction=%s strategy=%s)",
            ticker, direction, strategy,
        )
        _wiring_confirmed = True

    with _state_lock:
        state = _load_state()
        now = datetime.now(UTC)
        now_str = now.isoformat()

        # Update ticker trade timestamp
        key = f"{strategy}:{ticker}"
        if "ticker_trades" not in state:
            state["ticker_trades"] = {}
        state["ticker_trades"][key] = now_str

        # Update consecutive losses on SELL
        if direction == "SELL" and pnl_pct is not None:
            if "consecutive_losses" not in state:
                state["consecutive_losses"] = {"patient": 0, "bold": 0}
            if "last_loss_ts" not in state:
                state["last_loss_ts"] = {"patient": None, "bold": None}
            if pnl_pct < 0:
                state["consecutive_losses"][strategy] = (
                    state["consecutive_losses"].get(strategy, 0) + 1
                )
                state["last_loss_ts"][strategy] = now_str
            else:
                state["consecutive_losses"][strategy] = 0
                state["last_loss_ts"][strategy] = None

        # Track new position timestamps (BUY only)
        if direction == "BUY":
            if "new_position_timestamps" not in state:
                state["new_position_timestamps"] = {"patient": [], "bold": []}
            if strategy not in state["new_position_timestamps"]:
                state["new_position_timestamps"][strategy] = []
            state["new_position_timestamps"][strategy].append(now_str)

            # Prune old timestamps (keep last 24h).
            cutoff = now - timedelta(hours=24)
            pruned = []
            for ts in state["new_position_timestamps"][strategy]:
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    if dt >= cutoff:
                        pruned.append(ts)
                except (ValueError, TypeError):
                    continue
            state["new_position_timestamps"][strategy] = pruned

        _save_state(state)


def get_all_guard_warnings(signals, patient_pf, bold_pf, config=None):
    """Get trade guard warnings for all tickers with BUY/SELL signals.

    Args:
        signals: Dict of ticker -> signal data (from agent_summary).
        patient_pf: Patient portfolio state dict.
        bold_pf: Bold portfolio state dict.
        config: Optional config dict.

    Returns:
        dict with:
            - warnings: list of warning dicts
            - summary: str (human-readable summary)
    """
    cfg = (config or {}).get("trade_guards", {})
    if cfg.get("enabled") is False:
        return {"warnings": [], "summary": "Trade guards disabled"}

    all_warnings = []

    for ticker, sig in signals.items():
        action = sig.get("action", "HOLD")
        if action == "HOLD":
            continue

        for strategy, portfolio in [("patient", patient_pf), ("bold", bold_pf)]:
            warns = check_overtrading_guards(
                ticker, action, strategy, portfolio, config
            )
            all_warnings.extend(warns)

    summary_parts = []
    if all_warnings:
        by_guard = {}
        for w in all_warnings:
            guard = w["guard"]
            by_guard.setdefault(guard, []).append(w)
        for guard, warns in by_guard.items():
            summary_parts.append(f"{guard}: {len(warns)} warning(s)")

    # C4: Detect broken record_trade() wiring.
    # 2026-04-22: original check fired every cycle whenever state was empty,
    # even when no trades had happened yet (portfolios untouched) - noisy and
    # misleading post-BUG-219/PR-R4-4 which wired _record_new_trades().
    # Now only warn when portfolios DO have transactions but guard state is
    # still empty - that's the real signal the wiring is broken.
    with _state_lock:
        state = _load_state()
    if not state.get("ticker_trades") and all_warnings == [] and _portfolios_have_transactions():
        logger.warning(
            "C4: portfolios have transactions but trade_guard_state.json "
            "has no recorded trades - record_trade() wiring appears broken. "
            "Overtrading guards are NON-FUNCTIONAL."
        )

    return {
        "warnings": all_warnings,
        "summary": "; ".join(summary_parts) if summary_parts else "All clear",
    }


def should_block_trade(guard_result):
    """Check if any guard warning has 'block' severity.

    ARCH-29: Convenience function for Layer 2 go/no-go decisions.

    Args:
        guard_result: Return value from get_all_guard_warnings().

    Returns:
        True if any warning has severity="block", False otherwise.
    """
    warnings = guard_result.get("warnings", [])
    return any(w.get("severity") == "block" for w in warnings)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\trade_validation.py'" in Q:\fa-adv-2026-05-11
 succeeded in 784ms:
"""Pre-trade validation -- sanity checks before order placement.

Validates trade parameters (price, volume, spread, cash, position size) before
any order is placed. Returns a ValidationResult with pass/fail, reason, and
optional warnings for near-limit conditions.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("portfolio.trade_validation")


@dataclass
class ValidationResult:
    """Result of pre-trade validation."""
    valid: bool
    reason: str = ""
    warnings: list = field(default_factory=list)


def validate_trade(
    action: str,
    price: float,
    volume: float,
    cash_available: float,
    bid: float | None = None,
    ask: float | None = None,
    last_known_price: float | None = None,
    max_spread_pct: float = 2.0,
    max_cash_pct: float = 50.0,
    min_order_sek: float = 500.0,
    max_price_deviation_pct: float = 5.0,
) -> ValidationResult:
    """Validate a trade before placing it.

    Checks (in order):
    1. Basic parameter validation (positive price, volume, valid action)
    2. Minimum order size
    3. Cash sufficiency (BUY only)
    4. Position size as % of cash (BUY only)
    5. Bid/ask spread width
    6. Price sanity vs last known price

    Returns ValidationResult with valid=True/False, reason, and warnings.
    """
    warnings: list[str] = []

    # --- 1. Basic parameter validation ---
    if action not in ("BUY", "SELL"):
        return ValidationResult(False, f"Invalid action: {action}")
    if price <= 0:
        return ValidationResult(False, f"Invalid price: {price}")
    if volume <= 0:
        return ValidationResult(False, f"Invalid volume: {volume}")

    order_value = price * volume

    # --- 2. Minimum order size ---
    if order_value < min_order_sek:
        return ValidationResult(
            False,
            f"Order value {order_value:.0f} SEK below minimum {min_order_sek:.0f} SEK",
        )

    # --- 3 & 4. BUY-specific checks ---
    if action == "BUY":
        # Cash sufficiency
        if order_value > cash_available:
            return ValidationResult(
                False,
                f"Insufficient cash: need {order_value:.0f} SEK, have {cash_available:.0f} SEK",
            )
        # Position size limit
        if cash_available > 0:
            cash_pct = (order_value / cash_available) * 100
            if cash_pct > max_cash_pct:
                return ValidationResult(
                    False,
                    f"Position too large: {cash_pct:.1f}% of cash (max {max_cash_pct:.1f}%)",
                )

    # --- 5. Bid/ask spread check ---
    if bid is not None and ask is not None and bid > 0:
        spread_pct = ((ask - bid) / bid) * 100
        if spread_pct > max_spread_pct:
            return ValidationResult(
                False,
                f"Spread too wide: {spread_pct:.2f}% (max {max_spread_pct:.1f}%)",
            )
        if spread_pct > max_spread_pct * 0.7:
            warnings.append(f"Spread warning: {spread_pct:.2f}% approaching limit")

    # --- 6. Price sanity vs last known ---
    if last_known_price is not None and last_known_price > 0:
        deviation_pct = abs(price - last_known_price) / last_known_price * 100
        if deviation_pct > max_price_deviation_pct:
            return ValidationResult(
                False,
                f"Price deviation {deviation_pct:.2f}% from last known "
                f"{last_known_price:.2f} (max {max_price_deviation_pct:.1f}%)",
            )
        if deviation_pct > max_price_deviation_pct * 0.7:
            warnings.append(f"Price moved {deviation_pct:.2f}% from last known")

    logger.debug(
        "Trade validated: %s %.4f @ %.2f SEK (value %.0f SEK)%s",
        action,
        volume,
        price,
        order_value,
        f" -- warnings: {warnings}" if warnings else "",
    )
    return ValidationResult(True, "All checks passed", warnings)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\portfolio_validator.py'" in Q:\fa-adv-2026-05-11
 succeeded in 615ms:
"""Portfolio state validation utilities.

Validates portfolio state files for data integrity, ensuring that cash,
holdings, fees, and transaction records are all internally consistent.
"""

import pathlib
from collections import defaultdict

from portfolio.file_utils import load_json


def validate_portfolio(portfolio: dict) -> list[str]:
    """Validate portfolio state integrity.

    Performs comprehensive checks on a portfolio state dict to ensure
    all invariants hold. Returns a list of error messages -- an empty
    list means the portfolio is valid.

    Checks:
    1. Cash is non-negative
    2. All shares are non-negative
    3. Cash reconciliation: initial - sum(BUY allocs) + sum(SELL net_proceeds) = cash_sek
    4. Holdings shares reconciliation: sum(BUY shares) - sum(SELL shares) per ticker
    5. Fee reconciliation: total_fees_sek = sum(all fee_sek in transactions)
    6. No duplicate timestamps in transactions (same ticker + same timestamp)
    7. Transaction field completeness
    8. Holdings avg_cost_usd consistency

    Args:
        portfolio: Full portfolio state dict with keys:
            - cash_sek, holdings, transactions, initial_value_sek, total_fees_sek

    Returns:
        list of error message strings (empty = valid).
    """
    errors = []

    # --- Extract fields with defaults ---
    cash_sek = portfolio.get("cash_sek")
    holdings = portfolio.get("holdings", {})
    transactions = portfolio.get("transactions", [])
    initial_value = portfolio.get("initial_value_sek", 500_000)
    total_fees_sek = portfolio.get("total_fees_sek")

    # --- Check 0: Required fields exist ---
    if cash_sek is None:
        errors.append("Missing required field: cash_sek")
        cash_sek = 0
    if total_fees_sek is None:
        errors.append("Missing or null field: total_fees_sek (should be 0 if no fees)")
        total_fees_sek = 0
    if "initial_value_sek" not in portfolio:
        errors.append("Missing required field: initial_value_sek")

    # --- Check 1: Cash is non-negative ---
    if cash_sek < 0:
        errors.append(f"Cash is negative: {cash_sek:.2f} SEK")

    # --- Check 2: All shares are non-negative ---
    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares < 0:
            errors.append(f"Negative shares for {ticker}: {shares}")
        avg_cost = pos.get("avg_cost_usd")
        if avg_cost is not None and avg_cost < 0:
            errors.append(f"Negative avg_cost_usd for {ticker}: {avg_cost}")

    # --- Check 3: Cash reconciliation ---
    # cash = initial - sum(BUY total_sek) + sum(SELL total_sek)
    # Note: BUY total_sek = full allocation (including fee)
    # SELL total_sek = net proceeds (after fee deducted)
    total_buy_alloc = 0.0
    total_sell_proceeds = 0.0
    for tx in transactions:
        action = tx.get("action", "")
        total_sek = tx.get("total_sek", 0) or 0
        if action == "BUY":
            total_buy_alloc += total_sek
        elif action == "SELL":
            total_sell_proceeds += total_sek

    expected_cash = initial_value - total_buy_alloc + total_sell_proceeds
    cash_diff = abs(expected_cash - cash_sek)
    if cash_diff > 1.0:  # Allow 1 SEK tolerance for floating point
        errors.append(
            f"Cash reconciliation failed: expected {expected_cash:.2f} SEK "
            f"(initial {initial_value} - buys {total_buy_alloc:.2f} + sells {total_sell_proceeds:.2f}), "
            f"got {cash_sek:.2f} SEK (diff: {cash_diff:.2f})"
        )

    # --- Check 4: Holdings shares reconciliation ---
    # For each ticker: net_shares = sum(BUY shares) - sum(SELL shares)
    ticker_bought = defaultdict(float)
    ticker_sold = defaultdict(float)
    for tx in transactions:
        ticker = tx.get("ticker", "")
        action = tx.get("action", "")
        shares = tx.get("shares", 0) or 0
        if action == "BUY":
            ticker_bought[ticker] += shares
        elif action == "SELL":
            ticker_sold[ticker] += shares

    # Check tickers that appear in transactions
    all_tx_tickers = set(ticker_bought.keys()) | set(ticker_sold.keys())
    for ticker in all_tx_tickers:
        expected_shares = ticker_bought[ticker] - ticker_sold[ticker]

        # Get actual shares from holdings
        if ticker in holdings:
            actual_shares = holdings[ticker].get("shares", 0)
        else:
            actual_shares = 0

        # Compare (with tolerance for floating point and rounding from repeated partial sells)
        share_diff = abs(expected_shares - actual_shares)
        if share_diff > 1e-6:
            if expected_shares <= 1e-9 and actual_shares == 0:
                # Both effectively zero -- OK (sold all, removed from holdings)
                continue
            if expected_shares <= 1e-9 and ticker not in holdings:
                # Fully sold, ticker removed from holdings -- OK
                continue
            # Tolerance for small remainders from repeated partial sells (e.g.,
            # multiple 50% sells that don't sum exactly to total bought due to
            # floating-point rounding). Allow up to 1% of total bought shares.
            total_bought = ticker_bought[ticker]
            relative_diff = share_diff / total_bought if total_bought > 0 else float("inf")
            if actual_shares == 0 and ticker not in holdings and relative_diff < 0.01:
                # Small remainder from rounding, ticker removed -- acceptable
                continue
            errors.append(
                f"Holdings mismatch for {ticker}: expected {expected_shares:.8f} shares "
                f"(bought {ticker_bought[ticker]:.8f} - sold {ticker_sold[ticker]:.8f}), "
                f"got {actual_shares:.8f} in holdings (diff: {share_diff:.8f})"
            )

    # Check for holdings tickers not in transactions
    for ticker in holdings:
        shares = holdings[ticker].get("shares", 0)
        if shares > 0 and ticker not in all_tx_tickers:
            errors.append(
                f"Holdings contains {ticker} with {shares} shares but no matching transactions"
            )

    # --- Check 5: Fee reconciliation ---
    computed_fees = 0.0
    tx_with_fees = 0
    tx_without_fees = 0
    for tx in transactions:
        fee = tx.get("fee_sek")
        if fee is not None:
            computed_fees += fee
            tx_with_fees += 1
        else:
            tx_without_fees += 1

    if tx_without_fees > 0 and tx_with_fees > 0:
        errors.append(
            f"Inconsistent fee tracking: {tx_with_fees} transactions have fee_sek, "
            f"{tx_without_fees} do not"
        )

    # Only compare fees if transactions have fee_sek fields
    if tx_with_fees > 0:
        fee_diff = abs(computed_fees - total_fees_sek)
        if fee_diff > 0.01:  # 0.01 SEK tolerance
            errors.append(
                f"Fee reconciliation failed: sum of transaction fees = {computed_fees:.2f} SEK, "
                f"total_fees_sek = {total_fees_sek:.2f} SEK (diff: {fee_diff:.2f})"
            )
    elif len(transactions) > 0 and total_fees_sek == 0:
        # Transactions exist but no fee tracking at all -- warn
        errors.append(
            "No fee_sek fields in any transaction and total_fees_sek is 0, "
            "but transactions exist. Fees may not be tracked."
        )

    # --- Check 6: No duplicate timestamps per ticker ---
    seen_tx = set()
    for i, tx in enumerate(transactions):
        key = (tx.get("ticker", ""), tx.get("timestamp", ""), tx.get("action", ""))
        if key in seen_tx:
            errors.append(
                f"Duplicate transaction at index {i}: {key[2]} {key[0]} at {key[1]}"
            )
        seen_tx.add(key)

    # --- Check 7: Transaction field completeness ---
    required_tx_fields = ["timestamp", "ticker", "action", "shares", "price_usd",
                          "total_sek", "reason"]

    for i, tx in enumerate(transactions):
        for field in required_tx_fields:
            if field not in tx or tx[field] is None:
                errors.append(f"Transaction {i} missing required field: {field}")

        # Validate action value
        action = tx.get("action", "")
        if action not in ("BUY", "SELL"):
            errors.append(f"Transaction {i} has invalid action: '{action}' (expected BUY or SELL)")

        # Validate shares > 0
        shares = tx.get("shares", 0)
        if shares is not None and shares <= 0:
            errors.append(f"Transaction {i} has non-positive shares: {shares}")

        # Validate total_sek > 0
        total = tx.get("total_sek", 0)
        if total is not None and total <= 0:
            errors.append(f"Transaction {i} has non-positive total_sek: {total}")

    # --- Check 8: Holdings avg_cost_usd consistency ---
    # For tickers with multiple BUY transactions, verify avg_cost is plausible
    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue
        avg_cost = pos.get("avg_cost_usd")
        if avg_cost is None:
            errors.append(f"Holdings {ticker} missing avg_cost_usd")
            continue

        # Compute weighted average from BUY transactions
        total_cost = 0.0
        total_bought = 0.0
        for tx in transactions:
            if tx.get("ticker") != ticker or tx.get("action") != "BUY":
                continue
            tx_shares = tx.get("shares", 0) or 0
            tx_price = tx.get("price_usd", 0) or 0
            total_cost += tx_shares * tx_price
            total_bought += tx_shares

        if total_bought > 0:
            expected_avg = total_cost / total_bought
            avg_diff_pct = abs(expected_avg - avg_cost) / expected_avg * 100 if expected_avg > 0 else 0
            if avg_diff_pct > 1.0:  # More than 1% off
                errors.append(
                    f"Holdings {ticker} avg_cost_usd ({avg_cost:.4f}) differs from "
                    f"computed weighted average ({expected_avg:.4f}) by {avg_diff_pct:.2f}%"
                )

    return errors


def validate_portfolio_file(path: str) -> list[str]:
    """Validate a portfolio state JSON file.

    Convenience wrapper that loads the file and runs validate_portfolio().

    Args:
        path: Path to the portfolio state JSON file.

    Returns:
        list of error message strings (empty = valid).
    """
    # A-PR-3 (2026-04-11): Use file_utils.load_json instead of raw open()+
    # json.load(). The raw path had a TOCTOU race with concurrent saves
    # from portfolio_mgr.atomic_write_json: a partial write could be read
    # mid-flight, producing a JSONDecodeError that propagated as a "valid
    # but malformed" portfolio. load_json() retries on transient JSON
    # decode failures and uses the same atomic-rename window as the writer.
    p = pathlib.Path(path)
    if not p.exists():
        return [f"Portfolio file not found: {path}"]
    portfolio = load_json(p)
    if portfolio is None:
        return [f"Invalid or unreadable JSON in {path}"]

    return validate_portfolio(portfolio)


def validate_all() -> dict[str, list[str]]:
    """Validate both patient and bold portfolio files.

    Returns:
        dict with keys "patient" and "bold", each containing a list of errors.
    """
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    return {
        "patient": validate_portfolio_file(str(data_dir / "portfolio_state.json")),
        "bold": validate_portfolio_file(str(data_dir / "portfolio_state_bold.json")),
    }


if __name__ == "__main__":
    results = validate_all()
    for strategy, errs in results.items():
        print(f"\n{'='*60}")
        print(f"  {strategy.upper()} PORTFOLIO VALIDATION")
        print(f"{'='*60}")
        if errs:
            for e in errs:
                print(f"  ERROR: {e}")
        else:
            print("  VALID - all checks passed")
    print()
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’m through the guard and validation modules. There are already a couple of concrete candidates: the trade validator default minimum is below the stated 1000 SEK floor, and the risk classifier has a silent “unknown regime = zero risk” path. I’m checking whether the sizing and stop-loss code actually let those defects reach live decisions.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\kelly_sizing.py'" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\kelly_metals.py'" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\cost_model.py'" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\equity_curve.py'" in Q:\fa-adv-2026-05-11
 succeeded in 861ms:
"""Cost model for trade execution - fees, spread, and slippage.

Provides instrument-specific cost estimation for the exit optimizer.
Supports Avanza warrants, stocks, and crypto exchanges.

Usage:
    from portfolio.cost_model import get_cost_model
    costs = get_cost_model("warrant")
    exit_cost = costs.total_cost_sek(trade_value_sek=50000)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """All-in cost model for a single trade (one leg).

    Attributes:
        courtage_bps: Brokerage fee in basis points (e.g., 6.9 = 0.069%).
        min_fee_sek: Minimum fee in SEK (courtage floor).
        spread_bps: Half-spread cost in basis points. For a sell, you cross
            from mid to bid, losing half the spread.
        slippage_bps: Expected adverse price movement in basis points
            between decision and fill (market impact + latency).
        label: Human-readable name for this cost model.
    """
    courtage_bps: float = 0.0
    min_fee_sek: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    label: str = "default"

    def total_cost_sek(self, trade_value_sek: float) -> float:
        """Compute total one-way cost for a trade of given value.

        Returns:
            Total cost in SEK (always non-negative).
        """
        if trade_value_sek <= 0:
            return 0.0
        courtage = max(trade_value_sek * self.courtage_bps / 10_000, self.min_fee_sek)
        spread = trade_value_sek * self.spread_bps / 10_000
        slippage = trade_value_sek * self.slippage_bps / 10_000
        return courtage + spread + slippage

    def total_cost_pct(self) -> float:
        """Total cost as a percentage of trade value (excluding min fee)."""
        return (self.courtage_bps + self.spread_bps + self.slippage_bps) / 100.0

    def round_trip_pct(self) -> float:
        """Round-trip cost (buy + sell) as a percentage."""
        return self.total_cost_pct() * 2


# ---------------------------------------------------------------------------
# Preset cost models for known instrument types
# ---------------------------------------------------------------------------

# Avanza warrants/certificates: 0 courtage on many, spread is the real cost.
# Typical MINI silver spread: 0.6-1.0% (30-50 bps half-spread).
WARRANT_COSTS = CostModel(
    courtage_bps=0.0,
    min_fee_sek=0.0,
    spread_bps=40.0,    # 0.40% half-spread (conservative)
    slippage_bps=10.0,   # 0.10% slippage on market orders
    label="avanza_warrant",
)

# Avanza stocks (Mini courtage class): 0.069% with 1 SEK minimum
STOCK_COSTS = CostModel(
    courtage_bps=6.9,
    min_fee_sek=1.0,
    spread_bps=5.0,      # 0.05% half-spread (liquid US stocks)
    slippage_bps=2.0,     # 0.02% slippage
    label="avanza_stock",
)

# Crypto (Binance-equivalent fees, used for simulated portfolio)
CRYPTO_COSTS = CostModel(
    courtage_bps=5.0,     # 0.05% taker fee
    min_fee_sek=0.0,
    spread_bps=5.0,       # 0.05% half-spread
    slippage_bps=5.0,     # 0.05% slippage
    label="crypto",
)

# Elongir silver bot (specific spread from config)
ELONGIR_COSTS = CostModel(
    courtage_bps=25.0,    # 0.25% commission
    min_fee_sek=0.0,
    spread_bps=40.0,      # 0.40% half-spread
    slippage_bps=10.0,    # 0.10% slippage
    label="elongir_silver",
)

_COST_MODELS = {
    "warrant": WARRANT_COSTS,
    "stock": STOCK_COSTS,
    "crypto": CRYPTO_COSTS,
    "elongir": ELONGIR_COSTS,
}


def get_cost_model(instrument_type: str) -> CostModel:
    """Look up cost model by instrument type.

    Args:
        instrument_type: One of "warrant", "stock", "crypto", "elongir".

    Returns:
        CostModel for the instrument type. Falls back to STOCK_COSTS if unknown.
    """
    return _COST_MODELS.get(instrument_type, STOCK_COSTS)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 870ms:
"""Position sizing using Kelly criterion.

Kelly fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win

Uses signal accuracy as win probability and historical trade outcomes for avg_win/avg_loss.
Returns recommended position size as fraction of portfolio.
"""

from pathlib import Path

from portfolio.file_utils import load_json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PATIENT_FILE = DATA_DIR / "portfolio_state.json"
BOLD_FILE = DATA_DIR / "portfolio_state_bold.json"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"



def kelly_fraction(win_prob, avg_win_pct, avg_loss_pct):
    """Compute Kelly criterion fraction.

    The Kelly fraction gives the optimal bet size as a fraction of bankroll
    that maximizes long-run geometric growth rate.

    Formula: f* = (p * b - q) / b
    where p = win probability, q = 1 - p, b = avg_win / avg_loss

    Args:
        win_prob: Probability of winning (0.0 to 1.0).
        avg_win_pct: Average win as a positive percentage (e.g. 2.5 for +2.5%).
        avg_loss_pct: Average loss as a positive percentage (e.g. 1.8 for -1.8%).

    Returns:
        float: Kelly fraction (0.0 to 1.0). Clamped to [0, 1].
            Returns 0.0 if inputs are invalid or edge is negative.
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    if avg_win_pct <= 0 or avg_loss_pct <= 0:
        return 0.0

    # b = ratio of avg win to avg loss
    b = avg_win_pct / avg_loss_pct
    q = 1.0 - win_prob

    # Kelly formula: f* = (p * b - q) / b
    kelly = (win_prob * b - q) / b

    # Clamp to [0, 1] - negative Kelly means negative edge, don't bet
    return max(0.0, min(1.0, kelly))


def _compute_trade_stats(transactions, ticker=None):
    """Compute win rate and average win/loss from historical transactions.

    Pairs BUY and SELL transactions to compute realized P&L per round-trip.

    Args:
        transactions: List of transaction dicts from portfolio state.
        ticker: If specified, filter to this ticker only. None = all tickers.

    Returns:
        dict: {win_rate, avg_win_pct, avg_loss_pct, total_trades, wins, losses}
              Returns None if insufficient data (fewer than 2 round-trips).
    """
    # Group transactions by ticker
    from collections import defaultdict
    buys_by_ticker = defaultdict(list)
    sells_by_ticker = defaultdict(list)

    for t in transactions:
        t_ticker = t.get("ticker", "")
        if ticker and t_ticker != ticker:
            continue
        action = t.get("action", "")
        if action == "BUY":
            buys_by_ticker[t_ticker].append(t)
        elif action == "SELL":
            sells_by_ticker[t_ticker].append(t)

    # Compute P&L for each sell vs weighted average buy price
    pnl_list = []
    for t_ticker, sells in sells_by_ticker.items():
        buys = buys_by_ticker.get(t_ticker, [])
        if not buys:
            continue

        # Compute weighted average buy price (in SEK per share)
        total_shares_bought = sum(b.get("shares", 0) for b in buys)
        total_cost = sum(b.get("total_sek", 0) for b in buys)
        if total_shares_bought <= 0:
            continue
        avg_buy_price = total_cost / total_shares_bought

        for sell in sells:
            sell_shares = sell.get("shares", 0)
            sell_total = sell.get("total_sek", 0)
            if sell_shares <= 0:
                continue
            sell_price_per_share = sell_total / sell_shares
            pnl_pct = (sell_price_per_share - avg_buy_price) / avg_buy_price * 100
            pnl_list.append(pnl_pct)

    if len(pnl_list) < 2:
        return None

    wins = [p for p in pnl_list if p > 0]
    losses = [abs(p) for p in pnl_list if p <= 0]

    win_rate = len(wins) / len(pnl_list) if pnl_list else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return {
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "total_trades": len(pnl_list),
        "wins": len(wins),
        "losses": len(losses),
    }


def _get_signal_accuracy(agent_summary, ticker=None):
    """Extract consensus accuracy from agent_summary.

    Uses signal_accuracy_1d.consensus.accuracy as the win probability proxy.
    Falls back to weighted_confidence from the ticker's signal data.

    Args:
        agent_summary: Parsed agent_summary.json dict.
        ticker: Ticker to get accuracy for (used for weighted_confidence fallback).

    Returns:
        float: Estimated win probability (0.0 to 1.0).
    """
    # Primary: use overall consensus accuracy
    acc_data = agent_summary.get("signal_accuracy_1d", {})
    consensus_acc = acc_data.get("consensus", {}).get("accuracy")
    if consensus_acc is not None and consensus_acc > 0:
        return consensus_acc

    # Fallback: use weighted_confidence from the ticker's signals
    if ticker:
        signals = agent_summary.get("signals", {})
        ticker_data = signals.get(ticker, {})
        weighted_conf = ticker_data.get("weighted_confidence")
        if weighted_conf is not None:
            return weighted_conf

    # Last resort: 50/50
    return 0.5


def _get_ticker_signal_accuracy(agent_summary, ticker):
    """Get the accuracy of the most relevant signals for a specific ticker.

    Computes a weighted average of signal accuracies, weighted by each signal's
    normalized weight. Only considers signals that are actively voting (non-HOLD).

    P1-11 (2026-05-02 adversarial follow-up): per-signal accuracy is now
    looked up per-ticker first via `agent_summary["per_ticker_signal_accuracy"]`
    (produced by `accuracy_stats.accuracy_by_ticker_signal_cached()` upstream).
    If a ticker/signal isn't present per-ticker, OR has fewer than 5 samples
    on this ticker, it falls back to the system-wide
    `signal_accuracy_1d.signals` block. Without this, a signal that's 70%
    accurate on XAG-USD but 30% on BTC-USD shows up as ~50% in the
    system-wide aggregate, distorting Kelly sizing on both. The fallback
    keeps backwards compatibility with older agent_summary writers that
    don't yet emit the per-ticker block.

    Args:
        agent_summary: Parsed agent_summary.json dict.
        ticker: Ticker symbol.

    Returns:
        float: Weighted signal accuracy (0.0 to 1.0), or None if insufficient data.
    """
    acc_data = agent_summary.get("signal_accuracy_1d", {})
    sig_accuracies = acc_data.get("signals", {})
    sig_weights = agent_summary.get("signal_weights", {})

    # P1-11: per-ticker per-signal accuracy block (preferred when present).
    per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
    per_ticker_for_ticker = per_ticker_block.get(ticker, {}) or {}

    signals = agent_summary.get("signals", {})
    ticker_data = signals.get(ticker, {})
    extra = ticker_data.get("extra", {}) if isinstance(ticker_data, dict) else {}
    votes = extra.get("_votes", {})

    if not votes:
        return None
    # Either source can satisfy us - only bail if BOTH are empty.
    if not sig_accuracies and not per_ticker_for_ticker:
        return None

    weighted_sum = 0.0
    weight_total = 0.0

    for sig_name, vote in votes.items():
        if vote == "HOLD":
            continue

        # Prefer per-ticker accuracy when it has enough samples for THIS
        # ticker; otherwise fall back to the system-wide row.
        per_ticker_sig = per_ticker_for_ticker.get(sig_name) or {}
        per_ticker_samples = per_ticker_sig.get("samples", 0)
        if per_ticker_samples >= 5:
            accuracy = per_ticker_sig.get("accuracy", 0.5)
            samples = per_ticker_samples
        else:
            sig_acc = sig_accuracies.get(sig_name, {})
            accuracy = sig_acc.get("accuracy", 0.5)
            samples = sig_acc.get("samples", 0)
            if samples < 5:
                continue  # unreliable on both axes

        weight = sig_weights.get(sig_name, {}).get("normalized_weight", 1.0)
        weighted_sum += accuracy * weight
        weight_total += weight

    if weight_total <= 0:
        return None

    return weighted_sum / weight_total


def recommended_size(ticker, portfolio_path=None, agent_summary=None, strategy="patient"):
    """Compute recommended position size using Kelly criterion.

    Combines signal accuracy (as win probability) with historical trade
    performance (avg win/loss) to compute optimal position sizing.

    Args:
        ticker: Ticker symbol to compute sizing for.
        portfolio_path: Path to portfolio state JSON. If None, uses default
            based on strategy.
        agent_summary: Parsed agent_summary dict. If None, loads from file.
        strategy: "patient" or "bold" - determines default portfolio and max alloc.

    Returns:
        dict: {
            kelly_pct: Full Kelly fraction (0-1),
            half_kelly_pct: Half Kelly (more conservative),
            quarter_kelly_pct: Quarter Kelly (most conservative),
            recommended_sek: Recommended trade size in SEK (using half Kelly),
            max_alloc_sek: Maximum allocation per strategy rules,
            win_prob: Estimated win probability used,
            avg_win_pct: Average win percentage used,
            avg_loss_pct: Average loss percentage used,
            source: Description of data source used for estimates,
        }
    """
    # Load portfolio
    if portfolio_path is None:
        portfolio_path = BOLD_FILE if strategy == "bold" else PATIENT_FILE
    portfolio = load_json(portfolio_path, default={})
    cash_sek = portfolio.get("cash_sek", 0)
    transactions = portfolio.get("transactions", [])

    # Load agent summary
    if agent_summary is None:
        agent_summary = load_json(AGENT_SUMMARY_FILE, default={})

    # Max allocation per strategy rules
    alloc_frac = 0.30 if strategy == "bold" else 0.15
    max_alloc = cash_sek * alloc_frac

    # Estimate win probability
    # Priority: ticker-specific weighted signal accuracy > consensus accuracy > 50%
    # P1-11 (2026-05-02): _get_ticker_signal_accuracy now prefers
    # per_ticker_signal_accuracy over the system-wide aggregate when a
    # per-ticker block is supplied in agent_summary. Source string
    # advertises which path was used so operators can audit the decision.
    per_ticker_block = agent_summary.get("per_ticker_signal_accuracy", {}) or {}
    has_per_ticker = bool(per_ticker_block.get(ticker))
    win_prob = _get_ticker_signal_accuracy(agent_summary, ticker)
    if has_per_ticker:
        source = f"per-ticker weighted signal accuracy for {ticker}"
    else:
        source = f"weighted signal accuracy for {ticker} (system-wide fallback)"

    if win_prob is None:
        win_prob = _get_signal_accuracy(agent_summary, ticker)
        source = "consensus accuracy"

    # Estimate avg win/loss from historical trades
    trade_stats = _compute_trade_stats(transactions, ticker=ticker)
    if trade_stats is None:
        # Try all tickers if not enough ticker-specific data
        trade_stats = _compute_trade_stats(transactions, ticker=None)

    if trade_stats and trade_stats["avg_win_pct"] > 0 and trade_stats["avg_loss_pct"] > 0:
        avg_win = trade_stats["avg_win_pct"]
        avg_loss = trade_stats["avg_loss_pct"]
        source += f" + trade history ({trade_stats['total_trades']} trades)"
    else:
        # Default estimates based on typical crypto/stock moves
        # Use ATR from agent summary if available
        signals = agent_summary.get("signals", {})
        ticker_data = signals.get(ticker, {})
        atr_pct = ticker_data.get("atr_pct", 1.5) if isinstance(ticker_data, dict) else 1.5

        # Assume avg win = 1.5x ATR, avg loss = 1x ATR (realistic risk/reward)
        avg_win = atr_pct * 1.5
        avg_loss = atr_pct * 1.0
        source += f" + ATR-based estimates (win={avg_win:.1f}%, loss={avg_loss:.1f}%)"

    # Compute Kelly
    full_kelly = kelly_fraction(win_prob, avg_win, avg_loss)
    half_kelly = full_kelly / 2.0
    quarter_kelly = full_kelly / 4.0

    # Apply exposure ceiling from market health / exposure coach.
    # Scales position sizes down when the broad market is unhealthy.
    exposure_rec = agent_summary.get("exposure_recommendation", {})
    exposure_ceiling = exposure_rec.get("exposure_ceiling", 1.0)

    # Recommended size = half Kelly * cash * exposure_ceiling, capped at max allocation
    rec_sek = min(half_kelly * cash_sek * exposure_ceiling, max_alloc)

    # Minimum trade size check
    if rec_sek < 500:
        rec_sek = 0  # Below minimum trade size

    return {
        "kelly_pct": round(full_kelly, 4),
        "half_kelly_pct": round(half_kelly, 4),
        "quarter_kelly_pct": round(quarter_kelly, 4),
        "recommended_sek": round(rec_sek, 0),
        "max_alloc_sek": round(max_alloc, 0),
        "exposure_ceiling": round(exposure_ceiling, 2),
        "win_prob": round(win_prob, 4),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "source": source,
    }


def print_sizing_report(tickers=None, strategy="patient"):
    """Print Kelly sizing recommendations for given tickers.

    Args:
        tickers: List of ticker symbols. If None, uses all from agent_summary.
        strategy: "patient" or "bold".
    """
    agent_summary = load_json(AGENT_SUMMARY_FILE, default={})
    if tickers is None:
        tickers = list(agent_summary.get("signals", {}).keys())

    if not tickers:
        print("No tickers found in agent_summary.json")
        return

    print(f"=== Kelly Sizing Report ({strategy.title()}) ===")
    print()
    print(
        f"{'Ticker':<10} {'Kelly%':>7} {'Half-K%':>8} {'Rec SEK':>10} "
        f"{'Max SEK':>10} {'Win Prob':>9}"
    )
    print(
        f"{'------':<10} {'------':>7} {'-------':>8} {'-------':>10} "
        f"{'-------':>10} {'--------':>9}"
    )

    for ticker in sorted(tickers):
        rec = recommended_size(ticker, agent_summary=agent_summary, strategy=strategy)
        print(
            f"{ticker:<10} {rec['kelly_pct']*100:>6.1f}% {rec['half_kelly_pct']*100:>7.1f}% "
            f"{rec['recommended_sek']:>10,.0f} {rec['max_alloc_sek']:>10,.0f} "
            f"{rec['win_prob']*100:>8.1f}%"
        )

    print()
    print("Note: Recommended size uses Half Kelly (more conservative).")
    print("Full Kelly is optimal but volatile. Quarter Kelly is safest.")


if __name__ == "__main__":
    import sys

    strategy = "bold" if "--bold" in sys.argv else "patient"
    tickers = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not tickers:
        tickers = None
    print_sizing_report(tickers=tickers, strategy=strategy)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 880ms:
"""Kelly-optimal position sizing for metals warrant trading.

Computes leverage-adjusted Kelly fraction using real signal accuracy data
and historical outcome statistics from signal_log.db. Designed to replace
fixed position sizing (30% of buying power) with edge-aware sizing.

Usage::

    from portfolio.kelly_metals import recommended_metals_size

    rec = recommended_metals_size(
        ticker="XAG-USD",
        leverage=5.0,
        buying_power_sek=5000,
        consecutive_losses=0,
    )
    # rec["position_sek"]   -> how much to allocate (SEK)
    # rec["units"]          -> how many cert units at current ask
    # rec["kelly_pct"]      -> full Kelly fraction (0-1)
    # rec["half_kelly_pct"] -> half Kelly (recommended)
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from portfolio.file_utils import load_json
from portfolio.kelly_sizing import kelly_fraction

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ACCURACY_CACHE = DATA_DIR / "accuracy_cache.json"
SIGNAL_DB = DATA_DIR / "signal_log.db"
AGENT_SUMMARY = DATA_DIR / "agent_summary.json"

# Defaults when no historical data is available
_DEFAULT_AVG_WIN = {"XAG-USD": 3.09, "XAU-USD": 2.10}
_DEFAULT_AVG_LOSS = {"XAG-USD": 2.43, "XAU-USD": 1.80}
_DEFAULT_WIN_RATE = 0.52

# Conservative sizing limits
MIN_TRADE_SEK = 500.0
MAX_POSITION_FRACTION = 0.95  # never go above 95% of buying power
LOSS_REDUCTION_STEP = 0.25    # reduce Kelly by 25% per consecutive loss


def _get_ticker_accuracy(ticker: str) -> float | None:
    """Read per-ticker consensus accuracy from accuracy_cache.json."""
    cache = load_json(str(ACCURACY_CACHE), default={})
    per_ticker = cache.get("per_ticker_consensus", {})
    entry = per_ticker.get(ticker, {})
    if not isinstance(entry, dict):
        return None
    total = entry.get("total", 0)
    if total < 30:
        return None
    acc = entry.get("accuracy")
    if acc is not None and 0 < acc < 1:
        return acc
    return None


def _get_outcome_stats(ticker: str, horizon: str = "1d") -> dict | None:
    """Compute win rate and avg win/loss from signal_log.db outcomes.

    Joins ticker_signals (consensus BUY/SELL) with outcomes to measure
    how well the consensus predicted direction over the given horizon.

    Returns:
        dict with win_rate, avg_win_pct, avg_loss_pct, n_trades
        or None if insufficient data.
    """
    if not SIGNAL_DB.exists():
        return None

    try:
        conn = sqlite3.connect(str(SIGNAL_DB))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts.consensus, o.change_pct
            FROM ticker_signals ts
            JOIN outcomes o ON o.snapshot_id = ts.snapshot_id
                           AND o.ticker = ts.ticker
            WHERE ts.ticker = ? AND o.horizon = ?
              AND ts.consensus IN ('BUY', 'SELL')
            """,
            (ticker, horizon),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return None

    if len(rows) < 30:
        return None

    wins: list[float] = []
    losses: list[float] = []
    for consensus, change_pct in rows:
        correct = (consensus == "BUY" and change_pct > 0) or (
            consensus == "SELL" and change_pct < 0
        )
        if correct:
            wins.append(abs(change_pct))
        else:
            losses.append(abs(change_pct))

    total = len(wins) + len(losses)
    if total == 0:
        return None

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return {
        "win_rate": len(wins) / total,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "n_trades": total,
    }


def _loss_reduction(consecutive_losses: int) -> float:
    """Reduce position size after consecutive losses.

    Returns a multiplier in (0, 1]. After 4+ losses, position is 0 (sit out).
    """
    if consecutive_losses <= 0:
        return 1.0
    reduction = 1.0 - consecutive_losses * LOSS_REDUCTION_STEP
    return max(0.0, reduction)


def recommended_metals_size(
    ticker: str = "XAG-USD",
    leverage: float = 5.0,
    buying_power_sek: float = 0.0,
    ask_price_sek: float = 0.0,
    consecutive_losses: int = 0,
    agent_summary: dict | None = None,
    horizon: str = "1d",
) -> dict:
    """Compute Kelly-optimal position size for a metals warrant trade.

    Args:
        ticker: Underlying ticker (e.g. "XAG-USD").
        leverage: Certificate leverage (e.g. 5.0 for BULL SILVER X5).
        buying_power_sek: Available cash on Avanza.
        ask_price_sek: Current ask price of the certificate (SEK).
        consecutive_losses: Number of consecutive losses (reduces sizing).
        agent_summary: Optional agent_summary dict for signal-based win prob.
        horizon: Outcome horizon for accuracy stats ("1d", "3h").

    Returns:
        dict with sizing recommendation and all computation details.
    """
    # --- Step 1: Estimate win probability ---
    source_parts: list[str] = []
    win_rate = None

    # Try per-ticker accuracy from cache
    ticker_acc = _get_ticker_accuracy(ticker)
    if ticker_acc is not None:
        win_rate = ticker_acc
        source_parts.append(f"per_ticker_consensus ({win_rate:.1%})")

    # Try outcome stats from SQLite
    outcome_stats = _get_outcome_stats(ticker, horizon)
    if outcome_stats is not None:
        db_win_rate = outcome_stats["win_rate"]
        if win_rate is not None:
            # Blend: 60% DB (more data), 40% cache (includes recent)
            win_rate = 0.6 * db_win_rate + 0.4 * win_rate
            source_parts.append(f"blended with DB ({db_win_rate:.1%}, n={outcome_stats['n_trades']})")
        else:
            win_rate = db_win_rate
            source_parts.append(f"signal_log.db ({win_rate:.1%}, n={outcome_stats['n_trades']})")

    # Try agent_summary weighted confidence
    if win_rate is None and agent_summary:
        signals = agent_summary.get("signals", {})
        ticker_data = signals.get(ticker, {})
        if isinstance(ticker_data, dict):
            wc = ticker_data.get("weighted_confidence")
            if wc is not None and 0 < wc < 1:
                win_rate = wc
                source_parts.append(f"weighted_confidence ({win_rate:.1%})")

    # Fallback
    if win_rate is None:
        win_rate = _DEFAULT_WIN_RATE
        source_parts.append(f"default ({win_rate:.1%})")

    # --- Step 2: Estimate avg win/loss (underlying %) ---
    if outcome_stats and outcome_stats["avg_win_pct"] > 0 and outcome_stats["avg_loss_pct"] > 0:
        avg_win = outcome_stats["avg_win_pct"]
        avg_loss = outcome_stats["avg_loss_pct"]
        source_parts.append(f"outcome W={avg_win:.2f}% L={avg_loss:.2f}%")
    else:
        avg_win = _DEFAULT_AVG_WIN.get(ticker, 3.0)
        avg_loss = _DEFAULT_AVG_LOSS.get(ticker, 2.5)
        source_parts.append(f"default W={avg_win:.2f}% L={avg_loss:.2f}%")

    # --- Step 3: Compute Kelly fraction (on underlying) ---
    full_kelly = kelly_fraction(win_rate, avg_win, avg_loss)
    half_kelly = full_kelly / 2.0

    # --- Step 4: Convert to leveraged position fraction ---
    # Kelly says risk X% of capital. With leverage L, a loss of avg_loss%
    # on underlying = avg_loss * L % on the certificate.
    # Position fraction = half_kelly / (avg_loss * leverage / 100)
    cert_loss_frac = avg_loss * leverage / 100.0
    if cert_loss_frac > 0:
        position_fraction = half_kelly / cert_loss_frac
    else:
        position_fraction = 0.0

    position_fraction = min(position_fraction, MAX_POSITION_FRACTION)

    # --- Step 5: Apply consecutive loss reduction ---
    loss_mult = _loss_reduction(consecutive_losses)
    adjusted_fraction = position_fraction * loss_mult

    # --- Step 6: Compute final SEK allocation ---
    position_sek = buying_power_sek * adjusted_fraction
    if position_sek < MIN_TRADE_SEK:
        position_sek = 0.0

    # --- Step 7: Compute units ---
    units = 0
    if ask_price_sek > 0 and position_sek > 0:
        units = int(position_sek / ask_price_sek)
        if units <= 0:
            position_sek = 0.0

    # --- Step 8: Expected growth rate (daily log-growth at half-Kelly) ---
    cert_win_frac = avg_win * leverage / 100.0
    f = adjusted_fraction
    if f > 0 and cert_loss_frac > 0:
        daily_log_growth = (
            win_rate * math.log(1 + f * cert_win_frac)
            + (1 - win_rate) * math.log(max(1e-10, 1 - f * cert_loss_frac))
        )
    else:
        daily_log_growth = 0.0

    monthly_growth = math.exp(daily_log_growth * 22) - 1 if daily_log_growth > 0 else 0.0

    return {
        # Sizing
        "position_sek": round(position_sek, 0),
        "position_fraction": round(adjusted_fraction, 4),
        "units": units,
        # Kelly components
        "kelly_pct": round(full_kelly, 4),
        "half_kelly_pct": round(half_kelly, 4),
        "position_fraction_raw": round(position_fraction, 4),
        # Inputs
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "leverage": leverage,
        "consecutive_losses": consecutive_losses,
        "loss_multiplier": round(loss_mult, 2),
        # Growth projections
        "daily_log_growth": round(daily_log_growth, 6),
        "monthly_growth_pct": round(monthly_growth * 100, 1),
        # Provenance
        "source": " | ".join(source_parts),
    }


def format_kelly_line(rec: dict) -> str:
    """One-line summary for Telegram notifications."""
    k = rec["half_kelly_pct"] * 100
    wr = rec["win_rate"] * 100
    pos = rec["position_sek"]
    lm = rec["loss_multiplier"]
    parts = [f"Kelly:{k:.1f}% WR:{wr:.0f}% Pos:{pos:.0f}kr"]
    if lm < 1.0:
        parts.append(f"(x{lm:.2f} loss adj)")
    return " ".join(parts)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 905ms:
"""Equity curve analysis and portfolio metrics.

Loads portfolio value history from the JSONL log and computes performance
metrics useful for charting and strategy comparison.
"""

import datetime
import math
import pathlib
from collections import defaultdict

from portfolio.file_utils import load_jsonl

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

DEFAULT_HISTORY_PATH = DATA_DIR / "portfolio_value_history.jsonl"
INITIAL_VALUE = 500_000  # SEK
RISK_FREE_RATE_ANNUAL = 0.035  # 3.5% Swedish risk-free rate (approximate)

# The portfolio loop runs 24/7 (crypto always active), so
# portfolio_value_history.jsonl has entries every calendar day.
# Annualization must use 365, not 252 (stock-only convention).
ANNUALIZATION_DAYS = 365


def load_equity_curve(path: str | None = None) -> list[dict]:
    """Load portfolio value history for charting.

    Reads the JSONL file and returns a list of dicts sorted by timestamp.
    Each dict contains:
        - ts: ISO-8601 timestamp
        - patient_value_sek: float
        - bold_value_sek: float
        - patient_pnl_pct: float
        - bold_pnl_pct: float
        - fx_rate: float
        - prices: dict of ticker -> USD price

    Args:
        path: Path to the portfolio_value_history.jsonl file.
            Defaults to data/portfolio_value_history.jsonl.

    Returns:
        list of dicts sorted by timestamp (oldest first).
        Empty list if file doesn't exist or is empty.
    """
    if path is None:
        path = str(DEFAULT_HISTORY_PATH)

    result = load_jsonl(path)

    # Sort by timestamp
    result.sort(key=lambda x: x.get("ts", ""))
    return result


def _parse_ts(ts_str: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    dt = datetime.datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt


def _daily_returns(curve: list[dict], value_key: str) -> list[float]:
    """Compute daily returns from the equity curve.

    Groups entries by date, takes the last entry per day, and computes
    day-over-day percentage returns.

    Args:
        curve: Sorted equity curve list.
        value_key: Either "patient_value_sek" or "bold_value_sek".

    Returns:
        list of daily return percentages.
    """
    if not curve:
        return []

    # Group by date, take last value per day
    daily_values = {}
    for entry in curve:
        ts_str = entry.get("ts", "")
        if not ts_str:
            continue
        try:
            dt = _parse_ts(ts_str)
            date_key = dt.date()
            value = entry.get(value_key, 0)
            if value > 0:
                daily_values[date_key] = value
        except (ValueError, TypeError):
            continue

    if len(daily_values) < 2:
        return []

    sorted_dates = sorted(daily_values.keys())
    returns = []
    for i in range(1, len(sorted_dates)):
        prev_val = daily_values[sorted_dates[i - 1]]
        curr_val = daily_values[sorted_dates[i]]
        if prev_val > 0:
            daily_ret = ((curr_val - prev_val) / prev_val) * 100
            returns.append(daily_ret)
        else:
            returns.append(0.0)

    return returns


def compute_metrics(curve: list[dict], strategy: str) -> dict:
    """Compute portfolio metrics from equity curve.

    Args:
        curve: List of equity curve entries (from load_equity_curve).
        strategy: "patient" or "bold".

    Returns:
        dict with:
            - max_drawdown_pct: float -- maximum peak-to-trough drawdown
            - sharpe_ratio: float -- annualized Sharpe ratio (or None if insufficient data)
            - sortino_ratio: float -- annualized Sortino ratio (or None)
            - win_rate: float -- percentage of positive-return days
            - avg_daily_return_pct: float -- mean daily return
            - best_day_pct: float -- best single-day return
            - worst_day_pct: float -- worst single-day return
            - days_in_drawdown: int -- number of days below previous peak
            - total_return_pct: float -- total return from start to end
            - annualized_return_pct: float -- annualized return (or None if < 1 day)
            - volatility_annual_pct: float -- annualized daily volatility
            - num_data_points: int -- number of entries in curve
            - date_range: tuple of (first_ts, last_ts)
    """
    value_key = f"{strategy}_value_sek"

    result = {
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "win_rate": 0.0,
        "avg_daily_return_pct": 0.0,
        "best_day_pct": 0.0,
        "worst_day_pct": 0.0,
        "days_in_drawdown": 0,
        "total_return_pct": 0.0,
        "annualized_return_pct": None,
        "volatility_annual_pct": 0.0,
        "num_data_points": len(curve),
        "date_range": None,
    }

    if not curve:
        return result

    # Extract values
    values = []
    timestamps = []
    for entry in curve:
        val = entry.get(value_key)
        ts = entry.get("ts", "")
        if val is not None and val > 0 and ts:
            values.append(val)
            timestamps.append(ts)

    if not values:
        return result

    result["num_data_points"] = len(values)
    result["date_range"] = (timestamps[0], timestamps[-1])

    # --- Total return ---
    first_val = values[0]
    last_val = values[-1]
    total_return_pct = ((last_val - first_val) / first_val) * 100
    result["total_return_pct"] = round(total_return_pct, 4)

    # --- Annualized return ---
    try:
        first_dt = _parse_ts(timestamps[0])
        last_dt = _parse_ts(timestamps[-1])
        days_elapsed = (last_dt - first_dt).total_seconds() / 86400
        if days_elapsed >= 1:
            years = days_elapsed / 365.25
            # Annualized return = (final/initial)^(1/years) - 1
            if first_val > 0 and last_val > 0:
                annualized = (pow(last_val / first_val, 1 / years) - 1) * 100
                result["annualized_return_pct"] = round(annualized, 4)
    except (ValueError, TypeError, ZeroDivisionError):
        pass

    # --- Maximum drawdown ---
    peak = values[0]
    max_dd = 0.0
    # Group by date for drawdown day counting
    date_was_below = set()

    for i, val in enumerate(values):
        if val > peak:
            peak = val
        dd = ((peak - val) / peak) * 100
        if dd > max_dd:
            max_dd = dd
        if dd > 0.01:  # Meaningfully below peak
            try:
                dt = _parse_ts(timestamps[i])
                date_was_below.add(dt.date())
            except (ValueError, TypeError):
                pass

    result["max_drawdown_pct"] = round(max_dd, 4)
    result["days_in_drawdown"] = len(date_was_below)

    # --- Daily returns ---
    daily_rets = _daily_returns(curve, value_key)

    if daily_rets:
        result["avg_daily_return_pct"] = round(sum(daily_rets) / len(daily_rets), 6)
        result["best_day_pct"] = round(max(daily_rets), 4)
        result["worst_day_pct"] = round(min(daily_rets), 4)

        # Win rate
        positive_days = sum(1 for r in daily_rets if r > 0)
        result["win_rate"] = round((positive_days / len(daily_rets)) * 100, 2)

        # Volatility (annualized)
        if len(daily_rets) >= 2:
            mean_ret = sum(daily_rets) / len(daily_rets)
            variance = sum((r - mean_ret) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
            daily_vol = math.sqrt(variance)
            annual_vol = daily_vol * math.sqrt(ANNUALIZATION_DAYS)
            result["volatility_annual_pct"] = round(annual_vol, 4)

            # Sharpe ratio (annualized)
            daily_rf = RISK_FREE_RATE_ANNUAL / ANNUALIZATION_DAYS
            # Convert daily returns to decimal for Sharpe
            daily_rets_dec = [r / 100 for r in daily_rets]
            mean_excess = sum(r - daily_rf for r in daily_rets_dec) / len(daily_rets_dec)
            if daily_vol > 0:
                # Annualize Sharpe: mean_excess / daily_std * sqrt(ANNUALIZATION_DAYS)
                # BUG-225: extract mean to avoid O(n^2) recomputation inside generator
                mean_dec = sum(daily_rets_dec) / len(daily_rets_dec)
                daily_std_dec = math.sqrt(
                    sum((r - mean_dec) ** 2
                        for r in daily_rets_dec) / (len(daily_rets_dec) - 1)
                )
                if daily_std_dec > 0:
                    sharpe = (mean_excess / daily_std_dec) * math.sqrt(ANNUALIZATION_DAYS)
                    result["sharpe_ratio"] = round(sharpe, 4)

            # H19: Sortino - divide by TOTAL observations, not just downside count (standard formula)
            squared_devs = [min(r - daily_rf, 0) ** 2 for r in daily_rets_dec]
            downside_dev = math.sqrt(sum(squared_devs) / len(daily_rets_dec))
            if downside_dev > 0:
                sortino = (mean_excess / downside_dev) * math.sqrt(ANNUALIZATION_DAYS)
                result["sortino_ratio"] = round(sortino, 4)

    return result


def compare_strategies(curve: list[dict]) -> dict:
    """Compare patient vs bold strategy performance.

    Args:
        curve: Equity curve from load_equity_curve().

    Returns:
        dict with:
            - patient: metrics dict
            - bold: metrics dict
            - comparison: dict with relative performance
    """
    patient = compute_metrics(curve, "patient")
    bold = compute_metrics(curve, "bold")

    comparison = {
        "return_diff_pct": round(patient["total_return_pct"] - bold["total_return_pct"], 4),
        "leader": "patient" if patient["total_return_pct"] > bold["total_return_pct"] else "bold",
        "drawdown_diff_pct": round(patient["max_drawdown_pct"] - bold["max_drawdown_pct"], 4),
        "lower_drawdown": "patient" if patient["max_drawdown_pct"] < bold["max_drawdown_pct"] else "bold",
    }

    # Risk-adjusted comparison
    if patient.get("sharpe_ratio") is not None and bold.get("sharpe_ratio") is not None:
        comparison["sharpe_leader"] = "patient" if patient["sharpe_ratio"] > bold["sharpe_ratio"] else "bold"
        comparison["sharpe_diff"] = round(patient["sharpe_ratio"] - bold["sharpe_ratio"], 4)

    return {
        "patient": patient,
        "bold": bold,
        "comparison": comparison,
    }


def get_latest_values(curve: list[dict]) -> dict | None:
    """Get the most recent portfolio values from the curve.

    Returns:
        dict with patient_value_sek, bold_value_sek, ts, or None if curve is empty.
    """
    if not curve:
        return None
    latest = curve[-1]
    return {
        "ts": latest.get("ts"),
        "patient_value_sek": latest.get("patient_value_sek"),
        "bold_value_sek": latest.get("bold_value_sek"),
        "patient_pnl_pct": latest.get("patient_pnl_pct"),
        "bold_pnl_pct": latest.get("bold_pnl_pct"),
    }


def _pair_round_trips(transactions):
    """Match BUY and SELL transactions into round-trip pairs.

    Uses FIFO matching: each SELL is paired with the earliest available BUY
    shares for that ticker.

    Args:
        transactions: List of transaction dicts from portfolio state.

    Returns:
        list of round-trip dicts, each with:
            - ticker: str
            - buy_ts: str (ISO-8601)
            - sell_ts: str (ISO-8601)
            - buy_price_sek: float (per share)
            - sell_price_sek: float (per share)
            - shares: float
            - pnl_pct: float
            - pnl_sek: float
            - hold_hours: float
            - fee_sek: float (total fees for this round trip)
    """

    # Group BUYs by ticker - maintain FIFO order
    buy_queues = defaultdict(list)
    for tx in transactions:
        if tx.get("action") == "BUY":
            ticker = tx.get("ticker", "")
            shares = tx.get("shares", 0)
            total_sek = tx.get("total_sek", 0)
            fee = tx.get("fee_sek", 0) or 0
            if shares > 0:
                price_per_share = total_sek / shares
                buy_queues[ticker].append({
                    "ts": tx.get("timestamp", ""),
                    "remaining_shares": shares,
                    "original_shares": shares,
                    "price_per_share": price_per_share,
                    "fee_sek": fee,
                })

    round_trips = []

    for tx in transactions:
        if tx.get("action") != "SELL":
            continue
        ticker = tx.get("ticker", "")
        sell_shares = tx.get("shares", 0)
        sell_total = tx.get("total_sek", 0)
        sell_fee = tx.get("fee_sek", 0) or 0
        sell_ts = tx.get("timestamp", "")

        if sell_shares <= 0 or ticker not in buy_queues:
            continue

        sell_price_per_share = sell_total / sell_shares if sell_shares > 0 else 0
        shares_to_match = sell_shares

        while shares_to_match > 0 and buy_queues[ticker]:
            buy = buy_queues[ticker][0]
            matched = min(shares_to_match, buy["remaining_shares"])

            # Compute hold time
            hold_hours = 0
            try:
                buy_dt = datetime.datetime.fromisoformat(buy["ts"])
                sell_dt = datetime.datetime.fromisoformat(sell_ts)
                if buy_dt.tzinfo is None:
                    buy_dt = buy_dt.replace(tzinfo=datetime.UTC)
                if sell_dt.tzinfo is None:
                    sell_dt = sell_dt.replace(tzinfo=datetime.UTC)
                hold_hours = (sell_dt - buy_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

            buy_price = buy["price_per_share"]
            pnl_pct = ((sell_price_per_share - buy_price) / buy_price * 100) if buy_price > 0 else 0

            # Proportional fees - use original buy quantity as denominator
            # to avoid inflating fee allocation on subsequent partial matches
            # (BUG-37 regression covered by test_equity_curve_fifo.TestBug37FeeDoubleCount).
            #
            # P0-6 (2026-05-02): pnl_sek is now NET of buy+sell fees so
            # downstream metrics (profit_factor, total_pnl_sek, Calmar) report
            # realised SEK. `pnl_pct` remains the gross price-% (price-move
            # only) - used for streaks and expectancy. `fee_sek` field still
            # reports total fees so consumers can reconstruct gross if needed.
            # See tests/test_equity_curve_fifo.TestPnlSekNetOfFees and
            # tests/test_portfolio_metrics.TestProfitFactorNetOfFees.
            buy_fee_share = (buy["fee_sek"] * matched / buy["original_shares"]) if buy["original_shares"] > 0 else 0
            sell_fee_share = (sell_fee * matched / sell_shares) if sell_shares > 0 else 0
            pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share

            round_trips.append({
                "ticker": ticker,
                "buy_ts": buy["ts"],
                "sell_ts": sell_ts,
                "buy_price_sek": round(buy_price, 4),
                "sell_price_sek": round(sell_price_per_share, 4),
                "shares": round(matched, 8),
                "pnl_pct": round(pnl_pct, 4),
                "pnl_sek": round(pnl_sek, 2),
                "hold_hours": round(hold_hours, 2),
                "fee_sek": round(buy_fee_share + sell_fee_share, 2),
            })

            buy["remaining_shares"] -= matched
            shares_to_match -= matched

            if buy["remaining_shares"] <= 1e-10:
                buy_queues[ticker].pop(0)

    return round_trips


def compute_trade_metrics(transactions, initial_value=INITIAL_VALUE):
    """Compute per-trade performance metrics from transaction history.

    Args:
        transactions: List of transaction dicts from portfolio state.
        initial_value: Starting portfolio value in SEK.

    Returns:
        dict with:
            - profit_factor: gross_profit / gross_loss (None if no losses)
            - avg_hold_hours: average hold time per round trip
            - trade_frequency_per_week: trades per 7 calendar days
            - win_loss_ratio: avg_win_pct / avg_loss_pct (None if no losses)
            - max_consecutive_wins: longest win streak
            - max_consecutive_losses: longest loss streak
            - expectancy_pct: (win_rate * avg_win) - (loss_rate * avg_loss)
            - calmar_ratio: annualized_return / max_drawdown (None if insufficient data)
            - round_trips: int (number of paired trades)
            - total_pnl_sek: float (sum of round-trip P&L)
    """
    trips = _pair_round_trips(transactions)

    result = {
        "profit_factor": None,
        "avg_hold_hours": 0,
        "trade_frequency_per_week": 0,
        "win_loss_ratio": None,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "expectancy_pct": 0,
        "calmar_ratio": None,
        "round_trips": len(trips),
        "total_pnl_sek": 0,
    }

    if not trips:
        return result

    # Gross profit/loss
    gross_profit = sum(t["pnl_sek"] for t in trips if t["pnl_sek"] > 0)
    gross_loss = abs(sum(t["pnl_sek"] for t in trips if t["pnl_sek"] < 0))

    if gross_loss > 0:
        result["profit_factor"] = round(gross_profit / gross_loss, 4)

    # Average hold time
    hold_hours = [t["hold_hours"] for t in trips if t["hold_hours"] > 0]
    if hold_hours:
        result["avg_hold_hours"] = round(sum(hold_hours) / len(hold_hours), 2)

    # Trade frequency
    try:
        timestamps = []
        for t in trips:
            ts_str = t.get("sell_ts") or t.get("buy_ts")
            if ts_str:
                timestamps.append(datetime.datetime.fromisoformat(ts_str))
        if len(timestamps) >= 2:
            span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
            if span_days > 0:
                result["trade_frequency_per_week"] = round(len(trips) / span_days * 7, 2)
    except (ValueError, TypeError):
        pass

    # Win/loss stats
    wins = [t for t in trips if t["pnl_pct"] > 0]
    losses = [t for t in trips if t["pnl_pct"] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    total_count = len(trips)

    avg_win_pct = sum(t["pnl_pct"] for t in wins) / win_count if wins else 0
    avg_loss_pct = abs(sum(t["pnl_pct"] for t in losses) / loss_count) if losses else 0

    if avg_loss_pct > 0:
        result["win_loss_ratio"] = round(avg_win_pct / avg_loss_pct, 4)

    # Streaks
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    for t in trips:
        if t["pnl_pct"] > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    result["max_consecutive_wins"] = max_wins
    result["max_consecutive_losses"] = max_losses

    # Expectancy
    win_rate = win_count / total_count if total_count > 0 else 0
    loss_rate = loss_count / total_count if total_count > 0 else 0
    result["expectancy_pct"] = round(
        (win_rate * avg_win_pct) - (loss_rate * avg_loss_pct), 4
    )

    # Total P&L
    result["total_pnl_sek"] = round(sum(t["pnl_sek"] for t in trips), 2)

    # Calmar ratio (annualized return / max drawdown)
    # Compute a mini equity curve from round-trip PnLs
    if initial_value > 0 and len(trips) >= 2:
        equity = [initial_value]
        for t in trips:
            equity.append(equity[-1] + t["pnl_sek"])

        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else 0
        try:
            first_ts = datetime.datetime.fromisoformat(trips[0]["buy_ts"])
            last_ts = datetime.datetime.fromisoformat(trips[-1]["sell_ts"])
            days = (last_ts - first_ts).total_seconds() / 86400
            if days >= 1 and max_dd > 0:
                years = days / 365.25
                annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
                result["calmar_ratio"] = round(annualized / max_dd, 4)
        except (ValueError, TypeError):
            pass

    return result


if __name__ == "__main__":
    curve = load_equity_curve()
    if not curve:
        print("No equity curve data found in", DEFAULT_HISTORY_PATH)
        print("Run risk_management.log_portfolio_value() to start logging.")
    else:
        print(f"Loaded {len(curve)} data points")
        results = compare_strategies(curve)

        for strategy in ("patient", "bold"):
            m = results[strategy]
            print(f"\n{'='*50}")
            print(f"  {strategy.upper()} STRATEGY METRICS")
            print(f"{'='*50}")
            print(f"  Total return:     {m['total_return_pct']:+.2f}%")
            print(f"  Max drawdown:     {m['max_drawdown_pct']:.2f}%")
            if m["sharpe_ratio"] is not None:
                print(f"  Sharpe ratio:     {m['sharpe_ratio']:.2f}")
            if m["sortino_ratio"] is not None:
                print(f"  Sortino ratio:    {m['sortino_ratio']:.2f}")
            print(f"  Win rate:         {m['win_rate']:.1f}%")
            print(f"  Best day:         {m['best_day_pct']:+.2f}%")
            print(f"  Worst day:        {m['worst_day_pct']:+.2f}%")
            print(f"  Days in drawdown: {m['days_in_drawdown']}")
            print(f"  Volatility (ann): {m['volatility_annual_pct']:.2f}%")
            if m["date_range"]:
                print(f"  Date range:       {m['date_range'][0][:10]} to {m['date_range'][1][:10]}")

        c = results["comparison"]
        print(f"\n{'='*50}")
        print("  COMPARISON")
        print(f"{'='*50}")
        print(f"  Return leader:    {c['leader']} (by {abs(c['return_diff_pct']):.2f}%)")
        print(f"  Lower drawdown:   {c['lower_drawdown']} (by {abs(c['drawdown_diff_pct']):.2f}%)")
        if "sharpe_leader" in c:
            print(f"  Better Sharpe:    {c['sharpe_leader']} (by {abs(c['sharpe_diff']):.2f})")
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\exposure_coach.py'" in Q:\fa-adv-2026-05-11
 succeeded in 722ms:
"""Exposure coach - portfolio-level exposure recommendation.

Synthesizes market health score, regime detection, and portfolio state
into an exposure_ceiling multiplier (0.0-1.0) that scales maximum
allocation.  This is ADVISORY - it goes into agent_summary.json for
Layer 2 to consider, and optionally scales Kelly sizing.

Does NOT create new triggers or invoke Layer 2.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger("portfolio.exposure_coach")

# Exposure ceiling floors (never go below this)
MIN_CEILING = 0.2

# Zone-based base ceilings
_ZONE_CEILINGS = {
    "danger": 0.3,
    "caution": 0.6,
    "healthy": 1.0,
}

# Regime adjustments (multiplicative)
_REGIME_MULTIPLIERS = {
    "trending-down": 0.7,
    "high-vol": 0.8,
    "range-bound": 0.9,
    "trending-up": 1.0,
}

# Bias classification
_BIAS_MAP = {
    "danger": "defensive",
    "caution": "defensive",
    "healthy": "neutral",
}


def compute_exposure_recommendation(
    market_health: dict | None = None,
    regime: str = "range-bound",
    portfolio_concentration: float | None = None,
) -> dict:
    """Compute portfolio-level exposure recommendation.

    Args:
        market_health: output of market_health.get_market_health()
        regime: current regime from indicators.detect_regime()
        portfolio_concentration: fraction of portfolio in single largest
            position (0.0-1.0), used to flag concentration risk

    Returns:
        dict with exposure_ceiling, rationale, bias, etc.
    """
    # Default: no data = neutral recommendation
    if market_health is None:
        return {
            "exposure_ceiling": 1.0,
            "rationale": "No market health data available - using default exposure",
            "market_health_zone": "unknown",
            "market_health_score": None,
            "regime": regime,
            "new_entries_allowed": True,
            "bias": "neutral",
            "updated_at": datetime.now(UTC).isoformat(),
        }

    zone = market_health.get("zone", "healthy")
    score = market_health.get("score", 50)

    # Base ceiling from zone
    ceiling = _ZONE_CEILINGS.get(zone, 1.0)

    # Regime adjustment
    regime_mult = _REGIME_MULTIPLIERS.get(regime, 1.0)
    ceiling *= regime_mult

    # Floor enforcement
    ceiling = max(ceiling, MIN_CEILING)

    # Round to 2 decimal places
    ceiling = round(ceiling, 2)

    # New entries allowed?
    # Block new entries only in danger zone with bearish regime
    new_entries = not (zone == "danger" and regime in ("trending-down", "high-vol"))

    # Bias
    if zone == "danger" or (zone == "caution" and regime == "trending-down"):
        bias = "defensive"
    elif zone == "healthy" and regime == "trending-up":
        bias = "growth"
    else:
        bias = "neutral"

    # Build rationale
    parts = []
    parts.append(f"Market {zone} (score {score})")
    if regime != "range-bound":
        parts.append(f"{regime} regime")
    if portfolio_concentration and portfolio_concentration > 0.3:
        parts.append(f"high concentration ({portfolio_concentration:.0%})")
    rationale = " + ".join(parts)

    return {
        "exposure_ceiling": ceiling,
        "rationale": rationale,
        "market_health_zone": zone,
        "market_health_score": score,
        "regime": regime,
        "new_entries_allowed": new_entries,
        "bias": bias,
        "updated_at": datetime.now(UTC).isoformat(),
    }
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command 'rg -n "''^def _resolve_fx_rate|''^def _compute_portfolio_value|''^def check_drawdown|''^def compute_atr_stop_levels|''^def compute_atr_stops|''^def compute_mc_stop|''^def compute_cost_report|''^def check_concentration|''^def check_atr_stop_proximity|''^def check_all_risk_flags|''^def log_portfolio_value|''^def get_risk_summary|''^def compute_portfolio_concentration|''^def compute_monte_carlo" Q:'"\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
 succeeded in 634ms:
121:def _resolve_fx_rate(agent_summary: dict) -> float:
189:def _compute_portfolio_value(portfolio: dict, agent_summary: dict) -> float:
218:def check_drawdown(portfolio_path: str, max_drawdown_pct: float = 20.0,
565:def log_portfolio_value(patient_path: str | None = None,
734:def check_concentration_risk(ticker, action, portfolio, agent_summary, strategy="patient"):
873:def check_atr_stop_proximity(ticker, action, portfolio, agent_summary):
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py' | Select-Object -First 240" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py' | Select-Object -Skip 240 -First 220" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py' | Select-Object -Skip 460 -First 220" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py' | Select-Object -Skip 680 -First 220" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py' | Select-Object -Skip 900 -First 180" in Q:\fa-adv-2026-05-11
 succeeded in 945ms:

    if agent_summary_path is None:
        agent_summary_path = str(DATA_DIR / "agent_summary.json")

    # If portfolio has no holdings, value is just cash
    if not portfolio.get("holdings"):
        current_value = portfolio.get("cash_sek", initial_value)
    else:
        summary = load_json(agent_summary_path, default={})
        if summary:
            current_value = _compute_portfolio_value(portfolio, summary)
        else:
            # Fallback: cash only. This is NOT truly conservative -
            # it ignores unrealized P&L on holdings. If the price
            # feed is stale while holdings are underwater, the
            # drawdown reading will look tiny and the circuit
            # breaker will never trip. Surface the blind spot via
            # WARNING so dashboards/oncall can see "feed stale"
            # rather than silently trusting the number. (2026-04-17
            # adversarial review.)
            holding_count = sum(
                1 for h in portfolio.get("holdings", {}).values()
                if h.get("shares", 0) > 0
            )
            logger.warning(
                "check_drawdown: agent_summary empty at %s - falling "
                "back to cash-only value for %d live position(s). "
                "Drawdown circuit breaker reading may be optimistic.",
                agent_summary_path, holding_count,
            )
            current_value = portfolio.get("cash_sek", initial_value)

    # Determine peak value from history file or initial value
    history_path = DATA_DIR / "portfolio_value_history.jsonl"
    pf_name = pathlib.Path(portfolio_path).stem  # e.g. "portfolio_state" or "portfolio_state_bold"
    is_bold = "bold" in pf_name
    value_key = "bold_value_sek" if is_bold else "patient_value_sek"

    # A-PR-2: Stream the FULL history (not just last 2000 entries) to find
    # the true historical peak. The 2000-entry tail only covered ~33h, so
    # any rally older than that fell off the back and the drawdown circuit
    # breaker became blind to multi-day peaks.
    peak_value = _streaming_max(history_path, value_key, floor=initial_value)

    # Also compare against current value in case it's a new peak
    if current_value > peak_value:
        peak_value = current_value

    # Guard against NaN/Inf in peak_value or current_value - corrupted
    # history or failed computation. NaN silently passes all comparison
    # checks (NaN > 50.0 is False), bypassing the circuit breaker.
    if not math.isfinite(peak_value) or not math.isfinite(current_value):
        logger.critical(
            "check_drawdown: non-finite value detected (peak=%.2f, current=%.2f) "
            "- treating as 100%% drawdown (fail-safe).",
            peak_value, current_value,
        )
        return {
            "breached": True,
            "current_drawdown_pct": 100.0,
            "peak_value": round(peak_value, 2) if math.isfinite(peak_value) else 0.0,
            "current_value": round(current_value, 2) if math.isfinite(current_value) else 0.0,
            "initial_value": initial_value,
        }

    # Calculate drawdown
    if peak_value > 0:
        current_drawdown_pct = ((peak_value - current_value) / peak_value) * 100
    else:
        current_drawdown_pct = 0.0

    return {
        "breached": current_drawdown_pct > max_drawdown_pct,
        "current_drawdown_pct": round(current_drawdown_pct, 4),
        "peak_value": round(peak_value, 2),
        "current_value": round(current_value, 2),
        "initial_value": initial_value,
    }


def compute_stop_levels(holdings: dict, agent_summary: dict) -> dict:
    """Compute ATR-based stop-loss levels for all positions.

    For each holding with shares > 0, calculates stop-loss levels based on
    2x ATR (Average True Range) from the entry price.

    Args:
        holdings: The "holdings" dict from portfolio state.
            Each entry: {ticker: {"shares": N, "avg_cost_usd": X, ...}}
        agent_summary: Parsed agent_summary.json dict.

    Returns:
        dict keyed by ticker, each with:
            - entry_price_usd: float (avg_cost_usd)
            - current_price_usd: float (from agent_summary)
            - atr_pct: float (from agent_summary)
            - stop_price_usd: float (entry_price * (1 - 2 * atr_pct/100))
            - triggered: bool (current_price < stop_price)
            - distance_to_stop_pct: float (positive = above stop, negative = below)
            - pnl_pct: float (current vs entry)
    """
    signals = agent_summary.get("signals", {})
    result = {}

    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue

        entry_price = pos.get("avg_cost_usd", 0)
        if entry_price <= 0:
            continue

        # Get current price and ATR from agent_summary
        if ticker not in signals:
            # Ticker not in current summary (e.g., stock after hours)
            result[ticker] = {
                "entry_price_usd": entry_price,
                "current_price_usd": None,
                "atr_pct": None,
                "stop_price_usd": None,
                "triggered": False,
                "distance_to_stop_pct": None,
                "pnl_pct": None,
                "note": "No live data available (market closed or ticker not in summary)",
            }
            continue

        sig = signals[ticker]
        current_price = sig.get("price_usd", 0)
        atr_pct = sig.get("atr_pct", 0)

        # 2x ATR stop-loss - cap ATR at 15% to prevent meaninglessly wide stops for warrants
        atr_pct = min(atr_pct, 15.0)
        stop_price = entry_price * (1 - 2 * atr_pct / 100)

        # Distance from current price to stop
        if stop_price > 0:
            distance_to_stop_pct = ((current_price - stop_price) / current_price) * 100
        else:
            distance_to_stop_pct = float("inf")

        triggered = current_price < stop_price if current_price > 0 else False
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

        result[ticker] = {
            "entry_price_usd": round(entry_price, 6),
            "current_price_usd": round(current_price, 6),
            "atr_pct": round(atr_pct, 4),
            "stop_price_usd": round(stop_price, 6),
            "triggered": triggered,
            "distance_to_stop_pct": round(distance_to_stop_pct, 4),
            "pnl_pct": round(pnl_pct, 4),
        }

    return result


def compute_probabilistic_stops(holdings: dict, agent_summary: dict) -> dict:
    """Compute Monte Carlo probabilistic stop-loss quality for held positions.

    For each position, simulates remaining-session price paths and estimates
    the probability of hitting the ATR stop level, plus the expected time to
    hit if triggered.

    Args:
        holdings: The "holdings" dict from portfolio state.
        agent_summary: Parsed agent_summary.json dict.

    Returns:
        dict keyed by ticker, each with:
            - stop_price_usd: ATR-based stop level
            - stop_hit_prob: P(hitting stop this session), 0.0-1.0
            - expected_hit_time_min: E[time to stop | hit], or None
            - knockout_prob: For warrants, P(reaching financing level)
            - current_price_usd: current price
    """
    try:
        from portfolio.exit_optimizer import _first_hit_times, simulate_intraday_paths
        from portfolio.session_calendar import remaining_session_minutes
    except ImportError:
        logger.warning("exit_optimizer or session_calendar not available")
        return {}

    import numpy as np

    signals = agent_summary.get("signals", {})
    result = {}

    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue

        entry_price = pos.get("avg_cost_usd", 0)
        if entry_price <= 0:
            continue

        sig = signals.get(ticker)
        if not sig:
            continue

        current_price = sig.get("price_usd", 0)
        atr_pct = sig.get("atr_pct", 0)
        if current_price <= 0 or atr_pct <= 0:
            continue

        # Determine instrument type for session lookup
        if ticker.endswith("-USD") and ticker.startswith(("BTC", "ETH")):
            inst_type = "crypto"
        elif ticker.startswith(("XAG", "XAU")):
            inst_type = "warrant"
        else:
            inst_type = "stock"

        # Get remaining session minutes
        remaining = remaining_session_minutes(inst_type)
        if remaining < 2:
            continue

[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 948ms:
"""Risk management utilities for portfolio intelligence system.

Provides:
- Maximum drawdown circuit breaker
- ATR-based trailing stop-loss tracking
- Position age tracking
- Portfolio value history logging
- Transaction cost analysis
"""

import datetime
import json
import logging
import math
import pathlib
import threading

from portfolio.file_utils import atomic_append_jsonl, load_json
from portfolio.fx_rates import FX_RATE_FALLBACK, FX_RATE_MAX, FX_RATE_MIN

logger = logging.getLogger(__name__)


# FEAT-3: Cache peak values per (path, key) to avoid re-scanning the full
# JSONL on every 60s cycle. The file is append-only, so the peak can only
# increase. Cache stores: peak value + byte offset of last read position.
# Invalidated when the file shrinks (log rotation).
_peak_cache: dict[tuple, dict] = {}

# Adversarial review 04-29 PR-P1-2 (2026-05-02): the main loop's 8-worker
# ThreadPoolExecutor invokes check_drawdown() concurrently across the
# patient + bold portfolios, and update_health periodically calls
# _streaming_max. Without a lock, two threads could last-writer-wins on
# _peak_cache[cache_key] = {...}, occasionally losing a cached offset and
# forcing the next call to do a full O(file_size) scan instead of the
# O(delta) streaming read. Same class of bug as today's commit cdcbbd0f
# for signal_history.update_history. Lock scope: the entire read-decide-
# update sequence in _streaming_max - both reading the cached offset and
# writing back the new offset have to be atomic relative to other workers.
_peak_cache_lock = threading.Lock()


def _streaming_max(history_path: pathlib.Path, value_key: str, floor: float) -> float:
    """A-PR-2 (2026-04-11): Find the maximum value at `value_key` in a JSONL file.

    FEAT-3 (2026-04-29): Uses a byte-offset cache so subsequent calls only
    scan new entries appended since the last call. Falls back to a full scan
    if the file shrinks (rotation) or on any seek error.

    PR-P1-2 (2026-05-02): _peak_cache reads + writes are serialized under
    _peak_cache_lock to avoid the 8-worker ThreadPoolExecutor losing cached
    offsets to last-writer-wins races (see lock comment above).

    Streams line-by-line so memory stays O(1) regardless of file size.
    Returns `floor` (typically initial_value) if file missing/empty.
    """
    if not history_path.exists():
        return floor

    cache_key = (str(history_path), value_key)

    try:
        file_size = history_path.stat().st_size
    except OSError:
        file_size = 0

    # Snapshot cache under the lock - keeps the (read offset, decide
    # restart vs resume, hold last good peak) sequence consistent with
    # the matching write at the bottom of the function.
    with _peak_cache_lock:
        cached = _peak_cache.get(cache_key)
        if cached is not None:
            if file_size >= cached["offset"]:
                start_offset = cached["offset"]
                peak = cached["peak"]
            else:
                # File shrank (rotation) - full re-scan
                start_offset = 0
                peak = floor
        else:
            start_offset = 0
            peak = floor

    try:
        with open(history_path, encoding="utf-8") as f:
            if start_offset > 0:
                f.seek(start_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                val = entry.get(value_key, 0)
                if val > peak:
                    peak = val
            end_offset = f.tell()
    except OSError as e:
        logger.warning("Could not stream history file %s: %s", history_path.name, e)
        with _peak_cache_lock:
            cached_after = _peak_cache.get(cache_key)
        if cached_after is not None:
            return float(cached_after["peak"])
        return float(peak)

    with _peak_cache_lock:
        _peak_cache[cache_key] = {"peak": peak, "offset": end_offset}
    return float(peak)

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

INITIAL_VALUE_DEFAULT = 500_000  # SEK

# Adversarial review 05-01 P1-15 (2026-05-02): persistent fallback for fx_rate.
# FX constants imported from portfolio.fx_rates (single source of truth).
_FX_CACHE_FILENAME = "fx_rate_cache.json"


def _resolve_fx_rate(agent_summary: dict) -> float:
    """Return USDSEK rate, preferring (1) summary, (2) cached, (3) hardcoded.

    Adversarial review 05-01 P1-15: the original code did
    ``fx_rate = agent_summary.get("fx_rate", 1.0)``. When agent_summary was
    missing/empty/lacking the field - early loop cycle, agent_summary
    rotation, fx_rates.py crashed mid-fetch - the 1.0 default understated
    SEK valuations by ~10x and could trigger a false drawdown breach
    (a 5_435_000 SEK position valued at fx_rate=1.0 looks like 95%
    drawdown from the 500_000 SEK initial).

    Resolution order:
      1. ``agent_summary["fx_rate"]`` if it's a finite number in [7, 15].
         (Same sanity band as portfolio/fx_rates.py:42.) Successful values
         update the disk cache for future fallbacks.
      2. Cached rate from ``DATA_DIR/fx_rate_cache.json`` if present and
         in-band. The cache is best-effort: corrupt JSON or missing/invalid
         rate field is treated as no-cache.
      3. ``FX_RATE_FALLBACK`` (10.50) - matches
         portfolio/fx_rates.py:66 so both modules disagree-by-zero on the
         absolute worst-case path.

    Note: 1.0 is explicitly rejected by the sanity band, so the legacy
    pattern ``agent_summary.get("fx_rate", 1.0)`` continues to defer
    correctly even if a stale agent_summary still embeds 1.0 itself.
    """
    raw = agent_summary.get("fx_rate") if isinstance(agent_summary, dict) else None
    try:
        rate = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        rate = None
    if rate is not None and FX_RATE_MIN <= rate <= FX_RATE_MAX:
        # Cache the good rate for future fallback paths.
        try:
            from portfolio.file_utils import atomic_write_json
            atomic_write_json(DATA_DIR / _FX_CACHE_FILENAME, {
                "rate": rate,
                "ts": datetime.datetime.now(datetime.UTC).isoformat(),
            })
        except Exception as e:
            logger.debug("fx cache persist failed: %s", e)
        return rate

    # Try disk cache.
    cached = load_json(DATA_DIR / _FX_CACHE_FILENAME, default=None)
    if isinstance(cached, dict):
        # 2026-05-10 (codex re-review): cached.get("rate") is Any|None;
        # the except below catches the Nonefloat TypeError at runtime,
        # but mypy's strict arg-type check fires first. Explicit None
        # check makes the runtime guard visible to the type checker.
        rate_raw = cached.get("rate")
        if rate_raw is not None:
            try:
                cached_rate = float(rate_raw)
                if FX_RATE_MIN <= cached_rate <= FX_RATE_MAX:
                    return cached_rate
            except (TypeError, ValueError):
                pass

    logger.warning(
        "fx_rate fallback to hardcoded %.2f - agent_summary missing/invalid "
        "and no usable cache at %s. Portfolio valuations may be ~10%% off if "
        "SEK has moved.",
        FX_RATE_FALLBACK, DATA_DIR / _FX_CACHE_FILENAME,
    )
    return FX_RATE_FALLBACK


def _compute_portfolio_value(portfolio: dict, agent_summary: dict) -> float:
    """Compute current total portfolio value in SEK.

    Value = cash_sek + sum(shares * current_price_sek) for each holding.
    current_price_sek is derived from agent_summary prices * fx_rate.
    """
    cash = portfolio.get("cash_sek", 0)
    holdings = portfolio.get("holdings", {})
    # P1-15 (2026-05-02): use cached fallback chain instead of raw .get(..., 1.0).
    fx_rate = _resolve_fx_rate(agent_summary)
    signals = agent_summary.get("signals", {})

    holdings_value = 0.0
    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue
        # Try to get current price from agent_summary signals
        if ticker in signals:
            price_usd = signals[ticker].get("price_usd", 0)
            holdings_value += shares * price_usd * fx_rate
        else:
            # Fallback: use avg_cost_usd from holdings if no live price
            avg_cost = pos.get("avg_cost_usd", 0)
            holdings_value += shares * avg_cost * fx_rate

    return float(cash + holdings_value)


def check_drawdown(portfolio_path: str, max_drawdown_pct: float = 20.0,
                   agent_summary_path: str | None = None) -> dict:
    """Check if portfolio has exceeded maximum drawdown threshold.

    Computes current portfolio value against the initial value and the peak
    value recorded in portfolio_value_history.jsonl (if available).

    Args:
        portfolio_path: Path to portfolio_state JSON file.
        max_drawdown_pct: Maximum allowed drawdown percentage (default 20%).
        agent_summary_path: Path to agent_summary.json for live prices.
            If None, uses DATA_DIR / "agent_summary.json".

    Returns:
        dict with:
            - breached: bool -- True if drawdown exceeds threshold
            - current_drawdown_pct: float -- current drawdown from peak (positive number)
            - peak_value: float -- highest portfolio value seen
            - current_value: float -- current portfolio value in SEK
            - initial_value: float -- starting portfolio value
    """
    portfolio = load_json(portfolio_path, default={})
    initial_value = portfolio.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 923ms:
        # Estimate volatility from ATR
        import math
        vol = max(atr_pct / 100.0 * math.sqrt(252.0 / 14), 0.05)

        # ATR stop level
        stop_price = entry_price * (1 - 2 * atr_pct / 100)
        if stop_price <= 0:
            stop_price = entry_price * 0.01

        # Simulate paths
        paths = simulate_intraday_paths(
            price=current_price,
            volatility=vol,
            drift=0.0,
            remaining_minutes=int(remaining),
            instrument_type=inst_type,
            n_paths=2000,
        )

        # Stop hit probability
        session_min = np.min(paths[:, 1:], axis=1)
        stop_hit_prob = float(np.mean(session_min <= stop_price))

        # Expected time to hit stop
        hit_times = _first_hit_times(paths, stop_price, direction="below")
        hitting = hit_times[hit_times > 0]
        expected_hit_time = float(np.mean(hitting)) if len(hitting) > 0 else None

        result[ticker] = {
            "stop_price_usd": round(stop_price, 4),
            "stop_hit_prob": round(stop_hit_prob, 4),
            "expected_hit_time_min": round(expected_hit_time, 1) if expected_hit_time else None,
            "knockout_prob": None,
            "current_price_usd": round(current_price, 4),
        }

    return result


def get_position_ages(portfolio: dict) -> dict:
    """Calculate age of each position from first BUY transaction.

    Args:
        portfolio: Full portfolio state dict (with "holdings" and "transactions").

    Returns:
        dict keyed by ticker (only tickers currently held with shares > 0):
            - age_hours: float
            - age_days: float
            - first_buy: str (ISO-8601 timestamp of first BUY)
            - num_buys: int (total BUY transactions for this ticker)
            - num_sells: int (total SELL transactions for this ticker)
    """
    holdings = portfolio.get("holdings", {})
    transactions = portfolio.get("transactions", [])
    now = datetime.datetime.now(datetime.UTC)
    result = {}

    for ticker, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue

        # Find all BUY and SELL transactions for this ticker
        first_buy_ts = None
        num_buys = 0
        num_sells = 0

        for tx in transactions:
            if tx.get("ticker") != ticker:
                continue
            action = tx.get("action", "")
            ts_str = tx.get("timestamp", "")

            if action == "BUY":
                num_buys += 1
                try:
                    ts = datetime.datetime.fromisoformat(ts_str)
                    if first_buy_ts is None or ts < first_buy_ts:
                        first_buy_ts = ts
                except (ValueError, TypeError):
                    pass
            elif action == "SELL":
                num_sells += 1

        if first_buy_ts is not None:
            # Ensure timezone-aware comparison
            if first_buy_ts.tzinfo is None:
                first_buy_ts = first_buy_ts.replace(tzinfo=datetime.UTC)
            age_delta = now - first_buy_ts
            age_hours = age_delta.total_seconds() / 3600
            age_days = age_hours / 24

            result[ticker] = {
                "age_hours": round(age_hours, 2),
                "age_days": round(age_days, 2),
                "first_buy": first_buy_ts.isoformat(),
                "num_buys": num_buys,
                "num_sells": num_sells,
            }

    return result


def log_portfolio_value(patient_path: str | None = None,
                        bold_path: str | None = None,
                        agent_summary_path: str | None = None):
    """Append current portfolio values to data/portfolio_value_history.jsonl.

    Each entry contains:
        - ts: ISO-8601 UTC timestamp
        - patient_value_sek: total patient portfolio value
        - bold_value_sek: total bold portfolio value
        - patient_pnl_pct: patient P&L percentage
        - bold_pnl_pct: bold P&L percentage
        - prices: dict of current USD prices from agent_summary

    Args:
        patient_path: Path to patient portfolio state JSON.
        bold_path: Path to bold portfolio state JSON.
        agent_summary_path: Path to agent_summary.json.
    """
    if patient_path is None:
        patient_path = str(DATA_DIR / "portfolio_state.json")
    if bold_path is None:
        bold_path = str(DATA_DIR / "portfolio_state_bold.json")
    if agent_summary_path is None:
        agent_summary_path = str(DATA_DIR / "agent_summary.json")

    patient = load_json(patient_path, default={})
    bold = load_json(bold_path, default={})
    # P1-15 (2026-05-02): the {"fx_rate": 1.0} default here was load-bearing
    # for the false-circuit-breaker bug. Using a missing-fx_rate default lets
    # _resolve_fx_rate inside _compute_portfolio_value walk the cache chain
    # instead of taking the 1.0 at face value.
    summary = load_json(agent_summary_path, default={"signals": {}})

    patient_value = _compute_portfolio_value(patient, summary)
    bold_value = _compute_portfolio_value(bold, summary)

    patient_initial = patient.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
    bold_initial = bold.get("initial_value_sek", INITIAL_VALUE_DEFAULT)

    patient_pnl_pct = ((patient_value - patient_initial) / patient_initial) * 100 if patient_initial > 0 else 0
    bold_pnl_pct = ((bold_value - bold_initial) / bold_initial) * 100 if bold_initial > 0 else 0

    # Collect current prices
    prices = {}
    for ticker, sig in summary.get("signals", {}).items():
        price = sig.get("price_usd")
        if price is not None:
            prices[ticker] = price

    entry = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "patient_value_sek": round(patient_value, 2),
        "bold_value_sek": round(bold_value, 2),
        "patient_pnl_pct": round(patient_pnl_pct, 4),
        "bold_pnl_pct": round(bold_pnl_pct, 4),
        "fx_rate": _resolve_fx_rate(summary),  # P1-15 (2026-05-02)
        "prices": prices,
    }

    history_path = DATA_DIR / "portfolio_value_history.jsonl"
    atomic_append_jsonl(history_path, entry)


def transaction_cost_analysis(portfolio: dict) -> dict:
    """Analyze transaction costs for a portfolio.

    Examines all transactions and accumulated fees to produce a cost report.

    Args:
        portfolio: Full portfolio state dict.

    Returns:
        dict with:
            - total_fees_sek: float -- accumulated fees
            - fees_as_pct_of_initial: float -- total_fees / initial_value * 100
            - avg_fee_per_trade: float -- average fee per transaction
            - total_trades: int -- number of transactions
            - fees_as_pct_of_pnl: float | None -- total_fees / abs(pnl) * 100
              (None if no P&L to compare against)
            - total_buy_volume_sek: float -- sum of BUY allocs
            - total_sell_volume_sek: float -- sum of SELL proceeds
            - buy_count: int
            - sell_count: int
    """
    transactions = portfolio.get("transactions", [])
    initial_value = portfolio.get("initial_value_sek", INITIAL_VALUE_DEFAULT)
    cash = portfolio.get("cash_sek", initial_value)

    # total_fees_sek from portfolio state
    total_fees_from_state = portfolio.get("total_fees_sek", 0) or 0

    # Also compute fees from transaction records (fee_sek field)
    computed_fees = 0.0
    total_buy_volume = 0.0
    total_sell_volume = 0.0
    buy_count = 0
    sell_count = 0

    for tx in transactions:
        fee = tx.get("fee_sek", 0) or 0
        computed_fees += fee
        action = tx.get("action", "")
        total_sek = tx.get("total_sek", 0) or 0

        if action == "BUY":
            buy_count += 1
            total_buy_volume += total_sek
        elif action == "SELL":
            sell_count += 1
            total_sell_volume += total_sek

    # Use the larger of state fees vs computed fees (handles missing fee_sek fields)
    total_fees = max(total_fees_from_state, computed_fees)

    total_trades = len(transactions)
    avg_fee = total_fees / total_trades if total_trades > 0 else 0
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 942ms:
    # 2x ATR stop level
    stop_price = entry_price * (1 - 2 * atr_pct / 100)
    # Distance from current price to stop (in ATR units)
    atr_value = current_price * atr_pct / 100
    if atr_value <= 0:
        return None

    distance_to_stop = current_price - stop_price
    distance_in_atr = distance_to_stop / atr_value

    if distance_in_atr < 1.0:
        return {
            "flag": "atr_stop_proximity",
            "severity": "warning",
            "ticker": ticker,
            "current_price": round(current_price, 4),
            "stop_price": round(stop_price, 4),
            "distance_atr": round(distance_in_atr, 2),
            "message": (
                f"{ticker}: price ${current_price:.2f} is {distance_in_atr:.1f}x ATR "
                f"from stop ${stop_price:.2f} (danger zone < 1.0x ATR)"
            ),
        }
    return None


def compute_all_risk_flags(signals, patient_pf, bold_pf, agent_summary, config=None):
    """Compute all risk audit flags for all tickers.

    Args:
        signals: Dict of ticker -> signal data (from agent_summary).
        patient_pf: Patient portfolio state dict.
        bold_pf: Bold portfolio state dict.
        agent_summary: Full agent_summary dict.
        config: Optional config dict.

    Returns:
        dict with:
            - flags: list of flag dicts
            - summary: str (human-readable summary)
    """
    cfg = (config or {}).get("risk_audit", {})
    if cfg.get("enabled") is False:
        return {"flags": [], "summary": "Risk audit disabled"}

    all_flags = []

    for ticker, sig in signals.items():
        action = sig.get("action", "HOLD")
        if action == "HOLD":
            # Still check ATR proximity for held positions
            for strategy, pf in [("patient", patient_pf), ("bold", bold_pf)]:
                flag = check_atr_stop_proximity(ticker, "CHECK", pf, agent_summary)
                if flag:
                    flag["strategy"] = strategy
                    all_flags.append(flag)
            continue

        for strategy, pf in [("patient", patient_pf), ("bold", bold_pf)]:
            # Concentration
            flag = check_concentration_risk(ticker, action, pf, agent_summary, strategy)
            if flag:
                all_flags.append(flag)

            # Correlation
            flag = check_correlation_risk(ticker, action, pf, strategy)
            if flag:
                all_flags.append(flag)

            # ATR stop proximity
            flag = check_atr_stop_proximity(ticker, action, pf, agent_summary)
            if flag:
                flag["strategy"] = strategy
                all_flags.append(flag)

        # Regime mismatch (independent of strategy)
        flag = check_regime_mismatch(ticker, action, agent_summary)
        if flag:
            all_flags.append(flag)

    summary_parts = []
    if all_flags:
        from typing import Any as _Any
        by_flag: dict[str, list[dict[str, _Any]]] = {}
        for f in all_flags:
            by_flag.setdefault(f["flag"], []).append(f)
        for flag_name, flags in by_flag.items():
            summary_parts.append(f"{flag_name}: {len(flags)}")

    return {
        "flags": all_flags,
        "summary": "; ".join(summary_parts) if summary_parts else "All clear",
    }
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 982ms:

    fees_as_pct_initial = (total_fees / initial_value) * 100 if initial_value > 0 else 0

    # PnL: cash + holdings_value - initial_value
    # For simplicity here, we approximate with cash - initial (since holdings
    # value requires live prices). A full PnL needs _compute_portfolio_value.
    # However, if portfolio is all-cash (no holdings), this is exact.
    holdings = portfolio.get("holdings", {})
    has_open_positions = any(
        pos.get("shares", 0) > 0 for pos in holdings.values()
    )

    if has_open_positions:
        # PnL is approximate (doesn't include unrealized gains)
        pnl_note = "approximate (excludes unrealized gains/losses)"
        pnl = cash - initial_value  # unrealized not included
    else:
        pnl_note = "exact (all positions closed)"
        pnl = cash - initial_value

    if abs(pnl) > 0.01:
        fees_as_pct_pnl = (total_fees / abs(pnl)) * 100
    else:
        fees_as_pct_pnl = None

    return {
        "total_fees_sek": round(total_fees, 2),
        "fees_as_pct_of_initial": round(fees_as_pct_initial, 4),
        "avg_fee_per_trade": round(avg_fee, 2),
        "total_trades": total_trades,
        "fees_as_pct_of_pnl": round(fees_as_pct_pnl, 4) if fees_as_pct_pnl is not None else None,
        "total_buy_volume_sek": round(total_buy_volume, 2),
        "total_sell_volume_sek": round(total_sell_volume, 2),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "pnl_sek": round(pnl, 2),
        "pnl_note": pnl_note,
    }


# ---------------------------------------------------------------------------
# Risk Audit Flags - pre-trade risk checks for Layer 2
# ---------------------------------------------------------------------------

# Hard-coded correlation pairs based on known relationships
CORRELATED_PAIRS = {
    "ETH-USD": ["BTC-USD"],
    "BTC-USD": ["ETH-USD"],
    "XAG-USD": ["XAU-USD"],
    "XAU-USD": ["XAG-USD"],
}


def check_concentration_risk(ticker, action, portfolio, agent_summary, strategy="patient"):
    """Check if a new BUY would create excessive concentration.

    Args:
        ticker: Instrument to potentially buy.
        action: "BUY" or "SELL".
        portfolio: Portfolio state dict.
        agent_summary: Parsed agent_summary dict.
        strategy: "patient" or "bold".

    Returns:
        dict with flag info, or None if no risk.
    """
    if action != "BUY":
        return None

    cash = portfolio.get("cash_sek", 0)
    holdings = portfolio.get("holdings", {})
    # P1-15 (2026-05-02): use cached fallback chain instead of raw .get(..., 1.0).
    fx_rate = _resolve_fx_rate(agent_summary)
    signals = agent_summary.get("signals", {})

    # Compute current portfolio value
    total_value = cash
    for t, pos in holdings.items():
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue
        price = signals.get(t, {}).get("price_usd", pos.get("avg_cost_usd", 0))
        total_value += shares * price * fx_rate

    if total_value <= 0:
        return None

    # BUG-176: Compute proposed allocation as % of total portfolio, capped at cash
    alloc_pct = 0.30 if strategy == "bold" else 0.15
    proposed_alloc = min(total_value * alloc_pct, cash)

    # Existing position value for this ticker
    existing = holdings.get(ticker, {})
    existing_shares = existing.get("shares", 0)
    existing_price = signals.get(ticker, {}).get("price_usd", existing.get("avg_cost_usd", 0))
    existing_value = existing_shares * existing_price * fx_rate

    new_position_value = existing_value + proposed_alloc
    concentration_pct = (new_position_value / total_value) * 100

    if concentration_pct > 40:
        return {
            "flag": "concentration",
            "severity": "warning",
            "ticker": ticker,
            "strategy": strategy,
            "concentration_pct": round(concentration_pct, 1),
            "message": (
                f"{ticker} would be {concentration_pct:.1f}% of {strategy} portfolio "
                f"(>{40}% threshold)"
            ),
        }
    return None


def check_regime_mismatch(ticker, action, agent_summary):
    """Check if trade direction contradicts the market regime.

    BUY in trending-down (without volume confirmation) or
    SELL in trending-up is a regime mismatch.

    Returns:
        dict with flag info, or None if no mismatch.
    """
    if action == "HOLD":
        return None

    sig = agent_summary.get("signals", {}).get(ticker, {})
    regime = sig.get("regime", "ranging")
    extra = sig.get("extra", {})
    volume_ratio = extra.get("volume_ratio")

    mismatch = False
    reason = ""

    if action == "BUY" and regime == "trending-down":
        # BUY against downtrend - only flag with confirmed low volume.
        # Missing volume data (None) = unknown, not a mismatch.
        if volume_ratio is not None and volume_ratio < 1.5:
            mismatch = True
            reason = f"BUY in trending-down regime (RVOL={volume_ratio:.1f}, need >1.5x for reversal)"
    elif action == "SELL" and regime == "trending-up" and volume_ratio is not None and volume_ratio < 1.5:
        mismatch = True
        reason = f"SELL in trending-up regime (RVOL={volume_ratio:.1f}, need >1.5x for reversal)"

    if mismatch:
        return {
            "flag": "regime_mismatch",
            "severity": "warning",
            "ticker": ticker,
            "regime": regime,
            "action": action,
            "message": f"{ticker}: {reason}",
        }
    return None


def check_correlation_risk(ticker, action, portfolio, strategy="patient"):
    """Check if BUY would add correlated exposure to an existing position.

    Returns:
        dict with flag info, or None if no correlation risk.
    """
    if action != "BUY":
        return None

    correlated = CORRELATED_PAIRS.get(ticker, [])
    if not correlated:
        return None

    holdings = portfolio.get("holdings", {})
    held_correlated = []
    for t in correlated:
        pos = holdings.get(t, {})
        if pos.get("shares", 0) > 0:
            held_correlated.append(t)

    if held_correlated:
        return {
            "flag": "correlation",
            "severity": "warning",
            "ticker": ticker,
            "strategy": strategy,
            "correlated_held": held_correlated,
            "message": (
                f"{ticker}: correlated with held position(s) {', '.join(held_correlated)} "
                f"in {strategy} portfolio"
            ),
        }
    return None


def check_atr_stop_proximity(ticker, action, portfolio, agent_summary):
    """Check if current price is within 1x ATR of the computed stop level.

    This flags positions that are dangerously close to their stop-loss.

    Returns:
        dict with flag info, or None if no proximity risk.
    """
    if action == "HOLD":
        return None

    holdings = portfolio.get("holdings", {})
    pos = holdings.get(ticker, {})
    shares = pos.get("shares", 0)
    if shares <= 0:
        return None

    entry_price = pos.get("avg_cost_usd", 0)
    if entry_price <= 0:
        return None

    sig = agent_summary.get("signals", {}).get(ticker, {})
    current_price = sig.get("price_usd", 0)
    atr_pct = sig.get("atr_pct", 0)

    if current_price <= 0 or atr_pct <= 0:
        return None

[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py'" in Q:\fa-adv-2026-05-11
 succeeded in 643ms:
"""Warrant portfolio tracking - leverage-aware P&L for Avanza warrants.

Tracks actual warrant positions with leverage-multiplied P&L based on
the underlying instrument's price movement.
"""

import logging
from datetime import UTC
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.warrant_portfolio")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WARRANT_STATE_FILE = DATA_DIR / "portfolio_state_warrants.json"

_DEFAULT_STATE = {
    "holdings": {},
    "transactions": [],
}


def load_warrant_state():
    """Load warrant portfolio state from disk.

    Returns:
        dict with "holdings" and "transactions" keys.
    """
    state = load_json(WARRANT_STATE_FILE)
    if state is None:
        return _DEFAULT_STATE.copy()
    # Ensure required keys exist
    if "holdings" not in state:
        state["holdings"] = {}
    if "transactions" not in state:
        state["transactions"] = []
    return state


def save_warrant_state(state):
    """Atomically write warrant portfolio state.

    Args:
        state: dict with "holdings" and "transactions".
    """
    atomic_write_json(WARRANT_STATE_FILE, state)
    logger.info("Warrant state saved (%d holdings)", len(state.get("holdings", {})))


def warrant_pnl(holding, current_underlying_usd, fx_rate):
    """Compute P&L for a single warrant position.

    Uses the underlying price change multiplied by leverage factor.

    Args:
        holding: dict with keys:
            - units: number of warrant units held
            - entry_price_sek: price per unit at entry (SEK)
            - underlying: underlying ticker (e.g., "XAG-USD")
            - leverage: leverage factor (e.g., 5 for 5x)
            - underlying_entry_price_usd: underlying price at entry (USD)
            - name: human-readable name (optional)
        current_underlying_usd: current price of the underlying in USD.
        fx_rate: current USD/SEK exchange rate.

    Returns:
        dict: {
            "pnl_pct": float (percentage P&L),
            "pnl_sek": float (absolute P&L in SEK),
            "current_implied_sek": float (current implied value per unit),
            "total_value_sek": float (total current value),
            "entry_value_sek": float (total entry value),
            "underlying_change_pct": float (underlying price change %),
            "source": "implied"
        }
        Returns None if required data is missing.
    """
    if not holding or not current_underlying_usd or not fx_rate:
        return None

    units = holding.get("units", 0)
    entry_price_sek = holding.get("entry_price_sek", 0)
    leverage = holding.get("leverage", 1)
    underlying_entry = holding.get("underlying_entry_price_usd", 0)

    if not units or not entry_price_sek or not underlying_entry:
        return None

    # Underlying change
    underlying_change = (current_underlying_usd - underlying_entry) / underlying_entry
    underlying_change_pct = round(underlying_change * 100, 2)

    # Implied warrant P&L = underlying change * leverage
    implied_pnl_pct = underlying_change * leverage
    implied_pnl_pct_rounded = round(implied_pnl_pct * 100, 2)

    # Current implied value
    current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)
    total_value_sek = current_implied_sek * units
    entry_value_sek = entry_price_sek * units
    pnl_sek = round(total_value_sek - entry_value_sek, 2)

    return {
        "pnl_pct": implied_pnl_pct_rounded,
        "pnl_sek": pnl_sek,
        "current_implied_sek": round(current_implied_sek, 2),
        "total_value_sek": round(total_value_sek, 2),
        "entry_value_sek": round(entry_value_sek, 2),
        "underlying_change_pct": underlying_change_pct,
        "source": "implied",
    }


def get_warrant_summary(prices_usd, fx_rate):
    """Build a summary of all warrant positions with current P&L.

    Args:
        prices_usd: dict {ticker: price_usd} for all instruments.
        fx_rate: current USD/SEK exchange rate.

    Returns:
        dict: {
            "positions": {
                config_key: {
                    "name": "MINI L SILVER AVA 140",
                    "underlying": "XAG-USD",
                    "leverage": 5,
                    "units": 100,
                    "pnl": { ... warrant_pnl output ... },
                }
            },
            "total_value_sek": float,
            "total_pnl_sek": float,
        }
    """
    state = load_warrant_state()
    holdings = state.get("holdings", {})

    if not holdings:
        return {"positions": {}, "total_value_sek": 0, "total_pnl_sek": 0}

    positions = {}
    total_value = 0.0
    total_pnl = 0.0

    for key, holding in holdings.items():
        underlying = holding.get("underlying")
        if not underlying:
            continue

        current_price = prices_usd.get(underlying)
        if not current_price:
            continue

        pnl = warrant_pnl(holding, current_price, fx_rate)

        position = {
            "name": holding.get("name", key),
            "underlying": underlying,
            "leverage": holding.get("leverage", 1),
            "units": holding.get("units", 0),
        }

        if pnl:
            position["pnl"] = pnl
            total_value += pnl["total_value_sek"]
            total_pnl += pnl["pnl_sek"]
        else:
            position["pnl"] = None

        positions[key] = position

    return {
        "positions": positions,
        "total_value_sek": round(total_value, 2),
        "total_pnl_sek": round(total_pnl, 2),
    }


def record_warrant_transaction(config_key, action, units, price_sek, underlying_price_usd,
                                leverage, name=None, underlying=None):
    """Record a warrant buy/sell transaction.

    Args:
        config_key: Warrant config key (e.g., "MINI-SILVER").
        action: "BUY" or "SELL".
        units: Number of units.
        price_sek: Price per unit in SEK.
        underlying_price_usd: Underlying price at transaction time.
        leverage: Leverage factor.
        name: Human-readable name (optional).
        underlying: Underlying ticker (optional).
    """
    from datetime import datetime

    state = load_warrant_state()

    txn = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config_key": config_key,
        "action": action,
        "units": units,
        "price_sek": price_sek,
        "underlying_price_usd": underlying_price_usd,
        "leverage": leverage,
    }
    if name:
        txn["name"] = name
    if underlying:
        txn["underlying"] = underlying

    state["transactions"].append(txn)

    holdings = state["holdings"]
    if action == "BUY":
        if config_key in holdings:
            # Average in
            existing = holdings[config_key]
            old_units = existing.get("units", 0)
            old_price = existing.get("entry_price_sek", 0)
            new_units = old_units + units
            if new_units > 0:
                avg_price = (old_units * old_price + units * price_sek) / new_units
                existing["units"] = new_units
                existing["entry_price_sek"] = round(avg_price, 2)

                # PR-P1-1 (2026-05-02): also volume-weight the underlying
                # entry. Previously this stayed pinned to the FIRST entry's
                # spot price, causing the metals stop-loss reference (read
                # from `underlying_entry_price_usd` via warrant_pnl()) to
                # trip earlier than the VWAP entry implied - partial-add
                # positions would hard-stop on small underlying drawdowns.
                # Defensive: if either price is zero, fall back gracefully.
                # See tests/test_warrant_portfolio.TestWarrantAvgInUnderlyingEntry.
                old_underlying = existing.get("underlying_entry_price_usd", 0) or 0
                new_underlying = underlying_price_usd or 0
                if old_underlying > 0 and new_underlying > 0:
                    avg_underlying = (
                        old_units * old_underlying + units * new_underlying
                    ) / new_units
                    existing["underlying_entry_price_usd"] = round(avg_underlying, 4)
                elif new_underlying > 0:
                    # Existing was 0/missing - adopt the new valid spot.
                    existing["underlying_entry_price_usd"] = round(new_underlying, 4)
                # else: both zero - leave existing alone (degenerate input).
        else:
            holdings[config_key] = {
                "units": units,
                "entry_price_sek": price_sek,
                "underlying": underlying or "",
                "leverage": leverage,
                "underlying_entry_price_usd": underlying_price_usd,
                "name": name or config_key,
            }
    elif action == "SELL" and config_key in holdings:
        existing = holdings[config_key]
        remaining = existing.get("units", 0) - units
        if remaining <= 0:
            del holdings[config_key]
        else:
            existing["units"] = remaining

    save_warrant_state(state)
    logger.info("Warrant %s %s: %d units @ %.2f SEK", action, config_key, units, price_sek)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"financing|barrier|strike|leverage|underlying_entry_price_usd|entry_price_sek\" Q:\\finance-analyzer\\portfolio\\instrument_profile.py Q:\\finance-analyzer\\portfolio\\exit_optimizer.py Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py" in Q:\fa-adv-2026-05-11
 succeeded in 611ms:
Q:\finance-analyzer\portfolio\warrant_portfolio.py:1:"""Warrant portfolio tracking — leverage-aware P&L for Avanza warrants.
Q:\finance-analyzer\portfolio\warrant_portfolio.py:3:Tracks actual warrant positions with leverage-multiplied P&L based on
Q:\finance-analyzer\portfolio\warrant_portfolio.py:55:    Uses the underlying price change multiplied by leverage factor.
Q:\finance-analyzer\portfolio\warrant_portfolio.py:60:            - entry_price_sek: price per unit at entry (SEK)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:62:            - leverage: leverage factor (e.g., 5 for 5x)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:63:            - underlying_entry_price_usd: underlying price at entry (USD)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:84:    entry_price_sek = holding.get("entry_price_sek", 0)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:85:    leverage = holding.get("leverage", 1)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:86:    underlying_entry = holding.get("underlying_entry_price_usd", 0)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:88:    if not units or not entry_price_sek or not underlying_entry:
Q:\finance-analyzer\portfolio\warrant_portfolio.py:95:    # Implied warrant P&L = underlying change * leverage
Q:\finance-analyzer\portfolio\warrant_portfolio.py:96:    implied_pnl_pct = underlying_change * leverage
Q:\finance-analyzer\portfolio\warrant_portfolio.py:100:    current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:102:    entry_value_sek = entry_price_sek * units
Q:\finance-analyzer\portfolio\warrant_portfolio.py:129:                    "leverage": 5,
Q:\finance-analyzer\portfolio\warrant_portfolio.py:162:            "leverage": holding.get("leverage", 1),
Q:\finance-analyzer\portfolio\warrant_portfolio.py:183:                                leverage, name=None, underlying=None):
Q:\finance-analyzer\portfolio\warrant_portfolio.py:192:        leverage: Leverage factor.
Q:\finance-analyzer\portfolio\warrant_portfolio.py:207:        "leverage": leverage,
Q:\finance-analyzer\portfolio\warrant_portfolio.py:222:            old_price = existing.get("entry_price_sek", 0)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:227:                existing["entry_price_sek"] = round(avg_price, 2)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:232:                # from `underlying_entry_price_usd` via warrant_pnl()) to
Q:\finance-analyzer\portfolio\warrant_portfolio.py:237:                old_underlying = existing.get("underlying_entry_price_usd", 0) or 0
Q:\finance-analyzer\portfolio\warrant_portfolio.py:243:                    existing["underlying_entry_price_usd"] = round(avg_underlying, 4)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:246:                    existing["underlying_entry_price_usd"] = round(new_underlying, 4)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:251:                "entry_price_sek": price_sek,
Q:\finance-analyzer\portfolio\warrant_portfolio.py:253:                "leverage": leverage,
Q:\finance-analyzer\portfolio\warrant_portfolio.py:254:                "underlying_entry_price_usd": underlying_price_usd,
Q:\finance-analyzer\portfolio\exit_optimizer.py:67:    entry_price_sek: float               # What we paid per unit (SEK)
Q:\finance-analyzer\portfolio\exit_optimizer.py:71:    leverage: float = 1.0                # Effective leverage at entry
Q:\finance-analyzer\portfolio\exit_optimizer.py:72:    financing_level: float | None = None  # MINI future financing level (USD)
Q:\finance-analyzer\portfolio\exit_optimizer.py:312:        warrant_value = (underlying - financing_level) × usdsek
Q:\finance-analyzer\portfolio\exit_optimizer.py:320:    if position.instrument_type == "warrant" and position.financing_level is not None:
Q:\finance-analyzer\portfolio\exit_optimizer.py:321:        # MINI future: warrant price = (underlying - financing_level) × fx
Q:\finance-analyzer\portfolio\exit_optimizer.py:322:        exit_warrant_sek = (exit_price_usd - position.financing_level) * fx
Q:\finance-analyzer\portfolio\exit_optimizer.py:325:        entry_value = position.entry_price_sek * position.qty
Q:\finance-analyzer\portfolio\exit_optimizer.py:327:        # Leveraged product without explicit financing level
Q:\finance-analyzer\portfolio\exit_optimizer.py:329:        warrant_move = pct_move * position.leverage
Q:\finance-analyzer\portfolio\exit_optimizer.py:330:        exit_warrant_sek = position.entry_price_sek * (1 + warrant_move)
Q:\finance-analyzer\portfolio\exit_optimizer.py:333:        entry_value = position.entry_price_sek * position.qty
Q:\finance-analyzer\portfolio\exit_optimizer.py:345:    entry_value = position.entry_price_sek * position.qty
Q:\finance-analyzer\portfolio\exit_optimizer.py:373:    if position.financing_level and position.financing_level > 0:
Q:\finance-analyzer\portfolio\exit_optimizer.py:374:        distance_pct = (market.price - position.financing_level) / market.price * 100
Q:\finance-analyzer\portfolio\exit_optimizer.py:396:    if session_min is not None and position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:397:        stop_buffer = position.financing_level * 1.03  # 3% above financing
Q:\finance-analyzer\portfolio\exit_optimizer.py:419:    - Knock-out danger (< 3% from financing level)
Q:\finance-analyzer\portfolio\exit_optimizer.py:431:    if position.financing_level and position.financing_level > 0:
Q:\finance-analyzer\portfolio\exit_optimizer.py:432:        distance_pct = (market.price - position.financing_level) / market.price * 100
Q:\finance-analyzer\portfolio\exit_optimizer.py:434:            logger.warning("RISK OVERRIDE: Knock-out danger (%.1f%% from barrier), "
Q:\finance-analyzer\portfolio\exit_optimizer.py:445:    if session_min is not None and position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:446:        stop_buffer = position.financing_level * 1.03
Q:\finance-analyzer\portfolio\exit_optimizer.py:643:    elif position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:645:        stop_buffer = position.financing_level * 1.03
Q:\finance-analyzer\portfolio\exit_optimizer.py:686:    financing_level: float | None = None,
Q:\finance-analyzer\portfolio\exit_optimizer.py:687:    leverage: float = 1.0,
Q:\finance-analyzer\portfolio\exit_optimizer.py:701:        financing_level: For MINI futures, the knock-out level.
Q:\finance-analyzer\portfolio\exit_optimizer.py:702:        leverage: Effective leverage.
Q:\finance-analyzer\portfolio\exit_optimizer.py:731:    entry_price = position_state.get("entry_price_sek",
Q:\finance-analyzer\portfolio\exit_optimizer.py:744:        entry_price_sek=entry_price,
Q:\finance-analyzer\portfolio\exit_optimizer.py:748:        leverage=leverage,
Q:\finance-analyzer\portfolio\exit_optimizer.py:749:        financing_level=financing_level,
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py' | Select-Object -First 260" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py' | Select-Object -Skip 260 -First 260" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py' | Select-Object -Skip 520 -First 260" in Q:\fa-adv-2026-05-11
 succeeded in 733ms:
"""Quant exit engine - probabilistic exit optimization for intraday positions.

Three-layer architecture:
1. **Opportunity layer**: Monte Carlo path simulation for remaining-session
   price distribution (max/min/terminal).
2. **Execution layer**: Fill probability and time-to-hit estimation from
   simulated paths.
3. **Decision layer**: EV ranking of candidate exits, net of costs, with
   risk overrides (knock-out proximity, session end, volatility shock).

Designed for Avanza MINI futures (gold/silver warrants) but works for any
instrument with price, volatility, and session data.

Usage:
    from portfolio.exit_optimizer import compute_exit_plan, Position, MarketSnapshot
    plan = compute_exit_plan(position, market, session_end, cost_model)
    print(plan.recommended)  # Best exit by EV

Reference: docs/deep research/deep-research-report.md
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from portfolio.cost_model import CostModel, get_cost_model

logger = logging.getLogger("portfolio.exit_optimizer")

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSnapshot:
    """Current market state for the position's instrument.

    Prices are in the underlying's native currency (USD for metals/crypto).
    """
    asof_ts: datetime
    price: float              # Current underlying price (USD)
    bid: float | None = None
    ask: float | None = None
    volatility_annual: float | None = None   # Annualized vol (decimal)
    atr_pct: float | None = None             # ATR% for vol estimation
    usdsek: float = 10.85                       # FX rate
    drift: float = 0.0                          # Annualized drift (0 = neutral)


@dataclass(frozen=True)
class Position:
    """A held position to evaluate for exit.

    For warrants: prices in SEK, with underlying in USD.
    For stocks/crypto: prices in USD.
    """
    symbol: str                          # Underlying ticker (e.g., "XAG-USD")
    qty: float                           # Units held
    entry_price_sek: float               # What we paid per unit (SEK)
    entry_underlying_usd: float          # Underlying price at entry (USD)
    entry_ts: datetime
    instrument_type: str = "warrant"     # "warrant", "stock", "crypto"
    leverage: float = 1.0                # Effective leverage at entry
    financing_level: float | None = None  # MINI future financing level (USD)
    trailing_peak_usd: float | None = None  # Highest underlying since entry


@dataclass(frozen=True)
class CandidateExit:
    """A ranked exit candidate with probabilistic assessment.

    Attributes:
        price_usd: Target exit price in underlying USD.
        action: Exit method - "limit", "market", "hold_to_close".
        fill_prob: P(price reaches target before session end), 0.0-1.0.
        expected_fill_time_min: E[time to hit target | hit], in minutes.
        pnl_sek: Net P&L if filled at target price (after costs).
        ev_sek: Expected value = fill_prob x pnl + (1-fill_prob) x fallback.
        pnl_pct: P&L as percentage of position value.
        risk_flags: List of active risk warnings.
        quantile: Which quantile of session-max this candidate represents.
    """
    price_usd: float
    action: str
    fill_prob: float
    expected_fill_time_min: float
    pnl_sek: float
    ev_sek: float
    pnl_pct: float
    risk_flags: tuple[str, ...] = ()
    quantile: float | None = None


@dataclass
class ExitPlan:
    """Complete exit plan with ranked candidates.

    Attributes:
        symbol: Underlying ticker.
        asof_ts: When this plan was computed.
        remaining_minutes: Minutes until session close.
        candidates: All evaluated exit candidates, sorted by EV descending.
        recommended: The top candidate (highest EV, respecting risk overrides).
        market_exit: Immediate market exit candidate (always available).
        session_max_distribution: Quantiles of the remaining-session max price.
        session_min_distribution: Quantiles of the remaining-session min price.
        stop_hit_prob: P(price drops to stop level before session end).
        provenance: Audit trail (model version, parameters, data sources).
    """
    symbol: str
    asof_ts: datetime
    remaining_minutes: float
    candidates: list[CandidateExit]
    recommended: CandidateExit
    market_exit: CandidateExit
    session_max_distribution: dict[str, float] = field(default_factory=dict)
    session_min_distribution: dict[str, float] = field(default_factory=dict)
    stop_hit_prob: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary for logging/Telegram."""
        rec = self.recommended
        mkt = self.market_exit
        return (
            f"Exit plan: recommended {rec.action} @ ${rec.price_usd:.2f} "
            f"(EV {rec.ev_sek:+,.0f} SEK, fill {rec.fill_prob:.0%}, "
            f"{rec.expected_fill_time_min:.0f}min) | "
            f"market exit {mkt.pnl_sek:+,.0f} SEK | "
            f"{self.remaining_minutes:.0f}min left"
        )

    def to_dict(self) -> dict:
        """Serialize for JSON (agent_summary integration)."""
        return {
            "symbol": self.symbol,
            "remaining_min": round(self.remaining_minutes),
            "recommended": {
                "price": round(self.recommended.price_usd, 2),
                "action": self.recommended.action,
                "fill_prob": round(self.recommended.fill_prob, 3),
                "ev_sek": round(self.recommended.ev_sek),
                "pnl_pct": round(self.recommended.pnl_pct, 2),
                "time_min": round(self.recommended.expected_fill_time_min),
                "risk_flags": list(self.recommended.risk_flags),
            },
            "market_exit_sek": round(self.market_exit.pnl_sek),
            "stop_hit_prob": round(self.stop_hit_prob, 3),
            "session_max": self.session_max_distribution,
            "session_min": self.session_min_distribution,
            "n_candidates": len(self.candidates),
        }


# ---------------------------------------------------------------------------
# Intraday Monte Carlo path engine
# ---------------------------------------------------------------------------

# Trading minutes per day by instrument type (for annualization)
_TRADING_MINUTES = {
    "warrant": 820,    # 08:15-21:55 CET = ~13.67h
    "stock": 390,      # 6.5h
    "crypto": 1440,    # 24h
}
_TRADING_DAYS_PER_YEAR = 252
_MIN_VOLATILITY = 0.05  # 5% annualized floor


def _estimate_volatility(market: MarketSnapshot) -> float:
    """Get annualized volatility from market snapshot."""
    if market.volatility_annual and market.volatility_annual > _MIN_VOLATILITY:
        return market.volatility_annual
    if market.atr_pct and market.atr_pct > 0:
        # Convert ATR% (14-period) to annualized vol
        atr_frac = market.atr_pct / 100.0
        return max(atr_frac * math.sqrt(252.0 / 14), _MIN_VOLATILITY)
    return 0.20  # Default 20% annual vol


def simulate_intraday_paths(
    price: float,
    volatility: float,
    drift: float,
    remaining_minutes: int,
    instrument_type: str = "warrant",
    n_paths: int = 5000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM price paths at 1-minute resolution.

    Uses antithetic variates for variance reduction (~50% lower variance).

    Args:
        price: Current underlying price (USD).
        volatility: Annualized volatility (decimal, e.g., 0.25 = 25%).
        drift: Annualized drift (decimal). 0 = neutral.
        remaining_minutes: Minutes until session close.
        instrument_type: For annualization ("warrant", "stock", "crypto").
        n_paths: Number of paths to simulate. Even number recommended.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_paths, remaining_minutes + 1) where column 0 is
        the current price and each subsequent column is 1 minute later.
    """
    rng = np.random.default_rng(seed)

    n_steps = max(1, int(remaining_minutes))
    min_per_year = _TRADING_MINUTES.get(instrument_type, 390) * _TRADING_DAYS_PER_YEAR
    dt = 1.0 / min_per_year  # 1 minute in annualized trading time

    vol = max(volatility, _MIN_VOLATILITY)
    drift_per_step = (drift - 0.5 * vol ** 2) * dt
    vol_per_step = vol * math.sqrt(dt)

    # Antithetic variates: generate half, mirror the other half
    n_half = n_paths // 2
    Z = rng.standard_normal((n_half, n_steps))
    Z_all = np.vstack([Z, -Z])

    # If odd n_paths, add one extra
    if n_paths % 2 == 1:
        extra = rng.standard_normal((1, n_steps))
        Z_all = np.vstack([Z_all, extra])

    # Log-return increments  cumulative  price paths
    log_inc = drift_per_step + vol_per_step * Z_all  # (n_paths, n_steps)
    log_cum = np.cumsum(log_inc, axis=1)

    # Prepend zero column (current price)
    zeros = np.zeros((Z_all.shape[0], 1))
    log_paths = np.hstack([zeros, log_cum])

    paths = price * np.exp(log_paths)
    return paths


def _path_statistics(paths: np.ndarray) -> dict:
    """Extract key statistics from simulated paths.

    Returns:
        Dict with session_max, session_min, terminal arrays and quantile dicts.
    """
    session_max = np.max(paths[:, 1:], axis=1)  # Exclude t=0
    session_min = np.min(paths[:, 1:], axis=1)
    terminal = paths[:, -1]

    quantiles = [5, 10, 20, 35, 50, 65, 80, 90, 95]
    max_q = {f"p{q}": round(float(v), 4)
             for q, v in zip(quantiles, np.percentile(session_max, quantiles))}
    min_q = {f"p{q}": round(float(v), 4)
             for q, v in zip(quantiles, np.percentile(session_min, quantiles))}
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 735ms:

    return {
        "session_max": session_max,
        "session_min": session_min,
        "terminal": terminal,
        "max_quantiles": max_q,
        "min_quantiles": min_q,
    }


def _first_hit_times(paths: np.ndarray, target: float, direction: str = "above") -> np.ndarray:
    """Compute first passage time for each path to reach target.

    Args:
        paths: Price paths, shape (n_paths, n_steps+1).
        target: Price level to hit.
        direction: "above" (sell target) or "below" (stop level).

    Returns:
        Array of shape (n_paths,). Values are minute indices (1-based).
        -1 means the path never hit the target.
    """
    if direction == "above":
        hits = paths[:, 1:] >= target
    else:
        hits = paths[:, 1:] <= target

    # argmax on axis=1 returns first True index (0-based in the sliced array)
    first_idx = np.argmax(hits, axis=1)

    # Distinguish never-hit: if first_idx=0 but that cell isn't True  never hit
    never_hit = ~np.any(hits, axis=1)
    result = first_idx + 1  # Convert to 1-based minute index
    result[never_hit] = -1

    return result


# ---------------------------------------------------------------------------
# P&L computation
# ---------------------------------------------------------------------------

def _compute_pnl_sek(
    position: Position,
    exit_price_usd: float,
    market: MarketSnapshot,
    costs: CostModel,
) -> float:
    """Compute net P&L in SEK for exiting at given underlying price.

    For warrants (MINI futures):
        warrant_value = (underlying - financing_level) x usdsek
        pnl = (exit_value - entry_value) x qty - costs

    For stocks/crypto:
        pnl = (exit_price - entry_price) x qty x usdsek - costs
    """
    fx = market.usdsek

    if position.instrument_type == "warrant" and position.financing_level is not None:
        # MINI future: warrant price = (underlying - financing_level) x fx
        exit_warrant_sek = (exit_price_usd - position.financing_level) * fx
        exit_warrant_sek = max(exit_warrant_sek, 0)  # Can't go below 0 (knock-out)
        exit_value = exit_warrant_sek * position.qty
        entry_value = position.entry_price_sek * position.qty
    elif position.instrument_type == "warrant":
        # Leveraged product without explicit financing level
        pct_move = (exit_price_usd - position.entry_underlying_usd) / position.entry_underlying_usd
        warrant_move = pct_move * position.leverage
        exit_warrant_sek = position.entry_price_sek * (1 + warrant_move)
        exit_warrant_sek = max(exit_warrant_sek, 0)
        exit_value = exit_warrant_sek * position.qty
        entry_value = position.entry_price_sek * position.qty
    else:
        # Direct position (stock/crypto)
        exit_value = position.qty * exit_price_usd * fx
        entry_value = position.qty * position.entry_underlying_usd * fx

    cost = costs.total_cost_sek(exit_value)
    return exit_value - entry_value - cost


def _pnl_pct(pnl_sek: float, position: Position) -> float:
    """P&L as percentage of initial investment."""
    entry_value = position.entry_price_sek * position.qty
    if entry_value <= 0:
        return 0.0
    return pnl_sek / entry_value * 100.0


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------

def _compute_risk_flags(
    target_price: float | None,
    position: Position,
    market: MarketSnapshot,
    remaining_minutes: float,
    session_max: np.ndarray | None = None,
    session_min: np.ndarray | None = None,
) -> list[str]:
    """Generate risk warnings for a candidate exit."""
    flags = []

    # 1. Session end proximity
    if remaining_minutes < 30:
        flags.append("SESSION_END_IMMINENT")
    elif remaining_minutes < 60:
        flags.append("SESSION_END_NEAR")

    # 2. Knock-out proximity (MINI futures)
    if position.financing_level and position.financing_level > 0:
        distance_pct = (market.price - position.financing_level) / market.price * 100
        if distance_pct < 3:
            flags.append("KNOCKOUT_DANGER")
        elif distance_pct < 8:
            flags.append("KNOCKOUT_WARNING")

    # 3. Target far from current price (low fill probability expected)
    if target_price and market.price > 0:
        target_distance_pct = abs(target_price - market.price) / market.price * 100
        if target_distance_pct > 5:
            flags.append("TARGET_DISTANT")

    # 4. Underlying session mismatch (warrant still trading but underlying closed)
    # This would be detected by session_calendar, passed as a flag

    # 5. Position aging
    if position.entry_ts:
        hold_hours = (market.asof_ts - position.entry_ts).total_seconds() / 3600
        if hold_hours > 5:
            flags.append("HOLD_TIME_EXTENDED")

    # 6. Stop-loss proximity from MC paths
    if session_min is not None and position.financing_level:
        stop_buffer = position.financing_level * 1.03  # 3% above financing
        p_knockout = float(np.mean(session_min <= stop_buffer))
        if p_knockout > 0.10:
            flags.append(f"KNOCKOUT_PROB_{p_knockout:.0%}")

    return flags


# ---------------------------------------------------------------------------
# Risk overrides
# ---------------------------------------------------------------------------

def _apply_risk_overrides(
    candidates: list[CandidateExit],
    position: Position,
    market: MarketSnapshot,
    remaining_minutes: float,
    session_min: np.ndarray | None = None,
) -> CandidateExit:
    """Apply hard risk overrides and select recommended exit.

    Risk overrides can force a market exit even if EV says hold:
    - Knock-out danger (< 3% from financing level)
    - Session end imminent (< 5 min remaining)
    - Stop probability too high (> 25% chance of knock-out)
    """
    if not candidates:
        raise ValueError("No candidates to evaluate")

    # Find the market exit candidate
    market_exits = [c for c in candidates if c.action == "market"]
    market_exit = market_exits[0] if market_exits else candidates[-1]

    # Override 1: Knock-out danger  force market exit
    if position.financing_level and position.financing_level > 0:
        distance_pct = (market.price - position.financing_level) / market.price * 100
        if distance_pct < 3:
            logger.warning("RISK OVERRIDE: Knock-out danger (%.1f%% from barrier), "
                           "forcing market exit", distance_pct)
            return market_exit

    # Override 2: Session about to end  force market exit
    if remaining_minutes < 5:
        logger.info("RISK OVERRIDE: Session ending in %.0f min, forcing market exit",
                     remaining_minutes)
        return market_exit

    # Override 3: High knock-out probability  prefer market exit
    if session_min is not None and position.financing_level:
        stop_buffer = position.financing_level * 1.03
        p_knockout = float(np.mean(session_min <= stop_buffer))
        if p_knockout > 0.25:
            logger.warning("RISK OVERRIDE: %.0f%% knock-out probability, "
                           "forcing market exit", p_knockout * 100)
            return market_exit

    # No override triggered - return highest-EV candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_N_PATHS = 5000
DEFAULT_QUANTILES = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]


def compute_exit_plan(
    position: Position,
    market: MarketSnapshot,
    session_end: datetime,
    costs: CostModel | None = None,
    *,
    n_paths: int = DEFAULT_N_PATHS,
    quantiles: list[float] | None = None,
    stop_price_usd: float | None = None,
    seed: int | None = None,
) -> ExitPlan:
    """Compute a full exit plan for a held position.

    This is the main function. It:
    1. Simulates remaining-session price paths (Monte Carlo GBM)
    2. Extracts session-max/min distributions
    3. Generates candidate exits at quantile levels of session max
    4. Computes fill probability, time-to-hit, and EV for each
    5. Adds market exit and hold-to-close baselines
    6. Ranks by EV and applies risk overrides

    Args:
        position: The held position to evaluate.
        market: Current market snapshot.
        session_end: UTC datetime of session close.
        costs: Cost model. If None, auto-selects by instrument type.
        n_paths: Number of Monte Carlo paths.
        quantiles: Quantile levels for candidate generation.
        stop_price_usd: Explicit stop level (for stop-hit probability).
        seed: Random seed for reproducibility.

    Returns:
        ExitPlan with ranked candidates and recommendation.
    """
    if costs is None:
        costs = get_cost_model(position.instrument_type)

    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    now = market.asof_ts
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if session_end.tzinfo is None:
        session_end = session_end.replace(tzinfo=UTC)

    remaining_min = max(0, (session_end - now).total_seconds() / 60)

    # ---- Edge case: session over or almost over ----
    if remaining_min < 1:
        mkt_pnl = _compute_pnl_sek(position, market.bid or market.price, market, costs)
        mkt_candidate = CandidateExit(
            price_usd=market.bid or market.price,
            action="market",
            fill_prob=1.0,
            expected_fill_time_min=0,
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 718ms:
            pnl_sek=mkt_pnl,
            ev_sek=mkt_pnl,
            pnl_pct=_pnl_pct(mkt_pnl, position),
            risk_flags=("SESSION_ENDED",),
        )
        return ExitPlan(
            symbol=position.symbol,
            asof_ts=now,
            remaining_minutes=0,
            candidates=[mkt_candidate],
            recommended=mkt_candidate,
            market_exit=mkt_candidate,
            provenance={"reason": "session_ended"},
        )

    # ---- 1. Estimate volatility and simulate paths ----
    vol = _estimate_volatility(market)
    drift = market.drift

    paths = simulate_intraday_paths(
        price=market.price,
        volatility=vol,
        drift=drift,
        remaining_minutes=int(remaining_min),
        instrument_type=position.instrument_type,
        n_paths=n_paths,
        seed=seed,
    )

    # ---- 2. Extract path statistics ----
    stats = _path_statistics(paths)
    session_max = stats["session_max"]
    session_min = stats["session_min"]
    terminal = stats["terminal"]

    # ---- 3. Generate candidate exits at session-max quantiles ----
    target_prices = np.quantile(session_max, quantiles)
    candidates: list[CandidateExit] = []

    # Fallback P&L: what we get if we hold to close (median terminal price)
    median_terminal = float(np.median(terminal))
    fallback_pnl = _compute_pnl_sek(position, median_terminal, market, costs)

    for q, target in zip(quantiles, target_prices):
        target = float(target)

        # Skip if target is below current price (can't profit from selling below)
        if target <= market.price * 0.999:
            continue

        # Fill probability: fraction of paths where max >= target
        fill_prob = float(np.mean(session_max >= target))

        # Time to hit
        hit_times = _first_hit_times(paths, target, direction="above")
        hitting_times = hit_times[hit_times > 0]
        expected_time = float(np.mean(hitting_times)) if len(hitting_times) > 0 else remaining_min

        # P&L if filled
        pnl = _compute_pnl_sek(position, target, market, costs)

        # Expected value: fill_prob x conditional_pnl + (1-fill_prob) x fallback
        ev = fill_prob * pnl + (1 - fill_prob) * fallback_pnl

        flags = _compute_risk_flags(target, position, market, remaining_min,
                                     session_max, session_min)

        candidates.append(CandidateExit(
            price_usd=round(target, 4),
            action="limit",
            fill_prob=round(fill_prob, 4),
            expected_fill_time_min=round(expected_time, 1),
            pnl_sek=round(pnl, 2),
            ev_sek=round(ev, 2),
            pnl_pct=round(_pnl_pct(pnl, position), 2),
            risk_flags=tuple(flags),
            quantile=q,
        ))

    # ---- 4. Market exit candidate (immediate fill, certain) ----
    bid = market.bid or market.price
    mkt_pnl = _compute_pnl_sek(position, bid, market, costs)
    market_candidate = CandidateExit(
        price_usd=round(bid, 4),
        action="market",
        fill_prob=1.0,
        expected_fill_time_min=0,
        pnl_sek=round(mkt_pnl, 2),
        ev_sek=round(mkt_pnl, 2),
        pnl_pct=round(_pnl_pct(mkt_pnl, position), 2),
        risk_flags=tuple(_compute_risk_flags(None, position, market, remaining_min)),
    )
    candidates.append(market_candidate)

    # ---- 5. Hold-to-close candidate ----
    # EV of holding = mean terminal P&L (expected value across all paths)
    terminal_pnls = np.array([
        _compute_pnl_sek(position, float(p), market, costs)
        for p in np.percentile(terminal, [10, 25, 50, 75, 90])
    ])
    hold_ev = float(np.mean(terminal_pnls))

    hold_candidate = CandidateExit(
        price_usd=round(median_terminal, 4),
        action="hold_to_close",
        fill_prob=1.0,
        expected_fill_time_min=round(remaining_min, 1),
        pnl_sek=round(fallback_pnl, 2),
        ev_sek=round(hold_ev, 2),
        pnl_pct=round(_pnl_pct(fallback_pnl, position), 2),
        risk_flags=tuple(_compute_risk_flags(None, position, market, remaining_min,
                                              session_max, session_min)),
    )
    candidates.append(hold_candidate)

    # ---- 6. Sort by EV descending ----
    candidates.sort(key=lambda c: c.ev_sek, reverse=True)

    # ---- 7. Stop-loss hit probability ----
    stop_prob = 0.0
    if stop_price_usd and stop_price_usd > 0:
        stop_prob = float(np.mean(session_min <= stop_price_usd))
    elif position.financing_level:
        # Use knock-out level + 3% buffer as effective stop
        stop_buffer = position.financing_level * 1.03
        stop_prob = float(np.mean(session_min <= stop_buffer))

    # ---- 8. Apply risk overrides to select recommendation ----
    recommended = _apply_risk_overrides(
        candidates, position, market, remaining_min, session_min
    )

    return ExitPlan(
        symbol=position.symbol,
        asof_ts=now,
        remaining_minutes=round(remaining_min, 1),
        candidates=candidates,
        recommended=recommended,
        market_exit=market_candidate,
        session_max_distribution=stats["max_quantiles"],
        session_min_distribution=stats["min_quantiles"],
        stop_hit_prob=round(stop_prob, 4),
        provenance={
            "model": "GBM_antithetic",
            "n_paths": n_paths,
            "volatility": round(vol, 4),
            "drift": round(drift, 4),
            "remaining_min": round(remaining_min),
            "instrument_type": position.instrument_type,
            "cost_model": costs.label,
        },
    )


# ---------------------------------------------------------------------------
# Convenience: compute exit plan from existing system data
# ---------------------------------------------------------------------------

def compute_exit_plan_from_summary(
    ticker: str,
    agent_summary: dict,
    position_state: dict,
    session_end: datetime,
    *,
    instrument_type: str = "warrant",
    financing_level: float | None = None,
    leverage: float = 1.0,
    n_paths: int = DEFAULT_N_PATHS,
) -> ExitPlan | None:
    """Build exit plan from agent_summary and portfolio state data.

    Convenience wrapper that extracts price, volatility, and position data
    from the standard system data structures.

    Args:
        ticker: Underlying ticker (e.g., "XAG-USD").
        agent_summary: Agent summary dict with signals and prices.
        position_state: Position dict with shares, avg_cost, entry info.
        session_end: Session close time (UTC).
        instrument_type: "warrant", "stock", "crypto".
        financing_level: For MINI futures, the knock-out level.
        leverage: Effective leverage.
        n_paths: MC paths.

    Returns:
        ExitPlan or None if insufficient data.
    """
    signals = agent_summary.get("signals", {})
    ticker_data = signals.get(ticker, {})
    if not ticker_data:
        return None

    price = ticker_data.get("price_usd", 0)
    if price <= 0:
        return None

    extra = ticker_data.get("extra", {})
    atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct")
    fx_rate = agent_summary.get("fx_rate", 10.85)

    # Build MarketSnapshot
    market = MarketSnapshot(
        asof_ts=datetime.now(UTC),
        price=price,
        atr_pct=atr_pct,
        usdsek=fx_rate,
    )

    # Build Position
    shares = position_state.get("shares", position_state.get("qty", 0))
    entry_price = position_state.get("entry_price_sek",
                                      position_state.get("entry_price", 0))
    entry_underlying = position_state.get("entry_underlying_usd",
                                           position_state.get("entry_underlying", price))
    entry_ts_str = position_state.get("entry_ts")
    entry_ts = datetime.now(UTC)
    if entry_ts_str:
        with contextlib.suppress(ValueError, TypeError):
            entry_ts = datetime.fromisoformat(entry_ts_str)

    position = Position(
        symbol=ticker,
        qty=shares,
        entry_price_sek=entry_price,
        entry_underlying_usd=entry_underlying,
        entry_ts=entry_ts,
        instrument_type=instrument_type,
        leverage=leverage,
        financing_level=financing_level,
    )

    return compute_exit_plan(position, market, session_end, n_paths=n_paths)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\instrument_profile.py'" in Q:\fa-adv-2026-05-11
 succeeded in 721ms:
"""Instrument profiles for smart fishing - per-metal signal trust and behavior.

Each metal gets a "personality" defining:
- Which signals to trust (>70% historical accuracy on this ticker)
- Which signals to ignore (<45% accuracy, noise)
- Cross-asset drivers with correlation and lead time
- Regime-specific behavior (direction bias, TP multiplier)
- Precomputed context file path
- Typical volatility characteristics

Usage:
    from portfolio.instrument_profile import get_profile
    profile = get_profile("XAG-USD")
    trusted = profile["trusted_signals"]
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Signal trust tiers - derived from signal_reliability in agent_summary
# Updated periodically based on accuracy_cache.json
# ---------------------------------------------------------------------------

_SILVER_TRUSTED = [
    "econ_calendar",       # 94.7% (396 samples)
    "fear_greed",          # 91.5% (212)
    "claude_fundamental",  # 85.4% (247)
    "momentum_factors",    # 76.7% (630)
    "structure",           # 76.4% (467)
    "mean_reversion",      # 72.9% (3d horizon)
    "fibonacci",           # 68.2% (decent, context-dependent)
]

_SILVER_IGNORED = [
    "sentiment",           # 4.5% (89) - noise
    "ministral",           # 18.9% (95) - LLM fails on silver
    "oscillators",         # 40.6% (143) - below gate
    "custom_lora",         # disabled globally
    "ml",                  # disabled globally
]

_GOLD_TRUSTED = [
    "econ_calendar",       # high accuracy on metals broadly
    "macro_regime",        # gold responds to macro shifts
    "claude_fundamental",  # LLM analysis works for gold
    "fibonacci",           # structural levels reliable
    "structure",           # breakout detection
    "smart_money",         # institutional flow detection
]

_GOLD_IGNORED = [
    "sentiment",           # noise for metals
    "custom_lora",         # disabled
    "ml",                  # disabled
]

# ---------------------------------------------------------------------------
# Cross-asset drivers - what moves this instrument and how fast
# ---------------------------------------------------------------------------

_SILVER_CROSS_ASSETS = {
    "DXY": {
        "ticker": "DX-Y.NYB",
        "correlation": -0.65,
        "lead_minutes": 15,
        "description": "Strong USD = weak silver",
        "threshold_pct": 0.3,  # alert if DXY moves >0.3%
    },
    "copper": {
        "ticker": "HG=F",
        "correlation": 0.55,
        "lead_minutes": 30,
        "description": "Industrial demand proxy - copper leads silver",
        "threshold_pct": 1.5,
    },
    "gold": {
        "ticker": "GC=F",
        "correlation": 0.85,
        "lead_minutes": 5,
        "description": "Primary correlation - gold leads silver in most moves",
        "threshold_pct": 0.5,
    },
    "gold_silver_ratio": {
        "ticker": None,  # computed from GC=F / SI=F
        "mean": 62,
        "revert_threshold": 5,  # alert if ratio deviates >5 from mean
        "description": "High ratio = silver undervalued relative to gold",
    },
    "oil": {
        "ticker": "CL=F",
        "correlation": 0.30,
        "lead_minutes": 60,
        "description": "Inflation proxy - oil up = metals up",
        "threshold_pct": 2.0,
    },
}

_GOLD_CROSS_ASSETS = {
    "DXY": {
        "ticker": "DX-Y.NYB",
        "correlation": -0.70,
        "lead_minutes": 10,
        "description": "Strong USD = weak gold (strongest inverse correlation)",
        "threshold_pct": 0.3,
    },
    "real_yields": {
        "ticker": "^TNX",
        "correlation": -0.60,
        "lead_minutes": 60,
        "description": "Rising real yields = gold opportunity cost rises",
        "threshold_pct": 0.05,  # yield moves in absolute terms
    },
    "VIX": {
        "ticker": "^VIX",
        "correlation": 0.40,
        "lead_minutes": 15,
        "description": "Fear gauge - VIX spike = gold safe haven bid",
        "threshold_pct": 5.0,
    },
    "SPY": {
        "ticker": "SPY",
        "correlation": -0.25,
        "lead_minutes": 30,
        "description": "Risk-off = gold up, but weak correlation",
        "threshold_pct": 0.8,
    },
}

# ---------------------------------------------------------------------------
# Full instrument profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict[str, Any]] = {
    "XAG-USD": {
        "name": "Silver",
        "binance_symbol": "XAGUSDT",
        "trusted_signals": _SILVER_TRUSTED,
        "ignored_signals": _SILVER_IGNORED,
        "cross_asset_drivers": _SILVER_CROSS_ASSETS,
        "regime_behaviors": {
            "trending-up": {
                "preferred_direction": "LONG",
                "tp_multiplier": 1.5,
                "conviction_boost": 10,
            },
            "trending-down": {
                "preferred_direction": "SHORT",
                "tp_multiplier": 1.5,
                "conviction_boost": 10,
            },
            "ranging": {
                "preferred_direction": "BOTH",
                "tp_multiplier": 1.0,
                "conviction_boost": 0,
            },
            "high-vol": {
                "preferred_direction": "SHORT",  # vol expansion favors shorts
                "tp_multiplier": 2.0,
                "conviction_boost": -5,  # lower conviction in chaos
            },
        },
        "precompute_file": "data/silver_deep_context.json",
        "prophecy_key": "silver_bull_2026",
        "typical_daily_range_pct": 5.0,
        "typical_hourly_vol_pct": 0.4,
        "margin_hike_risk_levels": [100, 120],  # CME hikes near these
        "overnight_correlation_with_asia": 0.3,
    },
    "XAU-USD": {
        "name": "Gold",
        "binance_symbol": "XAUUSDT",
        "trusted_signals": _GOLD_TRUSTED,
        "ignored_signals": _GOLD_IGNORED,
        "cross_asset_drivers": _GOLD_CROSS_ASSETS,
        "regime_behaviors": {
            "trending-up": {
                "preferred_direction": "LONG",
                "tp_multiplier": 1.3,
                "conviction_boost": 10,
            },
            "trending-down": {
                "preferred_direction": "SHORT",
                "tp_multiplier": 1.3,
                "conviction_boost": 10,
            },
            "ranging": {
                "preferred_direction": "BOTH",
                "tp_multiplier": 1.0,
                "conviction_boost": 0,
            },
            "high-vol": {
                "preferred_direction": "LONG",  # gold is safe haven
                "tp_multiplier": 1.5,
                "conviction_boost": 5,
            },
        },
        "precompute_file": "data/gold_deep_context.json",
        "prophecy_key": None,
        "typical_daily_range_pct": 2.9,
        "typical_hourly_vol_pct": 0.25,
        "margin_hike_risk_levels": [],
        "overnight_correlation_with_asia": 0.5,  # gold more Asia-sensitive
    },
}


def get_profile(ticker: str) -> dict[str, Any] | None:
    """Get instrument profile for a ticker, or None if not profiled."""
    return PROFILES.get(ticker)


def get_trusted_signals(ticker: str) -> list[str]:
    """Get list of trusted signal names for this ticker."""
    profile = PROFILES.get(ticker)
    return profile["trusted_signals"] if profile else []


def get_ignored_signals(ticker: str) -> list[str]:
    """Get list of signals to ignore for this ticker."""
    profile = PROFILES.get(ticker)
    return profile["ignored_signals"] if profile else []


def get_cross_asset_drivers(ticker: str) -> dict[str, dict]:
    """Get cross-asset driver configuration for this ticker."""
    profile = PROFILES.get(ticker)
    return profile["cross_asset_drivers"] if profile else {}


def get_regime_behavior(ticker: str, regime: str) -> dict[str, Any]:
    """Get regime-specific behavior for this ticker."""
    profile = PROFILES.get(ticker)
    if not profile:
        return {"preferred_direction": "BOTH", "tp_multiplier": 1.0, "conviction_boost": 0}
    return profile["regime_behaviors"].get(
        regime,
        {"preferred_direction": "BOTH", "tp_multiplier": 1.0, "conviction_boost": 0},
    )


def format_profile_briefing(ticker: str, signal_data: dict | None = None) -> str:
    """Format a human-readable instrument briefing.

    Parameters
    ----------
    ticker : str
        Instrument ticker (e.g., "XAG-USD")
    signal_data : dict, optional
        Signal reliability data from agent_summary_compact.json

    Returns
    -------
    str
        Formatted briefing text
    """
    profile = PROFILES.get(ticker)
    if not profile:
        return f"No profile available for {ticker}"

    lines = [
        f"  Instrument: {profile['name']} ({ticker})",
        f"  Typical daily range: {profile['typical_daily_range_pct']:.1f}%",
        f"  Hourly vol: {profile['typical_hourly_vol_pct']:.2f}%",
    ]

    # Signal trust tiers
    if signal_data:
        reliability = signal_data.get("signal_reliability", {}).get(ticker, {})
        if reliability:
            trusted_with_acc = []
            for sig in profile["trusted_signals"]:
                acc_data = reliability.get(sig, {})
                if acc_data:
                    acc = acc_data.get("accuracy", 0)
                    n = acc_data.get("total", 0)
                    trusted_with_acc.append(f"{sig} ({acc:.0%}/{n})")
                else:
                    trusted_with_acc.append(sig)
            lines.append(f"  Trusted signals: {', '.join(trusted_with_acc)}")
        else:
            lines.append(f"  Trusted signals: {', '.join(profile['trusted_signals'])}")
    else:
        lines.append(f"  Trusted signals: {', '.join(profile['trusted_signals'])}")

    lines.append(f"  Ignored signals: {', '.join(profile['ignored_signals'])}")

    # Cross-asset drivers
    lines.append("  Cross-asset drivers:")
    for name, driver in profile["cross_asset_drivers"].items():
        corr = driver.get("correlation", 0)
        lead = driver.get("lead_minutes", 0)
        desc = driver.get("description", "")
        if lead:
            lines.append(f"    {name}: corr {corr:+.2f}, leads {lead}min -- {desc}")
        else:
            lines.append(f"    {name}: {desc}")

    # Prophecy
    if profile.get("prophecy_key"):
        lines.append(f"  Active prophecy: {profile['prophecy_key']}")

    return "\n".join(lines)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\monte_carlo.py'" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py' | Select-Object -First 280" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py' | Select-Object -Skip 280 -First 280" in Q:\fa-adv-2026-05-11
 succeeded in 757ms:
"""Portfolio VaR with t-copula correlated simulation.

Computes Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), and
joint drawdown probabilities for multi-position portfolios using a Student-t
copula for tail dependence.

The t-copula captures the empirical fact that assets crash together more
often than a Gaussian copula predicts (tail dependence lambda ~0.18 at
df=4 vs. 0 for Gaussian).

Usage:
    from portfolio.monte_carlo_risk import compute_portfolio_var
    result = compute_portfolio_var(portfolio_state, agent_summary)
    # result = {
    #   "var_95_usd": -1234.56,
    #   "cvar_95_usd": -1567.89,
    #   "var_99_usd": -2345.67,
    #   "total_exposure_usd": 50000.0,
    #   ...
    # }
"""

import logging
import math

import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_dist

from portfolio.fx_rates import FX_RATE_FALLBACK
from portfolio.monte_carlo import (
    MIN_VOLATILITY,
    drift_from_probability,
    volatility_from_atr,
)

logger = logging.getLogger("portfolio.monte_carlo_risk")

DEFAULT_DF = 4        # Degrees of freedom for t-copula (4 = moderate fat tails)
DEFAULT_N_PATHS = 10_000


# ---------------------------------------------------------------------------
# Correlation matrix estimation
# ---------------------------------------------------------------------------

def estimate_correlation_matrix(
    returns: dict[str, np.ndarray],
    tickers: list[str],
) -> np.ndarray:
    """Estimate correlation matrix from historical return series.

    For tickers with insufficient data, defaults to zero correlation
    (independent assumption) which is conservative for VaR.

    Args:
        returns: Dict mapping ticker  array of log-returns.
        tickers: Ordered list of tickers (defines matrix row/column order).

    Returns:
        Correlation matrix, shape (n, n), guaranteed symmetric and PSD.
    """
    n = len(tickers)
    if n == 0:
        return np.array([]).reshape(0, 0)

    corr = np.eye(n)

    # Build return matrix for tickers with data
    for i in range(n):
        for j in range(i + 1, n):
            ri = returns.get(tickers[i])
            rj = returns.get(tickers[j])
            if ri is not None and rj is not None:
                # Align lengths (use shorter)
                min_len = min(len(ri), len(rj))
                if min_len >= 20:  # Need at least 20 observations
                    c = np.corrcoef(ri[:min_len], rj[:min_len])[0, 1]
                    if np.isfinite(c):
                        corr[i, j] = c
                        corr[j, i] = c

    # Ensure PSD via eigenvalue clipping (Higham nearest PSD)
    corr = _nearest_psd(corr)
    return corr


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project matrix to nearest positive semi-definite correlation matrix.

    Uses eigenvalue clipping: set negative eigenvalues to a small positive
    value, then rescale diagonal to 1.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    # Reconstruct
    result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Rescale to correlation matrix (diagonal = 1)
    d = np.sqrt(np.diag(result))
    d[d == 0] = 1.0
    result = result / np.outer(d, d)
    # Enforce exact symmetry
    result = (result + result.T) / 2
    np.fill_diagonal(result, 1.0)
    return result


# ---------------------------------------------------------------------------
# Known correlation pairs (fallback when no historical data)
# ---------------------------------------------------------------------------

# From risk_management.py - approximate correlation strengths
CORRELATION_PRIORS = {
    ("BTC-USD", "ETH-USD"): 0.75,
    ("XAG-USD", "XAU-USD"): 0.85,
}


def _get_prior_correlation(ticker_a: str, ticker_b: str) -> float:
    """Look up prior correlation for a pair (order-independent)."""
    return CORRELATION_PRIORS.get(
        (ticker_a, ticker_b),
        CORRELATION_PRIORS.get((ticker_b, ticker_a), 0.0),
    )


def build_correlation_matrix(
    tickers: list[str],
    historical_returns: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build correlation matrix using historical data with prior fallback.

    If historical returns are available and sufficient, uses empirical
    estimates. Otherwise falls back to hardcoded priors from CORRELATED_PAIRS.

    Args:
        tickers: Ordered list of tickers.
        historical_returns: Optional dict of ticker  log-return arrays.

    Returns:
        Correlation matrix (n x n).
    """
    n = len(tickers)
    if n == 0:
        return np.array([]).reshape(0, 0)

    if historical_returns:
        # Check if we have sufficient data (>= 30 observations per ticker)
        has_data = sum(
            1 for t in tickers
            if t in historical_returns and len(historical_returns[t]) >= 30
        )
        if has_data >= 2:
            return estimate_correlation_matrix(historical_returns, tickers)

    # Fallback: use priors
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = _get_prior_correlation(tickers[i], tickers[j])
            corr[i, j] = c
            corr[j, i] = c

    return _nearest_psd(corr)


# ---------------------------------------------------------------------------
# Portfolio risk simulator
# ---------------------------------------------------------------------------

class PortfolioRiskSimulator:
    """Multi-position portfolio risk simulator using Student-t copula.

    Generates correlated terminal returns for all positions jointly,
    then computes portfolio-level P&L, VaR, and CVaR.

    The t-copula with df=4 captures tail dependence (assets crash together)
    that a Gaussian copula misses entirely.

    Args:
        positions: Dict mapping ticker  {shares, price_usd, volatility, drift}.
        correlation_matrix: Correlation matrix for position tickers.
        horizon_days: Simulation horizon.
        n_paths: Number of simulation paths.
        df: Degrees of freedom for t-copula (lower = fatter tails).
        seed: Random seed.
    """

    def __init__(
        self,
        positions: dict,
        correlation_matrix: np.ndarray,
        horizon_days: float = 1.0,
        n_paths: int = DEFAULT_N_PATHS,
        df: int = DEFAULT_DF,
        seed: int | None = None,
    ):
        # Filter out zero-share positions
        self._tickers = [t for t, p in positions.items() if p.get("shares", 0) != 0]
        self._positions = {t: positions[t] for t in self._tickers}
        self._n_assets = len(self._tickers)

        # If we filtered tickers, extract the sub-matrix
        if self._n_assets > 0 and correlation_matrix.size > 0:
            all_tickers = list(positions.keys())
            indices = [all_tickers.index(t) for t in self._tickers]
            self._corr = correlation_matrix[np.ix_(indices, indices)]
        else:
            self._corr = np.array([]).reshape(0, 0)

        self.horizon_days = horizon_days
        self.n_paths = n_paths
        self.df = df
        self.seed = seed

        self._returns = None  # (n_paths, n_assets) log-returns
        self._pnl = None      # (n_paths,) portfolio P&L in USD

    def simulate_correlated_returns(self) -> np.ndarray:
        """Generate correlated returns using t-copula.

        Algorithm:
        1. Cholesky decompose correlation matrix: L = chol(R)
        2. Generate independent standard normals: Z ~ N(0, I)
        3. Correlate: W = Z @ L^T
        4. Generate chi-squared: S ~ chi2(df)
        5. Scale to t: T = W * sqrt(df / S)  multivariate t
        6. Transform marginals: apply per-asset drift and volatility

        Returns:
            Array of shape (n_paths, n_assets) with log-returns.
        """
        if self._n_assets == 0:
            self._returns = np.empty((self.n_paths, 0))
            return self._returns

        rng = np.random.default_rng(self.seed)
        T = self.horizon_days / 252.0

        # Step 1: Cholesky decomposition
        try:
            L = np.linalg.cholesky(self._corr)
        except np.linalg.LinAlgError:
            logger.warning(
                "Cholesky decomposition failed for %d-asset correlation matrix - "
                "falling back to nearest PSD projection",
                self._n_assets,
            )
            L = np.linalg.cholesky(_nearest_psd(self._corr))

        # Step 2: Independent standard normals
        Z = rng.standard_normal((self.n_paths, self._n_assets))

        # Step 3: Correlate
        W = Z @ L.T

        # Step 4: Chi-squared scaling for t-distribution
        S = rng.chisquare(self.df, size=self.n_paths)

        # Step 5: Scale to multivariate t
        # T_i = W_i * sqrt(df / S) for each path
        scale = np.sqrt(self.df / S)[:, np.newaxis]
        T_samples = W * scale

        # Step 6: Transform to uniform via t CDF, then to Gaussian marginals
        # U = F_t(T; df)  uniform on [0,1]
        U = t_dist.cdf(T_samples, df=self.df)

        # Transform each marginal to GBM log-return
        # C9 FIX: Use norm.ppf (Gaussian inverse CDF), NOT t_dist.ppf.
        # Using t_dist.ppf(t_dist.cdf(x, df), df) is an identity transform -
        # it round-trips back to the same t-distributed samples, inflating
        # variance by ~sqrt(df/(df-2)) ÷ sqrt(2) at df=4. The correct
        # t-copula + GBM approach is: t-copula for dependence structure,
        # Gaussian marginals for the GBM diffusion term.
        returns = np.empty_like(U)
        for i, ticker in enumerate(self._tickers):
            pos = self._positions[ticker]
            sigma = max(pos["volatility"], MIN_VOLATILITY)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 762ms:
"""Monte Carlo price simulation engine.

Generates stochastic price paths using Geometric Brownian Motion (GBM)
with antithetic variates for variance reduction. Converts directional
signal probabilities into price quantile bands, stop-loss probabilities,
and expected return distributions.

Short-term focus: 3h, 1d, 3d horizons for active trading decisions.

Usage:
    from portfolio.monte_carlo import simulate_ticker
    result = simulate_ticker("BTC-USD", agent_summary)
    # result = {
    #   "price_bands_1d": {"p5": 82100, "p25": 84300, "p50": 86200, ...},
    #   "p_stop_hit_1d": 0.12,
    #   "expected_return_1d": {"mean_pct": 0.3, "std_pct": 2.1, "skew": 0.1},
    #   ...
    # }
"""

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger("portfolio.monte_carlo")

# Default parameters
DEFAULT_N_PATHS = 10_000   # 5K pairs with antithetic variates
DEFAULT_HORIZONS = [1, 3]  # days
MIN_VOLATILITY = 0.05      # 5% annualized floor (prevents degenerate sims)

# Per-asset-class ATR fallbacks when actual ATR is missing from signals.
# The generic 2.0% underestimates tail risk for crypto/metals.
_ATR_DEFAULT_BY_CLASS = {
    "crypto": 3.5,  # BTC 3-4% typical daily ATR
    "metals": 4.0,  # XAG 4-6%, XAU 2-3% - biased toward silver
    "stocks": 2.0,  # MSTR ~3% but broader equities lower
}


# ---------------------------------------------------------------------------
# Volatility & drift estimation from existing system data
# ---------------------------------------------------------------------------

def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
    """Convert ATR% (14-period) to annualized volatility.

    ATR measures average true range over `period` candles. For hourly candles
    (our primary timeframe), we annualize: vol = atr_frac * sqrt(trading_periods/period).

    Uses 252 trading days (standard for stocks/crypto).

    Args:
        atr_pct: ATR as percentage of price (e.g., 3.5 means 3.5%).
        period: ATR lookback period (default 14).

    Returns:
        Annualized volatility as a decimal (e.g., 0.20 = 20%).
    """
    atr_frac = atr_pct / 100.0
    annual_factor = math.sqrt(252.0 / period)
    vol = atr_frac * annual_factor
    return max(vol, MIN_VOLATILITY)


def drift_from_probability(p_up: float, volatility: float) -> float:
    """Convert directional probability P(up) into annualized drift.

    Uses the inverse of the GBM CDF relationship:
        P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / (sigma*sqrt(T)))

    For 1-day horizon (T = 1/252):
        mu = sigma * N_inv(p_up) * sqrt(252) + 0.5 * sigma^2

    This ensures the GBM simulation produces paths where the fraction
    ending above spot matches the input probability.

    Args:
        p_up: Probability of price being higher at horizon (0.0-1.0).
        volatility: Annualized volatility (decimal).

    Returns:
        Annualized drift (decimal). Positive = upward bias.
    """
    # Clamp p_up to avoid infinite drift at extremes
    p_up = max(0.01, min(0.99, p_up))

    # N_inv(p_up) gives the z-score for the desired probability
    z = norm.ppf(p_up)

    # mu = sigma * z * sqrt(252) + 0.5 * sigma^2
    # This is derived from P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / sigma*sqrt(T))
    mu = volatility * z * math.sqrt(252.0) + 0.5 * volatility**2

    return mu


def _atr_default_for_ticker(ticker: str) -> float:
    """Return the per-asset-class ATR fallback for a ticker."""
    from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS
    if ticker in CRYPTO_SYMBOLS:
        return _ATR_DEFAULT_BY_CLASS["crypto"]
    if ticker in METALS_SYMBOLS:
        return _ATR_DEFAULT_BY_CLASS["metals"]
    return _ATR_DEFAULT_BY_CLASS["stocks"]


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """Geometric Brownian Motion price path simulator with antithetic variates.

    Generates terminal price distributions for short-term horizons (hours to days).
    Uses antithetic variates for 50-75% variance reduction at zero extra cost.

    Args:
        price: Current asset price (USD).
        volatility: Annualized volatility (decimal, e.g., 0.20 = 20%).
        drift: Annualized drift (decimal). Use drift_from_probability() to derive.
        horizon_days: Simulation horizon in days (supports fractional, e.g., 0.125 = 3h).
        n_paths: Number of simulated terminal prices to generate.
        seed: Random seed for reproducibility.
    """

    def __init__(self, price: float, volatility: float, drift: float = 0.0,
                 horizon_days: float = 1.0, n_paths: int = DEFAULT_N_PATHS,
                 seed: int | None = None):
        self.price = price
        self.volatility = max(volatility, MIN_VOLATILITY)
        self.drift = drift
        self.horizon_days = horizon_days
        self.n_paths = n_paths
        self.seed = seed
        self._terminal_prices = None

    def simulate_paths(self) -> np.ndarray:
        """Generate terminal prices via GBM with antithetic variates.

        Formula: S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)

        Antithetic variates: for each random draw Z, we also compute the
        path with -Z. This creates negative correlation between paired
        estimates, reducing variance of the mean estimator.

        Returns:
            1D array of terminal prices, shape (n_paths,).
        """
        rng = np.random.default_rng(self.seed)

        T = self.horizon_days / 252.0  # Convert to years
        sigma = self.volatility
        mu = self.drift

        # Draw half the paths, use antithetic for the other half
        n_half = self.n_paths // 2
        Z = rng.standard_normal(n_half)

        # Deterministic component
        drift_term = (mu - 0.5 * sigma**2) * T
        vol_term = sigma * math.sqrt(T)

        # Original paths
        log_returns_pos = drift_term + vol_term * Z
        # Antithetic paths (negate Z)
        log_returns_neg = drift_term + vol_term * (-Z)

        # Combine
        log_returns = np.concatenate([log_returns_pos, log_returns_neg])
        terminal_prices = self.price * np.exp(log_returns)

        # If n_paths is odd, add one more path
        if self.n_paths % 2 == 1:
            extra_Z = rng.standard_normal(1)
            extra_price = self.price * np.exp(drift_term + vol_term * extra_Z)
            terminal_prices = np.concatenate([terminal_prices, extra_price])

        self._terminal_prices = terminal_prices
        return terminal_prices

    def _ensure_simulated(self):
        """Run simulation if not already done."""
        if self._terminal_prices is None:
            self.simulate_paths()

    def price_quantiles(self, percentiles: list[int] | None = None) -> dict:
        """Extract price quantile bands from simulated distribution.

        Args:
            percentiles: List of percentiles to compute (default: [5, 25, 50, 75, 95]).

        Returns:
            Dict mapping percentile  price (e.g., {5: 87.2, 50: 90.1, 95: 93.5}).
        """
        self._ensure_simulated()
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        values = np.percentile(self._terminal_prices, percentiles)
        return {p: round(float(v), 2) for p, v in zip(percentiles, values)}

    def probability_below(self, threshold: float) -> float:
        """Compute P(terminal price < threshold).

        Useful for stop-loss probability: "what's the chance price hits my stop?"

        Args:
            threshold: Price level to test.

        Returns:
            Probability (0.0-1.0).
        """
        self._ensure_simulated()
        if threshold <= 0:
            return 0.0
        return float(np.mean(self._terminal_prices < threshold))

    def probability_above(self, threshold: float) -> float:
        """Compute P(terminal price > threshold).

        Useful for profit target probability.

        Args:
            threshold: Price level to test.

        Returns:
            Probability (0.0-1.0).
        """
        self._ensure_simulated()
        if threshold <= 0:
            return 1.0
        return float(np.mean(self._terminal_prices > threshold))

    def expected_return(self) -> dict:
        """Compute return distribution statistics.

        Returns:
            Dict with mean_pct, std_pct, skew of the return distribution.
        """
        self._ensure_simulated()
        returns = (self._terminal_prices - self.price) / self.price * 100.0

        mean_pct = float(np.mean(returns))
        std_pct = float(np.std(returns))

        # Skewness
        if std_pct > 0:
            skew = float(np.mean(((returns - mean_pct) / std_pct) ** 3))
        else:
            skew = 0.0

        return {
            "mean_pct": round(mean_pct, 2),
            "std_pct": round(std_pct, 2),
            "skew": round(skew, 2),
        }


# ---------------------------------------------------------------------------
# Convenience: simulate a single ticker from agent_summary data
# ---------------------------------------------------------------------------

def simulate_ticker(ticker: str, agent_summary: dict,
                    n_paths: int = DEFAULT_N_PATHS,
                    horizons: list[int] | None = None,
                    seed: int | None = None) -> dict | None:
    """Simulate price distribution for a ticker using agent_summary data.

    Extracts price, ATR volatility, and directional probability from the
    signal data, then runs GBM simulation at each horizon.

    Args:
        ticker: Instrument ticker (e.g., "BTC-USD").
        agent_summary: Full agent_summary dict (or compact version).
        n_paths: Number of MC paths per horizon.
        horizons: List of horizon days (default: [1, 3]).
        seed: Random seed.

    Returns:
        Dict with price bands, stop probability, expected return per horizon.
        None if ticker not found in summary.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    signals = agent_summary.get("signals", {})
    ticker_data = signals.get(ticker)
    if not ticker_data:
        return None

    price = ticker_data.get("price_usd", 0)
    if price <= 0:
        return None

    extra = ticker_data.get("extra", {})
    atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct") or _atr_default_for_ticker(ticker)

    # Get volatility from ATR
    vol = volatility_from_atr(atr_pct)

    # Get directional probability for drift
    # Try to use existing probability computation if available
    p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
    drift = drift_from_probability(p_up, vol)

    # Compute ATR-based stop level (2x ATR below entry/current)
    stop_price = price * (1 - 2 * atr_pct / 100)

    result = {}

    for h in horizons:
        h_key = f"{h}d" if h >= 1 else f"{int(h * 24)}h"

        mc = MonteCarloEngine(
            price=price,
            volatility=vol,
            drift=drift,
            horizon_days=h,
            n_paths=n_paths,
            seed=seed,
        )
        mc.simulate_paths()

        result[f"price_bands_{h_key}"] = mc.price_quantiles()
        result[f"p_stop_hit_{h_key}"] = round(mc.probability_below(stop_price), 3)
        result[f"expected_return_{h_key}"] = mc.expected_return()

    # Add metadata
    result["price_usd"] = price
    result["atr_pct"] = round(atr_pct, 2)
    result["volatility_annual"] = round(vol, 3)
    result["p_up"] = round(p_up, 3)
    result["drift_annual"] = round(drift, 4)
    result["stop_price"] = round(stop_price, 2)

    return result


def _get_directional_probability(ticker: str, ticker_data: dict,
                                  agent_summary: dict) -> float:
    """Extract directional probability from available data.

    Priority:
    1. Pre-computed focus_probabilities (from ticker_accuracy.py)
    2. Weighted confidence + action from signal engine
    3. Fallback: 0.5 (no edge)
    """
    # 1. Check focus_probabilities in summary
    focus_probs = agent_summary.get("focus_probabilities", {})
    ticker_probs = focus_probs.get(ticker, {})
    prob_1d = ticker_probs.get("1d", {})
    if prob_1d and "probability" in prob_1d:
        return prob_1d["probability"]

    # 2. Derive from weighted confidence + action
    extra = ticker_data.get("extra", {})
    action = extra.get("_weighted_action") or ticker_data.get("action", "HOLD")
    conf = extra.get("_weighted_confidence") or ticker_data.get("weighted_confidence", 0.5)

    if action == "BUY":
        return 0.5 + conf * 0.3  # Scale to 0.5-0.8 range
    elif action == "SELL":
        return 0.5 - conf * 0.3  # Scale to 0.2-0.5 range
    else:
        return 0.5  # HOLD = no directional edge


# ---------------------------------------------------------------------------
# Batch simulation for all interesting tickers
# ---------------------------------------------------------------------------

def simulate_all(agent_summary: dict, tickers: list[str] | None = None,
                 n_paths: int = DEFAULT_N_PATHS, seed: int | None = None) -> dict:
    """Run MC simulation for multiple tickers.

    Args:
        agent_summary: Full agent summary dict.
        tickers: List of tickers to simulate. If None, uses held positions + focus tickers.
        n_paths: Paths per ticker per horizon.
        seed: Base seed (incremented per ticker for independence).

    Returns:
        Dict mapping ticker  simulation results.
    """
    if tickers is None:
        tickers = _interesting_tickers(agent_summary)

    results = {}
    for i, ticker in enumerate(tickers):
        try:
            # 2026-04-22: guard seed=None. reporting.py calls simulate_all
            # without a seed; `seed + i` was crashing every cycle with
            # TypeError and silently disabling MC risk sim for all tickers.
            ticker_seed = None if seed is None else seed + i
            result = simulate_ticker(ticker, agent_summary,
                                     n_paths=n_paths, seed=ticker_seed)
            if result:
                results[ticker] = result
        except Exception:
            logger.warning("MC simulation failed for %s", ticker, exc_info=True)

    return results


def _interesting_tickers(agent_summary: dict) -> list[str]:
    """Determine which tickers to simulate (held + focus + signaling)."""
    tickers = set()

    # Focus tickers (from config, surfaced in summary)
    for ft in agent_summary.get("focus_tickers", ["XAG-USD", "BTC-USD"]):
        tickers.add(ft)

    # Tickers with active consensus (BUY or SELL)
    for ticker, data in agent_summary.get("signals", {}).items():
        action = data.get("action", "HOLD")
        if action in ("BUY", "SELL"):
            tickers.add(ticker)

    return sorted(tickers)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 749ms:
            mu = pos.get("drift", 0.0)

            # Inverse normal CDF to get standard normal quantiles
            Z_marginal = norm.ppf(U[:, i])

            # GBM log-return: (mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z
            drift_term = (mu - 0.5 * sigma**2) * T
            vol_term = sigma * math.sqrt(T)
            returns[:, i] = drift_term + vol_term * Z_marginal

        self._returns = returns
        return returns

    def _ensure_simulated(self):
        """Run simulation if not already done."""
        if self._returns is None:
            self.simulate_correlated_returns()

    def portfolio_pnl(self) -> np.ndarray:
        """Compute portfolio P&L distribution in USD.

        P&L = sum over assets of (shares * price * (exp(log_return) - 1))

        Returns:
            1D array of portfolio P&L values, shape (n_paths,).
        """
        self._ensure_simulated()

        if self._n_assets == 0:
            self._pnl = np.zeros(self.n_paths)
            return self._pnl

        pnl = np.zeros(self.n_paths)
        for i, ticker in enumerate(self._tickers):
            pos = self._positions[ticker]
            shares = pos["shares"]
            price = pos["price_usd"]
            # P&L per path = shares * price * (exp(return) - 1)
            pnl += shares * price * (np.exp(self._returns[:, i]) - 1)

        self._pnl = pnl
        return pnl

    def _ensure_pnl(self):
        """Compute P&L if not already done."""
        if self._pnl is None:
            self.portfolio_pnl()

    def var(self, confidence: float = 0.95) -> float:
        """Compute Value-at-Risk at given confidence level.

        VaR is the loss at the (1-confidence) percentile of the P&L distribution.
        Returns a negative number (loss).

        Args:
            confidence: Confidence level (0.95 or 0.99 typically).

        Returns:
            VaR in USD (negative = loss).
        """
        self._ensure_pnl()
        if self._n_assets == 0:
            return 0.0
        percentile = (1 - confidence) * 100
        return float(np.percentile(self._pnl, percentile))

    def cvar(self, confidence: float = 0.95) -> float:
        """Compute Conditional VaR (Expected Shortfall) at given confidence.

        CVaR is the average loss in the worst (1-confidence) scenarios.
        Always worse than VaR (further from zero).

        Args:
            confidence: Confidence level.

        Returns:
            CVaR in USD (negative = loss).
        """
        self._ensure_pnl()
        if self._n_assets == 0:
            return 0.0
        var_threshold = self.var(confidence)
        tail = self._pnl[self._pnl <= var_threshold]
        if len(tail) == 0:
            return var_threshold
        return float(np.mean(tail))

    def drawdown_probability(self, threshold_pct: float) -> float:
        """Probability of portfolio losing more than threshold_pct.

        Args:
            threshold_pct: Loss threshold as percentage (e.g., 5.0 = 5%).

        Returns:
            Probability (0.0-1.0) of exceeding the loss threshold.
        """
        self._ensure_pnl()
        if self._n_assets == 0:
            return 0.0

        total_value = sum(
            p["shares"] * p["price_usd"] for p in self._positions.values()
        )
        if total_value <= 0:
            return 0.0

        loss_threshold = -total_value * threshold_pct / 100.0
        return float(np.mean(self._pnl < loss_threshold))


# ---------------------------------------------------------------------------
# Convenience: compute portfolio VaR from system data structures
# ---------------------------------------------------------------------------

def compute_portfolio_var(
    portfolio_state: dict,
    agent_summary: dict,
    n_paths: int = DEFAULT_N_PATHS,
    historical_returns: dict[str, np.ndarray] | None = None,
    seed: int | None = None,
) -> dict:
    """Compute portfolio VaR from portfolio state and agent summary.

    Extracts held positions, current prices, volatilities, and directional
    probabilities from the system data, then runs t-copula simulation.

    Args:
        portfolio_state: Portfolio state dict (holdings, cash_sek, etc.).
        agent_summary: Agent summary with signals, fx_rate.
        n_paths: Number of MC paths.
        historical_returns: Optional empirical return series for correlation.
        seed: Random seed.

    Returns:
        Dict with VaR metrics in both USD and SEK.
    """
    holdings = portfolio_state.get("holdings", {})
    signals = agent_summary.get("signals", {})
    fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)

    # Build positions dict
    positions = {}
    tickers = []

    for ticker, holding in holdings.items():
        shares = holding.get("shares", 0)
        if shares <= 0:
            continue

        ticker_data = signals.get(ticker, {})
        price = ticker_data.get("price_usd", 0)
        if price <= 0:
            continue

        extra = ticker_data.get("extra", {})
        atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
        vol = volatility_from_atr(atr_pct)

        # Get directional probability for drift
        from portfolio.monte_carlo import _get_directional_probability
        p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
        drift = drift_from_probability(p_up, vol)

        tickers.append(ticker)
        positions[ticker] = {
            "shares": shares,
            "price_usd": price,
            "volatility": vol,
            "drift": drift,
        }

    if not tickers:
        return {
            "var_95_usd": 0.0,
            "var_99_usd": 0.0,
            "cvar_95_usd": 0.0,
            "cvar_99_usd": 0.0,
            "var_95_sek": 0.0,
            "var_99_sek": 0.0,
            "cvar_95_sek": 0.0,
            "cvar_99_sek": 0.0,
            "total_exposure_usd": 0.0,
            "total_exposure_sek": 0.0,
            "n_positions": 0,
            "drawdown_1pct_prob": 0.0,
            "drawdown_5pct_prob": 0.0,
        }

    # Build correlation matrix
    corr = build_correlation_matrix(tickers, historical_returns)

    # Run simulation
    sim = PortfolioRiskSimulator(
        positions=positions,
        correlation_matrix=corr,
        horizon_days=1,
        n_paths=n_paths,
        df=DEFAULT_DF,
        seed=seed,
    )
    sim.simulate_correlated_returns()

    total_exposure = sum(p["shares"] * p["price_usd"] for p in positions.values())

    var95 = sim.var(0.95)
    var99 = sim.var(0.99)
    cvar95 = sim.cvar(0.95)
    cvar99 = sim.cvar(0.99)

    return {
        "var_95_usd": round(var95, 2),
        "var_99_usd": round(var99, 2),
        "cvar_95_usd": round(cvar95, 2),
        "cvar_99_usd": round(cvar99, 2),
        "var_95_sek": round(var95 * fx_rate, 2),
        "var_99_sek": round(var99 * fx_rate, 2),
        "cvar_95_sek": round(cvar95 * fx_rate, 2),
        "cvar_99_sek": round(cvar99 * fx_rate, 2),
        "total_exposure_usd": round(total_exposure, 2),
        "total_exposure_sek": round(total_exposure * fx_rate, 2),
        "n_positions": len(tickers),
        "drawdown_1pct_prob": round(sim.drawdown_probability(1.0), 3),
        "drawdown_5pct_prob": round(sim.drawdown_probability(5.0), 3),
    }
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\stats.py'" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py'" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\strategies\\golddigger_strategy.py' | Select-Object -First 260" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\strategies\\elongir_strategy.py' | Select-Object -First 220" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\strategies\\orchestrator.py'" in Q:\fa-adv-2026-05-11
 succeeded in 842ms:
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from portfolio.file_utils import load_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
TELEGRAMS_FILE = DATA_DIR / "telegram_messages.jsonl"


def invocation_stats():
    entries = load_jsonl(INVOCATIONS_FILE)
    if not entries:
        print("No invocations recorded yet.")
        return

    by_date = defaultdict(list)
    reason_counts = Counter()
    for e in entries:
        dt = datetime.fromisoformat(e["ts"])
        day = dt.strftime("%Y-%m-%d")
        by_date[day].append(e)
        for r in e.get("reasons", []):
            tag = r.split(" ")[0] if " " in r else r
            reason_counts[tag] += 1

    print("=== Agent Invocations ===\n")
    print(f"{'Date':<14} {'Count':>6}")
    print("-" * 22)
    total = 0
    for day in sorted(by_date.keys()):
        n = len(by_date[day])
        total += n
        print(f"{day:<14} {n:>6}")
    print("-" * 22)
    print(f"{'Total':<14} {total:>6}")

    days = len(by_date)
    if days > 0:
        print(f"{'Avg/day':<14} {total / days:>6.1f}")

    print(f"\n{'Trigger Reason':<24} {'Count':>6}")
    print("-" * 32)
    for reason, count in reason_counts.most_common():
        print(f"{reason:<24} {count:>6}")


def telegram_stats():
    entries = load_jsonl(TELEGRAMS_FILE)
    if not entries:
        print("\nNo Telegram messages recorded yet.")
        return

    by_date = defaultdict(list)
    decisions = Counter()
    for e in entries:
        dt = datetime.fromisoformat(e["ts"])
        day = dt.strftime("%Y-%m-%d")
        by_date[day].append(e)
        text = e.get("text", "")
        if text.startswith("*HOLD*"):
            decisions["HOLD"] += 1
        elif text.startswith("*BUY"):
            decisions["BUY"] += 1
        elif text.startswith("*SELL"):
            decisions["SELL"] += 1
        else:
            decisions["OTHER"] += 1

    print("\n=== Telegram Messages ===\n")
    print(f"{'Date':<14} {'Sent':>6}")
    print("-" * 22)
    total = 0
    for day in sorted(by_date.keys()):
        n = len(by_date[day])
        total += n
        print(f"{day:<14} {n:>6}")
    print("-" * 22)
    print(f"{'Total':<14} {total:>6}")

    print(f"\n{'Decision':<14} {'Count':>6} {'%':>7}")
    print("-" * 29)
    for dec, count in decisions.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"{dec:<14} {count:>6} {pct:>6.1f}%")


def full_report():
    invocation_stats()
    telegram_stats()

    invocations = load_jsonl(INVOCATIONS_FILE)
    telegrams = load_jsonl(TELEGRAMS_FILE)
    missed = len(invocations) - len(telegrams)
    if missed > 0:
        print(
            f"\nWARNING: {missed} invocation(s) without Telegram response (agent may have crashed)"
        )
    elif missed < 0:
        print(f"\nNOTE: {-missed} extra Telegram(s) beyond invocations (manual runs)")


if __name__ == "__main__":
    full_report()
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 844ms:
"""Portfolio state management - load, save, atomic writes, value calculation."""

import logging
import math
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("portfolio.portfolio_mgr")

from portfolio.file_utils import atomic_write_json as _atomic_write_json
from portfolio.file_utils import load_json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "portfolio_state.json"
BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"
INITIAL_CASH_SEK = 500_000

_DEFAULT_STATE = {
    "cash_sek": INITIAL_CASH_SEK,
    "holdings": {},
    "transactions": [],
    "initial_value_sek": INITIAL_CASH_SEK,
}

# C8: Per-file locks for concurrency safety
_state_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()

_MAX_BACKUPS = 3  # Rolling backup count


def _get_lock(path: Path) -> threading.Lock:
    """Get or create a per-file lock for concurrency safety (C8)."""
    key = str(path)
    with _locks_lock:
        if key not in _state_locks:
            _state_locks[key] = threading.Lock()
        return _state_locks[key]


def _rotate_backups(path: Path):
    """C7: Create rolling .bak backups before overwriting state.

    Maintains up to _MAX_BACKUPS copies: path.bak, path.bak2, path.bak3.
    Only backs up if the file exists and has content.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        # Rotate existing backups: .bak2  .bak3, .bak  .bak2
        for i in range(_MAX_BACKUPS, 1, -1):
            src = path.with_suffix(f".json.bak{i - 1}" if i > 2 else ".json.bak")
            dst = path.with_suffix(f".json.bak{i}")
            if src.exists():
                shutil.copy2(str(src), str(dst))
        # Current file  .bak
        shutil.copy2(str(path), str(path.with_suffix(".json.bak")))
    except OSError as e:
        logger.warning("Failed to rotate backups for %s: %s", path.name, e)


def _validated_state(loaded):
    """Merge loaded state with defaults to ensure all required keys exist."""
    if not loaded or not isinstance(loaded, dict):
        return {**_DEFAULT_STATE, "start_date": datetime.now(UTC).isoformat()}
    result = {**_DEFAULT_STATE, **loaded}
    # Ensure types are correct for critical fields
    if not isinstance(result.get("holdings"), dict):
        result["holdings"] = {}
    if not isinstance(result.get("transactions"), list):
        result["transactions"] = []
    return result


def _load_state_from(path: Path):
    """Load portfolio state from a specific file.

    C7: On corruption, logs CRITICAL and attempts recovery from backups.
    Returns validated defaults only if file AND all backups are missing/corrupt.
    """
    loaded = load_json(str(path), default=None)
    if loaded is not None:
        return _validated_state(loaded)

    # File is missing or corrupt - check if the file exists (corruption vs missing)
    if path.exists():
        logger.critical(
            "CORRUPT portfolio state file: %s - attempting backup recovery", path.name
        )
        # Try backups in order
        for i in range(1, _MAX_BACKUPS + 1):
            bak = path.with_suffix(f".json.bak{i}" if i > 1 else ".json.bak")
            if bak.exists():
                loaded = load_json(str(bak), default=None)
                if loaded is not None:
                    logger.warning("Recovered %s from backup %s", path.name, bak.name)
                    return _validated_state(loaded)
        logger.critical(
            "ALL backups corrupt/missing for %s - returning fresh defaults", path.name
        )

    return {**_DEFAULT_STATE, "start_date": datetime.now(UTC).isoformat()}


def _save_state_to(path: Path, state):
    """Save state with rolling backup (C7) and lock (C8)."""
    lock = _get_lock(path)
    with lock:
        _rotate_backups(path)
        _atomic_write_json(path, state)


def load_state():
    """Load Patient portfolio state. Returns validated defaults if missing or corrupt."""
    return _load_state_from(STATE_FILE)


def save_state(state):
    """Save Patient portfolio state with backup rotation (C7)."""
    _save_state_to(STATE_FILE, state)


def load_bold_state():
    """Load Bold portfolio state. Returns validated defaults if missing or corrupt."""
    return _load_state_from(BOLD_STATE_FILE)


def save_bold_state(state):
    """Save Bold portfolio state with backup rotation (C7)."""
    _save_state_to(BOLD_STATE_FILE, state)


def update_state(mutate_fn, bold=False):
    """Atomic read-modify-write for portfolio state (C8).

    Holds a lock for the entire read-modify-write cycle to prevent
    concurrent callers from overwriting each other's mutations.

    Args:
        mutate_fn: Callable that receives the current state dict and mutates it.
            The function should modify the dict in-place and optionally return it.
        bold: If True, operates on the Bold portfolio instead of Patient.

    Returns:
        The updated state dict.
    """
    path = BOLD_STATE_FILE if bold else STATE_FILE
    lock = _get_lock(path)
    with lock:
        state = _load_state_from(path)
        result = mutate_fn(state)
        if result is not None:
            state = result
        _rotate_backups(path)
        _atomic_write_json(path, state)
    return state


def portfolio_value(state, prices_usd, fx_rate):
    if not isinstance(fx_rate, (int, float)) or not math.isfinite(fx_rate) or fx_rate <= 0:
        logger.warning("portfolio_value: invalid fx_rate=%r, returning cash only", fx_rate)
        return state.get("cash_sek", 0)
    total = state.get("cash_sek", 0)
    for ticker, h in state.get("holdings", {}).items():
        try:
            shares = h.get("shares", 0)
            price = prices_usd.get(ticker)
            if shares > 0 and price is not None and price > 0:
                total += shares * price * fx_rate
            elif shares > 0 and (price is None or price <= 0):
                logger.warning(
                    "portfolio_value: invalid price for %s: %r (shares=%s)",
                    ticker, price, shares,
                )
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning("portfolio_value: error calculating %s: %s", ticker, e)
    return total
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 866ms:
"""GoldDigger strategy adapter - wraps GolddiggerBot for orchestrator integration.

Builds MarketSnapshot from SharedData + lightweight Binance fetch.
Trade actions are enqueued to metals_trade_queue.json for execution
by the metals loop's Playwright session.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.golddigger.bot import GolddiggerBot
from portfolio.golddigger.config import DATA_DIR, GolddiggerConfig
from portfolio.golddigger.data_provider import (
    MarketSnapshot,
    fetch_gold_price,
    fetch_us10y_context,
    fetch_usdsek,
    read_event_risk,
)
from portfolio.strategies.base import SharedData, StrategyBase

logger = logging.getLogger("portfolio.strategies.golddigger")

_DEFAULT_TRADE_QUEUE = str(DATA_DIR / "metals_trade_queue.json")


class GoldDiggerStrategy(StrategyBase):
    """Adapts GolddiggerBot as a strategy plugin.

    Data flow:
    - Gold spot: fetched from Binance FAPI at each tick (5s freshness)
    - FX (USD/SEK): from SharedData (metals loop cache)
    - US10Y: from GoldDigger's own provider (15min cache)
    - Cert bid/ask: from SharedData cert_prices (metals loop cache, 60s stale)
    - Trade execution: enqueued to metals_trade_queue.json
    """

    def __init__(
        self,
        config: dict,
        trade_queue_file: str = _DEFAULT_TRADE_QUEUE,
    ):
        self._cfg = GolddiggerConfig.from_config(config)
        # dry_run=False so bot tracks position state (prevents duplicate BUY signals).
        # Actual execution goes through trade queue, not Playwright.
        self._bot = GolddiggerBot(self._cfg, dry_run=False)
        self._trade_queue_file = trade_queue_file

    def name(self) -> str:
        return "golddigger"

    def poll_interval_seconds(self) -> float:
        return float(self._cfg.poll_seconds)

    def is_active(self) -> bool:
        return True

    def status_summary(self) -> str:
        state = self._bot.state
        pos = "flat"
        if state.has_position():
            pos = f"pos={state.position.quantity}x"
        return (
            f"golddigger: {pos}, "
            f"equity={state.equity_sek:.0f}, "
            f"trades={state.daily_trades}"
        )

    def tick(self, shared: SharedData) -> dict | None:
        """One poll cycle: fetch gold, build snapshot, run bot."""
        self._shared = shared  # store ref for trade queue lock
        gold = fetch_gold_price(self._cfg.binance_gold_symbol)
        if gold is None or gold <= 0:
            gold = shared.get_price("XAU-USD")
        if gold <= 0:
            return None

        snapshot = self._build_snapshot(shared, gold)
        action = self._bot.step(snapshot)

        if action is not None and action.get("action") in ("BUY", "SELL", "FLATTEN"):
            self._enqueue_trade(action, shared)

        return action

    def _build_snapshot(self, shared: SharedData, gold_price: float) -> MarketSnapshot:
        """Build a GoldDigger MarketSnapshot from shared data + own fetches."""
        # Fetch own FX rate (cached in fx_rates module, ~daily refresh)
        fx = fetch_usdsek()
        if fx is None or fx <= 0:
            fx = shared.fx_rate if shared.fx_rate > 0 else 10.5

        rate_ctx = None
        if self._cfg.fred_api_key:
            rate_ctx = fetch_us10y_context(
                self._cfg.fred_api_key,
                source=self._cfg.rates_source,
                yfinance_ticker=self._cfg.rates_proxy_ticker,
                interval=self._cfg.rates_proxy_interval,
                lookback_bars=self._cfg.rates_proxy_lookback_bars,
                ttl_seconds=self._cfg.rates_proxy_ttl_seconds,
                max_bar_age_minutes=self._cfg.rates_proxy_max_bar_age_minutes,
                fred_series=self._cfg.fred_series,
            )

        event_ctx = None
        if self._cfg.use_event_risk_gate:
            event_ctx = read_event_risk(
                hours_before=self._cfg.event_risk_hours_before,
                hours_after=self._cfg.event_risk_hours_after,
                block_types=self._cfg.event_risk_block_types,
            )

        cert = shared.get_cert(self._cfg.bull_orderbook_id)

        now = datetime.now(UTC)
        snap = MarketSnapshot(
            ts_utc=now,
            gold=gold_price,
            usdsek=fx,
            us10y=rate_ctx["value"] if rate_ctx else 0.0,
            us10y_source=rate_ctx.get("source") if rate_ctx else None,
            us10y_change_pct=rate_ctx.get("change_pct") if rate_ctx else None,
            next_event_type=event_ctx.get("event_type") if event_ctx else None,
            next_event_hours=event_ctx.get("hours_to_event") if event_ctx else None,
            event_risk_active=bool(event_ctx and event_ctx.get("active")),
            event_risk_phase=event_ctx.get("phase") if event_ctx else None,
            data_quality="ok",
            gold_fetch_ts=now,
            fx_fetch_ts=now,
        )
        if cert:
            snap.cert_bid = cert.get("bid")
            snap.cert_ask = cert.get("ask")
            snap.cert_last = cert.get("last")
            if snap.cert_bid and snap.cert_ask and snap.cert_bid > 0:
                snap.cert_spread_pct = (snap.cert_ask - snap.cert_bid) / snap.cert_bid

        return snap

    def _enqueue_trade(self, action: dict, shared: SharedData | None = None) -> None:
        """Write trade action to metals_trade_queue.json for execution.

        Uses shared.trade_queue_lock to prevent race with metals loop's
        process_trade_queue() running on the main thread.
        """
        lock = shared.trade_queue_lock if shared else None
        if lock:
            lock.acquire()
        try:
            self._enqueue_trade_locked(action)
        finally:
            if lock:
                lock.release()

    def _enqueue_trade_locked(self, action: dict) -> None:
        """Enqueue while holding the trade queue lock."""
        queue = load_json(self._trade_queue_file, default=None)
        if queue is None:
            queue = {"version": 1, "orders": []}

        order = {
            "id": str(uuid.uuid4()),
            "source": "golddigger",
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "pending",
            "action": action.get("action", action.get("type", "?")),
            "ob_id": self._cfg.bull_orderbook_id,
            "warrant_name": f"BULL GULD X{int(self._cfg.leverage)}",
            "quantity": action.get("quantity", 0),
            "price": action.get("price", 0),
            "account_id": self._cfg.avanza_account_id,
            "reason": action.get("reason", ""),
            "strategy_data": {
                "composite_s": action.get("composite_s"),
                "z_gold": action.get("z_gold"),
            },
        }
        queue["orders"].append(order)
        atomic_write_json(self._trade_queue_file, queue)
        logger.info(
            "Enqueued %s order: %s (qty=%d, ob=%s)",
            order["action"], order["id"][:8],
            order["quantity"], order["ob_id"],
        )
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 880ms:
"""Strategy orchestrator - daemon thread managing multiple strategy plugins."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from portfolio.strategies.base import SharedData, StrategyBase

logger = logging.getLogger("portfolio.strategies.orchestrator")

# Halt a strategy after this many consecutive errors
MAX_CONSECUTIVE_ERRORS = 10


class StrategyOrchestrator:
    """Manages strategy plugins in a daemon thread.

    Each strategy is ticked at its own poll interval. Errors in one
    strategy do not affect others. The thread stops cleanly on stop().
    """

    def __init__(
        self,
        strategies: list[StrategyBase],
        shared_data: SharedData,
        send_telegram: Callable[[str], object] | None = None,
    ):
        self._strategies = strategies
        self._shared = shared_data
        self._send_telegram = send_telegram
        self._last_tick: dict[str, float] = {}
        self._error_counts: dict[str, int] = {}
        self._halted: set[str] = set()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the orchestrator daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="strategy-orchestrator",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Strategy orchestrator started: %s",
            ", ".join(s.name() for s in self._strategies),
        )

    def stop(self) -> None:
        """Signal the thread to stop and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Strategy orchestrator stopped")

    def summary(self) -> str:
        """One-line summary of all strategies."""
        parts = []
        for s in self._strategies:
            status = "HALTED" if s.name() in self._halted else (
                "active" if s.is_active() else "inactive"
            )
            parts.append(f"{s.name()}({status}, {s.poll_interval_seconds()}s)")
        return f"{len(self._strategies)} strategies: " + ", ".join(parts)

    def _run_loop(self) -> None:
        """Main tick loop - runs until stop() is called."""
        while self._running:
            now = time.monotonic()
            for strategy in self._strategies:
                name = strategy.name()

                if name in self._halted:
                    continue
                if not strategy.is_active():
                    continue

                last = self._last_tick.get(name, 0.0)
                if now - last < strategy.poll_interval_seconds():
                    continue

                try:
                    action = strategy.tick(self._shared)
                    self._last_tick[name] = time.monotonic()
                    self._error_counts[name] = 0

                    if action is not None:
                        self._handle_action(strategy, action)

                except Exception:
                    count = self._error_counts.get(name, 0) + 1
                    self._error_counts[name] = count
                    logger.error(
                        "Strategy %s error (%d/%d)",
                        name, count, MAX_CONSECUTIVE_ERRORS,
                        exc_info=True,
                    )
                    if count >= MAX_CONSECUTIVE_ERRORS:
                        self._halted.add(name)
                        logger.error(
                            "Strategy %s HALTED after %d consecutive errors",
                            name, count,
                        )
                        if self._send_telegram:
                            self._send_telegram(
                                f"_Strategy {name} halted: {count} consecutive errors_"
                            )

            time.sleep(0.5)

    def _handle_action(self, strategy: StrategyBase, action: dict) -> None:
        """Process a trade action from a strategy."""
        logger.info(
            "Strategy %s action: %s",
            strategy.name(),
            action.get("type", action.get("action", "?")),
        )
        if self._send_telegram:
            action_type = action.get("type", action.get("action", "?"))
            reason = action.get("reason", "")
            self._send_telegram(
                f"*STRATEGY {strategy.name().upper()}* {action_type}\n_{reason}_"
            )


def load_strategies(config: dict) -> list[StrategyBase]:
    """Load enabled strategies from config.

    Reads config["strategies"]["golddigger_enabled"] and
    config["strategies"]["elongir_enabled"] to decide which to load.
    Defaults to enabled if the bot's config section exists.
    """
    strategies_cfg = config.get("strategies", {})
    strategies: list[StrategyBase] = []

    # GoldDigger
    gd_enabled = strategies_cfg.get("golddigger_enabled", "golddigger" in config)
    if gd_enabled:
        try:
            from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
            strategies.append(GoldDiggerStrategy(config))
            logger.info("Loaded strategy: golddigger")
        except Exception as e:
            logger.error("Failed to load golddigger strategy: %s", e)

    # Elongir
    el_enabled = strategies_cfg.get("elongir_enabled", "elongir" in config)
    if el_enabled:
        try:
            from portfolio.strategies.elongir_strategy import ElongirStrategy
            strategies.append(ElongirStrategy(config))
            logger.info("Loaded strategy: elongir")
        except Exception as e:
            logger.error("Failed to load elongir strategy: %s", e)

    return strategies
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 889ms:
"""Elongir strategy adapter - wraps ElongirBot for orchestrator integration.

Builds MarketSnapshot from SharedData + own kline fetches.
Fully simulated - no trade queue, no Avanza execution.
"""
from __future__ import annotations

import logging

from portfolio.elongir.bot import ElongirBot
from portfolio.elongir.config import ElongirConfig
from portfolio.elongir.data_provider import MarketSnapshot, fetch_klines, fetch_usdsek
from portfolio.strategies.base import SharedData, StrategyBase

logger = logging.getLogger("portfolio.strategies.elongir")


class ElongirStrategy(StrategyBase):
    """Adapts ElongirBot as a strategy plugin.

    Data flow:
    - Silver spot: from SharedData (metals loop cache, updated every 60s)
    - FX (USD/SEK): from SharedData
    - Klines (1m/5m/15m): fetched from Binance FAPI at each tick
    - Execution: fully simulated (ElongirBot manages own state)
    """

    def __init__(self, config: dict):
        self._cfg = ElongirConfig.from_config(config)
        self._bot = ElongirBot(self._cfg)

    def name(self) -> str:
        return "elongir"

    def poll_interval_seconds(self) -> float:
        return float(self._cfg.poll_seconds)

    def is_active(self) -> bool:
        return True

    def status_summary(self) -> str:
        state = self._bot.state
        pos = "flat"
        if state.has_position():
            pos = f"pos={state.position.quantity}x"
        wr = f"{state.wins}/{state.losses}" if (state.wins + state.losses) > 0 else "0/0"
        return (
            f"elongir: {pos}, "
            f"state={state.signal_state}, "
            f"pnl={state.total_pnl:+,.0f}, "
            f"W/L={wr}"
        )

    def tick(self, shared: SharedData) -> dict | None:
        """One poll cycle: build snapshot from shared data + klines, run bot."""
        silver = shared.get_price("XAG-USD")
        if silver <= 0:
            return None

        klines_1m = fetch_klines("1m", 100)
        klines_5m = fetch_klines("5m", 60)
        klines_15m = fetch_klines("15m", 40)

        snapshot = self._build_snapshot(shared, klines_1m, klines_5m, klines_15m)
        return self._bot.step(snapshot)

    def _build_snapshot(
        self,
        shared: SharedData,
        klines_1m: list | None,
        klines_5m: list | None,
        klines_15m: list | None,
    ) -> MarketSnapshot:
        """Build an Elongir MarketSnapshot from shared data + klines."""
        # Fetch own FX rate (cached in fx_rates module)
        fx = fetch_usdsek()
        if fx is None or fx <= 0:
            fx = shared.fx_rate if shared.fx_rate > 0 else 10.5
        return MarketSnapshot(
            silver_usd=shared.get_price("XAG-USD"),
            fx_rate=fx,
            klines_1m=klines_1m,
            klines_5m=klines_5m,
            klines_15m=klines_15m,
            xag_signals=None,
        )
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\strategies\\base.py'" in Q:\fa-adv-2026-05-11
 succeeded in 610ms:
"""Base protocol and shared data for strategy plugins."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock


@dataclass
class SharedData:
    """Data shared from metals loop to strategies.

    Updated by metals loop main thread, read by orchestrator thread.
    Dict fields must be replaced with copies (not mutated in place)
    to avoid concurrent-iteration RuntimeError.
    """
    underlying_prices: dict[str, float] = field(default_factory=dict)
    fx_rate: float = 0.0
    cert_prices: dict[str, dict] = field(default_factory=dict)
    is_market_hours: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    trade_queue_lock: Lock = field(default_factory=Lock)

    def get_price(self, ticker: str) -> float:
        """Get underlying price, 0.0 if missing."""
        return self.underlying_prices.get(ticker, 0.0)

    def get_cert(self, orderbook_id: str) -> dict | None:
        """Get certificate price data, None if missing."""
        return self.cert_prices.get(orderbook_id)


class StrategyBase(ABC):
    """Protocol for strategy plugins run by the StrategyOrchestrator."""

    @abstractmethod
    def name(self) -> str:
        """Unique strategy name for logging and config."""

    @abstractmethod
    def poll_interval_seconds(self) -> float:
        """Desired tick interval in seconds."""

    @abstractmethod
    def tick(self, shared: SharedData) -> dict | None:
        """Execute one poll cycle.

        Returns action dict if a trade happened, None otherwise.
        Must not call Playwright or block for more than a few seconds.
        """

    @abstractmethod
    def is_active(self) -> bool:
        """Whether this strategy should be ticked."""

    @abstractmethod
    def status_summary(self) -> str:
        """One-line status for Telegram/logging."""
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve moved into the valuation and P&L paths. There’s a stronger issue now than the earlier guard/default problems: the warrant portfolio code models MINI products as simple leverage multipliers and never tracks financing/barrier at all, which means knockouts can be missed entirely. I’m tracing whether that same simplification leaks into stop-loss and Monte Carlo decisions.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"load_state\\(|save_state\\(|load_bold_state\\(|save_bold_state\\(|update_state\\(\" Q:\\finance-analyzer\\portfolio\\portfolio_mgr.py Q:\\finance-analyzer\\portfolio\\portfolio_validator.py Q:\\finance-analyzer\\portfolio\\trade_guards.py Q:\\finance-analyzer\\portfolio\\trade_validation.py Q:\\finance-analyzer\\portfolio\\trade_risk_classifier.py Q:\\finance-analyzer\\portfolio\\risk_management.py Q:\\finance-analyzer\\portfolio\\monte_carlo.py Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py Q:\\finance-analyzer\\portfolio\\equity_curve.py Q:\\finance-analyzer\\portfolio\\exit_optimizer.py Q:\\finance-analyzer\\portfolio\\kelly_sizing.py Q:\\finance-analyzer\\portfolio\\kelly_metals.py Q:\\finance-analyzer\\portfolio\\exposure_coach.py Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py Q:\\finance-analyzer\\portfolio\\cost_model.py Q:\\finance-analyzer\\portfolio\\instrument_profile.py Q:\\finance-analyzer\\portfolio\\stats.py Q:\\finance-analyzer\\portfolio\\strategies" in Q:\fa-adv-2026-05-11
 succeeded in 842ms:
Q:\finance-analyzer\portfolio\trade_guards.py:35:def _load_state():
Q:\finance-analyzer\portfolio\trade_guards.py:45:def _save_state(state):
Q:\finance-analyzer\portfolio\trade_guards.py:127:        state = _load_state()
Q:\finance-analyzer\portfolio\trade_guards.py:265:        state = _load_state()
Q:\finance-analyzer\portfolio\trade_guards.py:312:        _save_state(state)
Q:\finance-analyzer\portfolio\trade_guards.py:362:        state = _load_state()
Q:\finance-analyzer\portfolio\portfolio_mgr.py:116:def load_state():
Q:\finance-analyzer\portfolio\portfolio_mgr.py:121:def save_state(state):
Q:\finance-analyzer\portfolio\portfolio_mgr.py:126:def load_bold_state():
Q:\finance-analyzer\portfolio\portfolio_mgr.py:131:def save_bold_state(state):
Q:\finance-analyzer\portfolio\portfolio_mgr.py:136:def update_state(mutate_fn, bold=False):
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"min_order_sek|500\\.0|Order value|minimum\" Q:\\finance-analyzer\\portfolio\\trade_validation.py Q:\\finance-analyzer\\portfolio\\kelly_sizing.py Q:\\finance-analyzer\\portfolio\\kelly_metals.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"regime_lower|_REGIME_SCORES|get\\(regime_lower, 0\\)|return \\{\\\"level\\\"\" Q:\\finance-analyzer\\portfolio\\trade_risk_classifier.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"avg_cost_usd|cash-only|Drawdown circuit breaker reading may be optimistic|current_value = portfolio.get\\(\\\"cash_sek\\\"|holdings_value \\+= shares \\* avg_cost|atr_pct = min|stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)|distance_in_atr\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"leverage-multiplied|current_underlying_usd|implied_pnl_pct|current_implied_sek|financing|barrier|strike|underlying_change \\* leverage|fx_rate\" Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py Q:\\finance-analyzer\\portfolio\\exit_optimizer.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"price_per_share = total_sek / shares|sell_price_per_share = sell_total / sell_shares|buy_fee_share|sell_fee_share|pnl_sek = \\(sell_price_per_share - buy_price\\) \\* matched - buy_fee_share - sell_fee_share|p_stop_hit|probability_below\\(stop_price\\)\" Q:\\finance-analyzer\\portfolio\\equity_curve.py Q:\\finance-analyzer\\portfolio\\monte_carlo.py" in Q:\fa-adv-2026-05-11
2026-05-11T15:32:04.298197Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:regime_lower|_REGIME_SCORES|get\(regime_lower, 0\)|return \{\)
    ^
error: unclosed group

 exited 1 in 778ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:regime_lower|_REGIME_SCORES|get\(regime_lower, 0\)|return \{\)
    ^
error: unclosed group

 succeeded in 797ms:
Q:\finance-analyzer\portfolio\kelly_metals.py:44:MIN_TRADE_SEK = 500.0
Q:\finance-analyzer\portfolio\kelly_sizing.py:327:        rec_sek = 0  # Below minimum trade size
Q:\finance-analyzer\portfolio\trade_validation.py:32:    min_order_sek: float = 500.0,
Q:\finance-analyzer\portfolio\trade_validation.py:60:    if order_value < min_order_sek:
Q:\finance-analyzer\portfolio\trade_validation.py:63:            f"Order value {order_value:.0f} SEK below minimum {min_order_sek:.0f} SEK",
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:32:04.334266Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:avg_cost_usd|cash-only|Drawdown circuit breaker reading may be optimistic|current_value = portfolio.get\(\)
    ^
error: unclosed group

 exited 1 in 814ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:avg_cost_usd|cash-only|Drawdown circuit breaker reading may be optimistic|current_value = portfolio.get\(\)
    ^
error: unclosed group

 succeeded in 813ms:
Q:\finance-analyzer\portfolio\warrant_portfolio.py:3:Tracks actual warrant positions with leverage-multiplied P&L based on
Q:\finance-analyzer\portfolio\warrant_portfolio.py:52:def warrant_pnl(holding, current_underlying_usd, fx_rate):
Q:\finance-analyzer\portfolio\warrant_portfolio.py:65:        current_underlying_usd: current price of the underlying in USD.
Q:\finance-analyzer\portfolio\warrant_portfolio.py:66:        fx_rate: current USD/SEK exchange rate.
Q:\finance-analyzer\portfolio\warrant_portfolio.py:72:            "current_implied_sek": float (current implied value per unit),
Q:\finance-analyzer\portfolio\warrant_portfolio.py:80:    if not holding or not current_underlying_usd or not fx_rate:
Q:\finance-analyzer\portfolio\warrant_portfolio.py:92:    underlying_change = (current_underlying_usd - underlying_entry) / underlying_entry
Q:\finance-analyzer\portfolio\warrant_portfolio.py:96:    implied_pnl_pct = underlying_change * leverage
Q:\finance-analyzer\portfolio\warrant_portfolio.py:97:    implied_pnl_pct_rounded = round(implied_pnl_pct * 100, 2)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:100:    current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)
Q:\finance-analyzer\portfolio\warrant_portfolio.py:101:    total_value_sek = current_implied_sek * units
Q:\finance-analyzer\portfolio\warrant_portfolio.py:106:        "pnl_pct": implied_pnl_pct_rounded,
Q:\finance-analyzer\portfolio\warrant_portfolio.py:108:        "current_implied_sek": round(current_implied_sek, 2),
Q:\finance-analyzer\portfolio\warrant_portfolio.py:116:def get_warrant_summary(prices_usd, fx_rate):
Q:\finance-analyzer\portfolio\warrant_portfolio.py:121:        fx_rate: current USD/SEK exchange rate.
Q:\finance-analyzer\portfolio\warrant_portfolio.py:157:        pnl = warrant_pnl(holding, current_price, fx_rate)
Q:\finance-analyzer\portfolio\exit_optimizer.py:72:    financing_level: float | None = None  # MINI future financing level (USD)
Q:\finance-analyzer\portfolio\exit_optimizer.py:312:        warrant_value = (underlying - financing_level) × usdsek
Q:\finance-analyzer\portfolio\exit_optimizer.py:320:    if position.instrument_type == "warrant" and position.financing_level is not None:
Q:\finance-analyzer\portfolio\exit_optimizer.py:321:        # MINI future: warrant price = (underlying - financing_level) × fx
Q:\finance-analyzer\portfolio\exit_optimizer.py:322:        exit_warrant_sek = (exit_price_usd - position.financing_level) * fx
Q:\finance-analyzer\portfolio\exit_optimizer.py:327:        # Leveraged product without explicit financing level
Q:\finance-analyzer\portfolio\exit_optimizer.py:373:    if position.financing_level and position.financing_level > 0:
Q:\finance-analyzer\portfolio\exit_optimizer.py:374:        distance_pct = (market.price - position.financing_level) / market.price * 100
Q:\finance-analyzer\portfolio\exit_optimizer.py:396:    if session_min is not None and position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:397:        stop_buffer = position.financing_level * 1.03  # 3% above financing
Q:\finance-analyzer\portfolio\exit_optimizer.py:419:    - Knock-out danger (< 3% from financing level)
Q:\finance-analyzer\portfolio\exit_optimizer.py:431:    if position.financing_level and position.financing_level > 0:
Q:\finance-analyzer\portfolio\exit_optimizer.py:432:        distance_pct = (market.price - position.financing_level) / market.price * 100
Q:\finance-analyzer\portfolio\exit_optimizer.py:434:            logger.warning("RISK OVERRIDE: Knock-out danger (%.1f%% from barrier), "
Q:\finance-analyzer\portfolio\exit_optimizer.py:445:    if session_min is not None and position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:446:        stop_buffer = position.financing_level * 1.03
Q:\finance-analyzer\portfolio\exit_optimizer.py:643:    elif position.financing_level:
Q:\finance-analyzer\portfolio\exit_optimizer.py:645:        stop_buffer = position.financing_level * 1.03
Q:\finance-analyzer\portfolio\exit_optimizer.py:686:    financing_level: float | None = None,
Q:\finance-analyzer\portfolio\exit_optimizer.py:701:        financing_level: For MINI futures, the knock-out level.
Q:\finance-analyzer\portfolio\exit_optimizer.py:719:    fx_rate = agent_summary.get("fx_rate", 10.85)
Q:\finance-analyzer\portfolio\exit_optimizer.py:726:        usdsek=fx_rate,
Q:\finance-analyzer\portfolio\exit_optimizer.py:749:        financing_level=financing_level,
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 814ms:
Q:\finance-analyzer\portfolio\equity_curve.py:346:                price_per_share = total_sek / shares
Q:\finance-analyzer\portfolio\equity_curve.py:369:        sell_price_per_share = sell_total / sell_shares if sell_shares > 0 else 0
Q:\finance-analyzer\portfolio\equity_curve.py:403:            buy_fee_share = (buy["fee_sek"] * matched / buy["original_shares"]) if buy["original_shares"] > 0 else 0
Q:\finance-analyzer\portfolio\equity_curve.py:404:            sell_fee_share = (sell_fee * matched / sell_shares) if sell_shares > 0 else 0
Q:\finance-analyzer\portfolio\equity_curve.py:405:            pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share
Q:\finance-analyzer\portfolio\equity_curve.py:417:                "fee_sek": round(buy_fee_share + sell_fee_share, 2),
Q:\finance-analyzer\portfolio\monte_carlo.py:15:    #   "p_stop_hit_1d": 0.12,
Q:\finance-analyzer\portfolio\monte_carlo.py:328:        result[f"p_stop_hit_{h_key}"] = round(mc.probability_below(stop_price), 3)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"_REGIME_SCORES|get\\(regime_lower, 0\\)|regime_lower = regime\\.lower\\(\\)\" Q:\\finance-analyzer\\portfolio\\trade_risk_classifier.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"avg_cost_usd|cash-only value|Drawdown circuit breaker reading may be optimistic|current_value = portfolio.get\\(\\\"cash_sek\\\", initial_value\\)|price_usd = signals\\[ticker\\]\\.get\\(\\\"price_usd\\\", 0\\)|stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)|atr_pct = min\\(atr_pct, 15\\.0\\)\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"fx_rate = agent_summary.get\\(\\\"fx_rate\\\", 10\\.85\\)|stop_buffer = position\\.financing_level \\* 1\\.03|distance_pct = \\(market\\.price - position\\.financing_level\\) / market\\.price \\* 100\" Q:\\finance-analyzer\\portfolio\\exit_optimizer.py" in Q:\fa-adv-2026-05-11
 succeeded in 769ms:
20:_REGIME_SCORES = {
80:    regime_lower = regime.lower()
81:    regime_score = _REGIME_SCORES.get(regime_lower, 0)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:32:13.509342Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:avg_cost_usd|cash-only value|Drawdown circuit breaker reading may be optimistic|current_value = portfolio.get\(\)
    ^
error: unclosed group

 exited 1 in 774ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:avg_cost_usd|cash-only value|Drawdown circuit breaker reading may be optimistic|current_value = portfolio.get\(\)
    ^
error: unclosed group

 exited 1 in 753ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:fx_rate = agent_summary.get\(\)
    ^
error: unclosed group

2026-05-11T15:32:13.510022Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:fx_rate = agent_summary.get\(\)
    ^
error: unclosed group

2026-05-11T15:32:21.915558Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Select-String -Path 'Q:\\finance-analyzer\\portfolio\\risk_management.py' -Pattern 'avg_cost_usd','cash-only value','Drawdown circuit breaker reading may be optimistic','current_value = portfolio.get(\"cash_sek\", initial_value)','price_usd = signals\\[ticker\\]\\.get\\(\"price_usd\", 0\\)','atr_pct = min\\(atr_pct, 15\\.0\\)','stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)'"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Select-String -Path 'Q:\\finance-analyzer\\portfolio\\risk_management.py' -Pattern 'avg_cost_usd','cash-only value','Drawdown circuit breaker reading may be optimistic','current_value = portfolio.get(\"cash_sek\", initial_value)','price_usd = signals\\[ticker\\]\\.get\\(\"price_usd\", 0\\)','atr_pct = min\\(atr_pct, 15\\.0\\)','stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)'" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Select-String -Path 'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py' -Pattern 'fx_rate = agent_summary.get\\(\"fx_rate\", 10\\.85\\)','stop_buffer = position.financing_level \\* 1.03','distance_pct = \\(market.price - position.financing_level\\) / market.price \\* 100'" in Q:\fa-adv-2026-05-11
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Select-String -Path 'Q:\\finance-analyzer\\portfolio\\risk_management.py' -Pattern 'avg_cost_usd','cash-only value','Drawdown circuit breaker reading may be optimistic','current_value = portfolio.get(\"cash_sek\", initial_value)','price_usd = signals\\[ticker\\]\\.get\\(\"price_usd\", 0\\)','atr_pct = min\\(atr_pct, 15\\.0\\)','stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)'"` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Select-String -Path 'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py' -Pattern 'fx_rate = agent_summary.get\\(\"fx_rate\", 10\\.85\\)','stop_buffer = position.financing_level \\* 1.03','distance_pct = \\(market.price - position.financing_level\\) / market.price \\* 100'"` rejected: blocked by policy
2026-05-11T15:32:21.915968Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Select-String -Path 'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py' -Pattern 'fx_rate = agent_summary.get\\(\"fx_rate\", 10\\.85\\)','stop_buffer = position.financing_level \\* 1.03','distance_pct = \\(market.price - position.financing_level\\) / market.price \\* 100'"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"avg_cost_usd\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"cash-only\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"atr_pct = min\\(atr_pct, 15\\.0\\)\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"fx_rate = agent_summary.get\\(\\\"fx_rate\\\", 10\\.85\\)\" Q:\\finance-analyzer\\portfolio\\exit_optimizer.py" in Q:\fa-adv-2026-05-11
 succeeded in 820ms:
211:            # Fallback: use avg_cost_usd from holdings if no live price
212:            avg_cost = pos.get("avg_cost_usd", 0)
329:            Each entry: {ticker: {"shares": N, "avg_cost_usd": X, ...}}
334:            - entry_price_usd: float (avg_cost_usd)
350:        entry_price = pos.get("avg_cost_usd", 0)
435:        entry_price = pos.get("avg_cost_usd", 0)
762:        price = signals.get(t, {}).get("price_usd", pos.get("avg_cost_usd", 0))
775:    existing_price = signals.get(ticker, {}).get("price_usd", existing.get("avg_cost_usd", 0))
890:    entry_price = pos.get("avg_cost_usd", 0)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 838ms:
267:                "back to cash-only value for %d live position(s). "
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 851ms:
375:        stop_price = entry_price * (1 - 2 * atr_pct / 100)
466:        stop_price = entry_price * (1 - 2 * atr_pct / 100)
902:    stop_price = entry_price * (1 - 2 * atr_pct / 100)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 851ms:
374:        atr_pct = min(atr_pct, 15.0)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:32:31.035068Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:fx_rate = agent_summary.get\(\)
    ^
error: unclosed group

 exited 1 in 848ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:fx_rate = agent_summary.get\(\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"agent_summary.get|stop_buffer|distance_pct = \\(market.price - position.financing_level\\) / market.price \\* 100\" Q:\\finance-analyzer\\portfolio\\exit_optimizer.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"current_value = portfolio.get\\(\\\"cash_sek\\\", initial_value\\)|summary = load_json\\(agent_summary_path, default=\\{\\}\\)|current_value = _compute_portfolio_value\\(portfolio, summary\\)\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
 succeeded in 604ms:
374:        distance_pct = (market.price - position.financing_level) / market.price * 100
397:        stop_buffer = position.financing_level * 1.03  # 3% above financing
398:        p_knockout = float(np.mean(session_min <= stop_buffer))
432:        distance_pct = (market.price - position.financing_level) / market.price * 100
446:        stop_buffer = position.financing_level * 1.03
447:        p_knockout = float(np.mean(session_min <= stop_buffer))
645:        stop_buffer = position.financing_level * 1.03
646:        stop_prob = float(np.mean(session_min <= stop_buffer))
708:    signals = agent_summary.get("signals", {})
719:    fx_rate = agent_summary.get("fx_rate", 10.85)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

2026-05-11T15:32:37.141676Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:current_value = portfolio.get\(\)
    ^
error: unclosed group

 exited 1 in 612ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:current_value = portfolio.get\(\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"current_value =\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"load_json\\(agent_summary_path|_compute_portfolio_value\\(portfolio, summary\\)\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
 succeeded in 579ms:
249:        summary = load_json(agent_summary_path, default={})
251:            current_value = _compute_portfolio_value(portfolio, summary)
596:    summary = load_json(agent_summary_path, default={"signals": {}})
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 600ms:
247:        current_value = portfolio.get("cash_sek", initial_value)
251:            current_value = _compute_portfolio_value(portfolio, summary)
271:            current_value = portfolio.get("cash_sek", initial_value)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"n_paths: int = DEFAULT_N_PATHS|self.n_paths = n_paths|if len\\(rows\\) < 30|if min_len >= 20|has_data >= 2|return estimate_correlation_matrix|var\\(|cvar\\(\" Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py Q:\\finance-analyzer\\portfolio\\kelly_metals.py" in Q:\fa-adv-2026-05-11
 succeeded in 617ms:
Q:\finance-analyzer\portfolio\kelly_metals.py:97:    if len(rows) < 30:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:13:    result = compute_portfolio_var(portfolio_state, agent_summary)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:77:                if min_len >= 20:  # Need at least 20 observations
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:154:        if has_data >= 2:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:155:            return estimate_correlation_matrix(historical_returns, tickers)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:195:        n_paths: int = DEFAULT_N_PATHS,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:213:        self.n_paths = n_paths
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:329:    def var(self, confidence: float = 0.95) -> float:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:347:    def cvar(self, confidence: float = 0.95) -> float:
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:362:        var_threshold = self.var(confidence)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:395:def compute_portfolio_var(
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:398:    n_paths: int = DEFAULT_N_PATHS,
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:485:    var95 = sim.var(0.95)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:486:    var99 = sim.var(0.99)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:487:    cvar95 = sim.cvar(0.95)
Q:\finance-analyzer\portfolio\monte_carlo_risk.py:488:    cvar99 = sim.cvar(0.99)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"price = ticker_data.get\\(\\\"price_usd\\\", 0\\)|if price <= 0:|continue\" Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py" in Q:\fa-adv-2026-05-11
2026-05-11T15:33:18.824403Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:price = ticker_data.get\(\)
    ^
error: unclosed group

 exited 1 in 622ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:price = ticker_data.get\(\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"ticker_data.get|price <= 0|signals.get\" Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py' | Select-Object -Skip 410 -First 40" in Q:\fa-adv-2026-05-11
 succeeded in 652ms:
430:        ticker_data = signals.get(ticker, {})
431:        price = ticker_data.get("price_usd", 0)
432:        if price <= 0:
435:        extra = ticker_data.get("extra", {})
436:        atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 690ms:
        historical_returns: Optional empirical return series for correlation.
        seed: Random seed.

    Returns:
        Dict with VaR metrics in both USD and SEK.
    """
    holdings = portfolio_state.get("holdings", {})
    signals = agent_summary.get("signals", {})
    fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)

    # Build positions dict
    positions = {}
    tickers = []

    for ticker, holding in holdings.items():
        shares = holding.get("shares", 0)
        if shares <= 0:
            continue

        ticker_data = signals.get(ticker, {})
        price = ticker_data.get("price_usd", 0)
        if price <= 0:
            continue

        extra = ticker_data.get("extra", {})
        atr_pct = extra.get("atr_pct") or ticker_data.get("atr_pct", 2.0)
        vol = volatility_from_atr(atr_pct)

        # Get directional probability for drift
        from portfolio.monte_carlo import _get_directional_probability
        p_up = _get_directional_probability(ticker, ticker_data, agent_summary)
        drift = drift_from_probability(p_up, vol)

        tickers.append(ticker)
        positions[ticker] = {
            "shares": shares,
            "price_usd": price,
            "volatility": vol,
            "drift": drift,
        }
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"rec_sek < 500|Recommended size = half Kelly|rec_sek = min|half_kelly = full_kelly / 2\\.0|quarter_kelly = full_kelly / 4\\.0\" Q:\\finance-analyzer\\portfolio\\kelly_sizing.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"MIN_TRADE_SEK|position_sek < MIN_TRADE_SEK|position_fraction = half_kelly / cert_loss_frac|half_kelly = full_kelly / 2\\.0\" Q:\\finance-analyzer\\portfolio\\kelly_metals.py" in Q:\fa-adv-2026-05-11
 succeeded in 976ms:
44:MIN_TRADE_SEK = 500.0
209:    half_kelly = full_kelly / 2.0
217:        position_fraction = half_kelly / cert_loss_frac
229:    if position_sek < MIN_TRADE_SEK:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 990ms:
314:    half_kelly = full_kelly / 2.0
315:    quarter_kelly = full_kelly / 4.0
322:    # Recommended size = half Kelly * cash * exposure_ceiling, capped at max allocation
323:    rec_sek = min(half_kelly * cash_sek * exposure_ceiling, max_alloc)
326:    if rec_sek < 500:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"def probability_below|terminal price < threshold|p_stop_hit|probability_below\\(stop_price\\)\" Q:\\finance-analyzer\\portfolio\\monte_carlo.py" in Q:\fa-adv-2026-05-11
 succeeded in 569ms:
15:    #   "p_stop_hit_1d": 0.12,
205:    def probability_below(self, threshold: float) -> float:
206:        """Compute P(terminal price < threshold).
328:        result[f"p_stop_hit_{h_key}"] = round(mc.probability_below(stop_price), 3)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"BUY total_sek = full allocation|SELL total_sek = net proceeds\" Q:\\finance-analyzer\\portfolio\\portfolio_validator.py" in Q:\fa-adv-2026-05-11
 succeeded in 565ms:
71:    # Note: BUY total_sek = full allocation (including fee)
72:    # SELL total_sek = net proceeds (after fee deducted)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"datetime.now\\(UTC\\)|fromisoformat|time.monotonic\" Q:\\finance-analyzer\\portfolio\\trade_guards.py Q:\\finance-analyzer\\portfolio\\strategies\\orchestrator.py" in Q:\fa-adv-2026-05-11
 succeeded in 605ms:
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:76:            now = time.monotonic()
Q:\finance-analyzer\portfolio\strategies\orchestrator.py:91:                    self._last_tick[name] = time.monotonic()
Q:\finance-analyzer\portfolio\trade_guards.py:88:        last_loss = datetime.fromisoformat(
Q:\finance-analyzer\portfolio\trade_guards.py:93:        elapsed_hours = (datetime.now(UTC) - last_loss).total_seconds() / 3600
Q:\finance-analyzer\portfolio\trade_guards.py:128:    now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\trade_guards.py:142:            last_trade = datetime.fromisoformat(last_trade_str)
Q:\finance-analyzer\portfolio\trade_guards.py:206:                ts = datetime.fromisoformat(ts_str)
Q:\finance-analyzer\portfolio\trade_guards.py:266:        now = datetime.now(UTC)
Q:\finance-analyzer\portfolio\trade_guards.py:303:                    dt = datetime.fromisoformat(ts)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"0\\.15|0\\.30|15%|30%|fixed|buying_power|position size|allocation|alloc_frac|alloc_pct\" Q:\\finance-analyzer\\portfolio\\kelly_sizing.py Q:\\finance-analyzer\\portfolio\\kelly_metals.py Q:\\finance-analyzer\\portfolio\\risk_management.py Q:\\finance-analyzer\\portfolio\\trade_validation.py Q:\\finance-analyzer\\portfolio\\exposure_coach.py Q:\\finance-analyzer\\portfolio\\strategies" in Q:\fa-adv-2026-05-11
 succeeded in 637ms:
Q:\finance-analyzer\portfolio\exposure_coach.py:5:allocation.  This is ADVISORY — it goes into agent_summary.json for
Q:\finance-analyzer\portfolio\trade_validation.py:3:Validates trade parameters (price, volume, spread, cash, position size) before
Q:\finance-analyzer\portfolio\risk_management.py:373:        # 2x ATR stop-loss — cap ATR at 15% to prevent meaninglessly wide stops for warrants
Q:\finance-analyzer\portfolio\risk_management.py:768:    # BUG-176: Compute proposed allocation as % of total portfolio, capped at cash
Q:\finance-analyzer\portfolio\risk_management.py:769:    alloc_pct = 0.30 if strategy == "bold" else 0.15
Q:\finance-analyzer\portfolio\risk_management.py:770:    proposed_alloc = min(total_value * alloc_pct, cash)
Q:\finance-analyzer\portfolio\kelly_metals.py:5:fixed position sizing (30% of buying power) with edge-aware sizing.
Q:\finance-analyzer\portfolio\kelly_metals.py:14:        buying_power_sek=5000,
Q:\finance-analyzer\portfolio\kelly_metals.py:127:    """Reduce position size after consecutive losses.
Q:\finance-analyzer\portfolio\kelly_metals.py:140:    buying_power_sek: float = 0.0,
Q:\finance-analyzer\portfolio\kelly_metals.py:146:    """Compute Kelly-optimal position size for a metals warrant trade.
Q:\finance-analyzer\portfolio\kelly_metals.py:151:        buying_power_sek: Available cash on Avanza.
Q:\finance-analyzer\portfolio\kelly_metals.py:227:    # --- Step 6: Compute final SEK allocation ---
Q:\finance-analyzer\portfolio\kelly_metals.py:228:    position_sek = buying_power_sek * adjusted_fraction
Q:\finance-analyzer\portfolio\kelly_sizing.py:6:Returns recommended position size as fraction of portfolio.
Q:\finance-analyzer\portfolio\kelly_sizing.py:169:    accurate on XAG-USD but 30% on BTC-USD shows up as ~50% in the
Q:\finance-analyzer\portfolio\kelly_sizing.py:232:    """Compute recommended position size using Kelly criterion.
Q:\finance-analyzer\portfolio\kelly_sizing.py:250:            max_alloc_sek: Maximum allocation per strategy rules,
Q:\finance-analyzer\portfolio\kelly_sizing.py:268:    # Max allocation per strategy rules
Q:\finance-analyzer\portfolio\kelly_sizing.py:269:    alloc_frac = 0.30 if strategy == "bold" else 0.15
Q:\finance-analyzer\portfolio\kelly_sizing.py:270:    max_alloc = cash_sek * alloc_frac
Q:\finance-analyzer\portfolio\kelly_sizing.py:318:    # Scales position sizes down when the broad market is unhealthy.
Q:\finance-analyzer\portfolio\kelly_sizing.py:322:    # Recommended size = half Kelly * cash * exposure_ceiling, capped at max allocation
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"hourly candles|annual_factor = math.sqrt\\(252\\.0 / period\\)|volatility_from_atr|math.sqrt\\(252\\.0 / 14\\)\" Q:\\finance-analyzer\\portfolio\\monte_carlo.py Q:\\finance-analyzer\\portfolio\\exit_optimizer.py Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\monte_carlo.py' | Select-Object -First 80" in Q:\fa-adv-2026-05-11
 succeeded in 608ms:
Q:\finance-analyzer\portfolio\risk_management.py:463:        vol = max(atr_pct / 100.0 * math.sqrt(252.0 / 14), 0.05)
Q:\finance-analyzer\portfolio\exit_optimizer.py:184:        return max(atr_frac * math.sqrt(252.0 / 14), _MIN_VOLATILITY)
Q:\finance-analyzer\portfolio\monte_carlo.py:47:def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
Q:\finance-analyzer\portfolio\monte_carlo.py:50:    ATR measures average true range over `period` candles. For hourly candles
Q:\finance-analyzer\portfolio\monte_carlo.py:63:    annual_factor = math.sqrt(252.0 / period)
Q:\finance-analyzer\portfolio\monte_carlo.py:302:    vol = volatility_from_atr(atr_pct)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 663ms:
"""Monte Carlo price simulation engine.

Generates stochastic price paths using Geometric Brownian Motion (GBM)
with antithetic variates for variance reduction. Converts directional
signal probabilities into price quantile bands, stop-loss probabilities,
and expected return distributions.

Short-term focus: 3h, 1d, 3d horizons for active trading decisions.

Usage:
    from portfolio.monte_carlo import simulate_ticker
    result = simulate_ticker("BTC-USD", agent_summary)
    # result = {
    #   "price_bands_1d": {"p5": 82100, "p25": 84300, "p50": 86200, ...},
    #   "p_stop_hit_1d": 0.12,
    #   "expected_return_1d": {"mean_pct": 0.3, "std_pct": 2.1, "skew": 0.1},
    #   ...
    # }
"""

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger("portfolio.monte_carlo")

# Default parameters
DEFAULT_N_PATHS = 10_000   # 5K pairs with antithetic variates
DEFAULT_HORIZONS = [1, 3]  # days
MIN_VOLATILITY = 0.05      # 5% annualized floor (prevents degenerate sims)

# Per-asset-class ATR fallbacks when actual ATR is missing from signals.
# The generic 2.0% underestimates tail risk for crypto/metals.
_ATR_DEFAULT_BY_CLASS = {
    "crypto": 3.5,  # BTC 3-4% typical daily ATR
    "metals": 4.0,  # XAG 4-6%, XAU 2-3% - biased toward silver
    "stocks": 2.0,  # MSTR ~3% but broader equities lower
}


# ---------------------------------------------------------------------------
# Volatility & drift estimation from existing system data
# ---------------------------------------------------------------------------

def volatility_from_atr(atr_pct: float, period: int = 14) -> float:
    """Convert ATR% (14-period) to annualized volatility.

    ATR measures average true range over `period` candles. For hourly candles
    (our primary timeframe), we annualize: vol = atr_frac * sqrt(trading_periods/period).

    Uses 252 trading days (standard for stocks/crypto).

    Args:
        atr_pct: ATR as percentage of price (e.g., 3.5 means 3.5%).
        period: ATR lookback period (default 14).

    Returns:
        Annualized volatility as a decimal (e.g., 0.20 = 20%).
    """
    atr_frac = atr_pct / 100.0
    annual_factor = math.sqrt(252.0 / period)
    vol = atr_frac * annual_factor
    return max(vol, MIN_VOLATILITY)


def drift_from_probability(p_up: float, volatility: float) -> float:
    """Convert directional probability P(up) into annualized drift.

    Uses the inverse of the GBM CDF relationship:
        P(S_T > S_0) = N((mu - 0.5*sigma^2)*sqrt(T) / (sigma*sqrt(T)))

    For 1-day horizon (T = 1/252):
        mu = sigma * N_inv(p_up) * sqrt(252) + 0.5 * sigma^2

    This ensures the GBM simulation produces paths where the fraction
    ending above spot matches the input probability.

    Args:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"zero correlation|independent assumption|if min_len >= 20|if has_data >= 2|return estimate_correlation_matrix|CORRELATION_PRIORS\" Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py" in Q:\fa-adv-2026-05-11
 succeeded in 587ms:
53:    For tickers with insufficient data, defaults to zero correlation
54:    (independent assumption) which is conservative for VaR.
77:                if min_len >= 20:  # Need at least 20 observations
114:CORRELATION_PRIORS = {
122:    return CORRELATION_PRIORS.get(
124:        CORRELATION_PRIORS.get((ticker_b, ticker_a), 0.0),
154:        if has_data >= 2:
155:            return estimate_correlation_matrix(historical_returns, tickers)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"def load_warrant_state|def save_warrant_state|state = load_warrant_state\\(|save_warrant_state\\(state\\)\" Q:\\finance-analyzer\\portfolio\\warrant_portfolio.py" in Q:\fa-adv-2026-05-11
 succeeded in 657ms:
25:def load_warrant_state():
42:def save_warrant_state(state):
138:    state = load_warrant_state()
198:    state = load_warrant_state()
265:    save_warrant_state(state)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"knockout_prob|inst_type = \\\"warrant\\\"|stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)|position.financing_level|3% above financing|3% from barrier\" Q:\\finance-analyzer\\portfolio\\risk_management.py Q:\\finance-analyzer\\portfolio\\exit_optimizer.py" in Q:\fa-adv-2026-05-11
2026-05-11T15:34:58.433691Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:knockout_prob|inst_type = \)
    ^
error: unclosed group

 exited 1 in 636ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:knockout_prob|inst_type = \)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"knockout_prob\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"inst_type = \\\"warrant\\\"\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"3% above financing|3% from barrier|distance_pct = \\(market.price - position.financing_level\\) / market.price \\* 100\" Q:\\finance-analyzer\\portfolio\\exit_optimizer.py" in Q:\fa-adv-2026-05-11
 succeeded in 768ms:
415:            - knockout_prob: For warrants, P(reaching financing level)
493:            "knockout_prob": None,
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 exited 1 in 765ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:inst_type = \)
    ^
error: unclosed group

2026-05-11T15:35:04.961355Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:inst_type = \)
    ^
error: unclosed group

 succeeded in 753ms:
374:        distance_pct = (market.price - position.financing_level) / market.price * 100
397:        stop_buffer = position.financing_level * 1.03  # 3% above financing
432:        distance_pct = (market.price - position.financing_level) / market.price * 100
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\risk_management.py' | Select-Object -Skip 448 -First 20" in Q:\fa-adv-2026-05-11
 succeeded in 634ms:
        if ticker.endswith("-USD") and ticker.startswith(("BTC", "ETH")):
            inst_type = "crypto"
        elif ticker.startswith(("XAG", "XAU")):
            inst_type = "warrant"
        else:
            inst_type = "stock"

        # Get remaining session minutes
        remaining = remaining_session_minutes(inst_type)
        if remaining < 2:
            continue

        # Estimate volatility from ATR
        import math
        vol = max(atr_pct / 100.0 * math.sqrt(252.0 / 14), 0.05)

        # ATR stop level
        stop_price = entry_price * (1 - 2 * atr_pct / 100)
        if stop_price <= 0:
            stop_price = entry_price * 0.01
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"ticker.startswith\\(\\(\\\"XAG\\\", \\\"XAU\\\"\\)\\)|vol = max\\(atr_pct / 100\\.0 \\* math\\.sqrt\\(252\\.0 / 14\\), 0\\.05\\)|stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
2026-05-11T15:35:15.436570Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:ticker.startswith\(\(\)
    ^
error: unclosed group

 exited 1 in 563ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:ticker.startswith\(\(\)
    ^
error: unclosed group

2026-05-11T15:35:21.422856Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"XAG\\\", \\\"XAU\" Q:\\finance-analyzer\\portfolio\\risk_management.py"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"XAG\\\", \\\"XAU\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"XAG\\\", \\\"XAU\" Q:\\finance-analyzer\\portfolio\\risk_management.py"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"vol = max\\(atr_pct / 100\\.0 \\* math.sqrt\\(252\\.0 / 14\\), 0\\.05\\)\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"stop_price = entry_price \\* \\(1 - 2 \\* atr_pct / 100\\)\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
 succeeded in 668ms:
375:        stop_price = entry_price * (1 - 2 * atr_pct / 100)
466:        stop_price = entry_price * (1 - 2 * atr_pct / 100)
902:    stop_price = entry_price * (1 - 2 * atr_pct / 100)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 696ms:
463:        vol = max(atr_pct / 100.0 * math.sqrt(252.0 / 14), 0.05)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"warrant\" Q:\\finance-analyzer\\portfolio\\risk_management.py" in Q:\fa-adv-2026-05-11
 succeeded in 559ms:
373:        # 2x ATR stop-loss — cap ATR at 15% to prevent meaninglessly wide stops for warrants
415:            - knockout_prob: For warrants, P(reaching financing level)
452:            inst_type = "warrant"
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve found a stronger sizing defect than the minimum-order floor: `kelly_sizing._compute_trade_stats()` builds one weighted average from all BUYs for a ticker and then scores every SELL against it, regardless of trade order. That means later scale-ins rewrite earlier win/loss history, so the Kelly inputs themselves are look-ahead biased. I’m pulling exact lines and checking whether the metals sizing path has the same issue.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"Compute P&L for each sell vs weighted average buy price|total_shares_bought = sum\\(b.get\\(\\\"shares\\\", 0\\) for b in buys\\)|avg_buy_price = total_cost / total_shares_bought|for sell in sells:\" Q:\\finance-analyzer\\portfolio\\kelly_sizing.py" in Q:\fa-adv-2026-05-11
2026-05-11T15:35:38.985513Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:Compute P&L for each sell vs weighted average buy price|total_shares_bought = sum\(b.get\(\)
    ^
error: unclosed group

 exited 1 in 553ms:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m
rg: regex parse error:
    (?:Compute P&L for each sell vs weighted average buy price|total_shares_bought = sum\(b.get\(\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"weighted average buy price\" Q:\\finance-analyzer\\portfolio\\kelly_sizing.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"total_shares_bought|avg_buy_price|for sell in sells\" Q:\\finance-analyzer\\portfolio\\kelly_sizing.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\kelly_sizing.py' | Select-Object -Skip 70 -First 40" in Q:\fa-adv-2026-05-11
 succeeded in 799ms:
83:    # Compute P&L for each sell vs weighted average buy price
90:        # Compute weighted average buy price (in SEK per share)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 801ms:
91:        total_shares_bought = sum(b.get("shares", 0) for b in buys)
93:        if total_shares_bought <= 0:
95:        avg_buy_price = total_cost / total_shares_bought
97:        for sell in sells:
103:            pnl_pct = (sell_price_per_share - avg_buy_price) / avg_buy_price * 100
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 850ms:
    sells_by_ticker = defaultdict(list)

    for t in transactions:
        t_ticker = t.get("ticker", "")
        if ticker and t_ticker != ticker:
            continue
        action = t.get("action", "")
        if action == "BUY":
            buys_by_ticker[t_ticker].append(t)
        elif action == "SELL":
            sells_by_ticker[t_ticker].append(t)

    # Compute P&L for each sell vs weighted average buy price
    pnl_list = []
    for t_ticker, sells in sells_by_ticker.items():
        buys = buys_by_ticker.get(t_ticker, [])
        if not buys:
            continue

        # Compute weighted average buy price (in SEK per share)
        total_shares_bought = sum(b.get("shares", 0) for b in buys)
        total_cost = sum(b.get("total_sek", 0) for b in buys)
        if total_shares_bought <= 0:
            continue
        avg_buy_price = total_cost / total_shares_bought

        for sell in sells:
            sell_shares = sell.get("shares", 0)
            sell_total = sell.get("total_sek", 0)
            if sell_shares <= 0:
                continue
            sell_price_per_share = sell_total / sell_shares
            pnl_pct = (sell_price_per_share - avg_buy_price) / avg_buy_price * 100
            pnl_list.append(pnl_pct)

    if len(pnl_list) < 2:
        return None

    wins = [p for p in pnl_list if p > 0]
    losses = [abs(p) for p in pnl_list if p <= 0]
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve got enough context to rank the main defects now. I’m validating two last edge cases before I write the report: whether the Monte Carlo stop/KO paths violate the explicit 50% de-risk tolerance, and whether the equity metrics are netting fees once or twice.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"0\\.25|0\\.10|knock-out probability|forcing market exit|KNOCKOUT_PROB\" Q:\\finance-analyzer\\portfolio\\exit_optimizer.py" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\exit_optimizer.py' | Select-Object -Skip 388 -First 65" in Q:\fa-adv-2026-05-11
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "Get-Content 'Q:\\finance-analyzer\\portfolio\\equity_curve.py' | Select-Object -Skip 338 -First 75" in Q:\fa-adv-2026-05-11
 succeeded in 638ms:
203:        volatility: Annualized volatility (decimal, e.g., 0.25 = 25%).
399:        if p_knockout > 0.10:
400:            flags.append(f"KNOCKOUT_PROB_{p_knockout:.0%}")
435:                           "forcing market exit", distance_pct)
440:        logger.info("RISK OVERRIDE: Session ending in %.0f min, forcing market exit",
444:    # Override 3: High knock-out probability → prefer market exit
448:        if p_knockout > 0.25:
449:            logger.warning("RISK OVERRIDE: %.0f%% knock-out probability, "
450:                           "forcing market exit", p_knockout * 100)
462:DEFAULT_QUANTILES = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 672ms:
    for tx in transactions:
        if tx.get("action") == "BUY":
            ticker = tx.get("ticker", "")
            shares = tx.get("shares", 0)
            total_sek = tx.get("total_sek", 0)
            fee = tx.get("fee_sek", 0) or 0
            if shares > 0:
                price_per_share = total_sek / shares
                buy_queues[ticker].append({
                    "ts": tx.get("timestamp", ""),
                    "remaining_shares": shares,
                    "original_shares": shares,
                    "price_per_share": price_per_share,
                    "fee_sek": fee,
                })

    round_trips = []

    for tx in transactions:
        if tx.get("action") != "SELL":
            continue
        ticker = tx.get("ticker", "")
        sell_shares = tx.get("shares", 0)
        sell_total = tx.get("total_sek", 0)
        sell_fee = tx.get("fee_sek", 0) or 0
        sell_ts = tx.get("timestamp", "")

        if sell_shares <= 0 or ticker not in buy_queues:
            continue

        sell_price_per_share = sell_total / sell_shares if sell_shares > 0 else 0
        shares_to_match = sell_shares

        while shares_to_match > 0 and buy_queues[ticker]:
            buy = buy_queues[ticker][0]
            matched = min(shares_to_match, buy["remaining_shares"])

            # Compute hold time
            hold_hours = 0
            try:
                buy_dt = datetime.datetime.fromisoformat(buy["ts"])
                sell_dt = datetime.datetime.fromisoformat(sell_ts)
                if buy_dt.tzinfo is None:
                    buy_dt = buy_dt.replace(tzinfo=datetime.UTC)
                if sell_dt.tzinfo is None:
                    sell_dt = sell_dt.replace(tzinfo=datetime.UTC)
                hold_hours = (sell_dt - buy_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

            buy_price = buy["price_per_share"]
            pnl_pct = ((sell_price_per_share - buy_price) / buy_price * 100) if buy_price > 0 else 0

            # Proportional fees - use original buy quantity as denominator
            # to avoid inflating fee allocation on subsequent partial matches
            # (BUG-37 regression covered by test_equity_curve_fifo.TestBug37FeeDoubleCount).
            #
            # P0-6 (2026-05-02): pnl_sek is now NET of buy+sell fees so
            # downstream metrics (profit_factor, total_pnl_sek, Calmar) report
            # realised SEK. `pnl_pct` remains the gross price-% (price-move
            # only) - used for streaks and expectancy. `fee_sek` field still
            # reports total fees so consumers can reconstruct gross if needed.
            # See tests/test_equity_curve_fifo.TestPnlSekNetOfFees and
            # tests/test_portfolio_metrics.TestProfitFactorNetOfFees.
            buy_fee_share = (buy["fee_sek"] * matched / buy["original_shares"]) if buy["original_shares"] > 0 else 0
            sell_fee_share = (sell_fee * matched / sell_shares) if sell_shares > 0 else 0
            pnl_sek = (sell_price_per_share - buy_price) * matched - buy_fee_share - sell_fee_share

            round_trips.append({
                "ticker": ticker,
                "buy_ts": buy["ts"],
                "sell_ts": sell_ts,
                "buy_price_sek": round(buy_price, 4),
                "sell_price_sek": round(sell_price_per_share, 4),
                "shares": round(matched, 8),
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 674ms:
    # 5. Position aging
    if position.entry_ts:
        hold_hours = (market.asof_ts - position.entry_ts).total_seconds() / 3600
        if hold_hours > 5:
            flags.append("HOLD_TIME_EXTENDED")

    # 6. Stop-loss proximity from MC paths
    if session_min is not None and position.financing_level:
        stop_buffer = position.financing_level * 1.03  # 3% above financing
        p_knockout = float(np.mean(session_min <= stop_buffer))
        if p_knockout > 0.10:
            flags.append(f"KNOCKOUT_PROB_{p_knockout:.0%}")

    return flags


# ---------------------------------------------------------------------------
# Risk overrides
# ---------------------------------------------------------------------------

def _apply_risk_overrides(
    candidates: list[CandidateExit],
    position: Position,
    market: MarketSnapshot,
    remaining_minutes: float,
    session_min: np.ndarray | None = None,
) -> CandidateExit:
    """Apply hard risk overrides and select recommended exit.

    Risk overrides can force a market exit even if EV says hold:
    - Knock-out danger (< 3% from financing level)
    - Session end imminent (< 5 min remaining)
    - Stop probability too high (> 25% chance of knock-out)
    """
    if not candidates:
        raise ValueError("No candidates to evaluate")

    # Find the market exit candidate
    market_exits = [c for c in candidates if c.action == "market"]
    market_exit = market_exits[0] if market_exits else candidates[-1]

    # Override 1: Knock-out danger  force market exit
    if position.financing_level and position.financing_level > 0:
        distance_pct = (market.price - position.financing_level) / market.price * 100
        if distance_pct < 3:
            logger.warning("RISK OVERRIDE: Knock-out danger (%.1f%% from barrier), "
                           "forcing market exit", distance_pct)
            return market_exit

    # Override 2: Session about to end  force market exit
    if remaining_minutes < 5:
        logger.info("RISK OVERRIDE: Session ending in %.0f min, forcing market exit",
                     remaining_minutes)
        return market_exit

    # Override 3: High knock-out probability  prefer market exit
    if session_min is not None and position.financing_level:
        stop_buffer = position.financing_level * 1.03
        p_knockout = float(np.mean(session_min <= stop_buffer))
        if p_knockout > 0.25:
            logger.warning("RISK OVERRIDE: %.0f%% knock-out probability, "
                           "forcing market exit", p_knockout * 100)
            return market_exit

    # No override triggered - return highest-EV candidate
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"fx_rate = agent_summary.get\\(\" Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py" in Q:\fa-adv-2026-05-11
 succeeded in 574ms:
419:    fx_rate = agent_summary.get("fx_rate", FX_RATE_FALLBACK)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -NoProfile -Command "rg -n \"var_95_sek|total_exposure_sek|cvar_95_sek|cvar_99_sek\" Q:\\finance-analyzer\\portfolio\\monte_carlo_risk.py" in Q:\fa-adv-2026-05-11
 succeeded in 588ms:
458:            "var_95_sek": 0.0,
460:            "cvar_95_sek": 0.0,
461:            "cvar_99_sek": 0.0,
463:            "total_exposure_sek": 0.0,
495:        "var_95_sek": round(var95 * fx_rate, 2),
497:        "cvar_95_sek": round(cvar95 * fx_rate, 2),
498:        "cvar_99_sek": round(cvar99 * fx_rate, 2),
500:        "total_exposure_sek": round(total_exposure * fx_rate, 2),
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
# Codex adversarial review: portfolio-risk
## Summary
2 P0s, 6 P1s, 4 P2s. The main failure modes are fail-open risk math when live prices disappear, inconsistent MINI warrant valuation/knockout handling, and biased sizing/Monte Carlo inputs.

## P0 — Blockers
- portfolio/risk_management.py:211, portfolio/risk_management.py:249, portfolio/monte_carlo_risk.py:431 — Why it bites: when live prices are missing, drawdown valuation falls back to `avg_cost_usd` or even cash-only, and VaR simply drops the position; a stale feed can therefore hide exposure and leave the circuit breaker effectively stuck open. Fix: make missing live prices a hard risk fault; either block/mark breached or require a freshness-checked last live tick before computing risk.
- portfolio/warrant_portfolio.py:92, portfolio/warrant_portfolio.py:96, portfolio/warrant_portfolio.py:100, portfolio/exit_optimizer.py:320 — Why it bites: MINI warrants are valued as `entry_price * (1 + underlying_change * leverage)` with no financing/barrier term, `fx_rate` is ignored, and large down moves can produce negative warrant prices instead of knockout-at-zero. Fix: persist financing/barrier metadata in holdings and value warrants as `(underlying - financing_level) * fx`, floored at zero, with explicit knockout detection.

## P1 — High
- portfolio/kelly_sizing.py:83, portfolio/kelly_sizing.py:90, portfolio/kelly_sizing.py:97 — Why it bites: each SELL is scored against one weighted-average price built from all BUYs for the ticker, including future adds and already-closed inventory, so later trades rewrite earlier P&L and Kelly sizes become look-ahead biased. Fix: pair BUY/SELL legs chronologically and compute realized trade stats from those pairs only.
- portfolio/trade_validation.py:32, portfolio/kelly_sizing.py:326, portfolio/kelly_metals.py:44, portfolio/kelly_metals.py:229 — Why it bites: three independent gates still treat 500 SEK as the minimum, so the subsystem can recommend and approve orders below the stated 1000 SEK floor. Fix: centralize one `MIN_ORDER_SEK = 1000` constant and enforce it after sizing and before validation.
- portfolio/monte_carlo.py:50, portfolio/monte_carlo.py:63, portfolio/exit_optimizer.py:184, portfolio/risk_management.py:463 — Why it bites: the docs say ATR is measured on hourly candles, but volatility is annualized with `sqrt(252/14)` as if the ATR were daily; stop-hit, VaR, and exit-path risk are therefore materially understated. Fix: carry the ATR timeframe explicitly and annualize from bars-per-year for that timeframe.
- portfolio/monte_carlo.py:205, portfolio/monte_carlo.py:328 — Why it bites: `p_stop_hit_*` is actually `P(terminal < stop)`, not `P(path ever touched stop)`; paths that break the stop intraday and recover are misclassified as safe. Fix: compute first-passage/min-path probabilities from full simulated paths before naming or consuming the metric as stop-hit risk.
- portfolio/exit_optimizer.py:399, portfolio/exit_optimizer.py:448 — Why it bites: the optimizer starts flagging KO risk at 10% and force-exits at 25%, which conflicts with the accepted “10-20% is okay, only de-risk at 50%+” rule for 5x certs. Fix: move KO thresholds into config/policy and set the hard override at or above 50%; lower levels should stay informational.
- portfolio/warrant_portfolio.py:198, portfolio/warrant_portfolio.py:265 — Why it bites: warrant state updates are plain load-mutate-save with no lock, so concurrent BUY/SELL recording can lose a transaction or clobber holdings even though the final write is atomic. Fix: add a per-file lock or an `update_warrant_state()` read-modify-write primitive equivalent to `portfolio_mgr.update_state()`.

## P2 — Medium
- portfolio/risk_management.py:415, portfolio/risk_management.py:452, portfolio/risk_management.py:463, portfolio/risk_management.py:493 — Why it bites: XAG/XAU are treated as `warrant`, but the probabilistic stop path still uses a plain ATR stop and hardcodes `knockout_prob: None`, so MINI barrier distance is never audited there. Fix: require financing/barrier data for warrant holdings and compute KO probability from that barrier; if missing, return an explicit risk fault instead of a partial result.
- portfolio/portfolio_validator.py:71, portfolio/portfolio_validator.py:72, portfolio/equity_curve.py:346, portfolio/equity_curve.py:369, portfolio/equity_curve.py:403 — Why it bites: validator semantics say BUY `total_sek` already includes fees and SELL `total_sek` is net of fees, but `_pair_round_trips()` subtracts prorated fees again, understating realized P&L and biasing profit factor/Calmar on partial fills. Fix: either derive prices from gross amounts and subtract fees once, or keep current per-share prices and stop subtracting `buy_fee_share`/`sell_fee_share` again.
- portfolio/trade_risk_classifier.py:80, portfolio/trade_risk_classifier.py:81 — Why it bites: unknown/renamed regimes silently score as `0`, which can downgrade a trade to LOW/MEDIUM instead of failing conservative. Fix: validate regime against the allowed set and treat unknown values as an explicit risk penalty or hard error.
- portfolio/monte_carlo_risk.py:53, portfolio/monte_carlo_risk.py:77, portfolio/monte_carlo_risk.py:154 — Why it bites: once the code enters empirical-correlation mode, any pair without enough overlap is left at zero correlation even if a strong prior exists, underestimating joint drawdowns for mixed-history portfolios. Fix: merge pairwise priors into the empirical matrix for short/missing pairs instead of defaulting them to independence.

## P3 — Low
None.

## Tests missing
- Feed-stale tests proving drawdown and VaR fail closed when a held ticker has no live price or `agent_summary` is empty.
- Warrant valuation tests for financing-level pricing, knockout-to-zero, non-negative value floor, and SEK P&L sensitivity to USD/SEK moves.
- Kelly chronology tests showing a later BUY cannot change an earlier SELL’s realized P&L.
- Shared minimum-order policy tests covering `trade_validation`, `kelly_sizing`, and `kelly_metals` with a 1000 SEK floor.
- Monte Carlo tests for hourly-ATR annualization and first-passage stop-hit probability vs terminal-below-stop probability.
- Concurrency tests proving parallel `record_warrant_transaction()` calls preserve both transactions and final units.
tokens used
175,178
# Codex adversarial review: portfolio-risk
## Summary
2 P0s, 6 P1s, 4 P2s. The main failure modes are fail-open risk math when live prices disappear, inconsistent MINI warrant valuation/knockout handling, and biased sizing/Monte Carlo inputs.

## P0 — Blockers
- portfolio/risk_management.py:211, portfolio/risk_management.py:249, portfolio/monte_carlo_risk.py:431 — Why it bites: when live prices are missing, drawdown valuation falls back to `avg_cost_usd` or even cash-only, and VaR simply drops the position; a stale feed can therefore hide exposure and leave the circuit breaker effectively stuck open. Fix: make missing live prices a hard risk fault; either block/mark breached or require a freshness-checked last live tick before computing risk.
- portfolio/warrant_portfolio.py:92, portfolio/warrant_portfolio.py:96, portfolio/warrant_portfolio.py:100, portfolio/exit_optimizer.py:320 — Why it bites: MINI warrants are valued as `entry_price * (1 + underlying_change * leverage)` with no financing/barrier term, `fx_rate` is ignored, and large down moves can produce negative warrant prices instead of knockout-at-zero. Fix: persist financing/barrier metadata in holdings and value warrants as `(underlying - financing_level) * fx`, floored at zero, with explicit knockout detection.

## P1 — High
- portfolio/kelly_sizing.py:83, portfolio/kelly_sizing.py:90, portfolio/kelly_sizing.py:97 — Why it bites: each SELL is scored against one weighted-average price built from all BUYs for the ticker, including future adds and already-closed inventory, so later trades rewrite earlier P&L and Kelly sizes become look-ahead biased. Fix: pair BUY/SELL legs chronologically and compute realized trade stats from those pairs only.
- portfolio/trade_validation.py:32, portfolio/kelly_sizing.py:326, portfolio/kelly_metals.py:44, portfolio/kelly_metals.py:229 — Why it bites: three independent gates still treat 500 SEK as the minimum, so the subsystem can recommend and approve orders below the stated 1000 SEK floor. Fix: centralize one `MIN_ORDER_SEK = 1000` constant and enforce it after sizing and before validation.
- portfolio/monte_carlo.py:50, portfolio/monte_carlo.py:63, portfolio/exit_optimizer.py:184, portfolio/risk_management.py:463 — Why it bites: the docs say ATR is measured on hourly candles, but volatility is annualized with `sqrt(252/14)` as if the ATR were daily; stop-hit, VaR, and exit-path risk are therefore materially understated. Fix: carry the ATR timeframe explicitly and annualize from bars-per-year for that timeframe.
- portfolio/monte_carlo.py:205, portfolio/monte_carlo.py:328 — Why it bites: `p_stop_hit_*` is actually `P(terminal < stop)`, not `P(path ever touched stop)`; paths that break the stop intraday and recover are misclassified as safe. Fix: compute first-passage/min-path probabilities from full simulated paths before naming or consuming the metric as stop-hit risk.
- portfolio/exit_optimizer.py:399, portfolio/exit_optimizer.py:448 — Why it bites: the optimizer starts flagging KO risk at 10% and force-exits at 25%, which conflicts with the accepted “10-20% is okay, only de-risk at 50%+” rule for 5x certs. Fix: move KO thresholds into config/policy and set the hard override at or above 50%; lower levels should stay informational.
- portfolio/warrant_portfolio.py:198, portfolio/warrant_portfolio.py:265 — Why it bites: warrant state updates are plain load-mutate-save with no lock, so concurrent BUY/SELL recording can lose a transaction or clobber holdings even though the final write is atomic. Fix: add a per-file lock or an `update_warrant_state()` read-modify-write primitive equivalent to `portfolio_mgr.update_state()`.

## P2 — Medium
- portfolio/risk_management.py:415, portfolio/risk_management.py:452, portfolio/risk_management.py:463, portfolio/risk_management.py:493 — Why it bites: XAG/XAU are treated as `warrant`, but the probabilistic stop path still uses a plain ATR stop and hardcodes `knockout_prob: None`, so MINI barrier distance is never audited there. Fix: require financing/barrier data for warrant holdings and compute KO probability from that barrier; if missing, return an explicit risk fault instead of a partial result.
- portfolio/portfolio_validator.py:71, portfolio/portfolio_validator.py:72, portfolio/equity_curve.py:346, portfolio/equity_curve.py:369, portfolio/equity_curve.py:403 — Why it bites: validator semantics say BUY `total_sek` already includes fees and SELL `total_sek` is net of fees, but `_pair_round_trips()` subtracts prorated fees again, understating realized P&L and biasing profit factor/Calmar on partial fills. Fix: either derive prices from gross amounts and subtract fees once, or keep current per-share prices and stop subtracting `buy_fee_share`/`sell_fee_share` again.
- portfolio/trade_risk_classifier.py:80, portfolio/trade_risk_classifier.py:81 — Why it bites: unknown/renamed regimes silently score as `0`, which can downgrade a trade to LOW/MEDIUM instead of failing conservative. Fix: validate regime against the allowed set and treat unknown values as an explicit risk penalty or hard error.
- portfolio/monte_carlo_risk.py:53, portfolio/monte_carlo_risk.py:77, portfolio/monte_carlo_risk.py:154 — Why it bites: once the code enters empirical-correlation mode, any pair without enough overlap is left at zero correlation even if a strong prior exists, underestimating joint drawdowns for mixed-history portfolios. Fix: merge pairwise priors into the empirical matrix for short/missing pairs instead of defaulting them to independence.

## P3 — Low
None.

## Tests missing
- Feed-stale tests proving drawdown and VaR fail closed when a held ticker has no live price or `agent_summary` is empty.
- Warrant valuation tests for financing-level pricing, knockout-to-zero, non-negative value floor, and SEK P&L sensitivity to USD/SEK moves.
- Kelly chronology tests showing a later BUY cannot change an earlier SELL’s realized P&L.
- Shared minimum-order policy tests covering `trade_validation`, `kelly_sizing`, and `kelly_metals` with a 1000 SEK floor.
- Monte Carlo tests for hourly-ATR annualization and first-passage stop-hit probability vs terminal-below-stop probability.
- Concurrency tests proving parallel `record_warrant_transaction()` calls preserve both transactions and final units.
