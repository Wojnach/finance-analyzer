# Signals-Modules Review — INCOMPLETE / LOW CONFIDENCE

Read-only review (caveman:cavecrew-reviewer) of `portfolio/signals/*.py` (58
modules) in worktree `Q:/fa-rev-0531`. **This review under-delivered**: the
agent returned a single finding for 58 files, and that finding is **REFUTED**
by main-thread cross-critique (below). Treat signals-modules as NOT yet
adequately reviewed — see backlog item in the synthesis.

## Reported finding — REFUTED
- `portfolio/signals/crypto_macro.py:228` — claimed "P1: `OPTIONS_TTL` used at
  line 228 before its definition at line 281 → NameError on first invocation."
  **FALSE POSITIVE.** Line 228 references `OPTIONS_TTL` *inside the body of*
  `compute_crypto_macro_signal()` (the `_cached(...)` call), i.e. at **call
  time**. Line 281 `OPTIONS_TTL = 900` is a **module-level** assignment executed
  at **import time**. Python resolves module globals at call time, not def time,
  so by the time the function runs the global exists. Confirmed empirically:
  crypto_macro is a *live, active* signal with 54.5% 1d accuracy — it could not
  have accumulated samples if it raised `NameError` at import/first call.
  (A genuine use-before-def would only bite for module-level execution or
  def-time-captured names: defaults, decorator factory args, class bodies. None
  apply here.) → No fix needed; the only nit is style: move the constant above
  its use for readability.

## Main-thread spot-checks (supplementing the thin agent pass)
Sampled the recent/active z-score & regime signals the agent skipped, hunting
the highest-value bug class (lookahead / in-sample contamination):
- `signals/absorption_ratio_regime.py:189-190` — **clean.** Mean/std use
  `ar_series.iloc[:-1]` — correctly EXCLUDES the current point being scored. Good
  practice; no contamination.
- `signals/vol_ratio_regime.py:53-88` — **clean.** Returns via `close.shift(1)`,
  rolling with `min_periods=window`; no forward leakage observed.
- `signals/eth_btc_ratio_roc_zscore.py:107-137` — minor: z-score window
  `recent` includes the latest point used as `iloc[-1]` (mild in-sample
  contamination, not future leakage; common and usually acceptable). Worth a
  closer look in the dedicated re-review but not a clear bug.
- `signals/mahalanobis_turbulence.py:285-291` — same mild in-sample pattern as
  above (current value may be in the mean/std window). Not strict lookahead.

## Verdict
No confirmed P0/P1 in the sampled set; the regime gates that were checked use
correct point-exclusion. But coverage is far too thin to clear 58 modules.
**Action: dedicated signals-modules re-review** (split into ~3 batches of ~20
files) focused on lookahead/shift-direction, NaN/empty-series guards, and
sign-direction of the vote — see synthesis backlog.
