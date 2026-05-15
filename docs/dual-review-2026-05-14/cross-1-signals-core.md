# Cross-Critique — 1 signals-core

Two independent reviews (`claude-1-signals-core.md`, `codex-1-signals-core.md`)
read separately. Below: where they agree, where they diverge, what each missed.

## Agreement — high-confidence findings (both reviewers)

- **`portfolio/forecast_accuracy.py:254-329` — `backfill_forecast_outcomes` truncates predictions file at `max_entries` break.** Both reviewers identify this as P0, both with the same mechanism (loop break before tail entries get appended, `_write_predictions` rewrites whole file with prefix only). Confidence: **very high — independent rediscovery of the exact line of bad code.** This was flagged in the 2026-05-04 review and the fix never landed. Action: **fix this first.**

- **Missing/zero `change_pct` poisons accuracy** — Claude flags `accuracy_stats.py:225` (null counter never fires), Codex flags both `forecast_accuracy.py:142,159` (BUY systematically miscredited) and `signal_db.py:288,319,347,387` (zero change_pct never counts as correct for BUY OR SELL). Different files, same root cause: **the codebase has inconsistent handling of flat/missing `change_pct`** — three different accuracy paths, three different bugs. Action: unify on `accuracy_stats._vote_correct`'s neutral-band semantics and migrate all callers.

## Codex found, Claude missed

- **`portfolio/forecast_signal.py:97` — bare `except (ImportError, Exception)`** silently downgrades Chronos-2 → v1 on OOM/CUDA/HF fault with only an INFO log. Same family as the 3-week "Not logged in" Layer 2 outage. Claude only flagged the `_load_candles` DEBUG path; missed this one. Codex is right — this is a P1 silent-failure surface. Action: narrow to `except ImportError` only.

- **`portfolio/signal_db.py:288,319,347,387` — `change_pct == 0` fails both branches.** Claude only looked at `accuracy_stats.py`'s flavour of the bug; missed that the SQL-backed mirror has the same shape minus the neutral-band guard. The dashboard `/api/accuracy` reads SQL, so the two paths report different numbers for the same underlying outcomes — a quiet credibility hole.

- **`portfolio/outcome_tracker.py:273` — UTC date vs yfinance NY exchange date.** Claude looked at yfinance specifically for timeout/leak (P1) but missed the timezone mismatch on `target_date = target_dt.date()`. Codex's catch: between 03:00-08:00 UTC the UTC date is 1 day ahead of the NY date the bar is keyed to. Affects MSTR outcome accuracy. Real, narrow, plausible — the symptom would be ~5% of MSTR outcomes silently mislabelled.

- **`portfolio/llm_calibration.py:117` + `llm_outcome_backfill.py:121,176,275` — non-atomic `read_text().splitlines()` on actively-appended JSONL.** Claude missed all four sites. Same anti-pattern that CLAUDE.md rule 4 documents as fixed elsewhere; LLM leaf code re-introduced it. Real P1.

- **`portfolio/meta_learner.py:42` + `ml_signal.py:17-27` — `_model_cache` unlocked.** Five ticker threads can race the cold-cache joblib.load; benign but wastes ~200MB. Claude said "no race conditions found" — that was wrong. Codex is correct, though the impact is small.

- **`portfolio/ml_signal.py:121,152-155` — in-progress 1h candle used for inference** while the classifier was trained on closed bars. Subtle look-ahead. Claude didn't look at ML inference path. Codex is right; needs `df = df.iloc[:-1]`.

## Claude found, Codex missed

- **`portfolio/forecast_signal.py:365,372 ↔ forecast_accuracy.py:145-148` — schema mismatch between writer (nested `chronos`/`prophet` payloads) and scorer (reads `sub_signals`/`raw_sub_signals`).** Codex looked at the *outcome assignment* in `forecast_accuracy.py` (the change_pct==0 issue) but missed that the forecast scorer reads keys the writer never emits. Result: every Chronos prediction contributes zero scored votes. Claude is right — this is P0, masks model degradation entirely. Codex's accuracy alerts would fire on a different (also broken) code path.

- **`portfolio/signal_engine.py:3069-3091` — direct `ind[...]` dict access on `rsi`/`macd_hist`/`ema9`/`ema21` without `.get()`.** Codex didn't dive into the legacy core-voter section. Claude is right that the surrounding `try/except Exception` converts a `KeyError` on a partial-data ticker into a silent total-signal-skip for that ticker, wiping enhanced signals too. Real P1.

- **`portfolio/ic_computation.py:127-128` — `ic_buy`/`ic_sell` are mean returns, not Spearman correlations, but `signal_engine._weighted_consensus` treats them as IC.** Codex didn't follow the IC → weighting → consensus chain. Claude is right — mean returns scale with market drift, so BUY signals on perma-bull tickers get a free weight bonus. This is a real model-quality bug (silent quality, not silent failure). P1.

- **`portfolio/signal_engine.py:954-956` — `MIN_VOTERS_METALS = 2` violates `.claude/rules/signals.md` ("MIN_VOTERS = 3 all classes").** Codex didn't cross-check rules. Claude is right; one of the two has to give. Real P1.

- **`accuracy_stats.py:148-153` — SQLite-first fallback to JSONL only when count == 0.** Codex missed it. Claude's catch: if SQLite has *some* rows but missed the last day of inserts (a prior P1 from 2026-05-04 still unfixed), the JSONL tail is silently ignored. Real P2.

## Disagreements

None substantive. Both reviewers explored largely disjoint code paths; the overlap is only the most visible bugs (forecast_accuracy backfill truncation, change_pct handling). No claim by one is contradicted by the other.

## What BOTH missed (third pass)

- **`portfolio/signal_engine.py` `_weighted_consensus` accuracy gate direction.** Neither reviewer audited whether the 47% gate is applied *before or after* per-direction accuracy splits. The codebase has per-direction accuracy in `ticker_accuracy.direction_probability` (Codex P2) and aggregate accuracy in `accuracy_stats.signal_accuracy`. If the gate compares against aggregate when it should compare against directional, a 70%-BUY / 30%-SELL signal passes the gate but votes wrong half the time. The fix in 2026-04 was supposed to address this — needs verification.

- **`portfolio/outcome_tracker.py` `backfill_outcomes` Phase 3 lock-release window.** Codex called the snapshot+rewrite "correct". Claude said the same. But neither checked that `concurrent_tail_bytes` parsing handles a JSONL row that crosses the snapshot byte boundary mid-line. If a writer flushed half a row before the snapshot and the rest after, the rewrite concatenates partial-row-bytes + fresh-tail-bytes, producing a corrupt JSONL line. The lock window is small but non-zero.

- **`portfolio/signal_engine.py` ADX cache key.** Claude flagged `_compute_adx`'s exception silencing (P2). Neither reviewer noted that the recent `b66375cb` commit ("content-based ADX cache key") implies a pre-existing cache-poisoning bug between tickers — review the new keying scheme against the cache's eviction policy.

- **`portfolio/signal_decay_alert.py`** — neither reviewer wrote any findings on this module. It has decay-detection logic that drives accuracy_degradation alerts. Unread = unreviewed.

## Verdict

**Codex review** = more breadth on silent-failure surfaces (LLM/Chronos/ML inference paths); **Claude review** = more depth on cross-module integration bugs (writer-vs-scorer schema, weighting math, rule-vs-code drift). Both miss the seam between `signal_decay_alert.py` and the accuracy_degradation gate.

P0 list after cross: **2 confirmed** (forecast_accuracy truncation, forecast_signal schema mismatch).
P1 list after cross: **~10 confirmed** (mix of accuracy-zeroing, schema, narrow-except).
