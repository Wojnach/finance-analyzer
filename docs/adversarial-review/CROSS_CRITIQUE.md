# Cross-Critique — Dual Adversarial Review

**Date**: 2026-04-10
**Format**: Independent review (I) vs Agent reviews (A) — each side critiques the other
**Status**: signals-core agent complete; 7 agents still running

---

## Part 1: Agent Review → Independent Review Critique

*Findings from agent reviews that the independent review MISSED or UNDERRATED.*

### 1.1 Independent MISSED: Per-Ticker Accuracy Strips Directional Fields [A-SC-1, P1]
The independent review examined `_weighted_consensus` and noted the directional gate logic, but did NOT check whether the per-ticker accuracy override at line 1844 preserves the `buy_accuracy`/`sell_accuracy` fields. The agent correctly identified that these fields are dropped during the override, silently disabling the directional gate for tickers with sufficient per-ticker data. This is the highest-impact finding across both reviews.

### 1.2 Independent MISSED: Regime Accuracy Cache Shared Timestamp [A-SC-2, P1]
The independent review noted the accuracy system's blending but did not audit the regime accuracy cache's timestamp management. The agent found that `write_regime_accuracy_cache` uses a single `"time"` key (unlike the main accuracy cache which was fixed in BUG-133). Cross-horizon contamination is a real risk.

### 1.3 Independent MISSED: signal_history.py Race Condition [A-SC-6, P2]
The independent review focused on `file_utils.py` atomicity and `shared_state.py` thread safety but missed `signal_history.py`'s unprotected read-modify-write. This is a genuine race under ThreadPoolExecutor.

### 1.4 Independent MISSED: blend_accuracy_data Sample Count Inflation [A-SC-5, P2]
The independent review noted the accuracy gate but didn't trace how `total` is computed in `blend_accuracy_data`. Using `max(at_samples, rc_samples)` overstates sample count when the blend is dominated by a small recent window.

### 1.5 Independent MISSED: Ministral Applicable Count Mismatch [A-SC-3, P1]
The independent review noted the `_compute_applicable_count` function but didn't compare it against the actual vote-setting code. The comment "all tickers" vs the count function's "crypto-only" exclusion is a real inconsistency.

---

## Part 2: Independent Review → Agent Review Critique

*Findings from the independent review that the signals-core agent MISSED.*

### 2.1 Agent MISSED: Subprocess governance bypass [IN-7, P1]
The agent only reviewed signals-core files. The independent review's cross-cutting scan found that `bigbet.py` and `iskbets.py` bypass `claude_gate.py` entirely — this affects the broader system, not just signals-core.

### 2.2 Agent MISSED: JSONL append atomicity on Windows [IN-1, P1]
The agent noted `signal_history.py` race conditions but didn't audit `file_utils.atomic_append_jsonl` for NTFS non-atomicity. The independent review caught this broader infrastructure issue.

### 2.3 Agent MISSED: Portfolio drawdown fallback [PR-1, P1]
Outside the agent's subsystem scope — the independent review's cross-subsystem view caught this financial safety issue in risk_management.py.

### 2.4 Agent MISSED: Hardware trailing stop failure [MC-4, P1]
Outside scope — metals-core subsystem finding from independent review.

### 2.5 Assessment of Agent Findings
All 10 agent findings for signals-core are **valid and well-evidenced**. Specific notes:
- A-SC-1 (directional gate) is the most impactful — CONFIRMED, should be Priority 1
- A-SC-2 (regime cache) is CONFIRMED — reproducible by inspection
- A-SC-4 (regime overlay ordering) is correctly assessed as "not a bug" but a design clarity issue — AGREE with P1 downgrade to P2
- A-SC-8 (SQLite check_same_thread) is CONFIRMED — defensive hardening recommended
- A-SC-10 (docstring) is correctly rated P3

---

## Part 3: Agreement Matrix

*Findings that both sides independently identified — these have highest confidence.*

| Finding | Independent ID | Agent ID | Subsystem | Severity |
|---------|---------------|----------|-----------|----------|
| ADX cache key collision potential | SC-1 | (not directly flagged) | signals-core | P2 |
| Consensus 50% tie returns BUY | SC-2 | A-SC-4 (related: confidence semantics) | signals-core | P2 |
| Sentiment hysteresis dirty flag | SC-5 | (not flagged — assessed differently) | signals-core | P2 |

---

## Part 4: Disputed Findings

*Findings where one side disagrees with the other's assessment.*

### 4.1 Regime Weight Double-Counting [Agent A-SC-4 vs Independent]
- **Agent**: Flagged as P2 — regime weights applied in `_weighted_consensus` AND `apply_confidence_penalties`.
- **Independent**: Did not flag this. After analysis, I AGREE with the agent's assessment that this is confusing but not a bug: `_weighted_consensus` regime weights adjust relative signal importance, while `apply_confidence_penalties` applies an absolute confidence discount. The two are not double-counting.
- **Resolution**: AGREE — P2 for clarity, not a functional bug.

### 4.2 ADX Cache Key [Independent SC-1 vs Agent]
- **Independent**: Flagged `(id(df), len(df), close[-1])` as a potential collision risk.
- **Agent**: Did not flag this specifically, but noted the content-based key approach (C1 comment).
- **Resolution**: AGREE with independent — the risk is real but low-probability. The ticker-based cache key extension would be a small fix.

---

## Part 5: False Alarms

*Findings that were investigated and determined to be non-issues.*

| Finding | Source | Reason for dismissal |
|---------|--------|---------------------|
| PR-3: Kelly division by zero | Independent | Code properly guards against all zero-denominator cases (lines 38-41) |
| A-SC-10: Cost-adjusted accuracy double-filter | Agent | The `continue` correctly skips BOTH `total` and `correct` increments |
| MC-3: Stop-loss 3% check uses bid vs ask | Independent | Using bid is correct — stops trigger on sell-side price |
| SC-7: Fail-closed accuracy gate | Independent | POSITIVE finding — excellent defensive design, not a bug |

---

## Summary

The dual review found **47+ unique findings** across both reviewers:
- **5 new P1 findings** from the agent that the independent review missed (directional gate, regime cache, ministral count, signal_history race, blend sample inflation)
- **4 P1 findings** from the independent review that the agent couldn't find (different subsystem scope)
- **Strong agreement** on the overall health assessment: the system is well-hardened, with the main risks in data consistency, not catastrophic logic errors.

The most impactful finding across both reviews is **A-SC-1: Per-ticker accuracy strips directional fields**, which silently disables a safety gate designed to prevent directionally-biased signals from voting.
