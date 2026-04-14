# Cross-Critique — Round 5 (2026-04-14)

## Methodology

Two independent review streams:
- **Agent stream**: 8 parallel `feature-dev:code-reviewer` subagents, one per subsystem
- **Manual stream**: Cross-cutting analysis focusing on systemic inter-subsystem issues

This document critiques each stream's findings from the other's perspective.

---

## Agent findings critiqued by Manual review

### Confirmed and strengthened

1. **PR-R5-1 (check_drawdown never called)** — Independently confirmed via grep. STILL unfixed
   from R4 (C6). This is the single most important finding across all rounds. Manual review
   elevates this from "one disconnected function" to "the entire risk subsystem is theater"
   (6 functions disconnected, not just one).

2. **OR-R5-1/OR-R5-2 (claude_gate bypass)** — Independently confirmed. Manual review identified
   the same 4+ bypass paths. The CLAUDECODE env var risk (OR-R5-10) is validated by the 34h
   outage documented in MEMORY.md.

3. **MC-R5-1 (stop-loss distance from entry, not bid)** — Confirmed. The user memory rule
   "NEVER place stop within 3% of current bid" is unambiguous. The code computes from entry
   price, which diverges from current bid as the position moves against the trader.

4. **SM-R5-1/SM-R5-2 (vix_term_structure noise)** — Confirmed structurally. The _Z_THRESHOLD=0.0
   makes _ratio_zscore a near-permanent voter, and _contango_depth creates BUY bias in the
   most common market state. Together they inject systematic noise into every signal computation.

5. **IN-R5-2 (journal context write_text non-atomic)** — Confirmed. This is an instance of the
   broader "raw file I/O violations" pattern found across 8+ locations in the codebase.

### Potentially overstated by agents

1. **IN-R5-1 (process_lock msvcrt 1-byte lock)** — Agent rates P1. Manual assessment: P2.
   The Windows msvcrt.locking documentation states the lock prevents ANY other locking call
   on the same range from the same byte position. The TOCTOU gap between open and lock is
   real but extremely narrow (microseconds). In practice, the singleton guard works — the
   scheduled task restarter waits 30s between attempts, so the window is never hit in normal
   operation. Downgraded to P2.

2. **SC-R5-4 (signal_history.py race)** — Agent rates P2. Manual assessment: P3. grep confirms
   `update_history()` has no callers in the codebase (dead code). The race can't fire on code
   that's never called. Downgraded to P3 (dead code, not a race condition).

3. **AV-R5-5 (portfolio/avanza/trading.py no whitelist)** — Agent rates P3. Confirmed as P3.
   The unified avanza package is not used in production. This is latent, not active risk.

### Missed by agents, found by Manual

1. **Error swallowing cascade (XC-R5-2)** — No agent identified the 25+ exception-swallowing
   blocks in main.py as a systemic pattern. Each agent reviewed its own subsystem's error
   handling but missed the cross-cutting pattern where persistent module failures generate
   WARNING logs every 60s without escalation or circuit-breaking.

2. **meta_learner.py raw JSON reads (XC-R5-5)** — No agent caught this because meta_learner.py
   doesn't cleanly fit into any of the 8 subsystems. It sits at the intersection of signals-core
   (it predicts from signal votes) and data-external (it reads metrics files). Two locations
   bypass file_utils.load_json().

---

## Manual findings critiqued by Agent review

### Confirmed by agents

1. **XC-R5-1 (risk management theater)** — PR-R5-1 through PR-R5-6 provide exhaustive
   per-function evidence. Agent review was MORE thorough here: identified 6 disconnected
   functions vs manual's 5 (missed classify_trade_risk).

2. **XC-R5-3 (stale data reads)** — DE-R5-2 (gold/BTC ratio), DE-R5-8 (crypto scheduler),
   OR-R5-9 (perception gate) all independently confirm the cross-subsystem stale data pattern.

3. **XC-R5-4 (min trade size inconsistency)** — PR-R5-3 provides the exact file:line evidence.

### Potentially overstated by Manual

1. **XC-R5-6 (atomic_append_jsonl not thread-safe)** — IN-R5-10 partially confirms but adds
   nuance: the main paths where ThreadPoolExecutor workers call atomic_append_jsonl are
   limited. signal_log writing happens on the main thread AFTER pool completion. The most
   likely concurrent-append path is agent_invocation.py during multi-agent mode. Maintained
   at P3 — real but low probability.

### Missed by Manual, found by Agents

1. **MC-R5-2 (BEAR MINI knockout `pass`)** — A critical metals-specific bug that the manual
   cross-cutting review missed. The `pass` on a detected knockout condition could lead to
   trading a worthless instrument.

2. **MC-R5-3 (limit sell + stop-loss overfill)** — The volume overlap creating potential short
   positions on leveraged certs is a genuine P1 that the cross-cutting review missed by not
   reading fin_snipe_manager.py deeply enough.

3. **SC-R5-1 (utility boost bypasses gate)** — A subtle signal engine math bug where a boost
   factor can push a sub-threshold signal past the accuracy gate. Cross-cutting review focused
   on the systemic "gate never called" pattern but missed this "gate called but bypassed" variant.

4. **DE-R5-1 (onchain_data cache still has ISO crash)** — The A-DE-5 fix was applied inconsistently.
   This type of "fix applied to path A but not path B" bug is a classic that subsystem-level
   review catches better than cross-cutting review.

5. **OR-R5-8 (wait_for_specialists sequential drain)** — A concurrency design flaw where
   sequential proc.wait() calls with a shared deadline starve later specialists. Deep subsystem
   knowledge required to catch this.

---

## Verdict

The dual approach produced complementary coverage:
- **Agents** excelled at deep subsystem-level bugs (math errors, missing guards, race conditions)
- **Manual** excelled at systemic cross-cutting patterns (risk theater, error swallowing, stale data)
- **Overlap** was productive: both independently confirmed the critical risk disconnection and
  Claude subprocess bypass issues, increasing confidence
- **Blind spots** existed in both: agents missed cross-subsystem error patterns; manual missed
  metals-specific volume overlap and signal engine math bypass
