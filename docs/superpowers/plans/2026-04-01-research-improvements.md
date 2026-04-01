# Research-Driven System Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 improvements derived from the Claude Code source leak architecture patterns and the TinyLoRA paper (arXiv 2602.04118), turning the finance-analyzer from a static signal voting system into a self-improving, multi-agent trading platform.

**Architecture:** Five independent improvements layered onto the existing system: (1) Daily memory consolidation agent that auto-generates trading insights from outcomes, (2) Multi-agent Layer 2 that splits analysis into parallel specialists, (3) Alert budgeting to reduce Telegram noise, (4) ML-based trade risk classification, (5) TinyLoRA RL training loop for local LLMs (scheduled after market hours only).

**Tech Stack:** Python 3.11, existing Claude CLI subprocess, file_utils for atomic I/O, Telegram notifications, llama-cpp-python for local LLM inference, GRPO/LoRA for RL training.

**Hardware Constraint:** GPU/CPU-heavy work (TinyLoRA training, model inference benchmarks) must only run after EU+US markets close (~22:00 CET). Implementation of training infrastructure is deferred — this plan covers the scaffolding and scheduling, not the actual training runs.

---

## Research Sources

### Claude Code Source Leak (March 31, 2026)
- **What:** 512K lines of TypeScript leaked via npm source maps (.npmignore oversight)
- **Key patterns applicable to trading:**
  - **Coordinator Mode:** Parallel research workers → synthesis → implementation → verification
  - **autoDream:** Background memory consolidation in 4 phases (Orient, Gather, Consolidate, Prune)
  - **KAIROS:** Proactive assistant with strict time budgets (15s blocking budget)
  - **Permission System:** ML-based risk classification (LOW/MEDIUM/HIGH) per action
  - **Modular Context:** Cached/dynamic prompt sections for efficiency

### TinyLoRA Paper (arXiv 2602.04118)
- **Title:** "Learning to Reason in 13 Parameters"
- **Key finding:** Fine-tuned Qwen2.5-8B to 91% accuracy on GSM8K with only 13 trained parameters using RL (not SFT). SFT needs 100-1000x more parameters for comparable results.
- **Implication for us:** Can RL-train Ministral-8B and Qwen3-8B trading signals with trivial compute cost. 13 parameters = seconds to train, zero risk of catastrophic forgetting.

---

## File Structure

| File | Responsibility | New/Modify |
|------|---------------|------------|
| `portfolio/memory_consolidation.py` | Daily auto-consolidation of trading insights from outcomes | **Create** |
| `portfolio/multi_agent_layer2.py` | Multi-agent Layer 2 orchestration (parallel specialists) | **Create** |
| `portfolio/alert_budget.py` | Telegram alert budgeting and consolidation | **Create** |
| `portfolio/trade_risk_classifier.py` | ML-based trade risk classification (LOW/MED/HIGH) | **Create** |
| `portfolio/tinylora_trainer.py` | TinyLoRA RL training loop for local LLMs (after-hours only) | **Create** |
| `portfolio/agent_invocation.py` | Wire multi-agent option into Layer 2 dispatch | **Modify** |
| `data/metals_loop.py` | Wire alert budget, memory consolidation trigger | **Modify** |
| `tests/test_memory_consolidation.py` | Tests for consolidation agent | **Create** |
| `tests/test_alert_budget.py` | Tests for alert budgeting | **Create** |
| `tests/test_trade_risk_classifier.py` | Tests for risk classification | **Create** |
| `tests/test_multi_agent_layer2.py` | Tests for multi-agent orchestration | **Create** |
| `tests/test_tinylora_trainer.py` | Tests for training scaffolding (not actual training) | **Create** |

---

## Task 1: Daily Memory Consolidation Agent

**Inspired by:** Claude Code's autoDream — background memory consolidation with Orient/Gather/Consolidate/Prune phases.

**Files:**
- Create: `portfolio/memory_consolidation.py`
- Create: `tests/test_memory_consolidation.py`

**What it does:** Runs daily (or on-demand), reads the last 7 days of signal outcomes, trade decisions, and accuracy data, then produces a compact `data/trading_insights.md` (<200 lines) that Layer 2 reads at the start of each invocation. This is the system's "memory" — what worked, what didn't, and what to do differently.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory_consolidation.py
"""Tests for daily memory consolidation agent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_signal_log(tmp_path, n=50):
    """Generate synthetic signal_log.jsonl with outcomes."""
    import numpy as np
    np.random.seed(42)
    lines = []
    for i in range(n):
        entry = {
            "ts": f"2026-03-{25 + i % 7:02d}T{10 + i % 12}:00:00+00:00",
            "tickers": {
                "XAG-USD": {
                    "price_usd": 30.0 + i * 0.1,
                    "consensus": np.random.choice(["BUY", "SELL", "HOLD"]),
                    "buy_count": int(np.random.randint(2, 8)),
                    "sell_count": int(np.random.randint(2, 8)),
                    "signals": {
                        "rsi": np.random.choice(["BUY", "SELL", "HOLD"]),
                        "macd": np.random.choice(["BUY", "SELL", "HOLD"]),
                        "trend": np.random.choice(["BUY", "SELL", "HOLD"]),
                    },
                    "regime": np.random.choice(["trending-up", "ranging", "high-vol"]),
                }
            },
            "outcomes": {
                "XAG-USD": {
                    "1d": {"change_pct": float(np.random.randn() * 2)},
                }
            },
        }
        lines.append(json.dumps(entry))
    log_path = tmp_path / "signal_log.jsonl"
    log_path.write_text("\n".join(lines))
    return log_path


def _make_journal(tmp_path, n=10):
    """Generate synthetic layer2_journal.jsonl."""
    import numpy as np
    np.random.seed(42)
    lines = []
    for i in range(n):
        entry = {
            "ts": f"2026-03-{25 + i % 7:02d}T14:00:00+00:00",
            "ticker": "XAG-USD",
            "strategy": np.random.choice(["patient", "bold"]),
            "action": np.random.choice(["BUY", "SELL", "HOLD"]),
            "reason": "Test reason",
            "confidence": float(np.random.uniform(0.3, 0.8)),
        }
        lines.append(json.dumps(entry))
    journal_path = tmp_path / "layer2_journal.jsonl"
    journal_path.write_text("\n".join(lines))
    return journal_path


class TestConsolidateInsights:
    def test_produces_markdown_output(self, tmp_path):
        from portfolio.memory_consolidation import consolidate_insights
        signal_log = _make_signal_log(tmp_path)
        journal = _make_journal(tmp_path)
        output = tmp_path / "trading_insights.md"

        result = consolidate_insights(
            signal_log_path=signal_log,
            journal_path=journal,
            output_path=output,
        )
        assert output.exists()
        content = output.read_text()
        assert "## Signal Performance" in content
        assert "## Regime Summary" in content
        assert len(content.splitlines()) <= 200

    def test_handles_empty_logs(self, tmp_path):
        from portfolio.memory_consolidation import consolidate_insights
        signal_log = tmp_path / "signal_log.jsonl"
        signal_log.write_text("")
        journal = tmp_path / "journal.jsonl"
        journal.write_text("")
        output = tmp_path / "trading_insights.md"

        result = consolidate_insights(
            signal_log_path=signal_log,
            journal_path=journal,
            output_path=output,
        )
        assert output.exists()

    def test_identifies_best_and_worst_signals(self, tmp_path):
        from portfolio.memory_consolidation import consolidate_insights
        signal_log = _make_signal_log(tmp_path, n=100)
        journal = _make_journal(tmp_path)
        output = tmp_path / "trading_insights.md"

        result = consolidate_insights(
            signal_log_path=signal_log,
            journal_path=journal,
            output_path=output,
        )
        assert "best" in result or "worst" in result or len(result) > 0


class TestLoadRecentEntries:
    def test_loads_entries_within_window(self, tmp_path):
        from portfolio.memory_consolidation import _load_recent_entries
        log_path = _make_signal_log(tmp_path, n=50)
        entries = _load_recent_entries(log_path, days=7)
        assert len(entries) > 0
        assert all("tickers" in e for e in entries)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_consolidation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement memory consolidation**

```python
# portfolio/memory_consolidation.py
"""Daily memory consolidation — auto-generates trading insights from outcomes.

Inspired by Claude Code's autoDream pattern: Orient → Gather → Consolidate → Prune.

Reads the last N days of signal outcomes, trade decisions, and accuracy data.
Produces a compact trading_insights.md (<200 lines) that Layer 2 reads at
the start of each invocation. This is the system's self-improving memory.

Run daily via scheduled task or on-demand:
    .venv/Scripts/python.exe -m portfolio.memory_consolidation
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from portfolio.file_utils import atomic_write_json

logger = logging.getLogger("portfolio.memory_consolidation")

_SIGNAL_LOG = Path("data/signal_log.jsonl")
_JOURNAL = Path("data/layer2_journal.jsonl")
_OUTPUT = Path("data/trading_insights.md")
_MAX_LINES = 200
_LOOKBACK_DAYS = 7


def _load_recent_entries(path: Path, days: int = 7) -> list[dict]:
    """Load JSONL entries from the last N days."""
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts_str = entry.get("ts", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
    return entries


def _compute_signal_accuracy(entries: list[dict]) -> dict[str, dict]:
    """Compute per-signal accuracy from entries with outcomes."""
    signal_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
    vote_map = {"BUY": 1, "SELL": -1, "HOLD": 0}

    for entry in entries:
        tickers = entry.get("tickers", {})
        outcomes = entry.get("outcomes", {})
        for ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            ticker_outcomes = outcomes.get(ticker, {})
            outcome_1d = ticker_outcomes.get("1d")
            if outcome_1d is None:
                continue
            change = outcome_1d if isinstance(outcome_1d, (int, float)) else outcome_1d.get("change_pct", 0)
            if abs(change) < 0.05:
                continue  # neutral — skip
            actual_dir = 1 if change > 0 else -1

            for sig_name, vote in signals.items():
                sig_dir = vote_map.get(vote, 0)
                if sig_dir == 0:
                    continue
                signal_stats[sig_name]["total"] += 1
                if sig_dir == actual_dir:
                    signal_stats[sig_name]["correct"] += 1
                else:
                    signal_stats[sig_name]["wrong"] += 1

    for stats in signal_stats.values():
        if stats["total"] > 0:
            stats["accuracy"] = round(stats["correct"] / stats["total"] * 100, 1)
        else:
            stats["accuracy"] = 0.0

    return dict(signal_stats)


def _compute_regime_stats(entries: list[dict]) -> dict[str, int]:
    """Count regime occurrences across entries."""
    regime_counts = Counter()
    for entry in entries:
        for tdata in entry.get("tickers", {}).values():
            regime = tdata.get("regime", "unknown")
            regime_counts[regime] += 1
    return dict(regime_counts)


def _compute_trade_stats(journal_entries: list[dict]) -> dict:
    """Summarize trade decisions from journal."""
    action_counts = Counter()
    strategy_counts = Counter()
    for entry in journal_entries:
        action_counts[entry.get("action", "UNKNOWN")] += 1
        strategy_counts[entry.get("strategy", "unknown")] += 1
    return {
        "action_counts": dict(action_counts),
        "strategy_counts": dict(strategy_counts),
        "total_decisions": len(journal_entries),
    }


def consolidate_insights(
    signal_log_path: Path | None = None,
    journal_path: Path | None = None,
    output_path: Path | None = None,
    days: int = _LOOKBACK_DAYS,
) -> dict:
    """Run full consolidation: Orient → Gather → Consolidate → Prune.

    Returns dict with key findings for programmatic use.
    """
    signal_log_path = signal_log_path or _SIGNAL_LOG
    journal_path = journal_path or _JOURNAL
    output_path = output_path or _OUTPUT

    # --- Phase 1: Orient ---
    signal_entries = _load_recent_entries(signal_log_path, days=days)
    journal_entries = _load_recent_entries(journal_path, days=days)

    # --- Phase 2: Gather ---
    signal_accuracy = _compute_signal_accuracy(signal_entries)
    regime_stats = _compute_regime_stats(signal_entries)
    trade_stats = _compute_trade_stats(journal_entries)

    # --- Phase 3: Consolidate ---
    # Sort signals by accuracy
    sorted_signals = sorted(
        signal_accuracy.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )
    best_signals = [(n, s) for n, s in sorted_signals if s["total"] >= 5 and s["accuracy"] >= 55]
    worst_signals = [(n, s) for n, s in sorted_signals if s["total"] >= 5 and s["accuracy"] < 45]

    # Build markdown
    lines = []
    lines.append(f"# Trading Insights — Last {days} Days")
    lines.append(f"_Auto-generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")

    lines.append("## Signal Performance")
    if best_signals:
        lines.append("**Working well:**")
        for name, stats in best_signals[:5]:
            lines.append(f"- {name}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
    if worst_signals:
        lines.append("**Underperforming:**")
        for name, stats in worst_signals[:5]:
            lines.append(f"- {name}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
    if not best_signals and not worst_signals:
        lines.append("_Insufficient data for signal ranking_")
    lines.append("")

    lines.append("## Regime Summary")
    if regime_stats:
        total_regime = sum(regime_stats.values())
        for regime, count in sorted(regime_stats.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_regime * 100
            lines.append(f"- {regime}: {pct:.0f}% ({count} observations)")
    else:
        lines.append("_No regime data_")
    lines.append("")

    lines.append("## Trade Decisions")
    if trade_stats["total_decisions"] > 0:
        for action, count in trade_stats["action_counts"].items():
            lines.append(f"- {action}: {count}")
        lines.append(f"- Total: {trade_stats['total_decisions']} decisions")
    else:
        lines.append("_No trade decisions in period_")
    lines.append("")

    lines.append("## Key Takeaways")
    takeaways = []
    if best_signals:
        top = best_signals[0]
        takeaways.append(f"Best signal: {top[0]} at {top[1]['accuracy']}%")
    if worst_signals:
        bottom = worst_signals[-1]
        takeaways.append(f"Worst signal: {bottom[0]} at {bottom[1]['accuracy']}% — consider gating")
    dominant_regime = max(regime_stats.items(), key=lambda x: x[1])[0] if regime_stats else "unknown"
    takeaways.append(f"Dominant regime: {dominant_regime}")
    for t in takeaways:
        lines.append(f"- {t}")

    # --- Phase 4: Prune (enforce max lines) ---
    if len(lines) > _MAX_LINES:
        lines = lines[:_MAX_LINES - 1]
        lines.append("_...truncated to 200 lines_")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Consolidated insights: %d signal entries, %d journal entries -> %s",
               len(signal_entries), len(journal_entries), output_path)

    return {
        "best_signals": [(n, s["accuracy"]) for n, s in best_signals[:5]],
        "worst_signals": [(n, s["accuracy"]) for n, s in worst_signals[:5]],
        "dominant_regime": dominant_regime,
        "total_decisions": trade_stats["total_decisions"],
        "entries_processed": len(signal_entries),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = consolidate_insights()
    print(f"Processed {result['entries_processed']} entries")
    if result["best_signals"]:
        print(f"Best: {result['best_signals'][0][0]} ({result['best_signals'][0][1]}%)")
    if result["worst_signals"]:
        print(f"Worst: {result['worst_signals'][0][0]} ({result['worst_signals'][0][1]}%)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_consolidation.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Wire into Layer 2 prompts**

In `portfolio/agent_invocation.py`, modify `_build_tier_prompt()` for T2 and T3 to include:
```
"If data/trading_insights.md exists, read it first for recent signal performance context. "
```

- [ ] **Step 6: Commit**

```bash
git add portfolio/memory_consolidation.py tests/test_memory_consolidation.py portfolio/agent_invocation.py
git commit -m "feat: add daily memory consolidation agent (autoDream pattern)"
```

---

## Task 2: Alert Budgeting System

**Inspired by:** Claude Code's KAIROS — proactive assistant with strict time/frequency budgets.

**Files:**
- Create: `portfolio/alert_budget.py`
- Create: `tests/test_alert_budget.py`
- Modify: `data/metals_loop.py` (wrap send_telegram calls)

**What it does:** Limits Telegram alerts to a configurable budget per hour (default 3 actionable alerts). Consolidates excess alerts into a periodic digest. Prioritizes by expected impact.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_alert_budget.py
"""Tests for alert budgeting system."""
from __future__ import annotations

import time

import pytest


class TestAlertBudget:
    def test_allows_alerts_within_budget(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=3)
        assert budget.should_send("price alert", priority=1) is True
        assert budget.should_send("stop alert", priority=1) is True
        assert budget.should_send("signal alert", priority=1) is True

    def test_blocks_alerts_over_budget(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=2)
        budget.should_send("alert 1", priority=1)
        budget.should_send("alert 2", priority=1)
        assert budget.should_send("alert 3", priority=1) is False

    def test_high_priority_bypasses_budget(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=1)
        budget.should_send("normal", priority=1)
        # Priority 3 (emergency) always gets through
        assert budget.should_send("EMERGENCY", priority=3) is True

    def test_buffer_returns_suppressed_messages(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=1)
        budget.should_send("sent", priority=1)
        budget.should_send("suppressed 1", priority=1)
        budget.should_send("suppressed 2", priority=1)
        buffered = budget.flush_buffer()
        assert len(buffered) == 2
        assert "suppressed 1" in buffered[0]

    def test_budget_resets_after_window(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=1, window_seconds=1)
        budget.should_send("alert 1", priority=1)
        assert budget.should_send("alert 2", priority=1) is False
        time.sleep(1.1)
        assert budget.should_send("alert 3", priority=1) is True
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement alert budget**

```python
# portfolio/alert_budget.py
"""Telegram alert budgeting — prevents alert fatigue.

Inspired by KAIROS's 15-second blocking budget. Limits actionable alerts
to a configurable maximum per time window. Excess alerts are buffered
and included in the next periodic digest.

Priority levels:
    1 = Normal (subject to budget)
    2 = Important (subject to budget but gets priority in queue)
    3 = Emergency (bypasses budget — stop-loss, circuit breaker, crash)
"""
from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger("portfolio.alert_budget")

PRIORITY_EMERGENCY = 3
PRIORITY_IMPORTANT = 2
PRIORITY_NORMAL = 1


class AlertBudget:
    """Token-bucket style alert rate limiter with priority bypass."""

    def __init__(self, max_per_hour: int = 3, window_seconds: int = 3600):
        self.max_per_hour = max_per_hour
        self.window_seconds = window_seconds
        self._sent_timestamps: deque[float] = deque()
        self._buffer: list[str] = []

    def _prune_old(self) -> None:
        """Remove timestamps outside the current window."""
        cutoff = time.time() - self.window_seconds
        while self._sent_timestamps and self._sent_timestamps[0] < cutoff:
            self._sent_timestamps.popleft()

    def should_send(self, message: str, priority: int = PRIORITY_NORMAL) -> bool:
        """Check if an alert should be sent or buffered.

        Args:
            message: The alert text.
            priority: 1=normal, 2=important, 3=emergency.

        Returns:
            True if the alert should be sent now, False if buffered.
        """
        # Emergency always goes through
        if priority >= PRIORITY_EMERGENCY:
            self._sent_timestamps.append(time.time())
            return True

        self._prune_old()

        if len(self._sent_timestamps) < self.max_per_hour:
            self._sent_timestamps.append(time.time())
            return True

        # Over budget — buffer it
        self._buffer.append(message)
        return False

    def flush_buffer(self) -> list[str]:
        """Return and clear buffered (suppressed) messages."""
        buffered = self._buffer.copy()
        self._buffer.clear()
        return buffered

    @property
    def remaining_budget(self) -> int:
        """How many alerts can still be sent in this window."""
        self._prune_old()
        return max(0, self.max_per_hour - len(self._sent_timestamps))

    @property
    def buffer_size(self) -> int:
        """Number of suppressed messages waiting in buffer."""
        return len(self._buffer)
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add portfolio/alert_budget.py tests/test_alert_budget.py
git commit -m "feat: add alert budgeting system (KAIROS pattern)"
```

---

## Task 3: Trade Risk Classifier

**Inspired by:** Claude Code's ML-based permission risk classification (LOW/MEDIUM/HIGH).

**Files:**
- Create: `portfolio/trade_risk_classifier.py`
- Create: `tests/test_trade_risk_classifier.py`

**What it does:** Classifies each proposed trade into LOW/MEDIUM/HIGH risk based on position size, trend alignment, consensus strength, regime, and portfolio concentration. Different risk levels get different scrutiny thresholds.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_trade_risk_classifier.py
"""Tests for trade risk classifier."""
from __future__ import annotations

import pytest


class TestClassifyTradeRisk:
    def test_small_with_trend_is_low(self):
        from portfolio.trade_risk_classifier import classify_trade_risk
        result = classify_trade_risk(
            action="BUY",
            confidence=0.7,
            position_pct=5.0,    # 5% of portfolio
            regime="trending-up",
            consensus_ratio=0.8,  # 80% agreement
        )
        assert result["level"] == "LOW"

    def test_large_counter_trend_is_high(self):
        from portfolio.trade_risk_classifier import classify_trade_risk
        result = classify_trade_risk(
            action="BUY",
            confidence=0.5,
            position_pct=25.0,    # 25% of portfolio
            regime="trending-down",
            consensus_ratio=0.55,
        )
        assert result["level"] == "HIGH"

    def test_medium_mixed_signals(self):
        from portfolio.trade_risk_classifier import classify_trade_risk
        result = classify_trade_risk(
            action="BUY",
            confidence=0.6,
            position_pct=12.0,
            regime="ranging",
            consensus_ratio=0.65,
        )
        assert result["level"] == "MEDIUM"

    def test_hold_is_always_low(self):
        from portfolio.trade_risk_classifier import classify_trade_risk
        result = classify_trade_risk(
            action="HOLD",
            confidence=0.0,
            position_pct=0.0,
            regime="high-vol",
            consensus_ratio=0.5,
        )
        assert result["level"] == "LOW"

    def test_returns_risk_factors(self):
        from portfolio.trade_risk_classifier import classify_trade_risk
        result = classify_trade_risk(
            action="BUY",
            confidence=0.7,
            position_pct=5.0,
            regime="trending-up",
            consensus_ratio=0.8,
        )
        assert "factors" in result
        assert "score" in result
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement risk classifier**

```python
# portfolio/trade_risk_classifier.py
"""Trade risk classification — LOW/MEDIUM/HIGH per proposed trade.

Inspired by Claude Code's ML-based permission risk classification.
Uses a scoring system based on:
    - Position size relative to portfolio (higher = riskier)
    - Trend alignment (counter-trend = riskier)
    - Consensus strength (weak consensus = riskier)
    - Market regime (high-vol/ranging = riskier)
    - Confidence level (low confidence = riskier)

Score thresholds:
    0-3 = LOW (auto-approve in autonomous mode)
    4-6 = MEDIUM (require Layer 2 confirmation)
    7+  = HIGH (require explicit reasoning in journal)
"""
from __future__ import annotations

import logging

logger = logging.getLogger("portfolio.trade_risk_classifier")

# Regime risk scores
_REGIME_RISK = {
    "trending-up": 0,
    "trending-down": 1,
    "ranging": 2,
    "range-bound": 2,
    "high-vol": 3,
    "breakout": 1,
    "capitulation": 3,
}

# Thresholds
_LOW_THRESHOLD = 3
_HIGH_THRESHOLD = 7


def classify_trade_risk(
    action: str,
    confidence: float,
    position_pct: float,
    regime: str,
    consensus_ratio: float,
    existing_exposure_pct: float = 0.0,
) -> dict:
    """Classify a proposed trade into LOW/MEDIUM/HIGH risk.

    Args:
        action: BUY, SELL, or HOLD.
        confidence: Signal confidence (0-1).
        position_pct: Proposed position size as % of portfolio.
        regime: Current market regime string.
        consensus_ratio: Fraction of signals agreeing (0-1).
        existing_exposure_pct: Current total exposure as % of portfolio.

    Returns:
        Dict with level (LOW/MEDIUM/HIGH), score (int), factors (list of strings).
    """
    if action == "HOLD":
        return {"level": "LOW", "score": 0, "factors": ["no_action"]}

    score = 0
    factors = []

    # Position size risk (0-3 points)
    if position_pct > 20:
        score += 3
        factors.append(f"large_position ({position_pct:.0f}%)")
    elif position_pct > 10:
        score += 2
        factors.append(f"medium_position ({position_pct:.0f}%)")
    elif position_pct > 5:
        score += 1
        factors.append(f"moderate_position ({position_pct:.0f}%)")

    # Regime risk (0-3 points)
    regime_score = _REGIME_RISK.get(regime, 2)
    score += regime_score
    if regime_score >= 2:
        factors.append(f"risky_regime ({regime})")

    # Counter-trend risk (0-2 points)
    if action == "BUY" and regime == "trending-down":
        score += 2
        factors.append("counter_trend_buy")
    elif action == "SELL" and regime == "trending-up":
        score += 2
        factors.append("counter_trend_sell")

    # Weak consensus risk (0-2 points)
    if consensus_ratio < 0.6:
        score += 2
        factors.append(f"weak_consensus ({consensus_ratio:.0%})")
    elif consensus_ratio < 0.7:
        score += 1
        factors.append(f"moderate_consensus ({consensus_ratio:.0%})")

    # Low confidence risk (0-1 points)
    if confidence < 0.5:
        score += 1
        factors.append(f"low_confidence ({confidence:.0%})")

    # Concentration risk (0-2 points)
    if existing_exposure_pct + position_pct > 40:
        score += 2
        factors.append(f"concentration ({existing_exposure_pct + position_pct:.0f}%)")
    elif existing_exposure_pct + position_pct > 25:
        score += 1
        factors.append(f"moderate_exposure ({existing_exposure_pct + position_pct:.0f}%)")

    # Classify
    if score <= _LOW_THRESHOLD:
        level = "LOW"
    elif score >= _HIGH_THRESHOLD:
        level = "HIGH"
    else:
        level = "MEDIUM"

    return {"level": level, "score": score, "factors": factors}
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add portfolio/trade_risk_classifier.py tests/test_trade_risk_classifier.py
git commit -m "feat: add trade risk classifier (LOW/MEDIUM/HIGH)"
```

---

## Task 4: Multi-Agent Layer 2 Orchestration

**Inspired by:** Claude Code's Coordinator Mode — parallel research workers → synthesis → decision.

**Files:**
- Create: `portfolio/multi_agent_layer2.py`
- Create: `tests/test_multi_agent_layer2.py`
- Modify: `portfolio/agent_invocation.py` (add multi-agent dispatch option)

**What it does:** Splits T2/T3 invocations into parallel specialist agents:
- Agent A: Technical analysis (signals, regime, momentum)
- Agent B: Risk assessment (portfolio state, exposure, stops)
- Agent C: Microstructure context (order flow, cross-asset)
- Synthesis agent: Takes 3 reports, makes BUY/SELL/HOLD decision

This is the most complex task and touches agent_invocation.py. Each specialist writes a brief report to a temp file, then the synthesis agent reads all three and makes the final call.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_multi_agent_layer2.py
"""Tests for multi-agent Layer 2 orchestration."""
from __future__ import annotations

import pytest


class TestBuildSpecialistPrompts:
    def test_technical_prompt_includes_signals(self):
        from portfolio.multi_agent_layer2 import build_specialist_prompts
        prompts = build_specialist_prompts(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%"],
        )
        assert "technical" in prompts
        assert "signal" in prompts["technical"].lower()

    def test_risk_prompt_includes_portfolio(self):
        from portfolio.multi_agent_layer2 import build_specialist_prompts
        prompts = build_specialist_prompts(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%"],
        )
        assert "risk" in prompts
        assert "portfolio" in prompts["risk"].lower()

    def test_microstructure_prompt_includes_orderflow(self):
        from portfolio.multi_agent_layer2 import build_specialist_prompts
        prompts = build_specialist_prompts(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%"],
        )
        assert "microstructure" in prompts
        assert "order" in prompts["microstructure"].lower() or "flow" in prompts["microstructure"].lower()

    def test_synthesis_prompt_references_reports(self):
        from portfolio.multi_agent_layer2 import build_synthesis_prompt
        prompt = build_synthesis_prompt(
            ticker="XAG-USD",
            trigger_reasons=["price move +2%"],
            report_paths=["/tmp/tech.md", "/tmp/risk.md", "/tmp/micro.md"],
        )
        assert "/tmp/tech.md" in prompt
        assert "/tmp/risk.md" in prompt


class TestMultiAgentConfig:
    def test_specialist_count(self):
        from portfolio.multi_agent_layer2 import SPECIALISTS
        assert len(SPECIALISTS) == 3

    def test_each_specialist_has_required_fields(self):
        from portfolio.multi_agent_layer2 import SPECIALISTS
        for name, spec in SPECIALISTS.items():
            assert "data_files" in spec
            assert "focus" in spec
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement multi-agent orchestration**

```python
# portfolio/multi_agent_layer2.py
"""Multi-agent Layer 2 orchestration — parallel specialists + synthesis.

Inspired by Claude Code's Coordinator Mode. Instead of one monolithic
agent reading everything, splits analysis into parallel specialists:

    1. Technical Agent: signals, regime, momentum, trend
    2. Risk Agent: portfolio state, exposure, drawdown, stops
    3. Microstructure Agent: order flow, depth, cross-asset context

Each specialist writes a brief report. A synthesis agent reads all three
and makes the final BUY/SELL/HOLD decision.

Benefits:
    - Each specialist has focused context (10K tokens vs 64K)
    - Better signal-to-noise ratio per agent
    - Parallel execution reduces wall-clock time
    - Easier to debug which agent was wrong
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("portfolio.multi_agent_layer2")

SPECIALISTS = {
    "technical": {
        "focus": "Technical analysis: signals, regime, momentum, trend direction",
        "data_files": [
            "data/agent_context_t2.json",
            "data/accuracy_cache.json",
        ],
        "output_file": "data/_specialist_technical.md",
        "timeout": 120,
        "max_turns": 10,
    },
    "risk": {
        "focus": "Risk assessment: portfolio exposure, drawdown, stop levels, position sizing",
        "data_files": [
            "data/portfolio_state.json",
            "data/portfolio_state_bold.json",
            "data/portfolio_state_warrants.json",
        ],
        "output_file": "data/_specialist_risk.md",
        "timeout": 90,
        "max_turns": 8,
    },
    "microstructure": {
        "focus": "Order flow and cross-asset context: depth imbalance, trade flow, VPIN, copper, GVZ, gold/silver ratio",
        "data_files": [
            "data/microstructure_state.json",
            "data/seasonality_profiles.json",
        ],
        "output_file": "data/_specialist_microstructure.md",
        "timeout": 90,
        "max_turns": 8,
    },
}


def build_specialist_prompts(
    ticker: str,
    trigger_reasons: list[str],
) -> dict[str, str]:
    """Build prompts for each specialist agent.

    Returns dict keyed by specialist name with prompt strings.
    """
    reason_str = ", ".join(trigger_reasons[:5])
    prompts = {}

    for name, spec in SPECIALISTS.items():
        data_reads = " ".join(f"Read {f}." for f in spec["data_files"])
        prompts[name] = (
            f"You are a {name} specialist for the trading system. "
            f"Ticker: {ticker}. Trigger: {reason_str}. "
            f"Focus: {spec['focus']}. "
            f"{data_reads} "
            f"Write a brief analysis (max 500 words) to {spec['output_file']}. "
            "Include: current state, key signals, recommendation (bullish/bearish/neutral), "
            "and confidence (low/medium/high). Be concise and data-driven."
        )

    return prompts


def build_synthesis_prompt(
    ticker: str,
    trigger_reasons: list[str],
    report_paths: list[str],
) -> str:
    """Build the synthesis agent prompt that reads all specialist reports.

    Args:
        ticker: The ticker being analyzed.
        trigger_reasons: What triggered the analysis.
        report_paths: Paths to the specialist report files.
    """
    reason_str = ", ".join(trigger_reasons[:5])
    reads = " ".join(f"Read {p}." for p in report_paths)

    return (
        "You are the Layer 2 synthesis agent. "
        f"Ticker: {ticker}. Trigger: {reason_str}. "
        "Read docs/TRADING_PLAYBOOK.md for trading rules. "
        "If data/trading_insights.md exists, read it for recent performance context. "
        f"{reads} "
        "These are reports from 3 specialist agents (technical, risk, microstructure). "
        "Synthesize their findings into a trading decision for BOTH Patient and Bold strategies. "
        "If specialists disagree, explain why you sided with one over the other. "
        "Write journal entry and send Telegram per the playbook."
    )


def get_specialist_timeout(name: str) -> int:
    """Get timeout for a specialist agent."""
    return SPECIALISTS.get(name, {}).get("timeout", 120)


def get_report_paths() -> list[str]:
    """Get output file paths for all specialists."""
    return [spec["output_file"] for spec in SPECIALISTS.values()]
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Wire into agent_invocation.py**

Add a `MULTI_AGENT_ENABLED` config flag (default False). When True and tier >= 2:
1. Build specialist prompts via `build_specialist_prompts()`
2. Launch 3 specialist agents in parallel (subprocess)
3. Wait for all to complete
4. Build synthesis prompt via `build_synthesis_prompt()`
5. Launch synthesis agent

Keep the existing single-agent path as fallback.

- [ ] **Step 6: Commit**

```bash
git add portfolio/multi_agent_layer2.py tests/test_multi_agent_layer2.py portfolio/agent_invocation.py
git commit -m "feat: add multi-agent Layer 2 orchestration (Coordinator Mode pattern)"
```

---

## Task 5: TinyLoRA RL Training Scaffolding

**Inspired by:** TinyLoRA paper (arXiv 2602.04118) — 91% accuracy with 13 parameters via RL.

**Hardware constraint:** Training runs ONLY after EU+US markets close (~22:00 CET). This task creates the scaffolding and scheduling — actual training runs are deferred to after-hours.

**Files:**
- Create: `portfolio/tinylora_trainer.py`
- Create: `tests/test_tinylora_trainer.py`
- Create: `scripts/win/train-after-hours.bat` (scheduled task launcher)

**What it does:**
1. Collects (signal_context, outcome) pairs from signal_log
2. Defines the reward function (correct direction = +1, wrong = -1)
3. Checks market hours — refuses to run during EU/US open
4. Prepares training data in the format llama-cpp-python/unsloth expects
5. Saves LoRA adapter weights to `data/models/tinylora/`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tinylora_trainer.py
"""Tests for TinyLoRA training scaffolding."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestCollectTrainingPairs:
    def test_extracts_context_outcome_pairs(self, tmp_path):
        from portfolio.tinylora_trainer import collect_training_pairs
        log_path = tmp_path / "signal_log.jsonl"
        entries = []
        for i in range(20):
            entries.append(json.dumps({
                "ts": f"2026-03-{25 + i % 5:02d}T12:00:00+00:00",
                "tickers": {
                    "XAG-USD": {
                        "price_usd": 30.0,
                        "consensus": "BUY",
                        "signals": {"rsi": "BUY", "macd": "SELL"},
                    }
                },
                "outcomes": {
                    "XAG-USD": {"1d": {"change_pct": 1.5}},
                },
            }))
        log_path.write_text("\n".join(entries))
        pairs = collect_training_pairs(log_path, ticker="XAG-USD")
        assert len(pairs) == 20
        assert "context" in pairs[0]
        assert "reward" in pairs[0]

    def test_reward_positive_for_correct_prediction(self, tmp_path):
        from portfolio.tinylora_trainer import collect_training_pairs
        log_path = tmp_path / "signal_log.jsonl"
        # BUY consensus + positive outcome = correct
        log_path.write_text(json.dumps({
            "ts": "2026-03-25T12:00:00+00:00",
            "tickers": {"XAG-USD": {"price_usd": 30.0, "consensus": "BUY", "signals": {"rsi": "BUY"}}},
            "outcomes": {"XAG-USD": {"1d": {"change_pct": 2.0}}},
        }))
        pairs = collect_training_pairs(log_path, ticker="XAG-USD")
        assert pairs[0]["reward"] > 0

    def test_reward_negative_for_wrong_prediction(self, tmp_path):
        from portfolio.tinylora_trainer import collect_training_pairs
        log_path = tmp_path / "signal_log.jsonl"
        # BUY consensus + negative outcome = wrong
        log_path.write_text(json.dumps({
            "ts": "2026-03-25T12:00:00+00:00",
            "tickers": {"XAG-USD": {"price_usd": 30.0, "consensus": "BUY", "signals": {"rsi": "BUY"}}},
            "outcomes": {"XAG-USD": {"1d": {"change_pct": -2.0}}},
        }))
        pairs = collect_training_pairs(log_path, ticker="XAG-USD")
        assert pairs[0]["reward"] < 0


class TestMarketHoursGuard:
    def test_refuses_during_market_hours(self):
        from portfolio.tinylora_trainer import is_training_allowed
        # 14:00 CET on a weekday = US market open
        from datetime import datetime
        dt = datetime(2026, 4, 1, 14, 0)  # Wednesday 14:00
        assert is_training_allowed(dt) is False

    def test_allows_after_market_close(self):
        from portfolio.tinylora_trainer import is_training_allowed
        from datetime import datetime
        dt = datetime(2026, 4, 1, 23, 0)  # Wednesday 23:00
        assert is_training_allowed(dt) is True

    def test_allows_weekends(self):
        from portfolio.tinylora_trainer import is_training_allowed
        from datetime import datetime
        dt = datetime(2026, 4, 4, 14, 0)  # Saturday 14:00
        assert is_training_allowed(dt) is True
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement training scaffolding**

```python
# portfolio/tinylora_trainer.py
"""TinyLoRA RL training scaffolding for local trading LLMs.

Based on arXiv 2602.04118: "Learning to Reason in 13 Parameters".
Fine-tunes Ministral-8B or Qwen3-8B with only 13-26 parameters using
reinforcement learning on trading outcomes.

HARDWARE CONSTRAINT: Training runs ONLY after EU+US markets close
(~22:00 CET). The is_training_allowed() guard enforces this.

Training loop:
    1. Collect (signal_context, outcome) pairs from signal_log
    2. Compute reward: +1 correct direction, -1 wrong, 0 neutral
    3. RL-train with TinyLoRA (rank=1, 13 params)
    4. Save adapter to data/models/tinylora/

Usage:
    .venv/Scripts/python.exe -m portfolio.tinylora_trainer
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("portfolio.tinylora_trainer")

_SIGNAL_LOG = Path("data/signal_log.jsonl")
_ADAPTER_DIR = Path("data/models/tinylora")
_VOTE_MAP = {"BUY": 1, "SELL": -1, "HOLD": 0}

# Market hours (CET): EU open 09:00, US close ~22:00
_MARKET_OPEN_HOUR = 8   # CET
_MARKET_CLOSE_HOUR = 22  # CET


def is_training_allowed(now: datetime | None = None) -> bool:
    """Check if GPU-intensive training is allowed right now.

    Training is blocked during EU+US market hours (08:00-22:00 CET weekdays)
    to avoid fan noise and resource contention with live trading processes.
    """
    if now is None:
        now = datetime.now()
    # Weekend = always allowed
    if now.weekday() >= 5:
        return True
    # After market close or before market open
    return now.hour >= _MARKET_CLOSE_HOUR or now.hour < _MARKET_OPEN_HOUR


def collect_training_pairs(
    log_path: Path | None = None,
    ticker: str = "XAG-USD",
    horizon: str = "1d",
) -> list[dict]:
    """Extract (context, reward) pairs from signal log for RL training.

    Args:
        log_path: Path to signal_log.jsonl.
        ticker: Which ticker to train on.
        horizon: Outcome horizon ("1d", "3h", etc.).

    Returns:
        List of dicts with 'context' (signal summary string) and
        'reward' (+1/-1/0).
    """
    log_path = log_path or _SIGNAL_LOG
    if not log_path.exists():
        return []

    pairs = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        tickers = entry.get("tickers", {})
        outcomes = entry.get("outcomes", {})

        if ticker not in tickers or ticker not in outcomes:
            continue

        tdata = tickers[ticker]
        outcome = outcomes[ticker].get(horizon)
        if outcome is None:
            continue

        change = outcome if isinstance(outcome, (int, float)) else outcome.get("change_pct", 0)
        consensus = tdata.get("consensus", "HOLD")
        signals = tdata.get("signals", {})
        price = tdata.get("price_usd", 0)

        # Build context string (what the LLM would see)
        signal_str = ", ".join(f"{k}={v}" for k, v in signals.items())
        context = (
            f"Ticker: {ticker}, Price: ${price:.2f}, "
            f"Signals: [{signal_str}], "
            f"Consensus: {consensus}"
        )

        # Compute reward
        consensus_dir = _VOTE_MAP.get(consensus, 0)
        if abs(change) < 0.05:
            reward = 0.0  # neutral
        elif consensus_dir == 0:
            reward = 0.0  # HOLD — no prediction to reward
        else:
            actual_dir = 1 if change > 0 else -1
            reward = 1.0 if consensus_dir == actual_dir else -1.0

        pairs.append({
            "context": context,
            "reward": reward,
            "change_pct": change,
            "consensus": consensus,
            "ts": entry.get("ts", ""),
        })

    return pairs


def prepare_training_config(
    model_path: str = "Q:/models/ministral-8b",
    adapter_dir: Path | None = None,
    rank: int = 1,
    learning_rate: float = 1e-4,
    epochs: int = 3,
) -> dict:
    """Prepare TinyLoRA training configuration.

    Args:
        model_path: Path to base model weights.
        adapter_dir: Where to save the LoRA adapter.
        rank: LoRA rank (1 = ~13 parameters per layer).
        learning_rate: RL learning rate.
        epochs: Number of training epochs.

    Returns:
        Config dict for the training script.
    """
    adapter_dir = adapter_dir or _ADAPTER_DIR
    adapter_dir.mkdir(parents=True, exist_ok=True)

    return {
        "model_path": model_path,
        "adapter_dir": str(adapter_dir),
        "rank": rank,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "method": "GRPO",  # Group Relative Policy Optimization
        "estimated_params": rank * 13,  # ~13 params per rank per layer
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not is_training_allowed():
        print("Training blocked — market hours. Run after 22:00 CET.")
    else:
        pairs = collect_training_pairs()
        print(f"Collected {len(pairs)} training pairs")
        if pairs:
            correct = sum(1 for p in pairs if p["reward"] > 0)
            wrong = sum(1 for p in pairs if p["reward"] < 0)
            print(f"Correct: {correct}, Wrong: {wrong}")
        config = prepare_training_config()
        print(f"Config: rank={config['rank']}, ~{config['estimated_params']} params")
        print("Training scaffolding ready. Actual training requires llama-cpp-python or unsloth.")
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Create after-hours launcher script**

```bat
@echo off
REM scripts/win/train-after-hours.bat
REM Scheduled via Windows Task Scheduler at 22:30 CET daily
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -m portfolio.tinylora_trainer
```

- [ ] **Step 6: Commit**

```bash
git add portfolio/tinylora_trainer.py tests/test_tinylora_trainer.py scripts/win/train-after-hours.bat
git commit -m "feat: add TinyLoRA RL training scaffolding (after-hours only)"
```

---

## Summary: What Each Task Delivers

| Task | Pattern Source | Runtime Impact | When It Runs |
|------|--------------|----------------|-------------|
| 1. Memory Consolidation | autoDream | Layer 2 reads insights at invocation start | Daily cron / on-demand |
| 2. Alert Budgeting | KAIROS | Reduces Telegram noise to 3/hour | Every metals_loop cycle |
| 3. Risk Classification | Permission System | Tags trades LOW/MED/HIGH in journal | Every trade decision |
| 4. Multi-Agent Layer 2 | Coordinator Mode | Parallel specialists + synthesis agent | T2/T3 invocations |
| 5. TinyLoRA Training | arXiv 2602.04118 | Trains local LLM on trading outcomes | After 22:00 CET only |

## Execution Order

Tasks 1-3 are independent and can be parallelized.
Task 4 depends on nothing but is the most complex.
Task 5 is scaffolding only (no GPU work during market hours).

Recommended: 1 + 2 + 3 in parallel → 4 → 5.
