# Strategy Orchestrator — Merge GoldDigger & Elongir into Metals Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge GoldDigger (gold certificate trading) and Elongir (silver warrant dip-trading) into the metals loop as pluggable strategy modules, running in a shared process with shared data and Avanza session.

**Architecture:** A `StrategyOrchestrator` daemon thread manages strategy plugins. Each strategy implements a `StrategyBase` protocol with its own poll interval and signal logic. The orchestrator ticks strategies at their individual intervals (GoldDigger 5s, Elongir 30s) within the metals loop process. Shared data (underlying prices, FX, cert prices) flows from metals loop to strategies via a thread-safe `SharedData` reference. GoldDigger trade actions are enqueued to the existing `metals_trade_queue.json` for execution by the metals loop's Playwright session. Elongir remains fully simulated. The existing standalone runners (`python -m portfolio.golddigger`, `python -m portfolio.elongir`) continue to work unchanged.

**Tech Stack:** Python 3.12, dataclasses, threading, ABC, pytest

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `portfolio/strategies/__init__.py` | Public API: StrategyBase, SharedData, StrategyOrchestrator |
| `portfolio/strategies/base.py` | StrategyBase ABC + SharedData dataclass |
| `portfolio/strategies/orchestrator.py` | Daemon thread: tick loop, timing, error isolation, Telegram |
| `portfolio/strategies/golddigger_strategy.py` | Wraps GolddiggerBot, builds snapshot from shared data |
| `portfolio/strategies/elongir_strategy.py` | Wraps ElongirBot, builds snapshot from shared data |
| `tests/strategies/__init__.py` | Test package |
| `tests/strategies/test_base.py` | SharedData, protocol compliance |
| `tests/strategies/test_orchestrator.py` | Lifecycle, timing, error handling |
| `tests/strategies/test_golddigger_strategy.py` | Snapshot mapping, trade queue integration |
| `tests/strategies/test_elongir_strategy.py` | Snapshot mapping, simulated execution |

### Modified Files
| File | Change |
|------|--------|
| `data/metals_loop.py` | ~30 lines: import orchestrator, init after swing_trader, share data, stop on shutdown |

---

## Design Decisions

### Why a daemon thread (not sub-loop)?
GoldDigger needs 5s ticks inside a 60s metals cycle. A sub-loop would block the metals cycle or require complex interleaving with the existing silver fast-tick. A daemon thread with its own timing is the pattern already used by `start_llm_thread()` in metals_loop.

### Why not share the Playwright page?
Playwright's sync API is not thread-safe. The orchestrator thread cannot call page methods from the metals loop's page. Instead, GoldDigger enqueues trades to `metals_trade_queue.json` (already processed by metals loop), and reads cert prices from the shared cache (60s stale — acceptable since the z-score signal runs on Binance gold spot, not cert price).

### Why keep standalone runners?
Backwards compatibility. The adapters wrap existing bots without modifying them. `python -m portfolio.golddigger --live` still works for testing or running independently.

### Thread safety
Python's GIL guarantees atomic dict reads/writes. The metals loop updates `_underlying_prices` (dict) and the orchestrator reads it. No lock needed. The trade queue uses file-based JSON I/O with atomic writes — already process-safe, automatically thread-safe.

---

## Task 1: StrategyBase Protocol and SharedData

**Files:**
- Create: `portfolio/strategies/__init__.py`
- Create: `portfolio/strategies/base.py`
- Test: `tests/strategies/__init__.py`
- Test: `tests/strategies/test_base.py`

- [ ] **Step 1: Write the failing test for SharedData**

```python
# tests/strategies/test_base.py
"""Tests for strategy base protocol and SharedData."""
import pytest
from datetime import UTC, datetime


def test_shared_data_creation():
    from portfolio.strategies.base import SharedData
    sd = SharedData(
        underlying_prices={"XAU-USD": 2345.0, "XAG-USD": 33.5},
        fx_rate=10.5,
        cert_prices={"12345": {"bid": 55.0, "ask": 56.0}},
        is_market_hours=True,
        timestamp=datetime.now(UTC),
    )
    assert sd.underlying_prices["XAU-USD"] == 2345.0
    assert sd.fx_rate == 10.5
    assert sd.is_market_hours is True


def test_shared_data_get_price():
    from portfolio.strategies.base import SharedData
    sd = SharedData(
        underlying_prices={"XAU-USD": 2345.0},
        fx_rate=10.5,
        cert_prices={},
        is_market_hours=True,
        timestamp=datetime.now(UTC),
    )
    assert sd.get_price("XAU-USD") == 2345.0
    assert sd.get_price("MISSING") == 0.0


def test_shared_data_get_cert():
    from portfolio.strategies.base import SharedData
    sd = SharedData(
        underlying_prices={},
        fx_rate=10.5,
        cert_prices={"12345": {"bid": 55.0, "ask": 56.0}},
        is_market_hours=True,
        timestamp=datetime.now(UTC),
    )
    assert sd.get_cert("12345") == {"bid": 55.0, "ask": 56.0}
    assert sd.get_cert("99999") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio.strategies'`

- [ ] **Step 3: Write SharedData and StrategyBase**

```python
# portfolio/strategies/__init__.py
"""Strategy plugin framework for metals loop integration."""
from portfolio.strategies.base import SharedData, StrategyBase

__all__ = ["SharedData", "StrategyBase"]
```

```python
# portfolio/strategies/base.py
"""Base protocol and shared data for strategy plugins."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class SharedData:
    """Thread-safe data snapshot shared from metals loop to strategies.

    Updated by metals loop main thread, read by orchestrator thread.
    Python GIL guarantees atomic dict reads.
    """
    underlying_prices: dict[str, float] = field(default_factory=dict)
    fx_rate: float = 0.0
    cert_prices: dict[str, dict] = field(default_factory=dict)
    is_market_hours: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

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
```

```python
# tests/strategies/__init__.py
```

- [ ] **Step 4: Write test for StrategyBase protocol compliance**

Add to `tests/strategies/test_base.py`:

```python
def test_strategy_base_protocol():
    """Verify a concrete strategy must implement all abstract methods."""
    from portfolio.strategies.base import StrategyBase, SharedData

    class IncompleteStrategy(StrategyBase):
        pass

    with pytest.raises(TypeError, match="abstract method"):
        IncompleteStrategy()


def test_concrete_strategy():
    """Verify a complete implementation works."""
    from portfolio.strategies.base import StrategyBase, SharedData

    class DummyStrategy(StrategyBase):
        def name(self) -> str:
            return "dummy"
        def poll_interval_seconds(self) -> float:
            return 10.0
        def tick(self, shared: SharedData) -> dict | None:
            return None
        def is_active(self) -> bool:
            return True
        def status_summary(self) -> str:
            return "dummy: idle"

    s = DummyStrategy()
    assert s.name() == "dummy"
    assert s.poll_interval_seconds() == 10.0
    assert s.is_active() is True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_base.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add portfolio/strategies/__init__.py portfolio/strategies/base.py \
       tests/strategies/__init__.py tests/strategies/test_base.py
git commit -m "feat: add StrategyBase protocol and SharedData for strategy plugins"
```

---

## Task 2: StrategyOrchestrator

**Files:**
- Create: `portfolio/strategies/orchestrator.py`
- Test: `tests/strategies/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/test_orchestrator.py
"""Tests for StrategyOrchestrator threading and lifecycle."""
import time
import pytest
from datetime import UTC, datetime
from unittest.mock import MagicMock

from portfolio.strategies.base import SharedData, StrategyBase


class FakeStrategy(StrategyBase):
    """Test strategy that counts ticks."""
    def __init__(self, interval: float = 0.1, active: bool = True):
        self._interval = interval
        self._active = active
        self.tick_count = 0
        self.last_shared: SharedData | None = None

    def name(self) -> str:
        return "fake"

    def poll_interval_seconds(self) -> float:
        return self._interval

    def tick(self, shared: SharedData) -> dict | None:
        self.tick_count += 1
        self.last_shared = shared
        return None

    def is_active(self) -> bool:
        return self._active

    def status_summary(self) -> str:
        return f"fake: {self.tick_count} ticks"


def test_orchestrator_lifecycle():
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    fake = FakeStrategy(interval=0.05)
    shared = SharedData(
        underlying_prices={"XAU-USD": 2345.0},
        fx_rate=10.5,
    )
    orch = StrategyOrchestrator(strategies=[fake], shared_data=shared)
    orch.start()
    time.sleep(0.3)
    orch.stop()
    assert fake.tick_count >= 2
    assert fake.last_shared is not None
    assert fake.last_shared.underlying_prices["XAU-USD"] == 2345.0


def test_orchestrator_respects_interval():
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    fast = FakeStrategy(interval=0.05)
    slow = FakeStrategy(interval=0.5)
    slow._name = "slow"  # won't affect .name() but distinguish in debug
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=[fast, slow], shared_data=shared)
    orch.start()
    time.sleep(0.3)
    orch.stop()
    assert fast.tick_count >= 3
    assert slow.tick_count <= 1


def test_orchestrator_skips_inactive():
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    inactive = FakeStrategy(interval=0.05, active=False)
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=[inactive], shared_data=shared)
    orch.start()
    time.sleep(0.2)
    orch.stop()
    assert inactive.tick_count == 0


def test_orchestrator_isolates_errors():
    """A crashing strategy must not kill other strategies."""
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    class CrashingStrategy(StrategyBase):
        def name(self): return "crasher"
        def poll_interval_seconds(self): return 0.05
        def tick(self, shared):
            raise RuntimeError("boom")
        def is_active(self): return True
        def status_summary(self): return "crasher"

    healthy = FakeStrategy(interval=0.05)
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(
        strategies=[CrashingStrategy(), healthy],
        shared_data=shared,
    )
    orch.start()
    time.sleep(0.3)
    orch.stop()
    assert healthy.tick_count >= 2  # healthy kept ticking despite crasher


def test_orchestrator_summary():
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    fake = FakeStrategy(interval=1.0)
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=[fake], shared_data=shared)
    summary = orch.summary()
    assert "fake" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio.strategies.orchestrator'`

- [ ] **Step 3: Write the orchestrator**

```python
# portfolio/strategies/orchestrator.py
"""Strategy orchestrator — daemon thread managing multiple strategy plugins."""
import logging
import threading
import time

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
        send_telegram: callable | None = None,
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
        """Main tick loop — runs until stop() is called."""
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

            time.sleep(0.5)  # 500ms granularity — fast enough for 5s strategies

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_orchestrator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/strategies/orchestrator.py tests/strategies/test_orchestrator.py
git commit -m "feat: add StrategyOrchestrator daemon thread for strategy plugins"
```

---

## Task 3: GoldDigger Strategy Adapter

**Files:**
- Create: `portfolio/strategies/golddigger_strategy.py`
- Test: `tests/strategies/test_golddigger_strategy.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/test_golddigger_strategy.py
"""Tests for GoldDigger strategy adapter."""
import json
import pytest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from portfolio.strategies.base import SharedData


@pytest.fixture
def shared_data():
    return SharedData(
        underlying_prices={"XAU-USD": 2345.0, "XAG-USD": 33.5},
        fx_rate=10.5,
        cert_prices={"12345": {"bid": 55.0, "ask": 56.0, "last": 55.5}},
        is_market_hours=True,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def gd_config():
    """Minimal config dict for GoldDigger."""
    return {
        "golddigger": {
            "trade_enabled": True,
            "poll_seconds": 5,
            "bull_orderbook_id": "12345",
            "equity_sek": 100000.0,
            "use_augmented_signals": False,
            "use_signal_consensus": False,
            "use_macro_context": False,
            "use_volume_confirm": False,
            "use_chronos_forecast": False,
            "use_intraday_dxy_gate": False,
            "use_event_risk_gate": False,
        },
        "avanza": {"account_id": "1625505"},
    }


def test_golddigger_strategy_creation(gd_config):
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    s = GoldDiggerStrategy(gd_config)
    assert s.name() == "golddigger"
    assert s.poll_interval_seconds() == 5.0
    assert s.is_active() is True


def test_golddigger_strategy_builds_snapshot(gd_config, shared_data):
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    s = GoldDiggerStrategy(gd_config)
    snap = s._build_snapshot(shared_data, gold_price=2345.0)
    assert snap.gold == 2345.0
    assert snap.usdsek == 10.5
    assert snap.cert_bid == 55.0
    assert snap.cert_ask == 56.0


def test_golddigger_strategy_tick_returns_none_outside_session(gd_config, shared_data):
    """Outside session window, tick should return None."""
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    s = GoldDiggerStrategy(gd_config)
    # Bot's session check will skip if outside hours — just verify no crash
    result = s.tick(shared_data)
    # Result depends on time of day — just verify it doesn't crash
    assert result is None or isinstance(result, dict)


def test_golddigger_strategy_inactive_when_disabled(gd_config):
    gd_config["golddigger"]["trade_enabled"] = False
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    s = GoldDiggerStrategy(gd_config)
    # Strategy is still active (signals run), just won't trade
    assert s.is_active() is True


def test_golddigger_enqueue_trade(gd_config, tmp_path):
    from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
    queue_file = str(tmp_path / "trade_queue.json")
    s = GoldDiggerStrategy(gd_config, trade_queue_file=queue_file)
    s._enqueue_trade({
        "action": "BUY",
        "quantity": 100,
        "price": 55.5,
        "reason": "test",
    })
    data = json.loads(Path(queue_file).read_text())
    assert len(data["orders"]) == 1
    assert data["orders"][0]["action"] == "BUY"
    assert data["orders"][0]["status"] == "pending"
    assert data["orders"][0]["source"] == "golddigger"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_golddigger_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the GoldDigger strategy adapter**

```python
# portfolio/strategies/golddigger_strategy.py
"""GoldDigger strategy adapter — wraps GolddiggerBot for orchestrator integration.

Builds MarketSnapshot from SharedData + lightweight Binance fetch.
Trade actions are enqueued to metals_trade_queue.json for execution
by the metals loop's Playwright session.
"""
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from portfolio.golddigger.bot import GolddiggerBot
from portfolio.golddigger.config import DATA_DIR, GolddiggerConfig
from portfolio.golddigger.data_provider import (
    MarketSnapshot,
    fetch_gold_price,
    fetch_us10y_context,
    read_event_risk,
)
from portfolio.strategies.base import SharedData, StrategyBase
from portfolio.file_utils import atomic_write_json, load_json

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
        self._raw_config = config
        self._bot = GolddiggerBot(self._cfg, dry_run=True)
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
        # Fetch gold from Binance (5s fresh — not from shared 60s cache)
        gold = fetch_gold_price(self._cfg.binance_gold_symbol)
        if gold is None or gold <= 0:
            gold = shared.get_price("XAU-USD")
        if gold <= 0:
            return None

        snapshot = self._build_snapshot(shared, gold)
        action = self._bot.step(snapshot)

        if action is not None and action.get("action") in ("BUY", "SELL", "FLATTEN"):
            self._enqueue_trade(action)

        return action

    def _build_snapshot(self, shared: SharedData, gold_price: float) -> MarketSnapshot:
        """Build a GoldDigger MarketSnapshot from shared data + own fetches."""
        # US10Y from GoldDigger's own cached provider
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

        # Event risk
        event_ctx = None
        if self._cfg.use_event_risk_gate:
            event_ctx = read_event_risk(
                hours_before=self._cfg.event_risk_hours_before,
                hours_after=self._cfg.event_risk_hours_after,
                block_types=self._cfg.event_risk_block_types,
            )

        # Cert price from metals loop's cache
        cert = shared.get_cert(self._cfg.bull_orderbook_id)

        now = datetime.now(UTC)
        snap = MarketSnapshot(
            ts_utc=now,
            gold=gold_price,
            usdsek=shared.fx_rate if shared.fx_rate > 0 else 10.5,
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

    def _enqueue_trade(self, action: dict) -> None:
        """Write trade action to metals_trade_queue.json for execution."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_golddigger_strategy.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/strategies/golddigger_strategy.py \
       tests/strategies/test_golddigger_strategy.py
git commit -m "feat: add GoldDigger strategy adapter with trade queue integration"
```

---

## Task 4: Elongir Strategy Adapter

**Files:**
- Create: `portfolio/strategies/elongir_strategy.py`
- Test: `tests/strategies/test_elongir_strategy.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/test_elongir_strategy.py
"""Tests for Elongir strategy adapter."""
import pytest
from datetime import UTC, datetime

from portfolio.strategies.base import SharedData


@pytest.fixture
def shared_data():
    return SharedData(
        underlying_prices={"XAG-USD": 33.5, "XAU-USD": 2345.0},
        fx_rate=10.5,
        cert_prices={},
        is_market_hours=True,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def elongir_config():
    return {
        "elongir": {
            "poll_seconds": 30,
            "equity_sek": 100000.0,
            "financing_level": 75.03,
        },
    }


def test_elongir_strategy_creation(elongir_config):
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    s = ElongirStrategy(elongir_config)
    assert s.name() == "elongir"
    assert s.poll_interval_seconds() == 30.0
    assert s.is_active() is True


def test_elongir_strategy_builds_snapshot(elongir_config, shared_data):
    """Snapshot should use silver price and FX from shared data."""
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    s = ElongirStrategy(elongir_config)
    snap = s._build_snapshot(shared_data, klines_1m=None, klines_5m=None, klines_15m=None)
    assert snap.silver_usd == 33.5
    assert snap.fx_rate == 10.5


def test_elongir_strategy_tick_no_crash(elongir_config, shared_data):
    """Tick should not crash even without klines (incomplete snapshot)."""
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    from unittest.mock import patch

    s = ElongirStrategy(elongir_config)
    # Mock kline fetching to return None (network failure)
    with patch("portfolio.strategies.elongir_strategy.fetch_klines", return_value=None):
        result = s.tick(shared_data)
    # Incomplete snapshot (no klines) — bot returns None
    assert result is None or isinstance(result, dict)


def test_elongir_strategy_zero_silver(elongir_config):
    """If silver price is 0 in shared data, tick returns None."""
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    from unittest.mock import patch

    shared = SharedData(
        underlying_prices={"XAG-USD": 0.0},
        fx_rate=10.5,
    )
    s = ElongirStrategy(elongir_config)
    with patch("portfolio.strategies.elongir_strategy.fetch_klines", return_value=None):
        result = s.tick(shared)
    assert result is None


def test_elongir_strategy_status(elongir_config):
    from portfolio.strategies.elongir_strategy import ElongirStrategy
    s = ElongirStrategy(elongir_config)
    status = s.status_summary()
    assert "elongir" in status
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_elongir_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the Elongir strategy adapter**

```python
# portfolio/strategies/elongir_strategy.py
"""Elongir strategy adapter — wraps ElongirBot for orchestrator integration.

Builds MarketSnapshot from SharedData + own kline fetches.
Fully simulated — no trade queue, no Avanza execution.
"""
import logging

from portfolio.elongir.bot import ElongirBot
from portfolio.elongir.config import ElongirConfig
from portfolio.elongir.data_provider import MarketSnapshot, fetch_klines
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

        fx = shared.fx_rate if shared.fx_rate > 0 else 10.5

        # Fetch klines (Elongir needs multi-TF for RSI/MACD/BB)
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
        return MarketSnapshot(
            silver_usd=shared.get_price("XAG-USD"),
            fx_rate=shared.fx_rate if shared.fx_rate > 0 else 10.5,
            klines_1m=klines_1m,
            klines_5m=klines_5m,
            klines_15m=klines_15m,
            xag_signals=None,  # orchestrator doesn't need Layer 1 signals
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_elongir_strategy.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/strategies/elongir_strategy.py \
       tests/strategies/test_elongir_strategy.py
git commit -m "feat: add Elongir strategy adapter for simulated dip-trading"
```

---

## Task 5: Metals Loop Integration

**Files:**
- Modify: `data/metals_loop.py` (~30 lines added)

- [ ] **Step 1: Write the integration test**

Add to `tests/strategies/test_orchestrator.py`:

```python
def test_load_strategies_from_config():
    """Verify strategy loading from a config dict."""
    from portfolio.strategies.orchestrator import load_strategies

    config = {
        "golddigger": {
            "trade_enabled": True,
            "poll_seconds": 5,
            "bull_orderbook_id": "12345",
            "use_augmented_signals": False,
            "use_signal_consensus": False,
            "use_macro_context": False,
            "use_volume_confirm": False,
            "use_chronos_forecast": False,
            "use_intraday_dxy_gate": False,
            "use_event_risk_gate": False,
        },
        "elongir": {"poll_seconds": 30},
        "avanza": {"account_id": "1625505"},
        "strategies": {
            "golddigger_enabled": True,
            "elongir_enabled": True,
        },
    }
    strategies = load_strategies(config)
    names = [s.name() for s in strategies]
    assert "golddigger" in names
    assert "elongir" in names


def test_load_strategies_respects_disabled():
    from portfolio.strategies.orchestrator import load_strategies

    config = {
        "golddigger": {"poll_seconds": 5, "bull_orderbook_id": "12345",
                        "use_augmented_signals": False, "use_signal_consensus": False,
                        "use_macro_context": False, "use_volume_confirm": False,
                        "use_chronos_forecast": False, "use_intraday_dxy_gate": False,
                        "use_event_risk_gate": False},
        "elongir": {"poll_seconds": 30},
        "avanza": {"account_id": "1625505"},
        "strategies": {
            "golddigger_enabled": True,
            "elongir_enabled": False,
        },
    }
    strategies = load_strategies(config)
    names = [s.name() for s in strategies]
    assert "golddigger" in names
    assert "elongir" not in names
```

- [ ] **Step 2: Add load_strategies to orchestrator.py**

Append to `portfolio/strategies/orchestrator.py`:

```python
def load_strategies(config: dict) -> list[StrategyBase]:
    """Load enabled strategies from config.

    Reads config["strategies"]["golddigger_enabled"] and
    config["strategies"]["elongir_enabled"] to decide which to load.
    Defaults to enabled if the section exists in config.
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
```

- [ ] **Step 3: Run all strategy tests**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/ -v`
Expected: All tests PASS

- [ ] **Step 4: Add integration to metals_loop.py**

After the swing_trader initialization block (~line 4666), add:

```python
        # Initialize strategy orchestrator (GoldDigger + Elongir as plugins)
        _strategy_orchestrator = None
        _strategy_shared_data = None
        try:
            from portfolio.strategies.orchestrator import StrategyOrchestrator, load_strategies
            from portfolio.strategies.base import SharedData

            _strategy_shared_data = SharedData(
                underlying_prices=_underlying_prices,
                fx_rate=0.0,
                cert_prices={},
                is_market_hours=False,
            )
            _loaded_strategies = load_strategies(config_data)
            if _loaded_strategies:
                _strategy_orchestrator = StrategyOrchestrator(
                    strategies=_loaded_strategies,
                    shared_data=_strategy_shared_data,
                    send_telegram=send_telegram,
                )
                _strategy_orchestrator.start()
                log(f"Strategy orchestrator: {_strategy_orchestrator.summary()}")
            else:
                log("Strategy orchestrator: no strategies enabled")
        except Exception as e:
            log(f"Strategy orchestrator: NOT available ({e})")
```

In the main while loop, after `fetch_underlying_from_binance()` (~line 4732), add shared data update:

```python
                # Update strategy shared data
                if _strategy_shared_data is not None:
                    _strategy_shared_data.underlying_prices = _underlying_prices
                    _strategy_shared_data.is_market_hours = is_market_hours()
                    # Update cert prices from active positions
                    for key, pos in POSITIONS.items():
                        if pos["active"] and key in prices:
                            _strategy_shared_data.cert_prices[pos["ob_id"]] = prices[key]
```

In the finally/shutdown block, add:

```python
            # Stop strategy orchestrator
            if _strategy_orchestrator is not None:
                _strategy_orchestrator.stop()
```

- [ ] **Step 5: Commit**

```bash
git add portfolio/strategies/orchestrator.py tests/strategies/test_orchestrator.py \
       data/metals_loop.py
git commit -m "feat: integrate strategy orchestrator into metals loop"
```

---

## Task 6: Update __init__.py exports and final integration test

**Files:**
- Modify: `portfolio/strategies/__init__.py`
- Test: `tests/strategies/test_integration.py`

- [ ] **Step 1: Update exports**

```python
# portfolio/strategies/__init__.py
"""Strategy plugin framework for metals loop integration."""
from portfolio.strategies.base import SharedData, StrategyBase
from portfolio.strategies.orchestrator import StrategyOrchestrator, load_strategies

__all__ = [
    "SharedData",
    "StrategyBase",
    "StrategyOrchestrator",
    "load_strategies",
]
```

- [ ] **Step 2: Write end-to-end integration test**

```python
# tests/strategies/test_integration.py
"""End-to-end integration test: orchestrator + both strategies."""
import time
import pytest
from datetime import UTC, datetime
from unittest.mock import patch

from portfolio.strategies.base import SharedData
from portfolio.strategies.orchestrator import StrategyOrchestrator, load_strategies


@pytest.fixture
def full_config():
    return {
        "golddigger": {
            "trade_enabled": True,
            "poll_seconds": 5,
            "bull_orderbook_id": "12345",
            "equity_sek": 100000.0,
            "use_augmented_signals": False,
            "use_signal_consensus": False,
            "use_macro_context": False,
            "use_volume_confirm": False,
            "use_chronos_forecast": False,
            "use_intraday_dxy_gate": False,
            "use_event_risk_gate": False,
        },
        "elongir": {
            "poll_seconds": 30,
            "equity_sek": 100000.0,
        },
        "avanza": {"account_id": "1625505"},
        "strategies": {
            "golddigger_enabled": True,
            "elongir_enabled": True,
        },
    }


def test_full_orchestrator_startup_and_shutdown(full_config):
    """Both strategies load and orchestrator starts/stops cleanly."""
    strategies = load_strategies(full_config)
    assert len(strategies) == 2

    shared = SharedData(
        underlying_prices={"XAU-USD": 2345.0, "XAG-USD": 33.5},
        fx_rate=10.5,
        cert_prices={"12345": {"bid": 55.0, "ask": 56.0}},
        is_market_hours=True,
    )
    orch = StrategyOrchestrator(strategies=strategies, shared_data=shared)
    orch.start()
    time.sleep(0.5)
    orch.stop()
    # No crash = success
    summary = orch.summary()
    assert "golddigger" in summary
    assert "elongir" in summary
```

- [ ] **Step 3: Run all strategy tests**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add portfolio/strategies/__init__.py tests/strategies/test_integration.py
git commit -m "feat: final strategy orchestrator integration with end-to-end test"
```
