"""Tests for StrategyOrchestrator threading and lifecycle."""
import time

from portfolio.strategies.base import SharedData, StrategyBase


class FakeStrategy(StrategyBase):
    """Test strategy that counts ticks."""
    def __init__(self, interval: float = 0.1, active: bool = True, sname: str = "fake"):
        self._interval = interval
        self._active = active
        self._name = sname
        self.tick_count = 0
        self.last_shared: SharedData | None = None

    def name(self) -> str:
        return self._name

    def poll_interval_seconds(self) -> float:
        return self._interval

    def tick(self, shared: SharedData) -> dict | None:
        self.tick_count += 1
        self.last_shared = shared
        return None

    def is_active(self) -> bool:
        return self._active

    def status_summary(self) -> str:
        return f"{self._name}: {self.tick_count} ticks"


class CrashingStrategy(StrategyBase):
    """Strategy that always raises."""
    def __init__(self):
        self.tick_count = 0

    def name(self):
        return "crasher"

    def poll_interval_seconds(self):
        return 0.05

    def tick(self, shared):
        self.tick_count += 1
        raise RuntimeError("boom")

    def is_active(self):
        return True

    def status_summary(self):
        return "crasher"


def test_orchestrator_lifecycle():
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    fake = FakeStrategy(interval=0.05)
    shared = SharedData(
        underlying_prices={"XAU-USD": 2345.0},
        fx_rate=10.5,
    )
    orch = StrategyOrchestrator(strategies=[fake], shared_data=shared)
    orch.start()
    time.sleep(1.5)
    orch.stop()
    assert fake.tick_count >= 2
    assert fake.last_shared is not None
    assert fake.last_shared.underlying_prices["XAU-USD"] == 2345.0


def test_orchestrator_respects_interval():
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    fast = FakeStrategy(interval=0.05, sname="fast")
    slow = FakeStrategy(interval=5.0, sname="slow")
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=[fast, slow], shared_data=shared)
    orch.start()
    time.sleep(1.5)
    orch.stop()
    assert fast.tick_count >= 2
    assert slow.tick_count <= 1


def test_orchestrator_skips_inactive():
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    inactive = FakeStrategy(interval=0.05, active=False)
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=[inactive], shared_data=shared)
    orch.start()
    time.sleep(0.3)
    orch.stop()
    assert inactive.tick_count == 0


def test_orchestrator_isolates_errors():
    """A crashing strategy must not kill other strategies."""
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    healthy = FakeStrategy(interval=0.05, sname="healthy")
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(
        strategies=[CrashingStrategy(), healthy],
        shared_data=shared,
    )
    orch.start()
    time.sleep(1.5)
    orch.stop()
    assert healthy.tick_count >= 2


def test_orchestrator_halts_after_max_errors():
    """Strategy should be halted after MAX_CONSECUTIVE_ERRORS crashes."""
    from portfolio.strategies.orchestrator import MAX_CONSECUTIVE_ERRORS, StrategyOrchestrator

    crasher = CrashingStrategy()
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=[crasher], shared_data=shared)
    orch.start()
    # Wait enough for MAX_CONSECUTIVE_ERRORS ticks (0.05s each + 0.5s sleep)
    time.sleep(MAX_CONSECUTIVE_ERRORS * 0.5 + 1.0)
    orch.stop()
    assert "crasher" in orch._halted


def test_orchestrator_summary():
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    fake = FakeStrategy(interval=1.0)
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=[fake], shared_data=shared)
    summary = orch.summary()
    assert "fake" in summary
    assert "1 strategies" in summary


def test_orchestrator_handles_action():
    """Verify action handling sends telegram."""
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    sent_messages = []

    class ActionStrategy(StrategyBase):
        def __init__(self):
            self._fired = False
        def name(self): return "actor"
        def poll_interval_seconds(self): return 0.05
        def tick(self, shared):
            if not self._fired:
                self._fired = True
                return {"type": "BUY", "reason": "test signal"}
            return None
        def is_active(self): return True
        def status_summary(self): return "actor"

    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(
        strategies=[ActionStrategy()],
        shared_data=shared,
        send_telegram=lambda msg: sent_messages.append(msg),
    )
    orch.start()
    time.sleep(0.4)
    orch.stop()
    assert len(sent_messages) >= 1
    assert "ACTOR" in sent_messages[0]
    assert "BUY" in sent_messages[0]


def test_orchestrator_double_start():
    """Starting an already-running orchestrator should be a no-op."""
    from portfolio.strategies.orchestrator import StrategyOrchestrator

    fake = FakeStrategy(interval=1.0)
    shared = SharedData(underlying_prices={}, fx_rate=10.5)
    orch = StrategyOrchestrator(strategies=[fake], shared_data=shared)
    orch.start()
    orch.start()  # should not raise or create a second thread
    orch.stop()
