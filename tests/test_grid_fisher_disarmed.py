"""The production grid config must stay disarmed while in surveillance mode.

2026-08-21: the operator confirmed nothing is meant to auto-trade at present.
`GRID_FISHER_PROBE_ONLY` had been left False, so the grid was armed for live
Avanza orders and inert only by accident — its host metals loop was not
running, the Avanza session was expired, and live buying power read 139 SEK
against a config written for ~7000. Restoring the session or funding the
account would have started placing real orders with no further approval.

This test is a tripwire, not a design constraint. When the grid is
deliberately taken live again, delete it in the same commit that flips the
flag — so arming is always an explicit, reviewed act rather than a drift.
"""

import portfolio.grid_fisher_config as gfc


def test_grid_fisher_is_probe_only():
    assert gfc.GRID_FISHER_PROBE_ONLY is True, (
        "grid_fisher is ARMED for live Avanza orders. The system is in "
        "surveillance mode; nothing should auto-trade. If arming is "
        "intentional, delete this test in the same commit."
    )


def test_probe_mode_actually_blocks_placement():
    """The flag must reach the engine, not just sit in the config module."""
    from portfolio.grid_fisher import GridFisher

    f = GridFisher.__new__(GridFisher)  # no __init__: we only want the wiring
    import inspect

    src = inspect.getsource(GridFisher.__init__)
    assert "self._probe_only = GRID_FISHER_PROBE_ONLY" in src, (
        "GridFisher no longer reads GRID_FISHER_PROBE_ONLY at init — the "
        "surveillance-mode tripwire above would be decorative."
    )
    del f


def test_reward_risk_is_recorded_as_unvalidated():
    """Target/stop geometry has never seen a real fill. Pin the numbers so a
    change to them is deliberate.

    +1.2% target vs -3.5% stop is 1:4 after costs (win +1.03%, loss -4.17%)
    and needs an 80.2% win rate. The barrier geometry alone is provably
    negative-EV; the only edge is half-spread capture at fill, whose
    break-even spread is ~0.538% against a flagship instrument at 0.50%.
    """
    assert gfc.GRID_TARGET_PCT == 1.2
    assert gfc.GRID_STOP_PCT == 3.5
