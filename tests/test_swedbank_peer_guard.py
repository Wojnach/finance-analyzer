"""Cross-machine guard: only one swedbank --loop may write Avanza at a time.

The singleton lock is a local PID file, so it cannot see the other machine.
Both the Deck and herc hold valid sessions for the same Avanza account, which
the real-money metals loop also uses.
"""

from __future__ import annotations

import datetime
import os
import sys

import pytest

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import swedbank_loop as sl


def _hb(status="ok", age_s=0.0, key="loop"):
    ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=age_s
    )
    return {key: {"status": status, "ts": ts.isoformat()}}


class TestIsPrimaryHost:
    def test_exact_match_is_primary(self):
        assert sl.is_primary_host(hostname="steamdeck", primary="steamdeck")

    def test_prefix_match_is_primary(self):
        # Hostnames pick up suffixes (steamdeck.local); the guard must not start
        # deferring just because DNS appended a domain.
        assert sl.is_primary_host(hostname="steamdeck.local", primary="steamdeck")

    def test_case_insensitive(self):
        assert sl.is_primary_host(hostname="SteamDeck", primary="steamdeck")

    def test_other_machine_is_not_primary(self):
        assert not sl.is_primary_host(hostname="herc2", primary="steamdeck")

    def test_empty_primary_is_never_primary(self):
        # An empty override must not silently promote every machine to primary,
        # which would disable the guard everywhere at once.
        assert not sl.is_primary_host(hostname="herc2", primary="")


class TestPeerLoopAlive:
    def test_running_peer_detected(self):
        assert sl.peer_loop_alive(url="x", fetch_fn=lambda u: _hb("ok"))

    def test_accepts_heartbeat_key_from_raw_file(self):
        # /api/swedbank nests under "loop"; the raw heartbeat file uses its own
        # shape. Both must be understood.
        assert sl.peer_loop_alive(url="x", fetch_fn=lambda u: _hb(key="heartbeat"))

    @pytest.mark.parametrize("status", ["stopped", "error", ""])
    def test_non_running_status_does_not_block(self, status):
        assert not sl.peer_loop_alive(url="x", fetch_fn=lambda u: _hb(status))

    def test_stale_heartbeat_does_not_block(self):
        stale = sl.PEER_HEARTBEAT_MAX_AGE_S + 60
        assert not sl.peer_loop_alive(url="x", fetch_fn=lambda u: _hb(age_s=stale))

    def test_unreachable_peer_does_not_block(self):
        # Deliberate: the Deck being off is exactly the case the other machine
        # must still cover, so an unknown peer must never block it.
        def boom(url):
            raise OSError("no route to host")

        assert not sl.peer_loop_alive(url="x", fetch_fn=boom)

    def test_malformed_payload_does_not_block(self):
        for payload in ({}, None, {"loop": {}}, {"loop": {"status": "ok"}}):
            assert not sl.peer_loop_alive(url="x", fetch_fn=lambda u, p=payload: p)

    def test_unparseable_timestamp_does_not_block(self):
        bad = {"loop": {"status": "ok", "ts": "not-a-timestamp"}}
        assert not sl.peer_loop_alive(url="x", fetch_fn=lambda u: bad)


class TestPeerGuardBlocks:
    @pytest.fixture(autouse=True)
    def _no_env_override(self, monkeypatch):
        monkeypatch.delenv("PF_SWEDBANK_IGNORE_PEER", raising=False)

    def test_primary_proceeds_when_peer_idle(self, monkeypatch):
        monkeypatch.setattr(sl, "_peer_config", lambda: ("steamdeck", "x", True))
        monkeypatch.setattr(sl, "is_primary_host", lambda primary=None: True)
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: False)
        assert not sl.peer_guard_blocks()

    def test_two_self_declared_primaries_both_refuse(self, monkeypatch):
        # config.json is per-machine, so promoting herc without demoting the Deck
        # leaves two primaries. "Both are primary" must not mean two writers.
        monkeypatch.setattr(sl, "_peer_config", lambda: ("steamdeck", "x", True))
        monkeypatch.setattr(sl, "is_primary_host", lambda primary=None: True)
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: True)
        assert sl.peer_guard_blocks()

    def test_primary_collision_is_still_overridable(self, monkeypatch):
        monkeypatch.setattr(sl, "_peer_config", lambda: ("steamdeck", "x", True))
        monkeypatch.setattr(sl, "is_primary_host", lambda primary=None: True)
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: True)
        assert not sl.peer_guard_blocks(ignore_peer=True)

    def test_secondary_defers_to_live_primary(self, monkeypatch):
        monkeypatch.setattr(sl, "_peer_config", lambda: ("steamdeck", "x", True))
        monkeypatch.setattr(sl, "is_primary_host", lambda primary=None: False)
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: True)
        assert sl.peer_guard_blocks()

    def test_secondary_proceeds_when_primary_idle(self, monkeypatch):
        monkeypatch.setattr(sl, "_peer_config", lambda: ("steamdeck", "x", True))
        monkeypatch.setattr(sl, "is_primary_host", lambda primary=None: False)
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: False)
        assert not sl.peer_guard_blocks()

    def test_cli_flag_overrides(self, monkeypatch):
        monkeypatch.setattr(sl, "_peer_config", lambda: ("steamdeck", "x", True))
        monkeypatch.setattr(sl, "is_primary_host", lambda primary=None: False)
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: True)
        assert not sl.peer_guard_blocks(ignore_peer=True)

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("PF_SWEDBANK_IGNORE_PEER", "1")
        monkeypatch.setattr(sl, "_peer_config", lambda: ("steamdeck", "x", True))
        monkeypatch.setattr(sl, "is_primary_host", lambda primary=None: False)
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: True)
        assert not sl.peer_guard_blocks()

    def test_config_can_disable_guard(self, monkeypatch):
        monkeypatch.setattr(sl, "_peer_config", lambda: ("steamdeck", "x", False))
        monkeypatch.setattr(sl, "is_primary_host", lambda primary=None: False)
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: True)
        assert not sl.peer_guard_blocks()

    def test_config_can_invert_which_machine_yields(self, monkeypatch):
        # Promoting herc to primary must make the Deck the one that defers,
        # without editing the loop.
        monkeypatch.setattr(sl, "_peer_config", lambda: ("herc2", "x", True))
        monkeypatch.setattr(sl, "peer_loop_alive", lambda *a, **k: True)
        assert sl.is_primary_host(hostname="herc2", primary="herc2")
        assert not sl.is_primary_host(hostname="steamdeck", primary="herc2")


class TestOneShotIsNotGuarded:
    def test_once_path_does_not_call_peer_guard(self, monkeypatch, tmp_path):
        # herc is the testing bench: --once, probes and CLI calls must stay free.
        # Only a persistent --loop is guarded.
        monkeypatch.setattr(
            sl,
            "peer_guard_blocks",
            lambda **k: pytest.fail("--once must not consult the peer guard"),
        )
        monkeypatch.setattr(sl, "cycle", lambda: None)
        monkeypatch.setattr(sl, "SINGLETON_LOCK_FILE", str(tmp_path / "s.lock"))
        monkeypatch.setattr(
            sl, "acquire_singleton_lock", lambda *a, **k: str(tmp_path / "s.lock")
        )
        monkeypatch.setattr(sl, "release_singleton_lock", lambda p: None)
        assert sl.main(["--once"]) == 0

    def test_loop_returns_peer_exit_code_when_blocked(self, monkeypatch):
        monkeypatch.setattr(sl, "peer_guard_blocks", lambda **k: True)
        monkeypatch.setattr(
            sl,
            "acquire_singleton_lock",
            lambda *a, **k: pytest.fail("must refuse before taking the lock"),
        )
        assert sl.main(["--loop"]) == sl.EXIT_PEER_ACTIVE
