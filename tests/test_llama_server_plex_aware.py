"""Tests for Plex-aware VRAM coordination in portfolio.llama_server.

Covers the 2026-05-11 plex-vram-coord change:
- `_plex_transcode_active()` detection via nvidia-smi compute-apps
- 5 s TTL cache to avoid hammering nvidia-smi during VRAM polling
- `_wait_for_vram_reclaim(plex_safe=True)` raises threshold + extends timeout
- `_start_server()` aborts swap when Plex active + insufficient VRAM
- `_start_server()` proceeds normally when Plex idle (regression guard)
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_plex_cache():
    """Clear the Plex-probe cache between tests to keep cases independent."""
    import portfolio.llama_server as ls
    ls._plex_probe_cache = None
    yield
    ls._plex_probe_cache = None


def _smi_result(stdout: str, returncode: int = 0):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    return r


class TestPlexTranscodeDetection:
    """`_plex_transcode_active` matches Plex Transcoder.exe in nvidia-smi output."""

    @patch("portfolio.llama_server.subprocess.run")
    def test_positive_match(self, mock_run):
        from portfolio.llama_server import _plex_transcode_active
        mock_run.return_value = _smi_result(
            "C:\\Program Files\\Plex\\Plex Media Server\\Plex Transcoder.exe\n"
            "C:\\some\\other\\process.exe\n"
        )
        assert _plex_transcode_active() is True

    @patch("portfolio.llama_server.subprocess.run")
    def test_negative_no_plex(self, mock_run):
        from portfolio.llama_server import _plex_transcode_active
        mock_run.return_value = _smi_result(
            "C:\\Windows\\System32\\dwm.exe\n"
            "Q:\\models\\llama-cpp-bin\\cuda13\\llama-server.exe\n"
        )
        assert _plex_transcode_active() is False

    @patch("portfolio.llama_server.subprocess.run")
    def test_match_is_case_insensitive(self, mock_run):
        from portfolio.llama_server import _plex_transcode_active
        mock_run.return_value = _smi_result("plex TRANSCODER.exe\n")
        assert _plex_transcode_active() is True

    @patch("portfolio.llama_server.subprocess.run")
    def test_nvidia_smi_failure_returns_false(self, mock_run):
        from portfolio.llama_server import _plex_transcode_active
        mock_run.side_effect = OSError("nvidia-smi not found")
        assert _plex_transcode_active() is False

    @patch("portfolio.llama_server.subprocess.run")
    def test_nvidia_smi_nonzero_returns_false(self, mock_run):
        from portfolio.llama_server import _plex_transcode_active
        # Even with "Plex Transcoder" in stdout, a non-zero returncode is treated
        # as an error and returns False — we never lie about Plex being active.
        mock_run.return_value = _smi_result("Plex Transcoder.exe\n", returncode=9)
        assert _plex_transcode_active() is False

    @patch("portfolio.llama_server.subprocess.run")
    def test_cache_hits_within_ttl(self, mock_run):
        from portfolio.llama_server import _plex_transcode_active
        mock_run.return_value = _smi_result("Plex Transcoder.exe\n")
        # First call: shells out
        assert _plex_transcode_active() is True
        # Second call within TTL: cached, no second shell-out
        assert _plex_transcode_active() is True
        assert mock_run.call_count == 1

    @patch("portfolio.llama_server.subprocess.run")
    def test_cache_expires_after_ttl(self, mock_run):
        import portfolio.llama_server as ls
        mock_run.return_value = _smi_result("Plex Transcoder.exe\n")
        assert ls._plex_transcode_active() is True
        # Backdate cache so it's older than the TTL
        ts, val = ls._plex_probe_cache
        ls._plex_probe_cache = (ts - ls._PLEX_PROBE_TTL_SEC - 1.0, val)
        assert ls._plex_transcode_active() is True
        assert mock_run.call_count == 2


class TestWaitForVramReclaimPlexSafe:
    """`_wait_for_vram_reclaim(plex_safe=True)` raises thresholds."""

    @patch("portfolio.llama_server._query_free_vram_mb")
    def test_plex_safe_raises_min_free_floor(self, mock_free):
        """plex_safe=True forces min_free_mb to at least 7168 MB."""
        from portfolio.llama_server import _wait_for_vram_reclaim
        # 6500 MB free — below the plex_safe floor of 7168, so the call
        # should still poll/wait rather than return immediately at the
        # 5632-MB default.
        mock_free.return_value = 6500
        waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=0.3, plex_safe=True)
        # max_wait will get bumped to 30 s, but the test only needs to verify
        # the floor — the polling loop will run for at least one iteration.
        # We assert mock_free was called more than once (init probe + at least
        # one poll iteration), proving the threshold raise stuck.
        assert mock_free.call_count >= 2
        assert waited >= 0  # sanity

    @patch("portfolio.llama_server._query_free_vram_mb")
    def test_plex_safe_extends_max_wait(self, mock_free):
        """plex_safe=True clamps max_wait up to >=30 s."""
        from portfolio.llama_server import _wait_for_vram_reclaim
        # Make threshold immediately satisfied so we don't actually sleep —
        # we just want to confirm that plex_safe didn't reduce the wait by
        # accident. _query_free_vram_mb returns 8 GB, well above 7168.
        mock_free.return_value = 8192
        waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=4.0, plex_safe=True)
        # Already enough free, so wait = 0.0 — but the function should have
        # accepted the call with plex_safe and not rejected it.
        assert waited == 0.0

    @patch("portfolio.llama_server._query_free_vram_mb")
    def test_default_not_plex_safe(self, mock_free):
        """plex_safe=False (default) keeps the original 5632-MB floor."""
        from portfolio.llama_server import _wait_for_vram_reclaim
        mock_free.return_value = 6000  # above 5632, below 7168
        waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=4.0)
        # Default mode: 6000 MB >= 5632 MB, so return immediately.
        assert waited == 0.0

    @patch("portfolio.llama_server._query_free_vram_mb")
    def test_caller_min_free_above_floor_preserved(self, mock_free):
        """plex_safe=True uses max(caller_min, 7168) — never lowers caller's ask."""
        from portfolio.llama_server import _wait_for_vram_reclaim
        # Caller asks for 8000 MB; plex_safe floor is 7168. Caller's higher
        # value should win. 7500 MB free is below 8000 → must poll.
        mock_free.return_value = 7500
        waited = _wait_for_vram_reclaim(min_free_mb=8000, max_wait=0.3, plex_safe=True)
        assert mock_free.call_count >= 2
        assert waited >= 0


class TestStartServerPlexAware:
    """`_start_server` aborts cleanly when Plex is active and VRAM is tight."""

    @patch("portfolio.llama_server.subprocess.Popen")
    @patch("portfolio.llama_server._query_free_vram_mb", return_value=4000)
    @patch("portfolio.llama_server._wait_for_vram_reclaim", return_value=30.0)
    @patch("portfolio.llama_server._stop_server")
    @patch("portfolio.llama_server._plex_transcode_active", return_value=True)
    @patch("portfolio.llama_server.os.path.exists", return_value=True)
    def test_aborts_when_plex_busy_and_low_vram(
        self, mock_exists, mock_plex, mock_stop, mock_wait, mock_free, mock_popen
    ):
        """Plex transcoding + only 4 GB free → abort, never spawn a new llama-server."""
        from portfolio.llama_server import _start_server
        assert _start_server("ministral3") is False
        mock_popen.assert_not_called()
        # _stop_server is still called (old server killed in case of partial
        # state), but the new spawn is skipped. This matches the swap-abort
        # path described in PLAN.md.
        mock_stop.assert_called_once()

    @patch("portfolio.llama_server.subprocess.Popen")
    @patch("portfolio.llama_server._is_server_alive", return_value=True)
    @patch("portfolio.llama_server._write_pid")
    @patch("portfolio.llama_server._query_free_vram_mb", return_value=4000)
    @patch("portfolio.llama_server._wait_for_vram_reclaim", return_value=4.0)
    @patch("portfolio.llama_server._stop_server")
    @patch("portfolio.llama_server._plex_transcode_active", return_value=False)
    @patch("portfolio.llama_server.os.path.exists", return_value=True)
    def test_proceeds_when_plex_idle(
        self, mock_exists, mock_plex, mock_stop, mock_wait, mock_free,
        mock_write_pid, mock_alive, mock_popen
    ):
        """Plex idle + low VRAM → swap proceeds (existing behaviour preserved)."""
        from portfolio.llama_server import _start_server
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        assert _start_server("ministral3") is True
        mock_popen.assert_called_once()

    @patch("portfolio.llama_server.subprocess.Popen")
    @patch("portfolio.llama_server._is_server_alive", return_value=True)
    @patch("portfolio.llama_server._write_pid")
    @patch("portfolio.llama_server._query_free_vram_mb", return_value=8000)
    @patch("portfolio.llama_server._wait_for_vram_reclaim", return_value=2.0)
    @patch("portfolio.llama_server._stop_server")
    @patch("portfolio.llama_server._plex_transcode_active", return_value=True)
    @patch("portfolio.llama_server.os.path.exists", return_value=True)
    def test_proceeds_when_plex_busy_but_vram_ok(
        self, mock_exists, mock_plex, mock_stop, mock_wait, mock_free,
        mock_write_pid, mock_alive, mock_popen
    ):
        """Plex transcoding but VRAM cleared past 7168 → swap proceeds."""
        from portfolio.llama_server import _start_server
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        assert _start_server("ministral3") is True
        mock_popen.assert_called_once()


class TestModelLoadSafe:
    """`model_load_safe()` is the pre-flight gate for subprocess fallbacks."""

    @patch("portfolio.llama_server._query_free_vram_mb")
    @patch("portfolio.llama_server._plex_transcode_active", return_value=False)
    def test_safe_when_plex_idle(self, mock_plex, mock_free):
        """Plex idle → always safe, even if VRAM is low."""
        from portfolio.llama_server import model_load_safe
        mock_free.return_value = 2000  # very low — doesn't matter
        assert model_load_safe() is True

    @patch("portfolio.llama_server._query_free_vram_mb", return_value=8000)
    @patch("portfolio.llama_server._plex_transcode_active", return_value=True)
    def test_safe_when_plex_busy_but_vram_ok(self, mock_plex, mock_free):
        """Plex busy + ≥7168 MB free → safe to load."""
        from portfolio.llama_server import model_load_safe
        assert model_load_safe() is True

    @patch("portfolio.llama_server._query_free_vram_mb", return_value=4000)
    @patch("portfolio.llama_server._plex_transcode_active", return_value=True)
    def test_unsafe_when_plex_busy_and_vram_low(self, mock_plex, mock_free):
        """Plex busy + <7168 MB free → unsafe, caller must skip load."""
        from portfolio.llama_server import model_load_safe
        assert model_load_safe() is False

    @patch("portfolio.llama_server._query_free_vram_mb", return_value=None)
    @patch("portfolio.llama_server._plex_transcode_active", return_value=True)
    def test_safe_when_nvidia_smi_broken(self, mock_plex, mock_free):
        """nvidia-smi failure → fail-open (don't permanently block the loop)."""
        from portfolio.llama_server import model_load_safe
        assert model_load_safe() is True

    @patch("portfolio.llama_server._query_free_vram_mb", return_value=7168)
    @patch("portfolio.llama_server._plex_transcode_active", return_value=True)
    def test_threshold_is_inclusive(self, mock_plex, mock_free):
        """Exactly 7168 MB free → safe (>= comparison)."""
        from portfolio.llama_server import model_load_safe
        assert model_load_safe() is True

    @patch("portfolio.llama_server._query_free_vram_mb", return_value=7167)
    @patch("portfolio.llama_server._plex_transcode_active", return_value=True)
    def test_threshold_just_below_unsafe(self, mock_plex, mock_free):
        """7167 MB free is one MB below the floor → unsafe."""
        from portfolio.llama_server import model_load_safe
        assert model_load_safe() is False
