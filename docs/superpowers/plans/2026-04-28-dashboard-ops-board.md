# Dashboard Operations Board — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Jenkins-style operational health view as the dashboard's new default tab — green/orange/red status cards for every running subsystem, drilldown strips for signals/LLMs/tickers, and an inline log inspector. Stay on the existing Flask + vanilla JS stack; reuse existing health helpers.

**Architecture:** Two new Flask endpoints (`/api/ops-status` rollup, `/api/logs/<source>` log tail) plus one additive field on `/api/health`. Frontend additions (CSS, HTML, JS) all go in the existing single-file `dashboard/static/index.html` under a `.ops-` / `OpsBoard` namespace. Static-export mode preserved by adding new endpoints to `dashboard/export_static.py`.

**Tech Stack:** Python 3.11 + Flask (existing), pytest (existing test patterns), vanilla JS + Chart.js 4.4.1 (already loaded), CSS variables (already in stylesheet). No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-04-28-dashboard-ops-board-design.md`

**Verification plugins available:** `chrome-devtools-mcp` (live screenshot, console, a11y audit), `frontend-design` (visual quality guidance).

**Security note:** All new frontend rendering uses safe DOM methods (`createElement` + `textContent` + `appendChild`). No `innerHTML` with interpolated content. The existing dashboard mixes both patterns; new code stays on the safe side.

---

## File Structure

**To create:**
- (none — all changes in existing files)

**To modify:**
| File | Change | Approx LOC |
|------|--------|-----------|
| `dashboard/app.py` | 5 new helpers + 2 new endpoints + extend `/api/health` | +250 |
| `dashboard/export_static.py` | Add 5 endpoints to `ENDPOINTS` list | +5 |
| `dashboard/static/index.html` | New CSS block + replace `#healthC` placeholder + change default tab | +700 |
| `tests/test_dashboard.py` | 4 new test classes (~12 tests) | +220 |
| `tests/test_dashboard_export_static.py` | Extend existing tests for new endpoints | +20 |
| `tests/test_dashboard_frontend.py` | 1 smoke test | +15 |

**Conventions:**
- All new helpers prefixed with `_compute_` and live as private module-level functions in `dashboard/app.py`
- All new CSS classes prefixed `ops-` (e.g. `.ops-card`, `.ops-dot`)
- All new JS attached to a single global `OpsBoard` namespace
- Status thresholds defined as module-level `OPS_THRESHOLDS` constant (config-extractable later)
- Frontend rendering uses safe DOM methods only — no `innerHTML` with interpolated user data

---

## Task 0: Worktree setup + design doc commit

**Files:**
- Existing: `docs/superpowers/specs/2026-04-28-dashboard-ops-board-design.md` (already written)
- Existing: `docs/superpowers/plans/2026-04-28-dashboard-ops-board.md` (this file)

- [ ] **Step 1: Create worktree** (per CLAUDE.md "Always use worktrees for changes")

```bash
cd /mnt/q/finance-analyzer
cmd.exe /c "git worktree add .worktrees/dashboard-ops-board -b feat/dashboard-ops-board"
```

Expected: new directory `.worktrees/dashboard-ops-board/` exists, on branch `feat/dashboard-ops-board`. All subsequent file edits use that path.

- [ ] **Step 2: Stage design + plan in worktree**

```bash
cd /mnt/q/finance-analyzer/.worktrees/dashboard-ops-board
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp /mnt/q/finance-analyzer/docs/superpowers/specs/2026-04-28-dashboard-ops-board-design.md docs/superpowers/specs/
cp /mnt/q/finance-analyzer/docs/superpowers/plans/2026-04-28-dashboard-ops-board.md docs/superpowers/plans/
```

- [ ] **Step 3: Commit design + plan**

```bash
cd /mnt/q/finance-analyzer/.worktrees/dashboard-ops-board
git add docs/superpowers/specs/2026-04-28-dashboard-ops-board-design.md
git add docs/superpowers/plans/2026-04-28-dashboard-ops-board.md
git commit -m "$(cat <<'EOF'
docs: dashboard ops board design + implementation plan

- Spec for Jenkins-style operations board (default landing tab)
- Step-ordered implementation plan with TDD checkpoints

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: `_status_color` helper + thresholds constant

The single function that maps a value + threshold dict to "green"/"orange"/"red". Used by every other helper. TDD this first.

**Files:**
- Modify: `dashboard/app.py` — add `OPS_THRESHOLDS` constant + `_status_color` function near top (after `_DEFAULT_TTL` declaration around line 60)
- Test: `tests/test_dashboard.py` — append new `class TestStatusColor` at end of file

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py` (after the last existing class):

```python
from dashboard.app import _status_color


class TestStatusColor:
    """Status color helper: maps a value + threshold dict to green/orange/red."""

    def test_under_direction_green(self):
        assert _status_color(30, {"green_under_s": 120, "orange_under_s": 300}, "under") == "green"

    def test_under_direction_orange(self):
        assert _status_color(200, {"green_under_s": 120, "orange_under_s": 300}, "under") == "orange"

    def test_under_direction_red(self):
        assert _status_color(400, {"green_under_s": 120, "orange_under_s": 300}, "under") == "red"

    def test_over_direction_green(self):
        assert _status_color(90, {"green_over_pct": 80, "orange_over_pct": 60}, "over") == "green"

    def test_over_direction_orange(self):
        assert _status_color(70, {"green_over_pct": 80, "orange_over_pct": 60}, "over") == "orange"

    def test_over_direction_red(self):
        assert _status_color(40, {"green_over_pct": 80, "orange_over_pct": 60}, "over") == "red"

    def test_none_value_is_red(self):
        # Missing data is not a green state.
        assert _status_color(None, {"green_under_s": 120, "orange_under_s": 300}, "under") == "red"

    def test_invalid_direction_raises(self):
        import pytest
        with pytest.raises(ValueError):
            _status_color(50, {"green_under_s": 120, "orange_under_s": 300}, "sideways")
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
cd /mnt/q/finance-analyzer
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestStatusColor -v
```

Expected: ImportError: cannot import name '_status_color' from 'dashboard.app'

- [ ] **Step 3: Implement the helper**

Find insertion point: `grep -n "_DEFAULT_TTL" dashboard/app.py | head -1`. Insert below that line.

```python
# Status color thresholds for /api/ops-status. green = healthy, orange = degraded,
# red = down or no data. "under" means lower is better (e.g. heartbeat age in seconds);
# "over" means higher is better (e.g. signal success rate %).
OPS_THRESHOLDS = {
    "layer1_heartbeat":  {"green_under_s": 120,  "orange_under_s": 300},
    "layer2_silence":    {"green_under_s": 7200, "orange_under_s": 14400},
    "metals_loop":       {"green_under_s": 120,  "orange_under_s": 300},
    "signals_aggregate": {"green_over_pct": 80,  "orange_over_pct": 60},
    "signal_individual": {"green_over_pct": 70,  "orange_over_pct": 50},
    "llm_individual":    {"green_over_pct": 90,  "orange_over_pct": 70},
    "ticker_accuracy":   {"green_over_pct": 55,  "orange_over_pct": 45},
    "accuracy_7d":       {"green_over_pct": 55,  "orange_over_pct": 45},
    "outcome_max_age_h": {"green_under_h": 6,    "orange_under_h": 12},
}


def _status_color(value, thresholds, direction):
    """Map a value to 'green' | 'orange' | 'red'.

    direction='under': lower is better. thresholds need 'green_under_*' and 'orange_under_*'.
    direction='over': higher is better. thresholds need 'green_over_*' and 'orange_over_*'.
    None value always returns 'red' (missing data is not a green state).
    """
    if value is None:
        return "red"
    if direction == "under":
        green_key = next(k for k in thresholds if k.startswith("green_under_"))
        orange_key = next(k for k in thresholds if k.startswith("orange_under_"))
        if value < thresholds[green_key]:
            return "green"
        if value < thresholds[orange_key]:
            return "orange"
        return "red"
    if direction == "over":
        green_key = next(k for k in thresholds if k.startswith("green_over_"))
        orange_key = next(k for k in thresholds if k.startswith("orange_over_"))
        if value >= thresholds[green_key]:
            return "green"
        if value >= thresholds[orange_key]:
            return "orange"
        return "red"
    raise ValueError(f"direction must be 'under' or 'over', got {direction!r}")
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestStatusColor -v
```

Expected: 8 passed.

- [ ] **Step 5: Run the full dashboard test suite — confirm no regressions**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -v --tb=short
```

Expected: all existing tests still pass; 8 new tests pass; no new failures.

- [ ] **Step 6: Commit**

```bash
cd /mnt/q/finance-analyzer/.worktrees/dashboard-ops-board
git add dashboard/app.py tests/test_dashboard.py
git commit -m "$(cat <<'EOF'
feat(dashboard): add OPS_THRESHOLDS + _status_color helper

Module-level constant + helper for green/orange/red status mapping.
Used by upcoming /api/ops-status helpers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_compute_metals_loop_status` helper

Reads `data/metals_invocations.jsonl` tail, returns the metals-loop subsystem status dict.

**Files:**
- Modify: `dashboard/app.py` — add helper after `_status_color`
- Test: `tests/test_dashboard.py` — append new test class

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
from dashboard.app import _compute_metals_loop_status
from portfolio import file_utils
import json
import time


class TestComputeMetalsLoopStatus:
    """_compute_metals_loop_status reads metals_invocations.jsonl tail."""

    def test_no_file_returns_red(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_metals_loop_status()
        assert result["status"] == "red"
        assert result["age_s"] is None
        assert "no data" in result["summary"].lower()

    def test_empty_file_returns_red(self, tmp_path, monkeypatch):
        (tmp_path / "metals_invocations.jsonl").write_text("")
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_metals_loop_status()
        assert result["status"] == "red"

    def test_recent_invocation_returns_green(self, tmp_path, monkeypatch):
        recent_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        entry = {"ts": recent_iso, "tier": 1, "trigger": "test"}
        (tmp_path / "metals_invocations.jsonl").write_text(json.dumps(entry) + "\n")
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_metals_loop_status()
        assert result["status"] == "green"
        assert result["age_s"] is not None
        assert result["age_s"] < 60

    def test_old_invocation_returns_red(self, tmp_path, monkeypatch):
        old_iso = "2020-01-01T00:00:00+00:00"
        entry = {"ts": old_iso, "tier": 1, "trigger": "test"}
        (tmp_path / "metals_invocations.jsonl").write_text(json.dumps(entry) + "\n")
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_metals_loop_status()
        assert result["status"] == "red"

    def test_corrupted_line_handled_gracefully(self, tmp_path, monkeypatch):
        recent_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        good_entry = {"ts": recent_iso}
        content = "this is not json\n" + json.dumps(good_entry) + "\n"
        (tmp_path / "metals_invocations.jsonl").write_text(content)
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_metals_loop_status()
        assert result["status"] == "green"
```

- [ ] **Step 2: Run the test — confirm failure**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestComputeMetalsLoopStatus -v
```

- [ ] **Step 3: Implement the helper**

Verify these are present in `dashboard/app.py` (top of file). Add any that are missing:
```python
from portfolio import file_utils
from datetime import datetime, timezone, timedelta
```

Then add `_compute_metals_loop_status` after `_status_color`:

```python
def _compute_metals_loop_status():
    """Return ops-status subsystem dict for the metals loop.

    Reads data/metals_invocations.jsonl tail; computes age of last invocation.
    Never raises — bad data → 'red' status with explanatory summary.
    """
    metals_log = DATA_DIR / "metals_invocations.jsonl"
    try:
        entries = file_utils.load_jsonl_tail(metals_log, max_entries=1)
    except FileNotFoundError:
        return {"status": "red", "summary": "no data (file missing)",
                "age_s": None, "details": {"file": str(metals_log)}}
    if not entries:
        return {"status": "red", "summary": "no data (empty file)",
                "age_s": None, "details": {}}
    last = entries[-1]
    ts = last.get("ts")
    if not ts:
        return {"status": "red", "summary": "no timestamp in last entry",
                "age_s": None, "details": {"last_entry": last}}
    parsed = _parse_iso8601(ts)
    if parsed is None:
        return {"status": "red", "summary": f"unparseable ts {ts!r}",
                "age_s": None, "details": {}}
    age_s = (datetime.now(parsed.tzinfo) - parsed).total_seconds()
    color = _status_color(age_s, OPS_THRESHOLDS["metals_loop"], "under")
    summary = f"last invocation {int(age_s)}s ago"
    return {"status": color, "summary": summary, "age_s": round(age_s, 1),
            "details": {"last_ts": ts, "last_tier": last.get("tier")}}
```

- [ ] **Step 4: Run the test — confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestComputeMetalsLoopStatus -v
```

Expected: 5 passed.

- [ ] **Step 5: Full suite check**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -x --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py tests/test_dashboard.py
git commit -m "$(cat <<'EOF'
feat(dashboard): _compute_metals_loop_status helper

Reads metals_invocations.jsonl tail, returns subsystem status dict
with age of last invocation. Handles missing file, empty file,
corrupted JSONL gracefully — never raises to caller.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `_compute_llm_health_summary` helper

Reads `data/forecast_health.jsonl` tail, groups by `model`, computes ok-rate per model, returns aggregate-and-per-model status.

**Files:**
- Modify: `dashboard/app.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
from dashboard.app import _compute_llm_health_summary


class TestComputeLlmHealthSummary:
    def _write_forecast_log(self, tmp_path, entries):
        path = tmp_path / "forecast_health.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        return path

    def test_no_file_returns_red_aggregate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_llm_health_summary()
        assert result["aggregate"]["status"] == "red"
        assert result["per_model"] == []

    def test_all_ok_returns_green(self, tmp_path, monkeypatch):
        entries = [
            {"ts": "2026-04-28T10:00:00+00:00", "model": "chronos", "ok": True, "ms": 50},
            {"ts": "2026-04-28T10:01:00+00:00", "model": "chronos", "ok": True, "ms": 45},
            {"ts": "2026-04-28T10:02:00+00:00", "model": "ministral", "ok": True, "ms": 800},
        ]
        self._write_forecast_log(tmp_path, entries)
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_llm_health_summary()
        assert result["aggregate"]["status"] == "green"
        assert len(result["per_model"]) == 2
        chronos = next(m for m in result["per_model"] if m["name"] == "chronos")
        assert chronos["ok_rate_pct"] == 100.0
        assert chronos["status"] == "green"

    def test_one_model_failing_lowers_aggregate(self, tmp_path, monkeypatch):
        entries = (
            [{"ts": "2026-04-28T10:00:00+00:00", "model": "chronos", "ok": True}] * 10 +
            [{"ts": "2026-04-28T10:00:00+00:00", "model": "qwen", "ok": False,
              "error": "timeout"}] * 10
        )
        self._write_forecast_log(tmp_path, entries)
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_llm_health_summary()
        qwen = next(m for m in result["per_model"] if m["name"] == "qwen")
        assert qwen["status"] == "red"
        assert result["aggregate"]["status"] in ("red", "orange")

    def test_per_model_includes_call_count(self, tmp_path, monkeypatch):
        entries = [
            {"ts": "2026-04-28T10:00:00+00:00", "model": "chronos", "ok": True}
        ] * 7
        self._write_forecast_log(tmp_path, entries)
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_llm_health_summary()
        chronos = next(m for m in result["per_model"] if m["name"] == "chronos")
        assert chronos["calls"] == 7
```

- [ ] **Step 2: Run — confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestComputeLlmHealthSummary -v
```

- [ ] **Step 3: Implement**

```python
def _compute_llm_health_summary(tail_size=200):
    """Return ops-status subsystem dict for LLMs.

    Reads forecast_health.jsonl tail, groups by model, computes ok-rate per model.
    Aggregate status is the worst individual model status.
    """
    log_path = DATA_DIR / "forecast_health.jsonl"
    try:
        entries = file_utils.load_jsonl_tail(log_path, max_entries=tail_size)
    except FileNotFoundError:
        return {"aggregate": {"status": "red", "summary": "no data", "age_s": None,
                              "details": {}}, "per_model": []}
    if not entries:
        return {"aggregate": {"status": "red", "summary": "no data", "age_s": None,
                              "details": {}}, "per_model": []}

    by_model = {}
    for entry in entries:
        model = entry.get("model")
        if not model:
            continue
        ok = entry.get("ok")
        bucket = by_model.setdefault(model, {"ok": 0, "fail": 0})
        if ok:
            bucket["ok"] += 1
        else:
            bucket["fail"] += 1

    per_model = []
    worst = "green"
    rank = {"green": 0, "orange": 1, "red": 2}
    for model, counts in sorted(by_model.items()):
        total = counts["ok"] + counts["fail"]
        rate = (counts["ok"] / total * 100) if total else 0
        color = _status_color(rate, OPS_THRESHOLDS["llm_individual"], "over")
        per_model.append({
            "name": model,
            "status": color,
            "ok_rate_pct": round(rate, 1),
            "calls": total,
        })
        if rank[color] > rank[worst]:
            worst = color

    n_failing = sum(1 for m in per_model if m["status"] != "green")
    summary = (f"all {len(per_model)} healthy" if n_failing == 0
               else f"{len(per_model) - n_failing}/{len(per_model)} healthy")
    return {
        "aggregate": {"status": worst, "summary": summary, "age_s": None,
                      "details": {"models_tracked": len(per_model)}},
        "per_model": per_model,
    }
```

- [ ] **Step 4: Run + pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestComputeLlmHealthSummary -v
```

Expected: 4 passed.

- [ ] **Step 5: Full suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -x --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py tests/test_dashboard.py
git commit -m "feat(dashboard): _compute_llm_health_summary helper

Reads forecast_health.jsonl tail; groups by model; aggregate status
is worst per-model status. Returns per-model details for drilldown.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `_compute_critical_error_window` helper

Counts entries in `data/critical_errors.jsonl` within last N seconds, grouped by level.

**Files:**
- Modify: `dashboard/app.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
from dashboard.app import _compute_critical_error_window


class TestComputeCriticalErrorWindow:
    def _write_log(self, tmp_path, entries):
        path = tmp_path / "critical_errors.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        return path

    def test_no_file_returns_green_zero_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_critical_error_window()
        assert result["status"] == "green"
        assert result["details"]["counts"] == {"critical": 0, "warning": 0, "info": 0}

    def test_no_recent_entries_returns_green(self, tmp_path, monkeypatch):
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        entries = [{"ts": old_ts, "level": "critical", "message": "old"}]
        self._write_log(tmp_path, entries)
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_critical_error_window(window_seconds=3600)
        assert result["status"] == "green"
        assert result["details"]["counts"]["critical"] == 0

    def test_recent_critical_returns_red(self, tmp_path, monkeypatch):
        recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        entries = [
            {"ts": recent_ts, "level": "critical", "message": "boom"},
            {"ts": recent_ts, "level": "info", "message": "fine"},
        ]
        self._write_log(tmp_path, entries)
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_critical_error_window()
        assert result["status"] == "red"
        assert result["details"]["counts"]["critical"] == 1

    def test_recent_warning_returns_orange(self, tmp_path, monkeypatch):
        recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        entries = [{"ts": recent_ts, "level": "warning"}]
        self._write_log(tmp_path, entries)
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_critical_error_window()
        assert result["status"] == "orange"

    def test_resolution_entries_excluded(self, tmp_path, monkeypatch):
        # Resolution entries document fixes, must not count as errors.
        recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        entries = [
            {"ts": recent_ts, "level": "info", "category": "resolution",
             "message": "fixed bug"}
        ]
        self._write_log(tmp_path, entries)
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_critical_error_window()
        assert result["details"]["counts"]["critical"] == 0
        assert result["details"]["counts"]["warning"] == 0
```

- [ ] **Step 2: Run — confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestComputeCriticalErrorWindow -v
```

- [ ] **Step 3: Implement**

```python
def _compute_critical_error_window(window_seconds=3600, tail_size=500):
    """Count critical_errors.jsonl entries in last `window_seconds`, grouped by level.

    Resolution entries (category=='resolution') are excluded — they document fixes,
    not errors.
    """
    log_path = DATA_DIR / "critical_errors.jsonl"
    counts = {"critical": 0, "warning": 0, "info": 0}
    try:
        entries = file_utils.load_jsonl_tail(log_path, max_entries=tail_size)
    except FileNotFoundError:
        return {"status": "green", "summary": "0 in last 1h", "age_s": None,
                "details": {"counts": counts, "window_s": window_seconds}}

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
    for entry in entries:
        if entry.get("category") == "resolution":
            continue
        ts_str = entry.get("ts")
        parsed = _parse_iso8601(ts_str)
        if parsed is None or parsed < cutoff:
            continue
        level = entry.get("level", "info")
        if level in counts:
            counts[level] += 1

    if counts["critical"] > 0:
        color = "red"
        summary = f"{counts['critical']} critical in last {window_seconds // 60}m"
    elif counts["warning"] > 0:
        color = "orange"
        summary = f"{counts['warning']} warning in last {window_seconds // 60}m"
    else:
        color = "green"
        summary = f"0 in last {window_seconds // 60}m"

    return {"status": color, "summary": summary, "age_s": None,
            "details": {"counts": counts, "window_s": window_seconds}}
```

- [ ] **Step 4: Run + pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestComputeCriticalErrorWindow -v
```

Expected: 5 passed.

- [ ] **Step 5: Full suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -x --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py tests/test_dashboard.py
git commit -m "feat(dashboard): _compute_critical_error_window helper

Counts critical_errors.jsonl entries in last N seconds grouped by level.
Resolution entries excluded (they document fixes, not failures).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `_compute_accuracy_7d` and `_compute_ticker_drilldown` helpers

7-day weighted accuracy rollup + per-ticker drilldown.

**Files:**
- Modify: `dashboard/app.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Locate the existing accuracy_stats function**

```bash
grep -nE "def per_ticker_accuracy|def consensus_accuracy|def signal_accuracy" /mnt/q/finance-analyzer/portfolio/accuracy_stats.py
```

Adjust the helper to consume whatever shape `per_ticker_accuracy()` actually returns.

- [ ] **Step 2: Write the failing test**

```python
from dashboard.app import _compute_accuracy_7d, _compute_ticker_drilldown
from unittest.mock import patch


class TestComputeAccuracy7d:
    def test_handles_missing_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_accuracy_7d()
        assert result["status"] == "red"
        assert result["details"]["weighted_pct"] is None

    def test_high_accuracy_returns_green(self, tmp_path, monkeypatch):
        cache = {
            "by_ticker": {
                "BTC-USD": {"7d": {"accuracy_pct": 65.0, "samples": 50}},
                "ETH-USD": {"7d": {"accuracy_pct": 58.0, "samples": 40}},
            }
        }
        (tmp_path / "accuracy_cache.json").write_text(json.dumps(cache))
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_accuracy_7d()
        assert result["status"] == "green"
        assert result["details"]["weighted_pct"] > 55

    def test_low_accuracy_returns_red(self, tmp_path, monkeypatch):
        cache = {
            "by_ticker": {
                "BTC-USD": {"7d": {"accuracy_pct": 38.0, "samples": 50}},
            }
        }
        (tmp_path / "accuracy_cache.json").write_text(json.dumps(cache))
        monkeypatch.setattr("dashboard.app.DATA_DIR", tmp_path)
        result = _compute_accuracy_7d()
        assert result["status"] == "red"


class TestComputeTickerDrilldown:
    def test_returns_list_with_one_per_ticker(self):
        fake = {
            "BTC-USD": {"accuracy_pct": 60.0, "samples": 100},
            "ETH-USD": {"accuracy_pct": 50.0, "samples": 80},
        }
        with patch("dashboard.app.accuracy_stats.per_ticker_accuracy",
                   return_value=fake):
            result = _compute_ticker_drilldown()
        assert len(result) == 2
        names = sorted(t["name"] for t in result)
        assert names == ["BTC-USD", "ETH-USD"]
        btc = next(t for t in result if t["name"] == "BTC-USD")
        assert btc["accuracy_pct"] == 60.0
        assert btc["status"] == "green"
```

- [ ] **Step 3: Run — confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestComputeAccuracy7d tests/test_dashboard.py::TestComputeTickerDrilldown -v
```

- [ ] **Step 4: Implement**

Verify import:
```bash
grep -nE "import accuracy_stats|from portfolio import accuracy_stats|from portfolio.accuracy_stats" dashboard/app.py | head -3
```

If absent, add `from portfolio import accuracy_stats` at top.

```python
def _compute_accuracy_7d():
    """Return ops-status subsystem dict for 7-day accuracy rollup."""
    cache_path = DATA_DIR / "accuracy_cache.json"
    if not cache_path.exists():
        return {"status": "red", "summary": "no accuracy cache", "age_s": None,
                "details": {"weighted_pct": None}}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "red", "summary": "cache unreadable", "age_s": None,
                "details": {"weighted_pct": None}}

    by_ticker = cache.get("by_ticker") or {}
    weighted_sum = 0.0
    weight_total = 0
    for ticker, horizons in by_ticker.items():
        h7 = horizons.get("7d") or {}
        pct = h7.get("accuracy_pct")
        samples = h7.get("samples", 0) or 0
        if pct is None or samples <= 0:
            continue
        weighted_sum += pct * samples
        weight_total += samples

    if weight_total == 0:
        return {"status": "red", "summary": "no 7d samples", "age_s": None,
                "details": {"weighted_pct": None}}

    weighted_pct = round(weighted_sum / weight_total, 2)
    color = _status_color(weighted_pct, OPS_THRESHOLDS["accuracy_7d"], "over")
    return {"status": color, "summary": f"{weighted_pct}%", "age_s": None,
            "details": {"weighted_pct": weighted_pct, "weight_total": weight_total}}


def _compute_ticker_drilldown():
    """Return list of {name, status, accuracy_pct} per Tier-1 ticker."""
    try:
        per_ticker = accuracy_stats.per_ticker_accuracy()
    except Exception as e:
        logger.warning("per_ticker_accuracy failed: %s", e)
        return []

    drilldown = []
    for ticker, data in sorted((per_ticker or {}).items()):
        if not isinstance(data, dict):
            continue
        pct = data.get("accuracy_pct")
        color = _status_color(pct, OPS_THRESHOLDS["ticker_accuracy"], "over")
        drilldown.append({
            "name": ticker,
            "status": color,
            "accuracy_pct": round(pct, 1) if pct is not None else None,
        })
    return drilldown
```

If `logger` isn't already defined in `dashboard/app.py`, add:
```python
import logging
logger = logging.getLogger(__name__)
```

- [ ] **Step 5: Run + pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestComputeAccuracy7d tests/test_dashboard.py::TestComputeTickerDrilldown -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py tests/test_dashboard.py
git commit -m "feat(dashboard): _compute_accuracy_7d + ticker drilldown helpers

7d weighted accuracy rollup from accuracy_cache.json (top-level status card)
plus per-ticker breakdown reusing accuracy_stats.per_ticker_accuracy()
(drilldown strip).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `/api/ops-status` endpoint

Compose the helpers into the rollup endpoint.

**Files:**
- Modify: `dashboard/app.py` — add new route after existing `/api/health` (around line 1130)
- Test: `tests/test_dashboard.py` — new `TestApiOpsStatus` class

- [ ] **Step 1: Write the failing test**

```python
class TestApiOpsStatus:
    """GET /api/ops-status — rollup endpoint feeding the ops board."""

    def test_requires_auth(self, client_with_token):
        client, _token = client_with_token
        resp = client.get("/api/ops-status")
        assert resp.status_code == 401

    def test_returns_subsystems_shape(self, client_with_token):
        client, token = client_with_token
        resp = client.get(f"/api/ops-status?token={token}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "subsystems" in data
        expected_keys = {
            "layer1", "layer2", "metals_loop", "data_feeds", "signals",
            "llms", "critical_errors", "outcome_backfill", "accuracy_7d"
        }
        assert set(data["subsystems"].keys()) >= expected_keys
        for name, sub in data["subsystems"].items():
            assert "status" in sub, f"{name} missing status"
            assert sub["status"] in ("green", "orange", "red"), f"{name} bad status {sub['status']}"
            assert "summary" in sub
            assert "details" in sub

    def test_returns_drilldowns_shape(self, client_with_token):
        client, token = client_with_token
        resp = client.get(f"/api/ops-status?token={token}")
        data = resp.get_json()
        assert "drilldowns" in data
        for k in ("signals", "llms", "tickers"):
            assert k in data["drilldowns"]
            assert isinstance(data["drilldowns"][k], list)

    def test_includes_timestamp(self, client_with_token):
        client, token = client_with_token
        resp = client.get(f"/api/ops-status?token={token}")
        data = resp.get_json()
        assert "timestamp" in data

    def test_helper_failure_isolated(self, client_with_token, monkeypatch):
        from dashboard import app as dashapp
        def boom():
            raise RuntimeError("simulated failure")
        monkeypatch.setattr(dashapp, "_compute_metals_loop_status", boom)

        client, token = client_with_token
        resp = client.get(f"/api/ops-status?token={token}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["subsystems"]["metals_loop"]["status"] == "red"
        assert data["subsystems"]["layer1"]["status"] in ("green", "orange", "red")
```

`client_with_token` fixture is used by other test classes — verify by greping. If absent,
copy the pattern from an existing class.

- [ ] **Step 2: Run — confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestApiOpsStatus -v
```

- [ ] **Step 3: Implement the endpoint**

In `dashboard/app.py`, find `api_health` and insert this after it:

```python
@app.route("/api/ops-status")
@require_auth
def api_ops_status():
    """Rollup status endpoint feeding the operations board UI."""

    def _safe(fn, fallback_subsystem_name):
        try:
            return fn()
        except Exception as e:
            logger.warning("ops-status helper %s failed: %s", fallback_subsystem_name, e)
            return {"status": "red", "summary": f"helper error: {e}",
                    "age_s": None, "details": {"error": str(e)}}

    def _safe_list(fn):
        try:
            return fn()
        except Exception as e:
            logger.warning("ops-status drilldown failed: %s", e)
            return []

    def _build():
        from portfolio import health as health_mod
        try:
            health = health_mod.get_health_summary()
        except Exception as e:
            logger.warning("get_health_summary failed: %s", e)
            health = {}

        # Layer 1
        l1_age = health.get("heartbeat_age_seconds")
        layer1 = {
            "status": _status_color(l1_age, OPS_THRESHOLDS["layer1_heartbeat"], "under"),
            "summary": (f"heartbeat {int(l1_age)}s ago" if l1_age is not None
                        else "no heartbeat"),
            "age_s": l1_age,
            "details": {
                "cycle_count": health.get("cycle_count"),
                "error_count": health.get("error_count"),
            },
        }

        # Layer 2
        l2_age = health.get("agent_silence_seconds")
        layer2 = {
            "status": _status_color(l2_age, OPS_THRESHOLDS["layer2_silence"], "under"),
            "summary": ("agent silent" if health.get("agent_silent")
                        else f"last invocation {int(l2_age or 0)}s ago"),
            "age_s": l2_age,
            "details": {"agent_silent": health.get("agent_silent")},
        }

        # Data feeds
        breakers = health.get("circuit_breakers") or {}
        statuses = list(breakers.values())
        if not statuses:
            data_feeds = {"status": "red", "summary": "no breaker data",
                          "age_s": None, "details": {}}
        else:
            opens = [b for b in statuses if str(b).lower() in ("open",)]
            half = [b for b in statuses if "half" in str(b).lower()]
            if opens:
                color = "red"
                summary = f"{len(opens)} feed(s) open"
            elif half:
                color = "orange"
                summary = f"{len(half)} feed(s) half-open"
            else:
                color = "green"
                summary = "all feeds closed (healthy)"
            data_feeds = {"status": color, "summary": summary, "age_s": None,
                          "details": {"breakers": breakers}}

        # Signals aggregate
        sig_health = health.get("signal_health") or {}
        if not sig_health:
            signals = {"status": "red", "summary": "no signal health data",
                       "age_s": None, "details": {}}
            sig_drilldown = []
        else:
            rates = [s["success_rate_pct"] for s in sig_health.values()
                     if s.get("success_rate_pct") is not None]
            avg = sum(rates) / len(rates) if rates else 0
            n_red = sum(1 for r in rates
                        if r < OPS_THRESHOLDS["signal_individual"]["orange_over_pct"])
            n_orange = sum(1 for r in rates
                           if OPS_THRESHOLDS["signal_individual"]["orange_over_pct"] <= r
                           < OPS_THRESHOLDS["signal_individual"]["green_over_pct"])
            color = _status_color(avg, OPS_THRESHOLDS["signals_aggregate"], "over")
            signals = {
                "status": color,
                "summary": f"{len(rates) - n_red - n_orange}/{len(rates)} healthy",
                "age_s": None,
                "details": {"avg_pct": round(avg, 1), "n_red": n_red,
                            "n_orange": n_orange, "n_total": len(rates)},
            }
            sig_drilldown = [
                {
                    "name": name,
                    "status": _status_color(s["success_rate_pct"],
                                            OPS_THRESHOLDS["signal_individual"], "over"),
                    "rate_pct": s["success_rate_pct"],
                    "calls": s.get("total_calls", 0),
                }
                for name, s in sorted(sig_health.items())
            ]

        metals_loop = _safe(_compute_metals_loop_status, "metals_loop")
        llm = _safe(_compute_llm_health_summary, "llms")
        critical_errors = _safe(_compute_critical_error_window, "critical_errors")

        try:
            outcome = health_mod.check_outcome_staleness()
            age_h = outcome.get("newest_outcome_age_hours", float("inf"))
            outcome_color = _status_color(age_h, OPS_THRESHOLDS["outcome_max_age_h"], "under")
            outcome_subsys = {
                "status": outcome_color,
                "summary": (f"newest outcome {age_h:.1f}h ago"
                            if age_h != float("inf") else "no outcomes"),
                "age_s": age_h * 3600 if age_h != float("inf") else None,
                "details": outcome,
            }
        except Exception as e:
            outcome_subsys = {"status": "red", "summary": f"helper error: {e}",
                              "age_s": None, "details": {"error": str(e)}}

        accuracy_7d = _safe(_compute_accuracy_7d, "accuracy_7d")
        ticker_drilldown = _safe_list(_compute_ticker_drilldown)

        # llm["aggregate"] may be a dict (ok) or a string fallback if _safe wrapped it
        llm_aggregate = llm["aggregate"] if isinstance(llm, dict) and "aggregate" in llm else llm
        llm_per_model = llm.get("per_model", []) if isinstance(llm, dict) else []

        return {
            "subsystems": {
                "layer1": layer1,
                "layer2": layer2,
                "metals_loop": metals_loop,
                "data_feeds": data_feeds,
                "signals": signals,
                "llms": llm_aggregate,
                "critical_errors": critical_errors,
                "outcome_backfill": outcome_subsys,
                "accuracy_7d": accuracy_7d,
            },
            "drilldowns": {
                "signals": sig_drilldown,
                "llms": llm_per_model,
                "tickers": ticker_drilldown,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return jsonify(_cached_read("ops_status", 5, _build))
```

Verify imports — `from flask import jsonify`, `from datetime import datetime, timezone` are
present (likely already there).

- [ ] **Step 4: Run + pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestApiOpsStatus -v
```

Expected: 5 passed.

- [ ] **Step 5: Full suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -x --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py tests/test_dashboard.py
git commit -m "feat(dashboard): add /api/ops-status rollup endpoint

Single endpoint feeding the new ops board UI. Composes get_health_summary,
_compute_metals_loop_status, _compute_llm_health_summary,
_compute_critical_error_window, _compute_accuracy_7d, check_outcome_staleness.
Helper failures isolated — endpoint never returns 5xx; broken subsystem
shows red, others continue.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `/api/logs/<source>` endpoint

Generic log tail with level / limit / since filtering.

**Files:**
- Modify: `dashboard/app.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestApiLogs:
    """GET /api/logs/<source> — generic log tail."""

    def test_requires_auth(self, client_with_token):
        client, _ = client_with_token
        assert client.get("/api/logs/critical-errors").status_code == 401

    def test_unknown_source_returns_404(self, client_with_token):
        client, token = client_with_token
        resp = client.get(f"/api/logs/foo-bar?token={token}")
        assert resp.status_code == 404

    def test_jsonl_source_returns_entries(self, client_with_token, tmp_path, monkeypatch):
        from dashboard import app as dashapp
        monkeypatch.setattr(dashapp, "DATA_DIR", tmp_path)
        log = tmp_path / "critical_errors.jsonl"
        log.write_text(
            '{"ts":"2026-04-28T10:00:00+00:00","level":"warning","message":"x"}\n'
            '{"ts":"2026-04-28T10:01:00+00:00","level":"critical","message":"y"}\n'
        )
        client, token = client_with_token
        resp = client.get(f"/api/logs/critical-errors?token={token}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["source"] == "critical-errors"
        assert len(data["entries"]) == 2
        assert data["entries"][-1]["level"] == "critical"

    def test_jsonl_source_filters_by_level(self, client_with_token, tmp_path, monkeypatch):
        from dashboard import app as dashapp
        monkeypatch.setattr(dashapp, "DATA_DIR", tmp_path)
        log = tmp_path / "critical_errors.jsonl"
        log.write_text(
            '{"ts":"2026-04-28T10:00:00+00:00","level":"warning"}\n'
            '{"ts":"2026-04-28T10:01:00+00:00","level":"critical"}\n'
            '{"ts":"2026-04-28T10:02:00+00:00","level":"info"}\n'
        )
        client, token = client_with_token
        resp = client.get(f"/api/logs/critical-errors?token={token}&level=critical")
        data = resp.get_json()
        assert all(e["level"] == "critical" for e in data["entries"])
        assert len(data["entries"]) == 1

    def test_text_source_returns_lines(self, client_with_token, tmp_path, monkeypatch):
        from dashboard import app as dashapp
        monkeypatch.setattr(dashapp, "DATA_DIR", tmp_path)
        (tmp_path / "loop_out.txt").write_text("line1\nline2\nline3\n")
        client, token = client_with_token
        resp = client.get(f"/api/logs/loop-stdout?token={token}")
        data = resp.get_json()
        assert "lines" in data
        assert data["lines"][-1] == "line3"

    def test_limit_param_clamped_to_max(self, client_with_token, tmp_path, monkeypatch):
        from dashboard import app as dashapp
        monkeypatch.setattr(dashapp, "DATA_DIR", tmp_path)
        big = "\n".join(f"line {i}" for i in range(1000))
        (tmp_path / "loop_out.txt").write_text(big)
        client, token = client_with_token
        resp = client.get(f"/api/logs/loop-stdout?token={token}&limit=99999")
        data = resp.get_json()
        assert len(data["lines"]) <= 500
```

- [ ] **Step 2: Run — confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestApiLogs -v
```

- [ ] **Step 3: Implement**

```python
LOG_SOURCES = {
    "critical-errors":    {"path": "critical_errors.jsonl",   "format": "jsonl"},
    "agent-log":          {"path": "agent.log",                "format": "jsonl"},
    "loop-stdout":        {"path": "loop_out.txt",             "format": "text"},
    "metals-loop-stdout": {"path": "metals_loop_out.txt",      "format": "text"},
}
LOGS_MAX_LIMIT = 500
LOGS_DEFAULT_LIMIT = 50


@app.route("/api/logs/<source>")
@require_auth
def api_logs(source):
    """Tail a known log source. Supports level (jsonl only), limit, since filters."""
    spec = LOG_SOURCES.get(source)
    if spec is None:
        return jsonify({"error": "unknown_source", "valid_sources": list(LOG_SOURCES)}), 404

    path = DATA_DIR / spec["path"]
    limit = _parse_limit_arg("limit", LOGS_DEFAULT_LIMIT, LOGS_MAX_LIMIT)

    if spec["format"] == "jsonl":
        levels_arg = request.args.get("level")
        levels_filter = (set(levels_arg.split(",")) if levels_arg else None)
        since_arg = request.args.get("since")
        since_dt = _parse_iso8601(since_arg) if since_arg else None

        raw = file_utils.load_jsonl_tail(path, max_entries=limit * 4) if path.exists() else []
        filtered = []
        for entry in raw:
            if levels_filter and entry.get("level") not in levels_filter:
                continue
            if since_dt:
                ts = _parse_iso8601(entry.get("ts"))
                if ts is None or ts < since_dt:
                    continue
            filtered.append(entry)
        return jsonify({
            "source": source,
            "entries": filtered[-limit:],
            "truncated": len(raw) > len(filtered),
        })

    if not path.exists():
        return jsonify({"source": source, "lines": [], "truncated": False})
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    truncated = len(lines) > limit
    return jsonify({
        "source": source,
        "lines": lines[-limit:],
        "truncated": truncated,
    })
```

If `request` isn't imported at top, add: `from flask import request`.

- [ ] **Step 4: Run + pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestApiLogs -v
```

Expected: 6 passed.

- [ ] **Step 5: Full suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -x --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py tests/test_dashboard.py
git commit -m "feat(dashboard): add /api/logs/<source> generic log tail endpoint

Sources: critical-errors, agent-log, loop-stdout, metals-loop-stdout.
JSONL sources support level/since filtering; both formats support limit
(default 50, max 500).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Extend `/api/health` with `metals_loop` field

Backward-compatible additive change.

**Files:**
- Modify: `dashboard/app.py` — `api_health()` around line 1118
- Test: `tests/test_dashboard.py` — extend existing `TestApiHealth` class

- [ ] **Step 1: Find current `api_health` implementation**

```bash
grep -n "def api_health\b" dashboard/app.py
sed -n '1116,1135p' dashboard/app.py
```

- [ ] **Step 2: Add a test**

Append to `class TestApiHealth` in `tests/test_dashboard.py`:

```python
    def test_includes_metals_loop_field(self, client_with_token):
        client, token = client_with_token
        resp = client.get(f"/api/health?token={token}")
        data = resp.get_json()
        assert "metals_loop" in data
        assert "status" in data["metals_loop"]
        assert data["metals_loop"]["status"] in ("green", "orange", "red")
```

- [ ] **Step 3: Run — confirm fail**

```bash
.venv/Scripts/python.exe -m pytest "tests/test_dashboard.py::TestApiHealth::test_includes_metals_loop_field" -v
```

- [ ] **Step 4: Add the field**

Use Edit to add a single line to `api_health()`. Find the existing return shape and add
`summary["metals_loop"] = _compute_metals_loop_status()` immediately before the `return`.

- [ ] **Step 5: Run + pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py::TestApiHealth -v
```

Expected: existing tests still pass + new test passes.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py tests/test_dashboard.py
git commit -m "feat(dashboard): add metals_loop field to /api/health

Additive, backward compatible. Existing consumers can ignore.
Frontend ops board uses this for one-roundtrip subsystem display.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Update `dashboard/export_static.py`

Add the new endpoints to the static-export list so GitHub Pages keeps working.

**Files:**
- Modify: `dashboard/export_static.py:25` (the `ENDPOINTS` constant)
- Test: `tests/test_dashboard_export_static.py`

- [ ] **Step 1: Read existing test pattern**

```bash
cat tests/test_dashboard_export_static.py
```

- [ ] **Step 2: Add a test**

Append to `tests/test_dashboard_export_static.py`:

```python
def test_export_includes_ops_status_and_log_files(tmp_path):
    from dashboard.export_static import export_all
    export_all(out_dir=tmp_path)
    expected_new = {
        "ops-status.json",
        "logs-critical-errors.json",
        "logs-agent-log.json",
        "logs-loop-stdout.json",
        "logs-metals-loop-stdout.json",
    }
    written = set(p.name for p in tmp_path.iterdir())
    missing = expected_new - written
    assert not missing, f"Static export missing new files: {missing}"
```

- [ ] **Step 3: Run — confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard_export_static.py::test_export_includes_ops_status_and_log_files -v
```

- [ ] **Step 4: Add to ENDPOINTS list**

Edit `dashboard/export_static.py`. Find the closing `]` of the `ENDPOINTS` list (after the
`("/api/health", "health.json"),` line) and insert before it:

```python
    ("/api/ops-status", "ops-status.json"),
    ("/api/logs/critical-errors", "logs-critical-errors.json"),
    ("/api/logs/agent-log", "logs-agent-log.json"),
    ("/api/logs/loop-stdout", "logs-loop-stdout.json"),
    ("/api/logs/metals-loop-stdout", "logs-metals-loop-stdout.json"),
```

- [ ] **Step 5: Run + pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard_export_static.py -v
```

Expected: all pass.

- [ ] **Step 6: Smoke test the actual export**

```bash
cd /mnt/q/finance-analyzer/.worktrees/dashboard-ops-board
.venv/Scripts/python.exe dashboard/export_static.py
ls dashboard/static/api-data/ops-status.json
ls dashboard/static/api-data/logs-critical-errors.json
```

Both files should exist.

- [ ] **Step 7: Commit**

```bash
git add dashboard/export_static.py tests/test_dashboard_export_static.py
git commit -m "feat(dashboard): export ops-status + log endpoints to static files

Keeps GitHub Pages mode working — every new live API endpoint has a
corresponding pre-baked JSON snapshot.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Frontend CSS — Ops board styles

Add `.ops-*` styles to the existing `<style>` block in `dashboard/static/index.html`.
No test for CSS — verified visually with chrome-devtools-mcp later.

**Files:**
- Modify: `dashboard/static/index.html`

- [ ] **Step 1: Find the closing `</style>` tag**

```bash
grep -n "</style>" dashboard/static/index.html | head -3
```

- [ ] **Step 2: Insert ops-* CSS block**

Use Edit to insert this CSS block immediately before the closing `</style>` tag. Reuse
existing CSS-variable names where present (search for `:root` block).

```css
/* ============ OPS BOARD ============ */
:root {
  --ops-green: #4caf50;
  --ops-orange: #ff9800;
  --ops-red: #e53935;
  --ops-card-bg: var(--bg-card, #1c1f24);
  --ops-card-border: var(--border, #2a2e36);
  --ops-text-dim: var(--txd, #888);
  --ops-text: var(--tx, #ddd);
}

.ops-tab { padding: 16px; max-width: 1400px; margin: 0 auto; }

.ops-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}
.ops-card {
  background: var(--ops-card-bg);
  border: 1px solid var(--ops-card-border);
  border-radius: 8px;
  padding: 16px;
  cursor: pointer;
  transition: border-color 0.15s ease;
  display: flex;
  align-items: center;
  gap: 12px;
}
.ops-card:hover { border-color: var(--ops-text); }
.ops-dot {
  width: 24px; height: 24px;
  border-radius: 50%;
  flex-shrink: 0;
  box-shadow: 0 0 8px currentColor;
}
.ops-dot--green  { background: var(--ops-green); color: var(--ops-green); }
.ops-dot--orange { background: var(--ops-orange); color: var(--ops-orange); }
.ops-dot--red    { background: var(--ops-red); color: var(--ops-red); animation: ops-pulse 1.5s infinite; }
@keyframes ops-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
.ops-card-body { flex: 1; min-width: 0; }
.ops-card-name { font-size: 14px; font-weight: 600; color: var(--ops-text); margin-bottom: 2px; }
.ops-card-summary { font-size: 12px; color: var(--ops-text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.ops-strip { display: flex; flex-wrap: wrap; gap: 4px; padding: 12px 16px; background: var(--ops-card-bg); border-radius: 6px; margin-bottom: 8px; align-items: center; }
.ops-strip-label { font-size: 11px; color: var(--ops-text-dim); margin-right: 12px; min-width: 80px; }
.ops-strip-dots { display: flex; flex-wrap: wrap; gap: 4px; }
.ops-mini-dot { width: 10px; height: 10px; border-radius: 50%; cursor: help; }
.ops-mini-dot--green  { background: var(--ops-green); }
.ops-mini-dot--orange { background: var(--ops-orange); }
.ops-mini-dot--red    { background: var(--ops-red); }

.ops-log {
  margin-top: 24px;
  background: var(--ops-card-bg);
  border: 1px solid var(--ops-card-border);
  border-radius: 8px;
  padding: 16px;
}
.ops-log-tabs { display: flex; gap: 8px; margin-bottom: 12px; border-bottom: 1px solid var(--ops-card-border); padding-bottom: 8px; }
.ops-log-tab { padding: 6px 12px; cursor: pointer; color: var(--ops-text-dim); font-size: 13px; border-radius: 4px; }
.ops-log-tab--active { background: var(--ops-card-border); color: var(--ops-text); }
.ops-log-controls { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
.ops-log-controls input, .ops-log-controls select, .ops-log-controls button {
  background: var(--bg, #0d0f12); border: 1px solid var(--ops-card-border);
  color: var(--ops-text); padding: 4px 8px; border-radius: 4px; font-size: 12px;
}
.ops-log-controls button { cursor: pointer; }
.ops-log-list { font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; max-height: 400px; overflow-y: auto; }
.ops-log-entry { padding: 4px 8px; border-bottom: 1px solid var(--ops-card-border); cursor: pointer; }
.ops-log-entry:hover { background: rgba(255,255,255,0.04); }
.ops-log-entry--critical { border-left: 3px solid var(--ops-red); }
.ops-log-entry--warning  { border-left: 3px solid var(--ops-orange); }
.ops-log-entry--info     { border-left: 3px solid transparent; }
.ops-log-entry-detail { display: none; padding: 8px; background: rgba(0,0,0,0.3); white-space: pre-wrap; word-wrap: break-word; }
.ops-log-entry--expanded .ops-log-entry-detail { display: block; }
```

- [ ] **Step 3: Sanity-check HTML still parses**

```bash
.venv/Scripts/python.exe -c "from html.parser import HTMLParser; HTMLParser().feed(open('dashboard/static/index.html').read())"
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/static/index.html
git commit -m "feat(dashboard): add ops-board CSS

Status grid, traffic-light dots, drilldown strips, log inspector styles.
All classes prefixed .ops-* to avoid collision with existing CSS.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Frontend HTML — Ops board structure + smoke test

Replace the empty `<div id="healthC">Loading...</div>` placeholder with the full 3-section
HTML shell.

**Files:**
- Modify: `dashboard/static/index.html`
- Test: `tests/test_dashboard_frontend.py`

- [ ] **Step 1: Add smoke test**

Append to `tests/test_dashboard_frontend.py`:

```python
def test_ops_board_html_present(client_with_token):
    """The ops board structural markers must be present in the served HTML."""
    client, token = client_with_token
    resp = client.get(f"/?token={token}")
    body = resp.data.decode("utf-8")
    assert 'id="opsGrid"' in body or 'class="ops-grid"' in body
    assert 'id="opsSignalsStrip"' in body
    assert 'id="opsLlmsStrip"' in body
    assert 'id="opsTickersStrip"' in body
    assert 'id="opsLogList"' in body
```

If `client_with_token` fixture isn't already imported, copy the standard pattern from
`tests/test_dashboard.py` or import from a conftest.

- [ ] **Step 2: Run — confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard_frontend.py -v
```

- [ ] **Step 3: Locate the placeholder**

```bash
grep -n 'id="healthC"' dashboard/static/index.html
```

- [ ] **Step 4: Replace placeholder with full structure**

Use Edit to replace the matched line(s). Old:
```html
<div id="healthC">Loading...</div>
```

New (this is pure structural HTML — no interpolation, safe to embed verbatim):
```html
<div id="healthC" class="ops-tab">
  <div id="opsGrid" class="ops-grid"></div>

  <div id="opsSignalsStrip" class="ops-strip">
    <span class="ops-strip-label">Signals</span>
    <div class="ops-strip-dots"></div>
  </div>
  <div id="opsLlmsStrip" class="ops-strip">
    <span class="ops-strip-label">LLMs</span>
    <div class="ops-strip-dots"></div>
  </div>
  <div id="opsTickersStrip" class="ops-strip">
    <span class="ops-strip-label">Tickers</span>
    <div class="ops-strip-dots"></div>
  </div>

  <div class="ops-log">
    <div class="ops-log-tabs">
      <div class="ops-log-tab ops-log-tab--active" data-source="critical-errors">Critical Errors</div>
      <div class="ops-log-tab" data-source="agent-log">System Log</div>
      <div class="ops-log-tab" data-source="loop-stdout">Main Loop stdout</div>
      <div class="ops-log-tab" data-source="metals-loop-stdout">Metals Loop stdout</div>
    </div>
    <div class="ops-log-controls">
      <input id="opsLogSearch" type="text" placeholder="Filter..." />
      <select id="opsLogLevel">
        <option value="">All levels</option>
        <option value="critical">Critical</option>
        <option value="warning">Warning</option>
        <option value="info">Info</option>
      </select>
      <button id="opsLogRefresh">Refresh</button>
    </div>
    <div id="opsLogList" class="ops-log-list">Loading...</div>
  </div>
</div>
```

- [ ] **Step 5: Run + pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_dashboard_frontend.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add dashboard/static/index.html tests/test_dashboard_frontend.py
git commit -m "feat(dashboard): replace empty Health placeholder with ops board HTML

Three sections: status grid, drilldown strips, log inspector.
JS to populate them lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Frontend JS — `OpsBoard` namespace (XSS-safe rendering)

Add the JS that fetches `/api/ops-status` and renders the status grid + drilldown strips.
**All rendering uses safe DOM methods** (`createElement` + `textContent` + `appendChild`).
No `innerHTML` with interpolated content.

**Files:**
- Modify: `dashboard/static/index.html` — add new JS in the existing `<script>` block

- [ ] **Step 1: Locate the closing `</script>` tag**

```bash
grep -n "</script>" dashboard/static/index.html | tail -3
```

- [ ] **Step 2: Insert OpsBoard JS**

Insert before the matched closing `</script>`:

```javascript
/* ============ OpsBoard ============ */
const OpsBoard = (() => {
  const REFRESH_MS = 5000;
  let pollTimer = null;
  let currentLogSource = "critical-errors";

  // Cards: each entry produces one status card. targetTab is the existing dashboard tab
  // to switch to on click (null = no link).
  const CARDS = [
    { key: "layer1",          label: "MainLoop (Layer 1)",   targetTab: null },
    { key: "layer2",          label: "Claude Agent",         targetTab: "decisions" },
    { key: "metals_loop",     label: "Metals Loop",          targetTab: "metals" },
    { key: "data_feeds",      label: "Data Feeds",           targetTab: null },
    { key: "signals",         label: "Signals",              targetTab: "signal-heatmap" },
    { key: "llms",            label: "LLMs",                 targetTab: null },
    { key: "critical_errors", label: "Critical Errors (1h)", targetTab: null },
    { key: "outcome_backfill",label: "Outcome Backfill",     targetTab: null },
    { key: "accuracy_7d",     label: "Accuracy 7d",          targetTab: "accuracy" },
  ];

  function init() {
    bindLogTabs();
    bindLogControls();
    refresh();
  }

  function start() {
    if (pollTimer) return;
    refresh();
    pollTimer = setInterval(refresh, REFRESH_MS);
  }
  function stop() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  async function refresh() {
    try {
      const data = await fetchJsonSafe("/api/ops-status");
      renderCards(data.subsystems || {});
      renderDrilldowns(data.drilldowns || {});
    } catch (e) {
      console.warn("[OpsBoard] /api/ops-status failed, trying static fallback", e);
      try {
        const fallback = await (await fetch("/static/api-data/ops-status.json")).json();
        renderCards(fallback.subsystems || {});
        renderDrilldowns(fallback.drilldowns || {});
      } catch (e2) {
        const grid = document.getElementById("opsGrid");
        if (grid) {
          grid.replaceChildren();
          const errDiv = document.createElement("div");
          errDiv.style.color = "var(--ops-red)";
          errDiv.style.padding = "16px";
          errDiv.textContent = "Ops status unavailable";
          grid.appendChild(errDiv);
        }
      }
    }
    refreshLogs();
  }

  // Build a status card using DOM API only — no innerHTML interpolation.
  function buildCard(card, sub) {
    const div = document.createElement("div");
    div.className = "ops-card";
    if (card.targetTab) div.dataset.targetTab = card.targetTab;

    const dot = document.createElement("div");
    dot.className = `ops-dot ops-dot--${sub.status || "red"}`;
    div.appendChild(dot);

    const body = document.createElement("div");
    body.className = "ops-card-body";

    const name = document.createElement("div");
    name.className = "ops-card-name";
    name.textContent = card.label;
    body.appendChild(name);

    const summary = document.createElement("div");
    summary.className = "ops-card-summary";
    summary.textContent = sub.summary || "";
    if (sub.summary) summary.title = sub.summary;
    body.appendChild(summary);

    div.appendChild(body);
    return div;
  }

  function renderCards(subsystems) {
    const grid = document.getElementById("opsGrid");
    if (!grid) return;
    grid.replaceChildren();
    for (const card of CARDS) {
      const sub = subsystems[card.key] || { status: "red", summary: "no data" };
      const el = buildCard(card, sub);
      if (card.targetTab) {
        el.addEventListener("click", () => {
          if (typeof switchTab === "function") switchTab(card.targetTab);
        });
      }
      grid.appendChild(el);
    }
  }

  function renderDrilldowns(drilldowns) {
    renderStrip("opsSignalsStrip", drilldowns.signals || [],
      d => `${d.name}: ${d.rate_pct ?? "—"}% (n=${d.calls ?? 0})`);
    renderStrip("opsLlmsStrip", drilldowns.llms || [],
      d => `${d.name}: ${d.ok_rate_pct ?? "—"}% (n=${d.calls ?? 0})`);
    renderStrip("opsTickersStrip", drilldowns.tickers || [],
      d => `${d.name}: ${d.accuracy_pct ?? "—"}%`);
  }

  function renderStrip(stripId, items, tooltipFn) {
    const container = document.querySelector(`#${stripId} .ops-strip-dots`);
    if (!container) return;
    container.replaceChildren();
    if (!items || items.length === 0) {
      const empty = document.createElement("span");
      empty.style.color = "var(--ops-text-dim)";
      empty.style.fontSize = "11px";
      empty.textContent = "no data";
      container.appendChild(empty);
      return;
    }
    for (const d of items) {
      const dot = document.createElement("div");
      dot.className = `ops-mini-dot ops-mini-dot--${d.status || "red"}`;
      dot.title = tooltipFn(d);  // .title is safe — set as attribute, not parsed as HTML
      container.appendChild(dot);
    }
  }

  /* ----- Log inspector (DOM-safe) ----- */
  function bindLogTabs() {
    document.querySelectorAll(".ops-log-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".ops-log-tab").forEach(t => t.classList.remove("ops-log-tab--active"));
        tab.classList.add("ops-log-tab--active");
        currentLogSource = tab.getAttribute("data-source");
        refreshLogs();
      });
    });
  }
  function bindLogControls() {
    const refreshBtn = document.getElementById("opsLogRefresh");
    const search = document.getElementById("opsLogSearch");
    const level = document.getElementById("opsLogLevel");
    if (refreshBtn) refreshBtn.addEventListener("click", refreshLogs);
    if (search) search.addEventListener("input", refreshLogs);
    if (level) level.addEventListener("change", refreshLogs);
  }

  async function refreshLogs() {
    const levelEl = document.getElementById("opsLogLevel");
    const searchEl = document.getElementById("opsLogSearch");
    const level = levelEl ? levelEl.value : "";
    const search = (searchEl ? searchEl.value : "").toLowerCase();
    let url = `/api/logs/${encodeURIComponent(currentLogSource)}?limit=100`;
    if (level) url += `&level=${encodeURIComponent(level)}`;
    let payload;
    try {
      payload = await fetchJsonSafe(url);
    } catch (e) {
      try {
        payload = await (await fetch(`/static/api-data/logs-${currentLogSource}.json`)).json();
      } catch (e2) {
        const list = document.getElementById("opsLogList");
        if (list) {
          list.replaceChildren();
          list.textContent = "Logs unavailable";
        }
        return;
      }
    }
    renderLogList(payload, search);
  }

  function buildJsonlEntry(entry, search) {
    const lvl = (entry.level || "info").toLowerCase();
    const json = JSON.stringify(entry, null, 2);
    if (search && !json.toLowerCase().includes(search)) return null;

    const div = document.createElement("div");
    div.className = `ops-log-entry ops-log-entry--${lvl}`;
    div.addEventListener("click", () => div.classList.toggle("ops-log-entry--expanded"));

    const tsSpan = document.createElement("span");
    tsSpan.style.color = "var(--ops-text-dim)";
    tsSpan.textContent = entry.ts || "";
    div.appendChild(tsSpan);

    const lvlSpan = document.createElement("span");
    lvlSpan.style.marginLeft = "8px";
    lvlSpan.style.textTransform = "uppercase";
    lvlSpan.style.fontSize = "10px";
    lvlSpan.textContent = lvl;
    div.appendChild(lvlSpan);

    const msgSpan = document.createElement("span");
    msgSpan.style.marginLeft = "8px";
    msgSpan.textContent = entry.message || entry.category || "";
    div.appendChild(msgSpan);

    const detail = document.createElement("pre");
    detail.className = "ops-log-entry-detail";
    detail.textContent = json;
    div.appendChild(detail);

    return div;
  }

  function buildTextEntry(line, search) {
    if (search && !line.toLowerCase().includes(search)) return null;
    let lvl = "info";
    if (/error|critical|exception|traceback/i.test(line)) lvl = "critical";
    else if (/warn/i.test(line)) lvl = "warning";

    const div = document.createElement("div");
    div.className = `ops-log-entry ops-log-entry--${lvl}`;
    div.textContent = line;
    return div;
  }

  function renderLogList(payload, search) {
    const list = document.getElementById("opsLogList");
    if (!list) return;
    list.replaceChildren();

    let any = false;
    if (Array.isArray(payload.entries)) {
      for (const entry of payload.entries.slice().reverse()) {
        const el = buildJsonlEntry(entry, search);
        if (el) { list.appendChild(el); any = true; }
      }
    } else if (Array.isArray(payload.lines)) {
      for (const line of payload.lines.slice().reverse()) {
        const el = buildTextEntry(line, search);
        if (el) { list.appendChild(el); any = true; }
      }
    }
    if (!any) {
      const empty = document.createElement("div");
      empty.style.color = "var(--ops-text-dim)";
      empty.style.padding = "12px";
      empty.textContent = "no entries";
      list.appendChild(empty);
    }
  }

  // Minimal safe fetch helper. Falls through to existing fetchJson if defined; else uses
  // built-in fetch() with status check.
  async function fetchJsonSafe(url) {
    if (typeof fetchJson === "function") return fetchJson(url);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  return { init, start, stop, refresh };
})();

// Wire up: when DOM ready, start polling. (Tab manager integration is verified manually.)
document.addEventListener("DOMContentLoaded", () => {
  if (typeof OpsBoard !== "undefined" && OpsBoard.init) {
    OpsBoard.init();
    OpsBoard.start();
  }
});
```

- [ ] **Step 3: HTML still parses**

```bash
.venv/Scripts/python.exe -c "from html.parser import HTMLParser; HTMLParser().feed(open('dashboard/static/index.html').read())"
```

- [ ] **Step 4: Manual verification using chrome-devtools-mcp**

Start dashboard:
```bash
cd /mnt/q/finance-analyzer/.worktrees/dashboard-ops-board
.venv/Scripts/python.exe dashboard/app.py &
sleep 3
```

Get the dashboard token (per CLAUDE.md feedback: NEVER cat config.json, use grep):
```bash
grep -E "^\s*\"dashboard_token\"" config.json | head -1
```

Then invoke the chrome-devtools-mcp skill:
```
Use chrome-devtools-mcp to:
1. Navigate to http://localhost:5055/?token=<TOKEN>
2. Take a screenshot of the Ops tab
3. Open the console and report any errors
4. Verify GET /api/ops-status returns 200 with { subsystems, drilldowns, timestamp }
```

Stop the dashboard:
```bash
kill %1 2>/dev/null
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/static/index.html
git commit -m "feat(dashboard): add OpsBoard JS (XSS-safe DOM rendering)

Fetches /api/ops-status every 5s; renders status grid, drilldown strips,
log inspector using only createElement + textContent + appendChild
(no innerHTML interpolation). Falls back to /static/api-data/ snapshots
when live API is unreachable. Cross-tab links from cards to existing tabs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Make Ops the default landing tab

**Files:**
- Modify: `dashboard/static/index.html`

- [ ] **Step 1: Find the current default-tab init**

```bash
grep -nE "default.*tab|switchTab\(|tabs\[0\]|active.*tab|accuracy.*default" dashboard/static/index.html | head -10
```

The previous commit `feat: accuracy tab as default + metals loop accuracy view` set
Accuracy as default. We're flipping it.

- [ ] **Step 2: Edit the default**

Use Edit to flip the default to whatever the existing tab id is for the `#healthC` pane
(likely `health` or `ops`). Typical pattern:

```javascript
// Old:
switchTab("accuracy");
// New:
switchTab("health");
```

- [ ] **Step 3: Verify**

Reload the dashboard at `http://localhost:5055/?token=...`. The Ops tab is active on
first load.

- [ ] **Step 4: Commit**

```bash
git add dashboard/static/index.html
git commit -m "feat(dashboard): make ops board the default landing tab

Replaces Accuracy tab as the default; existing tabs remain accessible.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Final verification + merge

- [ ] **Step 1: Full test suite**

```bash
cd /mnt/q/finance-analyzer/.worktrees/dashboard-ops-board
.venv/Scripts/python.exe -m pytest tests/test_dashboard.py tests/test_dashboard_export_static.py tests/test_dashboard_frontend.py -v --tb=short
```

Expected: all existing 97 tests pass + ~30 new tests pass. No regressions.

- [ ] **Step 2: Live verification with chrome-devtools-mcp**

```bash
.venv/Scripts/python.exe dashboard/app.py &
DASH_PID=$!
sleep 3
```

Run `chrome-devtools-mcp:chrome-devtools` skill:
- Navigate to `http://localhost:5055/?token=<token>`
- Take screenshot
- Console error count = 0
- Verify all 9 cards rendered
- Click a card with `data-target-tab` — confirm the relevant existing tab activates

Run `chrome-devtools-mcp:a11y-debugging` skill:
- Lighthouse a11y score on the Ops tab ≥ 90

```bash
kill $DASH_PID
```

- [ ] **Step 3: Manual flip test**

```bash
cmd.exe /c "schtasks /end /tn PF-MetalsLoop"
```

Wait 5 minutes. Refresh the dashboard. The "Metals Loop" card should be red (or orange
between 2 and 5 minutes).

```bash
cmd.exe /c "schtasks /run /tn PF-MetalsLoop"
```

Within 1–2 minutes, the card should return to green.

- [ ] **Step 4: Static-export end-to-end**

```bash
.venv/Scripts/python.exe dashboard/export_static.py
ls -la dashboard/static/api-data/ops-status.json
ls -la dashboard/static/api-data/logs-*.json
.venv/Scripts/python.exe -c "import json; d=json.load(open('dashboard/static/api-data/ops-status.json')); print('subsystems:', list(d.get('subsystems',{}).keys()))"
```

Expected: 5 new files exist; subsystems keys match the 9-card spec.

- [ ] **Step 5: Merge to main**

```bash
cd /mnt/q/finance-analyzer
cmd.exe /c "git fetch origin"
cmd.exe /c "git checkout main"
cmd.exe /c "git pull --ff-only"
cmd.exe /c "git merge --no-ff feat/dashboard-ops-board -m 'merge: dashboard ops board'"
cmd.exe /c "git push origin main"
```

If merge conflicts occur, resolve, run the test suite again, then push.

- [ ] **Step 6: Restart loops** (per CLAUDE.md infrastructure rules)

```bash
cmd.exe /c "schtasks /end /tn PF-DataLoop"
cmd.exe /c "schtasks /run /tn PF-DataLoop"
cmd.exe /c "schtasks /end /tn PF-MetalsLoop"
cmd.exe /c "schtasks /run /tn PF-MetalsLoop"
```

Refresh the browser to pick up the new dashboard.

- [ ] **Step 7: Cleanup worktree**

```bash
cd /mnt/q/finance-analyzer
cmd.exe /c "git worktree remove .worktrees/dashboard-ops-board"
cmd.exe /c "git branch -d feat/dashboard-ops-board"
```

---

## Self-Review Checklist (run by the executing agent before final commit)

- [ ] Every helper has a test that hits both happy and missing-data paths
- [ ] Every helper handles `FileNotFoundError` and corrupted JSONL gracefully (no 5xx)
- [ ] `/api/ops-status` never returns 5xx — broken helpers show as red, not 500
- [ ] All 9 status cards render with non-stale data when loops are running
- [ ] Default tab is Ops, not Accuracy, on a fresh page load
- [ ] Cross-tab links from cards activate the correct existing tab
- [ ] Static export produces 5 new files (ops-status.json + 4 log files)
- [ ] No new external JS or CSS dependencies (no React, no Tailwind, no shadcn)
- [ ] All existing 97 dashboard tests still pass
- [ ] Full pytest suite passes (`python -m pytest tests/`)
- [ ] chrome-devtools-mcp screenshot looks like a Jenkins-style ops board
- [ ] Lighthouse a11y score ≥ 90
- [ ] No `innerHTML` with interpolated content in any new JS — only safe DOM methods

---

## Backlog (post-MVP)

These are out of scope for this plan but tracked here:

- `PF-OutcomeCheck` heartbeat: write `data/outcome_check_heartbeat.json` on completion (~10 min)
- `PF-MLRetrain` heartbeat: same pattern (~10 min)
- `PF-FixAgentDispatcher` outcomes: structured success/failure log (~15 min)
- Sparkline trend in each status card (Chart.js mini-charts, ~30 min)
- Sound alert when a card flips red (browser Notification API, opt-in)
- Move CSS / JS / HTML out of the single-file `index.html` into separate files (refactor,
  ~2 hours, only worth doing if file grows past ~5,000 lines)
