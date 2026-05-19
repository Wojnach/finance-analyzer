# Dashboard Operations Board — Design Spec

**Date:** 2026-04-28
**Author:** Claude Opus 4.7 (paired with daisy)
**Status:** Approved for implementation
**Approved plan reference:** `/root/.claude/plans/i-need-us-to-dreamy-fiddle.md`

## Problem

The finance-analyzer dashboard surfaces 26 JSON endpoints across 10 tabs but has **no
at-a-glance operational health view**. The user must navigate to multiple tabs and visually
scan tables to answer the question "is the system running correctly right now?"

The Health tab exists in `dashboard/static/index.html` but is an empty
`<div id="healthC">Loading...</div>` placeholder — never implemented.

## Goal

Build a **Jenkins-style operations board** as the new default landing tab. At a glance, a
user must be able to see whether each major subsystem (MainLoop, MetalsLoop, LLMs, signals,
data feeds, errors) is **green / orange / red**, and drill into logs from the same view.

## Non-goals

- Full framework migration (Next.js / shadcn) — vanilla JS retained
- Mobile-responsive layout — desktop-only
- WebSocket / SSE push — current 5-second polling is fine for ops monitoring
- New instrumentation for `PF-OutcomeCheck`, `PF-MLRetrain`, `PF-FixAgentDispatcher`
  scheduled task heartbeats (in backlog, not blocking)
- Replacing or removing any existing tabs — they remain as deeper drilldowns

## Architecture

**Three-section vertical stack** in the Ops tab:

```
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 1: STATUS GRID (9 cards, 2x5 layout)                   │
│  Each card: traffic-light dot · name · summary · age            │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 2: DRILLDOWN STRIPS                                     │
│  Signals (33 dots) · LLMs (5 dots) · Tickers (5 dots)           │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 3: LOG INSPECTOR                                        │
│  Tabs: Critical Errors · System Log · Loop stdout                │
│  Search box + level filter                                       │
└─────────────────────────────────────────────────────────────────┘
```

Powered by **two new endpoints** (rollup + log tail) plus a small extension to the existing
`/api/health` endpoint.

## Component Design

### Backend: `dashboard/app.py`

#### `GET /api/ops-status`

Single rollup endpoint feeding the entire Ops board. Composed from existing helpers; no
new instrumentation.

**Response shape:**
```json
{
  "subsystems": {
    "layer1":         {"status": "green",  "summary": "heartbeat 45s ago", "age_s": 45,    "details": {...}},
    "layer2":         {"status": "green",  "summary": "agent ready",      "age_s": 312,   "details": {...}},
    "metals_loop":    {"status": "orange", "summary": "stale 4m",         "age_s": 245,   "details": {...}},
    "data_feeds":     {"status": "green",  "summary": "all feeds open",   "age_s": null,  "details": {...}},
    "signals":        {"status": "green",  "summary": "31/33 healthy",    "age_s": null,  "details": {...}},
    "llms":           {"status": "orange", "summary": "qwen 76%",         "age_s": null,  "details": {...}},
    "critical_errors":{"status": "green",  "summary": "0 in last 1h",     "age_s": null,  "details": {...}},
    "outcome_backfill":{"status": "green", "summary": "fresh",            "age_s": 1800,  "details": {...}},
    "accuracy_7d":    {"status": "orange", "summary": "53.2%",            "age_s": null,  "details": {...}}
  },
  "drilldowns": {
    "signals": [
      {"name": "trend",     "status": "green",  "rate_pct": 87.5, "calls": 248},
      {"name": "rsi",       "status": "green",  "rate_pct": 72.0, "calls": 200},
      {"name": "...",       "status": "...",    "rate_pct": ...,  "calls": ...}
    ],
    "llms": [
      {"name": "chronos",   "status": "green",  "ok_rate_pct": 98.2, "calls": 543},
      {"name": "ministral", "status": "orange", "ok_rate_pct": 84.0, "calls": 200},
      {"name": "...",       "status": "...",    "ok_rate_pct": ...,  "calls": ...}
    ],
    "tickers": [
      {"name": "BTC-USD",   "status": "green",  "accuracy_pct": 58.4},
      {"name": "...",       "status": "...",    "accuracy_pct": ...}
    ]
  },
  "timestamp": "2026-04-28T14:32:11+02:00"
}
```

**Status thresholds (module-level constants in `dashboard/app.py`):**

```python
OPS_THRESHOLDS = {
    "layer1_heartbeat":      {"green_under_s": 120, "orange_under_s": 300},
    "layer2_silence":        {"green_under_s": 7200, "orange_under_s": 14400},
    "metals_loop":           {"green_under_s": 120, "orange_under_s": 300},
    "signals_aggregate":     {"green_over_pct": 80, "orange_over_pct": 60},
    "signal_individual":     {"green_over_pct": 70, "orange_over_pct": 50},
    "llm_individual":        {"green_over_pct": 90, "orange_over_pct": 70},
    "ticker_accuracy":       {"green_over_pct": 55, "orange_over_pct": 45},
    "accuracy_7d":           {"green_over_pct": 55, "orange_over_pct": 45},
    "outcome_max_age_h":     {"green_under_h": 6,  "orange_under_h": 12},
}
```

**Reuses (no re-implementation):**
- `portfolio.health.get_health_summary()` for layer1, layer2, signals, breakers
- `portfolio.health.get_signal_health_summary()` for the per-signal drilldown
- `portfolio.health.check_outcome_staleness()` for outcome_backfill
- `portfolio.accuracy_stats.per_ticker_accuracy()` for ticker drilldown

**New helpers (private, in `dashboard/app.py`):**
- `_compute_metals_loop_status()` — reads `data/metals_invocations.jsonl` tail via
  `file_utils.load_jsonl_tail`
- `_compute_llm_health_summary()` — reads `data/forecast_health.jsonl` tail, groups by
  `model`, computes ok-rate
- `_compute_critical_error_window(window_seconds=3600)` — reads `data/critical_errors.jsonl`
  tail, filters by ts in last hour, groups by level
- `_compute_accuracy_7d()` — reads `data/accuracy_cache.json`, weights by recency
- `_status_color(value, thresholds, direction)` — one-liner that returns "green"/"orange"/"red"
  given a value, threshold dict, and direction ("over" or "under")

**Auth:** `@require_auth` decorator (existing, `dashboard/app.py:672`)
**Caching:** `_cached_read()` (existing, `dashboard/app.py:82`) with **5 second TTL**

#### `GET /api/logs/<source>?level=&limit=&since=`

Generic log tail. `<source>` is one of:
- `critical-errors` → `data/critical_errors.jsonl` (JSONL)
- `agent-log` → `data/agent.log` (JSONL)
- `loop-stdout` → `data/loop_out.txt` (plain text)
- `metals-loop-stdout` → `data/metals_loop_out.txt` (plain text)

Other sources return 404.

**Query params:**
- `level` (optional): comma-separated. For JSONL sources, filters by `level` field
  (e.g. `level=critical,warning`). Ignored for plain text.
- `limit` (default 50, max 500): max entries / lines to return.
- `since` (optional, ISO8601): only entries newer than this. Ignored for plain text
  (would require parsing every line for timestamp).

**Response shape (JSONL sources):**
```json
{ "entries": [ {...}, {...} ], "source": "critical-errors", "truncated": false }
```

**Response shape (plain text sources):**
```json
{ "lines": [ "...", "..." ], "source": "loop-stdout", "truncated": false }
```

**Auth:** `@require_auth` (same as everything else)
**Caching:** 2-second TTL (lower than ops-status because log views update faster)

#### Update to `GET /api/health`

Add one field, `metals_loop`, with shape:
```json
{ "last_invocation_ts": "...", "age_seconds": 45, "status": "green" }
```

Backward compatible — existing consumers see new field, can ignore.

### Backend: `dashboard/export_static.py`

Add to `ENDPOINTS` list (currently 20 entries, line 25):
```python
("/api/ops-status", "ops-status.json"),
("/api/logs/critical-errors", "logs-critical-errors.json"),
("/api/logs/agent-log", "logs-agent-log.json"),
("/api/logs/loop-stdout", "logs-loop-stdout.json"),
("/api/logs/metals-loop-stdout", "logs-metals-loop-stdout.json"),
```

This makes the GitHub Pages export include the Ops board data.

### Frontend: `dashboard/static/index.html`

Currently `dashboard/static/index.html` is a single 3,211-line file with inline `<style>`
and `<script>` blocks. Following the existing pattern (no extraction to separate files),
all changes go in this one file.

#### CSS additions (~250 lines)

New block in the existing `<style>` block, all classes prefixed with `ops-` to avoid
collision with the rest of the dashboard's CSS:

- `.ops-grid` — flex/grid layout for the 9 status cards (2 rows × 5 cols, last row has 4)
- `.ops-card` — individual card styling (border, padding, hover effect)
- `.ops-dot` — large traffic-light dot (`.ops-dot--green`, `.ops-dot--orange`, `.ops-dot--red`)
- `.ops-strip` — horizontal drilldown strip
- `.ops-mini-dot` — small dot for drilldown strips
- `.ops-log` — log inspector container
- `.ops-log-tabs`, `.ops-log-entry`, `.ops-log-filter` — log inspector children
- Color tokens (`--ops-green`, `--ops-orange`, `--ops-red`) added to existing CSS
  variable block

**Design principles** (from `frontend-design` skill):
- Restrained palette — only the status dots are saturated colors; everything else is
  greyscale
- 8px grid spacing (4/8/16/24/32)
- Typography hierarchy — single font, 4 sizes (12/14/18/24)
- High contrast (≥4.5:1) for accessibility

#### HTML additions

Replace `<div id="healthC">Loading...</div>` (current placeholder) with a complete
3-section structure: status grid, drilldown strips, log inspector.

#### JS additions (~500 lines)

Five small modules, all attached to a single `OpsBoard` namespace object:

- `OpsBoard.init()` — entrypoint, called when Ops tab is activated
- `OpsBoard.refreshStatus()` — fetches `/api/ops-status`, populates grid + strips,
  schedules next refresh (5s interval)
- `OpsBoard.renderCards(subsystems)` — DOM update for the 9 status cards
- `OpsBoard.renderDrilldowns(drilldowns)` — DOM update for signals/LLMs/tickers strips
- `OpsBoard.LogInspector` — sub-namespace with `.init()`, `.fetchSource(source)`,
  `.applyFilter()`, `.renderEntries()`

Cross-tab links: clicking a status card with `data-target-tab="signals"` calls the existing
tab manager to switch. Reuses the existing tab switcher (whatever function the existing
nav uses — TBD via grep during implementation).

#### Default tab change

Currently the Accuracy tab is the default per a previous commit ("accuracy tab as default").
Change the default tab init to point to Ops. Single-line change in the tab manager.

## Data Flow

```
Layer 1 loop             ─writes──►  data/health_state.json
                                             │
Metals loop              ─writes──►  data/metals_invocations.jsonl
                                             │
Layer 2 invocations      ─writes──►  data/invocations.jsonl
                                             │
LLM calls                ─writes──►  data/forecast_health.jsonl
                                             │
Critical errors          ─writes──►  data/critical_errors.jsonl
                                             │
Outcome backfill         ─writes──►  data/signal_log.jsonl (outcomes field)
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │  /api/ops-status   │
                                  │  (5s cache)        │
                                  └────────┬───────────┘
                                           │
                                           ▼
                                  ┌────────────────────┐
                                  │  Ops board UI      │
                                  │  (5s polling)      │
                                  └────────────────────┘
```

For static-export mode, the same data flows through `app.test_client()` in
`export_static.py` and lands as `dashboard/static/api-data/ops-status.json`.

## Error Handling

- **Missing data files** (e.g. `metals_invocations.jsonl` doesn't exist yet): helpers
  return `{"status": "red", "summary": "no data", "age_s": null, "details": {}}`. UI
  shows red dot + "no data" message.
- **Malformed entries** (corrupted JSONL line): use existing `file_utils.load_jsonl_tail`
  which already handles bad lines; helpers degrade gracefully.
- **Helper raises**: top-level try/except around each helper in `/api/ops-status`. Fail-safe
  is `{"status": "red", "summary": "error: <exception>", "details": {}}`. The endpoint
  itself never returns 5xx — it's a status board, errors must be VISIBLE not hidden.
- **Frontend fetch fails** (network error): existing fallback pattern (try live API,
  fall back to `static/api-data/ops-status.json`) — extends current behavior.
- **Invalid log source**: 404 with explanatory message in JSON body.

## Testing Strategy

### New tests in `tests/test_dashboard.py` (~12 new tests)

Following existing `class TestApiXxx` pattern, file ends at line 1586 currently:

- `class TestApiOpsStatus`
  - `test_requires_auth`
  - `test_returns_subsystems_shape` (validates the 9-key dict)
  - `test_returns_drilldowns_shape` (validates signals/llms/tickers arrays)
  - `test_handles_missing_metals_invocations` (file doesn't exist → red status, no 500)
  - `test_helper_failure_isolated` (one helper raises → that subsystem red, others green)

- `class TestApiLogs`
  - `test_requires_auth`
  - `test_unknown_source_returns_404`
  - `test_jsonl_source_returns_entries`
  - `test_jsonl_source_filters_by_level`
  - `test_text_source_returns_lines`
  - `test_limit_param_clamped_to_max`

- `class TestComputeMetalsLoopStatus` — direct helper tests (5 cases)

### New test in `tests/test_dashboard_export_static.py`

- Verify `ops-status.json` and 4 log-source JSON files appear in the output dir.

### Frontend smoke test in `tests/test_dashboard_frontend.py`

- `test_ops_board_html_present` — fetch `/`, assert response body contains `id="opsGrid"`,
  `class="ops-card"`, etc.

### Integration verification (manual + chrome-devtools-mcp)

After implementation:

1. Start dashboard: `.venv/Scripts/python.exe dashboard/app.py`
2. Use `chrome-devtools-mcp:chrome-devtools` skill: navigate to `http://localhost:5055/?token=...`,
   take screenshot of Ops tab, check console for errors, verify
   `GET /api/ops-status` returns 200.
3. Use `chrome-devtools-mcp:a11y-debugging` skill: confirm Lighthouse a11y score ≥90
   on the Ops tab.
4. **Flip test**: stop `PF-MetalsLoop` scheduled task. Within 5 minutes the Metals card
   should flip orange → red. Restart task; card returns to green within one cycle.
5. **Static export test**: `.venv/Scripts/python.exe dashboard/export_static.py` should
   write `ops-status.json` and 4 log files to `dashboard/static/api-data/`.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Existing 97 tests break due to changes in `/api/health` | New field is additive; existing assertions on existing keys keep passing. Run full suite after each step. |
| `_compute_critical_error_window` reads a 100MB+ JSONL file | Use `file_utils.load_jsonl_tail` (BUG-122 pattern, line 277 of `health.py`) — only reads tail, not full file. |
| 5s polling × 9 helpers could overload disk | All helpers go through the existing `_cached_read` 5s cache; one disk read per source per 5s. |
| Default tab change disorients users who relied on Accuracy as landing | Existing tabs remain accessible; user retains muscle memory. The tab nav itself doesn't change shape. |
| Frontend 500-line addition collides with the existing 3,211-line monolith | Strict `ops-` namespacing + `OpsBoard` JS namespace; sectioned with comment headers; can be extracted later. |

## Acceptance Criteria

- [ ] `/api/ops-status` returns the documented shape with all 9 subsystems
- [ ] `/api/logs/<source>` works for all 4 sources, with level/limit/since filters
- [ ] `/api/health` includes the new `metals_loop` field
- [ ] Ops tab is the default landing tab and renders all 3 sections without console errors
- [ ] Status cards flip color correctly when underlying state changes (verified via flip test)
- [ ] All 97 existing dashboard tests still pass; ~12 new tests added and passing
- [ ] Static export includes `ops-status.json` and 4 log-source JSON files
- [ ] Lighthouse a11y score ≥90 on the Ops tab
- [ ] No new external JS or CSS dependencies (no React, no Tailwind, no shadcn — pure
      vanilla extension of the existing dashboard)
