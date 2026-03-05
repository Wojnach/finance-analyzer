# Improvement Plan

Updated: 2026-03-05
Branch: improve/auto-session-2026-03-05

## 1) Bugs & Problems Found

Priority is critical runtime correctness first.

### P1 — Dashboard JSONL schema assumptions can crash endpoints (500)

- Evidence:
  - `dashboard/app.py:169`, `dashboard/app.py:171`, `dashboard/app.py:413`, `dashboard/app.py:425` use `entry.get(...)` directly.
  - `portfolio/file_utils.py:57-63` accepts any valid JSON line in `load_jsonl` (not guaranteed dict).
- Failure mode:
  - If a JSONL line is valid JSON but non-object (e.g., `"msg"`, `[]`, `123`), `entry.get` raises `AttributeError`.
- Affected APIs:
  - `/api/telegrams`, `/api/decisions`.
- Impact assessment:
  - Can break dashboard pages and static sync consumers during runtime if one malformed/non-dict entry is present.

### P1 — Accuracy endpoint can fail on one malformed JSONL line

- Evidence:
  - `portfolio/accuracy_stats.py:41` (`json.loads(line)` no guard in fallback path).
  - `/api/accuracy` in `dashboard/app.py:187-210` returns 500 when backend raises.
- Failure mode:
  - Single malformed line in `data/signal_log.jsonl` can break accuracy calculations entirely.
- Impact assessment:
  - Degrades monitoring and strategy calibration visibility; avoids graceful degradation.

### P2 — Static export tool lacks endpoint parity and auth support

- Evidence:
  - `dashboard/export_static.py:24-40` endpoint list misses frontend-used routes.
  - Frontend references `/api/metals-accuracy` and `/api/lora-status` (`dashboard/static/index.html:2129`, `dashboard/static/index.html:2553`).
  - Export requests do not pass dashboard token (`dashboard/export_static.py:64`), while auth is enforced when token configured (`dashboard/app.py:58-80`).
- Failure mode:
  - Incomplete or failed static exports for token-protected deployments.
- Impact assessment:
  - Public/static dashboards can show stale or missing sections despite live API availability.

### P3 — Config validation consistency gap outside loop mode

- Evidence:
  - `validate_config_file()` called only in loop startup (`portfolio/main.py:469`).
  - Non-loop command path ends at `run(force_report="--report" in args)` (`portfolio/main.py:640`) without strict validation pass.
- Impact assessment:
  - One-shot runs may fail later and less clearly compared with loop startup behavior.
- Decision:
  - Defer for this session due high blast radius across maintenance commands; document only.

## 2) Architecture Improvements

### A1 — Normalize dashboard JSONL ingestion boundaries

- Improvement:
  - Add explicit type filtering for JSONL-derived entries at dashboard endpoint layer before accessing fields.
- Why it matters:
  - Enforces API contract at system boundary and decouples endpoint correctness from log writer strictness.
- Enables:
  - Future coexistence with heterogeneous JSONL writers without endpoint fragility.
- Impact assessment:
  - Low risk; changes are additive and backward compatible.

### A2 — Make accuracy stats reader tolerant to partial log corruption

- Improvement:
  - Skip malformed lines in `accuracy_stats.load_entries()` JSONL fallback, optionally log a debug warning.
- Why it matters:
  - Keeps analytics partially available even when logs contain occasional corruption/truncation.
- Enables:
  - Resilient long-running operations and easier operational recovery.
- Impact assessment:
  - Low risk; preserves existing semantics for valid lines.

### A3 — Align static exporter with frontend data contract and auth model

- Improvement:
  - Include frontend-required endpoints and optional token propagation in static exporter.
- Why it matters:
  - Reduces coupling bugs between frontend assumptions and exported data availability.
- Enables:
  - Reliable static-site mode in authenticated environments.
- Impact assessment:
  - Medium-low risk; isolated to export script behavior.

## 3) Useful Features

### F1 — Exporter token support for authenticated dashboard

- Feature:
  - Export tool reads `dashboard_token` from config and appends token query param when present.
- Justification:
  - Needed for environments where API auth is enabled.
- Impact assessment:
  - Low; only affects export script requests.

### F2 — Dashboard hardening tests for malformed JSONL object types

- Feature:
  - Add endpoint-level tests proving `/api/telegrams` and `/api/decisions` ignore non-dict JSON lines instead of 500.
- Justification:
  - Prevents regression and captures real-world log variability.
- Impact assessment:
  - Low; tests only.

### F3 — Coverage for `/api/metals-accuracy` endpoint behavior

- Feature:
  - Add tests for present/missing data and auth behavior.
- Justification:
  - Current explicit gap in dashboard endpoint coverage.
- Impact assessment:
  - Low; tests only.

## 4) Refactoring TODOs

1. Consider central helper in dashboard for JSONL entry normalization to avoid repeated `isinstance(dict)` checks.
2. Standardize JSONL writer paths to prefer `atomic_append_jsonl` across legacy scripts under `data/`.
3. Reconcile `docs/architecture-plan.md` with actual trigger semantics (no cooldown trigger) and current symbol/signal inventories.
4. Review `portfolio/config_validator.py` required-key policy to support mode-specific runs (e.g., crypto-only or no-Alpaca scenarios).

## 5) Dependency / Ordering (Implementation Batches)

Batches are ordered by risk reduction and dependency.

### Batch 1 — Dashboard endpoint hardening + coverage

- Scope files (2):
  - `dashboard/app.py`
  - `tests/test_dashboard.py`
- Changes:
  - Guard JSONL-derived entries to process only dict objects in `/api/telegrams` and `/api/decisions`.
  - Add tests for non-dict JSONL lines and `/api/metals-accuracy` route.
- Why first:
  - Immediate user-visible stability improvement with minimal blast radius.
- Potential breakpoints:
  - Tests that implicitly assumed acceptance of non-dict payloads in endpoint output.

### Batch 2 — Accuracy JSONL fallback resilience + tests

- Scope files (2):
  - `portfolio/accuracy_stats.py`
  - `tests/test_signal_improvements.py` (or dedicated new accuracy-stats test file)
- Changes:
  - Skip malformed JSONL lines in `load_entries()` fallback path.
  - Add regression test ensuring malformed line does not break accuracy calculations.
- Dependency:
  - Independent from Batch 1.
- Potential breakpoints:
  - Any tests expecting hard failure on malformed logs (unlikely).

### Batch 3 — Static exporter parity/auth improvements + tests

- Scope files (3):
  - `dashboard/export_static.py`
  - `tests/test_dashboard.py` (or a new exporter test module)
  - `docs/CHANGELOG.md`
- Changes:
  - Add missing endpoint exports used by frontend.
  - Add token-aware export requests.
  - Add tests for successful export in token-enabled mode (via patched client behavior).
- Dependency:
  - No strict dependency, but scheduled last to keep runtime-critical fixes first.
- Potential breakpoints:
  - Existing automation expecting previous endpoint count or strict fail-fast behavior.

## Execution Notes

- Implementation follows: tests-first per batch, then code change, then full test suite run.
- Any risky or uncertain behavior discovered mid-batch will be marked `TODO: MANUAL REVIEW` and deferred with rationale.
