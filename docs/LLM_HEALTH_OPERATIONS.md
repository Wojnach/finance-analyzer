# LLM Health Operations — setup & runbook

Operational guide for the LLM-health pipeline added 2026-04-21. Covers
what the scripts do, how to schedule them, and how to interpret the
daily report.

## Quick status check

```bash
# Which shadows are stale (>30 d without resolution)?
.venv/Scripts/python.exe scripts/review_shadow_signals.py

# Per-model sentiment accuracy (needs backfill to have run at least once)
.venv/Scripts/python.exe scripts/backfill_sentiment_shadow.py --show-accuracy

# Full daily report as JSON
.venv/Scripts/python.exe -c "
import json
from portfolio.local_llm_report import build_local_llm_report
print(json.dumps(build_local_llm_report(days=30), indent=2, default=str))"
```

## The data flow

```
Layer 1 loop ────────► signal_engine central hook ─────► data/llm_probability_log.jsonl
                         (fire-and-forget log_vote)       (one row per LLM vote)
                                                                    │
                                                                    ▼
                       data/sentiment_ab_log.jsonl       scripts/backfill_llm_outcomes.py
                                │                                   │
                                ▼                                   ▼
           scripts/backfill_sentiment_shadow.py    data/llm_probability_outcomes.jsonl
                                │
                                ▼
           data/sentiment_shadow_outcomes.jsonl
                                │
                                └──────────────┬─────────┘
                                               ▼
                               portfolio/local_llm_report.py
                                               │
                                               ▼
                   data/local_llm_report_latest.json
                   data/local_llm_report_history.jsonl
```

## Scheduled tasks to register

`PF-DataLoop`, `PF-Dashboard`, `PF-OutcomeCheck`, `PF-MLRetrain` already
exist (see CLAUDE.md). Add one more:

### `PF-LLMBackfill` — hourly

Runs both outcome backfills. Idempotent; skips rows whose horizons haven't
elapsed and rows already backfilled.

Create via Task Scheduler GUI or:

```powershell
schtasks /create /tn "PF-LLMBackfill" /sc HOURLY /mo 1 /ru "%USERNAME%" `
  /tr "cmd.exe /c \"cd /d Q:\finance-analyzer && .venv\Scripts\python.exe scripts\backfill_llm_outcomes.py >> data\llm_backfill_out.txt 2>&1 && .venv\Scripts\python.exe scripts\backfill_sentiment_shadow.py --horizon 1d >> data\llm_backfill_out.txt 2>&1\""
```

Or copy the pattern from `PF-OutcomeCheck` in Task Scheduler, replace the
command with the two backfill scripts.

### `PF-ShadowReview` — daily

Alerts when a shadow signal exceeds 30 d without resolution. Exit code 1
on stale, suitable for email-on-error Windows scheduled task.

```powershell
schtasks /create /tn "PF-ShadowReview" /sc DAILY /st 07:00 /ru "%USERNAME%" `
  /tr "cmd.exe /c \"cd /d Q:\finance-analyzer && .venv\Scripts\python.exe scripts\review_shadow_signals.py >> data\shadow_review_out.txt 2>&1\""
```

## Interpreting the daily report

Fields added by the 2026-04-21 audit:

### `calibration`
Per-signal Brier score + log-loss, computed from
`llm_probability_log.jsonl` joined with outcomes. Populated only after
`backfill_llm_outcomes.py` runs. Before the first backfill, everything
reports under `missing_outcome`.

- **Brier score**: lower is better. Random 3-way guess ≈ 0.667. Perfect = 0.
  Maximally wrong = 2.0. Production-quality signals tend to land in 0.3-0.5.
- **Log-loss**: `-log P(true class)`. Lower is better. eps-clipped to stay
  finite when a prob is 0.
- **`buckets`**: per-chosen-action hit rate. Useful for the "confidently
  wrong" diagnostic — if a model chose BUY 200 times and was right 40 %
  of them, that's different from 200 mixed choices at 40 % aggregate.

### `shadow_registry`
Snapshot of the registry file plus a `stale` list (signals > 30 d in
shadow without resolution). Each stale row includes `signal` and
`days_in_shadow`.

### `sentiment_shadow_accuracy`
Per-model (primary + shadow) accuracy from the sentiment A/B log
backfill, bucketed by horizon (1d, 3d). Shows `samples`, `correct`,
`accuracy`, `agreement_with_primary`, `kind`.

Primary models appear with `kind: "primary"` and
`agreement_with_primary: null`.

**Interpretation caveat**: accuracy includes HOLD (neutral) predictions.
A signal that outputs 90 % neutral will score low by this metric even if
its rare BUY/SELL are excellent. The separate
`portfolio.accuracy_stats.accuracy_by_signal_ticker` layer filters HOLDs
and reports directional-only accuracy — use it for promotion-gate
evaluation.

## Promoting a shadow

When a shadow model clears its promotion criteria:

1. The daily report's recommendations will flag it:
   > "Shadow sentiment model `X` cleared promotion gate (62.1 % on 237 samples at 1d). Consider promoting."

2. Promote in code:
   - For sentiment: modify `portfolio/sentiment.py`'s primary-selection
     logic to route the ticker class to the new model.
   - For voting LLMs: adjust the dispatch in `portfolio/signal_engine.py`.

3. Update the registry:
   ```python
   from portfolio.shadow_registry import resolve_shadow
   resolve_shadow("<signal>", "promoted", notes="Accuracy X% on Y samples")
   ```

4. Keep the old shadow running in parallel for N more weeks as a
   regression canary before fully retiring its code path.

## Retiring a shadow

When a shadow has sat > 30 d with no promotion evidence OR has
persistent issues (Kronos at 59 % subprocess reliability was the
textbook case, though we didn't ultimately retire it — we isolated the
vote pool instead):

1. Decide: retire with full code removal, or isolate (let it log but
   not vote). Isolation is reversible, retirement is harder to undo.
2. If isolating: modify the composite vote function to exclude the
   shadow sub-signal (see `_health_weighted_vote` in forecast.py for
   the pattern).
3. Update the registry:
   ```python
   from portfolio.shadow_registry import resolve_shadow
   resolve_shadow("<signal>", "retired", notes="Reason + date")
   ```
4. Add a follow-up memory explaining why.

## Common issues

**"No outcomes yet" in backfill output**: price snapshots
(`data/price_snapshots_hourly.jsonl`) don't cover the log's time range,
OR the horizon hasn't elapsed. Both are expected early in the log's
life. Rerun daily — counts go up.

**`missing_price` count high**: ticker short-forms may need adding to
`_TICKER_EXPAND` in `portfolio/sentiment_shadow_backfill.py`. Current
map covers BTC/ETH/XAU/XAG.

**Scheduled task reports "Running" but no python process**: singleton
lock is held by a dead process. See CLAUDE.md's loop-restart protocol —
kill orphan pythons first, then `schtasks /run`.
