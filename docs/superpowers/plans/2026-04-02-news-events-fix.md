# Fix News/Event Gaps in Fish Monitor

## Problem
Monitor ignores news/events. We entered SHORT on Liberation Day without knowing.
econ_calendar says SELL but monitor only displays "E=SELL" without acting on it.
Headlines are fetched but never persisted — monitor can't see WHY signals fire.

## Findings
- econ_calendar DOES return event details in indicators dict (proximity_hours, event name)
- news_event DOES return headline counts, severity, keywords in indicators
- The monitor just doesn't READ these fields
- Headlines cached in-memory only — never written to disk

## Plan

### Batch 1: Monitor reads event/news details (fish_monitor_live.py)
- Read econ_calendar indicators: proximity_hours, risk_nearest_event, type_event_impact
- Read news_event indicators: severity_max_severity, velocity_article_count, severity_keywords_found
- Display as WARNING when:
  - Event within 24h: "!! OPEC 4h away — reduce hold time"
  - High severity news: "!! CRITICAL NEWS: tariff/crash keywords detected"
  - Headline velocity spike: "!! NEWS SPIKE: 15 articles (baseline 5)"
- Reduce max hold from 2h to 1h when high-impact event within 24h
- Do NOT hard-gate entries — just warn and reduce hold time

### Batch 2: Persist headlines to file (news_event.py or sentiment.py)
- After fetching headlines, write top 5 to data/headlines_latest.json:
  {ticker, timestamp, headlines: [{title, source, sentiment, severity}]}
- Monitor reads this file and displays latest headline in log
- This lets us see "Trump announces tariff action" instead of just "NEWS=SELL"

### Batch 3: Metals loop periodic news check
- Every 30 min in metals loop, fetch headlines for XAG/XAU
- Write to data/metals_news_summary.json
- Monitor reads alongside the main loop headlines
- Provides independent news check (metals loop runs separately)

### What could break
- Batch 1: Over-warning could cause alert fatigue. Keep to HIGH impact only.
- Batch 2: Writing headlines adds I/O to signal computation. Use atomic_write.
- Batch 3: NewsAPI rate limit (100/day). Metals loop already uses some quota.
  30min interval = 48/day for metals news. Should be within budget.
