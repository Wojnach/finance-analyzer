# Track 2 — Usage Signal (Internal Logs)

**Date:** 2026-05-03
**Author:** main agent (self-research)
**Status:** mostly a gap; documented findings + recommended instrumentation

## Question
Which dashboard tabs/endpoints actually get hit, and from what user-agent class? What's
the current phone-vs-desktop split?

## Findings

**No persistent request logging exists for the dashboard.**

What we found:
- `dashboard/app.py` only logs exceptions (`logger.exception(...)`). No request-level
  access log. There is no `@app.before_request` or `@app.after_request` that writes
  to a file (the existing `@app.after_request` at `dashboard/app.py:51` adds CORS
  headers, not logs).
- No `data/dashboard*.log` or equivalent in `data/`.
- No nginx/Apache layer in front: dashboard runs directly on Flask dev server on
  port 5055, exposed via Cloudflare Tunnel. Cloudflare Access logs are *not*
  pulled back into the project.
- `data/telegram_messages.jsonl` shows TG sends — useful for dashboard *Messages*
  tab content but not for dashboard usage itself.

**What we can infer indirectly:**

- The dashboard cookie has a 1-year rolling refresh (`dashboard/auth.py:41`,
  comment cites the Chrome 400-day cookie cap). That implies the user logs in
  rarely — supports the "open dashboard on phone, leave it logged in" hypothesis.
- The cookie-based auth was hardened on 2026-05-02 (commit `27398d44`,
  `9cdaa23a`) — recent activity suggests the user is actively using mobile/auth
  flows.
- Recent commits like `feat(dashboard): rolling 1-year cookie — re-auth practically
  never needed` (2026-04 era) directly motivated by user friction. So phone access
  *is* a real moment, not theoretical.

## Phone vs desktop traffic split — unknown

We have zero data. Either:
- A. Most usage is on desktop and phone is an edge case (mobile redesign is
  for-future-self).
- B. Phone usage is significant but we don't have the numbers.

Either way, "primarily phone" is the user's stated direction — design ahead of
the data.

## Recommendation: minimal instrumentation in this redesign

Because we are touching the dashboard anyway, add a tiny request log so we have
*next month's* numbers when we want to refine. Specifics for the implementation
batch:

- Add `@app.after_request` hook that appends one line per `/api/*` request to
  `data/dashboard_access.jsonl`: `{ts, path, method, status, ua_class,
  client_ip_present, response_ms}`. Strip identifying info (no full IP — just
  is_cloudflare_or_local; no UA string — just `phone`/`tablet`/`desktop`/`bot`
  classification by simple regex on UA).
- Skip non-`/api/*` paths — only the active fetches.
- Cap file at ~10 MB with daily rotation; the dashboard barely needs analytics
  granularity.
- This is **opt-in via config** so it can be disabled (the user's anti-PII memory
  in `feedback_no_api_keys_in_claude_settings.md` reflects appropriate caution).
- Do NOT log query params (some endpoints accept ticker filters with portfolio
  values — keep the surface minimal).

If the user prefers zero instrumentation, the redesign still ships — we just
keep flying blind for another iteration.

## Conclusion for the design

Without usage data, we *cannot* do data-driven home-screen prioritization.
Substitute by Track 3 (user-moment inference) and Track 4 (Telegram-overlap as
proxy for "what's already covered"). Re-evaluate after 30 days of access logs.
