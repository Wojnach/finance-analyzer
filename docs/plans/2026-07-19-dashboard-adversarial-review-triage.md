# Dashboard Adversarial Review — Verified Triage (2026-07-19/20)

Two independent hostile reviewers on dashboard/: **codex** (0 CRIT/12 HIGH/14 MED/2 LOW,
data-honesty focus) + **fresh Claude Fable 5** (0 CRIT/0 HIGH/2 MED/7 LOW, security focus,
empirically confirmed live). Orchestrator verified each finding below against code + live
`/api/*` before listing. Reviewer output is NOT trusted blind — items proven false or purely
latent-and-unreachable are marked SKIP/NOTE.

Decisions (user 2026-07-19): fix all confirmed now; health color must DISTINGUISH
systemd-disabled (user turned it off, neutral) from enabled-but-dead / missing-file (fault).

## Cluster A — system_status.py honesty (heredoc-only file)

1. **[HIGH] `_color` ignores sources/layer1/voters/avanza** (system_status.py:~1135). Only
   heartbeat+errors+violations+aggregate pct feed it. All subsystem loops frozen → still GREEN.
   FIX: feed loop-heartbeat freshness + layer1 + avanza into `_color`, BUT distinguish
   disabled-vs-dead: a loop whose systemd unit is `disabled` (read via control.py's
   `_systemctl_query`/loop-processes, or a new `enabled` flag per source) is NEUTRAL, not a
   fault; only `enabled && (frozen || missing)` bumps YELLOW/RED. In build/test mode (all
   disabled) the hero must NOT go red.
2. **[HIGH] Missing source file → `frozen:false`** (system_status.py:854 `_source_freshness`).
   OSError returns `{mtime:null,age_sec:null,frozen:false}`. A crashed loop that removed its
   heartbeat reads healthy. FIX: add `missing:true` and treat missing-of-an-enabled-loop as
   frozen/fault (compose with the disabled-vs-dead rule from #1).
3. **[HIGH] `VOTING` ≠ can-vote** (system_status.py:~994 `_voter_state`). Verified live:
   phi4_mini=VOTING while layer1.active=False. Rescued signals return VOTING without checking
   Layer 1 running or the shadow-registry throttle (phi4 status=shadow,cycle_modulo=10 →
   force-HOLD most cycles). FIX: new state `PAUSED_LOOP_DOWN` when layer1 inactive; surface
   shadow-throttle in reason (don't claim plain VOTING for a modulo'd shadow signal).
4. **[MED] Claude gate ACTIVE from incomplete evidence** (system_status.py:159 `_claude_gate`).
   `(config=None, module=True, metals=None)` → ACTIVE though 2 gates unknown. FIX: UNKNOWN
   label when any required input is None, only ACTIVE when all three readable-and-true.
5. **[MED] Lexicographic ISO windowing** (system_status.py:~1283 `_load_last_n_hours`). String
   compare assumes `+00:00`; a `Z` or offset row mis-windows. FIX: parse to aware datetime,
   compare instants. (Holds by convention today — real but low blast radius.)

## Cluster B — app.py / control.py / house_blueprint robustness (heredoc for app.py; control.py IS black-clean)

6. **[HIGH] Signal heatmap no freshness contract** (app.py:1253 `api_signal_heatmap`). Reads
   agent_summary.json, discards mtime. Producer dies → stale grid at 200 forever. FIX: attach
   `{data_ts, age_sec}` (file mtime) like /api/accuracy meta; frontend shows stale marker.
7. **[MED] Heatmap 500 on malformed shape** (app.py:1261). Truthy scalar/list into `.get`.
   FIX: isinstance guards → in-band `{"error":...}` not 500.
8. **[MED] Message search 500 on wrong-typed field** (app.py:1040). `{"category":1}`+`?category=`
   or `{"text":[]}`+`?search=` → `.lower()` on non-str. FIX: coerce/skip non-str.
9. **[MED] TTL cache lets older read overwrite newer** (app.py:157 `_cached_or`). Lock released
   during read_fn; slow old read finishes last, replaces fresh. FIX: re-check-and-keep-newest
   under lock by stamping each read with a monotonic start and discarding if a newer value
   landed. (Low frequency; fix cheaply.)
10. **[MED] control `/state` reports systemctl outage as definite stopped** (control.py:126;
    same at system_status.py:885 layer1). `_systemctl_query`→None → active:false. FIX: None →
    `"unknown"` (or `active:null`), UI renders "unknown" not "stopped".
11. **[MED] house `/` 500 on scalar manifest** (house_blueprint.py:118 `_list_runs`). `len(slugs)`
    on a non-list. FIX: isinstance guard, isolate the bad run.
12. **[MED→FIX] Audit logs loopback IP** (control.py:119). `request.remote_addr`=127.0.0.1
    (cloudflared→localhost). Real identity = CF-verified email (available in require_auth but
    not propagated). FIX: stash verified email into `flask.g` in auth.py CF branch; audit records
    `actor` (g email or "token"), `auth_method`, `cf_connecting_ip` (CF-Connecting-IP header,
    claimed), keep `remote_addr`.
13. **[LOW] /logout secure=True over LAN HTTP** (app.py:854/862). Other cookie writes use
    `_request_is_https()`; logout hardcodes True → can't clear cookie on plain-HTTP LAN. FIX:
    `secure=_request_is_https()`.

## Cluster C — frontend honesty (JS via Edit; prettier ok)

14. **[HIGH] Global error slot cleared by any success** (fetch.js:55). `Slots.ERROR` is global;
    every success `state.set(ERROR,null)`. /api/health fails → banner, then /api/loop_health
    success wipes it. FIX: key errors by url (map) or only clear on the same key's success.
15. **[MED] "last refresh now" on failed poll** (polling.js:96). `fj()` returns null on failure,
    poll resolves, LAST_REFRESH stamped. FIX: only stamp on non-null result; expose last-success
    vs last-attempt.
16. **[MED] Overlapping polls, old result wins** (polling.js:81). `runningP` tracked, never
    checked. FIX: skip a new `_fire` while one is in-flight, or ignore a resolved result older
    than the latest issued (sequence guard).
17. **[MED] localStorage/sessionStorage uncaught can blank SPA** (theme.js:18, freshness-banner.js:33).
    SecurityError (private mode / storage disabled) aborts boot / `_renderAll`. FIX: try/catch
    wrappers, degrade to defaults.
18. **[LOW] Malformed hash crashes routing** (router.js:129 `_decodeParams`). `decodeURIComponent`
    outside mount error handler; `#decisions/%E0%A4%A` → URIError, no view. FIX: try/catch → treat
    as raw / route to fallback.
19. **[HIGH] Silver Avanza box green on creds-only** (silver-pipeline.js:191/213). `credsOk?green:red`
    ignores unresolved_errors + session expiry. FIX: green only if creds AND no unresolved avanza
    errors AND (session fresh if that signal exists); else amber/red. (Latent now — creds off →
    already red — but fix the logic.)
20. **[HIGH] Component pills ignore voter_state** (silver-components.js:45/68). `enabled_default`
    → "enabled — voting"; phi4 SHADOW rendered as voting. FIX: color/label from voter*state
    (VOTING/SHADOW/DISABLED/PAUSED*\*) not just enabled_default.
21. **[LOW] prophecy innerHTML unescaped** (prophecy.html:84). `market_outlook`/`date`/`model`
    interpolated without `esc()`. Not attacker-reachable today (market_outlook never published)
    but latent stored-XSS on an authed page. FIX: wrap all three in `esc()`.

## NOTE / SKIP (not fixed — surfaced to user)

- **[MED] CF Access authZ scope** (auth.py:208). JWT verified; authZ fully delegated to the
  Cloudflare Access application policy. If that policy admits a whole email domain, any admitted
  identity gets the systemctl write surface. CONFIG, not code — user must verify the Access app
  is scoped to exactly one operator identity. SURFACED.
- **[LOW] Fail-open when dashboard_token unset/empty** (auth.py:181). `or None` → blank token =
  open access incl. write surface. Latent (token set). Optional hardening: if CF Access is
  configured, require it even when token blank. NOTED, not fixed (changes documented backward-compat).
- **[LOW] CSRF relies solely on SameSite=Lax** (auth.py:152). Mitigated today (Lax+httponly).
  Backstop (Origin check on control POSTs) is cheap insurance — FOLD INTO Cluster B if easy, else note.
- **[LOW] trading_status no holiday calendar** (trading_status.py `_in_session`). Dec-25-Friday →
  SCANNING. Feature gap, not a lie about current data. SKIP (backlog).

## SOLID (reviewer-confirmed, no action)

CF JWT verify (RS256 pinned, aud/iss/exp/email-claim checked, JWKS from config); systemctl
list-form + allowlist-before-use + pf-dashboard excluded; rate limiter (global, locked, post-auth);
hmac.compare_digest everywhere; path-traversal regexes anchored + secure_filename; nh3 sanitizer
on house markdown; SPA DOM via textContent (no innerHTML outside prophecy); no secret echo;
per-section 500 envelopes; MAX_CONTENT_LENGTH cap.
