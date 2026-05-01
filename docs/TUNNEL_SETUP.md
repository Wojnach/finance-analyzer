# Cloudflare Tunnel Setup — Dashboard at raanman.lol

Exposes the finance-analyzer dashboard (Flask, port 5055) to the internet via
Cloudflare Tunnel. As of 2026-04-30, the architecture is **apex-direct**:
typing `raanman.lol` in any browser hits the live dashboard, gated by a
30-day auth cookie set after a one-time `?token=` URL.

## Architecture

```
Browser → raanman.lol           → Cloudflare Edge → Tunnel → localhost:5055 (Windows)
Browser → bets.raanman.lol      → Cloudflare Edge → Tunnel → localhost:5055   (legacy alias)
```

Auth is on the Flask side (cookie / `?token=` query / `Authorization: Bearer`),
*not* at the Cloudflare edge — the tunnel is open, the dashboard gates.

### History

The original design (commit `ff2bc845`, 2026-04-28) was path-based:
`raanman.lol/bets` served from GitHub Pages and redirected to a
`bets.raanman.lol` subdomain. We pivoted to apex-direct on 2026-04-30 because:

- The GH Pages content was a stale clone of the dashboard SPA + 20 stale
  `api-data/*.json` snapshots, all reachable publicly with no auth — a real
  data leak that a redirect couldn't fix.
- Two hostnames doubled the auth surface area.
- Apex-direct gives the shortest possible URL.

The `bets.raanman.lol` subdomain is kept in `cloudflared` ingress as a legacy
alias so old bookmarks don't break, but `raanman.lol` is canonical.

## Prerequisites

- Cloudflare account (Free plan)
- The finance-analyzer dashboard running on port 5055
- Windows machine with internet access
- Administrator shell to run `setup_tunnel.bat` (needed for `cloudflared service install`)

## Step 1: Move `raanman.lol` to Cloudflare nameservers (~30 min, one-time)

Cloudflare's Argo Tunnel requires the zone to be on Cloudflare's nameservers
(it issues an origin cert tied to your zone, which it can't do for a domain
it doesn't authoritatively serve).

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com), log in or sign up
2. Click **Onboard a domain** (or whatever the current label is — it's been
   renamed several times). Enter `raanman.lol`. Choose **Free** plan.
3. Cloudflare scans existing DNS records. **Delete** anything that should not
   exist on Cloudflare's nameservers — for finance-analyzer, the only thing
   you want kept is records related to the tunnel (added later by the bat).
4. Cloudflare gives you two nameservers (e.g. `darl.ns.cloudflare.com`,
   `jacqueline.ns.cloudflare.com`). Copy both.
5. Log into your registrar (Porkbun, etc.). Find the **Authoritative
   Nameservers** field. **Replace** all existing entries with the two
   Cloudflare gave you.
6. Wait for propagation. Watch with:
   ```
   nslookup -type=NS raanman.lol 8.8.8.8
   ```
   When the response shows Cloudflare nameservers, you're live. Typical:
   10 min – 2h. Worst case: 24h.
7. **Important if DNSSEC was previously enabled at your registrar**: turn it
   off *before* changing nameservers, or validating resolvers will SERVFAIL
   your domain until DS records age out of caches.

### Required DNS records after Step 1

After Step 1 completes there is no `raanman.lol` content yet — the tunnel
adds records in Step 2. The records you should NOT have at this stage:

| Type      | Name | Notes                                               |
|-----------|------|-----------------------------------------------------|
| A / AAAA  | apex | If GH Pages was previously serving — delete         |
| CNAME     | www  | If `www → wojnach.github.io` existed — delete       |
| CNAME     | `*`  | Porkbun parking artifact (`pixie.porkbun.com`) — delete |

## Step 2: Run the setup bat (~5 min, one-time)

From an **elevated cmd** (Run as administrator):

```cmd
cd Q:\finance-analyzer
scripts\setup_tunnel.bat
```

The bat runs 7 steps:

| Step  | Action                                                                  |
|-------|-------------------------------------------------------------------------|
| [0/6] | Refuses to run if `dashboard_token` in `config.json` is empty/short     |
| [1/6] | `winget install` cloudflared if not on PATH                             |
| [2/6] | `cloudflared login` — opens browser, authorize raanman.lol zone         |
| [3/6] | `cloudflared tunnel create finance-dashboard` (writes creds JSON)       |
| [4/6] | Writes `%USERPROFILE%\.cloudflared\config.yml`                          |
| [5/6] | `cloudflared tunnel route dns` — creates apex CNAME pointing to tunnel  |
| [6/6] | Copies config + creds to `%SystemRoot%\System32\config\systemprofile\`, runs `cloudflared service install`, patches the service `ImagePath` via `fix_cloudflared_imagepath.ps1`, starts the service |

**Why the `ImagePath` patch matters** (added 2026-04-30, undocumented in
Cloudflare's Windows guide): `cloudflared service install` registers the
binary with no arguments, so the service starts the bare `cloudflared.exe`
which exits immediately ("process terminated unexpectedly"). The patch sets
`ImagePath` to include `--no-autoupdate tunnel run finance-dashboard` so the
service actually runs the tunnel. Re-applies idempotently on every bat run.

**Why config + creds need to live in systemprofile**: the cloudflared
service runs as `LocalSystem`, whose `%USERPROFILE%` is
`%SystemRoot%\System32\config\systemprofile\` — *not* the human user's
profile. `cloudflared service install` does NOT copy the config; the bat
does it explicitly.

## Step 3: First-time browser auth (one-time per device)

The dashboard is now reachable but gated. To set the 30-day auth cookie,
open this URL **once** in your browser:

```
https://raanman.lol/?token=<dashboard_token from config.json>
```

The dashboard validates the token, sets an `HttpOnly` `Secure` `SameSite=Lax`
cookie named `pf_dashboard_token`, then 302-redirects you to a clean
`https://raanman.lol/`. Bookmark *that*. From then on, just typing
`raanman.lol` in the same browser hits the dashboard directly. The cookie
expires after 30 days; re-visit the token URL once to refresh.

## Step 4: Verify

```cmd
.venv\Scripts\python.exe scripts\verify_tunnel.py
```

Expected output: `RESULT: all checks passed.` covering:
1. `raanman.lol/api/health?token=…` returns 200
2. The 200 response sets the `pf_dashboard_token` cookie
3. `raanman.lol/api/health` (no token) returns 401 — gate fires
4. `raanman.lol/api/health` with the cookie returns 200
5. `bets.raanman.lol/api/health` (legacy alias) returns 200

A red verifier means something's regressed — see the Troubleshooting
section below.

## Daily canary (optional but recommended)

`scripts/verify_tunnel_alerted.py` is a wrapper around `verify_tunnel.py`
that sends a Telegram alert (via the Bot API directly, bypassing
`telegram.mute_all`) when the verifier exits non-zero. To install as a
daily Windows scheduled task at 09:00 Stockholm time:

```cmd
powershell -NoProfile -Command "Register-ScheduledTask ^
  -TaskName 'PF-VerifyTunnel' ^
  -Action (New-ScheduledTaskAction ^
    -Execute 'Q:\finance-analyzer\.venv\Scripts\python.exe' ^
    -Argument 'Q:\finance-analyzer\scripts\verify_tunnel_alerted.py') ^
  -Trigger (New-ScheduledTaskTrigger -Daily -At 9am) ^
  -Force"
```

The wrapper always exits 0 (so Task Scheduler doesn't classify the run as
failed and trigger retries) — Telegram is the only failure signal. If
Telegram itself fails, the wrapper writes to `data/critical_errors.jsonl`
which feeds `PF-FixAgentDispatcher`.

## Management commands

```cmd
# Service status
sc query cloudflared

# Tunnel info (active connections, recent connectors)
cloudflared tunnel info finance-dashboard

# Restart service (after config changes)
net stop cloudflared
net start cloudflared

# View tunnel logs (foreground mode — Ctrl+C to exit)
cloudflared tunnel run finance-dashboard

# Re-apply the ImagePath patch by hand
powershell -File scripts\fix_cloudflared_imagepath.ps1 -TunnelName finance-dashboard

# Tear down (rarely needed)
cloudflared service uninstall
cloudflared tunnel delete finance-dashboard
```

## Troubleshooting

### "process terminated unexpectedly" when starting the service
The `ImagePath` was overwritten with the cloudflared default (no args). Most
common cause: someone re-ran `cloudflared service install` directly (not via
the bat) or `winget upgrade Cloudflare.cloudflared` reinstalled the service.

Fix: re-run the bat from an elevated shell, or apply just the patch:

```cmd
powershell -File scripts\fix_cloudflared_imagepath.ps1 -TunnelName finance-dashboard
net stop cloudflared
net start cloudflared
```

### `cloudflared tunnel info` shows no active connections, but service is RUNNING
Service started with bad config, exited cloudflared.exe immediately. Re-check:
- `%SystemRoot%\System32\config\systemprofile\.cloudflared\config.yml` exists
  and matches `%USERPROFILE%\.cloudflared\config.yml`
- The credentials JSON is alongside the config in systemprofile
- Service `ImagePath` ends with `tunnel run finance-dashboard`:
  ```
  sc qc cloudflared
  ```

### 502 / 503 from `bets.raanman.lol` or `raanman.lol`
Tunnel is up but origin (Flask) is down. Check:

```cmd
curl http://localhost:5055/api/health
schtasks /query /tn PF-Dashboard
```

### 401 with valid token in URL
`config.json[dashboard_token]` doesn't match what you're sending. Re-grab:

```cmd
.venv\Scripts\python.exe -c "import json; print(json.load(open('config.json'))['dashboard_token'])"
```

### Apex resolves to old GH Pages IPs (`185.199.108-111.153`)
You didn't delete the old `A`/`AAAA` records at apex. Open Cloudflare DNS
panel for `raanman.lol`, delete them, then `cloudflared tunnel route dns
--overwrite-dns finance-dashboard raanman.lol` to re-create the apex CNAME.

### Verifier reports `cookie-only returns 200` PASS but browser still asks for token
Browser cookie store probably wiped (private window, manual clear, expired).
Visit the `?token=…` URL once to re-set the cookie.

## Security notes

- The dashboard exposes `/api/portfolio`, `/api/trades`, `/api/decisions`, and
  similar to anyone holding the token. Treat the token like a password.
- The token is HMAC-compared in `dashboard/app.py:require_auth`
  (timing-safe). Cookie storage is HttpOnly + Secure + SameSite=Lax.
- `dashboard/static/api-data/` is gitignored (was the original leak path).
  If `dashboard/export_static.py` ever runs, it writes there but Flask still
  serves those files with no auth — don't enable that script.
- For stronger auth (email-based SSO, no token in URL ever), consider
  [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/)
  on the Free plan (up to 50 users). Add a Self-hosted application gating
  `raanman.lol` to your email; Cloudflare prompts for a one-time email PIN
  before traffic reaches the tunnel.
