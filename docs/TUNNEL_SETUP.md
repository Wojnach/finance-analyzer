# Cloudflare Tunnel Setup — Dashboard at bets.raanman.lol

Exposes the finance-analyzer dashboard (Flask, port 5055) to the internet via Cloudflare Tunnel.

## Architecture

```
Browser → bets.raanman.lol → Cloudflare Edge → Tunnel → localhost:5055 (herc)
Browser → raanman.lol/bets → GitHub Pages → redirect → bets.raanman.lol
```

## Prerequisites

- Cloudflare account (free plan works)
- The finance-analyzer dashboard running on port 5055
- Windows machine with internet access

## Step 1: Add raanman.lol to Cloudflare (~10 min, one-time)

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com)
2. Click **Add a site** → enter `raanman.lol`
3. Select the **Free** plan
4. Cloudflare will scan existing DNS records. Verify it found:
   - `CNAME www → Wojnach.github.io` (for GitHub Pages)
   - `A` records for the apex domain (if any)
5. Cloudflare gives you two nameservers (e.g., `ada.ns.cloudflare.com`, `bob.ns.cloudflare.com`)
6. Go to your domain registrar and update nameservers to the ones Cloudflare provided
7. Wait for propagation (usually < 30 min, can take up to 24h)
8. Verify: `raanman.lol` still loads the game via GitHub Pages

### DNS Records After Setup

| Type  | Name  | Target                    | Proxy |
|-------|-------|---------------------------|-------|
| CNAME | www   | Wojnach.github.io        | ✅    |
| CNAME | bets  | (auto-created by tunnel)  | ✅    |

## Step 2: Run the Setup Script (~5 min, one-time)

```cmd
cd Q:\CaludesRoom\finance-analyzer
scripts\setup_tunnel.bat
```

The script will:
1. Install `cloudflared` via winget (if not present)
2. Open a browser for Cloudflare authentication — **select raanman.lol**
3. Create a tunnel named `finance-dashboard`
4. Write the config to `%USERPROFILE%\.cloudflared\config.yml`
5. Create a DNS CNAME record for `bets.raanman.lol`
6. Install cloudflared as a Windows service (auto-starts on boot)

## Step 3: Verify

- Open `https://bets.raanman.lol` → live dashboard
- Open `https://raanman.lol/bets` → redirects to above
- Check all 8 tabs load data correctly
- Check auto-refresh works (real-time, not static)

## Management Commands

```cmd
# Check tunnel status
cloudflared tunnel info finance-dashboard

# View tunnel logs
cloudflared tunnel run finance-dashboard

# Restart the service
net stop cloudflared && net start cloudflared

# Check service status
sc query cloudflared

# Delete tunnel (if needed)
cloudflared service uninstall
cloudflared tunnel delete finance-dashboard
```

## Troubleshooting

### Dashboard not loading
1. Ensure the Flask dashboard is running: `curl http://localhost:5055/api/health`
2. Ensure cloudflared service is running: `sc query cloudflared`
3. Check Cloudflare DNS: `nslookup bets.raanman.lol` should return Cloudflare IPs

### "502 Bad Gateway" error
The tunnel is working but can't reach the Flask app.
- Check if the dashboard is running on port 5055
- Check `%USERPROFILE%\.cloudflared\config.yml` has `service: http://localhost:5055`

### raanman.lol stopped working after DNS change
Cloudflare DNS propagation can take time. Ensure the CNAME record for `www` still points to `Wojnach.github.io` with proxy enabled (orange cloud).

### SSL certificate errors
Cloudflare handles SSL automatically. In Cloudflare dashboard → SSL/TLS → set mode to **Full** (not Full Strict, since localhost is HTTP).

## Security Notes

- The dashboard is publicly accessible via the tunnel URL
- If `dashboard_token` is configured in `config.json`, visitors need `?token=XXX`
- For additional protection, consider [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/) (free for up to 50 users)
