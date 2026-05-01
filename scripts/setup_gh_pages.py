#!/usr/bin/env python3
"""DEPRECATED 2026-04-30 — architecture pivoted to apex-direct, no GH Pages.

This script was designed for a path-based architecture:
    Browser -> raanman.lol/bets -> GitHub Pages -> redirect -> bets.raanman.lol

The deployment instead pivoted to apex-direct:
    Browser -> raanman.lol -> Cloudflare tunnel -> http://localhost:5055

The apex GH Pages records (4 A + 4 AAAA + www CNAME + wildcard CNAME) were
deleted from Cloudflare DNS, so GitHub Pages no longer serves any content
under raanman.lol. Running this script today writes files to a gh-pages
worktree that nobody reads — the redirect would never be reached.

DO NOT RUN. Kept in the repo only as a paper trail of the original design;
delete in a future cleanup pass once it is clear the apex-direct path is
permanent.

If you ever revive GH Pages serving (e.g., to add a personal homepage at
the apex), the redirect logic here is still correct — but you'd also need
to re-add the GH Pages DNS records and reroute the tunnel to a subdomain.
See docs/TUNNEL_SETUP.md for the live architecture.
"""

from pathlib import Path

GH_PAGES_DIR = Path("Q:/wt/gh-pages")
BETS_DIR = GH_PAGES_DIR / "bets"
TARGET = "https://bets.raanman.lol"

BETS_DIR.mkdir(parents=True, exist_ok=True)

# Belt-and-braces redirect: meta-refresh as the no-JS fallback (search bots,
# privacy modes), location.replace() for users with JS so we can forward any
# ?token=... query string and #hash that was on the original URL — meta-refresh
# can't preserve those. .replace() (not .assign()) keeps the redirect out of
# back-button history so users don't bounce-loop.
redirect_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Redirecting to dashboard…</title>
<meta http-equiv="refresh" content="0;url={TARGET}/">
<meta name="robots" content="noindex">
<script>
  var qs = window.location.search || "";
  var h  = window.location.hash   || "";
  window.location.replace("{TARGET}/" + qs + h);
</script>
</head>
<body>
<p>Redirecting to <a href="{TARGET}/">{TARGET}</a>…</p>
</body>
</html>
"""

bets_index = BETS_DIR / "index.html"
bets_index.write_text(redirect_html, encoding="utf-8")

# .nojekyll prevents GH Pages' Jekyll layer from mangling files.
# Idempotent — safe to re-run.
(GH_PAGES_DIR / ".nojekyll").touch()

print(f"Wrote redirect: {bets_index} -> {TARGET}/")
print(f"Touched:        {GH_PAGES_DIR / '.nojekyll'}")
print()
print("NOT touched (preserved): apex index.html, CNAME.")
print()
print("Next:")
print("  cd /q/wt/gh-pages")
print("  cmd.exe /c \"git status\"")
print("  cmd.exe /c \"git add bets/index.html .nojekyll && git commit -m 'gh-pages: redirect /bets/ to tunnel'\"")
print("  cmd.exe /c \"git push origin gh-pages\"")
print()
print("If this gh-pages worktree previously had bets/api-data/*.json or a")
print("cloned dashboard/static/index.html copy, delete those manually before")
print("committing — they were the public-stale-snapshot leak we're fixing.")
