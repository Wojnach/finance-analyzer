#!/usr/bin/env python3
"""Set up the /bets/ path on the GitHub Pages site as a redirect to the tunnel.

The architecture (see docs/TUNNEL_SETUP.md):
    Browser -> raanman.lol/bets  -> GitHub Pages (this script) -> redirect ->
    Browser -> bets.raanman.lol -> Cloudflare Tunnel -> localhost:5055

Why redirect instead of cloning dashboard/static/index.html into gh-pages
(2026-04-28 rewrite — see git history for the prior clone-based version):
  1. The cloned SPA has no `dashboard_token` baked in, so every API call from
     it returns 401. We'd be serving a permanently-broken page.
  2. The clone would also need dashboard/static/api-data/*.json copied to be
     even half-functional, exposing stale public snapshots — defeating the
     point of having a live tunnel as the source of truth.
  3. Single source of truth: one URL (bets.raanman.lol), one auth model.

We deliberately do NOT touch:
  - The apex index.html — the existing game lives there.
  - The CNAME file — already configured for raanman.lol.
Only /bets/index.html and .nojekyll are written.
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
