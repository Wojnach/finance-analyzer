/*
 * sw.js — service worker for the Portfolio Intelligence mobile dashboard.
 *
 * Strategy (per Track-6 decisions):
 *   - Static shell (HTML / CSS / JS / icons / manifest) → cache-first.
 *   - Chart.js CDN → cache-first.
 *   - /api/* → network-first; never cached. Auth-required path; serving a
 *              stale 200 here would mask a missing CF Access cookie.
 *   - Navigation requests → network-first with cached HTML fallback so the
 *     phone shows the shell when offline (rather than Safari's error page).
 *
 * Versioned cache name flushes old shells on deploy. skipWaiting +
 * clients.claim activates the new SW immediately.
 */

const VERSION = "v3-2026-07-18";
const SHELL_CACHE = `pi-shell-${VERSION}`;

const SHELL_ASSETS = [
  "/",
  "/static/css/tokens.css",
  "/static/css/base.css",
  "/static/css/layout.css",
  "/static/css/components.css",
  "/static/css/responsive.css",
  "/static/js/main.js",
  "/static/js/state.js",
  "/static/js/fetch.js",
  "/static/js/format.js",
  "/static/js/theme.js",
  "/static/js/router.js",
  "/static/js/polling.js",
  "/static/manifest.webmanifest",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/apple-touch-icon-180.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(SHELL_CACHE);
      // Best-effort precache — individual failures are tolerated.
      await Promise.allSettled(
        SHELL_ASSETS.map((url) =>
          cache
            .add(new Request(url, { credentials: "same-origin" }))
            .catch((e) => console.warn("sw: precache failed", url, e)),
        ),
      );
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      // Drop old caches.
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((k) => k.startsWith("pi-shell-") && k !== SHELL_CACHE)
          .map((k) => caches.delete(k)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (
    url.origin !== location.origin &&
    !url.host.includes("cdn.jsdelivr.net")
  ) {
    return;
  }

  // /api/* — network-first, never cached. Auth-bearing — letting a cached
  // 200 reply through after the cookie expires would silently mask the
  // re-auth prompt.
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(_networkOnly(req));
    return;
  }

  // Chart.js CDN — cache-first, long-lived.
  if (
    url.host.includes("cdn.jsdelivr.net") &&
    url.pathname.includes("chart.js")
  ) {
    event.respondWith(_cacheFirst(req, SHELL_CACHE));
    return;
  }

  // Navigation requests — network-first, cached HTML fallback.
  if (req.mode === "navigate") {
    event.respondWith(_networkFirstWithCachedShell(req));
    return;
  }

  // All other static assets — cache-first.
  if (url.origin === location.origin) {
    event.respondWith(_cacheFirst(req, SHELL_CACHE));
  }
});

async function _networkOnly(req) {
  try {
    return await fetch(req);
  } catch (_) {
    return new Response(JSON.stringify({ error: "offline" }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }
}

async function _cacheFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(req);
  if (hit) return hit;
  try {
    const resp = await fetch(req);
    if (resp.ok) cache.put(req, resp.clone());
    return resp;
  } catch (e) {
    console.warn("sw: cache-first fetch failed", req.url, e);
    return new Response("", { status: 504 });
  }
}

async function _networkFirstWithCachedShell(req) {
  try {
    const resp = await fetch(req);
    if (resp.ok) {
      const cache = await caches.open(SHELL_CACHE);
      cache.put(req, resp.clone());
    }
    return resp;
  } catch (_) {
    const cache = await caches.open(SHELL_CACHE);
    const hit = await cache.match("/");
    if (hit) return hit;
    return new Response(
      "<h1>Offline</h1><p>The dashboard is offline. Reconnect and reload.</p>",
      { headers: { "Content-Type": "text/html" } },
    );
  }
}
