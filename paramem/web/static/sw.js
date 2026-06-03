/**
 * ParaMem PWA — service worker.
 *
 * Strategy:
 *   - Shell assets (/app/, /app/index.html, /app/app.js, /app/manifest.json,
 *     /app/icon-192.png, /app/icon-512.png): cache-first (precached on install).
 *   - API paths (/chat, /voice, /status, and any non-shell request): network-only.
 *
 * Cache versioning: bump CACHE_VERSION to force a full shell refresh on next
 * activation (old caches are deleted in the activate handler).
 *
 * Phase 5: Web Push — the push event handler shows a generic notification badge;
 * no personal content passes through the push relay.  The notificationclick
 * handler focuses or opens the PWA so the user can fetch real content.
 */

const CACHE_VERSION = "v7";
const CACHE_NAME = `paramem-shell-${CACHE_VERSION}`;

/** Shell assets to precache on install. */
const SHELL_ASSETS = [
  "/app/",
  "/app/index.html",
  "/app/app.js",
  "/app/manifest.json",
  "/app/icon-192.png",
  "/app/icon-512.png",
];

/**
 * URL prefixes that must always go to the network (API calls).
 * Any request whose pathname starts with one of these bypasses the cache.
 */
const NETWORK_ONLY_PREFIXES = [
  "/chat",
  "/voice",
  "/status",
  "/consolidate",
  "/gpu",
  "/debug",
  "/backup",
];

// ---------------------------------------------------------------------------
// Install — precache shell assets
// ---------------------------------------------------------------------------

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// ---------------------------------------------------------------------------
// Activate — clean up old caches
// ---------------------------------------------------------------------------

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k !== CACHE_NAME)
            .map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

// ---------------------------------------------------------------------------
// Fetch — cache-first for shell, network-only for API
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Push — show a generic notification badge; no personal content in the relay.
// ---------------------------------------------------------------------------

/**
 * Handles an incoming push message.
 *
 * The push payload is intentionally empty (contentless ping posture).
 * A generic title and body are shown so the user knows there is a new
 * message; real content is fetched when the user taps the notification.
 * The notification carries a data object so the notificationclick handler
 * can open the correct URL.
 */
self.addEventListener("push", (event) => {
  const title = "ParaMem";
  const options = {
    body: "New message",
    icon: "/app/icon-192.png",
    badge: "/app/icon-192.png",
    data: { url: "/app/" },
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

/**
 * Handles a tap on a displayed notification.
 *
 * Focuses an existing PWA window if one is open; otherwise opens a new one.
 * Closes the notification after handling the click.
 */
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const targetUrl = (event.notification.data && event.notification.data.url) || "/app/";
  event.waitUntil(
    clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((windowClients) => {
        for (const client of windowClients) {
          if (client.url.includes("/app") && "focus" in client) {
            return client.focus();
          }
        }
        return clients.openWindow(targetUrl);
      })
  );
});

// ---------------------------------------------------------------------------
// Fetch — cache-first for shell, network-only for API
// ---------------------------------------------------------------------------

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Only handle same-origin requests.
  if (url.origin !== self.location.origin) return;

  // API paths — always go to the network.
  if (NETWORK_ONLY_PREFIXES.some((p) => url.pathname.startsWith(p))) {
    // network-only: let the browser handle it (no event.respondWith).
    return;
  }

  // Non-GET requests — pass through.
  if (event.request.method !== "GET") return;

  // Shell assets — cache-first, fall back to network.
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        // Only cache successful, non-opaque responses for shell assets.
        if (
          response &&
          response.status === 200 &&
          response.type === "basic"
        ) {
          const clone = response.clone();
          caches
            .open(CACHE_NAME)
            .then((cache) => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});
