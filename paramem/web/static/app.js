/**
 * ParaMem PWA — chat client.
 *
 * Targets the /chat endpoint (ChatRequest / ChatResponse shapes):
 *   POST /chat  { text: string, conversation_id: string }
 *               → { text: string, escalated?: boolean, speaker?: string, follow_up?: string }
 *
 * Voice path:
 *   POST /voice  body = raw audio Blob (audio/mp4 on iOS, audio/webm;codecs=opus on Android)
 *               headers: Authorization, Content-Type, x-conversation-id
 *               → { transcript: string, reply: string, audio?: string, audio_format?: string, follow_up?: string }
 *   Push-to-talk: pointerdown → getUserMedia → MediaRecorder start
 *                 pointerup/pointercancel → stop recorder → POST /voice → render
 *
 * Auth: Authorization: Bearer <token>  (app.py:2352, auth.py:30)
 * Same-origin by default (StaticFiles at /app, API at same origin).
 *
 * Phase 5: push notifications — register after auth.
 * registerPush() is called once credentials are confirmed (token present, SW active).
 */

// ---------------------------------------------------------------------------
// Storage keys
// ---------------------------------------------------------------------------
const KEY_SERVER_URL = "paramem_server_url";
const KEY_TOKEN = "paramem_token";
const KEY_CONV_ID = "paramem_conversation_id";
const CACHE_NAME = "paramem-shell-v1";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let serverUrl = "";
let token = "";
let conversationId = "";
let inFlight = false;
let scannerStream = null;
let barcodeDetector = null;

// Voice / push-to-talk state
let micRecorder = null;       // active MediaRecorder instance while recording
let micChunks = [];           // audio chunks collected during capture
let micStream = null;         // getUserMedia stream (stopped on release)
let micVoiceSupported = null; // null = unchecked, true/false after first attempt

// Push notification state
let pushRegistered = false;   // true once registerPush() has succeeded

// ---------------------------------------------------------------------------
// DOM refs (resolved after DOMContentLoaded)
// ---------------------------------------------------------------------------
let logEl, textInput, sendBtn, micBtn;
let settingsOverlay, settingsDrawer, serverUrlInput, tokenInput;
let saveBtn, cancelBtn, qrBtn;
let scannerOverlay, scannerVideo, scannerCancel;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/**
 * Entry point. Wires all DOM refs, processes deep-link fragment, loads
 * persisted credentials, and opens settings if no token is stored.
 */
function init() {
  logEl = document.getElementById("log");
  textInput = document.getElementById("text-input");
  sendBtn = document.getElementById("send-btn");
  micBtn = document.getElementById("mic-btn");
  settingsOverlay = document.getElementById("settings-overlay");
  serverUrlInput = document.getElementById("server-url-input");
  tokenInput = document.getElementById("token-input");
  saveBtn = document.getElementById("save-btn");
  cancelBtn = document.getElementById("cancel-btn");
  qrBtn = document.getElementById("qr-btn");
  scannerOverlay = document.getElementById("scanner-overlay");
  scannerVideo = document.getElementById("scanner-video");
  scannerCancel = document.getElementById("scanner-cancel");

  processFragment();
  loadCredentials();
  ensureConversationId();
  registerServiceWorker();
  wireEvents();
  // Attempt push registration after credentials are loaded.  Safe to call
  // before the SW is active — registerPush() waits for the SW to be ready.
  registerPush();

  if (!token) {
    openSettings();
    appendMessage("system", "Enter your bearer token in Settings to get started.");
  }
}

/**
 * Processes a #token= deep-link fragment if present.
 * Accepts two forms:
 *   #token=<t>            (token only)
 *   #token=<t>&url=<u>    (token + server URL)
 * Clears the fragment after reading to avoid leaking credentials in history.
 */
function processFragment() {
  const hash = location.hash.slice(1); // strip leading '#'
  if (!hash) return;

  const params = new URLSearchParams(hash);
  const fragToken = params.get("token");
  if (!fragToken) return;

  localStorage.setItem(KEY_TOKEN, fragToken);
  const fragUrl = params.get("url");
  if (fragUrl) {
    localStorage.setItem(KEY_SERVER_URL, fragUrl);
  }
  // Clear fragment without adding a history entry.
  history.replaceState(null, "", location.pathname + location.search);
}

/**
 * Loads server URL and token from localStorage into module-level state.
 */
function loadCredentials() {
  serverUrl = localStorage.getItem(KEY_SERVER_URL) || "";
  token = localStorage.getItem(KEY_TOKEN) || "";
}

/**
 * Generates a stable conversation_id if one does not already exist.
 */
function ensureConversationId() {
  let id = localStorage.getItem(KEY_CONV_ID);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(KEY_CONV_ID, id);
  }
  conversationId = id;
}

/**
 * Registers the service worker if the browser supports it.
 */
function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/app/sw.js").catch((err) => {
      console.warn("SW registration failed:", err);
    });
  }
}

/**
 * Requests permission and subscribes to Web Push if supported.
 *
 * Idempotent: returns immediately if already subscribed (pushRegistered).
 * Requires a token to be set — skips silently when no credentials are present.
 *
 * Flow:
 *   1. Feature-detect serviceWorker + PushManager.
 *   2. Request Notification.requestPermission().  Gracefully handles denied /
 *      unsupported (logs to console, does not throw).
 *   3. GET /push/vapid-public-key — if the server returns 503 (push disabled),
 *      silently skip; this allows the client to run against a server that has
 *      push_enabled=false without errors.
 *   4. registration.pushManager.subscribe({userVisibleOnly, applicationServerKey}).
 *   5. POST /push/subscribe with the subscription JSON.
 */
async function registerPush() {
  if (pushRegistered) return;
  if (!token) return; // no credentials yet

  if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
    console.info("Web Push not supported in this browser.");
    return;
  }

  // Request notification permission.  A denied result is not an error.
  let permission;
  try {
    permission = await Notification.requestPermission();
  } catch (err) {
    console.warn("Notification.requestPermission() failed:", err);
    return;
  }
  if (permission !== "granted") {
    console.info("Push notifications not permitted (status:", permission, ")");
    return;
  }

  // Fetch the VAPID public key.  503 means push is not enabled server-side.
  let vapidKey;
  try {
    const base = serverUrl || "";
    const resp = await fetch(`${base}/push/vapid-public-key`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (resp.status === 503) {
      // push_enabled=false on the server — silently skip.
      return;
    }
    if (!resp.ok) {
      console.warn("GET /push/vapid-public-key failed:", resp.status);
      return;
    }
    const data = await resp.json();
    vapidKey = data.key;
  } catch (err) {
    console.warn("Could not fetch VAPID public key:", err);
    return;
  }

  // Convert unpadded base64url string to Uint8Array for PushManager.subscribe.
  const keyBytes = _base64urlToUint8Array(vapidKey);

  let registration;
  try {
    registration = await navigator.serviceWorker.ready;
  } catch (err) {
    console.warn("SW not ready:", err);
    return;
  }

  let subscription;
  try {
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: keyBytes,
    });
  } catch (err) {
    console.warn("pushManager.subscribe() failed:", err);
    return;
  }

  // POST the subscription to the server.
  try {
    const base = serverUrl || "";
    const resp = await fetch(`${base}/push/subscribe`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(subscription.toJSON()),
    });
    if (!resp.ok) {
      console.warn("POST /push/subscribe failed:", resp.status);
      return;
    }
    pushRegistered = true;
    console.info("Push subscription registered.");
  } catch (err) {
    console.warn("Could not register push subscription:", err);
  }
}

/**
 * Converts an unpadded base64url string to a Uint8Array.
 * Required to pass the VAPID public key to PushManager.subscribe().
 *
 * @param {string} base64url - Unpadded base64url-encoded string.
 * @returns {Uint8Array}
 */
function _base64urlToUint8Array(base64url) {
  // Re-pad to a multiple of 4 for atob.
  const padded = base64url.replace(/-/g, "+").replace(/_/g, "/");
  const padding = (4 - (padded.length % 4)) % 4;
  const base64 = padded + "=".repeat(padding);
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

/**
 * Wires all interactive event listeners.
 */
function wireEvents() {
  document.getElementById("settings-btn").addEventListener("click", openSettings);
  saveBtn.addEventListener("click", saveSettings);
  cancelBtn.addEventListener("click", closeSettings);
  settingsOverlay.addEventListener("click", (e) => {
    if (e.target === settingsOverlay) closeSettings();
  });

  sendBtn.addEventListener("click", handleSend);
  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
  // Auto-grow textarea
  textInput.addEventListener("input", () => {
    textInput.style.height = "auto";
    textInput.style.height = textInput.scrollHeight + "px";
  });

  // Mic: push-to-talk — hold to record, release to send.
  wireMicButton();

  scannerCancel.addEventListener("click", stopScanner);

  // Show QR button only if BarcodeDetector is available
  if ("BarcodeDetector" in window) {
    qrBtn.style.display = "";
    qrBtn.addEventListener("click", startScanner);
  }
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/**
 * Reads the text input and sends a message to /chat.
 * Appends the user's message and the server's response to the log.
 * Handles HTTP 401 (opens settings) and network errors gracefully.
 */
async function handleSend() {
  const text = textInput.value.trim();
  if (!text || inFlight) return;

  appendMessage("user", text);
  textInput.value = "";
  textInput.style.height = "auto";

  setInFlight(true);
  const typingEl = appendTyping();

  try {
    const base = serverUrl || "";
    const resp = await fetch(`${base}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ text, conversation_id: conversationId }),
    });

    typingEl.remove();

    if (resp.status === 401) {
      appendMessage("error", "Authentication failed — re-enter your token in Settings.");
      openSettings();
      return;
    }

    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      appendMessage("error", `Server error ${resp.status}${body ? ": " + body.slice(0, 120) : ""}`);
      return;
    }

    const data = await resp.json();
    // ChatResponse shape: { text, escalated?, speaker?, follow_up? }
    // (app.py:210-214)
    appendMessage("assistant", data.text);
    if (data.follow_up) {
      appendMessage("assistant", data.follow_up);
    }
  } catch (err) {
    typingEl.remove();
    appendMessage("error", `Network error: ${err.message || String(err)}`);
  } finally {
    setInFlight(false);
  }
}

/**
 * Toggles the in-flight state and disables/enables interactive controls.
 * Disabling the mic button while a text request is in flight prevents
 * concurrent submissions; re-enabling restores it unless it was already
 * marked unsupported.
 *
 * @param {boolean} state - true while a request is in progress.
 */
function setInFlight(state) {
  inFlight = state;
  sendBtn.disabled = state;
  if (!micBtn.classList.contains("unsupported")) {
    micBtn.disabled = state;
  }
}

// ---------------------------------------------------------------------------
// Log helpers
// ---------------------------------------------------------------------------

/**
 * Appends a message bubble to the log and scrolls to the bottom.
 *
 * @param {"user"|"assistant"|"error"|"system"} role - CSS class applied to the bubble.
 * @param {string} text - Message text (displayed as-is, not interpreted as HTML).
 * @returns {HTMLElement} The created element.
 */
function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
  return el;
}

/**
 * Appends an animated typing indicator while a request is in flight.
 *
 * @returns {HTMLElement} The indicator element (caller removes it on completion).
 */
function appendTyping() {
  const el = document.createElement("div");
  el.className = "msg assistant typing";
  el.innerHTML = "<span></span><span></span><span></span>";
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
  return el;
}

// ---------------------------------------------------------------------------
// Settings drawer
// ---------------------------------------------------------------------------

/**
 * Opens the settings drawer and pre-fills fields from current state.
 */
function openSettings() {
  serverUrlInput.value = serverUrl;
  tokenInput.value = token;
  settingsOverlay.classList.add("open");
  tokenInput.focus();
}

/**
 * Closes the settings drawer without saving.
 */
function closeSettings() {
  settingsOverlay.classList.remove("open");
}

/**
 * Saves server URL and token from the settings fields to localStorage
 * and updates module-level state, then closes the drawer.
 * Re-attempts push registration so a newly-entered token can subscribe.
 */
function saveSettings() {
  serverUrl = serverUrlInput.value.trim();
  token = tokenInput.value.trim();
  localStorage.setItem(KEY_SERVER_URL, serverUrl);
  localStorage.setItem(KEY_TOKEN, token);
  closeSettings();
  // Reset push state so a new token triggers a fresh subscription attempt.
  pushRegistered = false;
  registerPush();
}

// ---------------------------------------------------------------------------
// Voice capture (push-to-talk, MediaRecorder → POST /voice)
// ---------------------------------------------------------------------------

/**
 * Wires push-to-talk events onto the mic button.
 *
 * Uses pointer events (covers both mouse and touch with a single listener
 * pair). pointerdown starts capture; pointerup / pointercancel stops it and
 * submits.  If getUserMedia or MediaRecorder is unavailable the button is
 * marked unsupported once and further presses are ignored.
 */
function wireMicButton() {
  micBtn.addEventListener("pointerdown", handleMicDown);
  micBtn.addEventListener("pointerup", handleMicUp);
  micBtn.addEventListener("pointercancel", handleMicUp);
}

/**
 * Starts audio capture when the mic button is pressed.
 * Requests the mic, creates a MediaRecorder, and begins collecting chunks.
 * Marks the button unsupported if getUserMedia / MediaRecorder is absent.
 *
 * @param {PointerEvent} e
 */
async function handleMicDown(e) {
  e.preventDefault(); // prevent ghost click on touch devices

  if (inFlight || micRecorder) return;

  // One-time capability check
  if (micVoiceSupported === false) return;
  if (
    typeof navigator.mediaDevices === "undefined" ||
    typeof MediaRecorder === "undefined"
  ) {
    micVoiceSupported = false;
    micBtn.classList.add("unsupported");
    micBtn.disabled = true;
    micBtn.setAttribute("title", "Voice not supported in this context");
    micBtn.setAttribute("aria-label", "Voice not supported in this context");
    appendMessage("system", "Voice input is not available in this browser or context.");
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  } catch (err) {
    micVoiceSupported = false;
    micBtn.classList.add("unsupported");
    micBtn.disabled = true;
    appendMessage("error", `Microphone access denied: ${err.message || String(err)}`);
    return;
  }

  micVoiceSupported = true;
  micChunks = [];

  micRecorder = new MediaRecorder(micStream);
  micRecorder.addEventListener("dataavailable", (ev) => {
    if (ev.data && ev.data.size > 0) micChunks.push(ev.data);
  });
  micRecorder.start();

  micBtn.classList.add("recording");
  micBtn.setAttribute("aria-label", "Recording… release to send");
}

/**
 * Stops audio capture when the mic button is released (or the pointer leaves).
 * Assembles the recorded Blob and POSTs it to /voice.
 * Does nothing if no recording is in progress.
 *
 * @param {PointerEvent} e
 */
async function handleMicUp(e) {
  e.preventDefault();

  if (!micRecorder) return;

  const recorder = micRecorder;
  const stream = micStream;
  micRecorder = null;
  micStream = null;

  micBtn.classList.remove("recording");
  micBtn.setAttribute("aria-label", "Hold to record voice input");

  // Stop recorder and await the final dataavailable + stop events.
  await new Promise((resolve) => {
    recorder.addEventListener("stop", resolve, { once: true });
    recorder.stop();
  });

  // Stop all mic tracks so the browser indicator clears.
  for (const track of stream.getTracks()) track.stop();

  const mimeType = recorder.mimeType || "audio/webm";
  const blob = new Blob(micChunks, { type: mimeType });
  micChunks = [];

  // Guard: too-short / empty captures (e.g. accidental tap)
  if (blob.size < 1000) {
    appendMessage("system", "Recording too short — hold the button while speaking.");
    return;
  }

  await submitVoice(blob, mimeType);
}

/**
 * POSTs a recorded audio Blob to POST /voice and renders the result.
 * Reuses the same token / serverUrl / conversationId as the text path.
 * On success renders the transcript as a user bubble and the reply as an
 * assistant bubble via appendMessage (same helper as handleSend).
 *
 * Error handling:
 *   401 → auth failed message + openSettings
 *   503 (stt_unavailable in body) → "voice isn't available right now"
 *   other non-2xx → readable server error
 *   network failure → readable network error
 *
 * @param {Blob} blob - The recorded audio data.
 * @param {string} mimeType - MIME type reported by MediaRecorder (e.g. "audio/mp4").
 */
async function submitVoice(blob, mimeType) {
  if (inFlight) return;
  setInFlight(true);
  const typingEl = appendTyping();

  try {
    const base = serverUrl || "";
    const resp = await fetch(`${base}/voice`, {
      method: "POST",
      headers: {
        "Content-Type": mimeType,
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
        "x-conversation-id": conversationId,
      },
      body: blob,
    });

    typingEl.remove();

    if (resp.status === 401) {
      appendMessage("error", "Authentication failed — re-enter your token in Settings.");
      openSettings();
      return;
    }

    if (resp.status === 503) {
      appendMessage("system", "Voice isn't available right now — try text instead.");
      return;
    }

    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      appendMessage("error", `Server error ${resp.status}${body ? ": " + body.slice(0, 120) : ""}`);
      return;
    }

    const data = await resp.json();
    // VoiceResponse shape: { transcript, reply, audio?, audio_format?, follow_up? }
    if (data.transcript) {
      appendMessage("user", data.transcript);
    }
    if (data.reply) {
      appendMessage("assistant", data.reply);
    }
    if (data.follow_up) {
      appendMessage("assistant", data.follow_up);
    }
    // Play synthesized audio when the server provided a WAV payload.
    // The play() call sits inside the user-gesture chain (pointer-up → submitVoice)
    // so autoplay is permitted on iOS/Android.  If play() rejects (e.g. the page
    // was backgrounded) we swallow the error — the text reply is already rendered.
    if (data.audio) {
      try {
        const wavBytes = Uint8Array.from(atob(data.audio), c => c.charCodeAt(0));
        const wavBlob = new Blob([wavBytes], { type: "audio/wav" });
        const url = URL.createObjectURL(wavBlob);
        const audioEl = new Audio(url);
        audioEl.addEventListener("ended", () => URL.revokeObjectURL(url), { once: true });
        audioEl.play().catch(() => URL.revokeObjectURL(url));
      } catch (_e) {
        // Audio playback is best-effort; text reply already shown above.
      }
    }
  } catch (err) {
    typingEl.remove();
    appendMessage("error", `Network error: ${err.message || String(err)}`);
  } finally {
    setInFlight(false);
  }
}

// ---------------------------------------------------------------------------
// QR scanner (BarcodeDetector API — only shown when available)
// ---------------------------------------------------------------------------

/**
 * Opens a camera stream and starts scanning for a QR code using the
 * BarcodeDetector API.  Only called when `'BarcodeDetector' in window`.
 *
 * Accepts two QR payload formats:
 *   1. JSON: {"server_url": "...", "token": "..."}  (mint-user-token CLI output)
 *   2. URL with #token= fragment
 */
async function startScanner() {
  try {
    scannerStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
    });
    scannerVideo.srcObject = scannerStream;
    scannerOverlay.classList.add("open");

    barcodeDetector = new BarcodeDetector({ formats: ["qr_code"] });
    scanQrFrame();
  } catch (err) {
    appendMessage("error", `Camera access denied: ${err.message || String(err)}`);
  }
}

/**
 * Scans a single video frame for a QR code, then schedules the next frame.
 * Stops automatically once a valid onboarding QR is detected.
 */
function scanQrFrame() {
  if (!scannerStream) return;

  barcodeDetector
    .detect(scannerVideo)
    .then((barcodes) => {
      for (const bc of barcodes) {
        if (applyQrPayload(bc.rawValue)) {
          stopScanner();
          closeSettings();
          return;
        }
      }
      requestAnimationFrame(scanQrFrame);
    })
    .catch(() => {
      if (scannerStream) requestAnimationFrame(scanQrFrame);
    });
}

/**
 * Parses a QR raw value and applies server URL + token if valid.
 *
 * @param {string} raw - The raw QR code string.
 * @returns {boolean} true if a valid onboarding payload was found and applied.
 */
function applyQrPayload(raw) {
  // Try JSON payload: {"server_url": "...", "token": "..."}
  try {
    const obj = JSON.parse(raw);
    if (obj && typeof obj.token === "string") {
      token = obj.token;
      if (typeof obj.server_url === "string") serverUrl = obj.server_url;
      localStorage.setItem(KEY_TOKEN, token);
      localStorage.setItem(KEY_SERVER_URL, serverUrl);
      appendMessage("system", "Onboarding QR scanned. Token saved.");
      return true;
    }
  } catch (_) {
    // Not JSON — fall through.
  }

  // Try URL with #token= fragment
  try {
    const url = new URL(raw);
    const params = new URLSearchParams(url.hash.slice(1));
    const t = params.get("token");
    if (t) {
      token = t;
      const u = params.get("url");
      if (u) serverUrl = u;
      localStorage.setItem(KEY_TOKEN, token);
      localStorage.setItem(KEY_SERVER_URL, serverUrl);
      appendMessage("system", "Onboarding QR scanned. Token saved.");
      return true;
    }
  } catch (_) {
    // Not a URL — ignore.
  }

  return false;
}

/**
 * Stops the camera stream and closes the scanner overlay.
 */
function stopScanner() {
  if (scannerStream) {
    for (const track of scannerStream.getTracks()) track.stop();
    scannerStream = null;
  }
  scannerVideo.srcObject = null;
  scannerOverlay.classList.remove("open");
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
