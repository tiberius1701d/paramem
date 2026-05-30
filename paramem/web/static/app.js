/**
 * ParaMem PWA — chat client.
 *
 * Targets the /chat endpoint (ChatRequest / ChatResponse shapes):
 *   POST /chat  { text: string, conversation_id: string }
 *               → { text: string, escalated?: boolean, speaker?: string, follow_up?: string }
 *
 * Auth: Authorization: Bearer <token>  (app.py:2352, auth.py:30)
 * Same-origin by default (StaticFiles at /app, API at same origin).
 *
 * Phase 4: voice capture — not implemented.
 * Phase 5: push notifications — not implemented.
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

  // Mic: stubbed — Phase 4
  micBtn.addEventListener("click", () => {
    appendMessage("system", "Voice is coming in a later update.");
  });

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
 * Toggles the in-flight state and disables/enables the send button.
 *
 * @param {boolean} state - true while a request is in progress.
 */
function setInFlight(state) {
  inFlight = state;
  sendBtn.disabled = state;
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
 */
function saveSettings() {
  serverUrl = serverUrlInput.value.trim();
  token = tokenInput.value.trim();
  localStorage.setItem(KEY_SERVER_URL, serverUrl);
  localStorage.setItem(KEY_TOKEN, token);
  closeSettings();
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
