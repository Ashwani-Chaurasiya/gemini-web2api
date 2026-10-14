// Background Service Worker: Auto-syncs rotated Gemini cookies to local gemini-web2api server
const KEY_COOKIES = ["SID", "__Secure-1PSID", "__Secure-1PSIDTS", "SAPISID", "SIDCC", "__Secure-1PSIDCC"];
const PREFERRED_ORDER = [
  "SID", "HSID", "SSID", "APISID", "SAPISID",
  "__Secure-1PSID", "__Secure-3PSID",
  "__Secure-1PSIDTS", "__Secure-3PSIDTS",
  "__Secure-1PAPISID", "__Secure-3PAPISID",
  "SIDCC", "__Secure-1PSIDCC", "__Secure-3PSIDCC",
  "AEC", "NID"
];
const LOCAL_ENDPOINTS = ["http://127.0.0.1:8081/v1/cookies"];
let debounceTimer = null;

async function syncCookiesToLocalServer() {
  try {
    const [googleCookies, geminiCookies] = await Promise.all([
      chrome.cookies.getAll({ url: "https://google.com/" }),
      chrome.cookies.getAll({ url: "https://gemini.google.com/" })
    ]);
    const selected = new Map();
    for (const c of [...googleCookies, ...geminiCookies]) {
      if (!c || !c.name || typeof c.value !== "string" || !c.value.length) continue;
      const existing = selected.get(c.name);
      if (!existing || c.domain === "gemini.google.com") {
        selected.set(c.name, c);
      }
    }
    const sapisid = selected.get("SAPISID")?.value;
    const psid = selected.get("__Secure-1PSID")?.value;
    if (!sapisid || !psid) return;

    const ordered = [];
    const seen = new Set();
    for (const name of PREFERRED_ORDER) {
      if (selected.has(name)) {
        ordered.push(`${name}=${selected.get(name).value}`);
        seen.add(name);
      }
    }
    for (const [name, c] of selected.entries()) {
      if (!seen.has(name)) ordered.push(`${name}=${c.value}`);
    }

    const payload = {
      cookie: ordered.join("; "),
      sapisid
    };

    for (const url of LOCAL_ENDPOINTS) {
      try {
        await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
      } catch {
        // Local server may not be running; ignore.
      }
    }
    await chrome.storage.local.set({ lastCookieRotationAt: Date.now() });
  } catch {
    // Ignore background sync errors.
  }
}

chrome.runtime.onInstalled.addListener(() => {
  syncCookiesToLocalServer();
});

chrome.runtime.onStartup.addListener(() => {
  syncCookiesToLocalServer();
});

chrome.cookies.onChanged.addListener((changeInfo) => {
  if (changeInfo.removed) return;
  const { cookie } = changeInfo;
  const domain = (cookie.domain || "").replace(/^\./, "").toLowerCase();
  if (domain !== "google.com" && !domain.endsWith(".google.com")) return;
  if (!KEY_COOKIES.includes(cookie.name)) return;

  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    syncCookiesToLocalServer();
  }, 2000);
});
