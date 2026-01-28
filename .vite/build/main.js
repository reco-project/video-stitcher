import { dialog as p, app as d, BrowserWindow as k, ipcMain as m, shell as N } from "electron";
import { join as h, dirname as de } from "node:path";
import { existsSync as _, writeFileSync as R, unlinkSync as ue, readFileSync as $, mkdirSync as he, appendFileSync as fe, rmSync as ge, statSync as pe } from "node:fs";
import { fileURLToPath as me } from "url";
import { spawn as ee } from "node:child_process";
import L, { platform as q } from "node:os";
import z from "node:crypto";
const B = 1, we = /* @__PURE__ */ new Set([
  "app_open",
  "match_created",
  "processing_start",
  "processing_cancel",
  "processing_success",
  "processing_error",
  "context"
]);
function V(t) {
  he(S(t), { recursive: !0 });
}
function S(t) {
  return h(t.getPath("userData"), "telemetry");
}
function F(t) {
  return h(S(t), "events.jsonl");
}
function te(t) {
  return h(S(t), "client_id.txt");
}
function U(t) {
  V(t);
  const e = te(t);
  try {
    if (_(e)) {
      const r = $(e, "utf8").trim();
      if (r) return r;
    }
  } catch {
  }
  const n = z.randomUUID();
  try {
    R(e, `${n}
`, "utf8");
  } catch {
  }
  return n;
}
async function ne(t) {
  const e = L.cpus() || [], n = e[0]?.model || null, r = e.length || null, o = Math.round(L.totalmem() / 1024 ** 3);
  let a = null;
  const s = (l) => {
    if (!l || typeof l != "object") return null;
    const c = l.deviceString || l.description || l.name || null, u = typeof l.vendorId == "number" ? l.vendorId : typeof l.vendor_id == "number" ? l.vendor_id : null, g = typeof l.deviceId == "number" ? l.deviceId : typeof l.device_id == "number" ? l.device_id : null, b = typeof l.active == "boolean" ? l.active : null;
    return c ? { name: c, active: b } : u !== null || g !== null ? {
      vendor_id: u,
      device_id: g,
      active: b
    } : null;
  }, i = (l) => {
    const c = l?.gpuDevice ?? l?.gpuDevices ?? l?.devices ?? null;
    if (Array.isArray(c)) {
      const u = c.map(s).filter(Boolean);
      return u.length ? u : null;
    }
    if (c && typeof c == "object") {
      const u = s(c);
      return u ? [u] : null;
    }
    return null;
  };
  try {
    const l = await t.getGPUInfo("basic");
    if (a = i(l), !a || a.length === 0) {
      const c = await t.getGPUInfo("complete");
      a = i(c);
    }
    a && a.length === 0 && (a = null);
  } catch {
  }
  return {
    os: {
      platform: process.platform,
      release: L.release(),
      arch: process.arch
    },
    hardware: {
      cpu_model: n,
      cpu_threads: r,
      ram_gb: Number.isFinite(o) ? o : null,
      gpus: a
    }
  };
}
async function ye(t) {
  const e = await ne(t), n = JSON.stringify(e);
  return z.createHash("sha256").update(n).digest("hex").slice(0, 16);
}
function Y(t, e) {
  V(t), fe(F(t), `${JSON.stringify(e)}
`, "utf8");
}
function ve(t) {
  if (!t || typeof t != "object" || Array.isArray(t)) return null;
  try {
    const e = JSON.stringify(t);
    return e.length > 2048 ? null : JSON.parse(e);
  } catch {
    return null;
  }
}
function _e(t) {
  return we.has(t);
}
function ke({ ipcMain: t, app: e, shell: n }) {
  let r = !1;
  t.handle("telemetry:getInfo", async () => ({
    schema_version: B,
    telemetry_dir: S(e),
    events_path: F(e),
    client_id: U(e)
  })), t.handle("telemetry:openFolder", async () => {
    try {
      return V(e), await n.openPath(S(e)), !0;
    } catch {
      return !1;
    }
  }), t.handle("telemetry:deleteLocal", async () => {
    try {
      const o = F(e);
      return _(o) && R(o, "", "utf8"), { ok: !0 };
    } catch (o) {
      return { ok: !1, error: o?.message || "Failed to delete telemetry data" };
    }
  }), t.handle("telemetry:resetClientId", async () => {
    try {
      const o = te(e);
      return _(o) && ue(o), { ok: !0, client_id: U(e) };
    } catch (o) {
      return { ok: !1, error: o?.message || "Failed to reset client ID" };
    }
  }), t.handle("telemetry:track", async (o, a) => {
    try {
      const s = typeof a?.name == "string" ? a.name.trim() : "";
      if (!s || s.length > 64 || !_e(s)) return !1;
      const i = a?.include_system_info === !0, l = ve(a?.props);
      if (i && !r) {
        r = !0;
        const c = await ne(e);
        Y(e, {
          schema_version: B,
          ts: (/* @__PURE__ */ new Date()).toISOString(),
          name: "context",
          client_id: U(e),
          app: { version: e.getVersion() },
          ...c
        });
      }
      return Y(e, {
        schema_version: B,
        ts: (/* @__PURE__ */ new Date()).toISOString(),
        name: s,
        client_id: U(e),
        props: l
      }), !0;
    } catch {
      return !1;
    }
  });
}
const be = 200, G = globalThis.fetch;
function re(t) {
  return h(S(t), "upload_state.json");
}
function Se(t) {
  const e = re(t);
  try {
    if (!_(e)) return { last_uploaded_line: 0, last_success_at: null, last_hardware_hash: null };
    const n = $(e, "utf8"), r = JSON.parse(n);
    return {
      last_uploaded_line: Number.isFinite(r?.last_uploaded_line) ? r.last_uploaded_line : 0,
      last_success_at: typeof r?.last_success_at == "string" ? r.last_success_at : null,
      last_hardware_hash: typeof r?.last_hardware_hash == "string" ? r.last_hardware_hash : null
    };
  } catch {
    return { last_uploaded_line: 0, last_success_at: null, last_hardware_hash: null };
  }
}
function Ee(t, e) {
  const n = re(t);
  try {
    R(n, `${JSON.stringify(e, null, 2)}
`, "utf8");
  } catch {
  }
}
function Ue(t) {
  if (typeof t != "string") return null;
  const e = t.trim().replace(/\/+$/, "");
  return !e || !/^https?:\/\//i.test(e) ? null : e;
}
function Pe(t) {
  const e = F(t);
  if (!_(e)) return [];
  try {
    return $(e, "utf8").split(`
`).map((r) => r.trim()).filter(Boolean);
  } catch {
    return [];
  }
}
async function T({ app: t, endpointUrl: e }) {
  const n = Ue(e);
  if (!n)
    return { ok: !1, error: "Invalid endpoint URL (must start with http(s)://)" };
  const r = Pe(t), o = Se(t);
  let a = o.last_uploaded_line;
  a > r.length && (a = 0);
  const s = r.slice(a, a + be), i = [];
  for (const g of s)
    try {
      i.push(JSON.parse(g));
    } catch {
    }
  if (s.length === 0)
    return { ok: !0, sent: 0, remaining_lines: 0 };
  const l = await ye(t), c = o.last_hardware_hash !== l, u = {
    schema_version: B,
    client_id: U(t),
    app: {
      name: "video-stitcher",
      version: t.getVersion(),
      environment: t.isPackaged ? "production" : "dev"
    },
    sent_at: (/* @__PURE__ */ new Date()).toISOString(),
    batch_id: z.randomUUID(),
    hardware_changed: c,
    events: i
  };
  try {
    if (typeof G != "function")
      return { ok: !1, error: "fetch() is not available in this runtime" };
    const g = 3;
    let b = null;
    for (let P = 0; P < g; P++) {
      P > 0 && await new Promise((y) => setTimeout(y, Math.pow(2, P - 1) * 1e3));
      try {
        const y = await G(n, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(u)
        });
        if (y.status === 429)
          return console.warn("[TELEMETRY] Rate limited (429)."), { ok: !1, status: 429, error: await y.text().catch(() => "Rate limited") };
        if (!y.ok) {
          b = { ok: !1, status: y.status, error: await y.text().catch(() => y.statusText) };
          continue;
        }
        const C = a + s.length;
        return Ee(t, {
          last_uploaded_line: C,
          last_success_at: (/* @__PURE__ */ new Date()).toISOString(),
          last_hardware_hash: l
        }), console.log(`[TELEMETRY] Uploaded ${s.length} event(s). Remaining: ${Math.max(0, r.length - C)}`), { ok: !0, sent: s.length, remaining_lines: Math.max(0, r.length - C) };
      } catch (y) {
        b = { ok: !1, error: y?.message || "Network error" };
      }
    }
    return b || { ok: !1, error: "Upload failed after retries" };
  } catch (g) {
    return { ok: !1, error: g?.message || "Network error" };
  }
}
function Te({ ipcMain: t, app: e }) {
  let n = null;
  t.handle("telemetry:uploadNow", async (o, a) => T({ app: e, endpointUrl: a?.endpointUrl }));
  const r = () => {
    if (n) return;
    setTimeout(async () => {
      const { readSettings: a } = await Promise.resolve().then(() => O), s = a(e);
      if (s.telemetryEnabled && s.telemetryEndpointUrl) {
        console.log("[TELEMETRY] Immediate upload on app start...");
        try {
          await T({ app: e, endpointUrl: s.telemetryEndpointUrl });
        } catch (i) {
          console.warn("[TELEMETRY] Immediate upload failed:", i?.message);
        }
      }
    }, 2e3);
    const o = e.isPackaged ? 300 * 1e3 : 30 * 1e3;
    n = setInterval(async () => {
      const { readSettings: a } = await Promise.resolve().then(() => O), s = a(e);
      if (!(!s.telemetryEnabled || !s.telemetryAutoUpload || !s.telemetryEndpointUrl)) {
        console.log("[TELEMETRY] Periodic upload triggered...");
        try {
          await T({ app: e, endpointUrl: s.telemetryEndpointUrl });
        } catch (i) {
          console.warn("[TELEMETRY] Periodic upload failed:", i?.message);
        }
      }
    }, o);
  };
  e.isReady() ? r() : e.once("ready", r), e.on("before-quit", async (o) => {
    const { readSettings: a } = await Promise.resolve().then(() => O), s = a(e);
    if (!s.telemetryEnabled || !s.telemetryAutoUpload || !s.telemetryEndpointUrl) return;
    o.preventDefault(), console.log("[TELEMETRY] Upload on quit (3s timeout)...");
    const i = T({ app: e, endpointUrl: s.telemetryEndpointUrl }), l = new Promise((c) => setTimeout(c, 3e3));
    try {
      await Promise.race([i, l]);
    } catch (c) {
      console.warn("[TELEMETRY] Upload on quit failed:", c?.message);
    } finally {
      n && clearInterval(n), e.exit();
    }
  });
}
const A = {
  debugMode: !0,
  apiBaseUrl: "http://127.0.0.1:8000/api",
  encoderPreference: "auto",
  disableHardwareAcceleration: !1,
  telemetryEnabled: !1,
  telemetryIncludeSystemInfo: !1,
  telemetryEndpointUrl: "https://telemetry.reco-project.org/telemetry",
  telemetryAutoUpload: !1,
  telemetryPromptShown: !1
};
function oe(t) {
  return h(t.getPath("userData"), "settings.json");
}
function E(t) {
  const e = oe(t);
  try {
    if (!_(e)) return A;
    const n = $(e, "utf8"), r = JSON.parse(n);
    return { ...A, ...r };
  } catch {
    return A;
  }
}
function ae(t, e) {
  const n = oe(t);
  try {
    const r = { ...A, ...e };
    return R(n, JSON.stringify(r, null, 2), "utf8"), { ok: !0 };
  } catch (r) {
    return { ok: !1, error: r?.message || "Failed to write settings" };
  }
}
function se({ ipcMain: t, app: e, shell: n }) {
  t.handle("settings:read", async () => E(e)), t.handle("settings:write", async (r, o) => ae(e, o)), t.handle("settings:openUserDataFolder", async () => {
    try {
      const r = e.getPath("userData");
      return await n.openPath(r), { ok: !0 };
    } catch (r) {
      return { ok: !1, error: r?.message || "Failed to open folder" };
    }
  }), t.handle("settings:clearUserDataFolder", async (r) => {
    try {
      const o = e.getPath("userData");
      if ((await p.showMessageBox({
        type: "warning",
        buttons: ["Cancel", "Delete Everything"],
        defaultId: 0,
        cancelId: 0,
        title: "Clear All User Data",
        message: "This will permanently delete ALL data",
        detail: `This includes:
• All matches and videos
• All settings and preferences
• All telemetry data
• All logs and temporary files

The application will quit after deletion. This cannot be undone.

Are you absolutely sure?`
      })).response !== 1)
        return { ok: !1, cancelled: !0 };
      console.log("[Settings] Clearing user data folder:", o);
      try {
        const { readdirSync: s, statSync: i } = await import("node:fs"), c = s(o).filter(
          (u) => !u.startsWith(".") && u !== "Crashpad" && u !== "GPUCache"
          // Keep GPU cache
        );
        for (const u of c) {
          const g = h(o, u);
          console.log("[Settings] Deleting:", g), ge(g, { recursive: !0, force: !0 });
        }
        console.log("[Settings] User data cleared successfully");
      } catch (s) {
        return console.error("[Settings] Error during deletion:", s), { ok: !1, error: s.message };
      }
      return setTimeout(() => {
        e.quit();
      }, 500), { ok: !0 };
    } catch (o) {
      return console.error("[Settings] Error clearing user data:", o), { ok: !1, error: o?.message || "Failed to clear user data" };
    }
  }), t.handle("settings:getEncoderInfo", async () => {
    try {
      const r = E(e), o = r.apiBaseUrl || "http://127.0.0.1:8000/api", a = await fetch(`${o}/settings/encoders`);
      if (!a.ok)
        throw new Error("Failed to get encoder info from backend");
      const s = await a.json();
      return {
        ok: !0,
        current_encoder: r.encoderPreference || "auto",
        available_encoders: s.available_encoders || ["auto", "libx264"],
        encoder_descriptions: s.encoder_descriptions || {}
      };
    } catch (r) {
      return { ok: !1, error: r?.message || "Failed to get encoder info" };
    }
  });
}
const O = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  readSettings: E,
  registerSettingsIpc: se,
  writeSettings: ae
}, Symbol.toStringTag, { value: "Module" }));
let f = null, J = !1, Ie = !1, v = null, K = null;
try {
  const t = await import("electron-updater");
  console.log("[Updater] electron-updater loaded, pkg keys:", Object.keys(t)), f = t.autoUpdater || t.default?.autoUpdater, console.log("[Updater] autoUpdater:", f ? "found" : "not found"), f && (J = !0, f.setFeedURL({
    provider: "github",
    owner: "reco-project",
    repo: "video-stitcher"
  }), f.logger = console, f.autoDownload = !1, f.autoInstallOnAppQuit = !0, f.on("update-available", (e) => {
    console.log("[Updater] Update available:", e.version), Ie = !0, p.showMessageBox(v, {
      type: "info",
      title: "Update Available",
      message: `A new version (${e.version}) is available!`,
      detail: "Would you like to download and install it now?",
      buttons: ["Download", "Later"],
      defaultId: 0,
      cancelId: 1
    }).then((n) => {
      n.response === 0 && (console.log("[Updater] User chose to download update"), f.downloadUpdate());
    });
  }), f.on("update-not-available", (e) => {
    console.log("[Updater] No update available. Current version is latest.");
  }), f.on("download-progress", (e) => {
    const n = Math.round(e.percent);
    console.log(`[Updater] Download progress: ${n}%`), v && !v.isDestroyed() && v.setProgressBar(e.percent / 100);
  }), f.on("update-downloaded", (e) => {
    console.log("[Updater] Update downloaded:", e.version), v && !v.isDestroyed() && v.setProgressBar(-1), p.showMessageBox(v, {
      type: "info",
      title: "Update Ready",
      message: "Update downloaded!",
      detail: "The update will be installed when you restart the app. Restart now?",
      buttons: ["Restart Now", "Later"],
      defaultId: 0,
      cancelId: 1
    }).then((n) => {
      n.response === 0 && (console.log("[Updater] Quitting and installing update..."), f.quitAndInstall());
    });
  }), f.on("error", (e) => {
    console.error("[Updater] Error:", e.message);
  }));
} catch (t) {
  console.log("[Updater] electron-updater not available:", t.message);
}
function Be(t, e) {
  if (v = t, K = e, !J) {
    console.log("[Updater] Auto-updater not available in this build");
    return;
  }
  const n = E(e);
  if (n.autoUpdateEnabled === !1) {
    console.log("[Updater] Auto-update is disabled in settings");
    return;
  }
  setTimeout(() => {
    W(!1);
  }, 5e3);
  const r = n.autoUpdateCheckInterval || 4;
  setInterval(
    () => {
      E(K).autoUpdateEnabled !== !1 && W(!1);
    },
    r * 60 * 60 * 1e3
  );
}
function W(t = !0) {
  if (!J || !f) {
    console.log("[Updater] Auto-updater not available");
    return;
  }
  console.log("[Updater] Checking for updates..."), f.checkForUpdates().catch((e) => {
    console.error("[Updater] Error checking for updates:", e.message), t && p.showMessageBox(v, {
      type: "error",
      title: "Update Error",
      message: "Could not check for updates",
      detail: e.message
    });
  });
}
const j = globalThis.fetch;
q() === "linux" && d.commandLine.appendSwitch("no-sandbox");
if (q() === "win32")
  try {
    const { default: t } = await import("electron-squirrel-startup");
    t && d.quit();
  } catch {
  }
const Ae = me(import.meta.url), D = de(Ae);
let x = !1, Q = !1, w = null, I = 0;
const X = 3;
let H = !1;
async function De(t = "http://localhost:5173") {
  try {
    return typeof j != "function" ? !1 : (await j(t)).ok;
  } catch {
    return !1;
  }
}
const M = await De(), xe = E(d);
xe.disableHardwareAcceleration && (console.log("[Electron] Disabling hardware acceleration"), d.disableHardwareAcceleration());
async function le(t = 30, e = 1e3) {
  console.log("[Backend] Waiting for backend to be ready...");
  for (let n = 0; n < t; n++) {
    try {
      if ((await j("http://127.0.0.1:8000/api/health")).ok)
        return console.log("[Backend] Backend is ready!"), !0;
    } catch {
    }
    await new Promise((r) => setTimeout(r, e));
  }
  return console.error("[Backend] Backend failed to start within timeout"), !1;
}
function ie() {
  if (w) {
    console.log("[Backend] Process already running");
    return;
  }
  const t = q() === "win32", e = d.getPath("userData"), n = t ? ";" : ":";
  let r, o, a, s = [];
  if (M) {
    const l = h(D, "..");
    o = h(l, "backend"), r = t ? h(o, "venv", "Scripts", "python.exe") : h(o, "venv", "bin", "python"), s = ["-m", "app.main"], a = h(o, "bin");
  } else {
    const l = process.resourcesPath;
    o = h(l, "dist_bundle"), r = t ? h(o, "backend_server.exe") : h(o, "backend_server"), a = h(o, "bin");
  }
  const i = _(a) ? `${a}${n}${process.env.PATH || ""}` : process.env.PATH;
  console.log("[Backend] Starting backend..."), console.log("[Backend] isDev:", M), console.log("[Backend] Executable:", r), console.log("[Backend] Working dir:", o), console.log("[Backend] FFmpeg bin dir:", a, _(a) ? "(found)" : "(not found, using system)"), console.log("[Backend] User data path:", e), w = ee(r, s, {
    cwd: o,
    env: {
      ...process.env,
      USER_DATA_PATH: e,
      PATH: i
    },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: !0
  }), w.stdout?.on("data", (l) => {
    console.log("[Backend]", l.toString().trim());
  }), w.stderr?.on("data", (l) => {
    console.error("[Backend Error]", l.toString().trim());
  }), w.on("error", (l) => {
    console.error("[Backend] Failed to start:", l), w = null, H || (p.showErrorBox(
      "Backend Error",
      `Failed to start backend process:
${l.message}

Please contact the developer if this issue persists.`
    ), d.quit());
  }), w.on("exit", (l, c) => {
    if (console.log(`[Backend] Process exited with code ${l}, signal ${c}`), w = null, H) {
      console.log("[Backend] Shutdown initiated, not restarting");
      return;
    }
    l !== 0 && l !== null && (console.error(`[Backend] Crashed with exit code ${l}`), I < X ? (I++, console.log(`[Backend] Attempting restart (${I}/${X})...`), setTimeout(() => {
      ie(), le(10, 1e3).then((u) => {
        if (u) {
          console.log("[Backend] Restarted successfully"), I = 0;
          const g = k.getAllWindows();
          g.length > 0 && g[0].webContents.send("backend-reconnected");
        } else
          console.error("[Backend] Failed to restart"), p.showErrorBox(
            "Backend Connection Lost",
            `The backend process crashed and could not be restarted automatically.

The application will now close. Please restart it.

If this issue persists, please contact the developer.`
          ), d.quit();
      });
    }, 2e3)) : (console.error("[Backend] Max restart attempts reached"), p.showErrorBox(
      "Backend Connection Lost",
      `The backend process has crashed multiple times and cannot be restarted.

The application will now close. Please restart it.

If this issue persists, please contact the developer.`
    ), d.quit()));
  });
}
function ce() {
  w && (console.log("[Backend] Stopping process..."), H = !0, w.kill(), w = null);
}
const Z = () => {
  const t = new k({
    width: 800,
    height: 800,
    icon: h(D, "resources", "icon.png"),
    webPreferences: {
      preload: h(D, "preload.js")
    }
  });
  t.on("close", async (e) => {
    console.log("[Electron] Close event triggered. activeProcessing:", x), x && (e.preventDefault(), (await p.showMessageBox(t, {
      type: "warning",
      buttons: ["Keep processing", "Quit app"],
      defaultId: 0,
      cancelId: 0,
      title: "Processing in Progress",
      message: "Video processing is currently active",
      detail: `Closing the app will interrupt the current processing operation. You will need to restart it.

Are you sure you want to quit?`
    })).response === 1 && (x = !1, t.destroy()));
  }), t.loadFile(h(D, "../renderer/main_window/index.html"));
};
d.whenReady().then(async () => {
  if (se({ ipcMain: m, app: d, shell: N }), ke({ ipcMain: m, app: d, shell: N }), Te({ ipcMain: m, app: d }), ie(), !await le()) {
    p.showErrorBox(
      "Backend Failed to Start",
      "The backend server failed to start. Please check the logs and try again."
    ), d.quit();
    return;
  }
  if (Z(), !M) {
    const e = k.getAllWindows();
    e.length > 0 && Be(e[0], d);
  }
  d.on("activate", () => {
    k.getAllWindows().length === 0 && Z();
  });
});
d.on("window-all-closed", () => {
  process.platform !== "darwin" && (ce(), d.quit());
});
d.on("before-quit", () => {
  ce();
});
m.handle("dialog:selectVideoFile", async (t) => {
  const e = k.fromWebContents(t.sender), n = await p.showOpenDialog(e, {
    properties: ["openFile"],
    filters: [
      { name: "Videos", extensions: ["mp4", "mov", "avi", "mkv", "webm", "m3u8"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  return n.canceled ? null : n.filePaths[0];
});
m.handle("dialog:selectVideoFiles", async (t) => {
  const e = k.fromWebContents(t.sender), n = await p.showOpenDialog(e, {
    properties: ["openFile", "multiSelections"],
    filters: [
      { name: "Videos", extensions: ["mp4", "mov", "avi", "mkv", "webm", "m3u8"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  return n.canceled ? [] : n.filePaths;
});
m.handle("file:exists", async (t, e) => {
  try {
    return _(e);
  } catch {
    return !1;
  }
});
async function Fe(t) {
  return new Promise((e) => {
    const n = ee("ffprobe", [
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=width,height:format=duration",
      "-of",
      "json",
      t
    ]);
    let r = "";
    n.stdout.on("data", (o) => {
      r += o.toString();
    }), n.on("close", (o) => {
      if (o === 0 && r.trim())
        try {
          const a = JSON.parse(r), s = a.streams?.[0] || {}, i = a.format || {};
          e({
            duration: i.duration ? parseFloat(i.duration) : null,
            width: s.width || null,
            height: s.height || null
          });
        } catch {
          e({ duration: null, width: null, height: null });
        }
      else
        e({ duration: null, width: null, height: null });
    }), n.on("error", () => {
      e({ duration: null, width: null, height: null });
    }), setTimeout(() => {
      n.kill(), e({ duration: null, width: null, height: null });
    }, 5e3);
  });
}
function Me(t) {
  const e = new Date(t), n = e.getFullYear(), r = String(e.getMonth() + 1).padStart(2, "0"), o = String(e.getDate()).padStart(2, "0"), a = String(e.getHours()).padStart(2, "0"), s = String(e.getMinutes()).padStart(2, "0");
  return `${n}-${r}-${o} ${a}:${s}`;
}
m.handle("file:getMetadata", async (t, e) => {
  try {
    if (!_(e))
      return null;
    const n = pe(e), r = e.split(/[/\\]/).pop(), o = await Fe(e), a = n.birthtime && n.birthtime.getTime() > 0 ? n.birthtime : null, s = n.mtime, i = a && a < s ? a : s;
    return {
      name: r,
      size: n.size,
      sizeFormatted: Re(n.size),
      created: i.toISOString(),
      createdFormatted: Me(i),
      duration: o.duration,
      // in seconds
      width: o.width,
      height: o.height,
      resolution: o.width && o.height ? `${o.width}x${o.height}` : null
    };
  } catch (n) {
    return console.error("Failed to get file metadata:", n), null;
  }
});
m.handle("shell:openExternal", async (t, e) => {
  try {
    return await N.openExternal(e), !0;
  } catch (n) {
    return console.error("Failed to open external URL:", n), !1;
  }
});
m.handle("app:confirmCancelProcessing", async (t) => {
  const e = k.fromWebContents(t.sender);
  return (await p.showMessageBox(e, {
    type: "warning",
    buttons: ["Keep processing", "Cancel processing"],
    defaultId: 0,
    cancelId: 0,
    title: "Cancel Processing",
    message: "Are you sure you want to cancel processing?",
    detail: "This will stop the current transcoding operation."
  })).response === 1;
});
m.handle("app:setProcessingState", async (t, e, n = "unknown") => (x = e, e !== Q && (console.log(`[Electron] Processing state changed: ${e} (from: ${n})`), Q = e), !0));
m.handle("app:getVersion", () => d.getVersion());
m.handle("updater:checkForUpdates", () => M ? { success: !1, error: "Updates not available in development mode" } : (W(!0), { success: !0 }));
function Re(t) {
  if (t === 0) return "0 Bytes";
  const e = 1024, n = ["Bytes", "KB", "MB", "GB"], r = Math.floor(Math.log(t) / Math.log(e));
  return Math.round(t / Math.pow(e, r) * 100) / 100 + " " + n[r];
}
