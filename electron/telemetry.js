/**
 * Telemetry (local-first, opt-in)
 *
 * This module is intentionally separated from `electron/main.js` so it is easy to
 * audit. It implements a minimal event logger that writes to a local JSONL file.
 *
 * Privacy model
 * - The app never uploads telemetry in this repo version.
 * - The renderer process cannot write arbitrary files; it can only append events
 *   via IPC to this module.
 * - Events MUST NOT include personal/sensitive data.
 *   Specifically, do not include:
 *   - file paths, filenames, match names
 *   - raw error logs / stack traces
 *   - video metadata, URLs that may contain tokens
 *   - any stable hardware identifiers (serials, UUIDs, MAC addresses)
 *
 * Storage
 * - Directory:   app.getPath('userData')/telemetry/
 * - Events file: events.jsonl   (one JSON object per line)
 * - Client id:   client_id.txt  (random UUID generated once)
 *
 * Event schema (per line)
 * {
 *   schema_version: 1,
 *   ts: "ISO-8601",
 *   name: "app_open" | "match_created" | ...,
 *   client_id: "uuid-v4",
 *   props: object|null
 * }
 *
 * Optional context event (written at most once per app run)
 * {
 *   name: "context",
 *   os: { platform, release, arch },
 *   hardware: {
 *     cpu_model,
 *     cpu_threads,
 *     ram_gb,
 *     // GPU info is best-effort. Depending on platform/driver, Electron may only
 *     // provide numeric identifiers (vendorId/deviceId) rather than a name.
 *     gpus: [{ name?: string, vendor_id?: number, device_id?: number, active?: boolean|null }] | null
 *   }
 * }
 */

import os from 'node:os';
import crypto from 'node:crypto';
import { join } from 'node:path';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs';

export const TELEMETRY_SCHEMA_VERSION = 1;

// Keep this allowlist small on purpose.
// Extend consciously, and update docs/UI accordingly.
const ALLOWED_EVENT_NAMES = new Set([
	'app_open',
	'match_created',
	'processing_start',
	'processing_cancel',
	'processing_success',
	'processing_error',
	'context',
]);

function ensureTelemetryDir(app) {
	mkdirSync(getTelemetryDir(app), { recursive: true });
}

export function getTelemetryDir(app) {
	return join(app.getPath('userData'), 'telemetry');
}

export function getTelemetryEventsPath(app) {
	return join(getTelemetryDir(app), 'events.jsonl');
}

function getClientIdPath(app) {
	return join(getTelemetryDir(app), 'client_id.txt');
}

export function getOrCreateClientId(app) {
	ensureTelemetryDir(app);
	const clientIdPath = getClientIdPath(app);

	try {
		if (existsSync(clientIdPath)) {
			const val = readFileSync(clientIdPath, 'utf8').trim();
			if (val) return val;
		}
	} catch {
		// ignore
	}

	const newId = crypto.randomUUID();
	try {
		writeFileSync(clientIdPath, `${newId}\n`, 'utf8');
	} catch {
		// ignore
	}
	return newId;
}

async function getSystemInfoSnapshot(app) {
	const cpuList = os.cpus() || [];
	const cpuModel = cpuList[0]?.model || null;
	const cpuThreads = cpuList.length || null;
	const ramGb = Math.round(os.totalmem() / (1024 ** 3));

	let gpus = null;
	const normalizeGpuDevice = (d) => {
		if (!d || typeof d !== 'object') return null;
		const name = d.deviceString || d.description || d.name || null;
		const vendorId = typeof d.vendorId === 'number' ? d.vendorId : typeof d.vendor_id === 'number' ? d.vendor_id : null;
		const deviceId = typeof d.deviceId === 'number' ? d.deviceId : typeof d.device_id === 'number' ? d.device_id : null;
		const active = typeof d.active === 'boolean' ? d.active : null;
		if (name) return { name, active };
		if (vendorId !== null || deviceId !== null) {
			return {
				vendor_id: vendorId,
				device_id: deviceId,
				active,
			};
		}
		return null;
	};

	const extractGpus = (gpuInfo) => {
		const devices = gpuInfo?.gpuDevice ?? gpuInfo?.gpuDevices ?? gpuInfo?.devices ?? null;
		if (Array.isArray(devices)) {
			const normalized = devices.map(normalizeGpuDevice).filter(Boolean);
			return normalized.length ? normalized : null;
		}
		if (devices && typeof devices === 'object') {
			const single = normalizeGpuDevice(devices);
			return single ? [single] : null;
		}
		return null;
	};

	try {
		// Electron provides a cross-platform (best-effort) GPU description.
		// On some Linux setups, `basic` may return an empty device list; `complete` can be more informative.
		const basic = await app.getGPUInfo('basic');
		gpus = extractGpus(basic);

		if (!gpus || gpus.length === 0) {
			const complete = await app.getGPUInfo('complete');
			gpus = extractGpus(complete);
		}

		if (gpus && gpus.length === 0) gpus = null;
	} catch {
		// GPU info isn't always available.
	}

	return {
		os: {
			platform: process.platform,
			release: os.release(),
			arch: process.arch,
		},
		hardware: {
			cpu_model: cpuModel,
			cpu_threads: cpuThreads,
			ram_gb: Number.isFinite(ramGb) ? ramGb : null,
			gpus,
		},
	};
}

export async function getHardwareHash(app) {
	const snapshot = await getSystemInfoSnapshot(app);
	const str = JSON.stringify(snapshot);
	return crypto.createHash('sha256').update(str).digest('hex').slice(0, 16);
}

function appendTelemetryEvent(app, event) {
	ensureTelemetryDir(app);
	appendFileSync(getTelemetryEventsPath(app), `${JSON.stringify(event)}\n`, 'utf8');
}

function sanitizeProps(props) {
	if (!props || typeof props !== 'object' || Array.isArray(props)) return null;

	// Guardrails to avoid accidentally logging huge blobs.
	try {
		const serialized = JSON.stringify(props);
		if (serialized.length > 2048) return null;
		return JSON.parse(serialized);
	} catch {
		return null;
	}
}

function isAllowedEventName(name) {
	return ALLOWED_EVENT_NAMES.has(name);
}

export function registerTelemetryIpc({ ipcMain, app, shell }) {
	let wroteContextThisRun = false;

	ipcMain.handle('telemetry:getInfo', async () => {
		return {
			schema_version: TELEMETRY_SCHEMA_VERSION,
			telemetry_dir: getTelemetryDir(app),
			events_path: getTelemetryEventsPath(app),
			client_id: getOrCreateClientId(app),
		};
	});

	ipcMain.handle('telemetry:openFolder', async () => {
		try {
			ensureTelemetryDir(app);
			await shell.openPath(getTelemetryDir(app));
			return true;
		} catch {
			return false;
		}
	});

	ipcMain.handle('telemetry:deleteLocal', async () => {
		try {
			const eventsPath = getTelemetryEventsPath(app);
			if (existsSync(eventsPath)) {
				writeFileSync(eventsPath, '', 'utf8'); // Clear the file content
			}
			return { ok: true };
		} catch (err) {
			return { ok: false, error: err?.message || 'Failed to delete telemetry data' };
		}
	});

	ipcMain.handle('telemetry:resetClientId', async () => {
		try {
			const clientIdPath = getClientIdPath(app);
			if (existsSync(clientIdPath)) {
				unlinkSync(clientIdPath);
			}
			const newId = getOrCreateClientId(app);
			return { ok: true, client_id: newId };
		} catch (err) {
			return { ok: false, error: err?.message || 'Failed to reset client ID' };
		}
	});

	ipcMain.handle('telemetry:track', async (_event, payload) => {
		// Payload is renderer-controlled; keep it strictly validated.
		// Shape: { name: string, props?: object, include_system_info?: boolean }
		try {
			const name = typeof payload?.name === 'string' ? payload.name.trim() : '';
			if (!name || name.length > 64) return false;
			if (!isAllowedEventName(name)) return false;

			const includeSystemInfo = payload?.include_system_info === true;
			const props = sanitizeProps(payload?.props);

			// Write context only once per run when system info is enabled
			if (includeSystemInfo && !wroteContextThisRun) {
				wroteContextThisRun = true;
				const ctx = await getSystemInfoSnapshot(app);
				appendTelemetryEvent(app, {
					schema_version: TELEMETRY_SCHEMA_VERSION,
					ts: new Date().toISOString(),
					name: 'context',
					client_id: getOrCreateClientId(app),
					app: { version: app.getVersion() },
					...ctx,
				});
			}

			appendTelemetryEvent(app, {
				schema_version: TELEMETRY_SCHEMA_VERSION,
				ts: new Date().toISOString(),
				name,
				client_id: getOrCreateClientId(app),
				props,
			});

			return true;
		} catch {
			return false;
		}
	});
}
