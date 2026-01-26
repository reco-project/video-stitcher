/**
 * Telemetry uploader (optional)
 */

import crypto from 'node:crypto';
import { join } from 'node:path';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

import { getTelemetryDir, getTelemetryEventsPath, getOrCreateClientId, TELEMETRY_SCHEMA_VERSION, getHardwareHash } from './telemetry.js';

const MAX_EVENTS_PER_UPLOAD = 200;

const fetchImpl = globalThis.fetch;

function getUploadStatePath(app) {
	return join(getTelemetryDir(app), 'upload_state.json');
}

function readUploadState(app) {
	const statePath = getUploadStatePath(app);
	try {
		if (!existsSync(statePath)) return { last_uploaded_line: 0, last_success_at: null, last_hardware_hash: null };
		const raw = readFileSync(statePath, 'utf8');
		const parsed = JSON.parse(raw);
		return {
			last_uploaded_line: Number.isFinite(parsed?.last_uploaded_line) ? parsed.last_uploaded_line : 0,
			last_success_at: typeof parsed?.last_success_at === 'string' ? parsed.last_success_at : null,
			last_hardware_hash: typeof parsed?.last_hardware_hash === 'string' ? parsed.last_hardware_hash : null,
		};
	} catch {
		return { last_uploaded_line: 0, last_success_at: null, last_hardware_hash: null };
	}
}

function writeUploadState(app, state) {
	const statePath = getUploadStatePath(app);
	try {
		writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`, 'utf8');
	} catch { }
}

function normalizeEndpointUrl(endpointUrl) {
	if (typeof endpointUrl !== 'string') return null;
	const trimmed = endpointUrl.trim().replace(/\/+$/, '');
	if (!trimmed) return null;
	if (!/^https?:\/\//i.test(trimmed)) return null;
	return trimmed;
}

function readAllEvents(app) {
	const path = getTelemetryEventsPath(app);
	if (!existsSync(path)) return [];
	try {
		const raw = readFileSync(path, 'utf8');
		return raw.split('\n').map((l) => l.trim()).filter(Boolean);
	} catch {
		return [];
	}
}

export async function uploadTelemetryNow({ app, endpointUrl }) {
	const normalizedEndpoint = normalizeEndpointUrl(endpointUrl);
	if (!normalizedEndpoint) {
		return { ok: false, error: 'Invalid endpoint URL (must start with http(s)://)' };
	}

	const lines = readAllEvents(app);
	const state = readUploadState(app);

	let startLine = state.last_uploaded_line;
	if (startLine > lines.length) startLine = 0;

	const slice = lines.slice(startLine, startLine + MAX_EVENTS_PER_UPLOAD);
	const events = [];
	for (const line of slice) {
		try {
			events.push(JSON.parse(line));
		} catch { }
	}

	if (slice.length === 0) {
		return { ok: true, sent: 0, remaining_lines: 0 };
	}

	const currentHardwareHash = await getHardwareHash(app);
	const hardwareChanged = state.last_hardware_hash !== currentHardwareHash;

	const body = {
		schema_version: TELEMETRY_SCHEMA_VERSION,
		client_id: getOrCreateClientId(app),
		app: {
			name: 'video-stitcher',
			version: app.getVersion(),
			environment: app.isPackaged ? 'production' : 'dev',
		},
		sent_at: new Date().toISOString(),
		batch_id: crypto.randomUUID(),
		hardware_changed: hardwareChanged,
		events,
	};

	try {
		if (typeof fetchImpl !== 'function') {
			return { ok: false, error: 'fetch() is not available in this runtime' };
		}

		const maxRetries = 3;
		let lastError = null;

		for (let attempt = 0; attempt < maxRetries; attempt++) {
			if (attempt > 0) {
				await new Promise((resolve) => setTimeout(resolve, Math.pow(2, attempt - 1) * 1000));
			}

			try {
				const res = await fetchImpl(normalizedEndpoint, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify(body),
				});

				if (res.status === 429) {
					console.warn('[TELEMETRY] Rate limited (429).');
					return { ok: false, status: 429, error: await res.text().catch(() => 'Rate limited') };
				}

				if (!res.ok) {
					lastError = { ok: false, status: res.status, error: await res.text().catch(() => res.statusText) };
					continue;
				}

				const newLine = startLine + slice.length;
				writeUploadState(app, {
					last_uploaded_line: newLine,
					last_success_at: new Date().toISOString(),
					last_hardware_hash: currentHardwareHash,
				});

				console.log(`[TELEMETRY] Uploaded ${slice.length} event(s). Remaining: ${Math.max(0, lines.length - newLine)}`);

				return { ok: true, sent: slice.length, remaining_lines: Math.max(0, lines.length - newLine) };
			} catch (err) {
				lastError = { ok: false, error: err?.message || 'Network error' };
			}
		}

		return lastError || { ok: false, error: 'Upload failed after retries' };
	} catch (err) {
		return { ok: false, error: err?.message || 'Network error' };
	}
}

export function registerTelemetryUploadIpc({ ipcMain, app }) {
	let uploadInterval = null;

	ipcMain.handle('telemetry:uploadNow', async (_event, payload) => {
		return uploadTelemetryNow({ app, endpointUrl: payload?.endpointUrl });
	});

	const startPeriodicUpload = () => {
		if (uploadInterval) return;

		setTimeout(async () => {
			const { readSettings } = await import('./settings.js');
			const settings = readSettings(app);
			if (settings.telemetryEnabled && settings.telemetryEndpointUrl) {
				console.log('[TELEMETRY] Immediate upload on app start...');
				try {
					await uploadTelemetryNow({ app, endpointUrl: settings.telemetryEndpointUrl });
				} catch (err) {
					console.warn('[TELEMETRY] Immediate upload failed:', err?.message);
				}
			}
		}, 2000);

		// Auto-upload interval: 5 minutes in production, 30 seconds in development
		const uploadIntervalMs = app.isPackaged ? 5 * 60 * 1000 : 30 * 1000;

		uploadInterval = setInterval(async () => {
			const { readSettings } = await import('./settings.js');
			const settings = readSettings(app);
			if (!settings.telemetryEnabled || !settings.telemetryAutoUpload || !settings.telemetryEndpointUrl) return;
			console.log('[TELEMETRY] Periodic upload triggered...');
			try {
				await uploadTelemetryNow({ app, endpointUrl: settings.telemetryEndpointUrl });
			} catch (err) {
				console.warn('[TELEMETRY] Periodic upload failed:', err?.message);
			}
		}, uploadIntervalMs);
	};

	if (app.isReady()) {
		startPeriodicUpload();
	} else {
		app.once('ready', startPeriodicUpload);
	}

	app.on('before-quit', async (event) => {
		const { readSettings } = await import('./settings.js');
		const settings = readSettings(app);
		if (!settings.telemetryEnabled || !settings.telemetryAutoUpload || !settings.telemetryEndpointUrl) return;

		event.preventDefault();

		console.log('[TELEMETRY] Upload on quit (3s timeout)...');
		const uploadPromise = uploadTelemetryNow({ app, endpointUrl: settings.telemetryEndpointUrl });
		const timeoutPromise = new Promise((resolve) => setTimeout(resolve, 3000));

		try {
			await Promise.race([uploadPromise, timeoutPromise]);
		} catch (err) {
			console.warn('[TELEMETRY] Upload on quit failed:', err?.message);
		} finally {
			if (uploadInterval) clearInterval(uploadInterval);
			app.exit();
		}
	});
}
