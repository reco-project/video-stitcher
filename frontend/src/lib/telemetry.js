/**
 * Telemetry helper (renderer-side)
 *
 * This file is intentionally small. It does NOT store telemetry itself.
 * Instead it forwards events to Electron main via IPC, where the local JSONL
 * file is maintained.
 *
 * Privacy contract
 * - Telemetry is opt-in (default off).
 * - Do not send sensitive data in `props`.
 *   Never include: file paths, filenames, match names, raw logs/stack traces.
 * - Keep events event-based and minimal.
 * - There is no network upload code here.
 */

async function getTelemetrySettings() {
	if (!window.electronAPI?.readSettings) {
		return { enabled: false, includeSystemInfo: false };
	}
	try {
		const settings = await window.electronAPI.readSettings();
		return {
			enabled: !!settings.telemetryEnabled,
			includeSystemInfo: !!settings.telemetryIncludeSystemInfo,
		};
	} catch {
		return { enabled: false, includeSystemInfo: false };
	}
}

export async function trackTelemetryEvent(name, props = null) {
	const { enabled, includeSystemInfo } = await getTelemetrySettings();
	if (!enabled) return false;
	if (!window.electronAPI?.telemetryTrack) return false;

	return window.electronAPI.telemetryTrack({
		name,
		props,
		include_system_info: includeSystemInfo,
	});
}
