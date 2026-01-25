/**
 * Settings management (file-based)
 *
 * Stores app settings in userData/settings.json
 */

import { join } from 'node:path';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const defaultSettings = {
    debugMode: false,
    apiBaseUrl: 'http://127.0.0.1:8000/api',
    telemetryEnabled: false,
    telemetryIncludeSystemInfo: false,
    telemetryEndpointUrl: 'https://telemetry.reco-project.org/telemetry',
    telemetryAutoUpload: false,
    telemetryPromptShown: false,
};

function getSettingsPath(app) {
    return join(app.getPath('userData'), 'settings.json');
}

export function readSettings(app) {
    const path = getSettingsPath(app);
    try {
        if (!existsSync(path)) return defaultSettings;
        const raw = readFileSync(path, 'utf8');
        const parsed = JSON.parse(raw);
        return { ...defaultSettings, ...parsed };
    } catch {
        return defaultSettings;
    }
}

export function writeSettings(app, settings) {
    const path = getSettingsPath(app);
    try {
        const merged = { ...defaultSettings, ...settings };
        writeFileSync(path, JSON.stringify(merged, null, 2), 'utf8');
        return { ok: true };
    } catch (err) {
        return { ok: false, error: err?.message || 'Failed to write settings' };
    }
}

export function registerSettingsIpc({ ipcMain, app, shell }) {
    ipcMain.handle('settings:read', async () => {
        return readSettings(app);
    });

    ipcMain.handle('settings:write', async (_event, settings) => {
        return writeSettings(app, settings);
    });

    ipcMain.handle('settings:openUserDataFolder', async () => {
        try {
            const userDataPath = app.getPath('userData');
            await shell.openPath(userDataPath);
            return { ok: true };
        } catch (err) {
            return { ok: false, error: err?.message || 'Failed to open folder' };
        }
    });
}
