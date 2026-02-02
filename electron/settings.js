/**
 * Settings management (file-based)
 *
 * Stores app settings in:
 * - Production: userData/settings.json
 * - Development: devData/settings.json (at project root)
 */

import { join, dirname } from 'node:path';
import { existsSync, readFileSync, writeFileSync, rmSync, mkdirSync } from 'node:fs';
import { dialog } from 'electron';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const defaultSettings = {
    debugMode: true,
    apiBaseUrl: 'http://127.0.0.1:8000/api',
    encoderPreference: 'auto',
    disableHardwareAcceleration: false,
    telemetryEnabled: false,
    telemetryIncludeSystemInfo: false,
    telemetryEndpointUrl: 'https://telemetry.reco-project.org/telemetry',
    telemetryAutoUpload: false,
    telemetryPromptShown: false,
};

function getSettingsPath(app) {
    // In development, use devData/ at project root to avoid mixing with production data
    const isDev = !app.isPackaged;
    if (isDev) {
        // electron/ is one level below project root
        return join(__dirname, '..', 'devData', 'settings.json');
    }
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
        // Ensure directory exists
        const dir = dirname(path);
        if (!existsSync(dir)) {
            mkdirSync(dir, { recursive: true });
        }
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

    ipcMain.handle('settings:clearUserDataFolder', async (event) => {
        try {
            const userDataPath = app.getPath('userData');

            // Show confirmation dialog
            const result = await dialog.showMessageBox({
                type: 'warning',
                buttons: ['Cancel', 'Delete Everything'],
                defaultId: 0,
                cancelId: 0,
                title: 'Clear All User Data',
                message: 'This will permanently delete ALL data',
                detail: 'This includes:\n• All matches and videos\n• All settings and preferences\n• All telemetry data\n• All logs and temporary files\n\nThe application will quit after deletion. This cannot be undone.\n\nAre you absolutely sure?',
            });

            if (result.response !== 1) {
                return { ok: false, cancelled: true };
            }

            console.log('[Settings] Clearing user data folder:', userDataPath);

            // Delete the userData folder contents
            // We do this by deleting everything except Electron's internal files
            try {
                const { readdirSync, statSync } = await import('node:fs');
                const contents = readdirSync(userDataPath);

                // Delete all our custom directories/files
                const itemsToDelete = contents.filter(item =>
                    !item.startsWith('.')  // Keep hidden files (Electron internals)
                    && item !== 'Crashpad'  // Keep Electron crash reporter
                    && item !== 'GPUCache'  // Keep GPU cache
                );

                for (const item of itemsToDelete) {
                    const itemPath = join(userDataPath, item);
                    console.log('[Settings] Deleting:', itemPath);
                    rmSync(itemPath, { recursive: true, force: true });
                }

                console.log('[Settings] User data cleared successfully');
            } catch (deleteErr) {
                console.error('[Settings] Error during deletion:', deleteErr);
                return { ok: false, error: deleteErr.message };
            }

            // Quit the app after a short delay
            setTimeout(() => {
                app.quit();
            }, 500);

            return { ok: true };
        } catch (err) {
            console.error('[Settings] Error clearing user data:', err);
            return { ok: false, error: err?.message || 'Failed to clear user data' };
        }
    });

    ipcMain.handle('settings:getEncoderInfo', async () => {
        try {
            const settings = readSettings(app);
            // Query backend for available encoders
            const apiBaseUrl = settings.apiBaseUrl || 'http://127.0.0.1:8000/api';
            const response = await fetch(`${apiBaseUrl}/settings/encoders`);
            if (!response.ok) {
                throw new Error('Failed to get encoder info from backend');
            }
            const backendInfo = await response.json();
            return {
                ok: true,
                current_encoder: settings.encoderPreference || 'auto',
                available_encoders: backendInfo.available_encoders || ['auto', 'libx264'],
                encoder_descriptions: backendInfo.encoder_descriptions || {},
            };
        } catch (err) {
            return { ok: false, error: err?.message || 'Failed to get encoder info' };
        }
    });
}
