import { dialog } from 'electron';
import { readSettings } from './settings.js';
import log from 'electron-log';

let autoUpdater = null;
let updaterAvailable = false;
let updateAvailable = false;
let mainWindow = null;
let appInstance = null;

// Check if beta updates are enabled via environment variable
const betaUpdatesEnabled = process.env.VIDEO_STITCHER_BETA_UPDATES === '1';

// Constants for log truncation
const MAX_RELEASE_NOTES_LENGTH = 100;
const MAX_STACK_TRACE_LENGTH = 200;

// Try to load electron-updater (may not be available in all builds)
try {
    const pkg = await import('electron-updater');
    autoUpdater = pkg.autoUpdater || pkg.default?.autoUpdater;
    log.info('[Updater] electron-updater loaded successfully');
    
    // Log the electron-log file path for diagnostics
    const logFilePath = log.transports.file.getFile()?.path || 'unknown';
    log.info('[Updater] Log file location:', logFilePath);
    
    if (autoUpdater) {
        updaterAvailable = true;

        // Configure logging - electron-log for persistent logs (especially important on Windows)
        autoUpdater.logger = log;
        autoUpdater.logger.transports.file.level = 'info';
        autoUpdater.autoDownload = false; // Don't download automatically, ask user first
        autoUpdater.autoInstallOnAppQuit = true;
        
        // Configure beta/prerelease updates
        if (betaUpdatesEnabled) {
            autoUpdater.allowPrerelease = true;
            log.info('[Updater] Beta updates ENABLED - will receive prerelease versions');
        } else {
            autoUpdater.allowPrerelease = false;
            log.info('[Updater] Beta updates DISABLED - stable releases only');
        }

        // Checking for update
        autoUpdater.on('checking-for-update', () => {
            log.info('[Updater] Event: checking-for-update');
        });

        // Update available
        autoUpdater.on('update-available', (info) => {
            log.info('[Updater] Event: update-available');
            log.info('[Updater] Update available:', JSON.stringify({
                version: info.version,
                releaseDate: info.releaseDate,
                releaseName: info.releaseName,
                releaseNotes: info.releaseNotes?.substring(0, MAX_RELEASE_NOTES_LENGTH) || 'N/A'
            }, null, 2));
            updateAvailable = true;

            dialog
                .showMessageBox(mainWindow, {
                    type: 'info',
                    title: 'Update Available',
                    message: `A new version (${info.version}) is available!`,
                    detail: 'Would you like to download and install it now?',
                    buttons: ['Download', 'Later'],
                    defaultId: 0,
                    cancelId: 1,
                })
                .then((result) => {
                    if (result.response === 0) {
                        log.info('[Updater] User chose to download update');
                        autoUpdater.downloadUpdate().catch((err) => {
                            log.error('[Updater] Error downloading update:', err.message);
                            if (mainWindow && !mainWindow.isDestroyed()) {
                                dialog.showMessageBox(mainWindow, {
                                    type: 'error',
                                    title: 'Download Error',
                                    message: 'Failed to download update',
                                    detail: err.message,
                                });
                            }
                        });
                    }
                });
        });

        // No update available
        autoUpdater.on('update-not-available', (info) => {
            log.info('[Updater] Event: update-not-available');
            log.info('[Updater] No update available. Current version is latest:', JSON.stringify({
                version: info.version
            }, null, 2));
        });

        // Download progress
        autoUpdater.on('download-progress', (progress) => {
            const percent = Math.round(progress.percent);
            log.info(`[Updater] Event: download-progress - ${percent}% (${progress.transferred}/${progress.total} bytes, speed: ${Math.round(progress.bytesPerSecond / 1024)} KB/s)`);

            // Send progress to renderer if needed
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.setProgressBar(progress.percent / 100);
            }
        });

        // Update downloaded
        autoUpdater.on('update-downloaded', (info) => {
            log.info('[Updater] Event: update-downloaded');
            log.info('[Updater] Update downloaded successfully:', JSON.stringify({
                version: info.version,
                releaseDate: info.releaseDate,
                downloadedFile: info.downloadedFile || 'N/A'
            }, null, 2));

            // Clear progress bar
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.setProgressBar(-1);
            }

            dialog
                .showMessageBox(mainWindow, {
                    type: 'info',
                    title: 'Update Ready',
                    message: 'Update downloaded!',
                    detail: 'The update will be installed when you restart the app. Restart now?',
                    buttons: ['Restart Now', 'Later'],
                    defaultId: 0,
                    cancelId: 1,
                })
                .then((result) => {
                    if (result.response === 0) {
                        log.info('[Updater] Quitting and installing update...');
                        autoUpdater.quitAndInstall();
                    }
                });
        });

        // Error handling
        autoUpdater.on('error', (err) => {
            log.error('[Updater] Event: error');
            log.error('[Updater] Error details:', JSON.stringify({
                message: err.message,
                stack: err.stack?.substring(0, MAX_STACK_TRACE_LENGTH) || 'N/A'
            }, null, 2));
            // Show error to user if window is available
            if (mainWindow && !mainWindow.isDestroyed()) {
                dialog.showMessageBox(mainWindow, {
                    type: 'error',
                    title: 'Update Error',
                    message: 'An error occurred with the auto-updater',
                    detail: err.message,
                });
            }
        });
    }
} catch (err) {
    log.info('[Updater] electron-updater not available:', err.message);
}

export function initAutoUpdater(window, app) {
    mainWindow = window;
    appInstance = app;

    if (!updaterAvailable) {
        log.info('[Updater] Auto-updater not available in this build');
        return;
    }

    // Read settings to check if auto-update is enabled
    const settings = readSettings(app);
    if (settings.autoUpdateEnabled === false) {
        log.info('[Updater] Auto-update is disabled in settings');
        return;
    }

    // Check for updates on startup (with delay to not slow down launch)
    setTimeout(() => {
        checkForUpdates(false);
    }, 5000);

    // Check for updates every 4 hours (or configured interval)
    const intervalHours = settings.autoUpdateCheckInterval || 4;
    setInterval(
        () => {
            // Re-read settings in case user changed them
            const currentSettings = readSettings(appInstance);
            if (currentSettings.autoUpdateEnabled !== false) {
                checkForUpdates(false);
            }
        },
        intervalHours * 60 * 60 * 1000
    );
}

export function checkForUpdates(showNoUpdateDialog = true) {
    if (!updaterAvailable || !autoUpdater) {
        log.info('[Updater] Auto-updater not available');
        return;
    }

    log.info('[Updater] Checking for updates...');
    log.info('[Updater] Beta mode:', betaUpdatesEnabled ? 'ENABLED' : 'DISABLED');
    log.info('[Updater] allowPrerelease:', autoUpdater.allowPrerelease);

    autoUpdater.checkForUpdates().catch((err) => {
        log.error('[Updater] Error checking for updates:', err.message);
        log.error('[Updater] Error stack:', err.stack);
        if (showNoUpdateDialog) {
            dialog.showMessageBox(mainWindow, {
                type: 'error',
                title: 'Update Error',
                message: 'Could not check for updates',
                detail: err.message,
            });
        }
    });
}

export function isUpdateAvailable() {
    return updateAvailable;
}
