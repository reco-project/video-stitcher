import { dialog } from 'electron';
import { readSettings } from './settings.js';
import log from 'electron-log';

let autoUpdater = null;
let updaterAvailable = false;
let updateAvailable = false;
let mainWindow = null;
let appInstance = null;

// Try to load electron-updater (may not be available in all builds)
try {
    const pkg = await import('electron-updater');
    autoUpdater = pkg.autoUpdater || pkg.default?.autoUpdater;
    log.info('[Updater] electron-updater loaded successfully');
    if (autoUpdater) {
        updaterAvailable = true;

        // Configure logging - electron-log for persistent logs (especially important on Windows)
        autoUpdater.logger = log;
        autoUpdater.logger.transports.file.level = 'info';
        autoUpdater.autoDownload = false; // Don't download automatically, ask user first
        autoUpdater.autoInstallOnAppQuit = true;

        // Update available
        autoUpdater.on('update-available', (info) => {
            log.info('[Updater] Update available:', info.version);
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
            log.info('[Updater] No update available. Current version is latest.');
        });

        // Download progress
        autoUpdater.on('download-progress', (progress) => {
            const percent = Math.round(progress.percent);
            log.info(`[Updater] Download progress: ${percent}%`);

            // Send progress to renderer if needed
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.setProgressBar(progress.percent / 100);
            }
        });

        // Update downloaded
        autoUpdater.on('update-downloaded', (info) => {
            log.info('[Updater] Update downloaded:', info.version);

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
            log.error('[Updater] Error:', err.message);
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

    autoUpdater.checkForUpdates().catch((err) => {
        log.error('[Updater] Error checking for updates:', err.message);
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
