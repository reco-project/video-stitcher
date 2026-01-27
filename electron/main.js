import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron';
import { join, dirname } from 'node:path';
import { existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'url';
import { spawn } from 'node:child_process';
import { platform } from 'node:os';
import started from 'electron-squirrel-startup';
import { registerTelemetryIpc } from './telemetry.js';
import { registerTelemetryUploadIpc } from './telemetry_uploader.js';
import { registerSettingsIpc, readSettings } from './settings.js';
import { initAutoUpdater, checkForUpdates } from './updater.js';

const fetchImpl = globalThis.fetch;

// Add no-sandbox switch on Linux (env var is set by wrapper script)
if (platform() === 'linux') {
	app.commandLine.appendSwitch('no-sandbox');
}

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
	app.quit();
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const devServerUrl = 'http://localhost:5173';

// Track active processing states
let activeProcessing = false;
let lastLoggedState = false;

// Backend process management
let backendProcess = null;
let backendRestartAttempts = 0;
const MAX_RESTART_ATTEMPTS = 3;
let isShuttingDown = false;

// Function to check if Vite dev server is running
async function viteDevServerRunning(url = 'http://localhost:5173') {
	try {
		if (typeof fetchImpl !== 'function') return false;
		const res = await fetchImpl(url);
		return res.ok;
	} catch {
		return false;
	}
}

const isDev = await viteDevServerRunning();

// Apply hardware acceleration setting BEFORE app is ready
const initialSettings = readSettings(app);
if (initialSettings.disableHardwareAcceleration) {
	console.log('[Electron] Disabling hardware acceleration');
	app.disableHardwareAcceleration();
}

// Wait for backend to be ready
async function waitForBackend(maxAttempts = 30, delayMs = 1000) {
	console.log('[Backend] Waiting for backend to be ready...');

	for (let i = 0; i < maxAttempts; i++) {
		try {
			const response = await fetchImpl('http://127.0.0.1:8000/api/health');
			if (response.ok) {
				console.log('[Backend] Backend is ready!');
				return true;
			}
		} catch (error) {
			// Backend not ready yet, continue waiting
		}

		await new Promise(resolve => setTimeout(resolve, delayMs));
	}

	console.error('[Backend] Backend failed to start within timeout');
	return false;
}

// Start backend process
function startBackend() {
	if (backendProcess) {
		console.log('[Backend] Process already running');
		return;
	}

	const isWin = platform() === 'win32';

	// In development, use workspace root paths
	// In production, use paths relative to app resources
	let backendDir;
	let pythonPath;

	if (isDev) {
		// Development: backend is at workspace root
		const workspaceRoot = join(__dirname, '..');
		backendDir = join(workspaceRoot, 'backend');
		pythonPath = isWin
			? join(backendDir, 'venv', 'Scripts', 'python.exe')
			: join(backendDir, 'venv', 'bin', 'python');
	} else {
		// Production: backend is at app/backend (no asar with Vite plugin)
		// or at app.asar.unpacked/backend if asar is enabled
		const appPath = app.getAppPath();
		const unpackedPath = appPath + '.unpacked';

		// Check if unpacked exists (asar mode) or use app path directly
		backendDir = existsSync(join(unpackedPath, 'backend'))
			? join(unpackedPath, 'backend')
			: join(appPath, 'backend');

		// Check for portable Python setup (Linux with lib/.portable marker)
		const portableMarker = join(backendDir, 'lib', '.portable');
		if (!isWin && existsSync(portableMarker)) {
			// Linux portable: use system python3 with PYTHONPATH
			pythonPath = 'python3';
		} else {
			pythonPath = isWin
				? join(backendDir, 'venv', 'Scripts', 'python.exe')
				: join(backendDir, 'venv', 'bin', 'python');
		}
	}

	const userDataPath = app.getPath('userData');

	// Set up FFmpeg path - use bundled binaries in production
	const ffmpegBinDir = join(backendDir, 'bin');
	const pathSeparator = isWin ? ';' : ':';
	const newPath = existsSync(ffmpegBinDir)
		? `${ffmpegBinDir}${pathSeparator}${process.env.PATH || ''}`
		: process.env.PATH;

	// Set PYTHONPATH for production builds (both Windows embedded and Linux portable)
	// This ensures Python can find the app module and installed packages
	let pythonPathEnv = process.env.PYTHONPATH || '';
	if (!isDev) {
		const portableLibDir = join(backendDir, 'lib');
		if (isWin) {
			// Windows embedded: packages in venv/Scripts/Lib/site-packages, app in backendDir
			const sitePackages = join(backendDir, 'venv', 'Scripts', 'Lib', 'site-packages');
			pythonPathEnv = `${backendDir}${pathSeparator}${sitePackages}${pathSeparator}${pythonPathEnv}`;
		} else if (existsSync(join(portableLibDir, '.portable'))) {
			// Linux portable: packages in lib/, app in backendDir
			pythonPathEnv = `${backendDir}${pathSeparator}${portableLibDir}${pathSeparator}${pythonPathEnv}`;
		}
	}

	console.log('[Backend] Starting Python backend...');
	console.log('[Backend] isDev:', isDev);
	console.log('[Backend] Python path:', pythonPath);
	console.log('[Backend] Backend dir:', backendDir);
	console.log('[Backend] FFmpeg bin dir:', ffmpegBinDir, existsSync(ffmpegBinDir) ? '(found)' : '(not found, using system)');
	console.log('[Backend] User data path:', userDataPath);
	if (pythonPathEnv) {
		console.log('[Backend] PYTHONPATH:', pythonPathEnv);
	}

	backendProcess = spawn(pythonPath, ['-m', 'app.main'], {
		cwd: backendDir,
		env: {
			...process.env,
			USER_DATA_PATH: userDataPath,
			PATH: newPath,
			PYTHONPATH: pythonPathEnv,
			PYTHONUNBUFFERED: '1', // Force Python to flush output immediately
		},
		stdio: ['ignore', 'pipe', 'pipe'], // Capture stdout and stderr
		windowsHide: true, // Hide console window on Windows
	});

	// Log backend output
	backendProcess.stdout?.on('data', (data) => {
		console.log('[Backend]', data.toString().trim());
	});

	backendProcess.stderr?.on('data', (data) => {
		console.error('[Backend Error]', data.toString().trim());
	});

	backendProcess.on('error', (error) => {
		console.error('[Backend] Failed to start:', error);
		backendProcess = null;

		// Show error dialog to user
		if (!isShuttingDown) {
			dialog.showErrorBox(
				'Backend Error',
				`Failed to start backend process:\n${error.message}\n\nPlease contact the developer if this issue persists.`
			);
			app.quit();
		}
	});

	backendProcess.on('exit', (code, signal) => {
		console.log(`[Backend] Process exited with code ${code}, signal ${signal}`);
		backendProcess = null;

		// Don't restart if we're shutting down intentionally
		if (isShuttingDown) {
			console.log('[Backend] Shutdown initiated, not restarting');
			return;
		}

		// Restart on unexpected exit
		if (code !== 0 && code !== null) {
			console.error(`[Backend] Crashed with exit code ${code}`);

			if (backendRestartAttempts < MAX_RESTART_ATTEMPTS) {
				backendRestartAttempts++;
				console.log(`[Backend] Attempting restart (${backendRestartAttempts}/${MAX_RESTART_ATTEMPTS})...`);

				// Wait a bit before restarting
				setTimeout(() => {
					startBackend();

					// Wait for backend and notify user
					waitForBackend(10, 1000).then((ready) => {
						if (ready) {
							console.log('[Backend] Restarted successfully');
							backendRestartAttempts = 0; // Reset counter on success

							// Notify user of successful recovery
							const windows = BrowserWindow.getAllWindows();
							if (windows.length > 0) {
								windows[0].webContents.send('backend-reconnected');
							}
						} else {
							console.error('[Backend] Failed to restart');

							// Show error dialog and quit
							dialog.showErrorBox(
								'Backend Connection Lost',
								'The backend process crashed and could not be restarted automatically.\n\nThe application will now close. Please restart it.\n\nIf this issue persists, please contact the developer.'
							);
							app.quit();
						}
					});
				}, 2000); // Wait 2 seconds before restart
			} else {
				console.error('[Backend] Max restart attempts reached');

				// Show error dialog and quit
				dialog.showErrorBox(
					'Backend Connection Lost',
					'The backend process has crashed multiple times and cannot be restarted.\n\nThe application will now close. Please restart it.\n\nIf this issue persists, please contact the developer.'
				);
				app.quit();
			}
		}
	});
}

// Stop backend process
function stopBackend() {
	if (backendProcess) {
		console.log('[Backend] Stopping process...');
		isShuttingDown = true; // Prevent restart attempts
		backendProcess.kill();
		backendProcess = null;
	}
}

const createWindow = () => {
	// Create the browser window.
	const mainWindow = new BrowserWindow({
		width: 800,
		height: 800,
		icon: join(__dirname, 'resources', 'icon.png'),
		webPreferences: {
			preload: join(__dirname, 'preload.js'),
		},
	});

	// Prevent window close if processing is active
	mainWindow.on('close', async (event) => {
		console.log('[Electron] Close event triggered. activeProcessing:', activeProcessing);
		if (activeProcessing) {
			event.preventDefault();

			const response = await dialog.showMessageBox(mainWindow, {
				type: 'warning',
				buttons: ['Keep processing', 'Quit app'],
				defaultId: 0,
				cancelId: 0,
				title: 'Processing in Progress',
				message: 'Video processing is currently active',
				detail: 'Closing the app will interrupt the current processing operation. You will need to restart it.\n\nAre you sure you want to quit?',
			});

			if (response.response === 1) {
				// User clicked "Quit app"
				activeProcessing = false;
				mainWindow.destroy();
			}
			// Otherwise do nothing (cancel close)
		}
	});

	// and load the index.html of the app.
	if (isDev) {
		mainWindow.loadURL(devServerUrl);
	} else {
		// Load the built frontend from frontend/dist
		mainWindow.loadFile(join(app.getAppPath(), 'frontend', 'dist', 'index.html'));
	}

	// Open the DevTools.
	//mainWindow.webContents.openDevTools();
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(async () => {
	// Register settings IPC (file-based).
	registerSettingsIpc({ ipcMain, app, shell });
	// Register local-first telemetry IPC handlers (opt-in, no uploading).
	registerTelemetryIpc({ ipcMain, app, shell });
	// Optional telemetry upload IPC (manual trigger; reads ONLY telemetry files).
	registerTelemetryUploadIpc({ ipcMain, app });

	// Start backend (always, including dev mode)
	startBackend();

	// Wait for backend to be ready before creating window
	const backendReady = await waitForBackend();
	if (!backendReady) {
		dialog.showErrorBox(
			'Backend Failed to Start',
			'The backend server failed to start. Please check the logs and try again.'
		);
		app.quit();
		return;
	}

	createWindow();

	// Initialize auto-updater (only in production)
	if (!isDev) {
		const windows = BrowserWindow.getAllWindows();
		if (windows.length > 0) {
			initAutoUpdater(windows[0], app);
		}
	}

	// On OS X it's common to re-create a window in the app when the
	// dock icon is clicked and there are no other windows open.
	app.on('activate', () => {
		if (BrowserWindow.getAllWindows().length === 0) {
			createWindow();
		}
	});
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
	if (process.platform !== 'darwin') {
		stopBackend();
		app.quit();
	}
});

// Clean up backend on app quit
app.on('before-quit', () => {
	stopBackend();
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.

// IPC handlers for file dialogs
ipcMain.handle('dialog:selectVideoFile', async (event) => {
	const parentWindow = BrowserWindow.fromWebContents(event.sender);
	const result = await dialog.showOpenDialog(parentWindow, {
		properties: ['openFile'],
		filters: [
			{ name: 'Videos', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm3u8'] },
			{ name: 'All Files', extensions: ['*'] },
		],
	});

	if (result.canceled) {
		return null;
	}

	return result.filePaths[0];
});

// IPC handler for multi-select file dialog
ipcMain.handle('dialog:selectVideoFiles', async (event) => {
	const parentWindow = BrowserWindow.fromWebContents(event.sender);
	const result = await dialog.showOpenDialog(parentWindow, {
		properties: ['openFile', 'multiSelections'],
		filters: [
			{ name: 'Videos', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm3u8'] },
			{ name: 'All Files', extensions: ['*'] },
		],
	});

	if (result.canceled) {
		return [];
	}

	return result.filePaths;
});

// IPC handler to check if file exists
ipcMain.handle('file:exists', async (event, filePath) => {
	try {
		return existsSync(filePath);
	} catch {
		return false;
	}
});

// Helper function to get video metadata (duration, resolution) using ffprobe
async function getVideoMetadata(filePath) {
	return new Promise((resolve) => {
		const ffprobe = spawn('ffprobe', [
			'-v', 'error',
			'-select_streams', 'v:0',
			'-show_entries', 'stream=width,height:format=duration',
			'-of', 'json',
			filePath
		]);

		let output = '';
		ffprobe.stdout.on('data', (data) => {
			output += data.toString();
		});

		ffprobe.on('close', (code) => {
			if (code === 0 && output.trim()) {
				try {
					const data = JSON.parse(output);
					const stream = data.streams?.[0] || {};
					const format = data.format || {};
					resolve({
						duration: format.duration ? parseFloat(format.duration) : null,
						width: stream.width || null,
						height: stream.height || null,
					});
				} catch {
					resolve({ duration: null, width: null, height: null });
				}
			} else {
				resolve({ duration: null, width: null, height: null });
			}
		});

		ffprobe.on('error', () => {
			resolve({ duration: null, width: null, height: null });
		});

		// Timeout after 5 seconds
		setTimeout(() => {
			ffprobe.kill();
			resolve({ duration: null, width: null, height: null });
		}, 5000);
	});
}

// Helper to format date/time nicely
function formatDateTime(date) {
	const d = new Date(date);
	const year = d.getFullYear();
	const month = String(d.getMonth() + 1).padStart(2, '0');
	const day = String(d.getDate()).padStart(2, '0');
	const hours = String(d.getHours()).padStart(2, '0');
	const minutes = String(d.getMinutes()).padStart(2, '0');
	return `${year}-${month}-${day} ${hours}:${minutes}`;
}

// IPC handler to get file metadata
ipcMain.handle('file:getMetadata', async (event, filePath) => {
	try {
		if (!existsSync(filePath)) {
			return null;
		}

		const stats = statSync(filePath);
		const fileName = filePath.split(/[/\\]/).pop();

		// Get video metadata using ffprobe
		const videoMeta = await getVideoMetadata(filePath);

		// Use the older date between birthtime and mtime (more likely the actual recording date)
		const birthtime = stats.birthtime && stats.birthtime.getTime() > 0 ? stats.birthtime : null;
		const mtime = stats.mtime;
		const fileDate = birthtime && birthtime < mtime ? birthtime : mtime;

		return {
			name: fileName,
			size: stats.size,
			sizeFormatted: formatFileSize(stats.size),
			created: fileDate.toISOString(),
			createdFormatted: formatDateTime(fileDate),
			duration: videoMeta.duration, // in seconds
			width: videoMeta.width,
			height: videoMeta.height,
			resolution: videoMeta.width && videoMeta.height ? `${videoMeta.width}x${videoMeta.height}` : null,
		};
	} catch (error) {
		console.error('Failed to get file metadata:', error);
		return null;
	}
});

// IPC handler to open URL in external browser
ipcMain.handle('shell:openExternal', async (event, url) => {
	try {
		await shell.openExternal(url);
		return true;
	} catch (error) {
		console.error('Failed to open external URL:', error);
		return false;
	}
});

// Confirm cancelling processing (renderer-triggered).
// Kept in main process so we can control button order consistently within the app.
ipcMain.handle('app:confirmCancelProcessing', async (event) => {
	const parentWindow = BrowserWindow.fromWebContents(event.sender);
	const response = await dialog.showMessageBox(parentWindow, {
		type: 'warning',
		buttons: ['Keep processing', 'Cancel processing'],
		defaultId: 0,
		cancelId: 0,
		title: 'Cancel Processing',
		message: 'Are you sure you want to cancel processing?',
		detail: 'This will stop the current transcoding operation.',
	});

	return response.response === 1;
});

// IPC handler to set processing state
ipcMain.handle('app:setProcessingState', async (event, isProcessing, origin = 'unknown') => {
	activeProcessing = isProcessing;
	// Only log on state changes
	if (isProcessing !== lastLoggedState) {
		console.log(`[Electron] Processing state changed: ${isProcessing} (from: ${origin})`);
		lastLoggedState = isProcessing;
	}
	return true;
});

// IPC handler to get app version
ipcMain.handle('app:getVersion', () => {
	return app.getVersion();
});

// IPC handler to check for updates
ipcMain.handle('updater:checkForUpdates', () => {
	if (!isDev) {
		checkForUpdates(true);
		return { success: true };
	}
	return { success: false, error: 'Updates not available in development mode' };
});

// Helper function to format file size
function formatFileSize(bytes) {
	if (bytes === 0) return '0 Bytes';
	const k = 1024;
	const sizes = ['Bytes', 'KB', 'MB', 'GB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));
	return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}
