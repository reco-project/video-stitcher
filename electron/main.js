import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron';
import { join, dirname } from 'node:path';
import { existsSync, statSync } from 'node:fs';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import started from 'electron-squirrel-startup';

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

// Function to check if Vite dev server is running
async function viteDevServerRunning(url = 'http://localhost:5173') {
	try {
		const res = await fetch(url);
		return res.ok;
	} catch {
		return false;
	}
}

const isDev = await viteDevServerRunning();

const createWindow = () => {
	// Create the browser window.
	const mainWindow = new BrowserWindow({
		width: 800,
		height: 800,
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
		mainWindow.loadFile(join(__dirname, '../frontend/dist/index.html'));
	}

	// Open the DevTools.
	//mainWindow.webContents.openDevTools();
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
	createWindow();

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
		app.quit();
	}
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.

// IPC handlers for file dialogs
ipcMain.handle('dialog:selectVideoFile', async () => {
	const result = await dialog.showOpenDialog({
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
ipcMain.handle('dialog:selectVideoFiles', async () => {
	const result = await dialog.showOpenDialog({
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

// IPC handler to get file metadata
ipcMain.handle('file:getMetadata', async (event, filePath) => {
	try {
		if (!existsSync(filePath)) {
			return null;
		}

		const stats = statSync(filePath);
		const fileName = filePath.split(/[/\\]/).pop();

		return {
			name: fileName,
			size: stats.size,
			sizeFormatted: formatFileSize(stats.size),
			modified: stats.mtime.toISOString(),
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

// Helper function to format file size
function formatFileSize(bytes) {
	if (bytes === 0) return '0 Bytes';
	const k = 1024;
	const sizes = ['Bytes', 'KB', 'MB', 'GB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));
	return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}
