import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import { join, dirname } from 'node:path';
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
