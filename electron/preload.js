// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

const { contextBridge, ipcRenderer, webUtils } = require('electron');

// Expose file system APIs to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
	// Get file path from a dropped File object (required with contextIsolation)
	getPathForFile: (file) => webUtils.getPathForFile(file),
	// Open file dialog and return selected file path
	selectVideoFile: () => ipcRenderer.invoke('dialog:selectVideoFile'),
	// Open file dialog with multi-select and return array of paths
	selectVideoFiles: () => ipcRenderer.invoke('dialog:selectVideoFiles'),
	// Check if a file exists
	fileExists: (filePath) => ipcRenderer.invoke('file:exists', filePath),
	// Get file metadata
	getFileMetadata: (filePath) => ipcRenderer.invoke('file:getMetadata', filePath),
	// Open URL in default browser
	openExternal: (url) => ipcRenderer.invoke('shell:openExternal', url),
	// Settings (file-based)
	readSettings: () => ipcRenderer.invoke('settings:read'),
	writeSettings: (settings) => ipcRenderer.invoke('settings:write', settings),
	updateSetting: (key, value) => ipcRenderer.invoke('settings:write', { [key]: value }),
	getEncoderInfo: () => ipcRenderer.invoke('settings:getEncoderInfo'),
	openUserDataFolder: () => ipcRenderer.invoke('settings:openUserDataFolder'),
	clearUserDataFolder: () => ipcRenderer.invoke('settings:clearUserDataFolder'),
	// Set processing state to prevent accidental close
	setProcessingState: (isProcessing, origin) =>
		ipcRenderer.invoke('app:setProcessingState', isProcessing, origin),
	// Confirm cancelling processing (used by Processing page)
	confirmCancelProcessing: () => ipcRenderer.invoke('app:confirmCancelProcessing'),
	// Telemetry (local-first, opt-in)
	telemetryGetInfo: () => ipcRenderer.invoke('telemetry:getInfo'),
	telemetryOpenFolder: () => ipcRenderer.invoke('telemetry:openFolder'),
	telemetryTrack: (payload) => ipcRenderer.invoke('telemetry:track', payload),
	telemetryUploadNow: (payload) => ipcRenderer.invoke('telemetry:uploadNow', payload),
	telemetryDeleteLocal: () => ipcRenderer.invoke('telemetry:deleteLocal'),
	telemetryResetClientId: () => ipcRenderer.invoke('telemetry:resetClientId'),
	// App info
	getAppVersion: () => ipcRenderer.invoke('app:getVersion'),
	// Auto-updater
	checkForUpdates: () => ipcRenderer.invoke('updater:checkForUpdates'),
	// Backend status events
	onBackendReconnected: (callback) => {
		ipcRenderer.on('backend-reconnected', callback);
		// Return cleanup function
		return () => ipcRenderer.removeListener('backend-reconnected', callback);
	},
});
