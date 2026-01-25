// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

const { contextBridge, ipcRenderer } = require('electron');

// Expose file system APIs to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
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
	// Set processing state to prevent accidental close
	setProcessingState: (isProcessing, origin) =>
		ipcRenderer.invoke('app:setProcessingState', isProcessing, origin),
	// Confirm cancelling processing (used by Processing page)
	confirmCancelProcessing: () => ipcRenderer.invoke('app:confirmCancelProcessing'),
	// Telemetry (local-first, opt-in)
	//
	// Renderer can only append events through IPC; it cannot write arbitrary files.
	// There is no uploading code in this repo version.
	telemetryGetInfo: () => ipcRenderer.invoke('telemetry:getInfo'),
	telemetryOpenFolder: () => ipcRenderer.invoke('telemetry:openFolder'),
	telemetryTrack: (payload) => ipcRenderer.invoke('telemetry:track', payload),
	telemetryUploadNow: (payload) => ipcRenderer.invoke('telemetry:uploadNow', payload), telemetryDeleteLocal: () => ipcRenderer.invoke('telemetry:deleteLocal'), telemetryResetClientId: () => ipcRenderer.invoke('telemetry:resetClientId'), openUserDataFolder: () => ipcRenderer.invoke('settings:openUserDataFolder'),
});
