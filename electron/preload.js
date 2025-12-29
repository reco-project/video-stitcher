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
});
