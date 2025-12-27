// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

const { contextBridge, ipcRenderer } = require('electron');

// Expose file system APIs to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
	// Open file dialog and return selected file path
	selectVideoFile: () => ipcRenderer.invoke('dialog:selectVideoFile'),
});
