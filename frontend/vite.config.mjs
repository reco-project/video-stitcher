import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import jsConfigPaths from 'vite-jsconfig-paths';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ command }) => ({
	plugins: [react(), tailwindcss(), jsConfigPaths()],
	base: './', // Use relative paths for Electron compatibility
	root: __dirname, // Ensure Vite finds index.html in the frontend directory
	build: {
		// For standalone frontend dev, use 'dist'. For electron-forge packaging,
		// the VitePlugin sets outDir to .vite/renderer/main_window at project root
		outDir: command === 'serve' ? 'dist' : path.resolve(__dirname, '../.vite/renderer/main_window'),
	},
}));
