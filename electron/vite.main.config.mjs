import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import { builtinModules } from 'node:module';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    lib: {
      entry: path.resolve(__dirname, 'main.js'),
      formats: ['es'],
      fileName: () => 'main.js',
    },
    rollupOptions: {
      external: [
        'electron',
        'electron-updater',
        'electron-squirrel-startup',
        ...builtinModules,
        ...builtinModules.map(m => `node:${m}`),
      ],
    },
    outDir: path.resolve(__dirname, '../.vite/build'),
    emptyOutDir: false,
    target: 'node20',
    ssr: true, // Enable SSR mode for Node.js environment
  },
  resolve: {
    conditions: ['node'],
  },
});
