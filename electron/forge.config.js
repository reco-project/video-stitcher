// electron/forge.config.js
const path = require('path');
const { VitePlugin } = require('@electron-forge/plugin-vite');

module.exports = {
  packagerConfig: { asar: true },
  rebuildConfig: {},
  makers: [
    { name: '@electron-forge/maker-squirrel', config: {} },
    { name: '@electron-forge/maker-zip', platforms: ['darwin'] },
    { name: '@electron-forge/maker-deb', config: {} },
    { name: '@electron-forge/maker-rpm', config: {} },
  ],
  plugins: [
    new VitePlugin({
      build: [
        {
          entry: path.join(__dirname, 'main.js'),
          target: 'main',
        },
        {
          entry: path.join(__dirname, 'preload.js'),
          target: 'preload',
        },
      ],
      renderer: [
        {
          name: 'main_window',
          entry: path.resolve(__dirname, '../frontend/src/main.jsx'),
          config: path.resolve(__dirname, '../frontend/vite.config.js'),
        },
      ],
    }),
  ],
};
