// electron/forge.config.js
const path = require('path');
const fs = require('fs');
const { VitePlugin } = require('@electron-forge/plugin-vite');

module.exports = {
  packagerConfig: {
    asar: {
      // Unpack backend so Python can be executed
      unpack: '**/backend/**',
    },
    name: 'Video Stitcher',
    executableName: 'video-stitcher',
    icon: path.join(__dirname, 'resources', 'icon'),
    appBundleId: 'com.reco.video-stitcher',
    appCopyright: 'Copyright © 2026 Mohamed Taha GUELZIM',
  },
  hooks: {
    postPackage: async (config, options) => {
      // Debug: write to file
      fs.writeFileSync('/tmp/forge-hook-debug.txt', JSON.stringify(options, null, 2));
      
      // Create a wrapper script for Linux to set ELECTRON_DISABLE_SANDBOX
      if (options.platform === 'linux') {
        const appPath = options.outputPaths[0];
        const originalBinary = path.join(appPath, 'video-stitcher');
        const realBinary = path.join(appPath, 'video-stitcher.bin');
        
        // Rename original binary
        if (fs.existsSync(originalBinary) && !fs.existsSync(realBinary)) {
          fs.renameSync(originalBinary, realBinary);
          
          // Create wrapper script
          const wrapperScript = `#!/bin/bash
export ELECTRON_DISABLE_SANDBOX=1
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/video-stitcher.bin" "$@"
`;
          fs.writeFileSync(originalBinary, wrapperScript, { mode: 0o755 });
        }
      }
    },
  },
  rebuildConfig: {},
  makers: [
    {
      name: '@electron-forge/maker-squirrel',
      platforms: ['win32'],
      config: {
        name: 'VideoStitcher',
        setupIcon: path.join(__dirname, 'resources', 'icon.ico'),
      },
    },
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin', 'win32'],
    },
    {
      name: '@reforged/maker-appimage',
      platforms: ['linux'],
      config: {
        options: {
          icon: path.join(__dirname, 'resources', 'icon.png'),
          name: 'Video Stitcher',
          genericName: 'Video Editor',
          categories: ['AudioVideo', 'Video'],
        },
      },
    },
    {
      name: '@electron-forge/maker-deb',
      platforms: ['linux'],
      config: {
        options: {
          icon: path.join(__dirname, 'resources', 'icon.png'),
          maintainer: 'Mohamed Taha GUELZIM',
          homepage: 'https://github.com/reco-project/video-stitcher',
        },
      },
    },
    {
      name: '@electron-forge/maker-rpm',
      platforms: ['linux'],
      config: {
        options: {
          icon: path.join(__dirname, 'resources', 'icon.png'),
        },
      },
    },
  ],
  publishers: [
    {
      name: '@electron-forge/publisher-github',
      config: {
        repository: {
          owner: 'reco-project',
          name: 'video-stitcher',
        },
        prerelease: false,
        draft: true,
      },
    },
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
          config: path.resolve(__dirname, '../frontend/vite.config.mjs'),
        },
      ],
    }),
  ],
};
