// electron/forge.config.cjs
const path = require('path');
const fs = require('fs');
const { VitePlugin } = require('@electron-forge/plugin-vite');

module.exports = {
  packagerConfig: {
    asar: false, // Disable asar - PyInstaller bundle needs direct filesystem access
    name: 'Video Stitcher',
    executableName: 'video-stitcher',
    icon: path.join(__dirname, 'resources', 'icon'),
    appBundleId: 'com.reco.video-stitcher',
    appCopyright: 'Copyright © 2026 Mohamed Taha GUELZIM',
    // macOS code signing (requires Apple Developer certificate)
    osxSign: {
      identity: process.env.APPLE_SIGNING_IDENTITY || null,
      'hardened-runtime': true,
      entitlements: path.join(__dirname, 'entitlements.mac.plist'),
      'entitlements-inherit': path.join(__dirname, 'entitlements.mac.plist'),
      'signature-flags': 'library',
    },
    // macOS notarization (requires Apple Developer account secrets)
    ...(process.env.APPLE_ID && process.env.APPLE_ID_PASSWORD && process.env.APPLE_TEAM_ID ? {
      osxNotarize: {
        appleId: process.env.APPLE_ID,
        appleIdPassword: process.env.APPLE_ID_PASSWORD,
        teamId: process.env.APPLE_TEAM_ID,
      },
    } : {}),
    ignore: (filePath) => {
      // Always include root
      if (!filePath) return false;

      // Include package.json, .vite folder (Vite output)
      if (filePath === '/package.json') return false;
      if (filePath.startsWith('/.vite')) return false;

      // Exclude everything else from root that we don't need
      const excludePatterns = [
        /^\/backend/,        // Backend source (PyInstaller bundle is copied separately)
        /^\/frontend/,       // Frontend source (Vite builds to .vite)
        /^\/scripts/,
        /^\/docs/,
        /^\/\.git/,
        /^\/\.github/,
        /^\/node_modules/,
        /^\/electron\/(?!.vite)/,  // Exclude electron source except .vite output
        /\.md$/,
        /\.log$/,
        /^\/LICENSE$/,
      ];

      return excludePatterns.some(pattern => pattern.test(filePath));
    },
  },
  hooks: {
    // Cleanup is done in the GitHub Actions workflow after packaging
    // because VitePlugin doesn't work well with packagerConfig.ignore or packageAfterCopy hooks
    postPackage: async (config, options) => {
      const platform = options.platform;

      // Create a wrapper script for Linux to set ELECTRON_DISABLE_SANDBOX
      if (platform === 'linux') {
        const appPath = options.outputPaths[0];
        const originalBinary = path.join(appPath, 'video-stitcher');
        const realBinary = path.join(appPath, 'video-stitcher.bin');

        // Rename original binary
        if (fs.existsSync(originalBinary) && !fs.existsSync(realBinary)) {
          fs.renameSync(originalBinary, realBinary);

          // Create wrapper script that resolves symlinks and preserves GPU access
          // - Source profile to get CUDA/GPU library paths
          // - readlink -f resolves the full path even through symlinks
          const wrapperScript = `#!/bin/bash
# Source profile for CUDA/GPU library paths if launching from desktop
if [ -z "$LD_LIBRARY_PATH" ] && [ -f /etc/profile ]; then
    source /etc/profile 2>/dev/null || true
fi
export ELECTRON_DISABLE_SANDBOX=1
SCRIPT_PATH="$(readlink -f "$0")"
DIR="$(dirname "$SCRIPT_PATH")"
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
        // Creates Setup.exe installer
        authors: 'Mohamed Taha GUELZIM',
        description: 'Professional video stitching application',
      },
    },
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin', 'linux'],
    },
    {
      name: '@electron-forge/maker-deb',
      platforms: ['linux'],
      config: {
        options: {
          icon: path.join(__dirname, 'resources', 'icon.png'),
          maintainer: 'Mohamed Taha GUELZIM <mohamedtahaguelzim@gmail.com>',
          homepage: 'https://github.com/reco-project/video-stitcher',
          name: 'video-stitcher',
          productName: 'Video Stitcher',
          genericName: 'Video Editor',
          description: 'Professional video stitching application for creating panoramic and 360° videos',
          categories: ['AudioVideo', 'Video', 'Graphics'],
          section: 'video',
        },
      },
    },
    {
      name: '@electron-forge/maker-rpm',
      platforms: ['linux'],
      config: {
        options: {
          icon: path.join(__dirname, 'resources', 'icon.png'),
          name: 'video-stitcher',
          productName: 'Video Stitcher',
          genericName: 'Video Editor',
          description: 'Professional video stitching application for creating panoramic and 360° videos',
          categories: ['AudioVideo', 'Video', 'Graphics'],
          license: 'AGPL-3.0',
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
          config: path.join(__dirname, 'vite.main.config.mjs'),
        },
        {
          entry: path.join(__dirname, 'preload.js'),
          config: path.join(__dirname, 'vite.preload.config.mjs'),
        },
      ],
      renderer: [
        {
          name: 'main_window',
          config: path.resolve(__dirname, '../frontend/vite.config.mjs'),
        },
      ],
    }),
  ],
};
