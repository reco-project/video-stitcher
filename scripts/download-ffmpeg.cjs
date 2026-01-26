#!/usr/bin/env node
/**
 * Downloads FFmpeg binaries for the target platform.
 * 
 * Usage:
 *   node scripts/download-ffmpeg.js [platform]
 * 
 * Platform can be: linux, win32, darwin (defaults to current platform)
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

const FFMPEG_SOURCES = {
  'linux-x64': {
    url: 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz',
    extract: 'tar',
    binaries: ['ffmpeg', 'ffprobe'],
  },
  'win32-x64': {
    url: 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
    extract: 'zip',
    binaries: ['ffmpeg.exe', 'ffprobe.exe'],
  },
  'darwin-x64': {
    url: 'https://evermeet.cx/ffmpeg/getrelease/zip',
    urlProbe: 'https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip',
    extract: 'zip',
    binaries: ['ffmpeg', 'ffprobe'],
  },
  'darwin-arm64': {
    // For Apple Silicon, use the same URL (universal binary)
    url: 'https://evermeet.cx/ffmpeg/getrelease/zip',
    urlProbe: 'https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip',
    extract: 'zip',
    binaries: ['ffmpeg', 'ffprobe'],
  },
};

function getPlatformKey() {
  const platform = process.argv[2] || process.platform;
  const arch = process.arch;

  // Normalize platform names
  let key = `${platform}-${arch}`;

  // Fallback for common cases
  if (platform === 'linux') key = 'linux-x64';
  if (platform === 'win32') key = 'win32-x64';
  if (platform === 'darwin') key = arch === 'arm64' ? 'darwin-arm64' : 'darwin-x64';

  return key;
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    console.log(`Downloading: ${url}`);

    const file = fs.createWriteStream(dest);
    const protocol = url.startsWith('https') ? https : http;

    const request = protocol.get(url, (response) => {
      // Handle redirects (301, 302, 303, 307, 308)
      if ([301, 302, 303, 307, 308].includes(response.statusCode)) {
        file.close();
        fs.unlinkSync(dest);
        // Resolve relative redirects against the original URL
        let redirectUrl = response.headers.location;
        if (!redirectUrl.startsWith('http')) {
          const urlObj = new URL(url);
          redirectUrl = `${urlObj.protocol}//${urlObj.host}${redirectUrl}`;
        }
        console.log(`Redirecting to: ${redirectUrl}`);
        return download(redirectUrl, dest).then(resolve).catch(reject);
      }

      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download: ${response.statusCode}`));
        return;
      }

      const totalSize = parseInt(response.headers['content-length'], 10);
      let downloadedSize = 0;

      response.on('data', (chunk) => {
        downloadedSize += chunk.length;
        if (totalSize) {
          const percent = Math.round((downloadedSize / totalSize) * 100);
          process.stdout.write(`\rProgress: ${percent}% (${(downloadedSize / 1024 / 1024).toFixed(1)}MB)`);
        }
      });

      response.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log('\nDownload complete!');
        resolve();
      });
    });

    request.on('error', (err) => {
      fs.unlink(dest, () => { });
      reject(err);
    });
  });
}

// Recursively find a file by name in a directory
function findFile(dir, filename) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findFile(fullPath, filename);
      if (found) return found;
    } else if (entry.name.toLowerCase() === filename.toLowerCase()) {
      return fullPath;
    }
  }
  return null;
}

async function extractAndMove(archivePath, binDir, source) {
  const tempDir = path.join(path.dirname(archivePath), 'ffmpeg-temp');
  const isWindows = process.platform === 'win32';

  // Create temp directory
  if (fs.existsSync(tempDir)) {
    fs.rmSync(tempDir, { recursive: true });
  }
  fs.mkdirSync(tempDir, { recursive: true });

  console.log('Extracting...');

  if (source.extract === 'tar') {
    execSync(`tar xf "${archivePath}" -C "${tempDir}"`, { stdio: 'inherit' });
  } else if (source.extract === 'zip') {
    if (isWindows) {
      // Use PowerShell on Windows
      execSync(`powershell -command "Expand-Archive -Path '${archivePath}' -DestinationPath '${tempDir}'"`, { stdio: 'inherit' });
    } else {
      execSync(`unzip -q "${archivePath}" -d "${tempDir}"`, { stdio: 'inherit' });
    }
  }

  // Find and move binaries
  console.log('Moving binaries...');

  for (const binary of source.binaries) {
    // Find the binary in extracted files using our cross-platform function
    const foundPath = findFile(tempDir, binary);

    if (foundPath) {
      const destPath = path.join(binDir, binary);
      fs.copyFileSync(foundPath, destPath);
      if (!isWindows) {
        fs.chmodSync(destPath, 0o755);
      }
      console.log(`  ${binary} -> ${destPath}`);
    } else {
      console.warn(`  Warning: ${binary} not found in archive`);
    }
  }

  // Cleanup
  fs.rmSync(tempDir, { recursive: true });
  fs.unlinkSync(archivePath);
}

async function main() {
  const platformKey = getPlatformKey();
  const source = FFMPEG_SOURCES[platformKey];

  if (!source) {
    console.error(`Unsupported platform: ${platformKey}`);
    console.error('Supported platforms:', Object.keys(FFMPEG_SOURCES).join(', '));
    process.exit(1);
  }

  console.log(`\n=== Downloading FFmpeg for ${platformKey} ===\n`);

  const binDir = path.join(__dirname, '..', 'backend', 'bin');

  // Create bin directory
  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }

  // Check if already downloaded
  const allExist = source.binaries.every(b => fs.existsSync(path.join(binDir, b)));
  if (allExist) {
    console.log('FFmpeg binaries already exist. Skipping download.');
    console.log('Delete backend/bin/ to force re-download.');
    return;
  }

  // Download main archive
  const ext = source.extract === 'tar' ? '.tar.xz' : '.zip';
  const archivePath = path.join(binDir, `ffmpeg${ext}`);

  await download(source.url, archivePath);
  await extractAndMove(archivePath, binDir, source);

  // For macOS, ffprobe is a separate download
  if (source.urlProbe) {
    const probeArchive = path.join(binDir, 'ffprobe.zip');
    await download(source.urlProbe, probeArchive);
    await extractAndMove(probeArchive, binDir, { extract: 'zip', binaries: ['ffprobe'] });
  }

  console.log('\n=== FFmpeg download complete ===\n');

  // Verify
  for (const binary of source.binaries) {
    const binPath = path.join(binDir, binary);
    if (fs.existsSync(binPath)) {
      console.log(`✓ ${binary}`);
    } else {
      console.log(`✗ ${binary} (missing!)`);
    }
  }
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
