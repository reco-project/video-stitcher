#!/usr/bin/env node

/**
 * Generate latest.yml, latest-mac.yml, and latest-linux.yml files
 * for electron-updater to check for updates from GitHub releases.
 * 
 * This script should be run after the build artifacts are created.
 * It generates the metadata files that electron-updater needs to
 * detect and download new versions.
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Get version from package.json
const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'package.json'), 'utf8'));
const version = packageJson.version;

// Ensure version is set
if (!version) {
    console.error('Error: Version not found in package.json');
    process.exit(1);
}

console.log(`Generating latest yml files for version ${version}`);

/**
 * Calculate SHA512 hash of a file
 */
function calculateSha512(filePath) {
    const fileBuffer = fs.readFileSync(filePath);
    const hashSum = crypto.createHash('sha512');
    hashSum.update(fileBuffer);
    return hashSum.digest('base64');
}

/**
 * Get file size in bytes
 */
function getFileSize(filePath) {
    const stats = fs.statSync(filePath);
    return stats.size;
}

/**
 * Generate a latest yml file
 */
function generateLatestYml(outputPath, files, releaseDate = new Date().toISOString()) {
    const mainFile = files[0]; // First file is the main installer
    
    let content = `version: ${version}\n`;
    content += `files:\n`;
    
    files.forEach(file => {
        content += `  - url: ${file.name}\n`;
        content += `    sha512: ${file.sha512}\n`;
        content += `    size: ${file.size}\n`;
    });
    
    content += `path: ${mainFile.name}\n`;
    content += `sha512: ${mainFile.sha512}\n`;
    content += `releaseDate: '${releaseDate}'\n`;
    
    fs.writeFileSync(outputPath, content, 'utf8');
    console.log(`Generated: ${outputPath}`);
}

/**
 * Process artifacts for a platform
 */
function processArtifacts(makeDir, outputYmlPath, filePatterns) {
    if (!fs.existsSync(makeDir)) {
        console.warn(`Warning: ${makeDir} does not exist, skipping`);
        return false;
    }
    
    const files = [];
    
    for (const pattern of filePatterns) {
        const matches = findFiles(makeDir, pattern);
        for (const match of matches) {
            const relativeName = path.basename(match);
            const sha512 = calculateSha512(match);
            const size = getFileSize(match);
            
            files.push({
                name: relativeName,
                sha512: sha512,
                size: size,
                path: match
            });
            
            console.log(`  Found: ${relativeName} (${size} bytes)`);
        }
    }
    
    if (files.length === 0) {
        console.warn(`Warning: No files found matching patterns in ${makeDir}`);
        return false;
    }
    
    generateLatestYml(outputYmlPath, files);
    return true;
}

/**
 * Find files matching a pattern (simple glob)
 */
function findFiles(dir, pattern) {
    const results = [];
    
    function search(currentDir) {
        if (!fs.existsSync(currentDir)) return;
        
        const entries = fs.readdirSync(currentDir, { withFileTypes: true });
        
        for (const entry of entries) {
            const fullPath = path.join(currentDir, entry.name);
            
            if (entry.isDirectory()) {
                search(fullPath);
            } else if (entry.isFile()) {
                if (matchPattern(entry.name, pattern)) {
                    results.push(fullPath);
                }
            }
        }
    }
    
    search(dir);
    return results;
}

/**
 * Simple pattern matching (supports * wildcard)
 */
function matchPattern(filename, pattern) {
    const regexPattern = pattern
        .replace(/\./g, '\\.')
        .replace(/\*/g, '.*');
    const regex = new RegExp(`^${regexPattern}$`);
    return regex.test(filename);
}

// Main execution
const outDir = path.join(__dirname, '..', 'out');

if (!fs.existsSync(outDir)) {
    console.error('Error: out directory does not exist. Run electron-forge make first.');
    process.exit(1);
}

console.log('\n=== Generating Windows latest.yml ===');
const windowsMakeDir = path.join(outDir, 'make', 'squirrel.windows', 'x64');
const windowsYml = path.join(outDir, 'make', 'latest.yml');
processArtifacts(
    windowsMakeDir,
    windowsYml,
    ['*Setup.exe', '*.nupkg']
);

console.log('\n=== Generating macOS latest-mac.yml ===');
const macMakeDir = path.join(outDir, 'make', 'zip', 'darwin', 'arm64');
const macYml = path.join(outDir, 'make', 'latest-mac.yml');
processArtifacts(
    macMakeDir,
    macYml,
    ['*.zip']
);

console.log('\n=== Generating Linux latest-linux.yml ===');
const linuxMakeDir = path.join(outDir, 'make');
const linuxYml = path.join(outDir, 'make', 'latest-linux.yml');

// For Linux, we want to use the zip file (most universal)
const linuxZipDir = path.join(linuxMakeDir, 'zip', 'linux', 'x64');
if (fs.existsSync(linuxZipDir)) {
    processArtifacts(
        linuxZipDir,
        linuxYml,
        ['*.zip']
    );
} else {
    console.warn('Linux zip not found, skipping latest-linux.yml');
}

console.log('\n=== Done ===');
console.log('Upload these files to your GitHub release alongside the installers:');
console.log('  - latest.yml (for Windows)');
console.log('  - latest-mac.yml (for macOS)');
console.log('  - latest-linux.yml (for Linux)');
