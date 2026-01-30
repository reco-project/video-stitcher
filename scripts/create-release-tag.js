#!/usr/bin/env node

/**
 * Script to create a release tag matching the version in package.json
 * This will create a tag on the latest commit of the main branch
 */

import { readFileSync } from 'fs';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function exec(command, options = {}) {
  try {
    const result = execSync(command, { 
      encoding: 'utf-8', 
      stdio: options.silent ? 'pipe' : 'inherit',
      ...options 
    });
    return result ? result.trim() : '';
  } catch (error) {
    console.error(`Error executing: ${command}`);
    console.error(error.message);
    if (error.stdout) console.error(error.stdout);
    if (error.stderr) console.error(error.stderr);
    throw error;
  }
}

function main() {
  console.log('Creating release tag from package.json version...\n');

  // Read version from package.json
  const packageJsonPath = join(__dirname, '..', 'package.json');
  const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf-8'));
  const version = packageJson.version;
  const tag = `v${version}`;

  console.log(`Version from package.json: ${version}`);
  console.log(`Tag to create: ${tag}\n`);

  // Check if we're on the right branch or have access to main
  try {
    exec('git fetch origin main', { silent: true });
    console.log('Fetched latest main branch\n');
  } catch (error) {
    console.warn('Warning: Could not fetch main branch');
  }

  // Get the latest commit SHA from main
  let mainCommit;
  try {
    mainCommit = exec('git rev-parse main', { silent: true });
    console.log(`Latest commit on main: ${mainCommit}\n`);
  } catch (error) {
    console.error('Error: Could not find main branch');
    console.error('Make sure you have fetched the main branch or are on main');
    process.exit(1);
  }

  // Check if tag already exists
  try {
    const existingTagObject = exec(`git rev-parse ${tag}`, { silent: true });
    // For annotated tags, get the commit it points to
    const existingTagCommit = exec(`git rev-parse ${tag}^{commit}`, { silent: true });
    
    if (existingTagCommit) {
      console.log(`Tag ${tag} already exists`);
      console.log(`Tag object: ${existingTagObject}`);
      console.log(`Points to commit: ${existingTagCommit}`);
      
      if (existingTagCommit === mainCommit) {
        console.log('Tag is already pointing to the latest main commit.');
        
        // Check if tag exists on remote
        const remoteTag = exec(`git ls-remote --tags origin ${tag}`, { silent: true });
        if (remoteTag) {
          console.log('Tag already exists on remote. Nothing to do.');
          return;
        } else {
          console.log('Pushing tag to origin...\n');
          exec(`git push origin ${tag}`);
          console.log('\nTag pushed successfully!');
          return;
        }
      } else {
        console.error(`Error: Tag ${tag} exists but points to a different commit`);
        console.error(`Existing: ${existingTagCommit}`);
        console.error(`Expected: ${mainCommit}`);
        console.error('Please delete the existing tag first if you want to recreate it.');
        process.exit(1);
      }
    }
  } catch (error) {
    // Tag doesn't exist, which is what we want
    console.log(`Tag ${tag} does not exist yet. Creating...\n`);
  }

  // Create the tag
  console.log(`Creating tag ${tag} at commit ${mainCommit}...`);
  exec(`git tag -a ${tag} ${mainCommit} -m "Release ${version}"`);
  console.log('Tag created successfully!\n');

  // Push the tag to origin
  console.log('Pushing tag to origin...');
  exec(`git push origin ${tag}`);
  console.log('\nTag pushed successfully!');
  console.log(`\nThe tag ${tag} has been created and pushed.`);
  console.log('This will trigger the build workflow to create the release.');
  console.log(`Check the Actions tab: https://github.com/reco-project/video-stitcher/actions`);
}

main();
