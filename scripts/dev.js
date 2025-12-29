#!/usr/bin/env node

import { spawn } from 'child_process';
import { platform } from 'os';

// Set ELECTRON_DISABLE_SANDBOX on Linux
if (platform() === 'linux') {
	process.env.ELECTRON_DISABLE_SANDBOX = '1';
}

// Run the dev command
const child = spawn('npm', ['run', 'dev:base'], {
	stdio: 'inherit',
	shell: true,
	env: process.env,
});

child.on('exit', (code) => {
	process.exit(code);
});
