import { execSync } from 'node:child_process';
import { join } from 'node:path';
import os from 'node:os';

const isWin = os.platform() === 'win32';
const pythonPath = join('venv', isWin ? 'Scripts' : 'bin', isWin ? 'python.exe' : 'python');

const args = process.argv.slice(2); // pass extra args like --check

try {
	// check if venv exists
	execSync(`${pythonPath} --version`, {
		cwd: 'backend',
		stdio: 'ignore',
	});
	// if venv is not found, log error and exit
} catch (err) {
	console.error('Virtual environment not found. Please run "npm run setup or npm run backend-install" first.');
	process.exit(1);
}

try {
	// check if black is installed
	execSync(`${pythonPath} -m black --help`, {
		cwd: 'backend',
		stdio: 'ignore',
	});
	// if black is not found, log error and exit
} catch (err) {
	console.error('Black is not installed. Please run "npm run setup or npm run backend-install" first.');
	process.exit(1);
}

try {
	execSync(`${pythonPath} -m black ${args.join(' ') || '.'}`, {
		cwd: 'backend',
		stdio: 'inherit',
	});
} catch (err) {
	process.exit(1); // exit non-zero if formatting fails
}
