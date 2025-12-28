import { execSync } from 'node:child_process';
import { join } from 'node:path';
import os from 'node:os';

const isWin = os.platform() === 'win32';
const pythonPath = join('venv', isWin ? 'Scripts' : 'bin', isWin ? 'python.exe' : 'python');

try {
	// check if venv exists
	execSync(`${pythonPath} --version`, {
		cwd: 'backend',
		stdio: 'ignore',
	});
} catch (err) {
	console.error('Virtual environment not found. Please run "npm run setup" or "npm run backend-install" first.');
	process.exit(1);
}

try {
	// check if pytest is installed
	execSync(`${pythonPath} -m pytest --version`, {
		cwd: 'backend',
		stdio: 'ignore',
	});
} catch (err) {
	console.error('pytest is not installed. Please run "npm run setup" or "npm run backend-install" first.');
	process.exit(1);
}

try {
	console.log('Running backend tests...\n');
	execSync(`${pythonPath} -m pytest tests/ -v`, {
		cwd: 'backend',
		stdio: 'inherit',
	});
	console.log('\n✓ Backend tests passed');
} catch (err) {
	console.error('\n✗ Backend tests failed');
	process.exit(1);
}
