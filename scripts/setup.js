import { execSync } from 'node:child_process';
import { join } from 'node:path';
import os from 'node:os';

const isWin = os.platform() === 'win32';
const pythonPath = isWin ? join('venv', 'Scripts', 'python.exe') : join('venv', 'bin', 'python');

try {
	// create venv
	execSync('python -m venv venv', { cwd: 'backend', stdio: 'inherit' });

	// upgrade pip
	execSync(`${pythonPath} -m pip install --upgrade pip`, { cwd: 'backend', stdio: 'inherit' });

	// install requirements
	execSync(`${pythonPath} -m pip install -r requirements.txt`, { cwd: 'backend', stdio: 'inherit' });

	console.log('Backend setup completed successfully!');
} catch (err) {
	console.error(err);
	process.exit(1);
}
