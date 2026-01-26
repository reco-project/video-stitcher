import { execSync } from 'node:child_process';
import { join } from 'node:path';
import os from 'node:os';

const isWin = os.platform() === 'win32';
const pythonPath = isWin ? join('venv', 'Scripts', 'python.exe') : join('venv', 'bin', 'python');

const ENTRY_POINT = 'app.main'; // format: module.file

// Note: USER_DATA_PATH is not set when running backend standalone via npm script
// The backend will fall back to using backend/data directory in this case
// USER_DATA_PATH is only set when backend is started by Electron (via electron/main.js)
const env = {
	...process.env,
};

try {
	execSync(`cd backend && ${pythonPath} -m ${ENTRY_POINT}`, {
		stdio: 'inherit',
		env: env
	});
} catch (error) {
	console.error('Error starting backend:', error);
}
