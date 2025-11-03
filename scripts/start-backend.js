import { execSync } from 'node:child_process';
import { join } from 'node:path';
import os from 'node:os';

const isWin = os.platform() === 'win32';
const pythonPath = isWin ? join('venv', 'Scripts', 'python.exe') : join('venv', 'bin', 'python');

const ENTRY_POINT = 'app.main'; // format: module.file

try {
	execSync(`cd backend && ${pythonPath} -m ${ENTRY_POINT}`, { stdio: 'inherit' });
} catch (error) {
	console.error('Error starting backend:', error);
}
