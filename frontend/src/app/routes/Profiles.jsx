import React from 'react';
import ProfileManager from '@/features/profiles/components/ProfileManager';
import Health from '@/features/health/components/Health';
import { useNavigateTo } from '../Router';

export default function Profiles() {
	const navigate = useNavigateTo();

	return (
		<div className="flex flex-col items-center w-full p-4 gap-4">
			<h1 className="text-purple-600">Lens Profiles</h1>
			<p>Manage camera lens calibration profiles for distortion correction</p>

			<div className="flex gap-2">
				<button
					className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition"
					onClick={navigate.toHome}
				>
					← Back to Home
				</button>
			</div>

			<Health />

			<ProfileManager />
		</div>
	);
}
