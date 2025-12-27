import React, { useState } from 'react';
import Viewer from '@/features/viewer/components/Viewer.jsx';
import Health from '@/features/health/components/Health.jsx';
import { useNavigateTo } from '../Router';
import matches from '@/data/matches.js';

export default function Home() {
	const [selectedMatch, setSelectedMatch] = useState(null);
	const navigate = useNavigateTo();

	return (
		<div className="flex flex-col items-center w-full p-4 gap-4">
			<h1 className="text-purple-600">Video Stitcher</h1>
			<p>Welcome — this is the renderer application root.</p>

			<div className="flex gap-2">
				<button
					className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition"
					onClick={navigate.toProfiles}
				>
					Manage Lens Profiles
				</button>
			</div>

			<Health />

			<div className="w-full max-w-2xl">
				<label className="block mb-2 font-bold">Select match</label>
				<select
					className="w-full p-2 rounded border"
					value={selectedMatch ? selectedMatch.id : ''}
					onChange={(e) => {
						const id = e.target.value;
						const m = matches.find((mm) => mm.id === id) || null;
						setSelectedMatch(m);
					}}
				>
					<option value="">-- choose match --</option>
					{matches.map((m) => (
						<option key={m.id} value={m.id}>
							{m.label}
						</option>
					))}
				</select>
			</div>

			{selectedMatch && (
				<section className={'w-full aspect-video h-full flex flex-col items-center align-middle'}>
					<Viewer key={selectedMatch.id} selectedMatch={selectedMatch} />
				</section>
			)}
		</div>
	);
}
