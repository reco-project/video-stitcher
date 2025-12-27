import React, { useState } from 'react';
import ProfileList from './ProfileList';
import ProfileDetail from './ProfileDetail';
import ProfileBrowser from './ProfileBrowser';

export default function ProfileManager() {
	const [selectedProfileId, setSelectedProfileId] = useState(null);
	const [viewMode, setViewMode] = useState('list'); // 'list' or 'browse'

	const handleSelectProfile = (profile) => {
		setSelectedProfileId(profile.id);
	};

	return (
		<div className="w-full max-w-6xl">
			<div className="flex items-center justify-between mb-4">
				<h2 className="text-xl font-bold">Lens Profile Manager</h2>
				<div className="flex gap-2">
					<button
						className={`px-3 py-1 rounded ${
							viewMode === 'list' ? 'bg-purple-600 text-white' : 'bg-gray-200'
						}`}
						onClick={() => setViewMode('list')}
					>
						All Profiles
					</button>
					<button
						className={`px-3 py-1 rounded ${
							viewMode === 'browse' ? 'bg-purple-600 text-white' : 'bg-gray-200'
						}`}
						onClick={() => setViewMode('browse')}
					>
						Browse by Camera
					</button>
				</div>
			</div>

			<div className="grid md:grid-cols-2 gap-4">
				<div>
					{viewMode === 'list' ? (
						<ProfileList onSelect={handleSelectProfile} />
					) : (
						<ProfileBrowser onSelect={handleSelectProfile} />
					)}
				</div>
				<div>
					<ProfileDetail profileId={selectedProfileId} />
				</div>
			</div>
		</div>
	);
}
