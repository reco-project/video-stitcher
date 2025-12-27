import React, { useState } from 'react';
import ProfileDetail from './ProfileDetail';
import ProfileBrowser from './ProfileBrowser';
import ProfileForm from './ProfileForm';
import { useProfileMutations } from '../hooks/useProfiles';
import { Button } from '@/components/ui/button';

export default function ProfileManager() {
	const [selectedProfileId, setSelectedProfileId] = useState(null);
	const [showForm, setShowForm] = useState(false);
	const [editingProfile, setEditingProfile] = useState(null);

	const { create, update, delete: deleteProfile } = useProfileMutations();

	const handleSelectProfile = (profile) => {
		setSelectedProfileId(profile.id);
		setShowForm(false);
	};

	const handleCreateNew = () => {
		setEditingProfile(null);
		setSelectedProfileId(null);
		setShowForm(true);
	};

	const handleEdit = (profile) => {
		setEditingProfile(profile);
		setShowForm(true);
	};

	const handleDelete = async (profileId) => {
		if (!confirm('Are you sure you want to delete this profile?')) {
			return;
		}

		try {
			await deleteProfile(profileId);
			setSelectedProfileId(null);
		} catch (err) {
			alert(`Failed to delete profile: ${err.message}`);
		}
	};

	const handleFormSubmit = async (profileData) => {
		try {
			if (editingProfile) {
				await update(editingProfile.id, profileData);
			} else {
				await create(profileData);
			}
			setShowForm(false);
			setEditingProfile(null);
		} catch (err) {
			alert(`Failed to save profile: ${err.message}`);
		}
	};

	const handleFormCancel = () => {
		setShowForm(false);
		setEditingProfile(null);
	};

	return (
		<div className="w-full max-w-6xl">
			<div className="flex items-center justify-between mb-4">
				<h2 className="text-xl font-bold">Lens Profile Manager</h2>
				{!showForm && <Button onClick={handleCreateNew}>+ Create New Profile</Button>}
			</div>

			{showForm ? (
				<ProfileForm profile={editingProfile} onSubmit={handleFormSubmit} onCancel={handleFormCancel} />
			) : (
				<div className="grid md:grid-cols-2 gap-4">
					<div>
						<ProfileBrowser onSelect={handleSelectProfile} />
					</div>
					<div>
						<ProfileDetail profileId={selectedProfileId} onEdit={handleEdit} onDelete={handleDelete} />
					</div>
				</div>
			)}
		</div>
	);
}
