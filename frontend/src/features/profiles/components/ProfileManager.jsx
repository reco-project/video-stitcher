import React, { useState } from 'react';
import ProfileDetail from './ProfileDetail';
import ProfileBrowser from './ProfileBrowser';
import ProfileForm from './ProfileForm';
import { useProfileMutations } from '../hooks/useProfiles';
import { Button } from '@/components/ui/button';
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
	DialogFooter,
} from '@/components/ui/dialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';

export default function ProfileManager() {
	const [selectedProfileId, setSelectedProfileId] = useState(null);
	const [showForm, setShowForm] = useState(false);
	const [editingProfile, setEditingProfile] = useState(null);
	const [deleteDialog, setDeleteDialog] = useState({ open: false, profileId: null, profileName: '' });
	const [errorDialog, setErrorDialog] = useState({ open: false, message: '' });

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

	const handleDelete = async (profileId, profileName) => {
		setDeleteDialog({ open: true, profileId, profileName: profileName || profileId });
	};

	const confirmDelete = async () => {
		const { profileId } = deleteDialog;
		setDeleteDialog({ open: false, profileId: null, profileName: '' });

		try {
			await deleteProfile(profileId);
			setSelectedProfileId(null);
		} catch (err) {
			setErrorDialog({ open: true, message: `Failed to delete profile: ${err.message}` });
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
			setErrorDialog({ open: true, message: `Failed to save profile: ${err.message}` });
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
						<ProfileBrowser onSelect={handleSelectProfile} selectedProfileId={selectedProfileId} />
					</div>
					<div>
						<ProfileDetail profileId={selectedProfileId} onEdit={handleEdit} onDelete={handleDelete} />
					</div>
				</div>
			)}

			{/* Delete Confirmation Dialog */}
			<Dialog
				open={deleteDialog.open}
				onOpenChange={(open) => !open && setDeleteDialog({ open: false, profileId: null, profileName: '' })}
			>
				<DialogContent>
					<DialogHeader>
						<DialogTitle>Delete Profile</DialogTitle>
						<DialogDescription>
							Are you sure you want to delete the profile <strong>{deleteDialog.profileName}</strong>?
							<br />
							This action cannot be undone.
						</DialogDescription>
					</DialogHeader>
					<DialogFooter>
						<Button
							variant="outline"
							onClick={() => setDeleteDialog({ open: false, profileId: null, profileName: '' })}
						>
							Cancel
						</Button>
						<Button variant="destructive" onClick={confirmDelete}>
							Delete
						</Button>
					</DialogFooter>
				</DialogContent>
			</Dialog>

			{/* Error Dialog */}
			<Dialog
				open={errorDialog.open}
				onOpenChange={(open) => !open && setErrorDialog({ open: false, message: '' })}
			>
				<DialogContent>
					<DialogHeader>
						<DialogTitle className="flex items-center gap-2 text-red-600">
							<AlertCircle className="h-5 w-5" />
							Error
						</DialogTitle>
					</DialogHeader>
					<Alert variant="destructive">
						<AlertDescription>{errorDialog.message}</AlertDescription>
					</Alert>
					<DialogFooter>
						<Button onClick={() => setErrorDialog({ open: false, message: '' })}>Close</Button>
					</DialogFooter>
				</DialogContent>
			</Dialog>
		</div>
	);
}
