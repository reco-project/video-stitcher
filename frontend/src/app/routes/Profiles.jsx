import React from 'react';
import ProfileManager from '@/features/profiles/components/ProfileManager';
import AppSettings from '@/features/settings/components/AppSettings';
import About from '@/features/settings/components/About';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function Profiles() {
	return (
		<div className="w-full h-full">
			<div className="max-w-7xl mx-auto p-6">
				<Tabs defaultValue="general" className="space-y-6">
					<TabsList>
						<TabsTrigger value="general">General</TabsTrigger>
						<TabsTrigger value="profiles">Lens Profiles</TabsTrigger>
						<TabsTrigger value="about">About</TabsTrigger>
					</TabsList>

					<TabsContent value="general" className="space-y-6">
						<AppSettings />
					</TabsContent>

					<TabsContent value="profiles" className="space-y-6">
						<ProfileManager />
					</TabsContent>

					<TabsContent value="about" className="space-y-6">
						<About />
					</TabsContent>
				</Tabs>
			</div>
		</div>
	);
}
