import React from 'react';
import { createBrowserRouter, createHashRouter, useNavigate } from 'react-router';
import { RouterProvider } from 'react-router-dom';

import AppLayout from './AppLayout';
import MatchList from './routes/MatchList';
import CreateMatch from './routes/CreateMatch';
import EditMatch from './routes/EditMatch';
import ProcessingMatch from './routes/ProcessingMatch';
import MatchViewer from './routes/MatchViewer';
import Profiles from './routes/Profiles';
import NotFound from './routes/NotFound';

// Use HashRouter for Electron (file:// protocol) and BrowserRouter for dev server
const isElectron = window.location.protocol === 'file:';

/*
 * Simplified routing with dedicated routes for each view
 */

const paths = {
	home: {
		pattern: '/',
		build: () => '/',
		title: 'Home',
	},
	create: {
		pattern: '/create',
		build: () => '/create',
		title: 'Create Match',
	},
	edit: {
		pattern: '/edit/:id',
		build: (id) => `/edit/${id}`,
		title: 'Edit Match',
	},
	processing: {
		pattern: '/processing/:id',
		build: (id) => `/processing/${id}`,
		title: 'Processing',
	},
	viewer: {
		pattern: '/viewer/:id',
		build: (id) => `/viewer/${id}`,
		title: 'Viewer',
	},
	profiles: {
		pattern: '/profiles',
		build: () => '/profiles',
		title: 'Settings',
	},
};

/**
 * Custom hook for navigation. Usage:
 * ```js
 * const navigate = useNavigateTo();
 * navigate.toHome();
 * ```
 * @returns {Object} - Navigation functions.
 */
export const useNavigateTo = () => {
	const navigate = useNavigate();
	return {
		toHome: () => navigate(paths.home.build()),
		toCreate: () => navigate(paths.create.build()),
		toEdit: (id) => navigate(paths.edit.build(id)),
		toProcessing: (id) => navigate(paths.processing.build(id)),
		toViewer: (id) => navigate(paths.viewer.build(id)),
		toProfiles: () => navigate(paths.profiles.build()),
	};
};

const routes = [
	{
		path: paths.home.pattern,
		element: (
			<AppLayout>
				<MatchList />
			</AppLayout>
		),
	},
	{
		path: paths.create.pattern,
		element: (
			<AppLayout>
				<CreateMatch />
			</AppLayout>
		),
	},
	{
		path: paths.edit.pattern,
		element: (
			<AppLayout>
				<EditMatch />
			</AppLayout>
		),
	},
	{
		path: paths.processing.pattern,
		element: (
			<AppLayout>
				<ProcessingMatch />
			</AppLayout>
		),
	},
	{
		path: paths.viewer.pattern,
		element: (
			<AppLayout>
				<MatchViewer />
			</AppLayout>
		),
	},
	{
		path: paths.profiles.pattern,
		element: (
			<AppLayout>
				<Profiles />
			</AppLayout>
		),
	},
	{
		path: '*',
		element: <NotFound />,
	},
];

// Use HashRouter for Electron (file:// protocol), BrowserRouter for dev server
const router = isElectron ? createHashRouter(routes) : createBrowserRouter(routes);

export default function AppRouter() {
	return <RouterProvider router={router} />;
}
