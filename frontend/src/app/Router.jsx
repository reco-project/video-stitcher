import React from 'react';
import { createBrowserRouter, useNavigate } from 'react-router';
import { RouterProvider } from 'react-router/dom';

import Home from './routes/Home';
import Profiles from './routes/Profiles';
import NotFound from './routes/NotFound';

/*
 * For the moment, I don't need complex routing, so I'll keep this centralized.
 * When the app grows, we can split (see https://github.com/alan2207/bulletproof-react/blob/master/apps/react-vite/src/app/router.tsx).
 */

const paths = {
	home: {
		pattern: '/',
		build: () => '/',
		title: 'Home',
	},
	profiles: {
		pattern: '/profiles',
		build: () => '/profiles',
		title: 'Lens Profiles',
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
		toProfiles: () => navigate(paths.profiles.build()),
	};
};

const router = createBrowserRouter([
	{
		path: paths.home.pattern,
		element: <Home />,
	},
	{
		path: paths.profiles.pattern,
		element: <Profiles />,
	},
	{
		path: '*',
		element: <NotFound />,
	},
]);

export default function AppRouter() {
	return <RouterProvider router={router} />;
}
