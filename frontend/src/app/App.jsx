import React from 'react';
import { AppProvider } from './Provider';
import AppRouter from './Router';

export default function App() {
	return (
		<AppProvider>
			<AppRouter />
		</AppProvider>
	);
}
