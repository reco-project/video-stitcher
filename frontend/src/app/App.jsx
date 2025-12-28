import React from 'react';
import { AppProvider } from './Provider';
import AppRouter from './Router';
import { ToastProvider } from '@/components/ui/toast';

export default function App() {
	return (
		<ToastProvider>
			<AppProvider>
				<AppRouter />
			</AppProvider>
		</ToastProvider>
	);
}
