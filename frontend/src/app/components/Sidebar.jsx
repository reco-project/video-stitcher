import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Home, Plus, Settings, Moon, Sun } from 'lucide-react';
import { cn } from '@/lib/cn';
import { useDarkMode } from '@/hooks/useDarkMode';

/**
 * Footer Navigation with health status
 * Simplified to use proper route-based navigation
 */
export default function Sidebar() {
	const location = useLocation();
	const navigate = useNavigate();
	const [healthStatus, setHealthStatus] = useState('checking');
	const { isDark, toggle } = useDarkMode();

	// Check backend health
	useEffect(() => {
		const checkHealth = async () => {
			try {
				const apiBaseUrl = localStorage.getItem('app-settings')
					? JSON.parse(localStorage.getItem('app-settings')).apiBaseUrl
					: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';
				const response = await fetch(`${apiBaseUrl}/health`, { timeout: 3000 });
				setHealthStatus(response.ok ? 'connected' : 'disconnected');
			} catch {
				setHealthStatus('disconnected');
			}
		};

		checkHealth();
		const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
		return () => clearInterval(interval);
	}, []);

	const navItems = [
		{
			id: 'home',
			label: 'Home',
			icon: Home,
			path: '/',
		},
		{
			id: 'create',
			label: 'Create Match',
			icon: Plus,
			path: '/create',
		},
		{
			id: 'profiles',
			label: 'Settings',
			icon: Settings,
			path: '/profiles',
		},
	];

	const isActive = (path) => {
		if (path === '/') {
			return location.pathname === '/' || location.pathname.startsWith('/viewer');
		}
		return location.pathname === path;
	};

	const healthColor = healthStatus === 'connected' ? 'bg-green-500' : 'bg-red-500';
	const healthLabel = healthStatus === 'connected' ? 'Connected' : 'Offline';

	return (
		<footer className="border-t bg-background">
			<div className="flex items-center justify-between px-6 py-4">
				{/* Left: Health Status */}
				<div className="flex items-center gap-2 text-xs text-muted-foreground">
					<div className={cn('w-2 h-2 rounded-full', healthColor)} />
					<span>{healthLabel}</span>
				</div>

				{/* Center: Navigation */}
				<nav className="flex gap-2">
					{navItems.map((item) => {
						const Icon = item.icon;
						const active = isActive(item.path);

						return (
							<Button
								key={item.id}
								size="sm"
								title={item.label}
								variant={active ? 'default' : 'ghost'}
								className="h-9 w-9 rounded-md transition-all duration-200"
								onClick={() => navigate(item.path)}
							>
								<Icon className="h-4 w-4" />
							</Button>
						);
					})}
				</nav>

				{/* Right: Dark mode toggle */}
				<Button
					variant="ghost"
					size="sm"
					onClick={toggle}
					title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
					className="h-9 w-9 p-0"
				>
					{isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
				</Button>
			</div>
		</footer>
	);
}
