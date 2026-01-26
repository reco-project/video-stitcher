import React from 'react';
import { useNavigate } from 'react-router';
import { LucideHome } from 'lucide-react';
import { Button } from '@/components/ui/button';

const NotFound = () => {
	const navigate = useNavigate();

	return (
		<div className="flex flex-col items-center justify-center min-h-screen gap-6 p-8">
			<div className="text-center space-y-4">
				<h1 className="text-9xl font-bold text-muted-foreground/20">404</h1>
				<h2 className="text-3xl font-semibold">Page Not Found</h2>
				<p className="text-muted-foreground max-w-md">
					The page you&apos;re looking for doesn&apos;t exist or has been moved.
				</p>
			</div>
			<Button onClick={() => navigate('/')} size="lg" className="gap-2">
				<LucideHome className="h-4 w-4" />
				Go Back Home
			</Button>
		</div>
	);
};

export default NotFound;
