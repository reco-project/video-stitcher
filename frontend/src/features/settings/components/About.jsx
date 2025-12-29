import React from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Github, ExternalLink, AlertCircle, Heart } from 'lucide-react';

export default function About() {
	const openLink = (url) => {
		try {
			// Open in default browser instead of Electron window
			if (window.electronAPI && typeof window.electronAPI.openExternal === 'function') {
				window.electronAPI.openExternal(url);
			} else {
				window.open(url, '_blank', 'noopener,noreferrer');
			}
		} catch (error) {
			console.error('Error opening link:', error);
			// Final fallback
			window.open(url, '_blank', 'noopener,noreferrer');
		}
	};

	return (
		<div className="space-y-6">
			{/* Project Info */}
			<Card>
				<CardHeader>
					<div className="flex items-center justify-between">
						<div className="flex items-center gap-2">
							<Github className="h-5 w-5 text-muted-foreground" />
							<CardTitle>reco-project/video-stitcher</CardTitle>
						</div>
						<Badge variant="secondary">Beta</Badge>
					</div>
					<CardDescription>
						An open-source dual-camera video stitching tool with lens calibration support
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="flex items-start gap-3 p-4 rounded-lg bg-muted/50">
						<AlertCircle className="h-5 w-5 text-blue-500 mt-0.5" />
						<div className="flex-1 space-y-1">
							<p className="text-sm font-medium">This is beta software</p>
							<p className="text-xs text-muted-foreground">
								Features are actively being developed. You may encounter bugs or incomplete
								functionality. Please report issues on GitHub to help improve the project.
							</p>
						</div>
					</div>

					<Separator />

					<div className="space-y-2">
						<h4 className="text-sm font-semibold">Quick Links</h4>
						<div className="grid gap-2">
							<Button
								variant="outline"
								className="justify-start"
								onClick={() => openLink('https://github.com/reco-project/video-stitcher')}
							>
								<Github className="h-4 w-4 mr-2" />
								View on GitHub
								<ExternalLink className="h-3 w-3 ml-auto" />
							</Button>
							<Button
								variant="outline"
								className="justify-start"
								onClick={() => openLink('https://github.com/reco-project/video-stitcher/issues')}
							>
								<AlertCircle className="h-4 w-4 mr-2" />
								Report an Issue
								<ExternalLink className="h-3 w-3 ml-auto" />
							</Button>
							<Button
								variant="outline"
								className="justify-start"
								onClick={() =>
									openLink('https://github.com/reco-project/video-stitcher/blob/main/README.md')
								}
							>
								Documentation
								<ExternalLink className="h-3 w-3 ml-auto" />
							</Button>
						</div>
					</div>
				</CardContent>
			</Card>

			{/* Credits & Attribution */}
			<Card>
				<CardHeader>
					<div className="flex items-center gap-2">
						<Heart className="h-5 w-5 text-muted-foreground" />
						<CardTitle>Credits</CardTitle>
					</div>
					<CardDescription>Built with open-source tools and libraries</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="space-y-3">
						<div>
							<h4 className="text-sm font-semibold mb-1">Core Technologies</h4>
							<ul className="text-sm text-muted-foreground space-y-1">
								<li>• React + Vite - Modern frontend framework</li>
								<li>• FastAPI + Python - High-performance backend</li>
								<li>• Three.js + React Three Fiber - 3D rendering and video processing</li>
								<li>• OpenCV - Computer vision and camera calibration</li>
							</ul>
						</div>

						<Separator />

						<div>
							<h4 className="text-sm font-semibold mb-1">UI Components</h4>
							<ul className="text-sm text-muted-foreground space-y-1">
								<li>• shadcn/ui - Accessible component library</li>
								<li>• Radix UI - Unstyled component primitives</li>
								<li>• Tailwind CSS - Utility-first styling</li>
								<li>• Lucide React - Beautiful icon library</li>
							</ul>
						</div>

						<Separator />

						<div>
							<h4 className="text-sm font-semibold mb-1">Special Thanks</h4>
							<ul className="text-sm text-muted-foreground space-y-1">
								<li>• Gyroflow project for lens calibration profiles</li>
								<li>• Open-source community for tools and inspiration</li>
							</ul>
						</div>
					</div>
				</CardContent>
			</Card>

			{/* Version Info */}
			<Card>
				<CardContent className="pt-6">
					<div className="text-center text-sm text-muted-foreground">
						<p>Version 0.1.0</p>
						<p className="mt-1">
							Made with <Heart className="h-3 w-3 inline text-red-500" /> by the RECO team
						</p>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
