import React from 'react';
import { cn } from '@/lib/cn';
import { Check, Loader2 } from 'lucide-react';

/**
 * StepIndicator shows progress through a multi-step wizard
 * Shows current step, total steps, and visual progress bar
 */
export default function StepIndicator({ currentStep, totalSteps, steps, stepBusy = false }) {
	return (
		<div className="space-y-4">
			{/* Step Progress Header */}
			<div className="flex items-center justify-between">
				<div>
					<h3 className="text-sm font-semibold text-muted-foreground">
						Step {currentStep} of {totalSteps}
					</h3>
					{steps && steps[currentStep - 1] && (
						<p className="text-lg font-bold mt-1">{steps[currentStep - 1]}</p>
					)}
				</div>
				<div className="text-sm text-muted-foreground">
					{Math.round((currentStep / totalSteps) * 100)}% complete
				</div>
			</div>

			{/* Progress Bar */}
			<div className="w-full bg-muted rounded-full h-2 overflow-hidden">
				<div
					className="bg-purple-600 h-full transition-all duration-300 ease-out"
					style={{ width: `${(currentStep / totalSteps) * 100}%` }}
				/>
			</div>

			{/* Step Indicators */}
			<div className="flex justify-between gap-2">
				{Array.from({ length: totalSteps }).map((_, idx) => {
					const stepNum = idx + 1;
					const isCompleted = stepNum < currentStep;
					const isCurrent = stepNum === currentStep;

					return (
						<div
							key={stepNum}
							className={cn(
								'flex-1 h-10 rounded-lg flex items-center justify-center font-semibold text-sm transition-all relative',
								isCurrent && 'bg-purple-600 text-white ring-2 ring-purple-400 ring-offset-2',
								isCompleted && 'bg-green-500 text-white',
								!isCurrent && !isCompleted && 'bg-muted text-muted-foreground'
							)}
						>
							{isCompleted ? (
								<Check className="h-5 w-5" />
							) : (
								<>
									{stepNum}
									{isCurrent && stepBusy && (
										<Loader2 className="h-4 w-4 absolute right-2 top-2 animate-spin text-white" />
									)}
								</>
							)}
						</div>
					);
				})}
			</div>
		</div>
	);
}
