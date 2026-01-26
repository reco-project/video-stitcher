import React, { createContext, useContext, useState, useCallback } from 'react';

const ToastContext = createContext();

export function ToastProvider({ children }) {
	const [toasts, setToasts] = useState([]);

	const showToast = useCallback((toast) => {
		const id = toast.id || `toast-${Date.now()}`;
		setToasts((prev) => [...prev, { ...toast, id }]);
		setTimeout(() => {
			setToasts((prev) => prev.filter((t) => t.id !== id));
		}, toast.duration || 3500);
		return id;
	}, []);

	return (
		<ToastContext.Provider value={{ showToast }}>
			{children}
			<div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-50 flex flex-col gap-2 items-center">
				{toasts.map((toast) => (
					<div
						key={toast.id}
						className={`px-4 py-2 rounded shadow-lg bg-card border flex items-center gap-2 min-w-[200px] max-w-xs text-sm ${toast.type === 'error' ? 'bg-red-50 border-red-300 text-red-800 dark:bg-red-900 dark:text-red-100' : toast.type === 'success' ? 'bg-green-50 border-green-300 text-green-800 dark:bg-green-900 dark:text-green-100' : toast.type === 'warning' ? 'bg-yellow-50 border-yellow-300 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100' : 'bg-blue-50 border-blue-300 text-blue-800 dark:bg-blue-900 dark:text-blue-100'}`}
					>
						{toast.icon && <span>{toast.icon}</span>}
						<span>{toast.message}</span>
					</div>
				))}
			</div>
		</ToastContext.Provider>
	);
}

export function useToast() {
	return useContext(ToastContext);
}
