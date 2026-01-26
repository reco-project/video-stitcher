import { useState, useCallback, useEffect } from 'react';
import { useToast } from '@/components/ui/toast';

/**
 * Hook to manage video files for a single side (left or right)
 * Handles file selection, drag-drop, metadata loading, and reordering
 */
export function useVideoSide(initialPaths = []) {
	const [paths, setPaths] = useState(initialPaths);
	const [metadata, setMetadata] = useState([]);

	const loadMetadata = useCallback(async (filePath, index) => {
		if (!filePath || !window.electronAPI?.getFileMetadata) return;

		try {
			const meta = await window.electronAPI.getFileMetadata(filePath);
			if (meta) {
				setMetadata((prev) => {
					const newMeta = [...prev];
					newMeta[index] = meta;
					return newMeta;
				});
			}
		} catch (err) {
			console.warn('Failed to load metadata:', err);
		}
	}, []);

	const addFiles = useCallback(
		(filePaths) => {
			if (!filePaths || filePaths.length === 0) return;
			const existingPaths = paths.filter((p) => p);
			const newPaths = [...existingPaths, ...filePaths];
			setPaths(newPaths);
			filePaths.forEach((path, idx) => {
				loadMetadata(path, existingPaths.length + idx);
			});
		},
		[paths, loadMetadata]
	);

	const removeFile = useCallback((index) => {
		setPaths((prev) => prev.filter((_, i) => i !== index));
		setMetadata((prev) => prev.filter((_, i) => i !== index));
	}, []);

	const reorder = useCallback((fromIndex, toIndex) => {
		setPaths((prev) => {
			const newPaths = [...prev];
			const [moved] = newPaths.splice(fromIndex, 1);
			newPaths.splice(toIndex, 0, moved);
			return newPaths;
		});
		setMetadata((prev) => {
			const newMeta = [...prev];
			const [moved] = newMeta.splice(fromIndex, 1);
			newMeta.splice(toIndex, 0, moved);
			return newMeta;
		});
	}, []);

	const insertAt = useCallback((index, filePath, meta) => {
		setPaths((prev) => {
			const newPaths = [...prev];
			newPaths.splice(index, 0, filePath);
			return newPaths;
		});
		setMetadata((prev) => {
			const newMeta = [...prev];
			newMeta.splice(index, 0, meta);
			return newMeta;
		});
	}, []);

	const removeAt = useCallback((index) => {
		let removedPath, removedMeta;
		setPaths((prev) => {
			const newPaths = [...prev];
			[removedPath] = newPaths.splice(index, 1);
			return newPaths;
		});
		setMetadata((prev) => {
			const newMeta = [...prev];
			[removedMeta] = newMeta.splice(index, 1);
			return newMeta;
		});
		return { path: removedPath, meta: removedMeta };
	}, []);

	// Load metadata for initial paths
	useEffect(() => {
		paths.forEach((path, index) => {
			if (path && path.trim()) {
				loadMetadata(path, index);
			}
		});
	}, []); // Only on mount

	return {
		paths,
		setPaths,
		metadata,
		setMetadata,
		addFiles,
		removeFile,
		reorder,
		insertAt,
		removeAt,
		loadMetadata,
	};
}

/**
 * Hook to manage video files for both sides with cross-side drag-drop support
 */
export function useVideoManager(initialLeftPaths = [], initialRightPaths = []) {
	const { showToast } = useToast();
	const left = useVideoSide(initialLeftPaths);
	const right = useVideoSide(initialRightPaths);

	const [draggedIndex, setDraggedIndex] = useState(null);
	const [draggedSide, setDraggedSide] = useState(null);

	const handleSelectFiles = useCallback(
		async (side) => {
			try {
				if (!window.electronAPI || !window.electronAPI.selectVideoFiles) {
					throw new Error('File selection not available. Please run in Electron.');
				}

				const filePaths = await window.electronAPI.selectVideoFiles();
				if (filePaths && filePaths.length > 0) {
					if (side === 'left') {
						left.addFiles(filePaths);
					} else {
						right.addFiles(filePaths);
					}
				}
			} catch (err) {
				console.error('[useVideoManager] Error in handleSelectFiles:', err);
				showToast({ message: err.message, type: 'error' });
			}
		},
		[left, right, showToast]
	);

	const handleFileDrop = useCallback(
		(side, e) => {
			e.preventDefault();
			e.stopPropagation();

			const files = Array.from(e.dataTransfer.files);

			// With contextIsolation enabled, file.path is not accessible
			// Use Electron's webUtils.getPathForFile via the preload API
			const filePaths = files
				.map((f) => {
					// Try Electron's getPathForFile API first (works with contextIsolation)
					if (window.electronAPI?.getPathForFile) {
						try {
							return window.electronAPI.getPathForFile(f);
						} catch {
							return null;
						}
					}
					// Fallback to file.path (only works without contextIsolation)
					return f.path;
				})
				.filter(Boolean);

			if (filePaths.length > 0) {
				if (side === 'left') {
					left.addFiles(filePaths);
				} else {
					right.addFiles(filePaths);
				}
			}
		},
		[left, right]
	);

	const handleRemoveVideo = useCallback(
		(side, index) => {
			if (side === 'left') {
				left.removeFile(index);
			} else {
				right.removeFile(index);
			}
		},
		[left, right]
	);

	const handleDragStart = useCallback((side, index, e) => {
		e.dataTransfer.effectAllowed = 'move';
		e.dataTransfer.setData('text/plain', `reorder-${side}-${index}`);
		setDraggedIndex(index);
		setDraggedSide(side);
	}, []);

	const handleDragOver = useCallback((e) => {
		e.preventDefault();
		e.stopPropagation();
		e.dataTransfer.dropEffect = 'move';
	}, []);

	const handleDrop = useCallback(
		(side, dropIndex, e = null, filePaths = null) => {
			// If filePaths are provided (external drop), add them directly
			if (Array.isArray(filePaths) && filePaths.length > 0) {
				if (side === 'left') {
					left.addFiles(filePaths);
				} else {
					right.addFiles(filePaths);
				}
				return;
			}

			if (e) {
				e.preventDefault();
				e.stopPropagation();

				const dragData = e.dataTransfer.getData('text/plain');

				// Check if this is a file drop from filesystem
				if (!dragData && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
					handleFileDrop(side, e);
					return;
				}
			}

			// Check for invalid drop
			if (draggedSide === null || draggedIndex === null) {
				return;
			}

			// Same side, same position - no change
			if (draggedSide === side && draggedIndex === dropIndex) {
				setDraggedIndex(null);
				setDraggedSide(null);
				return;
			}

			const source = draggedSide === 'left' ? left : right;
			const target = side === 'left' ? left : right;

			if (draggedSide === side) {
				// Same side reorder
				const adjustedIndex = dropIndex > draggedIndex ? dropIndex - 1 : dropIndex;
				source.reorder(draggedIndex, adjustedIndex);
			} else {
				// Cross-side move
				const sourcePaths = [...source.paths];
				const sourceMetadata = [...source.metadata];
				const [movedPath] = sourcePaths.splice(draggedIndex, 1);
				const [movedMeta] = sourceMetadata.splice(draggedIndex, 1);

				source.setPaths(sourcePaths);
				source.setMetadata(sourceMetadata);
				target.insertAt(dropIndex, movedPath, movedMeta);
			}

			setDraggedIndex(null);
			setDraggedSide(null);
		},
		[draggedIndex, draggedSide, left, right, handleFileDrop]
	);

	return {
		left: {
			paths: left.paths,
			metadata: left.metadata,
		},
		right: {
			paths: right.paths,
			metadata: right.metadata,
		},
		dragState: {
			draggedIndex,
			draggedSide,
		},
		handlers: {
			handleSelectFiles,
			handleFileDrop,
			handleRemoveVideo,
			handleDragStart,
			handleDragOver,
			handleDrop,
		},
	};
}
