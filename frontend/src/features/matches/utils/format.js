/**
 * Shared formatting utilities for video metadata
 */

export function formatDuration(seconds) {
    if (!seconds) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

export function calculateVideoTotals(metadata) {
    const validMetadata = metadata.filter(Boolean);
    const totalDuration = validMetadata.reduce((sum, m) => sum + (m.duration || 0), 0);
    const totalSize = validMetadata.reduce((sum, m) => sum + (m.size || 0), 0);

    return {
        count: validMetadata.length,
        totalDuration,
        totalSize,
        durationFormatted: formatDuration(totalDuration),
        sizeFormatted: formatFileSize(totalSize),
    };
}
