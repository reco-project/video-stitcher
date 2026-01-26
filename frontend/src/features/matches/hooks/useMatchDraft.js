import { useEffect, useCallback } from 'react';

const DRAFT_KEY = 'matchCreationDraft';

/**
 * Hook to manage match creation draft persistence
 */
export function useMatchDraft() {
    const loadDraft = useCallback(() => {
        try {
            const draft = localStorage.getItem(DRAFT_KEY);
            return draft ? JSON.parse(draft) : null;
        } catch (err) {
            console.warn('Failed to load draft:', err);
            return null;
        }
    }, []);

    const saveDraft = useCallback((data) => {
        try {
            localStorage.setItem(DRAFT_KEY, JSON.stringify(data));
        } catch (err) {
            console.warn('Failed to save draft:', err);
        }
    }, []);

    const clearDraft = useCallback(() => {
        try {
            localStorage.removeItem(DRAFT_KEY);
        } catch (err) {
            console.warn('Failed to clear draft:', err);
        }
    }, []);

    return { loadDraft, saveDraft, clearDraft };
}

/**
 * Hook to auto-save draft data with debounce
 */
export function useAutoSaveDraft(data, saveDraft, delay = 500) {
    useEffect(() => {
        const timeoutId = setTimeout(() => {
            saveDraft(data);
        }, delay);

        return () => clearTimeout(timeoutId);
    }, [data, saveDraft, delay]);
}
