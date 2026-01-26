import { useState, useEffect, useMemo, useCallback } from 'react';
import { getEncoderSettings } from '@/features/settings/api/settings';

// Preset to bitrate mapping
const PRESET_BITRATES = {
    '720p': '30M',
    '1080p': '50M',
    '1440p': '70M',
};

/**
 * Hook to manage quality settings state and encoder info
 */
export function useQualitySettings(initialValues = {}) {
    const [preset, setPreset] = useState(initialValues.preset || '1080p');
    const [customBitrate, setCustomBitrate] = useState(initialValues.customBitrate || '30M');
    const [customPreset, setCustomPreset] = useState(initialValues.customPreset || 'medium');
    const [customResolution, setCustomResolution] = useState(initialValues.customResolution || '1080p');
    const [customUseGpuDecode, setCustomUseGpuDecode] = useState(initialValues.customUseGpuDecode ?? true);

    // Encoder info
    const [encoderInfo, setEncoderInfo] = useState(null);
    const [loadingEncoder, setLoadingEncoder] = useState(true);

    // Load encoder settings on mount
    useEffect(() => {
        getEncoderSettings()
            .then(setEncoderInfo)
            .catch((err) => console.error('Failed to load encoder settings:', err))
            .finally(() => setLoadingEncoder(false));
    }, []);

    const handleCustomChange = useCallback((changes) => {
        if ('bitrate' in changes) setCustomBitrate(changes.bitrate);
        if ('preset' in changes) setCustomPreset(changes.preset);
        if ('resolution' in changes) setCustomResolution(changes.resolution);
        if ('useGpuDecode' in changes) setCustomUseGpuDecode(changes.useGpuDecode);
    }, []);

    // Build the quality settings object for submission
    const qualitySettings = useMemo(() => {
        if (preset === 'custom') {
            return {
                preset: 'custom',
                bitrate: customBitrate,
                speed_preset: customPreset,
                resolution: customResolution,
                use_gpu_decode: customUseGpuDecode,
            };
        }
        return {
            preset,
            bitrate: PRESET_BITRATES[preset] || '50M',
            speed_preset: 'superfast',
            resolution: preset,
            use_gpu_decode: false,
        };
    }, [preset, customBitrate, customPreset, customResolution, customUseGpuDecode]);

    // Values for draft saving
    const draftValues = useMemo(
        () => ({
            qualityPreset: preset,
            customBitrate,
            customPreset,
            customResolution,
            customUseGpuDecode,
        }),
        [preset, customBitrate, customPreset, customResolution, customUseGpuDecode]
    );

    return {
        // State
        preset,
        setPreset,
        customBitrate,
        customPreset,
        customResolution,
        customUseGpuDecode,
        handleCustomChange,

        // Encoder
        encoderInfo,
        loadingEncoder,

        // Computed
        qualitySettings,
        draftValues,
    };
}
