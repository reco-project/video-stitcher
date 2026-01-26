import { useState, useRef, useCallback } from 'react';

/**
 * Hook to record the canvas/WebGL content as a video file
 * @param {Object} options - Configuration options
 * @param {number} options.fps - Frames per second (default: 30)
 * @param {string} options.mimeType - Video MIME type (default: 'video/webm')
 * @param {number} options.videoBitsPerSecond - Bitrate (default: 5000000 = 5Mbps)
 */
export function useCanvasRecorder({
    fps = 30,
    mimeType = 'video/webm;codecs=vp9',
    videoBitsPerSecond = 5000000,
} = {}) {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingDuration, setRecordingDuration] = useState(0);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);
    const durationIntervalRef = useRef(null);
    const startTimeRef = useRef(null);

    /**
     * Start recording the canvas
     * @param {HTMLCanvasElement} canvas - The canvas element to record
     * @param {HTMLVideoElement} [audioSource] - Optional video element to capture audio from
     * @param {MediaStream} [micStream] - Optional microphone stream to include
     */
    const startRecording = useCallback((canvas, audioSource = null, micStream = null) => {
        if (!canvas) {
            console.error('Canvas element is required for recording');
            return false;
        }

        // Check for supported MIME types
        let selectedMimeType = mimeType;
        if (!MediaRecorder.isTypeSupported(mimeType)) {
            const fallbacks = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
            selectedMimeType = fallbacks.find(type => MediaRecorder.isTypeSupported(type));
            if (!selectedMimeType) {
                console.error('No supported video MIME type found');
                return false;
            }
            console.warn(`Using fallback MIME type: ${selectedMimeType}`);
        }

        try {
            // Capture the canvas stream
            const canvasStream = canvas.captureStream(fps);

            // Handle audio mixing if needed
            let finalAudioStream = null;

            // If we have both video audio and microphone, we need to mix them
            if (audioSource && audioSource.captureStream && micStream) {
                const audioContext = new AudioContext();
                const destination = audioContext.createMediaStreamDestination();

                // Add video audio
                const videoAudioStream = audioSource.captureStream();
                const videoAudioTracks = videoAudioStream.getAudioTracks();
                if (videoAudioTracks.length > 0) {
                    const videoSource = audioContext.createMediaStreamSource(new MediaStream([videoAudioTracks[0]]));
                    videoSource.connect(destination);
                }

                // Add microphone audio
                const micTracks = micStream.getAudioTracks();
                if (micTracks.length > 0) {
                    const micSource = audioContext.createMediaStreamSource(new MediaStream([micTracks[0]]));
                    micSource.connect(destination);
                }

                finalAudioStream = destination.stream;
            } else if (audioSource && audioSource.captureStream) {
                // Only video audio
                finalAudioStream = audioSource.captureStream();
            } else if (micStream) {
                // Only microphone
                finalAudioStream = micStream;
            }

            // Add the audio track(s) to canvas stream
            if (finalAudioStream) {
                const audioTracks = finalAudioStream.getAudioTracks();
                if (audioTracks.length > 0) {
                    audioTracks.forEach(track => canvasStream.addTrack(track));
                }
            }

            // Create MediaRecorder
            const mediaRecorder = new MediaRecorder(canvasStream, {
                mimeType: selectedMimeType,
                videoBitsPerSecond,
            });

            chunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                // Create blob and trigger download
                const blob = new Blob(chunksRef.current, { type: selectedMimeType });

                if (blob.size === 0) {
                    console.error('Recording resulted in 0 bytes. No data was captured.');
                    alert('Recording failed: No data was captured. Make sure the canvas is rendering.');
                    chunksRef.current = [];
                    return;
                }

                const url = URL.createObjectURL(blob);

                // Create download link
                const a = document.createElement('a');
                a.href = url;
                a.download = `recording-${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.webm`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);

                // Cleanup
                URL.revokeObjectURL(url);
                chunksRef.current = [];
            };

            mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                setIsRecording(false);
                clearInterval(durationIntervalRef.current);
            };

            // Start recording
            mediaRecorder.start(100); // Collect data every 100ms
            mediaRecorderRef.current = mediaRecorder;
            setIsRecording(true);
            setRecordingDuration(0);
            startTimeRef.current = Date.now();

            // Update duration counter
            durationIntervalRef.current = setInterval(() => {
                setRecordingDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
            }, 1000);

            return true;
        } catch (error) {
            console.error('Failed to start recording:', error);
            return false;
        }
    }, [fps, mimeType, videoBitsPerSecond]);

    /**
     * Stop recording and download the video
     */
    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
            mediaRecorderRef.current = null;
        }
        setIsRecording(false);
        clearInterval(durationIntervalRef.current);
        setRecordingDuration(0);
    }, []);

    /**
     * Toggle recording state
     */
    const toggleRecording = useCallback((canvas, audioSource = null, micStream = null) => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording(canvas, audioSource, micStream);
        }
    }, [isRecording, startRecording, stopRecording]);

    return {
        isRecording,
        recordingDuration,
        startRecording,
        stopRecording,
        toggleRecording,
    };
}
