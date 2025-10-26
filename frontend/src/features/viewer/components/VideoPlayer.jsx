import React, { useEffect, useRef, useState } from "react";
import { cn } from "../../../lib/cn";
import { useViewerStore } from "../stores/store";
import {
  LucidePlay,
  LucidePause,
  LucideVolume,
  LucideMaximize,
} from "lucide-react";

export default function VideoPlayer({ children, className }) {
  const videoRef = useViewerStore((state) => state.videoRef);
  const containerRef = useRef(null);
  const playing = useViewerStore((state) => state.playing);
  const setPlaying = useViewerStore((state) => state.setPlaying);

  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const progress =
    videoRef && videoRef.duration ? (currentTime / videoRef.duration) * 100 : 0;

  useEffect(() => {
    if (!videoRef) return;
    let interval = null;
    if (playing) {
      interval = setInterval(() => {
        if (videoRef) {
          setCurrentTime(videoRef.currentTime);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [videoRef, playing]);

  const togglePlay = () => {
    if (!videoRef) return;
    if (videoRef.paused) {
      videoRef.play();
      setPlaying(true);
    } else {
      videoRef.pause();
      setPlaying(false);
    }
  };

  const handleVolumeToggle = () => {
    if (!videoRef) return;
    videoRef.muted = !videoRef.muted;
    setMuted(videoRef.muted);
  };

  const handleVolumeChange = (e) => {
    if (!videoRef) return;
    videoRef.volume = parseFloat(e.target.value);
    setVolume(videoRef.volume);
  };

  const formatTime = (time) => {
    if (isNaN(time)) return "00:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  };

  const handleSeek = (e) => {
    if (!videoRef || !videoRef.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const seekPercent = Math.max(0, Math.min(1, clickX / rect.width));
    videoRef.currentTime = seekPercent * videoRef.duration;
    setCurrentTime(videoRef.currentTime);
  };

  const handleMouseDown = (e) => {
    if (!videoRef || !videoRef.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const startX = e.clientX;
    const seek = (clientX) => {
      const clickX = clientX - rect.left;
      const seekPercent = Math.max(0, Math.min(1, clickX / rect.width));
      videoRef.currentTime = seekPercent * videoRef.duration;
      setCurrentTime(videoRef.currentTime);
    };
    seek(startX);

    const onMouseMove = (moveEvent) => {
      seek(moveEvent.clientX);
    };
    const onMouseUp = (upEvent) => {
      seek(upEvent.clientX);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  };

  const handleFullscreen = () => {
    const el = containerRef.current;
    let attemptedFullscreen = !isFullscreen;
    if (!el) return;
    if (isFullscreen) {
      document.exitFullscreen();
    } else if (el.requestFullscreen) {
      el.requestFullscreen();
    } else if (el.webkitRequestFullscreen) {
      el.webkitRequestFullscreen();
    } else if (el.msRequestFullscreen) {
      el.msRequestFullscreen();
    } else {
      console.warn("Fullscreen API is not supported.");
      attemptedFullscreen = isFullscreen;
    }
    setIsFullscreen(attemptedFullscreen);
  };

  // Optional: subscribe to video time updates
  useEffect(() => {
    if (!videoRef) return;
    const onTimeUpdate = () => setPlaying(!videoRef.paused);
    videoRef.addEventListener("timeupdate", onTimeUpdate);
    return () => videoRef.removeEventListener("timeupdate", onTimeUpdate);
  }, [videoRef, setPlaying]);

const [isControlsVisible, setIsControlsVisible] = useState(true);

const handleMouseEnter = () => setIsControlsVisible(true);
const handleMouseLeave = () => setIsControlsVisible(false); // TODO: refine better logic for hiding controls after timeout

return (
    <div
        className={cn(
            "relative w-full max-w-3xl bg-black rounded-lg overflow-hidden",
            className
        )}
        ref={containerRef}
        tabIndex={0}
        onKeyDown={(e) => {
            if (e.key === " ") {
                e.preventDefault();
                togglePlay();
            }
            if (e.key === "ArrowRight") {
                if (videoRef) {
                    videoRef.currentTime = Math.min(
                        videoRef.currentTime + 5,
                        videoRef.duration
                    );
                    setCurrentTime(videoRef.currentTime);
                }
            }
            if (e.key === "ArrowLeft") {
                if (videoRef) {
                    videoRef.currentTime = Math.max(videoRef.currentTime - 5, 0);
                    setCurrentTime(videoRef.currentTime);
                }
            }
            if (e.key === "m") {
                handleVolumeToggle();
            }
            if (e.key === "f") {
                handleFullscreen();
            }
        }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
    >
        <div className="w-full h-full aspect-video">{children}</div>

        {/* Controls Overlay */}
        {isControlsVisible && (
            <div className="absolute bottom-0 left-0 w-full bg-black/60 backdrop-blur-sm text-white px-4 py-2 flex items-center space-x-4">
                {/* Play/Pause */}
                <button
                    onClick={togglePlay}
                    className="p-1 hover:text-primary transition"
                    aria-label={playing ? "Pause" : "Play"}
                >
                    {playing ? <LucidePause /> : <LucidePlay />}
                </button>
                {/* Progress Bar with Drag Seek */}
                <div
                    className="flex-1 h-2 bg-white/30 rounded cursor-pointer relative"
                    onClick={handleSeek}
                    onMouseDown={handleMouseDown}
                    role="slider"
                    aria-valuemin="0"
                    aria-valuemax="100"
                    aria-valuenow={progress}
                    tabIndex={0}
                    onKeyDown={(e) => {
                        if (e.key === "ArrowRight") {
                            handleSeek({
                                currentTarget: e.currentTarget,
                                clientX: e.clientX + 30,
                            });
                        }
                        if (e.key === "ArrowLeft") {
                            handleSeek({
                                currentTarget: e.currentTarget,
                                clientX: e.clientX - 30,
                            });
                        }
                    }}
                >
                    <div
                        className="h-2 bg-primary rounded pointer-events-none"
                        style={{ width: `${progress}%` }}
                    />
                    {/* Cursor/Thumb */}
                    <div
                        className="absolute top-1/2 -translate-y-1/2"
                        style={{ left: `calc(${progress}% - 8px)` }}
                    >
                        <div className="w-4 h-4 bg-primary rounded-full pointer-events-none border-2 border-white shadow" />
                    </div>
                </div>
                {/* Time Display */}
                <div className="flex items-center space-x-2">
                    <span>{formatTime(currentTime)}</span>
                    <span>/</span>
                    <span>{videoRef ? formatTime(videoRef.duration) : "00:00"}</span>
                </div>
                <div className="flex items-center space-x-2">
                    <LucideVolume
                        onClick={handleVolumeToggle}
                        className="cursor-pointer"
                        strokeWidth={muted ? 0 : 2}
                        aria-label={muted ? "Unmute" : "Mute"}
                    />
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={muted ? 0 : volume}
                        onChange={handleVolumeChange}
                        className="w-24"
                        aria-label="Volume control"
                    />
                </div>
                {/* Fullscreen */}
                <button
                    onClick={handleFullscreen}
                    className="p-1 hover:text-primary transition"
                    aria-label="Toggle Fullscreen"
                >
                    <LucideMaximize />
                </button>
            </div>
        )}
    </div>
);
}
