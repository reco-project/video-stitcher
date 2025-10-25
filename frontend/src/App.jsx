import React, { useState, useEffect } from "react";
import Viewer from "./features/viewer/components/Viewer.jsx";
import matches from "./app/data/matches.js";

export default function App() {
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // TODO: possibly should move this to a custom hook or elsewhere
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
    document.addEventListener("msfullscreenchange", handleFullscreenChange);

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener(
        "webkitfullscreenchange",
        handleFullscreenChange
      );
      document.removeEventListener(
        "msfullscreenchange",
        handleFullscreenChange
      );
    };
  }, []);

  return (
    <div className="flex flex-col items-center w-full p-4 gap-4">
      <h1 className="text-purple-600">Video Stitcher</h1>
      <p>Welcome — this is the renderer application root.</p>

      <div className="w-full max-w-2xl">
        <label className="block mb-2 font-bold">Select match</label>
        <select
          className="w-full p-2 rounded border"
          value={selectedMatch ? selectedMatch.id : ""}
          onChange={(e) => {
            const id = e.target.value;
            const m = matches.find((mm) => mm.id === id) || null;
            setSelectedMatch(m);
          }}
        >
          <option value="">-- choose match --</option>
          {matches.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {selectedMatch && (
        <section
          className={
            "w-full aspect-video h-fit" +
            (isFullscreen ? " absolute top-0 left-0 z-50" : "")
          }
        >
          <Viewer key={selectedMatch.id} selectedMatch={selectedMatch} />
        </section>
      )}
    </div>
  );
}
