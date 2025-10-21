import React from 'react'
import Viewer from './features/viewer/components/Viewer';

export default function App() {
  return (
    <div className="flex flex-col items-center w-full h-full p-4 gap-4">
      <h1 className="text-purple-600">Video Stitcher</h1>
      <p>Welcome — this is the renderer application root.</p>
      <section className='w-full lg:w-3/4 aspect-video h-fit max-h-screen'>
        <Viewer
          cameraAxisOffset={0.23}
        />
      </section>
    </div>
  );
}
