import React from 'react';

export default function App() {
  return (
    <div style={{fontFamily: 'system-ui, sans-serif', padding: 24}}>
      <h1 style={{color: '#7c3aed'}}>Video Stitcher</h1>
      <p>Welcome — this is the renderer application root.</p>
      <section style={{marginTop: 16}}>
        <button
          onClick={() => alert('This is a placeholder action')}
          className='bg-red-500 text-white px-3 py-2 rounded-md hover:bg-cyan-600 transition-colors'
        >
          Test Action
        </button>
      </section>
    </div>
  );
}
