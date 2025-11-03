import React from 'react';
import { createRoot } from 'react-dom/client';
import './styles/globals.css';
import App from './app/App.jsx';

console.log('👋 renderer starting - mounting React App');

const rootEl = document.getElementById('root');
if (rootEl) {
  const root = createRoot(rootEl);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
} else {
  console.error('Root element not found: #root');
}
