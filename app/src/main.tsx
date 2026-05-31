import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/index.css';

// Default to whatever Electron has already wired up (may be 0 during bootstrap).
// App.tsx listens for amv:bootstrap-ready and updates __AMV_API_PORT__ before
// any API call fires.
const fromEnv = Number(import.meta.env.VITE_API_PORT);
(window as any).__AMV_API_PORT__ = Number.isFinite(fromEnv) && fromEnv > 0 ? fromEnv : 0;

if (window.amvBridge?.getApiPort) {
  window.amvBridge.getApiPort().then((port) => {
    if (port > 0) (window as any).__AMV_API_PORT__ = port;
  }).catch(() => {});
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
