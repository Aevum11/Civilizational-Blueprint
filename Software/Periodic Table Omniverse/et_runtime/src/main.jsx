import React from 'react' 
import { createRoot } from 'react-dom/client' 
 
// ERROR REPORTER 
const reportError = (err) => { 
  const msg = err instanceof Error ? err.stack : String(err); 
  fetch('/__client_log', { method: 'POST', body: msg }).catch(()=>{}); 
  console.error(err); 
}; 
window.addEventListener('error', (e) => reportError(e.error)); 
window.addEventListener('unhandledrejection', (e) => reportError(e.reason)); 
 
// GLOBAL SHIMS 
window.React = React; 
window.ReactDOM = { createRoot }; 
 
// LOAD USER SCRIPT 
import('./UserScript.jsx').catch(reportError); 
