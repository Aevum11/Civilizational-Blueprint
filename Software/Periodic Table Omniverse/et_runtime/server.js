import fs from 'fs'; 
import path from 'path'; 
import { fileURLToPath } from 'url'; 
import { createServer } from 'vite'; 
import { exec } from 'child_process'; 
 
const __dirname = path.dirname(fileURLToPath(import.meta.url)); 
 
(async () => { 
  const vite = await createServer({ 
    configFile: path.resolve(__dirname, 'vite.config.js'), 
    root: __dirname, 
    server: { port: 5173 } 
  }); 
 
  // LOGGING MIDDLEWARE 
  vite.middlewares.use('/__client_log', (req, res, next) => { 
    if (req.method === 'POST') { 
      let body = ''; 
      req.on('data', c => { body += c.toString(); }); 
      req.on('end', () => { 
        console.log('\x1b[31m[BROWSER ERROR]\x1b[0m'); 
        console.log('\x1b[31m' + body + '\x1b[0m'); 
        res.end('ok'); 
      }); 
      return; 
    } 
    next(); 
  }); 
 
  await vite.listen(); 
  console.log('\x1b[32m[SERVER]\x1b[0m ET Omniverse running at http://localhost:5173'); 
  vite.printUrls(); 
  const start = (process.platform == 'darwin'? 'open': process.platform == 'win32'? 'start': 'xdg-open'); 
  exec(start + ' http://localhost:5173'); 
})(); 
