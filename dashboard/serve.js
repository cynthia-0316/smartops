// Zero-dependency static server for local use:  node dashboard/serve.js [port]
// (The dashboard also works by opening dashboard/index.html directly in a browser.)
'use strict';
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const root = __dirname;
const port = Number(process.argv[2] || process.env.PORT || 8080);
const types = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml', '.csv': 'text/csv'};

http.createServer((req, res) => {
  const urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const rel = path.normalize(urlPath === '/' ? '/index.html' : urlPath);
  const file = path.join(root, rel);
  if (!file.startsWith(root + path.sep) || file.includes(path.sep + 'tests' + path.sep)) { res.writeHead(403).end('Forbidden'); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404).end('Not found'); return; }
    res.writeHead(200, {'Content-Type': types[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store'});
    res.end(data);
  });
}).listen(port, '127.0.0.1', () => console.log('SmartOps dashboard: http://localhost:' + port));
