import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const app = express();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST_DIR = path.join(__dirname, '..');

app.use(express.static(DIST_DIR, {
  index: ['index.html']
}));

const PORT = process.env.PORT || 8088;

app.listen(PORT, () => {
  console.log(`CRM frontend server running on port ${PORT}`);
  console.log(`Open http://localhost:${PORT} in your browser`);
});
