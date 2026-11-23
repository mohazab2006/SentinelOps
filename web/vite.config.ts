import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the console runs on Vite and talks to `sentinelops serve` on port 8000.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // The lazily loaded globe chunk is mostly three.js (~2 MB); the app shell stays small.
  build: { chunkSizeWarningLimit: 2500 },
  server: {
    proxy: {
      "/api": api,
      "/ws": { target: api, ws: true },
    },
  },
});
