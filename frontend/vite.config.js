import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/decide": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/decisions": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/map": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/customers": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/steer": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/corrections": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
