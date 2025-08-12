import { fileURLToPath } from "url";
import { dirname } from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": `${__dirname}/src`,
    },
  },
  server: {
    port: 5500,
    proxy: {
      "/api": "http://127.0.0.1:5000",
    },
  },
});
