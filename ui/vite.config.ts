import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, import.meta.dirname)
  return {
    plugins: [react()],
    resolve: {
      alias: { '@': path.resolve(import.meta.dirname, './src') },
    },
    server: {
      // localhost only: never expose the dev server to the network (no `--host`).
      host: '127.0.0.1',
      port: 5173,
      // During `pnpm dev` the browser calls /api on this server and Vite forwards it to the
      // Modal gateway, so the page is same-origin and CORS never applies, whatever the port.
      proxy: env.VITE_API_URL ? { '/api': { target: env.VITE_API_URL, changeOrigin: true } } : undefined,
    },
  }
})
