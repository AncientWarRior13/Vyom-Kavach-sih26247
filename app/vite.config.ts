import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@tokens': path.resolve(__dirname, '../packages/design-tokens'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8090',
      '/aar/session': 'http://localhost:8090',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
