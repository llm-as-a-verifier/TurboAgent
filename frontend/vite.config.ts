import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  base: '/visualizer/',
  root: '.',
  build: {
    outDir: '../turbo_agent/visualizer-dist',
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@ui': path.resolve(__dirname, 'src/visualizer/ui'),
    },
  },
  server: {
    port: 8887,
    proxy: {
      '/visualizer/api': {
        target: 'http://localhost:8888',
        changeOrigin: true,
      },
    },
  },
});
