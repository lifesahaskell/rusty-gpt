import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: process.env.VITE_API_PROXY_TARGET
      ? {
          '/api': process.env.VITE_API_PROXY_TARGET,
        }
      : undefined,
  },
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.test.{ts,tsx}'],
    setupFiles: './tests/setup.ts',
    globals: true,
  },
})
