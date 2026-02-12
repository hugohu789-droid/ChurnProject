import { fileURLToPath } from 'node:url'
import { mergeConfig, defineConfig, configDefaults } from 'vitest/config'
import viteConfig from './vite.config'

// vite.config may export a function (UserConfigFn) or a config object. If it's a function, call it for a 'test' mode to obtain a config object.
const baseViteConfig =
  typeof viteConfig === 'function' ? viteConfig({ mode: 'test' } as any) : viteConfig

export default mergeConfig(
  baseViteConfig as any,
  defineConfig({
    test: {
      environment: 'jsdom',
      exclude: [...configDefaults.exclude, 'e2e/**'],
      root: fileURLToPath(new URL('./', import.meta.url)),
      globals: true,
    },
  }),
)
