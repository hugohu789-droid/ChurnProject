// @ts-nocheck
// Note: dev mock worker; types intentionally disabled to avoid project ts config issues
import { setupWorker } from 'msw'
import { handlers } from './handlers'

export const worker = setupWorker(...handlers)
