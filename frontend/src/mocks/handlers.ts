// @ts-nocheck
/* eslint-disable @typescript-eslint/no-explicit-any */
// Note: msw types can cause type-check issues in this project config; file is intentionally nochecked for dev mocks.
import { rest } from 'msw'
import type { TrainingRecord } from '../api/modelTraining'

// Create an in-memory list of mock records (no external deps)
const makeRecord = (i: number): TrainingRecord => {
  const id = `${Date.now()}-${i}-${Math.floor(Math.random() * 10000)}`
  const now = Date.now() - i * 1000 * 60 * 60 * 24
  return {
    id,
    uploadDate: new Date(now).toISOString(),
    modelName: `mock-model-${i}`,
    trainingDate: i % 3 === 0 ? new Date(now + 1000 * 60 * 60).toISOString() : null,
    recordCount: 100 + i,
    auc: i % 2 === 0 ? +(0.7 + (i % 10) * 0.01).toFixed(3) : null,
    recall: i % 2 === 0 ? +(0.6 + (i % 10) * 0.01).toFixed(3) : null,
    accuracy: i % 2 === 0 ? +(0.75 + (i % 10) * 0.01).toFixed(3) : null,
    status: i % 5 === 0 ? 'training' : i % 4 === 0 ? 'failed' : 'trained',
  }
}

const MOCK_DB: TrainingRecord[] = Array.from({ length: 57 }).map((_, i) => makeRecord(i + 1))

export const handlers = [
  // GET history with pagination
  rest.get('/api/training/history', (req: any, res: any, ctx: any) => {
    const page = Number(req.url.searchParams.get('page') ?? '1')
    const pageSize = Number(req.url.searchParams.get('pageSize') ?? '10')
    const start = (page - 1) * pageSize
    const end = start + pageSize
    const slice = MOCK_DB.slice(start, end)
    return res(ctx.status(200), ctx.json({ records: slice, total: MOCK_DB.length }))
  }),

  // POST upload
  rest.post('/api/training/upload', async (req: any, res: any, ctx: any) => {
    // pretend to process multipart; add a new record to the head
    const id = `${Date.now()}-${Math.floor(Math.random() * 10000)}`
    const now = new Date().toISOString()
    const record: TrainingRecord = {
      id,
      uploadDate: now,
      modelName: 'uploaded-model',
      trainingDate: null,
      recordCount: 0,
      auc: null,
      recall: null,
      accuracy: null,
      status: 'uploaded',
    }
    MOCK_DB.unshift(record)
    return res(ctx.status(200), ctx.json({ success: true, id }))
  }),

  // DELETE record
  rest.delete('/api/training/:id', (req: any, res: any, ctx: any) => {
    const { id } = req.params as { id: string }
    const idx = MOCK_DB.findIndex((r) => r.id === id)
    if (idx >= 0) {
      MOCK_DB.splice(idx, 1)
      return res(ctx.status(200), ctx.json({ success: true }))
    }
    return res(ctx.status(404), ctx.json({ error: 'Not found' }))
  }),

  // POST train
  rest.post('/api/training/:id/train', (req: any, res: any, ctx: any) => {
    const { id } = req.params as { id: string }
    const r = MOCK_DB.find((x) => x.id === id)
    if (r) {
      r.status = 'training'
      // after a short delay simulate completion (not actually delayed here)
      return res(ctx.status(200), ctx.json({ success: true }))
    }
    return res(ctx.status(404), ctx.json({ error: 'Not found' }))
  }),

  // GET details
  rest.get('/api/training/:id', (req: any, res: any, ctx: any) => {
    const { id } = req.params as { id: string }
    const r = MOCK_DB.find((x) => x.id === id)
    if (r) {
      return res(ctx.status(200), ctx.json(r))
    }
    return res(ctx.status(404), ctx.json({ error: 'Not found' }))
  }),

  // POST predict - accepts multipart form (file + modelId) and returns CSV
  rest.post('/api/predict', async (req: any, res: any, ctx: any) => {
    // In a real server you'd parse the file and run the model.
    // Here we simulate work and return a small CSV as a text/csv response.
    const csv = 'id,prediction\n1001,0\n1002,1\n1003,0\n'
    return res(ctx.delay(1000), ctx.status(200), ctx.set('Content-Type', 'text/csv'), ctx.body(csv))
  }),
]
