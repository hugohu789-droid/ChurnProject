import { client } from './client'

export type ModelMetadata = {
  parameters: {
    epochs: number
    batchSize: number
    learningRate: number
  }
  features: string[]
}

export type Model = {
  id: string
  file_id: number
  record_number: number
  accuracy: number
  recall_rate: number
  precision: number
  model_name: string
  train_date: string // ISO date string
  metadata?: ModelMetadata
}

export type Prediction = {
  id: string
  file_path: number
  predict_date: string // ISO date string
  result1_path: number
  result2_path: number
  status: 'predicting' | 'completed' | 'failed'
}

export type FetchModelsResponse = {
  records: Model[]
  total: number
}

export type FetchPredictionsResponse = {
  records: Prediction[]
  total: number
}

export async function uploadPredictFile(fd: FormData) {
  const res = await client.post('/predict', fd)
  return res.data
}

const mockModels: Model[] = Array.from({ length: 20 }).map((_, i) => ({
  id: `model-${i + 1}`,
  file_id: i + 1,
  record_number: i + 1,
  model_name: `ChurnModel_v${i + 1}`,
  accuracy: 0.75 + Math.random() * 0.2,
  recall_rate: 0.7 + Math.random() * 0.25,
  train_date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
  metadata: {
    parameters: {
      epochs: 100,
      batchSize: 32,
      learningRate: 0.001,
    },
    features: ['age', 'tenure', 'totalCharges', 'monthlyCharges'],
  },
}))

export async function fetchModels(page = 1, pageSize = 10): Promise<FetchModelsResponse> {
  // Use mock data in development
  // if (import.meta.env.DEV) {
  //   const start = (page - 1) * pageSize
  //   const end = start + pageSize
  //   return {
  //     models: mockModels.slice(start, end),
  //     total: mockModels.length,
  //   }
  // }
  const body = {
    page: page,
    page_size: pageSize,
  }
  const res = await client.post('/models/list', body)
  return res.data
}

export async function fetchTrainedModels(): Promise<FetchModelsResponse> {
  const res = await client.post('/trained/models')
  return res.data
}

export async function fetchPredictions(page = 1, pageSize = 10): Promise<FetchPredictionsResponse> {
  const body = {
    page: page,
    page_size: pageSize,
  }
  const res = await client.post('/predictions', body)
  return res.data
}

export async function getModelDetails(id: string): Promise<Model> {
  // if (import.meta.env.DEV) {
  //   const model = mockModels.find((m) => m.id === id)
  //   if (!model) throw new Error('Model not found')
  //   return model
  // }
  const res = await client.get(`/models/${id}`)
  console.log('Fetched model details:', res.data)
  return res.data
}

export default {
  fetchModels,
  getModelDetails,
  fetchPredictions,
  uploadPredictFile,
  fetchTrainedModels,
}
