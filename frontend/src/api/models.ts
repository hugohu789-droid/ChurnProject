import { client } from './client'

export type ModelMetadata = {
  parameters: Record<string, unknown>
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
  train_date: string
  metadata?: ModelMetadata | null
}

export type Prediction = {
  id: string
  train_model_id: number
  train_model_name: string
  result1_path: string
  result2_path: string
  predict_date: string
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

export async function fetchModels(page = 1, pageSize = 10): Promise<FetchModelsResponse> {
  const res = await client.post('/models/list', { page, page_size: pageSize })
  return res.data
}

export async function fetchTrainedModels(): Promise<FetchModelsResponse> {
  const res = await client.post('/models/list', { page: 1, page_size: 100 })
  return res.data
}

export async function getModelDetails(id: string): Promise<Model> {
  const res = await client.get(`/models/${id}`)
  return res.data
}

export async function fetchPredictions(page = 1, pageSize = 10): Promise<FetchPredictionsResponse> {
  const res = await client.post('/predictions/list', { page, page_size: pageSize })
  return res.data
}

export async function uploadPredictFile(fd: FormData) {
  const res = await client.post('/predictions/run', fd)
  return res.data
}

export default { fetchModels, fetchTrainedModels, getModelDetails, fetchPredictions, uploadPredictFile }
