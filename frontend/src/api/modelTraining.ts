import { client } from './client'
import { mockTrainingRecords } from './mockData'

export type TrainingRecord = {
  id: number
  original_filename: string // ISO date
  saved_filename: string
  file_path?: string | null
  upload_time: string
  file_size?: number | null
  status: 'uploaded' | 'training' | 'trained' | 'failed'
  modelName?: string
}

export type FetchHistoryResponse = {
  records: TrainingRecord[]
  total: number
}

// using shared client from src/api/client.ts

export async function uploadFile(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await client.post('/upload', fd)
  return res.data
}

export async function fetchHistory(page = 1, pageSize = 10): Promise<FetchHistoryResponse> {
  // use Vite's runtime flag for development
  const body = {
    page: page,
    page_size: pageSize,
  }
  // if (import.meta.env.DEV) {
  // const start = (page - 1) * pageSize
  // const end = start + pageSize
  // return {
  //   records: mockTrainingRecords.slice(start, end),
  //   total: mockTrainingRecords.length,
  // }
  // }
  const res = await client.post('/modeltraining/list', body)
  return res.data
}

export async function deleteRecord(id: string) {
  const res = await client.delete(`/modeltraining/${id}`)
  return res.data
}

export async function triggerTrain(id: string, modelName?: string) {
  // send optional modelName in body so backend can name the model/run
  const body = {
    id: id,
    modelName: modelName,
  }
  //modelName ? { modelName } : undefined
  const res = await client.post(`/modeltraining/train`, body)
  return res.data
}

export async function getDetails(id: string) {
  const res = await client.get(`/training/${id}`)
  return res.data
}

export default { uploadFile, fetchHistory, deleteRecord, triggerTrain, getDetails }
