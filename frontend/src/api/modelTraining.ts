import { client } from './client'

export type TrainingRecord = {
  id: number
  original_filename: string
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

export async function uploadFile(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await client.post('/datasets/upload', fd)
  return res.data
}

export async function fetchHistory(page = 1, pageSize = 10): Promise<FetchHistoryResponse> {
  const res = await client.post('/training/list', { page, page_size: pageSize })
  return res.data
}

export async function getDetails(id: number): Promise<TrainingRecord> {
  const res = await client.get(`/datasets/${id}`)
  return res.data
}

export async function deleteRecord(id: number) {
  const res = await client.delete(`/datasets/${id}`)
  return res.data
}

export async function triggerTrain(id: number, modelName?: string) {
  const res = await client.post('/training/train', { id, model_name: modelName })
  return res.data
}

export default { uploadFile, fetchHistory, getDetails, deleteRecord, triggerTrain }
