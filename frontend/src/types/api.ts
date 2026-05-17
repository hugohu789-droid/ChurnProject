export interface PagedResponse<T> {
  page: number
  page_size: number
  total: number
  records: T[]
}

export interface ApiError {
  detail: string
}
