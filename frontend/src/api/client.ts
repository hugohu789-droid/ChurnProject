import axios from 'axios'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string) || '/api'

export const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000,
})

export default client
