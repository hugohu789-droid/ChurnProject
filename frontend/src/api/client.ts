import axios, { type AxiosError } from 'axios'
import { useAuthStore } from '@/stores/auth'

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string) || '/api/v1'

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000,
})

// Attach the access token to every request
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Attempt a token refresh on 401, then retry the original request once
client.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const original = error.config as typeof error.config & { _retry?: boolean }
    if (error.response?.status === 401 && !original?._retry) {
      original._retry = true
      const refreshToken = localStorage.getItem('refresh_token')
      if (refreshToken) {
        try {
          const { data } = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken,
          })
          localStorage.setItem('access_token', data.access_token)
          localStorage.setItem('refresh_token', data.refresh_token)
          if (original?.headers) {
            original.headers.Authorization = `Bearer ${data.access_token}`
          }
          return client(original!)
        } catch {
          useAuthStore().logout()
        }
      }
    }
    return Promise.reject(error)
  },
)

export default client
