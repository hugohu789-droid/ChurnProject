import axios, { type AxiosError } from 'axios'

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string) || '/api/v1'

export const client = axios.create({
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

// On 401: try to refresh the token, then retry the original request once.
// Importing useAuthStore here would create a circular dependency
// (client → auth store → api/auth → client), so we clear tokens directly.
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
          // Refresh failed — clear session and redirect to login
          localStorage.removeItem('access_token')
          localStorage.removeItem('refresh_token')
          window.location.href = '/#/login'
        }
      }
    }
    return Promise.reject(error)
  },
)

export default client
