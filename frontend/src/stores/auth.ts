import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { authApi } from '@/api/auth'
import type { User, LoginPayload, RegisterPayload } from '@/types/auth'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const router = useRouter()

  const isAuthenticated = computed(() => !!user.value)

  function setTokens(accessToken: string, refreshToken: string) {
    localStorage.setItem('access_token', accessToken)
    localStorage.setItem('refresh_token', refreshToken)
  }

  function clearTokens() {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
  }

  async function fetchMe() {
    const token = localStorage.getItem('access_token')
    if (!token) return
    try {
      const { data } = await authApi.me()
      user.value = data
    } catch {
      clearTokens()
    }
  }

  async function login(payload: LoginPayload) {
    isLoading.value = true
    error.value = null
    try {
      const { data } = await authApi.login(payload)
      setTokens(data.access_token, data.refresh_token)
      await fetchMe()
      router.push('/dashboard')
    } catch (e: any) {
      error.value = e?.response?.data?.detail ?? 'Login failed'
    } finally {
      isLoading.value = false
    }
  }

  async function register(payload: RegisterPayload) {
    isLoading.value = true
    error.value = null
    try {
      await authApi.register(payload)
      // Auto-login: get tokens directly then fetch user
      const { data } = await authApi.login({ email: payload.email, password: payload.password })
      setTokens(data.access_token, data.refresh_token)
      await fetchMe()
      await router.push('/dashboard')
    } catch (e: any) {
      error.value = e?.response?.data?.detail ?? 'Registration failed'
    } finally {
      isLoading.value = false
    }
  }

  function logout() {
    user.value = null
    clearTokens()
    router.push('/login')
  }

  return { user, isLoading, error, isAuthenticated, fetchMe, login, register, logout }
})
