import { storeToRefs } from 'pinia'
import { useAuthStore } from '@/stores/auth'

export function useAuth() {
  const store = useAuthStore()
  const { user, isLoading, error, isAuthenticated } = storeToRefs(store)
  return { user, isLoading, error, isAuthenticated, login: store.login, register: store.register, logout: store.logout }
}
