import client from './client'
import type { LoginPayload, RegisterPayload, TokenResponse, User } from '@/types/auth'

export const authApi = {
  register: (payload: RegisterPayload) =>
    client.post<User>('/auth/register', payload),

  login: (payload: LoginPayload) =>
    client.post<TokenResponse>('/auth/login', payload),

  refresh: (refreshToken: string) =>
    client.post<TokenResponse>('/auth/refresh', { refresh_token: refreshToken }),

  me: () => client.get<User>('/auth/me'),
}
