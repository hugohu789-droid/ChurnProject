import { client } from './client'

export const getDashboardStats = () => {
  return client.get('/dashboard/stats')
}

export default getDashboardStats