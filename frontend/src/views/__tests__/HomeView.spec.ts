// /\ChurnProject\frontend\src\views\__tests__\HomeView.spec.ts
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import HomeView from '../HomeView.vue'
import { ElCard, ElRow, ElCol, ElTable, ElTableColumn, ElTag } from 'element-plus'

// Mock API Module
vi.mock('../api/dashboard', () => ({
  getDashboardStats: vi.fn(() => Promise.resolve({
    data: {
      total_files: 10,
      total_models: 5,
      total_predictions: 120,
      average_accuracy: 0.885,
      recent_models: [
        {
          id: 1,
          model_name: 'Test Model',
          accuracy: 0.95,
          precision: 0.92,
          train_date: '2023-01-01T12:00:00'
        }
      ]
    }
  }))
}))

describe('HomeView', () => {
  it('renders dashboard title properly', () => {
    // Use shallowMount or register component stubs to avoid Element Plus rendering issues.
    const wrapper = mount(HomeView, {
      global: {
        stubs: {
          ElCard: true,
          ElRow: true,
          ElCol: true,
          ElTable: true,
          ElTableColumn: true,
          ElTag: true
        }
      }
    })
    
    expect(wrapper.text()).toContain('Dashboard Overview')
    expect(wrapper.text()).toContain('Welcome back')
  })
})
