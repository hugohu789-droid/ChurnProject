<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getDashboardStats } from '../api/dashboard'
import { ElMessage } from 'element-plus'

// define data structure for dashboard stats
interface DashboardStats {
  total_files: number
  total_models: number
  total_predictions: number
  average_accuracy: number
  recent_models: Array<{
    id: number
    model_name: number
    accuracy: number
    precision: number
    train_date: string
  }>
}

const stats = ref<DashboardStats>({
  total_files: 0,
  total_models: 0,
  total_predictions: 0,
  average_accuracy: 0,
  recent_models: []
})

const loading = ref(false)

const fetchStats = async () => {
  loading.value = true
  try {
    const res = await getDashboardStats()
    stats.value = res.data
  } catch (error) {
    console.error(error)
    ElMessage.error('Failed to load dashboard data')
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchStats()
})
</script>

<template>
  <div class="dashboard-container">
    <div class="header">
      <h2 class="title">Dashboard Overview</h2>
      <p class="subtitle">Welcome back! Here is the latest update on your churn prediction models.</p>
    </div>

    <!-- statistics cards -->
    <el-row :gutter="24">
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <template #header>
            <div class="card-header">
              <span>Total Files</span>
            </div>
          </template>
          <div class="stat-value">{{ stats.total_files }}</div>
        </el-card>
      </el-col>
      
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <template #header>
            <div class="card-header">
              <span>Trained Models</span>
            </div>
          </template>
          <div class="stat-value">{{ stats.total_models }}</div>
        </el-card>
      </el-col>
      
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <template #header>
            <div class="card-header">
              <span>Predictions</span>
            </div>
          </template>
          <div class="stat-value">{{ stats.total_predictions }}</div>
        </el-card>
      </el-col>
      
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card highlight-card">
          <template #header>
            <div class="card-header">
              <span>Avg Accuracy</span>
            </div>
          </template>
          <div class="stat-value">
            {{ (stats.average_accuracy * 100).toFixed(1) }}<span class="unit">%</span>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- latest trained models-->
    <div class="recent-section">
      <h3 class="section-title">Recent Trained Models</h3>
      <el-card shadow="never" class="table-card">
        <el-table :data="stats.recent_models" style="width: 100%" v-loading="loading">
          <el-table-column prop="model_name" label="Model Name" min-width="180" />
          <el-table-column prop="accuracy" label="Accuracy" width="150">
            <template #default="scope">
              <el-tag :type="scope.row.accuracy > 0.8 ? 'success' : 'warning'">
                {{ (scope.row.accuracy * 100).toFixed(2) }}%
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="precision" label="Precision" width="150">
            <template #default="scope">
              {{ (scope.row.precision * 100).toFixed(2) }}%
            </template>
          </el-table-column>
          <el-table-column prop="train_date" label="Date" width="200">
            <template #default="scope">
              {{ new Date(scope.row.train_date).toLocaleDateString() }} 
              {{ new Date(scope.row.train_date).toLocaleTimeString() }}
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>
  </div>
</template>

<style scoped>
.dashboard-container {
  padding: 8px;
}

.header {
  margin-bottom: 32px;
}

.title {
  font-size: 1.75rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 8px;
}

.subtitle {
  color: #64748b;
  font-size: 0.95rem;
}

.stat-card {
  border: none;
  border-radius: 12px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.card-header {
  font-size: 0.875rem;
  font-weight: 600;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat-value {
  font-size: 2.5rem;
  font-weight: 700;
  color: #0f172a;
  margin-top: 8px;
}

.unit {
  font-size: 1.25rem;
  color: #64748b;
  margin-left: 4px;
}

.highlight-card .stat-value {
  color: var(--theme-primary); 
}

.recent-section {
  margin-top: 40px;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #334155;
  margin-bottom: 16px;
  padding-left: 4px;
}

.table-card {
  border: 1px solid #f1f5f9;
  border-radius: 12px;
}
</style>