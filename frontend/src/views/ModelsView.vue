<template>
  <div class="page">
    <h1>Models</h1>
    <el-card class="models-list">
      <el-table :data="models" style="width: 100%">
        <el-table-column prop="model_name" label="Model Name" />
        <el-table-column prop="accuracy" label="Accuracy" width="120">
          <template #default="{ row }"> {{ (row.accuracy * 100).toFixed(2) }}% </template>
        </el-table-column>
        <el-table-column prop="recall_rate" label="Recall Rate" width="120">
          <template #default="{ row }"> {{ (row.recall_rate * 100).toFixed(2) }}% </template>
        </el-table-column>
        <el-table-column prop="precision" label="Precision" width="120">
          <template #default="{ row }"> {{ (row.precision * 100).toFixed(2) }}% </template>
        </el-table-column>
        <el-table-column prop="train_date" label="Training Time" width="180">
          <template #default="{ row }">
            {{ formatDate(row.train_date) }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="120">
          <template #default="{ row }">
            <el-button size="mini" type="primary" @click="showDetails(row.id)"> Details </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div style="margin-top: 12px; text-align: right">
        <el-pagination
          background
          layout="total, sizes, prev, pager, next, jumper"
          :total="total"
          :page-size="pageSize"
          :current-page="currentPage"
          :page-sizes="[5, 10, 20, 50]"
          @current-change="onPageChange"
          @size-change="onPageSizeChange"
        />
      </div>
    </el-card>

    <!-- Details Dialog -->
    <el-dialog width="60%" v-model="detailsDialogVisible" title="Model Details">
      <template v-if="selectedModel">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="Model Name">{{
            selectedModel.model_name
          }}</el-descriptions-item>
          <el-descriptions-item label="ID">{{ selectedModel.id }}</el-descriptions-item>
          <el-descriptions-item label="Accuracy"
            >{{ (selectedModel.accuracy * 100).toFixed(2) }}%</el-descriptions-item
          >
          <el-descriptions-item label="Recall Rate"
            >{{ (selectedModel.recall_rate * 100).toFixed(2) }}%</el-descriptions-item
          >
          <el-descriptions-item label="Precision"
            >{{ (selectedModel.precision * 100).toFixed(2) }}%</el-descriptions-item
          >
          <el-descriptions-item label="Training Time">{{
            formatDate(selectedModel.train_date)
          }}</el-descriptions-item>
          <el-descriptions-item label="File ID">#{{ selectedModel.file_id }}</el-descriptions-item>
        </el-descriptions>

        <div v-if="selectedModel.metadata" class="metadata-section">
          <h3>Additional Information</h3>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="Training Parameters">
              <pre>{{ JSON.stringify(selectedModel.metadata.parameters, null, 2) }}</pre>
            </el-descriptions-item>
            <el-descriptions-item label="Features">
              {{ selectedModel.metadata.features.join(', ') }}
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </template>
      <template #footer>
        <el-button @click="detailsDialogVisible = false">Close</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import type { Model } from '../api/models'
import api from '../api/models'

const models = ref<Model[]>([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const detailsDialogVisible = ref(false)
const selectedModel = ref<Model | null>(null)

async function loadModels() {
  loading.value = true
  try {
    const resp = await api.fetchModels(currentPage.value, pageSize.value)
    models.value = resp.records
    total.value = resp.total
    console.log('Fetched models', models.value)
  } catch (err) {
    console.error(err)
    ElMessage({ message: 'Failed to fetch models', type: 'error' })
  } finally {
    loading.value = false
  }
}

async function showDetails(id: string) {
  try {
    const model = await api.getModelDetails(id)
    selectedModel.value = model
    detailsDialogVisible.value = true
  } catch (err) {
    console.error(err)
    ElMessage({ message: 'Failed to fetch model details', type: 'error' })
  }
}

function formatDate(d?: string | null) {
  if (!d) return '-'
  try {
    return new Date(d).toLocaleString()
  } catch {
    return d
  }
}

function onPageChange(page: number) {
  currentPage.value = page
  loadModels()
}

function onPageSizeChange(size: number) {
  pageSize.value = size
  currentPage.value = 1
  loadModels()
}

onMounted(() => {
  loadModels()
})
</script>

<style scoped>
.page {
  padding: 18px;
}
.models-list {
  margin-top: 16px;
}
.metadata-section {
  margin-top: 24px;
}
pre {
  background: #f6f8fa;
  padding: 12px;
  border-radius: 4px;
  margin: 0;
}
</style>
