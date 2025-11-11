<template>
  <div class="page">
    <h1>Model Training</h1>
    <el-row :gutter="16" style="margin-bottom: 16px">
      <el-col :span="16">
        <UploadTrainingData @uploaded="onUploaded" />
      </el-col>
    </el-row>
    <el-row :gutter="24">
      <el-col :span="24">
        <TrainingHistoryTable
          :records="records"
          :current-page="currentPage"
          :page-size="pageSize"
          :total="total"
          @delete="onDelete"
          @train="onTrain"
          @details="onDetails"
          @refresh="load"
          @page-change="onPageChange"
          @size-change="onPageSizeChange"
        />
      </el-col>
    </el-row>

    <el-dialog width="60%" v-model:visible="detailsDialogVisible" title="Details">
      <pre style="white-space: pre-wrap">{{ selectedDetails }}</pre>
      <template #footer>
        <el-button @click="detailsDialogVisible = false">Close</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessageBox, ElMessage } from 'element-plus'
import UploadTrainingData from '../components/UploadTrainingData.vue'
import TrainingHistoryTable from '../components/TrainingHistoryTable.vue'
import api from '../api/modelTraining'
import type { TrainingRecord } from '../api/modelTraining'

const records = ref<TrainingRecord[]>([])
const loading = ref(false)
const selectedDetails = ref<Record<string, unknown> | null>(null)
const detailsDialogVisible = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

async function load() {
  loading.value = true
  try {
    const resp = await api.fetchHistory(currentPage.value, pageSize.value)
    records.value = resp.records
    total.value = resp.total
  } catch (err) {
    console.error(err)
    ElMessage({ message: 'Failed to fetch history', type: 'error' })
  } finally {
    loading.value = false
  }
}

function onUploaded() {
  // refresh history
  currentPage.value = 1
  load()
}

async function onDelete(id: string) {
  try {
    await ElMessageBox.confirm('Are you sure you want to delete this record?', 'Warning', {
      type: 'warning',
    })
  } catch {
    return
  }
  try {
    await api.deleteRecord(id)
    await load()
    ElMessage({ message: 'Deleted', type: 'success' })
  } catch (err) {
    console.error(err)
    ElMessage({ message: 'Delete failed', type: 'error' })
  }
}

async function onTrain(id: string) {
  try {
    await api.triggerTrain(id)
    // quick refresh to show training state
    await load()
  } catch (err) {
    console.error(err)
    ElMessage({ message: 'Failed to trigger training', type: 'error' })
  }
}

async function onDetails(id: string) {
  try {
    const d = await api.getDetails(id)
    selectedDetails.value = d
    detailsDialogVisible.value = true
  } catch (err) {
    console.error(err)
    ElMessage({ message: 'Failed to fetch details', type: 'error' })
  }
}

onMounted(() => {
  load()
})

function onPageChange(page: number) {
  currentPage.value = page
  load()
}

function onPageSizeChange(size: number) {
  pageSize.value = size
  currentPage.value = 1
  load()
}
</script>

<style scoped>
.page {
  padding: 18px;
}
.details-panel {
  margin-top: 16px;
  border: 1px solid #e5e7eb;
  padding: 12px;
  border-radius: 6px;
  background: white;
}
</style>
