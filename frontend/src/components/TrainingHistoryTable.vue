<template>
  <el-card class="history-card">
    <template #header>
      <span>Upload History</span>
    </template>
    <el-table :data="records" style="width: 100%">
      <el-table-column prop="upload_time" label="Upload Time" width="160">
        <template #default="{ row }">{{ formatDate(row.upload_time) }}</template>
      </el-table-column>
      <el-table-column prop="original_filename" label="File Name" width="240" />
      <el-table-column prop="status" label="Status" width="100" />
      <!-- <el-table-column prop="trainingDate" label="Training Date" width="120">
        <template #default="{ row }">{{
          row.trainingDate ? formatDate(row.trainingDate) : '-'
        }}</template>
      </el-table-column> -->
      <el-table-column label="Actions" width="220">
        <template #default="{ row }">
          <!-- <el-button size="mini" @click="$emit('details', row.id)">Details</el-button> -->
          <el-button
            v-if="row.status === 'uploaded' || row.status === 'failed'"
            size="mini"
            type="primary"
            @click="openTrainDialog(row)"
            >Train Model</el-button
          >
          <el-button
            v-if="row.status !== 'trained' && row.status !== 'training'"
            size="mini"
            type="danger"
            @click="$emit('delete', row.id.toString())"
            >Delete</el-button
          >
          <span v-if="row.status === 'training'" style="margin-left: 8px">Training...</span>
        </template>
      </el-table-column>
    </el-table>
    <div v-if="records.length === 0" style="padding: 12px">No records</div>
    <div style="margin-top: 12px; text-align: right">
      <el-pagination
        background
        layout="total, sizes, prev, pager, next, jumper"
        :total="total"
        :page-size="pageSize"
        :current-page="currentPage"
        :page-sizes="[5, 10, 20, 50]"
        @current-change="handlePageChange"
        @size-change="handleSizeChange"
      />
    </div>

    <!-- Train dialog -->
    <el-dialog title="Start Training" v-model="dialogVisible">
      <div>
        <!-- No validation rules are required for the model name input in this form -->
        <el-form ref="formRef" :model="formModel" :rules="rules" label-position="top">
          <!-- Validation rules are defined for the model name input in this form -->
          <el-form-item label="Model name" prop="modelName">
            <el-input
              v-model="formModel.modelName"
              placeholder="Enter a name for the model"
            ></el-input>
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="isSubmitting" @click="submitTrain"
          >Start Training</el-button
        >
      </template>
    </el-dialog>
  </el-card>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElNotification } from 'element-plus'
import type { TrainingRecord } from '../api/modelTraining'
import { triggerTrain } from '../api/modelTraining'

// Define props and emits using runtime declarations for better type inference
defineProps<{
  records: TrainingRecord[]
  currentPage?: number
  pageSize?: number
  total?: number
}>()

const emit = defineEmits<{
  (e: 'refresh'): void
  (e: 'details', id: string): void
  (e: 'delete', id: string): void
  (e: 'page-change', page: number): void
  (e: 'size-change', size: number): void
}>()

function handlePageChange(page: number) {
  emit('page-change', page)
}

function handleSizeChange(size: number) {
  emit('size-change', size)
}

function formatDate(d?: string | null) {
  if (!d) return '-'
  try {
    return new Date(d).toLocaleString()
  } catch {
    return d
  }
}

const dialogVisible = ref(false)
const selectedId = ref<number | null>(null)
const formRef = ref<{ validate?: () => Promise<void> } | null>(null)
const formModel = reactive({ modelName: '' })
const isSubmitting = ref(false)

const rules = {
  modelName: [
    { required: true, message: 'Model name is required', trigger: 'blur' },
    { min: 1, message: 'Model name cannot be empty', trigger: 'blur' },
  ],
}

function openTrainDialog(row: TrainingRecord) {
  selectedId.value = row.id
  formModel.modelName = row.modelName || ''
  dialogVisible.value = true
}

async function submitTrain() {
  if (!selectedId.value) return

  // validate form
  try {
    if (formRef.value && typeof formRef.value.validate === 'function') {
      console.log('Validating form...')
      await formRef.value.validate()
    }
  } catch {
    return
  }

  isSubmitting.value = true
  try {
    await triggerTrain(selectedId.value.toString(), formModel.modelName || undefined)
    ElNotification({ title: 'Training', message: 'Training started successfully', type: 'success' })
    dialogVisible.value = false
    formModel.modelName = ''
    selectedId.value = null
    emit('refresh')
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    ElNotification({
      title: 'Error',
      message: message || 'Failed to start training',
      type: 'error',
    })
  } finally {
    isSubmitting.value = false
  }
}
</script>

<style scoped>
.history-card {
  margin-top: 16px;
  border: 1px solid #e5e7eb;
  padding: 12px;
  border-radius: 6px;
  background: white;
}
.history-table {
  width: 100%;
  border-collapse: collapse;
}
.history-table th,
.history-table td {
  padding: 8px;
  border-bottom: 1px solid #f3f4f6;
  text-align: left;
}
.actions {
  display: flex;
  gap: 8px;
  align-items: center;
}
.actions button {
  background: #111827;
  color: white;
  border-radius: 4px;
  padding: 4px 8px;
  border: none;
}
</style>
