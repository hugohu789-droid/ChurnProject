<template>
  <el-card class="upload-card">
    <template #header>
      <span>Upload Training Data</span>
    </template>

    <el-form :model="{ modelName }" label-width="120px">
      <el-form-item label="Select file (CSV)">
        <input
          ref="fileInput"
          type="file"
          accept=".csv"
          @change="onFileChange"
          style="display: block"
        />
        <div v-if="selectedFile" style="margin-top: 8px">Selected: {{ selectedFile.name }}</div>
      </el-form-item>
      <el-form-item class="model-select-section">
        <label class="label">Select model</label>
        <el-select
          v-model="selectedModel"
          placeholder="Choose a trained model"
          clearable
          style="width: 100%"
        >
          <el-option
            v-for="model in models"
            :key="model.id"
            :label="model.model_name"
            :value="model.id"
          />
        </el-select>
      </el-form-item>
      <el-form-item>
        <!-- <el-button
          type="primary"
          :disabled="!selectedFile || uploading"
          :loading="uploading"
          @click="upload"
          >Upload and Save</el-button
        > -->
        <el-button
          type="primary"
          :disabled="!canPredict || uploading"
          :loading="uploading"
          @click="upload"
          >Predict</el-button
        >
      </el-form-item>
    </el-form>
  </el-card>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import type { Model } from '../api/models'
import api from '../api/models'

const selectedModel = ref<string>('')
const models = ref<Model[]>([])
const emit = defineEmits(['uploaded'])

const selectedFile = ref<File | null>(null)

const canPredict = computed(() => selectedFile.value !== null && selectedModel.value !== '')
const modelName = ref('')
const uploading = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

async function loadModels() {
  try {
    const resp = await api.fetchTrainedModels()
    models.value = resp.records
  } catch (err) {
    console.error(err)
    ElMessage({ message: 'Failed to fetch trained models', type: 'error' })
  } finally {
  }
}

function onFileChange(e: Event) {
  const t = e.target as HTMLInputElement
  selectedFile.value = t.files && t.files[0] ? t.files[0] : null
}

async function upload() {
  if (!selectedFile.value) return
  uploading.value = true
  try {
    const form = new FormData()
    form.append('file', selectedFile.value)
    form.append('modelId', selectedModel.value)

    const res = await api.uploadPredictFile(form)
    emit('uploaded', res)
    modelName.value = ''
    if (fileInput.value) fileInput.value.value = ''
    selectedFile.value = null
    selectedModel.value = ''
    ElMessage({ message: 'Upload successful', type: 'success' })
  } catch (err) {
    console.error(err)
    ElMessage({
      message: 'Upload failed: ' + (err instanceof Error ? err.message : String(err)),
      type: 'error',
    })
  } finally {
    uploading.value = false
  }
}

onMounted(() => {
  loadModels()
})
</script>

<style scoped>
.upload-card {
  border: 1px solid #e5e7eb;
  padding: 16px;
  border-radius: 6px;
  background: white;
}
.form-row {
  margin-bottom: 8px;
  display: flex;
  gap: 8px;
  align-items: center;
}
.form-row label {
  width: 120px;
}
.actions {
  margin-top: 8px;
}
button {
  background: #2563eb;
  color: white;
  padding: 6px 10px;
  border-radius: 4px;
  border: none;
}
button:disabled {
  opacity: 0.6;
}
</style>
