<template>
  <div class="page">
    <h1>Predict</h1>
    <el-card class="predict-view">
      <h2>Model Prediction</h2>
      <el-row :gutter="16" style="margin-bottom: 16px">
        <el-col :span="16">
          <UploadPredictData @uploaded="onUploaded" />
        </el-col>
      </el-row>
      <!-- <div class="upload-section">
        <label class="label">Test data file</label>
        <el-upload
          class="upload"
          :before-upload="beforeUpload"
          :on-remove="handleRemove"
          :file-list="fileList"
          :auto-upload="false"
          :on-change="handleChange"
          :limit="1"
          drag
        >
          <i class="el-icon-upload"></i>
          <div class="el-upload__text">Drop file here or <em>click to upload</em></div>
          <div class="el-upload__tip">Only CSV files are supported</div>
        </el-upload>
      </div> -->

      <!-- <div class="model-select-section">
        <label class="label">Select model</label>
        <el-select
          v-model="selectedModel"
          placeholder="Choose a trained model"
          clearable
          style="width: 100%"
        >
          <el-option
            v-for="model in trainedModels"
            :key="model.id"
            :label="model.modelName"
            :value="model.id"
          />
        </el-select>
      </div>

      <el-button
        type="primary"
        :disabled="!canPredict || isLoading"
        :loading="isLoading"
        @click="handlePredict"
        >Predict</el-button
      > -->

      <!-- <div v-if="downloadUrl" class="result-section">
        <p>Prediction result is ready:</p>
        <a :href="downloadUrl" download>Download prediction results</a>
      </div> -->
    </el-card>

    <el-card class="prediction-list">
      <h2>Prediction History</h2>
      <el-table :data="predictions" style="width: 100%">
        <el-table-column prop="train_model_name" label="Model Name" />
        <el-table-column prop="predict_date" label="Training Time" width="180">
          <template #default="{ row }">
            {{ formatDate(row.predict_date) }}
          </template>
        </el-table-column>
        <el-table-column prop="result1_path" label="LightGBM Model Result" width="180">
          <template #default="{ row }">
            <a
              v-if="row.result1_path"
              :href="getDownloadUrl(row.result1_path)"
              target="_blank"
              rel="noopener noreferrer"
              >Download File
            </a>
          </template>
        </el-table-column>
        <el-table-column prop="result2_path" label="Optuna Model Result" width="180">
          <template #default="{ row }">
            <a
              v-if="row.result2_path"
              :href="getDownloadUrl(row.result2_path)"
              target="_blank"
              rel="noopener noreferrer"
              >Download File
            </a>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">{{ row.status }}</template>
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
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

import UploadPredictData from '../components/UploadPredictData.vue'
import type { Prediction } from '../api/models'
import api from '../api/models'

const predictions = ref<Prediction[]>([])
//const selectedModel = ref<string>('')
//const file = ref<File | null>(null)
//type UploadFileLike = { name: string; size: number; status?: string; raw?: File }
//const fileList = ref<UploadFileLike[]>([])
//const downloadUrl = ref<string>('')

//const isLoading = ref(false)

const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

function onUploaded() {
  loadPredictions()
}

const getDownloadUrl = (filePath: string) => {
  if (!filePath) return '#'

  const encodedPath = encodeURIComponent(filePath)

  return `/api/download?file_path=${encodedPath}`
}

async function loadPredictions() {
  loading.value = true
  try {
    const resp = await api.fetchPredictions(currentPage.value, pageSize.value)
    predictions.value = resp.records
    total.value = resp.total
    console.log('Fetched models', predictions.value)
  } catch (err) {
    console.error(err)
    ElMessage({ message: 'Failed to fetch predictions', type: 'error' })
  } finally {
    loading.value = false
  }
}

// function beforeUpload(uploadFile: File) {
//   // store the file locally and prevent automatic upload
//   file.value = uploadFile
//   fileList.value = [
//     {
//       name: uploadFile.name,
//       size: uploadFile.size,
//       status: 'ready',
//       raw: uploadFile,
//     },
//   ]
//   return false
// }

function formatDate(d?: string | null) {
  if (!d) return '-'
  try {
    return new Date(d).toLocaleString()
  } catch {
    return d
  }
}

// function handleChange(hfile: File) {
//   // store the file locally and prevent automatic upload
//   file.value = hfile
//   fileList.value = [
//     {
//       name: hfile.name,
//       size: hfile.size,
//       status: 'ready',
//       raw: hfile,
//     },
//   ]
//   return false
// }

// function handleRemove() {
//   file.value = null
//   fileList.value = []
// }

import { ElMessage } from 'element-plus'
//import type { File } from 'node:buffer'

// async function handlePredict() {
//   if (!file.value || !selectedModel.value) return
//   // call mocked backend endpoint
//   const form = new FormData()
//   form.append('file', file.value)
//   form.append('modelId', selectedModel.value)

//   isLoading.value = true
//   try {
//     const res = await fetch('/api/predict', { method: 'POST', body: form })
//     if (!res.ok) throw new Error(`Server responded ${res.status}`)
//     const blob = await res.blob()
//     downloadUrl.value = URL.createObjectURL(blob)
//     ElNotification({
//       title: 'Success',
//       message: 'Prediction completed. Download is ready.',
//       type: 'success',
//     })
//   } catch (err) {
//     const message = err instanceof Error ? err.message : String(err)
//     ElNotification({ title: 'Error', message: message || 'Prediction failed', type: 'error' })
//   } finally {
//     isLoading.value = false
//   }
// }

function onPageChange(page: number) {
  currentPage.value = page
  loadPredictions()
}

function onPageSizeChange(size: number) {
  pageSize.value = size
  currentPage.value = 1
  loadPredictions()
}

onMounted(() => {
  loadPredictions()
})
</script>

<style scoped>
.page {
  padding: 18px;
}
.predict-view {
  width: 100%;
  max-width: 800px;
  margin: 0;
  padding: 2em;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
.upload-section,
.model-select-section {
  margin-bottom: 1em;
}
.result-section {
  margin-top: 2em;
  background: #f6f8fa;
  padding: 1em;
  border-radius: 6px;
}
a {
  color: #409eff;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
</style>
