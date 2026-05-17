<script setup lang="ts">
import { computed } from 'vue'
import { RouterView, useRoute } from 'vue-router'
import MainLayout from './views/MainLayout.vue'

const route = useRoute()

// Auth pages (login, register) render without the sidebar
const useAppLayout = computed(() => !!route.meta.requiresAuth)
</script>

<template>
  <MainLayout v-if="useAppLayout">
    <RouterView />
  </MainLayout>
  <RouterView v-else />
</template>

<style>
@import 'element-plus/dist/index.css';

.el-table {
  --el-table-border-color: #f1f5f9;
  --el-table-header-bg-color: #f8fafc;
  --el-table-row-hover-bg-color: rgba(56, 189, 248, 0.08);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.el-table th.el-table__cell {
  background-color: var(--el-table-header-bg-color) !important;
  color: #64748b;
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  height: 48px;
}

.el-table .el-table__cell {
  padding: 12px 0;
}

.el-button--primary {
  --el-button-bg-color: #38bdf8;
  --el-button-border-color: #38bdf8;
}

.el-pagination.is-background .el-pager li:not(.is-disabled).is-active {
  background-color: #38bdf8;
}
</style>
