<script setup lang="ts">
import { RouterView } from 'vue-router'
import MainLayout from './views/MainLayout.vue'
</script>

<template>
  <MainLayout>
    <RouterView />
  </MainLayout>
</template>

<style>
@import 'element-plus/dist/index.css';

/* 全局样式覆盖 - 适配 Tech Theme */
:root {
  --theme-primary: #38bdf8; /* 对应左侧菜单的高亮蓝 */
  --theme-primary-fade: rgba(56, 189, 248, 0.08); /* 悬停时的淡蓝光晕 */
}

/* 表格整体美化 */
.el-table {
  --el-table-border-color: #f1f5f9;
  --el-table-header-bg-color: #f8fafc;
  --el-table-row-hover-bg-color: var(--theme-primary-fade);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

/* 表头样式 - 简洁、大写、弱化视觉噪音 */
.el-table th.el-table__cell {
  background-color: var(--el-table-header-bg-color) !important;
  color: #64748b;
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  height: 48px;
}

/* 单元格内边距调整，增加呼吸感 */
.el-table .el-table__cell {
  padding: 12px 0;
}

/* 按钮主色调统一 - 确保表格内的按钮也符合主题 */
.el-button--primary {
  --el-button-bg-color: var(--theme-primary);
  --el-button-border-color: var(--theme-primary);
}

/* 分页选中态颜色 */
.el-pagination.is-background .el-pager li:not(.is-disabled).is-active {
  background-color: var(--theme-primary);
}
</style>
