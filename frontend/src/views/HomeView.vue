<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute, RouterView } from 'vue-router'

const route = useRoute()

const menuItems = [
  {
    index: 'model-training',
    label: 'Model Training',
    path: '/model-training',
    icon: 'el-icon-s-data',
  },
  {
    index: 'models',
    label: 'Models',
    path: '/models',
    icon: 'el-icon-s-grid',
  },
  { index: 'predict', label: 'Prediction', path: '/predict', icon: 'el-icon-s-operation' },
]

const isCollapsed = ref(false)

function updateCollapse() {
  isCollapsed.value = window.innerWidth < 880
}

function toggleCollapse() {
  isCollapsed.value = !isCollapsed.value
}

onMounted(() => {
  updateCollapse()
  window.addEventListener('resize', updateCollapse)
})

onUnmounted(() => {
  window.removeEventListener('resize', updateCollapse)
})
</script>

<template>
  <el-container class="home-layout">
    <el-aside :width="isCollapsed ? '64px' : '220px'" class="sidebar">
      <div class="brand" :class="{ collapsed: isCollapsed }">
        <div class="logo">Telco AI</div>
        <div class="subtitle">Churn Prediction Dashboard</div>
      </div>

      <el-menu
        :default-active="route.name as string"
        :collapse="isCollapsed"
        class="menu"
        router
        background-color="transparent"
        text-color="#9fb8ff"
        active-text-color="#fff"
      >
        <el-menu-item
          v-for="item in menuItems"
          :key="item.index"
          :index="item.path"
          :route="item.path"
        >
          <i :class="item.icon" style="margin-right: 8px"></i>
          <span>{{ item.label }}</span>
        </el-menu-item>
      </el-menu>

      <div class="sidebar-footer">Powered by Element Plus</div>
    </el-aside>

    <el-container class="main-container" width="100%">
      <el-header class="header">
        <div class="header-left">
          <el-button type="text" @click="toggleCollapse" class="collapse-btn">
            <i class="el-icon-s-fold"></i>
          </el-button>
          <div class="header-title">Telecommunications · Predictive Modeling</div>
        </div>
      </el-header>

      <el-main class="main-content">
        <RouterView />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.home-layout {
  min-height: 100vh;
  min-width: 1068px;
  width: 100%;
  background: radial-gradient(ellipse at center, #071029 0%, #021122 60%);
  color: #cfe6ff;
  display: flex;
}
.sidebar {
  padding: 1.5rem 1rem;
  border-right: 1px solid rgba(255, 255, 255, 0.04);
  background: linear-gradient(180deg, rgba(10, 30, 60, 0.6), rgba(5, 15, 30, 0.4));
  flex-shrink: 0;
  transition: width 0.3s;
}
.main-container {
  flex: 1;
  min-width: 0;
  overflow-x: auto;
}
.brand {
  transition: opacity 0.3s;
}
.brand.collapsed .subtitle {
  opacity: 0;
}
.brand .logo {
  font-weight: 700;
  font-size: 1.25rem;
  color: #7fd1ff;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.brand .subtitle {
  font-size: 0.8rem;
  color: #8fbfff;
  margin-bottom: 1rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: opacity 0.3s;
}
.menu {
  border: none;
  background: transparent;
}
.sidebar-footer {
  margin-top: 2rem;
  font-size: 0.75rem;
  color: #7ea8d6;
}
.header {
  background: transparent;
  color: #e6f4ff;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}
.header-title {
  font-weight: 600;
  letter-spacing: 0.6px;
}
.main-content {
  padding: 1.5rem;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.collapse-btn {
  color: #9fb8ff;
}
</style>
