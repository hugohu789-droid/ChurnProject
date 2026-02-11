<script setup lang="ts">
import { useRouter, useRoute } from 'vue-router'

// Import the tech theme styles
import '../assets/tech-theme.css'

const router = useRouter()
const route = useRoute()

// Menu Configuration (English)
const menuItems = [
  { name: 'Dashboard', path: '/dashboard', icon: '📊' },
  { name: 'Model Training', path: '/training', icon: '🧠' },
  { name: 'Models', path: '/models', icon: '📜' },
  { name: 'Predictions', path: '/predict', icon: '🔮' },
]

const navigate = (path: string) => {
  router.push(path)
}
</script>

<template>
  <div class="layout-container">
    <!-- Left Sidebar -->
    <aside class="sidebar">
      <div class="logo-container">
        <div class="logo-icon">C</div>
        <span class="logo-text">Churn<span class="highlight">AI</span></span>
      </div>
      
      <nav class="nav-menu">
        <div 
          v-for="item in menuItems" 
          :key="item.path"
          class="nav-item"
          :class="{ active: route.path === item.path }"
          @click="navigate(item.path)"
        >
          <span class="icon">{{ item.icon }}</span>
          <span class="label">{{ item.name }}</span>
          <!-- Glowing bar for active state -->
          <div class="glow-bar"></div>
        </div>
      </nav>

      <div class="sidebar-footer">
        <div class="status-dot"></div>
        <span>System Online</span>
      </div>
    </aside>

    <!-- Right Main Content -->
    <main class="main-content">
      <header class="top-header">
        <h2 class="page-title">{{ route.name || 'Dashboard' }}</h2>
        <div class="user-profile">
          <div class="avatar">Admin</div>
        </div>
      </header>
      
      <div class="content-wrapper">
        <!-- Page Content Slot -->
        <slot></slot> 
      </div>
    </main>
  </div>
</template>

<style scoped>
.layout-container {
  display: flex;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background-color: var(--bg-color);
}

/* Sidebar Styles */
.sidebar {
  width: var(--sidebar-width);
  background-color: var(--color-surface);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  z-index: 10;
}

.logo-container {
  height: var(--header-height);
  display: flex;
  align-items: center;
  padding: 0 24px;
  border-bottom: 1px solid var(--color-border);
}

.logo-icon {
  width: 32px;
  height: 32px;
  background: var(--color-primary);
  color: var(--bg-color);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  margin-right: 12px;
}

.logo-text {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--color-text-main);
}

.highlight {
  color: var(--color-primary);
}

.nav-menu {
  flex: 1;
  padding: 24px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.nav-item {
  position: relative;
  display: flex;
  align-items: center;
  padding: 12px 16px;
  border-radius: 8px;
  cursor: pointer;
  color: var(--color-text-muted);
  transition: all 0.3s ease;
  overflow: hidden;
}

.nav-item:hover {
  background-color: rgba(255, 255, 255, 0.03);
  color: var(--color-text-main);
}

.nav-item.active {
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.1) 0%, transparent 100%);
  color: var(--color-primary);
}

.icon {
  margin-right: 12px;
  font-size: 1.1rem;
}

.glow-bar {
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 0%;
  background-color: var(--color-primary);
  border-radius: 0 4px 4px 0;
  transition: height 0.3s ease;
  box-shadow: 0 0 8px var(--color-primary);
}

.nav-item.active .glow-bar {
  height: 60%;
}

.sidebar-footer {
  padding: 24px;
  border-top: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  font-size: 0.875rem;
  color: var(--color-text-muted);
}

.status-dot {
  width: 8px;
  height: 8px;
  background-color: #10b981;
  border-radius: 50%;
  margin-right: 8px;
  box-shadow: 0 0 8px #10b981;
}

/* Main Content Styles */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
}

.top-header {
  height: var(--header-height);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  border-bottom: 1px solid var(--color-border);
  background-color: rgba(15, 23, 42, 0.8); /* Translucent */
  backdrop-filter: blur(8px);
  z-index: 5;
}

.user-profile .avatar {
  width: 36px;
  height: 36px;
  background-color: var(--color-surface-hover);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  border: 1px solid var(--color-border);
}

.content-wrapper {
  flex: 1;
  overflow-y: auto;
  padding: 32px;
  position: relative;
}
</style>