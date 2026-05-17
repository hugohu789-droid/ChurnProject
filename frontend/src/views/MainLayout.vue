<script setup lang="ts">
import { useRouter, useRoute } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useAuthStore } from '@/stores/auth'
import { useTheme } from '@/composables/useTheme'

import '../assets/tech-theme.css'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const { user } = storeToRefs(authStore)
const { theme, toggleTheme } = useTheme()

const menuItems = [
  { name: 'Dashboard', path: '/dashboard', icon: '📊' },
  { name: 'Model Training', path: '/training', icon: '🧠' },
  { name: 'Models', path: '/models', icon: '📜' },
  { name: 'Predictions', path: '/predict', icon: '🔮' },
]

const navigate = (path: string) => router.push(path)
const logout = () => authStore.logout()
</script>

<template>
  <div class="layout-container">
    <!-- Left Sidebar -->
    <aside class="sidebar">
      <div class="logo-container">
        <div class="logo-icon">M</div>
        <span class="logo-text">ML<span class="highlight">Platform</span></span>
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
          <button
            class="theme-toggle-btn"
            @click="toggleTheme"
            :title="theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'"
          >
            {{ theme === 'dark' ? '☀️' : '🌙' }}
          </button>
          <div class="avatar">{{ user?.username?.slice(0, 2).toUpperCase() ?? '?' }}</div>
          <span class="username">{{ user?.username }}</span>
          <button class="logout-btn" @click="logout" title="Sign out">⏻</button>
        </div>
      </header>

      <div class="content-wrapper">
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
  transition: background-color var(--transition-speed);
}

/* Sidebar Styles */
.sidebar {
  width: var(--sidebar-width);
  background-color: var(--color-surface);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  z-index: 10;
  transition: background-color var(--transition-speed), border-color var(--transition-speed);
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
  color: #ffffff;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  margin-right: 12px;
  flex-shrink: 0;
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
  background-color: var(--color-surface-hover);
  color: var(--color-text-main);
}

.nav-item.active {
  background: linear-gradient(90deg, var(--color-primary-glow) 0%, transparent 100%);
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
  background-color: var(--bg-color);
  transition: background-color var(--transition-speed);
}

.top-header {
  height: var(--header-height);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  border-bottom: 1px solid var(--color-border);
  background-color: var(--color-surface);
  backdrop-filter: blur(8px);
  z-index: 5;
  transition: background-color var(--transition-speed), border-color var(--transition-speed);
}

.page-title {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--color-text-main);
}

.user-profile {
  display: flex;
  align-items: center;
  gap: 10px;
}

.user-profile .avatar {
  width: 36px;
  height: 36px;
  background-color: var(--color-surface-hover);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
  border: 1px solid var(--color-border);
  color: var(--color-primary);
  flex-shrink: 0;
}

.username {
  font-size: 0.875rem;
  color: var(--color-text-muted);
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.theme-toggle-btn {
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: border-color 0.15s, background-color 0.15s;
  flex-shrink: 0;
  line-height: 1;
}

.theme-toggle-btn:hover {
  border-color: var(--color-primary);
  background-color: var(--color-surface-hover);
}

.logout-btn {
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  color: var(--color-text-muted);
  cursor: pointer;
  font-size: 1rem;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: border-color 0.15s, color 0.15s;
  flex-shrink: 0;
}

.logout-btn:hover {
  border-color: #ef4444;
  color: #ef4444;
}

.content-wrapper {
  flex: 1;
  overflow-y: auto;
  padding: 32px;
  position: relative;
}
</style>
