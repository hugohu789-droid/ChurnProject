import './assets/main.css'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'

// Apply stored theme before mount to avoid flash of wrong theme
;(() => {
  const saved = localStorage.getItem('theme')
  const theme = saved === 'light' || saved === 'dark' ? saved : 'dark'
  document.documentElement.classList.add(theme)
})()

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'

import App from './App.vue'
import router from './router'


const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(ElementPlus)

app.mount('#app')
