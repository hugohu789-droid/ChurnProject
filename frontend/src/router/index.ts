import { createRouter, createWebHashHistory } from 'vue-router'

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      redirect: '/training',
      children: [
        {
          path: '/dashboard',
          name: 'dashboard',
          component: () => import('../views/HomeView.vue')
        },
        {
          path: '/training',
          name: 'training',
          component: () => import('../views/ModelTrainingView.vue')
        },
        {
          path: '/models',
          name: 'models',
          component: () => import('../views/ModelsView.vue')
        },
        {
          path: '/predict',
          name: 'predict',
          component: () => import('../views/PredictView.vue')
        }
      ]
    }
  ],
})

export default router
