import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
      redirect: '/model-training',
      children: [
        {
          path: '/model-training',
          name: 'model-training',
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
