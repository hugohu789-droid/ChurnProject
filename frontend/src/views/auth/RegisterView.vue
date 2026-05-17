<template>
  <div class="auth-page">
    <div class="auth-card">
      <div class="auth-brand">
        <span class="brand-icon">⚡</span>
        <h1 class="brand-name">ML Platform</h1>
        <p class="brand-tagline">Create your account</p>
      </div>

      <form class="auth-form" @submit.prevent="handleSubmit">
        <div class="form-group">
          <label for="email">Email</label>
          <input
            id="email"
            v-model="form.email"
            type="email"
            placeholder="you@example.com"
            required
            autocomplete="email"
          />
        </div>

        <div class="form-group">
          <label for="username">Username</label>
          <input
            id="username"
            v-model="form.username"
            type="text"
            placeholder="mlhero"
            required
            minlength="3"
            autocomplete="username"
          />
        </div>

        <div class="form-group">
          <label for="password">Password</label>
          <input
            id="password"
            v-model="form.password"
            type="password"
            placeholder="Min. 8 characters"
            required
            minlength="8"
            autocomplete="new-password"
          />
        </div>

        <!-- CAPTCHA -->
        <div class="form-group">
          <label>Verification Code</label>
          <CaptchaWidget @change="onCaptchaChange" />
          <input
            v-model="captchaInput"
            type="text"
            placeholder="Enter the code above"
            autocomplete="off"
            maxlength="5"
            class="captcha-input"
            :class="{ 'input-error': captchaError }"
            @input="captchaError = ''"
          />
          <span v-if="captchaError" class="field-error">{{ captchaError }}</span>
        </div>

        <p v-if="error" class="error-msg">{{ error }}</p>

        <button type="submit" class="btn-primary" :disabled="isLoading">
          <span v-if="isLoading" class="spinner" />
          <span v-else>Create account</span>
        </button>
      </form>

      <p class="auth-footer">
        Already have an account?
        <RouterLink to="/login">Sign in</RouterLink>
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { useAuth } from '@/composables/useAuth'
import CaptchaWidget from '@/components/CaptchaWidget.vue'

const { register, isLoading, error } = useAuth()

const form = reactive({ email: '', username: '', password: '' })
const captchaInput = ref('')
const captchaCode = ref('')
const captchaError = ref('')

function onCaptchaChange(code: string) {
  captchaCode.value = code
  captchaInput.value = ''
  captchaError.value = ''
}

async function handleSubmit() {
  if (captchaInput.value.toUpperCase() !== captchaCode.value) {
    captchaError.value = 'Incorrect verification code, please try again.'
    captchaInput.value = ''
    return
  }
  await register(form)
}
</script>

<style scoped>
.auth-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg);
  padding: 1.5rem;
}
.auth-card {
  width: 100%;
  max-width: 400px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 1rem;
  padding: 2.5rem 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}
.auth-brand { text-align: center; margin-bottom: 2rem; }
.brand-icon { font-size: 2rem; }
.brand-name { font-size: 1.5rem; font-weight: 700; color: var(--color-text-primary); margin: 0.25rem 0; letter-spacing: -0.02em; }
.brand-tagline { font-size: 0.875rem; color: var(--color-text-muted); margin: 0; }
.auth-form { display: flex; flex-direction: column; gap: 1.25rem; }
.form-group { display: flex; flex-direction: column; gap: 0.375rem; }
.form-group label { font-size: 0.8125rem; font-weight: 500; color: var(--color-text-secondary); }
.form-group input { padding: 0.625rem 0.875rem; background: var(--color-input-bg); border: 1px solid var(--color-border); border-radius: 0.5rem; color: var(--color-text-primary); font-size: 0.9375rem; transition: border-color 0.15s, box-shadow 0.15s; outline: none; }
.form-group input:focus { border-color: var(--color-accent); box-shadow: 0 0 0 3px var(--color-accent-glow); }
.form-group input::placeholder { color: var(--color-text-muted); }
.captcha-input { letter-spacing: 0.15em; font-family: 'Courier New', monospace; text-transform: uppercase; }
.input-error { border-color: #ef4444 !important; }
.field-error { font-size: 0.75rem; color: #ef4444; margin-top: 2px; }
.error-msg { font-size: 0.8125rem; color: var(--color-danger); margin: 0; padding: 0.5rem 0.75rem; background: rgba(239, 68, 68, 0.1); border-radius: 0.375rem; border: 1px solid rgba(239, 68, 68, 0.2); }
.btn-primary { padding: 0.75rem; background: var(--color-accent); color: #fff; border: none; border-radius: 0.5rem; font-size: 0.9375rem; font-weight: 600; cursor: pointer; transition: opacity 0.15s, transform 0.1s; display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
.btn-primary:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
.btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
.spinner { width: 1rem; height: 1rem; border: 2px solid rgba(255, 255, 255, 0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.auth-footer { text-align: center; margin-top: 1.5rem; font-size: 0.875rem; color: var(--color-text-muted); }
.auth-footer a { color: var(--color-accent); font-weight: 500; text-decoration: none; }
.auth-footer a:hover { text-decoration: underline; }
</style>
