<script setup lang="ts">
import { ref, onMounted } from 'vue'

const emit = defineEmits<{
  (e: 'change', code: string): void
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const currentCode = ref('')

const CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
const CODE_LEN = 5

function generateCode(): string {
  return Array.from({ length: CODE_LEN }, () =>
    CHARS[Math.floor(Math.random() * CHARS.length)],
  ).join('')
}

function rand(min: number, max: number) {
  return Math.random() * (max - min) + min
}

function draw(code: string) {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')!
  const W = canvas.width
  const H = canvas.height

  // Background
  ctx.fillStyle = '#1e293b'
  ctx.fillRect(0, 0, W, H)

  // Noise lines
  for (let i = 0; i < 5; i++) {
    ctx.beginPath()
    ctx.moveTo(rand(0, W), rand(0, H))
    ctx.lineTo(rand(0, W), rand(0, H))
    ctx.strokeStyle = `hsla(${rand(180, 220)},60%,${rand(40, 70)}%,0.5)`
    ctx.lineWidth = rand(1, 2)
    ctx.stroke()
  }

  // Noise dots
  for (let i = 0; i < 40; i++) {
    ctx.beginPath()
    ctx.arc(rand(0, W), rand(0, H), rand(0.5, 1.5), 0, Math.PI * 2)
    ctx.fillStyle = `hsla(${rand(180, 220)},50%,${rand(50, 80)}%,0.6)`
    ctx.fill()
  }

  // Characters
  ctx.textBaseline = 'middle'
  const charW = W / (CODE_LEN + 1)
  for (let i = 0; i < code.length; i++) {
    const x = charW * (i + 0.9) + rand(-4, 4)
    const y = H / 2 + rand(-4, 4)
    const angle = rand(-0.25, 0.25)
    const size = Math.floor(rand(20, 26))

    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle)
    ctx.font = `bold ${size}px 'Courier New', monospace`
    ctx.fillStyle = `hsl(${rand(185, 210)},80%,${rand(65, 85)}%)`
    ctx.fillText(code.charAt(i), 0, 0)
    ctx.restore()
  }
}

function refresh() {
  const code = generateCode()
  currentCode.value = code
  draw(code)
  emit('change', code)
}

onMounted(() => refresh())
</script>

<template>
  <div class="captcha-widget">
    <canvas ref="canvasRef" width="160" height="48" class="captcha-canvas" />
    <button type="button" class="captcha-refresh" title="Refresh CAPTCHA" @click="refresh">
      ↻
    </button>
  </div>
</template>

<style scoped>
.captcha-widget {
  display: flex;
  align-items: center;
  gap: 8px;
}

.captcha-canvas {
  border-radius: 6px;
  border: 1px solid var(--color-border);
  cursor: pointer;
  flex-shrink: 0;
}

.captcha-refresh {
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  color: var(--color-text-muted);
  cursor: pointer;
  font-size: 1.1rem;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: border-color 0.15s, color 0.15s;
  flex-shrink: 0;
}

.captcha-refresh:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
}
</style>
