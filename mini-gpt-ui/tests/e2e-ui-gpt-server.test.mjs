import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const UI_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const ROOT_DIR = resolve(UI_DIR, '..')

function testPort(offset) {
  return 30_000 + ((process.pid + offset) % 20_000)
}

function startProcess(command, args, options) {
  const child = spawn(command, args, {
    ...options,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let output = ''
  child.stdout.on('data', (chunk) => {
    output += chunk
  })
  child.stderr.on('data', (chunk) => {
    output += chunk
  })
  return { child, output: () => output }
}

async function stopProcess(childProcess) {
  if (childProcess.exitCode !== null) {
    return
  }
  try {
    globalThis.process.kill(-childProcess.pid, 'SIGTERM')
  } catch (error) {
    if (error.code !== 'ESRCH') {
      throw error
    }
    return
  }
  await new Promise((resolveStop) => {
    const timeout = setTimeout(() => {
      try {
        globalThis.process.kill(-childProcess.pid, 'SIGKILL')
      } catch (error) {
        if (error.code !== 'ESRCH') {
          throw error
        }
      }
      resolveStop()
    }, 2_000)
    childProcess.once('exit', () => {
      clearTimeout(timeout)
      resolveStop()
    })
  })
}

async function waitForFetch(url, options = {}) {
  const deadline = Date.now() + (options.timeoutMs ?? 60_000)
  let lastError

  while (Date.now() < deadline) {
    try {
      const response = await fetch(url)
      if (response.ok) {
        return response
      }
      lastError = new Error(`${url} returned ${response.status}`)
    } catch (error) {
      lastError = error
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 250))
  }

  throw lastError ?? new Error(`timed out waiting for ${url}`)
}

test(
  'UI dev server proxies generate requests to the GPT server',
  { timeout: 180_000 },
  async () => {
    const apiPort = testPort(0)
    const uiPort = testPort(1)
    const apiBaseUrl = `http://127.0.0.1:${apiPort}`
    const uiBaseUrl = `http://127.0.0.1:${uiPort}`

    const api = startProcess(
      'cargo',
      [
        'run',
        '--quiet',
        '--',
        '--serve',
        '--input',
        'tests/fixtures/input.txt',
        '--server-addr',
        `127.0.0.1:${apiPort}`,
      ],
      { cwd: ROOT_DIR },
    )
    const ui = startProcess(
      'npm',
      ['run', 'dev', '--', '--host', '127.0.0.1', '--port', String(uiPort)],
      {
        cwd: UI_DIR,
        env: {
          ...process.env,
          VITE_API_PROXY_TARGET: apiBaseUrl,
        },
      },
    )

    try {
      await waitForFetch(`${apiBaseUrl}/api/info`, { timeoutMs: 120_000 })
      const uiRoot = await waitForFetch(uiBaseUrl)
      assert.match(await uiRoot.text(), /<div id="root"><\/div>/)

      const response = await fetch(`${uiBaseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: 'hello',
          max_tokens: 1,
          temperature: 1,
        }),
      })
      const data = await response.json()

      assert.equal(response.status, 200)
      assert.equal(typeof data.generated, 'string')
      assert.ok(data.generated.startsWith('hello'))
      assert.ok(Array.isArray(data.tokens))
      assert.ok(data.tokens.length >= 6)
      assert.ok(Array.isArray(data.attention))
      assert.ok(data.attention.length > 0)
    } catch (error) {
      error.message = `${error.message}\n\nAPI output:\n${api.output()}\n\nUI output:\n${ui.output()}`
      throw error
    } finally {
      await Promise.all([stopProcess(ui.child), stopProcess(api.child)])
    }
  },
)
