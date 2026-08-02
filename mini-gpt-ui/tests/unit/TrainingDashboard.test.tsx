import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import TrainingDashboard from '../../src/components/TrainingDashboard'

const RUN_ID = '6c4b1e2a-0000-4000-8000-000000000001'
const OTHER_RUN_ID = '6c4b1e2a-0000-4000-8000-000000000002'
const POLL_INTERVAL_MS = 1000

type StatusOverrides = Partial<{
  run_id: string
  status: string
  steps_completed: number
  total_steps: number
  training_loss: number | null
  value_loss: number | null
  steps_per_second: number | null
  eta_seconds: number | null
  ended_at_unix: number | null
  checkpoints: string[]
  error: string
}>

function runStatus(overrides: StatusOverrides = {}) {
  return {
    run_id: RUN_ID,
    status: 'running',
    request: {
      train_steps: 250,
      learning_rate: 0.0001,
      checkpoint_interval: 100,
      eval_interval: 100,
    },
    started_at_unix: 1_750_000_000,
    ended_at_unix: null,
    steps_completed: 100,
    total_steps: 250,
    training_loss: 2.5,
    value_loss: 2.7,
    steps_per_second: 5,
    checkpoints: [],
    eta_seconds: 180,
    ...overrides,
  }
}

function jsonResponse(payload: unknown) {
  return { json: vi.fn().mockResolvedValue(payload) }
}

function errorResponse(status: number, statusText: string, body: unknown) {
  return {
    ok: false,
    status,
    statusText,
    text: vi.fn().mockResolvedValue(typeof body === 'string' ? body : JSON.stringify(body)),
  }
}

function stubFetch(handler: (url: string, init?: RequestInit) => unknown) {
  const fetchMock = vi
    .fn()
    .mockImplementation((url: string, init?: RequestInit) => Promise.resolve(handler(url, init)))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** Drives the polling loop: advances fake timers and flushes the fetch promises. */
async function flush(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

function startRun() {
  fireEvent.change(screen.getByLabelText(/train steps/i), { target: { value: '250' } })
  fireEvent.click(screen.getByRole('button', { name: /start training/i }))
}

describe('TrainingDashboard', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('posts the form, polls status, and stops polling once the run completes', async () => {
    const statuses = [
      runStatus({ steps_completed: 100, training_loss: 2.5 }),
      runStatus({ steps_completed: 200, training_loss: 2.1 }),
      runStatus({
        status: 'completed',
        steps_completed: 250,
        training_loss: 1.8,
        eta_seconds: null,
        ended_at_unix: 1_750_000_600,
        checkpoints: ['mini_gpt.step-250.mpk'],
      }),
    ]
    let polls = 0
    const fetchMock = stubFetch((url, init) => {
      if (url === '/api/train' && init?.method === 'POST') {
        return jsonResponse({ run_id: RUN_ID })
      }

      if (url === `/api/train/${RUN_ID}/status`) {
        return jsonResponse(statuses[Math.min(polls++, statuses.length - 1)])
      }

      throw new Error(`unexpected request: ${url}`)
    })

    render(<TrainingDashboard />)

    fireEvent.change(screen.getByLabelText(/resume from/i), {
      target: { value: 'mini_gpt.step-5000' },
    })
    startRun()
    await flush()

    expect(fetchMock).toHaveBeenCalledWith('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        train_steps: 250,
        learning_rate: 0.0001,
        checkpoint_interval: 100,
        eval_interval: 100,
        resume_from: 'mini_gpt.step-5000',
      }),
    })
    expect(screen.getByText(/run running — step 100 \/ 250/i)).toBeInTheDocument()
    expect(screen.getByText('2.5000')).toBeInTheDocument()
    expect(screen.getByText('3m 0s')).toBeInTheDocument()
    // One sample is not a line yet.
    expect(screen.queryByRole('img', { name: /training loss curve/i })).not.toBeInTheDocument()

    await flush(POLL_INTERVAL_MS)

    expect(screen.getByText(/step 200 \/ 250/i)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /training loss curve over 2 samples/i })).toBeInTheDocument()

    await flush(POLL_INTERVAL_MS)

    expect(screen.getByText(/run completed — step 250 \/ 250/i)).toBeInTheDocument()
    expect(screen.getByText('mini_gpt.step-250.mpk')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /stop/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /start training/i })).toBeEnabled()

    const callsWhenFinished = fetchMock.mock.calls.length
    await flush(POLL_INTERVAL_MS * 5)

    expect(fetchMock.mock.calls.length).toBe(callsWhenFinished)
  })

  it('stops an active run and follows it to interrupted', async () => {
    let stopRequested = false
    const fetchMock = stubFetch((url, init) => {
      if (url === '/api/train' && init?.method === 'POST') {
        return jsonResponse({ run_id: RUN_ID })
      }

      if (url === `/api/train/${RUN_ID}` && init?.method === 'DELETE') {
        stopRequested = true
        return { status: 202 }
      }

      if (url === `/api/train/${RUN_ID}/status`) {
        return jsonResponse(
          stopRequested
            ? runStatus({
                status: 'interrupted',
                steps_completed: 120,
                eta_seconds: null,
                ended_at_unix: 1_750_000_400,
                checkpoints: ['mini_gpt.interrupted-step-120.mpk'],
              })
            : runStatus(),
        )
      }

      throw new Error(`unexpected request: ${url}`)
    })

    render(<TrainingDashboard />)
    startRun()
    await flush()

    fireEvent.click(screen.getByRole('button', { name: /stop/i }))
    await flush()

    expect(fetchMock).toHaveBeenCalledWith(`/api/train/${RUN_ID}`, { method: 'DELETE' })

    await flush(POLL_INTERVAL_MS)

    expect(screen.getByText(/run interrupted — step 120 \/ 250/i)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('treats a 404 from stop as "already finished" and refreshes instead of erroring', async () => {
    // The run finishes server-side between the last poll and the Stop click.
    let stopAttempted = false
    const fetchMock = stubFetch((url, init) => {
      if (url === '/api/train' && init?.method === 'POST') {
        return jsonResponse({ run_id: RUN_ID })
      }

      if (url === `/api/train/${RUN_ID}` && init?.method === 'DELETE') {
        stopAttempted = true
        return errorResponse(404, 'Not Found', '')
      }

      if (url === `/api/train/${RUN_ID}/status`) {
        return jsonResponse(
          stopAttempted
            ? runStatus({ status: 'completed', steps_completed: 250, eta_seconds: null })
            : runStatus(),
        )
      }

      throw new Error(`unexpected request: ${url}`)
    })

    render(<TrainingDashboard />)
    startRun()
    await flush()

    fireEvent.click(screen.getByRole('button', { name: /stop/i }))
    await flush()

    expect(fetchMock).toHaveBeenCalledWith(`/api/train/${RUN_ID}`, { method: 'DELETE' })
    expect(screen.getByText(/run completed — step 250 \/ 250/i)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('follows the active run when the server reports one is already in progress', async () => {
    const fetchMock = stubFetch((url, init) => {
      if (url === '/api/train' && init?.method === 'POST') {
        return errorResponse(409, 'Conflict', { error: 'run_in_progress', run_id: OTHER_RUN_ID })
      }

      if (url === `/api/train/${OTHER_RUN_ID}/status`) {
        return jsonResponse(runStatus({ run_id: OTHER_RUN_ID, steps_completed: 42 }))
      }

      throw new Error(`unexpected request: ${url}`)
    })

    render(<TrainingDashboard />)
    startRun()
    await flush()

    // A notice, not an error: the panel switched to the run that is actually going.
    expect(screen.getByRole('status')).toHaveTextContent(/already in progress/i)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(`/api/train/${OTHER_RUN_ID}/status`)
    expect(screen.getByText(/run running — step 42 \/ 250/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /stop/i })).toBeEnabled()
  })

  it('disables the form when the server has no training runner', async () => {
    stubFetch((url, init) => {
      if (url === '/api/train' && init?.method === 'POST') {
        return errorResponse(503, 'Service Unavailable', {
          error: 'training_unavailable',
          message: 'server was started without a training runner',
        })
      }

      throw new Error(`unexpected request: ${url}`)
    })

    render(<TrainingDashboard />)
    startRun()
    await flush()

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Training is unavailable on this server: server was started without a training runner',
    )
    expect(screen.getByRole('button', { name: /start training/i })).toBeDisabled()
    expect(screen.queryByLabelText(/train steps/i)).not.toBeInTheDocument()
  })

  it('surfaces range validation errors from the training API', async () => {
    stubFetch((url, init) => {
      if (url === '/api/train' && init?.method === 'POST') {
        return errorResponse(400, 'Bad Request', {
          error: 'train_steps_out_of_range',
          max_allowed: 100_000,
          requested: 250_000,
        })
      }

      throw new Error(`unexpected request: ${url}`)
    })

    render(<TrainingDashboard />)
    fireEvent.change(screen.getByLabelText(/train steps/i), { target: { value: '250000' } })
    fireEvent.click(screen.getByRole('button', { name: /start training/i }))
    await flush()

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Train steps must be at most 100000 (requested 250000).',
    )
    expect(screen.getByRole('button', { name: /start training/i })).toBeEnabled()
  })
})
