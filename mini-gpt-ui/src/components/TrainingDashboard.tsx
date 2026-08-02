import { useCallback, useEffect, useState } from 'react'
import { ApiError, getTrainStatus, startTraining, stopTraining, type TrainStatus } from '../api'

/** The status route is exempt from the generate rate limiter, so a 1s poll is cheap. */
const POLL_INTERVAL_MS = 1000

type LossSample = {
  step: number
  loss: number
}

function formatLoss(loss: number | null) {
  return loss === null ? '—' : loss.toFixed(4)
}

function formatEta(seconds: number | null) {
  if (seconds === null) {
    return '—'
  }

  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainder = seconds % 60

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }

  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`
}

function trainErrorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) {
    return error instanceof Error ? error.message : 'Training request failed.'
  }

  const body = error.body
  switch (body?.error) {
    case 'train_steps_out_of_range':
      return `Train steps must be at most ${body.max_allowed} (requested ${body.requested}).`
    case 'learning_rate_out_of_range':
      return `Learning rate must be at most ${body.max_allowed} (requested ${body.requested}).`
    case 'run_not_found':
      return 'That training run is no longer available on the server.'
    default:
      return body?.message ?? error.message
  }
}

const CHART_WIDTH = 420
const CHART_HEIGHT = 140
/** Keeps the stroke off the viewBox edge, where half of it would be clipped. */
const CHART_INSET = 3

// ponytail: an inline polyline instead of a charting dependency. Swap in a real
// chart library only if this needs axes, zoom, or multiple series.
function LossCurve({ samples }: { samples: LossSample[] }) {
  const latest = samples.at(-1)

  if (samples.length < 2 || latest === undefined) {
    return <p className="loss-chart-empty">Waiting for training loss samples...</p>
  }

  const steps = samples.map((sample) => sample.step)
  const losses = samples.map((sample) => sample.loss)
  const minStep = Math.min(...steps)
  const minLoss = Math.min(...losses)
  // A flat curve has a zero span; fall back to 1 so every point lands mid-chart.
  const stepSpan = Math.max(...steps) - minStep || 1
  const lossSpan = Math.max(...losses) - minLoss || 1
  const plotHeight = CHART_HEIGHT - CHART_INSET * 2

  const points = samples
    .map((sample) => {
      const x = ((sample.step - minStep) / stepSpan) * CHART_WIDTH
      const y = CHART_HEIGHT - CHART_INSET - ((sample.loss - minLoss) / lossSpan) * plotHeight
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  return (
    <svg
      className="loss-chart"
      viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`Training loss curve over ${samples.length} samples, latest ${latest.loss.toFixed(
        4,
      )} at step ${latest.step}`}
    >
      <polyline
        points={points}
        fill="none"
        stroke="var(--accent)"
        strokeWidth={2}
        strokeLinejoin="round"
      />
    </svg>
  )
}

export default function TrainingDashboard() {
  const [trainSteps, setTrainSteps] = useState('1000')
  const [learningRate, setLearningRate] = useState('0.0001')
  const [checkpointInterval, setCheckpointInterval] = useState('100')
  const [evalInterval, setEvalInterval] = useState('100')
  const [resumeFrom, setResumeFrom] = useState('')

  const [runId, setRunId] = useState<string | null>(null)
  const [status, setStatus] = useState<TrainStatus | null>(null)
  const [samples, setSamples] = useState<LossSample[]>([])
  const [isStartingRun, setIsStartingRun] = useState(false)
  const [isStoppingRun, setIsStoppingRun] = useState(false)
  const [trainingError, setTrainingError] = useState<string | null>(null)
  const [trainingUnavailable, setTrainingUnavailable] = useState<string | null>(null)
  // Kept apart from trainingError: a successful poll clears errors, and this
  // note about *which* run is on screen has to survive that.
  const [runNotice, setRunNotice] = useState<string | null>(null)

  // A freshly started run has no status yet, so treat "unknown" as running.
  const isRunActive = runId !== null && (status === null || status.status === 'running')

  const applyStatus = useCallback((next: TrainStatus) => {
    setStatus(next)
    // The server only reports the latest point; the history is ours to keep.
    setSamples((currentSamples) =>
      next.training_loss === null || (currentSamples.at(-1)?.step ?? -1) >= next.steps_completed
        ? currentSamples
        : [...currentSamples, { step: next.steps_completed, loss: next.training_loss }],
    )
  }, [])

  useEffect(() => {
    if (runId === null || !isRunActive) {
      return
    }

    let isCurrent = true

    const poll = async () => {
      try {
        const next = await getTrainStatus(runId)
        if (isCurrent) {
          applyStatus(next)
          setTrainingError(null)
        }
      } catch (error) {
        if (!isCurrent) {
          return
        }

        setTrainingError(trainErrorMessage(error))
        if (error instanceof ApiError && error.status === 404) {
          // Terminal: there is nothing left to poll for this id.
          setRunId(null)
        }
      }
    }

    void poll()
    const timer = setInterval(() => void poll(), POLL_INTERVAL_MS)

    return () => {
      isCurrent = false
      clearInterval(timer)
    }
  }, [runId, isRunActive, applyStatus])

  const trackRun = (nextRunId: string) => {
    setRunId(nextRunId)
    setStatus(null)
    setSamples([])
    setRunNotice(null)
  }

  const handleStart = async () => {
    setTrainingError(null)
    setIsStartingRun(true)

    try {
      const accepted = await startTraining({
        train_steps: Number(trainSteps),
        learning_rate: Number(learningRate),
        checkpoint_interval: Number(checkpointInterval),
        eval_interval: Number(evalInterval),
        ...(resumeFrom.trim() === '' ? {} : { resume_from: resumeFrom.trim() }),
      })
      trackRun(accepted.run_id)
    } catch (error) {
      const body = error instanceof ApiError ? error.body : null

      if (body?.error === 'run_in_progress' && body.run_id !== undefined) {
        // Someone else already started one — follow that run instead of erroring.
        trackRun(body.run_id)
        setRunNotice('A training run was already in progress; showing that run.')
      } else if (body?.error === 'training_unavailable') {
        setTrainingUnavailable(
          body.message ?? 'This server was started without a training runner.',
        )
      } else {
        setTrainingError(trainErrorMessage(error))
      }
    } finally {
      setIsStartingRun(false)
    }
  }

  const handleStop = async () => {
    if (runId === null) {
      return
    }

    setTrainingError(null)
    setIsStoppingRun(true)

    try {
      await stopTraining(runId)
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        // The run finished between the last poll and this click — not an error.
        try {
          applyStatus(await getTrainStatus(runId))
        } catch (refreshError) {
          setTrainingError(trainErrorMessage(refreshError))
        }
      } else {
        setTrainingError(trainErrorMessage(error))
      }
    } finally {
      setIsStoppingRun(false)
    }
  }

  return (
    <section className="training-dashboard" aria-labelledby="training-heading">
      <div className="training-header">
        <div>
          <h2 id="training-heading">Training Dashboard</h2>
          <p>Start a MiniGPT run on the server corpus and watch it progress.</p>
        </div>
        <button
          type="button"
          onClick={handleStart}
          disabled={isStartingRun || isRunActive || trainingUnavailable !== null}
        >
          {isStartingRun ? 'Starting...' : 'Start training'}
        </button>
      </div>

      {trainingUnavailable === null ? (
        <div className="training-form">
          <label htmlFor="train-steps">
            Train steps
            <input
              id="train-steps"
              min="1"
              type="number"
              value={trainSteps}
              onChange={(event) => setTrainSteps(event.target.value)}
              disabled={isRunActive}
            />
          </label>
          <label htmlFor="learning-rate">
            Learning rate
            <input
              id="learning-rate"
              min="0"
              step="0.0001"
              type="number"
              value={learningRate}
              onChange={(event) => setLearningRate(event.target.value)}
              disabled={isRunActive}
            />
          </label>
          <label htmlFor="checkpoint-interval">
            Checkpoint interval
            <input
              id="checkpoint-interval"
              min="1"
              type="number"
              value={checkpointInterval}
              onChange={(event) => setCheckpointInterval(event.target.value)}
              disabled={isRunActive}
            />
          </label>
          <label htmlFor="eval-interval">
            Eval interval
            <input
              id="eval-interval"
              min="1"
              type="number"
              value={evalInterval}
              onChange={(event) => setEvalInterval(event.target.value)}
              disabled={isRunActive}
            />
          </label>
          <label htmlFor="resume-from">
            Resume from
            <input
              id="resume-from"
              type="text"
              placeholder="Fresh model"
              value={resumeFrom}
              onChange={(event) => setResumeFrom(event.target.value)}
              disabled={isRunActive}
            />
          </label>
        </div>
      ) : (
        <p className="form-error" role="alert">
          Training is unavailable on this server: {trainingUnavailable}
        </p>
      )}

      {trainingError !== null && (
        <p className="form-error" role="alert">
          {trainingError}
        </p>
      )}

      {runNotice !== null && (
        <p className="training-notice" role="status">
          {runNotice}
        </p>
      )}

      {runId !== null && (
        <div className="training-run">
          <div className="training-summary" aria-live="polite">
            <span>
              {status === null
                ? 'Run accepted; waiting for the first status update.'
                : `Run ${status.status} — step ${status.steps_completed} / ${status.total_steps}`}
            </span>
            <button type="button" onClick={handleStop} disabled={!isRunActive || isStoppingRun}>
              {isStoppingRun ? 'Stopping...' : 'Stop'}
            </button>
          </div>

          {status !== null && (
            <dl className="training-metrics">
              <div>
                <dt>Training loss</dt>
                <dd>{formatLoss(status.training_loss)}</dd>
              </div>
              <div>
                <dt>Value loss</dt>
                <dd>{formatLoss(status.value_loss)}</dd>
              </div>
              <div>
                <dt>Steps / second</dt>
                <dd>
                  {status.steps_per_second === null ? '—' : status.steps_per_second.toFixed(2)}
                </dd>
              </div>
              <div>
                <dt>ETA</dt>
                <dd>{formatEta(status.eta_seconds)}</dd>
              </div>
              <div>
                <dt>Latest checkpoint</dt>
                <dd>{status.checkpoints.at(-1) ?? '—'}</dd>
              </div>
            </dl>
          )}

          <LossCurve samples={samples} />

          {status?.error !== undefined && (
            <p className="form-error" role="alert">
              Training failed: {status.error}
            </p>
          )}
        </div>
      )}
    </section>
  )
}
