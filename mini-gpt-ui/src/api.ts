export type GenerateResponse = {
  generated: string
  tokens: string[]
  attention: Array<{
    layer: number
    head: number
    weights: number[][]
  }>
  routing?: Array<{
    layer: number
    experts: number[][]
    weights: number[][]
  }>
}

export type ModelInfo = {
  model_kind: string
  vocab_size: number
  num_layers: number
  num_heads: number
  block_size: number
  num_experts: number
  moe_top_k: number
  tokenizer_vocab_size: number
  model_tokenizer_vocab_match: boolean
}

export type TrainRequestPayload = {
  train_steps: number
  learning_rate: number
  checkpoint_interval: number
  eval_interval: number
  resume_from?: string
}

export type TrainAccepted = {
  run_id: string
}

/** A stop lands on `interrupted` — the server has no separate "stopped" state. */
export type TrainRunState = 'running' | 'completed' | 'interrupted' | 'failed'

export type TrainStatus = {
  run_id: string
  status: TrainRunState
  request: TrainRequestPayload
  started_at_unix: number
  ended_at_unix: number | null
  steps_completed: number
  total_steps: number
  training_loss: number | null
  value_loss: number | null
  steps_per_second: number | null
  checkpoints: string[]
  eta_seconds: number | null
  /** Absent from the JSON entirely unless `status` is `failed`. */
  error?: string
}

/** Shape of the `/api/train` JSON error bodies, e.g. `{"error":"run_in_progress","run_id":"..."}`. */
export type TrainErrorBody = {
  error: string
  run_id?: string
  message?: string
  max_allowed?: number
  requested?: number
}

type GenerateOptions = {
  baseUrl?: string
  maxTokens?: number
  temperature?: number
  topK?: number
}

export class ApiError extends Error {
  readonly status: number
  /** Parsed error body when the server sent JSON; `null` for plain-text errors. */
  readonly body: TrainErrorBody | null

  constructor(status: number, message: string, body: TrainErrorBody | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

export async function generateText(
  prompt: string,
  { baseUrl = '', maxTokens = 80, temperature = 1.0, topK }: GenerateOptions = {},
): Promise<GenerateResponse> {
  const response = await fetch(`${baseUrl}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      max_tokens: maxTokens,
      temperature,
      ...(topK === undefined ? {} : { top_k: topK }),
    }),
  })

  await throwIfRequestFailed(response)
  const payload: unknown = await response.json()
  if (!isGenerateResponse(payload)) {
    throw new ApiError(0, 'Server returned an invalid generation response.')
  }

  return payload
}

export async function getModelInfo(baseUrl = ''): Promise<ModelInfo> {
  const response = await fetch(`${baseUrl}/api/info`)

  await throwIfRequestFailed(response)
  const payload: unknown = await response.json()
  if (!isModelInfo(payload)) {
    throw new ApiError(0, 'Server returned invalid model info.')
  }

  return payload
}

/** `POST /api/train` — start a run. `202` carries the run id to poll. */
export async function startTraining(
  request: TrainRequestPayload,
  baseUrl = '',
): Promise<TrainAccepted> {
  const response = await fetch(`${baseUrl}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  await throwIfRequestFailed(response)
  const payload: unknown = await response.json()
  if (!isTrainAccepted(payload)) {
    throw new ApiError(0, 'Server returned an invalid training response.')
  }

  return payload
}

/** `GET /api/train/{run_id}/status` — exempt from the generate rate limiter, safe to poll. */
export async function getTrainStatus(runId: string, baseUrl = ''): Promise<TrainStatus> {
  const response = await fetch(`${baseUrl}/api/train/${encodeURIComponent(runId)}/status`)

  await throwIfRequestFailed(response)
  const payload: unknown = await response.json()
  if (!isTrainStatus(payload)) {
    throw new ApiError(0, 'Server returned an invalid training status.')
  }

  return payload
}

/**
 * `DELETE /api/train/{run_id}` — request a stop. `202` means "requested", not
 * "stopped": the run reaches `interrupted` at its next step boundary, so keep
 * polling status afterwards. `404` means it is no longer the running run.
 */
export async function stopTraining(runId: string, baseUrl = ''): Promise<void> {
  const response = await fetch(`${baseUrl}/api/train/${encodeURIComponent(runId)}`, {
    method: 'DELETE',
  })

  await throwIfRequestFailed(response)
}

async function throwIfRequestFailed(response: Response): Promise<void> {
  if (response.ok !== false) {
    return
  }

  let message = response.statusText || `Request failed with status ${response.status}`
  let body: TrainErrorBody | null = null
  try {
    const text = await response.text()
    if (text.trim()) {
      message = text
      body = parseErrorBody(text)
    }
  } catch {
    // Fall back to the status text when the body cannot be read.
  }

  throw new ApiError(response.status, message, body)
}

function parseErrorBody(text: string): TrainErrorBody | null {
  try {
    const payload: unknown = JSON.parse(text)
    return isRecord(payload) && typeof payload.error === 'string'
      ? (payload as TrainErrorBody)
      : null
  } catch {
    return null
  }
}

function isGenerateResponse(value: unknown): value is GenerateResponse {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.generated === 'string' &&
    Array.isArray(value.tokens) &&
    value.tokens.every((token) => typeof token === 'string') &&
    Array.isArray(value.attention) &&
    value.attention.every(isAttentionData) &&
    (value.routing === undefined ||
      (Array.isArray(value.routing) && value.routing.every(isRoutingData)))
  )
}

function isAttentionData(value: unknown): value is GenerateResponse['attention'][number] {
  if (!isRecord(value)) {
    return false
  }

  return (
    Number.isInteger(value.layer) &&
    Number.isInteger(value.head) &&
    Array.isArray(value.weights) &&
    value.weights.every(
      (row) => Array.isArray(row) && row.every((weight) => typeof weight === 'number'),
    )
  )
}

function isRoutingData(value: unknown): value is NonNullable<GenerateResponse['routing']>[number] {
  if (!isRecord(value)) {
    return false
  }

  return (
    Number.isInteger(value.layer) &&
    Array.isArray(value.experts) &&
    value.experts.every(
      (row) => Array.isArray(row) && row.every((expert) => Number.isInteger(expert)),
    ) &&
    Array.isArray(value.weights) &&
    value.weights.every(
      (row) => Array.isArray(row) && row.every((weight) => typeof weight === 'number'),
    )
  )
}

function isModelInfo(value: unknown): value is ModelInfo {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.model_kind === 'string' &&
    Number.isInteger(value.vocab_size) &&
    Number.isInteger(value.num_layers) &&
    Number.isInteger(value.num_heads) &&
    Number.isInteger(value.block_size) &&
    Number.isInteger(value.num_experts) &&
    Number.isInteger(value.moe_top_k) &&
    Number.isInteger(value.tokenizer_vocab_size) &&
    typeof value.model_tokenizer_vocab_match === 'boolean'
  )
}

function isTrainAccepted(value: unknown): value is TrainAccepted {
  return isRecord(value) && typeof value.run_id === 'string'
}

function isTrainStatus(value: unknown): value is TrainStatus {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.run_id === 'string' &&
    isTrainRunState(value.status) &&
    isTrainRequestPayload(value.request) &&
    Number.isInteger(value.started_at_unix) &&
    isNullableInteger(value.ended_at_unix) &&
    Number.isInteger(value.steps_completed) &&
    Number.isInteger(value.total_steps) &&
    isNullableNumber(value.training_loss) &&
    isNullableNumber(value.value_loss) &&
    isNullableNumber(value.steps_per_second) &&
    isNullableInteger(value.eta_seconds) &&
    Array.isArray(value.checkpoints) &&
    value.checkpoints.every((checkpoint) => typeof checkpoint === 'string') &&
    (value.error === undefined || typeof value.error === 'string')
  )
}

function isTrainRequestPayload(value: unknown): value is TrainRequestPayload {
  if (!isRecord(value)) {
    return false
  }

  return (
    Number.isInteger(value.train_steps) &&
    typeof value.learning_rate === 'number' &&
    Number.isInteger(value.checkpoint_interval) &&
    Number.isInteger(value.eval_interval) &&
    (value.resume_from === undefined || typeof value.resume_from === 'string')
  )
}

function isTrainRunState(value: unknown): value is TrainRunState {
  return (
    value === 'running' || value === 'completed' || value === 'interrupted' || value === 'failed'
  )
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || typeof value === 'number'
}

function isNullableInteger(value: unknown): value is number | null {
  return value === null || Number.isInteger(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
