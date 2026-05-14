export type GenerateResponse = {
  generated: string
  tokens: string[]
  attention: Array<{
    layer: number
    head: number
    weights: number[][]
  }>
}

export type ModelInfo = {
  vocab_size: number
  num_layers: number
  num_heads: number
  block_size: number
  tokenizer_vocab_size: number
  model_tokenizer_vocab_match: boolean
}

type GenerateOptions = {
  baseUrl?: string
  maxTokens?: number
  temperature?: number
  topK?: number
}

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
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

async function throwIfRequestFailed(response: Response): Promise<void> {
  if (response.ok !== false) {
    return
  }

  let message = response.statusText || `Request failed with status ${response.status}`
  try {
    const body = await response.text()
    if (body.trim()) {
      message = body
    }
  } catch {
    // Fall back to the status text when the body cannot be read.
  }

  throw new ApiError(response.status, message)
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
    value.attention.every(isAttentionData)
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

function isModelInfo(value: unknown): value is ModelInfo {
  if (!isRecord(value)) {
    return false
  }

  return (
    Number.isInteger(value.vocab_size) &&
    Number.isInteger(value.num_layers) &&
    Number.isInteger(value.num_heads) &&
    Number.isInteger(value.block_size) &&
    Number.isInteger(value.tokenizer_vocab_size) &&
    typeof value.model_tokenizer_vocab_match === 'boolean'
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
