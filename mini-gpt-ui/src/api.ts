export type GenerateResponse = {
  generated: string
  tokens: string[]
  attention: Array<{
    layer: number
    head: number
    weights: number[][]
  }>
}

type GenerateOptions = {
  baseUrl?: string
  maxTokens?: number
  temperature?: number
  topK?: number
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

  return response.json()
}
