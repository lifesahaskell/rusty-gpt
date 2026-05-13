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
}

export async function generateText(
  prompt: string,
  { baseUrl = '', maxTokens = 80, temperature = 1.0 }: GenerateOptions = {},
): Promise<GenerateResponse> {
  const response = await fetch(`${baseUrl}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_tokens: maxTokens, temperature }),
  })

  return response.json()
}
