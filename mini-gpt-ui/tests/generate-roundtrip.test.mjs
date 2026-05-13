import assert from 'node:assert/strict'
import { pathToFileURL } from 'node:url'
import test from 'node:test'

test('generateText posts the server contract and reads the generated response', async () => {
  const originalFetch = globalThis.fetch
  let receivedRequest

  globalThis.fetch = async (url, options) => {
    receivedRequest = {
      url,
      method: options.method,
      contentType: options.headers['Content-Type'],
      body: JSON.parse(options.body),
    }

    return {
      async json() {
        return {
          generated: 'ROMEO: hi',
          tokens: ['R', 'O'],
          attention: [{ layer: 0, head: 0, weights: [[1]] }],
        }
      },
    }
  }

  try {
    const { generateText } = await import(
      pathToFileURL('/tmp/rusty-gpt-mini-gpt-ui-roundtrip/api.js').href
    )
    const result = await generateText('ROMEO:', {
      baseUrl: 'http://example.test',
    })

    assert.deepEqual(receivedRequest, {
      url: 'http://example.test/api/generate',
      method: 'POST',
      contentType: 'application/json',
      body: { prompt: 'ROMEO:', max_tokens: 80, temperature: 1 },
    })
    assert.deepEqual(result, {
      generated: 'ROMEO: hi',
      tokens: ['R', 'O'],
      attention: [{ layer: 0, head: 0, weights: [[1]] }],
    })
  } finally {
    globalThis.fetch = originalFetch
  }
})
