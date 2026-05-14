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
      ok: true,
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

test('getModelInfo reads the server model metadata contract', async () => {
  const originalFetch = globalThis.fetch
  let receivedUrl

  globalThis.fetch = async (url) => {
    receivedUrl = url

    return {
      ok: true,
      async json() {
        return {
          vocab_size: 2048,
          num_layers: 4,
          num_heads: 4,
          block_size: 128,
          tokenizer_vocab_size: 2048,
          model_tokenizer_vocab_match: true,
        }
      },
    }
  }

  try {
    const { getModelInfo } = await import(
      pathToFileURL('/tmp/rusty-gpt-mini-gpt-ui-roundtrip/api.js').href
    )
    const result = await getModelInfo('http://example.test')

    assert.equal(receivedUrl, 'http://example.test/api/info')
    assert.deepEqual(result, {
      vocab_size: 2048,
      num_layers: 4,
      num_heads: 4,
      block_size: 128,
      tokenizer_vocab_size: 2048,
      model_tokenizer_vocab_match: true,
    })
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('generateText throws server validation errors', async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = async () => ({
    ok: false,
    status: 400,
    statusText: 'Bad Request',
    async text() {
      return 'prompt must not be empty'
    },
  })

  try {
    const { generateText } = await import(
      pathToFileURL('/tmp/rusty-gpt-mini-gpt-ui-roundtrip/api.js').href
    )

    await assert.rejects(
      () => generateText(''),
      (error) => error.name === 'ApiError' && error.status === 400 && error.message === 'prompt must not be empty',
    )
  } finally {
    globalThis.fetch = originalFetch
  }
})
