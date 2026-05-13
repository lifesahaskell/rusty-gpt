import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../../src/App'

const generateResponse = {
  generated: 'ROMEO: hi',
  tokens: ['R', 'O', 'M', 'E', 'O'],
  attention: [
    {
      layer: 0,
      head: 0,
      weights: [
        [1, 0, 0],
        [0.5, 0.5, 0],
        [0.2, 0.3, 0.5],
      ],
    },
    {
      layer: 1,
      head: 0,
      weights: [
        [1, 0, 0],
        [0.8, 0.2, 0],
        [0.4, 0.4, 0.2],
      ],
    },
  ],
}

describe('App generation flow', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('posts the prompt and renders generated text, tokens, and attention controls', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn().mockResolvedValue({
      json: vi.fn().mockResolvedValue(generateResponse),
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await user.clear(screen.getByLabelText(/prompt/i))
    await user.type(screen.getByLabelText(/prompt/i), 'hello')
    await user.click(screen.getByRole('button', { name: /generate/i }))

    await screen.findByText('ROMEO: hi')

    expect(fetchMock).toHaveBeenCalledWith('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: 'hello', max_tokens: 80, temperature: 1 }),
    })
    expect(screen.getByRole('list', { name: /generated tokens/i })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: /layer \/ head/i })).toHaveValue('0')
    expect(screen.getByRole('img', { name: /attention weight heatmap/i })).toBeInTheDocument()
  })

  it('resets token and attention selection after a new generation request', async () => {
    const user = userEvent.setup()
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(generateResponse) })
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(generateResponse) })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await user.click(screen.getByRole('button', { name: /generate/i }))
    await screen.findByText('ROMEO: hi')
    await user.click(screen.getByRole('button', { name: 'E' }))
    await user.selectOptions(screen.getByRole('combobox', { name: /layer \/ head/i }), '1')

    expect(screen.getByText('Selected Token: "E"')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: /layer \/ head/i })).toHaveValue('1')

    await user.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(screen.queryByText(/selected token/i)).not.toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: /layer \/ head/i })).toHaveValue('0')
  })

  it('keeps generated token buttons inside the generated-token list', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ json: vi.fn().mockResolvedValue(generateResponse) }),
    )

    render(<App />)

    await user.click(screen.getByRole('button', { name: /generate/i }))

    const tokenList = await screen.findByRole('list', { name: /generated tokens/i })
    expect(within(tokenList).getAllByRole('button')).toHaveLength(generateResponse.tokens.length)
  })
})
