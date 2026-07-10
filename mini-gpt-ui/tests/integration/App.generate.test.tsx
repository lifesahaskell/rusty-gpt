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

const moeGenerateResponse = {
  ...generateResponse,
  routing: [
    {
      layer: 0,
      experts: [
        [1, 3],
        [2, 0],
        [1, 2],
      ],
      weights: [
        [0.7, 0.3],
        [0.6, 0.4],
        [0.8, 0.2],
      ],
    },
  ],
}

const modelInfo = {
  model_kind: 'minigpt',
  vocab_size: 2048,
  num_layers: 4,
  num_heads: 4,
  block_size: 128,
  num_experts: 0,
  moe_top_k: 0,
  tokenizer_vocab_size: 2048,
  model_tokenizer_vocab_match: true,
}

const moeModelInfo = {
  ...modelInfo,
  model_kind: 'moe-gpt',
  num_experts: 4,
  moe_top_k: 2,
}

describe('App generation flow', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('posts the prompt and renders generated text, tokens, and attention controls', async () => {
    const user = userEvent.setup()
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(modelInfo) })
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(generateResponse) })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await screen.findByText('Ready')
    expect(screen.getByRole('tab', { name: /prompt/i })).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByRole('heading', { name: /training dashboard/i })).not.toBeInTheDocument()

    await user.clear(screen.getByRole('textbox', { name: /prompt/i }))
    await user.type(screen.getByRole('textbox', { name: /prompt/i }), 'hello')
    await user.clear(screen.getByLabelText(/max tokens/i))
    await user.type(screen.getByLabelText(/max tokens/i), '12')
    await user.clear(screen.getByLabelText(/temperature/i))
    await user.type(screen.getByLabelText(/temperature/i), '0.75')
    await user.type(screen.getByLabelText(/top k/i), '4')
    await user.click(screen.getByRole('button', { name: /generate/i }))

    await screen.findByText('ROMEO: hi')

    expect(fetchMock).toHaveBeenCalledWith('/api/info')
    expect(fetchMock).toHaveBeenCalledWith('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: 'hello', max_tokens: 12, temperature: 0.75, top_k: 4 }),
    })
    expect(screen.getByRole('list', { name: /generated tokens/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^attention$/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^attention$/i }))

    expect(screen.getByRole('combobox', { name: /layer \/ head/i })).toHaveValue('0')
    expect(screen.getByRole('img', { name: /attention weight heatmap/i })).toBeInTheDocument()
  })

  it('renders expert routing when the generation response includes MoE data', async () => {
    const user = userEvent.setup()
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(moeModelInfo) })
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(moeGenerateResponse) })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await screen.findByText('4 / top 2')
    await user.click(screen.getByRole('button', { name: /generate/i }))
    await screen.findByText('ROMEO: hi')
    await user.click(screen.getByRole('button', { name: /^attention$/i }))

    expect(screen.getByRole('heading', { name: /expert routing/i })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /expert routing for layer 0/i })).toBeInTheDocument()
    expect(screen.getByLabelText(/token 0 expert 1 weight 0.700/i)).toBeInTheDocument()
  })

  it('resets token and attention selection after a new generation request', async () => {
    const user = userEvent.setup()
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(modelInfo) })
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(generateResponse) })
      .mockResolvedValueOnce({ json: vi.fn().mockResolvedValue(generateResponse) })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await screen.findByText('Ready')
    await user.click(screen.getByRole('button', { name: /generate/i }))
    await screen.findByText('ROMEO: hi')
    await user.click(screen.getByRole('button', { name: 'E' }))
    await user.click(screen.getByRole('button', { name: /^attention$/i }))
    await user.selectOptions(screen.getByRole('combobox', { name: /layer \/ head/i }), '1')

    expect(screen.getByText('Focused token: "E"')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: /layer \/ head/i })).toHaveValue('1')

    await user.click(screen.getByRole('button', { name: /back to tokens/i }))
    await user.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    expect(screen.queryByText(/selected token/i)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /^attention$/i }))
    expect(screen.getByRole('combobox', { name: /layer \/ head/i })).toHaveValue('0')
  })

  it('keeps prompt, attention, and training concerns in separate workspaces', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: vi.fn().mockResolvedValue(modelInfo) }))

    render(<App />)

    await screen.findByText('Ready')
    expect(screen.getByRole('heading', { name: /generation/i })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /attention visualization/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /training dashboard/i })).not.toBeInTheDocument()

    screen.getByRole('tab', { name: /prompt/i }).focus()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('heading', { name: /attention visualization/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^prompt$/i })).toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: /prompt/i })).not.toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: /training/i }))
    expect(screen.getByRole('heading', { name: /training dashboard/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /start training/i })).toBeDisabled()
    expect(screen.queryByRole('heading', { name: /generation/i })).not.toBeInTheDocument()
  })

  it('keeps generated token buttons inside the generated-token list', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => {
      if (url === '/api/info') {
        return Promise.resolve({ json: vi.fn().mockResolvedValue(modelInfo) })
      }

      return Promise.resolve({ json: vi.fn().mockResolvedValue(generateResponse) })
    }))

    render(<App />)

    await user.click(screen.getByRole('button', { name: /generate/i }))

    const tokenList = await screen.findByRole('list', { name: /generated tokens/i })
    expect(within(tokenList).getAllByRole('button')).toHaveLength(generateResponse.tokens.length)
  })

  it('surfaces API validation errors from generation requests', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => {
      if (url === '/api/info') {
        return Promise.resolve({ json: vi.fn().mockResolvedValue(modelInfo) })
      }

      return Promise.resolve({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        text: vi.fn().mockResolvedValue('temperature must be greater than zero'),
      })
    }))

    render(<App />)

    await screen.findByText('Ready')
    await user.click(screen.getByRole('button', { name: /generate/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'temperature must be greater than zero',
    )
  })
})
