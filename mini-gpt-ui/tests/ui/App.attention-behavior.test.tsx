import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../../src/App'

const responseWithContextWindow = {
  generated: 'ABCDE',
  tokens: ['A', 'B', 'C', 'D', 'E'],
  attention: [
    {
      layer: 0,
      head: 0,
      weights: [
        [1, 0, 0],
        [0.6, 0.4, 0],
        [0.1, 0.3, 0.6],
      ],
    },
  ],
}

function mockGenerateResponse() {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      json: vi.fn().mockResolvedValue(responseWithContextWindow),
    }),
  )
}

describe('attention UI behavior', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('highlights the attention row for selected tokens inside the attention context window', async () => {
    const user = userEvent.setup()
    mockGenerateResponse()

    render(<App />)

    await user.click(screen.getByRole('button', { name: /generate/i }))
    await screen.findByText('ABCDE')
    await user.click(screen.getByRole('button', { name: 'D' }))

    const cells = screen.getByRole('img', { name: /attention weight heatmap/i }).querySelectorAll('rect')

    expect(screen.getByText('Selected Token: "D"')).toBeInTheDocument()
    expect(cells[0]).toHaveAttribute('opacity', '0.22')
    expect(cells[1]).toHaveAttribute('opacity', '0.22')
    expect(cells[2]).toHaveAttribute('opacity', '0.22')
    expect(cells[3]).toHaveAttribute('opacity', '1')
    expect(cells[4]).toHaveAttribute('opacity', '1')
    expect(cells[5]).toHaveAttribute('opacity', '1')
    expect(cells[6]).toHaveAttribute('opacity', '0.22')
    expect(cells[7]).toHaveAttribute('opacity', '0.22')
    expect(cells[8]).toHaveAttribute('opacity', '0.22')
  })

  it('does not dim the heatmap when the selected token is outside the attention context window', async () => {
    const user = userEvent.setup()
    mockGenerateResponse()

    render(<App />)

    await user.click(screen.getByRole('button', { name: /generate/i }))
    await screen.findByText('ABCDE')
    await user.click(screen.getByRole('button', { name: 'B' }))

    const cells = screen.getByRole('img', { name: /attention weight heatmap/i }).querySelectorAll('rect')

    expect(screen.getByText('Selected Token: "B"')).toBeInTheDocument()
    expect([...cells].every((cell) => cell.getAttribute('opacity') === '1')).toBe(true)
  })
})
