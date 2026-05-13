import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import AttentionHeatmap from '../../src/components/AttentionHeatmap'

describe('AttentionHeatmap', () => {
  it('renders one cell for each attention weight', () => {
    render(
      <AttentionHeatmap
        weights={[
          [1, 0],
          [0.25, 0.75],
        ]}
        tokens={['A', 'B']}
        selectedRow={null}
        size={120}
      />,
    )

    const heatmap = screen.getByRole('img', { name: /attention weight heatmap/i })

    expect(heatmap).toHaveAttribute('width', '120')
    expect(heatmap.querySelectorAll('rect')).toHaveLength(4)
    expect(screen.getByText('A attends to B: 0.000')).toBeInTheDocument()
    expect(screen.getByText('B attends to A: 0.250')).toBeInTheDocument()
  })

  it('dims unselected rows when a row is selected', () => {
    render(
      <AttentionHeatmap
        weights={[
          [1, 0],
          [0.25, 0.75],
        ]}
        tokens={['A', 'B']}
        selectedRow={1}
      />,
    )

    const cells = screen.getByRole('img').querySelectorAll('rect')

    expect(cells[0]).toHaveAttribute('opacity', '0.22')
    expect(cells[1]).toHaveAttribute('opacity', '0.22')
    expect(cells[2]).toHaveAttribute('opacity', '1')
    expect(cells[3]).toHaveAttribute('opacity', '1')
  })

  it('shows an empty state for an empty matrix', () => {
    render(<AttentionHeatmap weights={[]} tokens={[]} selectedRow={null} />)

    expect(screen.getByText(/no attention weights returned/i)).toBeInTheDocument()
  })
})
