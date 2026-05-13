type AttentionHeatmapProps = {
  weights: number[][]
  tokens: string[]
  selectedRow: number | null
  size?: number
}

function clampUnit(value: number) {
  if (!Number.isFinite(value)) {
    return 0
  }

  return Math.min(1, Math.max(0, value))
}

function heatColor(value: number) {
  const normalized = clampUnit(value)
  const hue = 225 - normalized * 175
  const lightness = 18 + normalized * 50

  return `hsl(${hue} 82% ${lightness}%)`
}

export default function AttentionHeatmap({
  weights,
  tokens,
  selectedRow,
  size = 320,
}: AttentionHeatmapProps) {
  const cellCount = weights.length

  if (cellCount === 0) {
    return <p className="attention-empty">No attention weights returned.</p>
  }

  const cellSize = size / cellCount
  const labelledTokens = tokens.slice(-cellCount)

  return (
    <svg
      className="attention-heatmap"
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label="Attention weight heatmap"
    >
      {weights.map((row, rowIndex) =>
        row.map((value, columnIndex) => {
          const rowToken = labelledTokens[rowIndex] ?? `token ${rowIndex + 1}`
          const columnToken = labelledTokens[columnIndex] ?? `token ${columnIndex + 1}`

          return (
            <rect
              key={`${rowIndex}-${columnIndex}`}
              x={columnIndex * cellSize}
              y={rowIndex * cellSize}
              width={cellSize}
              height={cellSize}
              fill={heatColor(value)}
              opacity={selectedRow === null || selectedRow === rowIndex ? 1 : 0.22}
            >
              <title>
                {rowToken} attends to {columnToken}: {value.toFixed(3)}
              </title>
            </rect>
          )
        }),
      )}
    </svg>
  )
}
