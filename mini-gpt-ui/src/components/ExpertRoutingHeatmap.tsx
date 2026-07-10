import type { GenerateResponse } from '../api'

type RoutingLayer = NonNullable<GenerateResponse['routing']>[number]

type ExpertRoutingHeatmapProps = {
  routing: RoutingLayer[]
  tokens: string[]
}

const palette = [
  '#2563eb',
  '#dc2626',
  '#16a34a',
  '#ca8a04',
  '#9333ea',
  '#0891b2',
  '#ea580c',
  '#4f46e5',
]

export default function ExpertRoutingHeatmap({ routing, tokens }: ExpertRoutingHeatmapProps) {
  if (routing.length === 0) {
    return null
  }

  return (
    <section className="routing-panel" aria-labelledby="routing-heading">
      <h3 id="routing-heading">Expert Routing</h3>
      <div className="routing-layers">
        {routing.map((layer) => (
          <div className="routing-layer" key={layer.layer}>
            <h4>Layer {layer.layer}</h4>
            <div className="routing-grid" role="img" aria-label={`Expert routing for layer ${layer.layer}`}>
              {layer.experts.map((experts, tokenIndex) => (
                <div className="routing-token" key={`${layer.layer}-${tokenIndex}`}>
                  <span className="routing-token-label">
                    {tokens[tokenIndex] ?? `#${tokenIndex}`}
                  </span>
                  <div className="routing-experts">
                    {experts.map((expert, expertIndex) => {
                      const weight = layer.weights[tokenIndex]?.[expertIndex] ?? 0
                      return (
                        <span
                          aria-label={`token ${tokenIndex} expert ${expert} weight ${weight.toFixed(3)}`}
                          className="routing-expert"
                          key={`${expert}-${expertIndex}`}
                          style={{
                            backgroundColor: palette[expert % palette.length],
                            opacity: Math.max(0.25, Math.min(1, weight)),
                          }}
                          title={`expert ${expert}: ${weight.toFixed(3)}`}
                        >
                          {expert}
                        </span>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
