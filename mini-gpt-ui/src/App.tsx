import { useState } from 'react'
import { generateText, type GenerateResponse } from './api'
import AttentionHeatmap from './components/AttentionHeatmap'
import TrainingDashboard from './components/TrainingDashboard'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('ROMEO:')
  const [response, setResponse] = useState<GenerateResponse | null>(null)
  const [selectedToken, setSelectedToken] = useState<number | null>(null)
  const [selectedAttentionIndex, setSelectedAttentionIndex] = useState(0)
  
  // TODO: bootstrap the model and tokenizer on app load, and reuse them for each generation request
  const handleGenerate = async () => {
    setResponse(await generateText(prompt))
    setSelectedToken(null)
    setSelectedAttentionIndex(0)
  }

  const selectedAttention = response?.attention[selectedAttentionIndex] ?? response?.attention[0]
  const attentionTokenOffset =
    response && selectedAttention ? Math.max(0, response.tokens.length - selectedAttention.weights.length) : 0
  const selectedAttentionRow =
    selectedToken !== null && selectedToken >= attentionTokenOffset ? selectedToken - attentionTokenOffset : null

  return (
    <div className="App">
      <h1>MiniGPT UI</h1>
      <TrainingDashboard />
      <div className="prompt-controls">
        <label htmlFor="prompt">Prompt</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
        />
        <button type="button" onClick={handleGenerate}>
          Generate
        </button>
      </div>
      {response && (
        <div className="response">
          <div className="generation-panel">
            <h2>Generated Text:</h2>
            <p>{response.generated}</p>
            <h2>Generated Tokens:</h2>
            <ul className="token-list" aria-label="Generated tokens">
              {response.tokens.map((token, index) => (
                <li
                  key={index}
                  className={index === selectedToken ? 'selected' : ''}
                >
                  <button type="button" onClick={() => setSelectedToken(index)}>
                    {token}
                  </button>
                </li>
              ))}
            </ul>
            {selectedToken !== null && (
              <div className="logprobs">
                <h3>Selected Token: "{response.tokens[selectedToken]}"</h3>
              </div>
            )}
          </div>
          {selectedAttention && (
            <section className="attention-panel" aria-labelledby="attention-heading">
              <div className="attention-header">
                <h2 id="attention-heading">Attention</h2>
                <label>
                  Layer / head
                  <select
                    value={selectedAttentionIndex}
                    onChange={(event) => setSelectedAttentionIndex(Number(event.target.value))}
                  >
                    {response.attention.map((attention, index) => (
                      <option key={`${attention.layer}-${attention.head}`} value={index}>
                        Layer {attention.layer}, head {attention.head}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <AttentionHeatmap
                weights={selectedAttention.weights}
                tokens={response.tokens}
                selectedRow={selectedAttentionRow}
              />
            </section>
          )}
        </div>
      )}
    </div>
  )
}

export default App
