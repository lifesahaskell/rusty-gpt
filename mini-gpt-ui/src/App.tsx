import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { generateText, getModelInfo, type GenerateResponse, type ModelInfo } from './api'
import AttentionHeatmap from './components/AttentionHeatmap'
import TrainingDashboard from './components/TrainingDashboard'
import './App.css'

type Workspace = 'prompt' | 'attention' | 'training'

const workspaces: Array<{ id: Workspace; label: string }> = [
  { id: 'prompt', label: 'Prompt' },
  { id: 'attention', label: 'Attention' },
  { id: 'training', label: 'Training' },
]

function App() {
  const [activeWorkspace, setActiveWorkspace] = useState<Workspace>('prompt')
  const workspaceTabRefs = useRef<Array<HTMLButtonElement | null>>([])
  const [prompt, setPrompt] = useState('ROMEO:')
  const [maxTokens, setMaxTokens] = useState('80')
  const [temperature, setTemperature] = useState('1')
  const [topK, setTopK] = useState('')
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [modelInfoError, setModelInfoError] = useState<string | null>(null)
  const [response, setResponse] = useState<GenerateResponse | null>(null)
  const [selectedToken, setSelectedToken] = useState<number | null>(null)
  const [selectedAttentionIndex, setSelectedAttentionIndex] = useState(0)
  const [generationError, setGenerationError] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)

  useEffect(() => {
    let isCurrent = true

    getModelInfo()
      .then((info) => {
        if (isCurrent) {
          setModelInfo(info)
          setModelInfoError(null)
        }
      })
      .catch((error: unknown) => {
        if (isCurrent) {
          setModelInfoError(error instanceof Error ? error.message : 'Unable to load model info.')
        }
      })

    return () => {
      isCurrent = false
    }
  }, [])

  const handleGenerate = async () => {
    setGenerationError(null)
    setIsGenerating(true)

    try {
      setResponse(
        await generateText(prompt, {
          maxTokens: Number(maxTokens),
          temperature: Number(temperature),
          topK: topK.trim() === '' ? undefined : Number(topK),
        }),
      )
      setSelectedToken(null)
      setSelectedAttentionIndex(0)
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : 'Generation failed.')
    } finally {
      setIsGenerating(false)
    }
  }

  const selectedAttention = response?.attention[selectedAttentionIndex] ?? response?.attention[0]
  const attentionTokenOffset =
    response && selectedAttention ? Math.max(0, response.tokens.length - selectedAttention.weights.length) : 0
  const selectedAttentionRow =
    selectedToken !== null && selectedToken >= attentionTokenOffset ? selectedToken - attentionTokenOffset : null

  const activateWorkspace = (index: number) => {
    const nextIndex = (index + workspaces.length) % workspaces.length
    setActiveWorkspace(workspaces[nextIndex].id)
    workspaceTabRefs.current[nextIndex]?.focus()
  }

  const handleWorkspaceKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === 'ArrowRight') {
      event.preventDefault()
      activateWorkspace(index + 1)
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault()
      activateWorkspace(index - 1)
    } else if (event.key === 'Home') {
      event.preventDefault()
      activateWorkspace(0)
    } else if (event.key === 'End') {
      event.preventDefault()
      activateWorkspace(workspaces.length - 1)
    }
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>MiniGPT UI</h1>
        <section className="model-info" aria-labelledby="model-info-heading">
          <h2 id="model-info-heading">Model</h2>
          {modelInfo ? (
            <dl>
              <div>
                <dt>Vocabulary</dt>
                <dd>{modelInfo.vocab_size}</dd>
              </div>
              <div>
                <dt>Tokenizer</dt>
                <dd>{modelInfo.tokenizer_vocab_size}</dd>
              </div>
              <div>
                <dt>Compatibility</dt>
                <dd>{modelInfo.model_tokenizer_vocab_match ? 'Ready' : 'Mismatch'}</dd>
              </div>
              <div>
                <dt>Layers</dt>
                <dd>{modelInfo.num_layers}</dd>
              </div>
              <div>
                <dt>Heads</dt>
                <dd>{modelInfo.num_heads}</dd>
              </div>
              <div>
                <dt>Context</dt>
                <dd>{modelInfo.block_size}</dd>
              </div>
            </dl>
          ) : (
            <p role={modelInfoError ? 'alert' : undefined}>
              {modelInfoError ?? 'Loading model info...'}
            </p>
          )}
        </section>
      </header>

      <nav className="workspace-tabs" aria-label="MiniGPT workspaces" role="tablist">
        {workspaces.map((workspace, index) => (
          <button
            key={workspace.id}
            ref={(element) => {
              workspaceTabRefs.current[index] = element
            }}
            id={`${workspace.id}-tab`}
            type="button"
            role="tab"
            aria-selected={activeWorkspace === workspace.id}
            aria-controls={`${workspace.id}-panel`}
            onClick={() => setActiveWorkspace(workspace.id)}
            onKeyDown={(event) => handleWorkspaceKeyDown(event, index)}
          >
            {workspace.label}
          </button>
        ))}
      </nav>

      <main className="workspace">
        <section
          id="prompt-panel"
          className="workspace-panel"
          role="tabpanel"
          aria-labelledby="prompt-tab"
          hidden={activeWorkspace !== 'prompt'}
        >
          <div className="prompt-controls">
            <div>
              <h2>Generation</h2>
            </div>
            <label htmlFor="prompt">Prompt</label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
            />
            <div className="generation-options">
              <label htmlFor="max-tokens">
                Max tokens
                <input
                  id="max-tokens"
                  min="1"
                  type="number"
                  value={maxTokens}
                  onChange={(event) => setMaxTokens(event.target.value)}
                />
              </label>
              <label htmlFor="temperature">
                Temperature
                <input
                  id="temperature"
                  min="0.01"
                  step="0.05"
                  type="number"
                  value={temperature}
                  onChange={(event) => setTemperature(event.target.value)}
                />
              </label>
              <label htmlFor="top-k">
                Top K
                <input
                  id="top-k"
                  min="1"
                  placeholder="Any"
                  type="number"
                  value={topK}
                  onChange={(event) => setTopK(event.target.value)}
                />
              </label>
            </div>
            <button type="button" onClick={handleGenerate} disabled={isGenerating}>
              {isGenerating ? 'Generating...' : 'Generate'}
            </button>
            {generationError && (
              <p className="form-error" role="alert">
                {generationError}
              </p>
            )}
          </div>

          {response && (
            <section className="generation-panel" aria-labelledby="generated-text-heading">
              <h2 id="generated-text-heading">Generated Text</h2>
              <p>{response.generated}</p>
              <h2>Generated Tokens</h2>
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
              {response.attention.length > 0 && (
                <button
                  className="secondary-action"
                  type="button"
                  onClick={() => setActiveWorkspace('attention')}
                >
                  Attention
                </button>
              )}
            </section>
          )}
        </section>

        <section
          id="attention-panel"
          className="workspace-panel"
          role="tabpanel"
          aria-labelledby="attention-tab"
          hidden={activeWorkspace !== 'attention'}
        >
          {response && selectedAttention ? (
            <section className="attention-panel" aria-labelledby="attention-heading">
              <div className="attention-header">
                <div>
                  <h2 id="attention-heading">Attention Visualization</h2>
                  <p>
                    {selectedToken === null
                      ? 'All attention rows are visible.'
                      : `Focused token: "${response.tokens[selectedToken]}"`}
                  </p>
                </div>
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
              <button
                className="secondary-action"
                type="button"
                onClick={() => setActiveWorkspace('prompt')}
              >
                Back to tokens
              </button>
            </section>
          ) : (
            <div className="empty-state">
              <h2>Attention Visualization</h2>
              <button type="button" onClick={() => setActiveWorkspace('prompt')}>
                Prompt
              </button>
            </div>
          )}
        </section>

        <section
          id="training-panel"
          className="workspace-panel"
          role="tabpanel"
          aria-labelledby="training-tab"
          hidden={activeWorkspace !== 'training'}
        >
          <TrainingDashboard />
        </section>
      </main>
    </div>
  )
}

export default App
