import { useMemo, useState, type DragEvent, type ChangeEvent } from 'react'

type TrainingFile = {
  id: string
  name: string
  size: number
  type: string
}

function fileId(file: File) {
  return `${file.name}-${file.size}-${file.lastModified}`
}

function formatBytes(size: number) {
  if (size < 1024) {
    return `${size} B`
  }

  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`
  }

  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

function toTrainingFile(file: File): TrainingFile {
  return {
    id: fileId(file),
    name: file.name,
    size: file.size,
    type: file.type || 'text/plain',
  }
}

export default function TrainingDashboard() {
  const [files, setFiles] = useState<TrainingFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [status, setStatus] = useState('Add training data to prepare a run.')

  const totalSize = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files])

  const addFiles = (incomingFiles: FileList | File[]) => {
    const nextFiles = Array.from(incomingFiles).map(toTrainingFile)

    setFiles((currentFiles) => {
      const existingIds = new Set(currentFiles.map((file) => file.id))
      const uniqueFiles = nextFiles.filter((file) => !existingIds.has(file.id))

      if (uniqueFiles.length === 0) {
        setStatus('Those files are already in the training set.')
        return currentFiles
      }

      setStatus(`${uniqueFiles.length} file${uniqueFiles.length === 1 ? '' : 's'} ready for training.`)
      return [...currentFiles, ...uniqueFiles]
    })
  }

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.currentTarget.files) {
      addFiles(event.currentTarget.files)
    }
    event.currentTarget.value = ''
  }

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    setIsDragging(false)
    addFiles(event.dataTransfer.files)
  }

  const removeFile = (fileIdToRemove: string) => {
    setFiles((currentFiles) => currentFiles.filter((file) => file.id !== fileIdToRemove))
    setStatus('Training data updated.')
  }

  const startTraining = () => {
    setStatus(`${files.length} file${files.length === 1 ? '' : 's'} queued for training.`)
  }

  return (
    <section className="training-dashboard" aria-labelledby="training-heading">
      <div className="training-header">
        <div>
          <h2 id="training-heading">Training Dashboard</h2>
          <p>Prepare local text data before starting a MiniGPT training run.</p>
        </div>
        <button type="button" onClick={startTraining} disabled={files.length === 0}>
          Start training
        </button>
      </div>

      <label
        className={`training-dropzone${isDragging ? ' is-dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <span>Drop training files here</span>
        <small>TXT, Markdown, JSON, JSONL, and CSV files are good inputs.</small>
        <input
          type="file"
          multiple
          accept=".txt,.md,.json,.jsonl,.csv,text/plain,text/markdown,application/json,text/csv"
          onChange={handleFileInput}
        />
      </label>

      <div className="training-summary" aria-live="polite">
        <span>{status}</span>
        <span>
          {files.length} file{files.length === 1 ? '' : 's'} · {formatBytes(totalSize)}
        </span>
      </div>

      {files.length > 0 && (
        <ul className="training-file-list" aria-label="Training files">
          {files.map((file) => (
            <li key={file.id}>
              <div>
                <strong>{file.name}</strong>
                <span>
                  {formatBytes(file.size)} · {file.type}
                </span>
              </div>
              <button type="button" onClick={() => removeFile(file.id)} aria-label={`Remove ${file.name}`}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
