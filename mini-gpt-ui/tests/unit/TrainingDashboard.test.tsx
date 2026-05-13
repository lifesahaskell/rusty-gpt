import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import TrainingDashboard from '../../src/components/TrainingDashboard'

function trainingFile(name: string, contents: string, type = 'text/plain') {
  return new File([contents], name, { type, lastModified: 1_700_000_000_000 })
}

describe('TrainingDashboard', () => {
  it('starts empty with training disabled', () => {
    render(<TrainingDashboard />)

    expect(screen.getByRole('heading', { name: /training dashboard/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /start training/i })).toBeDisabled()
    expect(screen.getByText(/0 files/i)).toBeInTheDocument()
  })

  it('adds files from the file picker and queues them for training', async () => {
    const user = userEvent.setup()
    const file = trainingFile('input.txt', 'training text')

    render(<TrainingDashboard />)

    await user.upload(screen.getByLabelText(/drop training files here/i), file)
    await user.click(screen.getByRole('button', { name: /start training/i }))

    const fileList = screen.getByRole('list', { name: /training files/i })

    expect(within(fileList).getByText('input.txt')).toBeInTheDocument()
    expect(screen.getByText(/1 file queued for training/i)).toBeInTheDocument()
  })

  it('adds files dropped onto the dropzone', () => {
    const file = trainingFile('tiny-shakespeare.txt', 'ROMEO:')

    render(<TrainingDashboard />)

    fireEvent.drop(screen.getByText(/drop training files here/i), {
      dataTransfer: {
        files: [file],
      },
    })

    expect(screen.getByText('tiny-shakespeare.txt')).toBeInTheDocument()
    expect(screen.getByText(/1 file ready for training/i)).toBeInTheDocument()
  })

  it('does not add the same file twice', async () => {
    const user = userEvent.setup()
    const file = trainingFile('dedupe.jsonl', '{"text":"hello"}', 'application/json')

    render(<TrainingDashboard />)

    await user.upload(screen.getByLabelText(/drop training files here/i), file)
    await user.upload(screen.getByLabelText(/drop training files here/i), file)

    expect(screen.getAllByText('dedupe.jsonl')).toHaveLength(1)
    expect(screen.getByText(/already in the training set/i)).toBeInTheDocument()
  })

  it('removes files from the training set', async () => {
    const user = userEvent.setup()
    const file = trainingFile('remove-me.csv', 'text')

    render(<TrainingDashboard />)

    await user.upload(screen.getByLabelText(/drop training files here/i), file)
    await user.click(screen.getByRole('button', { name: /remove remove-me.csv/i }))

    expect(screen.queryByText('remove-me.csv')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /start training/i })).toBeDisabled()
  })
})
