import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { ArrowLeft, Play, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react'
import clsx from 'clsx'
import { useCreateJob, useModels } from '../api/hooks'
import type { TrainingJobCreate, TrainingType } from '../api/types'

const defaultConfig: TrainingJobCreate = {
  model_path: '',
  dataset_path: '',
  training_type: 'sft',
  epochs: 3,
  learning_rate: 2e-5,
  batch_size: 4,
  lora_enabled: true,
  lora_r: 16,
  lora_alpha: 32,
  max_seq_length: 2048,
  gradient_accumulation_steps: 4,
  warmup_ratio: 0.1,
  fp16: false,
  bf16: true,
}

export default function NewJobPage() {
  const navigate = useNavigate()
  const createJob = useCreateJob()
  const { data: models } = useModels()

  const [config, setConfig] = useState<TrainingJobCreate>(defaultConfig)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  const updateConfig = <K extends keyof TrainingJobCreate>(
    key: K,
    value: TrainingJobCreate[K]
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
    setErrors((prev) => ({ ...prev, [key]: '' }))
  }

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (!config.model_path) {
      newErrors.model_path = 'Model path is required'
    }
    if (!config.dataset_path) {
      newErrors.dataset_path = 'Dataset path is required'
    }
    if (config.epochs < 1 || config.epochs > 100) {
      newErrors.epochs = 'Epochs must be between 1 and 100'
    }
    if (config.batch_size < 1 || config.batch_size > 128) {
      newErrors.batch_size = 'Batch size must be between 1 and 128'
    }
    if (config.learning_rate <= 0 || config.learning_rate > 1) {
      newErrors.learning_rate = 'Learning rate must be between 0 and 1'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validate()) return

    try {
      const job = await createJob.mutateAsync(config)
      navigate(`/jobs/${job.job_id}`)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create job'
      setErrors({ submit: message })
    }
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center space-x-4 mb-6">
        <Link
          to="/jobs"
          className="p-2 rounded-md hover:bg-gray-100 transition-colors"
        >
          <ArrowLeft className="h-5 w-5 text-gray-500" />
        </Link>
        <h1 className="text-2xl font-bold text-gray-900">New Training Job</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="card p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Basic Configuration</h2>

          <div className="space-y-4">
            {/* Model Path */}
            <div>
              <label className="label">Model Path</label>
              <input
                type="text"
                value={config.model_path}
                onChange={(e) => updateConfig('model_path', e.target.value)}
                placeholder="Qwen/Qwen2.5-7B-Instruct or /path/to/model"
                className={clsx('input', errors.model_path && 'border-red-500')}
                list="model-suggestions"
              />
              <datalist id="model-suggestions">
                {models?.map((m) => (
                  <option key={m.path} value={m.path}>
                    {m.name}
                  </option>
                ))}
                <option value="Qwen/Qwen2.5-7B-Instruct" />
                <option value="Qwen/Qwen2.5-3B-Instruct" />
                <option value="meta-llama/Llama-3.2-3B-Instruct" />
              </datalist>
              {errors.model_path && (
                <p className="mt-1 text-sm text-red-600">{errors.model_path}</p>
              )}
            </div>

            {/* Dataset Path */}
            <div>
              <label className="label">Dataset Path</label>
              <input
                type="text"
                value={config.dataset_path}
                onChange={(e) => updateConfig('dataset_path', e.target.value)}
                placeholder="/path/to/dataset.jsonl"
                className={clsx('input', errors.dataset_path && 'border-red-500')}
              />
              {errors.dataset_path && (
                <p className="mt-1 text-sm text-red-600">{errors.dataset_path}</p>
              )}
            </div>

            {/* Training Type */}
            <div>
              <label className="label">Training Type</label>
              <div className="grid grid-cols-2 gap-4">
                {(['sft', 'dpo'] as TrainingType[]).map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => updateConfig('training_type', type)}
                    className={clsx(
                      'p-4 rounded-lg border-2 text-left transition-colors',
                      config.training_type === type
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    )}
                  >
                    <span className="block text-sm font-medium text-gray-900 uppercase">
                      {type}
                    </span>
                    <span className="block text-xs text-gray-500 mt-1">
                      {type === 'sft'
                        ? 'Supervised Fine-Tuning'
                        : 'Direct Preference Optimization'}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Basic Parameters */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Epochs</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => updateConfig('epochs', parseInt(e.target.value) || 1)}
                  min={1}
                  max={100}
                  className={clsx('input', errors.epochs && 'border-red-500')}
                />
                {errors.epochs && (
                  <p className="mt-1 text-sm text-red-600">{errors.epochs}</p>
                )}
              </div>
              <div>
                <label className="label">Batch Size</label>
                <input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => updateConfig('batch_size', parseInt(e.target.value) || 1)}
                  min={1}
                  max={128}
                  className={clsx('input', errors.batch_size && 'border-red-500')}
                />
                {errors.batch_size && (
                  <p className="mt-1 text-sm text-red-600">{errors.batch_size}</p>
                )}
              </div>
            </div>

            {/* Learning Rate */}
            <div>
              <label className="label">Learning Rate</label>
              <input
                type="text"
                value={config.learning_rate}
                onChange={(e) => updateConfig('learning_rate', parseFloat(e.target.value) || 0)}
                className={clsx('input', errors.learning_rate && 'border-red-500')}
              />
              {errors.learning_rate && (
                <p className="mt-1 text-sm text-red-600">{errors.learning_rate}</p>
              )}
              <p className="mt-1 text-xs text-gray-500">Recommended: 2e-5 for SFT, 5e-7 for DPO</p>
            </div>

            {/* LoRA Toggle */}
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <span className="text-sm font-medium text-gray-900">Enable LoRA</span>
                <p className="text-xs text-gray-500">
                  Low-Rank Adaptation for efficient fine-tuning
                </p>
              </div>
              <button
                type="button"
                onClick={() => updateConfig('lora_enabled', !config.lora_enabled)}
                className={clsx(
                  'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
                  config.lora_enabled ? 'bg-primary-600' : 'bg-gray-200'
                )}
              >
                <span
                  className={clsx(
                    'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
                    config.lora_enabled ? 'translate-x-5' : 'translate-x-0'
                  )}
                />
              </button>
            </div>

            {/* LoRA Parameters */}
            {config.lora_enabled && (
              <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
                <div>
                  <label className="label">LoRA Rank (r)</label>
                  <input
                    type="number"
                    value={config.lora_r}
                    onChange={(e) => updateConfig('lora_r', parseInt(e.target.value) || 8)}
                    min={1}
                    max={256}
                    className="input"
                  />
                </div>
                <div>
                  <label className="label">LoRA Alpha</label>
                  <input
                    type="number"
                    value={config.lora_alpha}
                    onChange={(e) => updateConfig('lora_alpha', parseInt(e.target.value) || 16)}
                    min={1}
                    className="input"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Advanced Options */}
        <div className="card">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full px-6 py-4 flex items-center justify-between text-left"
          >
            <span className="text-lg font-medium text-gray-900">Advanced Options</span>
            {showAdvanced ? (
              <ChevronUp className="h-5 w-5 text-gray-500" />
            ) : (
              <ChevronDown className="h-5 w-5 text-gray-500" />
            )}
          </button>

          {showAdvanced && (
            <div className="px-6 pb-6 space-y-4 border-t border-gray-200 pt-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Max Sequence Length</label>
                  <input
                    type="number"
                    value={config.max_seq_length}
                    onChange={(e) =>
                      updateConfig('max_seq_length', parseInt(e.target.value) || 2048)
                    }
                    min={128}
                    max={32768}
                    className="input"
                  />
                </div>
                <div>
                  <label className="label">Gradient Accumulation Steps</label>
                  <input
                    type="number"
                    value={config.gradient_accumulation_steps}
                    onChange={(e) =>
                      updateConfig('gradient_accumulation_steps', parseInt(e.target.value) || 1)
                    }
                    min={1}
                    max={64}
                    className="input"
                  />
                </div>
              </div>

              <div>
                <label className="label">Warmup Ratio</label>
                <input
                  type="number"
                  value={config.warmup_ratio}
                  onChange={(e) => updateConfig('warmup_ratio', parseFloat(e.target.value) || 0)}
                  min={0}
                  max={1}
                  step={0.01}
                  className="input"
                />
              </div>

              <div className="flex items-center space-x-6">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.bf16}
                    onChange={(e) => {
                      updateConfig('bf16', e.target.checked)
                      if (e.target.checked) updateConfig('fp16', false)
                    }}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-2 text-sm text-gray-900">BF16 (Recommended)</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.fp16}
                    onChange={(e) => {
                      updateConfig('fp16', e.target.checked)
                      if (e.target.checked) updateConfig('bf16', false)
                    }}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-2 text-sm text-gray-900">FP16</span>
                </label>
              </div>

              <div>
                <label className="label">Output Directory (Optional)</label>
                <input
                  type="text"
                  value={config.output_dir || ''}
                  onChange={(e) => updateConfig('output_dir', e.target.value || undefined)}
                  placeholder="./output (defaults to auto-generated path)"
                  className="input"
                />
              </div>
            </div>
          )}
        </div>

        {/* Submit Error */}
        {errors.submit && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
            <AlertCircle className="h-5 w-5 text-red-500 mr-3 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-medium text-red-800">Error creating job</h4>
              <p className="text-sm text-red-600 mt-1">{errors.submit}</p>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-end space-x-4">
          <Link to="/jobs" className="btn btn-secondary">
            Cancel
          </Link>
          <button
            type="submit"
            disabled={createJob.isPending}
            className="btn btn-primary"
          >
            {createJob.isPending ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Creating...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Training
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
