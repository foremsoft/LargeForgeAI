import { useParams, Link, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  CheckCircle,
  Clock,
  AlertCircle,
  Activity,
  XCircle,
  Download,
  Trash2,
  Play,
} from 'lucide-react'
import clsx from 'clsx'
import { useJob, useCancelJob, useDeleteJob } from '../api/hooks'
import { useJobWebSocket } from '../api/websocket'
import type { JobStatus, TrainingProgress } from '../api/types'
import { format } from 'date-fns'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { useState, useEffect } from 'react'

const statusConfig: Record<JobStatus, { color: string; bgColor: string; icon: typeof CheckCircle; text: string }> = {
  pending: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: Clock, text: 'Pending' },
  running: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Activity, text: 'Running' },
  completed: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle, text: 'Completed' },
  failed: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle, text: 'Failed' },
  cancelled: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: XCircle, text: 'Cancelled' },
}

interface LossDataPoint {
  step: number
  loss: number
}

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { data: job, isLoading, error } = useJob(id || '')
  const cancelJob = useCancelJob()
  const deleteJob = useDeleteJob()

  const [lossHistory, setLossHistory] = useState<LossDataPoint[]>([])

  const { progress: wsProgress, logs, isConnected } = useJobWebSocket(
    job?.status === 'running' ? id || null : null
  )

  const currentProgress: TrainingProgress | undefined = wsProgress || job?.progress

  // Update loss history when progress changes
  useEffect(() => {
    if (currentProgress && currentProgress.step > 0) {
      setLossHistory((prev) => {
        const exists = prev.some((p) => p.step === currentProgress.step)
        if (exists) return prev
        return [...prev, { step: currentProgress.step, loss: currentProgress.loss }].slice(-100)
      })
    }
  }, [currentProgress])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (error || !job) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-lg font-medium text-gray-900 mb-2">Job not found</h2>
        <Link to="/jobs" className="text-primary-600 hover:text-primary-800">
          Back to jobs
        </Link>
      </div>
    )
  }

  const config = statusConfig[job.status]
  const StatusIcon = config.icon
  const progressPct = currentProgress
    ? Math.round((currentProgress.step / currentProgress.total_steps) * 100)
    : 0

  const handleDelete = async () => {
    if (window.confirm('Are you sure you want to delete this job?')) {
      await deleteJob.mutateAsync(job.job_id)
      navigate('/jobs')
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link
            to="/jobs"
            className="p-2 rounded-md hover:bg-gray-100 transition-colors"
          >
            <ArrowLeft className="h-5 w-5 text-gray-500" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Job {job.job_id.slice(0, 8)}
            </h1>
            <p className="text-sm text-gray-500">{job.request.model_path}</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          {job.status === 'running' && (
            <button
              onClick={() => cancelJob.mutate(job.job_id)}
              className="btn btn-secondary"
              disabled={cancelJob.isPending}
            >
              <XCircle className="h-4 w-4 mr-2" />
              Cancel
            </button>
          )}
          {job.status === 'completed' && job.output_path && (
            <Link
              to={`/models?path=${encodeURIComponent(job.output_path)}`}
              className="btn btn-primary"
            >
              <Download className="h-4 w-4 mr-2" />
              View Model
            </Link>
          )}
          {['completed', 'failed', 'cancelled'].includes(job.status) && (
            <button
              onClick={handleDelete}
              className="btn btn-danger"
              disabled={deleteJob.isPending}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Status Card */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-6">
              <span className={clsx(
                'inline-flex items-center px-3 py-1 rounded-full text-sm font-medium',
                config.bgColor,
                config.color
              )}>
                <StatusIcon className="h-4 w-4 mr-2" />
                {config.text}
              </span>
              {job.status === 'running' && (
                <span className={clsx(
                  'text-sm',
                  isConnected ? 'text-green-600' : 'text-gray-500'
                )}>
                  {isConnected ? 'Live' : 'Connecting...'}
                </span>
              )}
            </div>

            {job.status === 'running' && currentProgress && (
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-500">
                      Epoch {currentProgress.epoch + 1}/{currentProgress.total_epochs}
                    </span>
                    <span className="text-gray-900 font-medium">{progressPct}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-primary-600 h-3 rounded-full transition-all duration-300"
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Step</span>
                    <p className="text-gray-900 font-medium">
                      {currentProgress.step}/{currentProgress.total_steps}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">Loss</span>
                    <p className="text-gray-900 font-medium">{currentProgress.loss.toFixed(4)}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Learning Rate</span>
                    <p className="text-gray-900 font-medium">{currentProgress.learning_rate.toExponential(2)}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">ETA</span>
                    <p className="text-gray-900 font-medium">
                      {Math.round(currentProgress.eta_seconds / 60)} min
                    </p>
                  </div>
                </div>
              </div>
            )}

            {job.status === 'failed' && job.error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <h4 className="text-sm font-medium text-red-800 mb-1">Error</h4>
                <p className="text-sm text-red-600">{job.error}</p>
              </div>
            )}

            {job.status === 'completed' && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 className="text-sm font-medium text-green-800 mb-1">Training Complete</h4>
                <p className="text-sm text-green-600">
                  Output saved to: {job.output_path || 'N/A'}
                </p>
              </div>
            )}
          </div>

          {/* Loss Chart */}
          {lossHistory.length > 1 && (
            <div className="card p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Training Loss</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" label={{ value: 'Step', position: 'bottom' }} />
                    <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="loss"
                      stroke="#0284c7"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Logs */}
          {logs.length > 0 && (
            <div className="card">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">Training Logs</h3>
              </div>
              <div className="p-4 bg-gray-900 rounded-b-lg max-h-64 overflow-y-auto font-mono text-sm">
                {logs.map((log, i) => (
                  <div
                    key={i}
                    className={clsx(
                      'py-0.5',
                      log.level === 'error' && 'text-red-400',
                      log.level === 'warning' && 'text-yellow-400',
                      log.level === 'info' && 'text-gray-300',
                      !log.level && 'text-gray-400'
                    )}
                  >
                    {log.line}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Configuration Sidebar */}
        <div className="space-y-6">
          <div className="card p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Configuration</h3>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-500">Training Type</dt>
                <dd className="text-gray-900 font-medium uppercase">{job.training_type}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Epochs</dt>
                <dd className="text-gray-900 font-medium">{job.request.epochs}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Batch Size</dt>
                <dd className="text-gray-900 font-medium">{job.request.batch_size}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Learning Rate</dt>
                <dd className="text-gray-900 font-medium">{job.request.learning_rate}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Max Seq Length</dt>
                <dd className="text-gray-900 font-medium">{job.request.max_seq_length}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">LoRA Enabled</dt>
                <dd className="text-gray-900 font-medium">{job.request.lora_enabled ? 'Yes' : 'No'}</dd>
              </div>
              {job.request.lora_enabled && (
                <>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">LoRA r</dt>
                    <dd className="text-gray-900 font-medium">{job.request.lora_r}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">LoRA Alpha</dt>
                    <dd className="text-gray-900 font-medium">{job.request.lora_alpha}</dd>
                  </div>
                </>
              )}
              <div className="flex justify-between">
                <dt className="text-gray-500">BF16</dt>
                <dd className="text-gray-900 font-medium">{job.request.bf16 ? 'Yes' : 'No'}</dd>
              </div>
            </dl>
          </div>

          <div className="card p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Timeline</h3>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-500">Created</dt>
                <dd className="text-gray-900">
                  {format(new Date(job.created_at), 'MMM d, yyyy HH:mm')}
                </dd>
              </div>
              {job.started_at && (
                <div className="flex justify-between">
                  <dt className="text-gray-500">Started</dt>
                  <dd className="text-gray-900">
                    {format(new Date(job.started_at), 'MMM d, yyyy HH:mm')}
                  </dd>
                </div>
              )}
              {job.completed_at && (
                <div className="flex justify-between">
                  <dt className="text-gray-500">Completed</dt>
                  <dd className="text-gray-900">
                    {format(new Date(job.completed_at), 'MMM d, yyyy HH:mm')}
                  </dd>
                </div>
              )}
            </dl>
          </div>
        </div>
      </div>
    </div>
  )
}
