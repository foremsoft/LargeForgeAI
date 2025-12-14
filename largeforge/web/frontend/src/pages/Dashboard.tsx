import { Link } from 'react-router-dom'
import {
  Briefcase,
  CheckCircle,
  Clock,
  AlertCircle,
  Plus,
  Cpu,
  HardDrive,
  Activity,
} from 'lucide-react'
import clsx from 'clsx'
import { useJobs, useSystemInfo, useGPUInfo } from '../api/hooks'
import type { TrainingJob, JobStatus } from '../api/types'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

const statusConfig: Record<JobStatus, { color: string; icon: typeof CheckCircle }> = {
  pending: { color: 'text-yellow-500', icon: Clock },
  running: { color: 'text-blue-500', icon: Activity },
  completed: { color: 'text-green-500', icon: CheckCircle },
  failed: { color: 'text-red-500', icon: AlertCircle },
  cancelled: { color: 'text-gray-500', icon: AlertCircle },
}

function StatCard({
  title,
  value,
  icon: Icon,
  color,
}: {
  title: string
  value: string | number
  icon: typeof Briefcase
  color: string
}) {
  return (
    <div className="card p-6">
      <div className="flex items-center">
        <div className={clsx('p-3 rounded-lg', color)}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  )
}

function RecentJobsCard({ jobs }: { jobs: TrainingJob[] }) {
  return (
    <div className="card">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900">Recent Jobs</h3>
          <Link to="/jobs" className="text-sm text-primary-600 hover:text-primary-700">
            View all
          </Link>
        </div>
      </div>
      <div className="divide-y divide-gray-200">
        {jobs.slice(0, 5).map((job) => {
          const config = statusConfig[job.status]
          const StatusIcon = config.icon
          return (
            <Link
              key={job.job_id}
              to={`/jobs/${job.job_id}`}
              className="flex items-center px-6 py-4 hover:bg-gray-50 transition-colors"
            >
              <StatusIcon className={clsx('h-5 w-5 mr-3', config.color)} />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {job.request.model_path.split('/').pop()}
                </p>
                <p className="text-xs text-gray-500">
                  {job.training_type.toUpperCase()} - {job.request.epochs} epochs
                </p>
              </div>
              <div className="text-right">
                <span
                  className={clsx(
                    'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize',
                    job.status === 'running' && 'bg-blue-100 text-blue-800',
                    job.status === 'completed' && 'bg-green-100 text-green-800',
                    job.status === 'failed' && 'bg-red-100 text-red-800',
                    job.status === 'pending' && 'bg-yellow-100 text-yellow-800',
                    job.status === 'cancelled' && 'bg-gray-100 text-gray-800'
                  )}
                >
                  {job.status}
                </span>
                {job.progress && job.status === 'running' && (
                  <p className="text-xs text-gray-500 mt-1">
                    {((job.progress.step / job.progress.total_steps) * 100).toFixed(0)}%
                  </p>
                )}
              </div>
            </Link>
          )
        })}
        {jobs.length === 0 && (
          <div className="px-6 py-8 text-center text-gray-500">
            <Briefcase className="h-8 w-8 mx-auto mb-2 text-gray-400" />
            <p>No training jobs yet</p>
            <Link
              to="/jobs/new"
              className="mt-2 inline-flex items-center text-sm text-primary-600 hover:text-primary-700"
            >
              <Plus className="h-4 w-4 mr-1" />
              Start your first training
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}

function GPUCard() {
  const { data: gpus } = useGPUInfo()

  if (!gpus || gpus.length === 0) {
    return (
      <div className="card p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">GPU Status</h3>
        <p className="text-gray-500">No GPU available</p>
      </div>
    )
  }

  // Mock data for GPU usage chart
  const chartData = Array.from({ length: 20 }, (_, i) => ({
    time: i,
    memory: Math.random() * 30 + 40,
    utilization: Math.random() * 40 + 30,
  }))

  return (
    <div className="card p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">GPU Status</h3>
      <div className="space-y-4">
        {gpus.map((gpu) => (
          <div key={gpu.id} className="border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-900">{gpu.name}</span>
              <span className="text-xs text-gray-500">GPU {gpu.id}</span>
            </div>
            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Memory</span>
                  <span>
                    {gpu.memory_used_gb.toFixed(1)} / {gpu.memory_total_gb.toFixed(1)} GB
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-primary-600 h-2 rounded-full"
                    style={{ width: `${(gpu.memory_used_gb / gpu.memory_total_gb) * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Utilization</span>
                  <span>{gpu.utilization_percent}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${gpu.utilization_percent}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <XAxis dataKey="time" hide />
            <YAxis hide domain={[0, 100]} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="memory"
              stroke="#0284c7"
              strokeWidth={2}
              dot={false}
              name="Memory %"
            />
            <Line
              type="monotone"
              dataKey="utilization"
              stroke="#22c55e"
              strokeWidth={2}
              dot={false}
              name="Utilization %"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { data: jobsData } = useJobs()
  const { data: systemInfo } = useSystemInfo()

  const jobs = jobsData?.jobs || []
  const runningJobs = jobs.filter((j) => j.status === 'running').length
  const completedJobs = jobs.filter((j) => j.status === 'completed').length
  const failedJobs = jobs.filter((j) => j.status === 'failed').length

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <Link to="/jobs/new" className="btn btn-primary">
          <Plus className="h-4 w-4 mr-2" />
          New Training
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Active Jobs"
          value={runningJobs}
          icon={Activity}
          color="bg-blue-500"
        />
        <StatCard
          title="Completed"
          value={completedJobs}
          icon={CheckCircle}
          color="bg-green-500"
        />
        <StatCard
          title="Failed"
          value={failedJobs}
          icon={AlertCircle}
          color="bg-red-500"
        />
        <StatCard
          title="GPU Memory"
          value={
            systemInfo?.gpu_available
              ? `${systemInfo.gpu_memory_used_gb?.toFixed(1) || 0} GB`
              : 'N/A'
          }
          icon={Cpu}
          color="bg-purple-500"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <RecentJobsCard jobs={jobs} />
        <GPUCard />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="card p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Quick Actions</h3>
          <div className="space-y-3">
            <Link
              to="/jobs/new"
              className="flex items-center p-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
            >
              <Plus className="h-5 w-5 text-primary-600 mr-3" />
              <span className="text-sm font-medium text-gray-900">Start New Training</span>
            </Link>
            <Link
              to="/models"
              className="flex items-center p-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
            >
              <HardDrive className="h-5 w-5 text-primary-600 mr-3" />
              <span className="text-sm font-medium text-gray-900">Browse Models</span>
            </Link>
          </div>
        </div>

        <div className="card p-6 lg:col-span-2">
          <h3 className="text-lg font-medium text-gray-900 mb-4">System Status</h3>
          {systemInfo ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500">Status</p>
                <p className="text-lg font-semibold text-green-600 capitalize">
                  {systemInfo.status}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Version</p>
                <p className="text-lg font-semibold text-gray-900">{systemInfo.version}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">CPU Usage</p>
                <p className="text-lg font-semibold text-gray-900">
                  {systemInfo.cpu_percent?.toFixed(1) || 0}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Memory Usage</p>
                <p className="text-lg font-semibold text-gray-900">
                  {systemInfo.memory_percent?.toFixed(1) || 0}%
                </p>
              </div>
            </div>
          ) : (
            <p className="text-gray-500">Loading system information...</p>
          )}
        </div>
      </div>
    </div>
  )
}
