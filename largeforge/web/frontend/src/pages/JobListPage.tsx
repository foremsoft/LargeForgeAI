import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Plus,
  Search,
  Filter,
  CheckCircle,
  Clock,
  AlertCircle,
  Activity,
  Trash2,
  XCircle,
} from 'lucide-react'
import clsx from 'clsx'
import { useJobs, useCancelJob, useDeleteJob } from '../api/hooks'
import type { TrainingJob, JobStatus } from '../api/types'
import { formatDistanceToNow } from 'date-fns'

const statusConfig: Record<JobStatus, { color: string; bgColor: string; icon: typeof CheckCircle }> = {
  pending: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: Clock },
  running: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Activity },
  completed: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle },
  failed: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle },
  cancelled: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: XCircle },
}

function JobRow({ job, onCancel, onDelete }: {
  job: TrainingJob
  onCancel: (id: string) => void
  onDelete: (id: string) => void
}) {
  const config = statusConfig[job.status]
  const StatusIcon = config.icon
  const progress = job.progress
  const progressPct = progress
    ? Math.round((progress.step / progress.total_steps) * 100)
    : 0

  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap">
        <Link to={`/jobs/${job.job_id}`} className="text-sm font-medium text-primary-600 hover:text-primary-800">
          {job.job_id.slice(0, 8)}...
        </Link>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="text-sm text-gray-900 truncate max-w-xs">
          {job.request.model_path}
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 uppercase">
          {job.training_type}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <span className={clsx(
          'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize',
          config.bgColor,
          config.color
        )}>
          <StatusIcon className="h-3 w-3 mr-1" />
          {job.status}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        {job.status === 'running' && progress ? (
          <div className="w-full">
            <div className="flex justify-between text-xs text-gray-500 mb-1">
              <span>Step {progress.step}/{progress.total_steps}</span>
              <span>{progressPct}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div
                className="bg-primary-600 h-1.5 rounded-full transition-all"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        ) : (
          <span className="text-sm text-gray-500">-</span>
        )}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
        <div className="flex items-center justify-end space-x-2">
          <Link
            to={`/jobs/${job.job_id}`}
            className="text-primary-600 hover:text-primary-900"
          >
            View
          </Link>
          {job.status === 'running' && (
            <button
              onClick={() => onCancel(job.job_id)}
              className="text-yellow-600 hover:text-yellow-900"
            >
              Cancel
            </button>
          )}
          {['completed', 'failed', 'cancelled'].includes(job.status) && (
            <button
              onClick={() => onDelete(job.job_id)}
              className="text-red-600 hover:text-red-900"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}
        </div>
      </td>
    </tr>
  )
}

export default function JobListPage() {
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [search, setSearch] = useState('')
  const { data, isLoading } = useJobs(statusFilter || undefined)
  const cancelJob = useCancelJob()
  const deleteJob = useDeleteJob()

  const jobs = data?.jobs || []
  const filteredJobs = jobs.filter((job) =>
    job.request.model_path.toLowerCase().includes(search.toLowerCase()) ||
    job.job_id.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Training Jobs</h1>
        <Link to="/jobs/new" className="btn btn-primary">
          <Plus className="h-4 w-4 mr-2" />
          New Training
        </Link>
      </div>

      <div className="card">
        <div className="p-4 border-b border-gray-200">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search jobs..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="input pl-10"
              />
            </div>
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="input pl-10 pr-8 appearance-none cursor-pointer"
              >
                <option value="">All Status</option>
                <option value="pending">Pending</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="cancelled">Cancelled</option>
              </select>
            </div>
          </div>
        </div>

        {isLoading ? (
          <div className="p-8 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
            <p className="mt-2 text-gray-500">Loading jobs...</p>
          </div>
        ) : filteredJobs.length === 0 ? (
          <div className="p-8 text-center">
            <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-1">No jobs found</h3>
            <p className="text-gray-500 mb-4">
              {search || statusFilter
                ? 'Try adjusting your filters'
                : 'Get started by creating your first training job'}
            </p>
            {!search && !statusFilter && (
              <Link to="/jobs/new" className="btn btn-primary">
                <Plus className="h-4 w-4 mr-2" />
                New Training
              </Link>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Model
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Progress
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredJobs.map((job) => (
                  <JobRow
                    key={job.job_id}
                    job={job}
                    onCancel={(id) => cancelJob.mutate(id)}
                    onDelete={(id) => deleteJob.mutate(id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
