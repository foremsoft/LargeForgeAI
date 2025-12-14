import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Box,
  Search,
  ExternalLink,
  HardDrive,
  Cloud,
  CheckCircle,
  AlertCircle,
} from 'lucide-react'
import clsx from 'clsx'
import { useModels } from '../api/hooks'
import type { ModelInfo } from '../api/types'

function ModelCard({ model }: { model: ModelInfo }) {
  return (
    <div className="card p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-center">
          {model.is_local ? (
            <HardDrive className="h-8 w-8 text-primary-600" />
          ) : (
            <Cloud className="h-8 w-8 text-gray-400" />
          )}
          <div className="ml-4">
            <h3 className="text-lg font-medium text-gray-900">{model.name}</h3>
            <p className="text-sm text-gray-500 truncate max-w-xs">{model.path}</p>
          </div>
        </div>
        {model.has_adapter && (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
            LoRA
          </span>
        )}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Architecture</span>
          <p className="text-gray-900 font-medium">{model.architecture || 'Unknown'}</p>
        </div>
        <div>
          <span className="text-gray-500">Size</span>
          <p className="text-gray-900 font-medium">
            {model.size_gb ? `${model.size_gb.toFixed(1)} GB` : 'N/A'}
          </p>
        </div>
      </div>

      {model.base_model && (
        <div className="mt-3 text-sm">
          <span className="text-gray-500">Base Model:</span>
          <span className="ml-2 text-gray-900">{model.base_model}</span>
        </div>
      )}

      <div className="mt-4 flex items-center justify-between pt-4 border-t border-gray-200">
        <span className={clsx(
          'inline-flex items-center text-xs',
          model.is_local ? 'text-green-600' : 'text-gray-500'
        )}>
          {model.is_local ? (
            <>
              <CheckCircle className="h-4 w-4 mr-1" />
              Local
            </>
          ) : (
            <>
              <Cloud className="h-4 w-4 mr-1" />
              Remote
            </>
          )}
        </span>
        <div className="flex space-x-2">
          <Link
            to={`/jobs/new?model=${encodeURIComponent(model.path)}`}
            className="btn btn-primary text-xs py-1 px-3"
          >
            Train
          </Link>
          {!model.is_local && (
            <a
              href={`https://huggingface.co/${model.path}`}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary text-xs py-1 px-3"
            >
              <ExternalLink className="h-3 w-3 mr-1" />
              View
            </a>
          )}
        </div>
      </div>
    </div>
  )
}

export default function ModelsPage() {
  const [searchParams] = useSearchParams()
  const highlightPath = searchParams.get('path')
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<'all' | 'local' | 'remote'>('all')

  const { data: models, isLoading, error } = useModels()

  const filteredModels = (models || []).filter((model) => {
    const matchesSearch =
      model.name.toLowerCase().includes(search.toLowerCase()) ||
      model.path.toLowerCase().includes(search.toLowerCase())

    const matchesFilter =
      filter === 'all' ||
      (filter === 'local' && model.is_local) ||
      (filter === 'remote' && !model.is_local)

    return matchesSearch && matchesFilter
  })

  // Add some default models if none exist
  const displayModels = filteredModels.length > 0 ? filteredModels : [
    {
      name: 'Qwen2.5-7B-Instruct',
      path: 'Qwen/Qwen2.5-7B-Instruct',
      size_gb: 14.5,
      architecture: 'Qwen2',
      is_local: false,
      has_adapter: false,
    },
    {
      name: 'Qwen2.5-3B-Instruct',
      path: 'Qwen/Qwen2.5-3B-Instruct',
      size_gb: 6.2,
      architecture: 'Qwen2',
      is_local: false,
      has_adapter: false,
    },
    {
      name: 'Llama-3.2-3B-Instruct',
      path: 'meta-llama/Llama-3.2-3B-Instruct',
      size_gb: 6.4,
      architecture: 'Llama',
      is_local: false,
      has_adapter: false,
    },
  ] as ModelInfo[]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Models</h1>
      </div>

      {/* Search and Filter */}
      <div className="card p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="input pl-10"
            />
          </div>
          <div className="flex gap-2">
            {(['all', 'local', 'remote'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={clsx(
                  'px-4 py-2 rounded-md text-sm font-medium transition-colors capitalize',
                  filter === f
                    ? 'bg-primary-100 text-primary-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                )}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Highlighted Model */}
      {highlightPath && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-start">
          <CheckCircle className="h-5 w-5 text-green-500 mr-3 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-green-800">Model Output</h4>
            <p className="text-sm text-green-600 mt-1">
              Your trained model is available at: <code className="bg-green-100 px-1 rounded">{highlightPath}</code>
            </p>
          </div>
        </div>
      )}

      {/* Models Grid */}
      {isLoading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-2 text-gray-500">Loading models...</p>
        </div>
      ) : error ? (
        <div className="text-center py-12">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-1">Error loading models</h3>
          <p className="text-gray-500">Please try again later</p>
        </div>
      ) : displayModels.length === 0 ? (
        <div className="text-center py-12">
          <Box className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-1">No models found</h3>
          <p className="text-gray-500">Try adjusting your search or filters</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {displayModels.map((model) => (
            <ModelCard key={model.path} model={model} />
          ))}
        </div>
      )}

      {/* Info Section */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">About Models</h3>
        <div className="text-sm text-gray-600 space-y-2">
          <p>
            <strong>Local models</strong> are stored on your system and ready for immediate use.
          </p>
          <p>
            <strong>Remote models</strong> will be downloaded from HuggingFace when you start training.
          </p>
          <p>
            Models with the <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">LoRA</span> badge
            are fine-tuned adapters that can be merged with their base model.
          </p>
        </div>
      </div>
    </div>
  )
}
