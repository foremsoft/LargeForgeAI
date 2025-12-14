// Job types
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
export type TrainingType = 'sft' | 'dpo'

export interface TrainingProgress {
  epoch: number
  total_epochs: number
  step: number
  total_steps: number
  loss: number
  learning_rate: number
  elapsed_seconds: number
  eta_seconds: number
  metrics: Record<string, number>
}

export interface TrainingJob {
  job_id: string
  status: JobStatus
  training_type: TrainingType
  created_at: string
  started_at?: string
  completed_at?: string
  progress?: TrainingProgress
  error?: string
  output_path?: string
  request: TrainingJobCreate
}

export interface TrainingJobCreate {
  model_path: string
  dataset_path: string
  training_type: TrainingType
  epochs: number
  learning_rate: number
  batch_size: number
  lora_enabled: boolean
  lora_r: number
  lora_alpha: number
  max_seq_length: number
  output_dir?: string
  gradient_accumulation_steps: number
  warmup_ratio: number
  fp16: boolean
  bf16: boolean
}

// Auth types
export interface User {
  id: string
  username: string
  email: string
  is_active: boolean
  is_admin: boolean
  created_at: string
  last_login?: string
}

export interface Token {
  access_token: string
  token_type: string
  expires_in: number
}

export interface LoginCredentials {
  username: string
  password: string
}

export interface UserCreate {
  username: string
  email: string
  password: string
}

// System types
export interface SystemInfo {
  status: string
  version: string
  uptime_seconds: number
  active_jobs: number
  gpu_available: boolean
  gpu_count: number
  gpu_memory_total_gb: number
  gpu_memory_used_gb: number
  cpu_percent: number
  memory_percent: number
}

export interface GPUInfo {
  id: number
  name: string
  memory_total_gb: number
  memory_used_gb: number
  memory_free_gb: number
  utilization_percent: number
  temperature: number
}

// Model types
export interface ModelInfo {
  name: string
  path: string
  size_gb: number
  architecture: string
  is_local: boolean
  has_adapter: boolean
  base_model?: string
}

// API response types
export interface JobListResponse {
  jobs: TrainingJob[]
  total: number
  page: number
  per_page: number
}

export interface ApiError {
  detail: string
  status_code: number
}

// WebSocket message types
export type WSMessageType = 'progress' | 'status' | 'log' | 'ping' | 'pong'

export interface WSMessage {
  type: WSMessageType
  data: unknown
}

export interface WSProgressMessage extends WSMessage {
  type: 'progress'
  data: TrainingProgress
}

export interface WSStatusMessage extends WSMessage {
  type: 'status'
  data: {
    status: JobStatus
    error?: string
  }
}

export interface WSLogMessage extends WSMessage {
  type: 'log'
  data: {
    line: string
    level: string
  }
}
