import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { get, post, del } from './client'
import type {
  TrainingJob,
  TrainingJobCreate,
  JobListResponse,
  SystemInfo,
  GPUInfo,
  ModelInfo,
  Token,
  LoginCredentials,
  User,
  UserCreate,
} from './types'

// Job hooks
export function useJobs(status?: string, page = 1, perPage = 20) {
  return useQuery({
    queryKey: ['jobs', status, page, perPage],
    queryFn: () =>
      get<JobListResponse>('/jobs', {
        status,
        page,
        per_page: perPage,
      }),
    refetchInterval: 5000, // Refetch every 5 seconds
  })
}

export function useJob(jobId: string) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => get<TrainingJob>(`/jobs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: 2000, // Refetch every 2 seconds for active monitoring
  })
}

export function useCreateJob() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: TrainingJobCreate) => post<TrainingJob>('/jobs', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

export function useCancelJob() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => post<void>(`/jobs/${jobId}/cancel`),
    onSuccess: (_, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['job', jobId] })
    },
  })
}

export function useDeleteJob() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => del<void>(`/jobs/${jobId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

// System hooks
export function useSystemInfo() {
  return useQuery({
    queryKey: ['system', 'info'],
    queryFn: () => get<SystemInfo>('/system/info'),
    refetchInterval: 10000,
  })
}

export function useGPUInfo() {
  return useQuery({
    queryKey: ['system', 'gpu'],
    queryFn: () => get<GPUInfo[]>('/system/gpu'),
    refetchInterval: 5000,
  })
}

// Model hooks
export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => get<ModelInfo[]>('/models'),
    staleTime: 60000, // Models don't change often
  })
}

export function useModel(modelId: string) {
  return useQuery({
    queryKey: ['model', modelId],
    queryFn: () => get<ModelInfo>(`/models/${encodeURIComponent(modelId)}/info`),
    enabled: !!modelId,
  })
}

// Auth hooks
export function useLogin() {
  return useMutation({
    mutationFn: async (credentials: LoginCredentials) => {
      const formData = new URLSearchParams()
      formData.append('username', credentials.username)
      formData.append('password', credentials.password)

      const response = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Login failed')
      }

      return response.json() as Promise<Token>
    },
  })
}

export function useRegister() {
  return useMutation({
    mutationFn: (data: UserCreate) => post<User>('/auth/register/first', data),
  })
}

export function useCurrentUser() {
  return useQuery({
    queryKey: ['user', 'me'],
    queryFn: () => get<User>('/auth/me'),
    retry: false,
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => get<{ status: string }>('/health'),
    refetchInterval: 30000,
  })
}
