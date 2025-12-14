import { useEffect, useRef, useState, useCallback } from 'react'
import type { TrainingProgress, JobStatus, WSMessage } from './types'
import { getAccessToken } from './client'

type OnProgressCallback = (progress: TrainingProgress) => void
type OnStatusCallback = (status: JobStatus, error?: string) => void
type OnLogCallback = (line: string, level: string) => void

interface UseJobWebSocketOptions {
  onProgress?: OnProgressCallback
  onStatus?: OnStatusCallback
  onLog?: OnLogCallback
}

interface UseJobWebSocketReturn {
  isConnected: boolean
  progress: TrainingProgress | null
  status: JobStatus | null
  logs: Array<{ line: string; level: string }>
  error: string | null
}

export function useJobWebSocket(
  jobId: string | null,
  options: UseJobWebSocketOptions = {}
): UseJobWebSocketReturn {
  const { onProgress, onStatus, onLog } = options
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)

  const [isConnected, setIsConnected] = useState(false)
  const [progress, setProgress] = useState<TrainingProgress | null>(null)
  const [status, setStatus] = useState<JobStatus | null>(null)
  const [logs, setLogs] = useState<Array<{ line: string; level: string }>>([])
  const [error, setError] = useState<string | null>(null)

  const connect = useCallback(() => {
    if (!jobId) return

    const token = getAccessToken()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/jobs/${jobId}${token ? `?token=${token}` : ''}`

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      setError(null)
    }

    ws.onclose = () => {
      setIsConnected(false)
      // Attempt to reconnect after 3 seconds
      reconnectTimeoutRef.current = window.setTimeout(() => {
        connect()
      }, 3000)
    }

    ws.onerror = () => {
      setError('WebSocket connection error')
    }

    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data)

        switch (message.type) {
          case 'progress': {
            const progressData = message.data as TrainingProgress
            setProgress(progressData)
            onProgress?.(progressData)
            break
          }
          case 'status': {
            const statusData = message.data as { status: JobStatus; error?: string }
            setStatus(statusData.status)
            onStatus?.(statusData.status, statusData.error)
            if (statusData.error) {
              setError(statusData.error)
            }
            break
          }
          case 'log': {
            const logData = message.data as { line: string; level: string }
            setLogs((prev) => [...prev.slice(-999), logData])
            onLog?.(logData.line, logData.level)
            break
          }
          case 'ping':
            // Respond with pong
            ws.send(JSON.stringify({ type: 'pong' }))
            break
        }
      } catch {
        console.error('Failed to parse WebSocket message')
      }
    }
  }, [jobId, onProgress, onStatus, onLog])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connect])

  return {
    isConnected,
    progress,
    status,
    logs,
    error,
  }
}

// Simple reconnecting WebSocket wrapper
export class ReconnectingWebSocket {
  private url: string
  private ws: WebSocket | null = null
  private reconnectTimeout: number | null = null
  private listeners: Map<string, Set<(data: unknown) => void>> = new Map()

  constructor(url: string) {
    this.url = url
    this.connect()
  }

  private connect() {
    this.ws = new WebSocket(this.url)

    this.ws.onopen = () => {
      this.emit('open', null)
    }

    this.ws.onclose = () => {
      this.emit('close', null)
      this.reconnectTimeout = window.setTimeout(() => this.connect(), 3000)
    }

    this.ws.onerror = (error) => {
      this.emit('error', error)
    }

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        this.emit('message', data)
      } catch {
        this.emit('message', event.data)
      }
    }
  }

  on(event: string, callback: (data: unknown) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set())
    }
    this.listeners.get(event)!.add(callback)
  }

  off(event: string, callback: (data: unknown) => void) {
    this.listeners.get(event)?.delete(callback)
  }

  private emit(event: string, data: unknown) {
    this.listeners.get(event)?.forEach((callback) => callback(data))
  }

  send(data: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }

  close() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
    }
    this.ws?.close()
  }
}
