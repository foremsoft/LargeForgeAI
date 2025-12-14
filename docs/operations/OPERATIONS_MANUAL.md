# LargeForgeAI Operations Manual

## Overview

This manual provides comprehensive guidance for operating LargeForgeAI in production environments. It covers deployment, monitoring, maintenance, and troubleshooting procedures.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Deployment](#deployment)
3. [Monitoring](#monitoring)
4. [Scaling](#scaling)
5. [Backup and Recovery](#backup-and-recovery)
6. [Maintenance](#maintenance)
7. [Security Operations](#security-operations)
8. [Performance Tuning](#performance-tuning)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Load Balancer                                │
│                      (nginx/HAProxy/ALB)                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                         Router Service                               │
│                    (Query Classification)                            │
│                         Port: 8080                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│  Inference #1   │   │  Inference #2   │   │  Inference #3   │
│  (Code Expert)  │   │ (Write Expert)  │   │(General Expert) │
│   Port: 8001    │   │   Port: 8002    │   │   Port: 8003    │
│   GPU: 0        │   │   GPU: 1        │   │   GPU: 2        │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      Monitoring Stack                                │
│              Prometheus │ Grafana │ AlertManager                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Purpose | Scaling |
|-----------|---------|---------|
| Load Balancer | Traffic distribution, SSL termination | Horizontal |
| Router Service | Query classification, expert selection | Horizontal |
| Inference Service | Model serving, token generation | Horizontal (per GPU) |
| Prometheus | Metrics collection | Vertical |
| Grafana | Visualization, dashboards | Vertical |
| AlertManager | Alert routing | Vertical |

---

## Deployment

### Prerequisites

1. **Infrastructure**
   - Kubernetes cluster (1.24+) or Docker Swarm
   - NVIDIA GPU operator installed
   - Persistent storage (NFS/EBS/GCS)
   - Load balancer (nginx/HAProxy/cloud LB)

2. **Resources per node**
   - 4+ CPU cores
   - 32GB+ RAM
   - 100GB+ SSD
   - NVIDIA GPU (A100 recommended)

### Kubernetes Deployment

#### 1. Create Namespace

```bash
kubectl create namespace largeforge
kubectl label namespace largeforge istio-injection=enabled  # If using Istio
```

#### 2. Deploy Secrets

```bash
# Create secrets
kubectl create secret generic largeforge-secrets \
  --namespace largeforge \
  --from-literal=api-key=$LARGEFORGE_API_KEY \
  --from-literal=hf-token=$HF_TOKEN

# Create TLS secret (if not using cert-manager)
kubectl create secret tls largeforge-tls \
  --namespace largeforge \
  --cert=./tls.crt \
  --key=./tls.key
```

#### 3. Deploy with Helm

```bash
# Add Helm repository
helm repo add largeforge https://charts.largeforge.ai
helm repo update

# Install
helm install largeforge largeforge/largeforge \
  --namespace largeforge \
  --values values.yaml
```

**values.yaml:**

```yaml
# Inference service
inference:
  replicas: 3
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: 80Gi
    requests:
      nvidia.com/gpu: 1
      memory: 40Gi

  models:
    - name: code-expert
      path: s3://models/code-expert
      gpu: 0
    - name: write-expert
      path: s3://models/write-expert
      gpu: 1
    - name: general
      path: s3://models/general
      gpu: 2

  config:
    backend: vllm
    maxModelLen: 4096
    gpuMemoryUtilization: 0.9

# Router service
router:
  replicas: 2
  resources:
    limits:
      memory: 4Gi
      cpu: 2
    requests:
      memory: 2Gi
      cpu: 1

  config:
    classifierType: hybrid
    defaultExpert: general

# Monitoring
monitoring:
  enabled: true
  prometheus:
    retention: 30d
    storage: 100Gi
  grafana:
    enabled: true
    adminPassword: ${GRAFANA_PASSWORD}

# Ingress
ingress:
  enabled: true
  className: nginx
  hosts:
    - api.largeforge.example.com
  tls:
    - secretName: largeforge-tls
      hosts:
        - api.largeforge.example.com

# Autoscaling
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilization: 70
  targetMemoryUtilization: 80
```

#### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n largeforge

# Check services
kubectl get svc -n largeforge

# Check ingress
kubectl get ingress -n largeforge

# View logs
kubectl logs -f deployment/largeforge-inference -n largeforge

# Port forward for local testing
kubectl port-forward svc/largeforge-router 8080:80 -n largeforge
```

### Docker Compose Deployment

For smaller deployments or development:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  inference-code:
    image: largeforgeai/largeforge:latest
    command: serve inference --model /models/code-expert --port 8001
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  inference-write:
    image: largeforgeai/largeforge:latest
    command: serve inference --model /models/write-expert --port 8002
    environment:
      - CUDA_VISIBLE_DEVICES=1
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  router:
    image: largeforgeai/largeforge:latest
    command: serve router --config /config/router.yaml
    ports:
      - "8080:8080"
    volumes:
      - ./config:/config:ro
    depends_on:
      inference-code:
        condition: service_healthy
      inference-write:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - router

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  prometheus_data:
  grafana_data:
```

---

## Monitoring

### Key Metrics

#### Inference Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `inference_requests_total` | Total requests | N/A |
| `inference_latency_seconds` | Request latency | p99 > 5s |
| `tokens_generated_total` | Tokens generated | N/A |
| `tokens_per_second` | Generation speed | < 20 tok/s |
| `gpu_memory_used_bytes` | GPU memory usage | > 95% |
| `active_requests` | Concurrent requests | > 100 |
| `queue_length` | Pending requests | > 50 |
| `error_rate` | Failed requests % | > 1% |

#### Router Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `routing_decisions_total` | Routing decisions | N/A |
| `routing_latency_seconds` | Classification time | p99 > 100ms |
| `expert_availability` | Expert health | any < 1 |
| `fallback_rate` | Fallback usage | > 10% |

#### System Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `cpu_usage_percent` | CPU utilization | > 90% |
| `memory_usage_percent` | Memory utilization | > 90% |
| `disk_usage_percent` | Disk utilization | > 85% |
| `network_errors_total` | Network errors | > 0 |

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - '/etc/prometheus/rules/*.yml'

scrape_configs:
  - job_name: 'largeforge-inference'
    static_configs:
      - targets: ['inference-code:8001', 'inference-write:8002']
    metrics_path: /metrics

  - job_name: 'largeforge-router'
    static_configs:
      - targets: ['router:8080']
    metrics_path: /metrics

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['nvidia-exporter:9400']
```

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: largeforge
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "p99 latency is {{ $value }}s"

      - alert: HighErrorRate
        expr: rate(inference_requests_total{status="error"}[5m]) / rate(inference_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: GPUMemoryHigh
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage high"
          description: "GPU {{ $labels.device }} at {{ $value | humanizePercentage }}"

      - alert: ExpertDown
        expr: up{job="largeforge-inference"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Expert service down"
          description: "{{ $labels.instance }} is not responding"

      - alert: QueueBacklog
        expr: queue_length > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Request queue backlog"
          description: "Queue has {{ $value }} pending requests"
```

### Grafana Dashboards

Import these dashboards:
- **LargeForgeAI Overview** - High-level system health
- **Inference Performance** - Latency, throughput, errors
- **GPU Monitoring** - GPU utilization, memory, temperature
- **Expert Routing** - Routing decisions, expert load

---

## Scaling

### Horizontal Scaling

#### Manual Scaling

```bash
# Kubernetes
kubectl scale deployment largeforge-inference --replicas=5 -n largeforge

# Docker Compose
docker-compose up -d --scale inference=5
```

#### Autoscaling (Kubernetes)

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: largeforge-inference-hpa
  namespace: largeforge
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: largeforge-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: inference_requests_per_second
        target:
          type: AverageValue
          averageValue: 100
```

### Vertical Scaling

For inference services, vertical scaling means:
- More GPU memory = Larger models or more concurrent requests
- More GPUs = Tensor parallelism for larger models

```yaml
# For larger models (tensor parallelism)
inference:
  config:
    tensorParallelSize: 2  # Use 2 GPUs per replica
  resources:
    limits:
      nvidia.com/gpu: 2
```

### Multi-Region Deployment

```
                    Global Load Balancer
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    US-EAST           US-WEST           EU-WEST
    ┌─────┐           ┌─────┐           ┌─────┐
    │ LF  │           │ LF  │           │ LF  │
    │Cluster│         │Cluster│         │Cluster│
    └─────┘           └─────┘           └─────┘
         │                 │                 │
    ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
    │ Models  │       │ Models  │       │ Models  │
    │  (S3)   │◄──────┤ (Sync)  ├──────►│ (GCS)   │
    └─────────┘       └─────────┘       └─────────┘
```

---

## Backup and Recovery

### What to Backup

| Data | Frequency | Retention | Method |
|------|-----------|-----------|--------|
| Model weights | On change | 5 versions | S3/GCS |
| Configuration | Daily | 30 days | Git + S3 |
| Prometheus data | Hourly | 30 days | S3 |
| Secrets | On change | 10 versions | Vault |

### Backup Procedures

#### Model Backup

```bash
# Backup model to S3
aws s3 sync ./models s3://backups/models/$(date +%Y%m%d)/

# Automated backup script
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=/models
BACKUP_BUCKET=s3://backups/models

# Create backup
aws s3 sync $MODEL_DIR $BACKUP_BUCKET/$BACKUP_DATE/

# Keep only last 5 backups
aws s3 ls $BACKUP_BUCKET/ | head -n -5 | awk '{print $2}' | \
  xargs -I {} aws s3 rm --recursive $BACKUP_BUCKET/{}
```

#### Configuration Backup

```bash
# Backup Kubernetes resources
kubectl get all,configmap,secret -n largeforge -o yaml > backup.yaml

# Backup Helm values
helm get values largeforge -n largeforge > values-backup.yaml
```

### Recovery Procedures

#### Model Recovery

```bash
# List available backups
aws s3 ls s3://backups/models/

# Restore specific version
aws s3 sync s3://backups/models/20241201_120000/ ./models/

# Restart services to load new models
kubectl rollout restart deployment/largeforge-inference -n largeforge
```

#### Full Cluster Recovery

```bash
# 1. Recreate namespace
kubectl create namespace largeforge

# 2. Restore secrets from Vault
vault read -format=json secret/largeforge | kubectl apply -f -

# 3. Deploy with Helm
helm install largeforge largeforge/largeforge \
  --namespace largeforge \
  --values values-backup.yaml

# 4. Restore models
aws s3 sync s3://backups/models/latest/ /models/

# 5. Verify deployment
kubectl get pods -n largeforge
largeforge health check --all
```

---

## Maintenance

### Routine Maintenance

#### Daily

- [ ] Review error logs
- [ ] Check alert status
- [ ] Verify backup completion
- [ ] Review resource utilization

#### Weekly

- [ ] Review performance trends
- [ ] Check disk space
- [ ] Update security patches
- [ ] Review access logs

#### Monthly

- [ ] Model performance review
- [ ] Capacity planning
- [ ] Security audit
- [ ] Documentation update

### Rolling Updates

```bash
# Update with zero downtime
kubectl set image deployment/largeforge-inference \
  inference=largeforgeai/largeforge:1.1.0 \
  -n largeforge

# Watch rollout status
kubectl rollout status deployment/largeforge-inference -n largeforge

# Rollback if issues
kubectl rollout undo deployment/largeforge-inference -n largeforge
```

### Model Updates

```bash
# 1. Upload new model
aws s3 sync ./new-model s3://models/code-expert-v2/

# 2. Update configuration
kubectl edit configmap largeforge-config -n largeforge
# Change model path to s3://models/code-expert-v2/

# 3. Rolling restart
kubectl rollout restart deployment/largeforge-inference -n largeforge

# 4. Verify new model
curl http://api.largeforge.ai/v1/models | jq .
```

---

## Security Operations

### Access Control

```yaml
# RBAC for Kubernetes
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: largeforge-operator
  namespace: largeforge
rules:
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: largeforge-operator-binding
  namespace: largeforge
subjects:
  - kind: Group
    name: operators
roleRef:
  kind: Role
  name: largeforge-operator
  apiGroup: rbac.authorization.k8s.io
```

### API Key Rotation

```bash
# 1. Generate new key
NEW_KEY=$(openssl rand -hex 32)

# 2. Add to secrets (both old and new active)
kubectl patch secret largeforge-secrets -n largeforge \
  -p '{"data":{"api-key-new":"'"$(echo -n $NEW_KEY | base64)"'"}}'

# 3. Update client applications

# 4. Remove old key after migration
kubectl patch secret largeforge-secrets -n largeforge \
  -p '{"data":{"api-key":null}}'
```

### Security Incident Response

1. **Detection** - Alert triggers or manual discovery
2. **Containment** - Isolate affected systems
3. **Investigation** - Review logs, identify scope
4. **Eradication** - Remove threat
5. **Recovery** - Restore services
6. **Lessons Learned** - Update procedures

---

## Performance Tuning

### GPU Optimization

```yaml
# vLLM tuning for A100
vllm:
  tensorParallelSize: 1
  maxModelLen: 8192
  gpuMemoryUtilization: 0.92
  enforceEager: false
  enablePrefixCaching: true
  maxNumBatchedTokens: 32768
  maxNumSeqs: 512
```

### Request Optimization

```yaml
# Batching configuration
batching:
  maxBatchSize: 64
  maxWaitTime: 50ms  # Wait up to 50ms to form batch
  dynamicBatching: true
```

### Memory Optimization

```bash
# Monitor GPU memory fragmentation
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
```

---

## Appendix: Useful Commands

```bash
# Check system status
largeforge status --all

# View real-time logs
kubectl logs -f deployment/largeforge-inference -n largeforge

# Check GPU usage
nvidia-smi -l 1

# Test endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","prompt":"hello","max_tokens":10}'

# Prometheus query examples
# Request rate: rate(inference_requests_total[5m])
# p99 latency: histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m]))
# Error rate: rate(inference_requests_total{status="error"}[5m]) / rate(inference_requests_total[5m])
```

---

*For incident-specific procedures, see the [Runbook](./RUNBOOK.md).*
