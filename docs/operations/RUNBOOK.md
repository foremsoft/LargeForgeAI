# LargeForgeAI Runbook

Incident response and operational procedures for on-call engineers.

---

## Quick Reference

### Emergency Contacts

| Role | Contact | Escalation Time |
|------|---------|-----------------|
| On-Call Engineer | PagerDuty | Immediate |
| Platform Lead | @platform-lead | 15 min |
| Engineering Manager | @eng-manager | 30 min |
| VP Engineering | @vp-eng | Critical only |

### Service Endpoints

| Service | Production | Staging |
|---------|------------|---------|
| API Gateway | api.largeforge.ai | staging-api.largeforge.ai |
| Grafana | grafana.largeforge.ai | - |
| Prometheus | prometheus.largeforge.ai | - |
| AlertManager | alerts.largeforge.ai | - |

### Quick Commands

```bash
# Check all services
kubectl get pods -n largeforge

# View logs
kubectl logs -f deployment/largeforge-inference -n largeforge

# Restart service
kubectl rollout restart deployment/largeforge-inference -n largeforge

# Scale up
kubectl scale deployment/largeforge-inference --replicas=5 -n largeforge

# Check GPU status
nvidia-smi
```

---

## Incident Response Procedures

### INC-001: High Latency

**Severity:** P2
**Alert:** `HighLatency` - p99 latency > 5s

**Symptoms:**
- Slow response times
- User complaints
- Timeouts in client applications

**Diagnosis:**

```bash
# 1. Check current latency
curl -w "\nTime: %{time_total}s\n" \
  http://localhost:8000/health

# 2. Check Prometheus
# Query: histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m]))

# 3. Check queue length
curl http://localhost:8000/metrics | grep queue_length

# 4. Check GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

# 5. Check for OOM in logs
kubectl logs deployment/largeforge-inference -n largeforge | grep -i "out of memory"
```

**Resolution:**

1. **If queue is backed up:**
   ```bash
   # Scale up inference replicas
   kubectl scale deployment/largeforge-inference --replicas=+2 -n largeforge
   ```

2. **If GPU memory is high:**
   ```bash
   # Reduce batch size via config
   kubectl edit configmap largeforge-config -n largeforge
   # Set maxNumSeqs: 128 (lower value)

   # Rolling restart
   kubectl rollout restart deployment/largeforge-inference -n largeforge
   ```

3. **If single slow requests:**
   - Check for unusually long prompts
   - Review max_tokens settings

**Escalation:** If latency persists > 30 minutes, escalate to Platform Lead.

---

### INC-002: High Error Rate

**Severity:** P1
**Alert:** `HighErrorRate` - Error rate > 1%

**Symptoms:**
- HTTP 500 errors
- Client retries increasing
- Error rate dashboard showing spikes

**Diagnosis:**

```bash
# 1. Check error breakdown
curl http://localhost:8000/metrics | grep 'status="error"'

# 2. View error logs
kubectl logs deployment/largeforge-inference -n largeforge | grep -i error | tail -50

# 3. Check specific error types
kubectl logs deployment/largeforge-inference -n largeforge | \
  grep -E "(CUDA|OOM|timeout|connection)" | tail -20

# 4. Check pod health
kubectl describe pod -l app=largeforge-inference -n largeforge
```

**Resolution:**

1. **If CUDA/GPU errors:**
   ```bash
   # Check GPU health
   nvidia-smi -q | grep -A 5 "GPU Error"

   # If GPU issues, restart affected pods
   kubectl delete pod <pod-name> -n largeforge
   ```

2. **If OOM errors:**
   ```bash
   # Reduce memory pressure
   kubectl edit configmap largeforge-config -n largeforge
   # Reduce gpuMemoryUtilization to 0.85

   kubectl rollout restart deployment/largeforge-inference -n largeforge
   ```

3. **If connection errors:**
   ```bash
   # Check network
   kubectl exec -it <pod-name> -n largeforge -- ping <target>

   # Check service endpoints
   kubectl get endpoints -n largeforge
   ```

**Escalation:** If error rate > 5% for > 10 minutes, escalate immediately.

---

### INC-003: Service Down

**Severity:** P0
**Alert:** `ExpertDown` - Service not responding

**Symptoms:**
- Health check failures
- 503 Service Unavailable
- No pods running

**Diagnosis:**

```bash
# 1. Check pod status
kubectl get pods -n largeforge

# 2. Check events
kubectl get events -n largeforge --sort-by='.lastTimestamp' | tail -20

# 3. Check pod details
kubectl describe pod -l app=largeforge-inference -n largeforge

# 4. Check node status
kubectl get nodes
```

**Resolution:**

1. **If pods are CrashLoopBackOff:**
   ```bash
   # Check logs for crash reason
   kubectl logs <pod-name> -n largeforge --previous

   # Common causes:
   # - OOM: Increase memory limits
   # - Model not found: Check model path
   # - GPU unavailable: Check node GPU status
   ```

2. **If pods are Pending:**
   ```bash
   # Check resources
   kubectl describe pod <pod-name> -n largeforge | grep -A 10 Events

   # If GPU unavailable, check node
   kubectl describe node <node-name> | grep nvidia
   ```

3. **If no pods exist:**
   ```bash
   # Check deployment
   kubectl get deployment largeforge-inference -n largeforge -o yaml

   # Force recreation
   kubectl rollout restart deployment/largeforge-inference -n largeforge
   ```

**Escalation:** Immediate escalation to Platform Lead.

---

### INC-004: GPU Memory Exhaustion

**Severity:** P2
**Alert:** `GPUMemoryHigh` - GPU memory > 95%

**Symptoms:**
- OOM errors in logs
- Increasing error rate
- Degraded performance

**Diagnosis:**

```bash
# 1. Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 2. Check memory breakdown
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 3. Check for memory leaks (growing over time)
watch -n 5 nvidia-smi --query-gpu=memory.used --format=csv
```

**Resolution:**

1. **Immediate relief:**
   ```bash
   # Reduce concurrent requests
   kubectl edit configmap largeforge-config -n largeforge
   # Set maxNumSeqs: 64

   kubectl rollout restart deployment/largeforge-inference -n largeforge
   ```

2. **If persistent issue:**
   ```bash
   # Enable memory optimization
   # Set in environment:
   # PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

   # Consider model quantization
   largeforge quantize awq --model <path> --output <new-path>
   ```

---

### INC-005: Routing Failures

**Severity:** P2
**Alert:** High fallback rate or routing errors

**Symptoms:**
- All requests going to default expert
- Classification errors
- High routing latency

**Diagnosis:**

```bash
# 1. Check router health
curl http://localhost:8080/health

# 2. Check expert availability
curl http://localhost:8080/experts | jq '.experts[] | {name, status}'

# 3. Check routing metrics
curl http://localhost:8080/metrics | grep routing

# 4. Check router logs
kubectl logs deployment/largeforge-router -n largeforge | tail -50
```

**Resolution:**

1. **If expert unreachable:**
   ```bash
   # Check expert endpoint
   curl http://<expert-endpoint>/health

   # If down, see INC-003 for service recovery
   ```

2. **If classifier failing:**
   ```bash
   # Check classifier model
   kubectl exec -it <router-pod> -n largeforge -- \
     python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"

   # Restart router
   kubectl rollout restart deployment/largeforge-router -n largeforge
   ```

---

### INC-006: Certificate Expiry

**Severity:** P1
**Alert:** Certificate expiring in < 7 days

**Symptoms:**
- TLS handshake failures
- Browser security warnings

**Diagnosis:**

```bash
# Check certificate expiry
echo | openssl s_client -connect api.largeforge.ai:443 2>/dev/null | \
  openssl x509 -noout -dates
```

**Resolution:**

1. **If using cert-manager:**
   ```bash
   # Check certificate status
   kubectl get certificate -n largeforge

   # Force renewal
   kubectl delete certificate largeforge-tls -n largeforge
   # cert-manager will recreate
   ```

2. **If manual certificates:**
   ```bash
   # Update secret with new cert
   kubectl create secret tls largeforge-tls \
     --cert=new-cert.pem \
     --key=new-key.pem \
     -n largeforge \
     --dry-run=client -o yaml | kubectl apply -f -

   # Restart ingress controller
   kubectl rollout restart deployment/nginx-ingress -n ingress-nginx
   ```

---

### INC-007: Disk Space Low

**Severity:** P2
**Alert:** Disk usage > 85%

**Symptoms:**
- Checkpoint saves failing
- Log rotation issues
- Performance degradation

**Diagnosis:**

```bash
# Check disk usage
df -h

# Find large files
du -sh /* | sort -rh | head -20

# Check model cache
du -sh ~/.cache/huggingface/

# Check logs
du -sh /var/log/largeforge/
```

**Resolution:**

1. **Clear model cache:**
   ```bash
   # Remove old model versions
   huggingface-cli cache purge
   ```

2. **Rotate logs:**
   ```bash
   # Force log rotation
   logrotate -f /etc/logrotate.d/largeforge
   ```

3. **Clean old checkpoints:**
   ```bash
   # Keep only last 3 checkpoints
   ls -t ./checkpoints/ | tail -n +4 | xargs rm -rf
   ```

---

## Maintenance Procedures

### MAINT-001: Rolling Update

```bash
# 1. Check current state
kubectl get pods -n largeforge
kubectl rollout history deployment/largeforge-inference -n largeforge

# 2. Apply update
kubectl set image deployment/largeforge-inference \
  inference=largeforgeai/largeforge:1.1.0 \
  -n largeforge

# 3. Monitor rollout
kubectl rollout status deployment/largeforge-inference -n largeforge

# 4. If issues, rollback
kubectl rollout undo deployment/largeforge-inference -n largeforge
```

### MAINT-002: Model Update

```bash
# 1. Upload new model
aws s3 sync ./new-model s3://models/expert-v2/

# 2. Verify upload
aws s3 ls s3://models/expert-v2/

# 3. Update config (gradually)
# First update 1 pod to test
kubectl edit deployment largeforge-inference -n largeforge
# Update one replica's model path

# 4. Verify new model works
curl http://localhost:8000/v1/models | jq .

# 5. Roll out to all replicas
kubectl edit configmap largeforge-config -n largeforge
kubectl rollout restart deployment/largeforge-inference -n largeforge
```

### MAINT-003: Scale Up for High Traffic

```bash
# 1. Check current capacity
kubectl get pods -n largeforge
kubectl top pods -n largeforge

# 2. Scale up
kubectl scale deployment/largeforge-inference --replicas=10 -n largeforge

# 3. Verify scaling
kubectl get pods -n largeforge -w

# 4. Monitor capacity
watch -n 10 'kubectl top pods -n largeforge'
```

### MAINT-004: Drain Node for Maintenance

```bash
# 1. Cordon node (prevent new pods)
kubectl cordon <node-name>

# 2. Drain pods (with grace period)
kubectl drain <node-name> \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --grace-period=300

# 3. Verify pods migrated
kubectl get pods -n largeforge -o wide

# 4. Perform maintenance on node

# 5. Uncordon when ready
kubectl uncordon <node-name>
```

---

## Recovery Procedures

### REC-001: Restore from Backup

```bash
# 1. List available backups
aws s3 ls s3://backups/models/

# 2. Choose backup timestamp
BACKUP_DATE=20241201_120000

# 3. Download models
aws s3 sync s3://backups/models/$BACKUP_DATE/ ./models/

# 4. Verify model integrity
largeforge model verify ./models/

# 5. Deploy
kubectl rollout restart deployment/largeforge-inference -n largeforge

# 6. Verify functionality
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","prompt":"hello","max_tokens":10}'
```

### REC-002: Cluster Recovery

```bash
# 1. Verify cluster access
kubectl cluster-info

# 2. Recreate namespace if needed
kubectl create namespace largeforge

# 3. Restore secrets
vault read -format=json secret/largeforge | \
  jq -r '.data | to_entries[] | "kubectl create secret generic " + .key + " --from-literal=value=" + .value' | \
  bash

# 4. Deploy via Helm
helm install largeforge largeforge/largeforge \
  --namespace largeforge \
  --values values-production.yaml

# 5. Wait for rollout
kubectl rollout status deployment/largeforge-inference -n largeforge --timeout=600s

# 6. Verify all services
largeforge health check --all
```

---

## Post-Incident Procedures

### After Every Incident

1. **Update Status Page**
   - Mark incident resolved
   - Post summary for users

2. **Create Incident Report**
   - Timeline of events
   - Root cause
   - Resolution steps
   - Action items

3. **Review and Improve**
   - Update runbook if needed
   - Add monitoring if gap found
   - Schedule blameless postmortem

### Incident Report Template

```markdown
# Incident Report: [INC-XXXX]

## Summary
Brief description of what happened.

## Timeline
- HH:MM - Alert triggered
- HH:MM - On-call acknowledged
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix deployed
- HH:MM - Incident resolved

## Impact
- Duration: X hours
- Users affected: X%
- Requests failed: X

## Root Cause
Detailed explanation of why the incident occurred.

## Resolution
Steps taken to resolve the incident.

## Action Items
- [ ] Action 1 - Owner - Due Date
- [ ] Action 2 - Owner - Due Date

## Lessons Learned
What we learned and how we'll prevent recurrence.
```

---

*Keep this runbook updated. Last reviewed: December 2024*
