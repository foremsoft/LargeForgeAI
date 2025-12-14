# Case Study: AI-Powered Customer Support

## Overview

**Company**: TechServe Inc. (Fictional)
**Industry**: B2B SaaS
**Challenge**: Scaling customer support while maintaining quality
**Solution**: LargeForgeAI-powered expert routing system
**Results**: 60% reduction in response time, 40% cost savings

---

## Background

### Company Profile

TechServe Inc. provides cloud infrastructure management tools to mid-market businesses. With 5,000+ customers and growing, their support team was struggling to keep up with demand.

### The Challenge

- **Volume**: 2,000+ support tickets per day
- **Complexity**: Mix of technical and billing questions
- **Response time**: Average 4 hours, target was 1 hour
- **Cost**: Support team growing faster than revenue
- **Quality**: Inconsistent responses across agents

### Previous Attempts

1. **Generic chatbot**: 20% deflection rate, low customer satisfaction
2. **Outsourced support**: Quality issues, high escalation rate
3. **Knowledge base**: Helpful but couldn't handle complex queries

---

## Solution Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Customer Support System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Customer Query → [Intent Classifier] → [Expert Router]       │
│                           │                    │                │
│                           ▼                    ▼                │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Expert Models                         │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │  │
│   │  │Technical │  │ Billing  │  │ Account  │  │ General │ │  │
│   │  │  Expert  │  │  Expert  │  │  Expert  │  │ Expert  │ │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│   [Response Generation] → [Quality Check] → [Customer/Agent]   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Expert Models

| Expert | Training Data | Specialization |
|--------|--------------|----------------|
| Technical | 50K resolved tickets | Server issues, API errors, integrations |
| Billing | 20K billing interactions | Invoices, upgrades, refunds |
| Account | 15K account queries | Password resets, permissions, settings |
| General | 30K general inquiries | Product info, feature questions |

### Implementation Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Data Preparation | 2 weeks | Export tickets, clean data, create training sets |
| Model Training | 1 week | SFT + DPO for each expert |
| Integration | 2 weeks | API integration, testing, monitoring setup |
| Pilot | 4 weeks | 10% traffic, iteration |
| Full Rollout | 2 weeks | Gradual increase to 100% |

---

## Technical Implementation

### Data Preparation

```python
# ticket_processor.py
import pandas as pd
from llm.data import DatasetFormatter

# Load historical tickets
tickets = pd.read_csv("support_tickets.csv")

# Filter for resolved tickets with high satisfaction
quality_tickets = tickets[
    (tickets["status"] == "resolved") &
    (tickets["satisfaction_score"] >= 4) &
    (tickets["response_quality"] == "approved")
]

# Format for training
formatter = DatasetFormatter(format="sharegpt")

training_data = []
for _, ticket in quality_tickets.iterrows():
    training_data.append({
        "conversations": [
            {"from": "system", "value": f"You are a {ticket['category']} support specialist for TechServe."},
            {"from": "human", "value": ticket["customer_query"]},
            {"from": "gpt", "value": ticket["agent_response"]}
        ]
    })

# Split by category for expert training
categories = ["technical", "billing", "account", "general"]
for category in categories:
    category_data = [t for t in training_data if category in t["conversations"][0]["value"]]
    formatter.save(category_data, f"{category}_training.json")
```

### Model Training

```bash
#!/bin/bash
# train_experts.sh

BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"

# Train each expert
for expert in technical billing account general; do
    echo "Training $expert expert..."

    # SFT Phase
    largeforge train sft \
        --model $BASE_MODEL \
        --dataset ./${expert}_training.json \
        --output ./experts/${expert}-sft \
        --num-epochs 3 \
        --lora-r 16 \
        --learning-rate 2e-5

    # DPO Phase (using preference data)
    largeforge train dpo \
        --model ./experts/${expert}-sft \
        --dataset ./${expert}_preferences.json \
        --output ./experts/${expert}-final \
        --beta 0.1 \
        --num-epochs 1
done

echo "All experts trained!"
```

### Router Configuration

```yaml
# router_config.yaml
router:
  classifier_type: hybrid  # keyword + neural

  # Keyword matching
  keyword_config:
    technical:
      keywords: ["error", "API", "server", "crash", "bug", "integration", "webhook"]
      weight: 0.4
    billing:
      keywords: ["invoice", "payment", "subscription", "upgrade", "refund", "charge"]
      weight: 0.4
    account:
      keywords: ["password", "login", "permission", "user", "role", "access"]
      weight: 0.4
    general:
      keywords: ["feature", "how to", "what is", "documentation"]
      weight: 0.2

  # Neural classifier
  neural_config:
    model: ./classifier_model
    threshold: 0.7

  # Hybrid settings
  hybrid_weights:
    keyword: 0.3
    neural: 0.7

  # Fallback
  fallback_expert: general
  confidence_threshold: 0.6

# Load balancing
experts:
  - name: technical
    endpoint: http://technical-expert:8000
    replicas: 3

  - name: billing
    endpoint: http://billing-expert:8000
    replicas: 2

  - name: account
    endpoint: http://account-expert:8000
    replicas: 2

  - name: general
    endpoint: http://general-expert:8000
    replicas: 2
```

### Integration with Ticketing System

```python
# support_integration.py
import requests
from zendesk_api import ZendeskClient

class AISupport:
    def __init__(self):
        self.router_url = "http://router:8080"
        self.zendesk = ZendeskClient()

    def process_ticket(self, ticket_id: str):
        # Get ticket details
        ticket = self.zendesk.get_ticket(ticket_id)

        # Route to appropriate expert
        route_response = requests.post(
            f"{self.router_url}/route",
            json={"query": ticket["description"]}
        )
        expert = route_response.json()["expert"]
        confidence = route_response.json()["confidence"]

        # Generate response
        response = requests.post(
            f"{self.router_url}/generate",
            json={
                "query": ticket["description"],
                "expert": expert,
                "context": {
                    "customer_id": ticket["customer_id"],
                    "product_tier": ticket["product_tier"]
                }
            }
        )

        ai_response = response.json()["response"]

        # Quality check - if low confidence, route to human
        if confidence < 0.7:
            return self.escalate_to_human(ticket_id, ai_response, expert)

        # Auto-respond or draft for review
        if confidence > 0.9 and self.passes_quality_check(ai_response):
            self.zendesk.reply_ticket(ticket_id, ai_response, auto=True)
        else:
            self.zendesk.create_draft(ticket_id, ai_response)

        return {"status": "processed", "expert": expert, "confidence": confidence}

    def passes_quality_check(self, response: str) -> bool:
        # Check for prohibited content, length, etc.
        checks = [
            len(response) > 50,
            "I don't know" not in response.lower(),
            not any(word in response.lower() for word in ["competitor", "lawsuit"]),
        ]
        return all(checks)
```

---

## Results

### Quantitative Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Avg Response Time | 4 hours | 1.5 hours | 62% faster |
| First Contact Resolution | 45% | 72% | +27 points |
| Customer Satisfaction | 3.8/5 | 4.4/5 | +0.6 points |
| Tickets per Agent | 40/day | 65/day | 62% increase |
| Cost per Ticket | $12 | $7 | 42% reduction |

### Routing Accuracy

| Expert | Routing Accuracy | Avg Confidence |
|--------|-----------------|----------------|
| Technical | 94% | 0.87 |
| Billing | 96% | 0.91 |
| Account | 92% | 0.85 |
| General | 88% | 0.79 |

### Customer Feedback

> "The support responses are now much more relevant to my actual question. I used to get generic responses, now I get specific solutions."
> — Customer Survey Response

### Agent Feedback

> "The AI drafts save me so much time. I can handle twice as many tickets, and the suggestions are usually spot-on."
> — Sarah M., Support Engineer

---

## Lessons Learned

### What Worked Well

1. **Domain-specific experts** outperformed a single general model
2. **Hybrid routing** (keyword + neural) was more robust than either alone
3. **Human-in-the-loop** for low-confidence responses maintained quality
4. **Gradual rollout** allowed iteration without customer impact

### Challenges Faced

1. **Initial data quality**: 30% of historical tickets had poor responses
   - **Solution**: Used customer satisfaction as quality filter

2. **Edge cases**: Complex multi-topic queries confused router
   - **Solution**: Added "general" fallback and multi-expert routing

3. **Context understanding**: AI didn't know customer history
   - **Solution**: Added customer context injection

### Recommendations

1. **Start with your best data**: Quality > quantity for training
2. **Monitor continuously**: Set up alerts for routing failures
3. **Iterate quickly**: Weekly retraining with new examples
4. **Keep humans involved**: AI assists, humans verify

---

## Cost Analysis

### Implementation Costs

| Item | Cost |
|------|------|
| GPU compute (training) | $2,000 |
| Development time (3 engineers, 6 weeks) | $90,000 |
| Integration and testing | $15,000 |
| **Total Implementation** | **$107,000** |

### Ongoing Costs

| Item | Monthly Cost |
|------|--------------|
| GPU inference (4x A10G) | $3,200 |
| Monitoring and maintenance | $1,500 |
| Retraining (weekly) | $400 |
| **Total Monthly** | **$5,100** |

### ROI

- Previous support cost: $180,000/month
- New support cost: $108,000/month + $5,100 AI = $113,100/month
- **Monthly savings**: $66,900
- **Payback period**: 1.6 months

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                          Production Setup                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│  │   Zendesk    │───▶│ Integration  │───▶│   Load Balancer     │ │
│  │   Webhook    │    │   Service    │    │   (nginx/traefik)   │ │
│  └──────────────┘    └──────────────┘    └──────────────────────┘ │
│                                                    │               │
│                                                    ▼               │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                      Router Service                          │ │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │ │
│  │  │ Keyword Matcher │  │ Neural Classifier│                   │ │
│  │  └─────────────────┘  └─────────────────┘                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│           │              │              │              │          │
│           ▼              ▼              ▼              ▼          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐ │
│  │  Technical   │ │   Billing    │ │   Account    │ │ General │ │
│  │  (3 pods)    │ │  (2 pods)    │ │  (2 pods)    │ │(2 pods) │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘ │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              Monitoring & Observability                     │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │   │
│  │  │ Prometheus │  │  Grafana   │  │  Alert Manager     │   │   │
│  │  └────────────┘  └────────────┘  └────────────────────┘   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Configuration Files

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: technical-expert
spec:
  replicas: 3
  selector:
    matchLabels:
      app: technical-expert
  template:
    spec:
      containers:
        - name: inference
          image: largeforge/inference:latest
          args:
            - --model=/models/technical-expert
            - --port=8000
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: 24Gi
          volumeMounts:
            - name: model-storage
              mountPath: /models
```

---

*Case study based on composite experiences. Specific metrics are illustrative.*

*Last Updated: December 2024*
