# Architecture Decision Records (ADRs)

## Overview

This directory contains Architecture Decision Records (ADRs) documenting significant architectural decisions made during the development of LargeForgeAI.

ADRs capture the context, decision, and consequences of important technical choices, providing:
- Historical record of why decisions were made
- Onboarding context for new team members
- Reference for revisiting decisions when context changes

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [ADR-001](./ADR-001-inference-backend-selection.md) | Inference Backend Selection | Accepted | 2024-06-15 |
| [ADR-002](./ADR-002-training-framework-selection.md) | Training Framework Selection | Accepted | 2024-06-20 |
| [ADR-003](./ADR-003-expert-routing-architecture.md) | Expert Routing Architecture | Accepted | 2024-07-01 |

## Status Definitions

| Status | Description |
|--------|-------------|
| **Proposed** | Under discussion, not yet accepted |
| **Accepted** | Approved and implemented |
| **Deprecated** | No longer recommended but still in use |
| **Superseded** | Replaced by a newer ADR |

## Creating a New ADR

1. Copy the [ADR template](./ADR_TEMPLATE.md)
2. Name it `ADR-XXX-short-description.md` (XXX = next number)
3. Fill in all sections
4. Submit for review via PR
5. Update this index when merged

### Guidelines

- **Be specific**: Include concrete details and examples
- **Document alternatives**: Show why other options weren't chosen
- **Include consequences**: Both positive and negative
- **Keep it focused**: One decision per ADR
- **Reference related ADRs**: Link to related decisions

## When to Write an ADR

Write an ADR for decisions that:
- Are hard to change later
- Affect multiple components
- Have significant trade-offs
- Were debated by the team
- Would benefit future developers

Examples:
- Framework/library selection
- API design patterns
- Data storage strategies
- Security approaches
- Deployment architectures

## ADR Lifecycle

```
┌──────────┐     ┌──────────┐     ┌────────────┐
│ Proposed │────▶│ Accepted │────▶│ Deprecated │
└──────────┘     └──────────┘     └────────────┘
      │                │                 │
      │                │                 ▼
      │                │          ┌────────────┐
      │                └─────────▶│ Superseded │
      │                           └────────────┘
      │
      ▼
┌──────────┐
│ Rejected │
└──────────┘
```

## Further Reading

- [Michael Nygard's ADR article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
- [Lightweight Architecture Decision Records](https://www.thoughtworks.com/radar/techniques/lightweight-architecture-decision-records)

---

*Last Updated: December 2024*
