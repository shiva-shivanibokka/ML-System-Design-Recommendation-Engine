# ADR 002: Thompson Sampling for A/B Model Selection

**Status:** Accepted  
**Date:** 2026-06-27

## Context

We maintain two recommendation models — SVD (matrix factorization, fast, interpretable) and NeuMF (neural collaborative filtering, more expressive, slower). We want to route traffic between them based on which one is currently driving higher click-through rate, without human intervention to flip a flag, and without committing 100% of traffic to an unvalidated model.

Classical A/B testing requires fixing a split (e.g. 50/50), running for a fixed duration, then applying a frequentist significance test. This has two costs:
1. **Opportunity cost**: during the experiment, roughly half the traffic goes to the inferior model.
2. **Manual overhead**: someone must decide when to stop the experiment and read the result.

## Decision

We use **Thompson Sampling**, a Bayesian multi-armed bandit algorithm, as a traffic router between SVD and NeuMF.

**Mechanism:**
- Each model maintains Beta distribution parameters (α, β) representing its click-through performance. α increments on click, β increments on no-click.
- At inference time, we draw a sample from each model's Beta(α, β) distribution and route to the model with the higher sample. This naturally shifts traffic toward the better-performing model over time.
- We declare a statistical winner (95% confidence via Monte Carlo simulation: 10,000 draws, model A wins if P(A > B) ≥ 0.95) before fully committing.
- State (α, β per model) persists in a JSON file on disk. In a multi-instance deployment, this should be moved to Redis (INCR operations are atomic).

**Traffic evolution:** With equal starting priors (α=β=1), traffic splits approximately 50/50. As one model accumulates more clicks, it receives progressively more traffic — converging toward 100% of traffic on the winner without a hard switch.

## Consequences

**Positive:**
- Regret minimization: the bandit exploits the current best model while still exploring. Expected regret is O(log T) vs. O(T) for always-explore or O(√T) for epsilon-greedy.
- Self-correcting: if NeuMF degrades after a bad retrain, SVD's click rate improves relative to NeuMF's, and traffic automatically shifts back.
- No fixed experiment windows or manual decisions — continuous adaptation.
- The state file exposes `get_state()` via the `/bandit/state` API endpoint, giving full observability into α, β, CTR, and confidence for each model.

**Negative / Tradeoffs:**
- Thompson Sampling requires logged click feedback per model. Without a working click feedback loop (Kafka consumer → bandit update endpoint), the model never improves past its initial prior. The endpoint `POST /feedback` must be called by the client.
- Priors are not pre-set from historical data, so the system starts with 50/50 traffic regardless of known model quality. An offline evaluation pass before deployment could initialize α/β from historical CTR.
- The JSON-file state store is not thread-safe in multi-process deployments. The architecture doc in the Kafka consumer notes this limitation explicitly.
- CTR is a proxy metric for satisfaction. Thompson Sampling optimizes click probability, not long-term retention or rating quality.

## Alternatives Considered

| Approach | Why Rejected |
|---|---|
| Manual A/B flag | No automatic adaptation; requires human decision cycle |
| Epsilon-greedy bandit | Wastes O(ε) fraction of traffic forever; Thompson achieves better regret bounds |
| UCB1 (Upper Confidence Bound) | Requires deterministic initialization; Thompson handles cold start more naturally with Beta priors |
| Full Bayesian AB test (Bayesian t-test) | More complex, similar properties; Thompson is simpler to implement and equally principled for Bernoulli reward signals |
