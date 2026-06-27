# ADR 001: Two-Stage Retrieval (FAISS ANN → Model Reranking)

**Status:** Accepted  
**Date:** 2026-06-27

## Context

The catalog contains ~9,000 movies. A naive approach scores every item for every user request — O(N) model inferences per request, which at NCF forward-pass cost (~2ms each) would take 18 seconds for a single request. This is not feasible for a real-time serving system targeting sub-100ms end-to-end latency.

Industry systems (YouTube, Pinterest, DoorDash) uniformly solve this with a **two-stage funnel**: a fast, approximate first-stage retriever narrows candidates, followed by a slower, accurate second-stage ranker applied to the small candidate set.

## Decision

We implement a two-stage pipeline:

1. **Stage 1 — Candidate Retrieval (FAISS IVF+PQ ANN):** A 128-dimensional user embedding is queried against a FAISS index of item embeddings. Approximate nearest-neighbor search using Inverted File Index (IVF) with 256 Voronoi cells and Product Quantization (PQ, 16 sub-quantizers) returns the top-500 candidate items in <5ms. The index is ~0.5MB in memory (vs. 4MB+ for flat L2) and achieves ~95% recall@500.

2. **Stage 2 — Reranking (SVD or NeuMF via Thompson Sampling):** The full collaborative-filtering model (SVD matrix factorization or NeuMF neural model, selected by a Thompson Sampling bandit) scores all 500 candidates and reranks them, returning the top-50. Per-inference latency is ~0.1ms for SVD, ~2ms for NeuMF, making 500-candidate scoring feasible at <100ms total.

3. **Stage 3 — Post-Ranking (MMR):** Maximal Marginal Relevance post-processes the top-50 to enforce diversity (λ=0.7, genre cap of 3 per genre), producing the final top-10 returned to the user.

## Consequences

**Positive:**
- End-to-end p99 latency budget is met (<100ms) across all pipeline stages.
- FAISS retrieval is parallelizable and scales to millions of items without code changes (swap to HNSW for M+ scale).
- Stage 2 model is fully hot-swappable — the bandit can shift traffic between SVD and NeuMF without a restart.
- The funnel is interpretable: we can log stage exit counts (500 → 50 → 10) to detect retrieval or ranking regressions independently.

**Negative / Tradeoffs:**
- ~5% of truly relevant items may be missed in Stage 1 (ANN approximation error). At 500 candidates from ~9,000 items, coverage is ~5.6% of catalog per request — acceptable recall.
- The FAISS index must be rebuilt whenever item embeddings change (nightly, aligned with retraining). There is a brief window where new items aren't retrievable.
- The IVF index requires a minimum catalog size (~1,000 items) to train meaningfully. For catalogs below this, use a flat index (automatically handled in `training/build_faiss_index.py`).

## Alternatives Considered

| Approach | Why Rejected |
|---|---|
| Brute-force scoring all 9K items | ~18s latency — not interactive |
| HNSW (no PQ) | Better recall but 8× more RAM; overkill at 9K items |
| BM25 keyword retrieval | No semantic similarity; poor quality for cold start |
| Two-tower model (separate retrieval model) | Would require training a third model; FAISS on existing embeddings achieves comparable recall |
