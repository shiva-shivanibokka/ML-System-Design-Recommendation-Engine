# ADR 003: FAISS IVF+PQ Index Configuration

**Status:** Accepted  
**Date:** 2026-06-27

## Context

Stage 1 retrieval requires a vector index over item embeddings (128 dimensions, ~9,000 items). The index must meet four constraints simultaneously:
1. **Recall:** ≥95% recall@500 (the Stage 2 ranker needs good candidates)
2. **Latency:** <5ms query latency (contributing <50ms total to the pipeline budget)
3. **Memory:** Fit in container RAM alongside the two ML models (target <100MB total)
4. **Build time:** Rebuild overnight with retraining (< 30 seconds for 9K items)

## Decision

We use **FAISS IndexIVFPQ** with the following parameters, set in `training/build_faiss_index.py`:

| Parameter | Value | Reasoning |
|---|---|---|
| Index type | `IndexIVFPQ` | IVF for fast cell pruning + PQ for memory compression |
| `nlist` (Voronoi cells) | 256 | Rule of thumb: √N where N=items; 256 ≈ √(9000). More cells = faster search, but requires more training data |
| `n_probe` | 32 | Probes 32/256 = 12.5% of cells; empirically achieves ~95% recall on MovieLens embeddings |
| `M` (PQ sub-quantizers) | 16 | 128 dims / 16 = 8 dims per sub-quantizer; 256 centroids per sub-quantizer |
| `nbits` | 8 | 256 centroids per PQ code (standard) |
| Metric | `METRIC_INNER_PRODUCT` | User and item embeddings are L2-normalized; inner product = cosine similarity |

**Memory footprint:** Each item vector is encoded as 16 bytes (16 sub-quantizers × 1 byte each) vs. 512 bytes for float32. Total index ≈ 9000 × 16 = 144KB, plus IVF cell structures ≈ 0.5MB total.

**Fallback:** For catalog sizes below 1,000 items, the build script falls back to `IndexFlatIP` (exact brute-force). The threshold at which IVF training is meaningful requires at least `nlist × 39` training points (FAISS guidance: ≥39× nlist for reliable cell training).

## Consequences

**Positive:**
- Index fits entirely in L1 cache on modern CPUs — query time dominated by I/O scheduling, not computation.
- PQ compression is lossless in its distance approximation properties (error bounded by sub-quantizer centroids), giving predictable recall.
- FAISS is the production standard at Meta, Spotify, and Pinterest for billion-scale retrieval — the same API handles scale-up from 9K items to 9B items by switching quantization parameters.
- The entire index serializes to a ~0.5MB file, trivially stored in S3/GCS for persistent reload.

**Negative / Tradeoffs:**
- IVF+PQ is an **approximate** index. The 5% recall gap (95% vs 100%) means that for any given query, ~25 of 500 returned candidates may not be the true top-500 nearest neighbors. For a reranking funnel this is acceptable — the Stage 2 model compensates.
- The IVF index must be **trained** (k-means clustering of embeddings into Voronoi cells) before items can be added. Training time is ~2s for 9K items and must re-run whenever the embedding space changes (i.e., after model retraining).
- `n_probe=32` is tuned for MovieLens-1M embeddings. If item embeddings cluster very differently after domain shift, this value should be recalibrated via a recall benchmark (`faiss.knn_gpu` for ground truth vs. IVFPQquery output).

## Alternatives Considered

| Index Type | Recall@500 | Memory | Latency | Why Rejected |
|---|---|---|---|---|
| `IndexFlatIP` (exact) | 100% | 4.5MB | 1-2ms | Scales O(N); acceptable now but not at 100K+ items |
| `IndexHNSWFlat` | 99% | 36MB | 1ms | Better recall, but 72× more memory; no compression |
| `IndexIVFFlat` | 97% | 4.5MB | 2ms | Better recall than IVF+PQ but same memory as flat |
| `IndexIVFPQ` (chosen) | ~95% | 0.5MB | 3-5ms | Best memory efficiency; 95% recall sufficient for our funnel |
