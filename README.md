# ML System Design: Real-Time Recommendation Engine

A production-grade recommendation system designed to the same architectural standards as Netflix, Spotify, and YouTube. Built to answer the most common ML system design interview question at FAANG companies.

---

## Architecture

```
User Request (user_id)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 0: Redis Cache Check                       │
│     hit → return in <5ms    │    miss → continue pipeline        │
└───────────────────────────────────────────┬─────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 1: Candidate Generation (FAISS IVF+PQ)           │
│  Cold user? → popularity fallback (top-200 items)                │
│  Warm user  → NCF GMF embedding → ANN query → top-500            │
│  Budget: 20ms                                                     │
└───────────────────────────────────────────┬─────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│        STAGE 2: Feature Fetch (Feast → Redis Online Store)        │
│  User features + item features from Redis (<5ms)                 │
│  Same features as training → zero training-serving skew          │
│  Budget: 10ms                                                     │
└───────────────────────────────────────────┬─────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│        STAGE 3: Ranking — Thompson Sampling Bandit                │
│  Selects: SVD (arm A) or NeuMF (arm B)                           │
│  SVD:  dot product over 500 item factors  ~2ms                   │
│  NCF:  full NeuMF forward pass over 500   ~50ms                  │
│  Auto-shifts traffic to winning arm over time                    │
│  Budget: 50ms                                                     │
└───────────────────────────────────────────┬─────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 4: Post-Ranking                           │
│  1. Freshness boost  (+10% score for recently seen items)        │
│  2. MMR diversity    (λ=0.7 — penalizes same-genre repeats)      │
│  3. Genre hard cap   (max 3 items per genre)                     │
│  Budget: 10ms                                                     │
└───────────────────────────────────────────┬─────────────────────┘
                                             │
                                             ▼
                        Top-10 recommendations

── Feedback Loop ──────────────────────────────────────────────────
Click event → POST /feedback/click
  → Thompson Sampling bandit: α += 1 (reward)
  → PostgreSQL click log
  → Kafka consumer → Feast offline store (new training row)
  → Nightly Airflow DAG: retrain → validate → promote if better

── Staleness Detection (background thread) ────────────────────────
  CTR rolling 7-day  → alert if < 80% of baseline
  Catalog coverage   → alert if < 10% of catalog recommended (collapse)
  PSI drift score    → alert if > 0.2 (distribution shift)
```

---

## What Makes This Different From a Simple RecSys

| System Design Problem | Solution Implemented |
|---|---|
| Can't run NCF over all items in <100ms | Two-stage: FAISS ANN → top-500 → NCF reranks |
| New users have no history | Cold-start: popularity-based fallback (top-200 global) |
| New items have no interactions | Content-based fallback: genre embedding cosine similarity |
| Fixed A/B wastes traffic on worse model | Thompson Sampling bandit: auto-shifts to winner |
| Top results all same genre | MMR diversity re-ranking (Maximal Marginal Relevance) |
| Training features ≠ serving features | Feast: single feature definition, offline + online store |
| Models decay silently in production | 3-signal staleness detection: CTR + coverage + PSI |
| No latency visibility | Per-stage Prometheus histograms + budget enforcement |
| Static model in production | Kafka → Feast feedback loop → nightly Airflow retrain |

---

## Models

### SVD — Matrix Factorization Baseline (arm A)
- TruncatedSVD via scikit-learn on the user-item interaction matrix
- User factors: (6040 × 128) in memory — ~3MB, trivial
- Serving latency: ~2ms (dot product)
- Role: fast, cheap baseline; wins when NCF provides no improvement

### NeuMF — Neural Collaborative Filtering (arm B)
- Architecture: GMF path (element-wise embedding product) + MLP path (concat → deep layers)
- GMF embeddings extracted → indexed in FAISS for Stage 1 retrieval
- Full NeuMF forward pass for Stage 2 reranking over 500 candidates
- Evaluation protocol: HR@10, NDCG@10 on leave-one-out chronological splits

### FAISS IVF+PQ Index (Stage 1)
- 3,706 item embeddings (64-dim NCF GMF vectors)
- IVF256 + PQ16: ~0.5MB, ~20ms ANN retrieval, ~95% recall vs exact
- Query: user GMF embedding → top-500 nearest items

---

## Dataset

**MovieLens 1M** (GroupLens Research)
- 1,000,209 ratings | 6,040 users | 3,706 movies
- Implicit feedback: rating ≥ 4 → positive interaction
- Split: chronological leave-one-out per user (industry standard for RecSys eval)

---

## Stack

| Component | Technology |
|---|---|
| Event streaming | Kafka (Confluent) |
| Feature store | Feast (offline: Parquet, online: Redis) |
| Stage 1 retrieval | FAISS IVF+PQ |
| Models | PyTorch (NeuMF), scikit-learn (SVD) |
| A/B routing | Thompson Sampling (custom implementation) |
| Post-ranking | MMR + freshness + genre cap |
| Experiment tracking | MLflow |
| API serving | FastAPI + Uvicorn |
| Cache | Redis |
| Database | PostgreSQL |
| Monitoring | Prometheus + Grafana |
| Structured logging | structlog (JSON) |
| Containerization | Docker + docker-compose |
| Demo UI | Gradio |
| Deployment | Railway (docker-compose) |

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and preprocess MovieLens 1M
```bash
python scripts/download_movielens.py
```

### 3. Train models
```bash
python training/train.py --model all
# Trains SVD + NCF, builds FAISS index, registers in MLflow
```

### 4. Start full stack
```bash
docker-compose up --build
```

### 5. Access services
| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| Gradio UI | http://localhost:7860 |
| MLflow | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### 6. Test the API
```bash
# Get recommendations for user 1
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_n": 10}'

# Register a click (updates bandit)
curl -X POST http://localhost:8000/feedback/click \
  -H "Content-Type: application/json" \
  -d '{"request_id": "...", "user_id": 1, "item_id": 1196, "model_used": "ncf", "rank_shown": 1}'

# Check bandit state
curl http://localhost:8000/bandit/state

# Check staleness signals
curl http://localhost:8000/monitoring/health
```

### 7. Simulate user events (Kafka producer)
```bash
python kafka/producer.py --n-events 1000 --rate 10
```

---

## Evaluation Results

| Model | HR@10 | NDCG@10 | Serving latency (p99) |
|---|---|---|---|
| SVD (128 components) | ~0.68 | ~0.38 | ~2ms |
| NeuMF (64-dim, MLP [128,64,32]) | ~0.74 | ~0.44 | ~50ms |
| Popularity fallback (cold-start) | ~0.31 | ~0.15 | <1ms |

*Evaluated on MovieLens 1M with leave-one-out chronological splits (NCF paper protocol).*

---

## Key Design References

- **NeuMF architecture**: He et al., "Neural Collaborative Filtering", WWW 2017
- **YouTube DNN two-stage**: Covington et al., "Deep Neural Networks for YouTube Recommendations", RecSys 2016
- **Thompson Sampling**: Chapelle & Li, "An Empirical Evaluation of Thompson Sampling", NeurIPS 2011
- **MMR diversity**: Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking", SIGIR 1998
- **PSI drift detection**: standard metric in credit risk ML, adopted by Evidently AI and WhyLabs
- **Feast feature store**: eliminates training-serving skew — the most common silent failure in production ML

---

## Interview Talking Points

**"How does this scale to 100M users?"**
- FAISS IVF+PQ scales to billions of vectors (Facebook AI Research)
- Redis online store handles millions of feature lookups/sec
- Kafka decouples event ingestion from serving — backpressure handled independently
- Serving layer is stateless → horizontal scaling via multiple replicas behind load balancer

**"What happens when a model goes stale?"**
- Three signals: CTR drop, coverage collapse, PSI score distribution shift
- Kafka consumer continuously writes new interactions to Feast offline store
- Airflow DAG retrains nightly with fresh data, validates against held-out set, promotes if better

**"How do you handle the cold-start problem?"**
- New users: popularity-based fallback (top-200 globally popular items)
- New items: genre embedding cosine similarity to find proxy collaborative signal
- Threshold-based routing: switch to collaborative filtering once N interactions collected

**"Why Thompson Sampling over a fixed A/B test?"**
- Fixed 50/50: continues sending half traffic to worse model even after statistical significance
- Thompson Sampling: automatically converges to winner — regret-minimizing
- Bayesian credible interval (P > 0.95) used to formally declare winner
