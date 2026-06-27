"""
FastAPI Recommendation Gateway — Full Serving Pipeline.

Serves recommendations via a 5-stage pipeline:

  Stage 0: Cache Check (Redis)
    If user's recommendations are cached (TTL=5min) → return immediately (<5ms)
    Else → proceed to Stage 1

  Stage 1: Candidate Generation (FAISS ANN)
    User's GMF embedding (NCF) queried against FAISS IVF+PQ index
    Returns top-500 candidate items in ~20ms
    Cold users → skip to popularity fallback

  Stage 2: Feature Fetch (Redis Online Store via Feast)
    Fetch user + item features from Redis in ~10ms
    Used for ranking features and cold-start detection

  Stage 3: Ranking (Thompson Sampling → SVD or NCF)
    Bandit selects which model scores the 500 candidates
    NCF: full NeuMF forward pass over 500 items (~50ms)
    SVD: dot product over 500 item vectors (~2ms)
    Returns top-50 scored candidates

  Stage 4: Post-Ranking
    Freshness boost → MMR diversity → genre hard cap (~10ms)
    Returns final top-10 recommendations

  Stage 5: Cache Write + Log
    Write results to Redis (TTL=5min)
    Log request to PostgreSQL for analytics and retraining

Total budget: p99 < 100ms (cache miss path)

Endpoints:
  POST /recommend                  — main recommendation endpoint
  POST /feedback/click             — click feedback (updates bandit)
  GET  /bandit/state               — Thompson Sampling state
  GET  /monitoring/health          — staleness detection report
  GET  /monitoring/latency         — p50/p95/p99 latency by stage
  GET  /metrics                    — Prometheus scrape endpoint
  POST /admin/bandit/reset         — reset bandit for new experiment
"""

from __future__ import annotations

import asyncio
import json
import pickle
import sys
import time
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import redis
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings
from serving.bandit import bandit
from serving.cold_start import cold_start_handler
from serving.monitoring import (
    staleness_detector,
    PIPELINE_LATENCY,
    RECOMMENDATION_COUNTER,
    CACHE_HITS,
    CACHE_MISSES,
    COLD_START_COUNTER,
    BANDIT_PULLS,
    BANDIT_REWARDS,
    STAGE_CACHE_CHECK,
    STAGE_FEATURE_FETCH,
    STAGE_CANDIDATE_GEN,
    STAGE_RANKING,
    STAGE_POST_RANKING,
    STAGE_TOTAL,
)
from serving.post_ranking import post_ranker
from training.ncf_model import NeuMF
from training.svd_model import SVDRecommender

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Shared state (loaded once at startup)
# ---------------------------------------------------------------------------


class AppState:
    ncf_model: Optional[NeuMF] = None
    svd_model: Optional[SVDRecommender] = None
    faiss_index = None
    faiss_id_map: Dict[int, int] = {}  # FAISS internal idx → item_id
    faiss_id_to_idx: Dict[int, int] = {}  # item_id → FAISS internal idx
    redis_client: Optional[redis.Redis] = None
    db_engine = None
    feast_store = None  # Feast FeatureStore for online feature retrieval
    user_id_map: Dict[int, int] = {}  # user_id → user_idx
    item_id_map: Dict[int, int] = {}  # item_id → item_idx
    idx_to_item: Dict[int, int] = {}  # item_idx → item_id
    item_genres: Dict[int, str] = {}  # item_id → primary genre
    n_total_items: int = 0
    user_interaction_counts: Dict[int, int] = {}
    item_interaction_counts: Dict[int, int] = {}
    user_history: Dict[int, set] = {}  # user_idx → set of item_idxs seen


state = AppState()


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and indexes at startup."""
    log.info("startup.begin")
    t0 = time.time()
    proc = Path(settings.data.processed_dir)

    # Load NCF model
    if Path(settings.ncf.model_path).exists():
        state.ncf_model = NeuMF.load()
        log.info("startup.ncf_loaded")
    else:
        log.warning("startup.ncf_not_found", path=settings.ncf.model_path)

    # Load SVD model
    if Path(settings.svd.model_path).exists():
        state.svd_model = SVDRecommender.load()
        log.info("startup.svd_loaded")
    else:
        log.warning("startup.svd_not_found", path=settings.svd.model_path)

    # Load FAISS index
    if Path(settings.faiss.index_path).exists():
        state.faiss_index = faiss.read_index(settings.faiss.index_path)
        state.faiss_index.nprobe = settings.faiss.n_probe
        with open(settings.faiss.docid_map_path, "rb") as f:
            state.faiss_id_map = pickle.load(f)  # idx → item_id
        state.faiss_id_to_idx = {v: k for k, v in state.faiss_id_map.items()}
        log.info("startup.faiss_loaded", n_vectors=state.faiss_index.ntotal)
    else:
        log.warning("startup.faiss_not_found")

    # Redis client
    try:
        state.redis_client = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            decode_responses=False,
            socket_connect_timeout=2,
        )
        state.redis_client.ping()
        log.info("startup.redis_connected")
    except Exception as e:
        log.warning("startup.redis_failed", error=str(e))
        state.redis_client = None

    # PostgreSQL
    try:
        state.db_engine = create_engine(settings.postgres.url, pool_pre_ping=True)
        _init_db(state.db_engine)
        log.info("startup.postgres_connected")
    except Exception as e:
        log.warning("startup.postgres_failed", error=str(e))

    # Load ID maps and item metadata
    import pandas as pd

    user_map = pd.read_parquet(proc / "user_id_map.parquet")
    item_map = pd.read_parquet(proc / "item_id_map.parquet")
    movies = pd.read_parquet(proc / "movies.parquet")
    user_stats = pd.read_parquet(proc / "user_stats.parquet")
    item_stats = pd.read_parquet(proc / "item_stats.parquet")
    train = pd.read_parquet(proc / "train.parquet")

    state.user_id_map = dict(zip(user_map["user_id"], user_map["user_idx"]))
    state.item_id_map = dict(zip(item_map["item_id"], item_map["item_idx"]))
    state.idx_to_item = dict(zip(item_map["item_idx"], item_map["item_id"]))
    state.item_genres = dict(zip(movies["item_id"], movies["primary_genre"]))
    state.n_total_items = len(item_map)
    state.user_interaction_counts = dict(
        zip(user_stats["user_id"], user_stats["interaction_count"])
    )
    state.item_interaction_counts = dict(
        zip(item_stats["item_id"], item_stats["interaction_count"])
    )

    # User history for negative sampling / seen item exclusion
    state.user_history = train.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # Cold-start handler
    cold_start_handler.load()

    # Monitoring
    staleness_detector.set_total_items(state.n_total_items)

    # Background staleness check thread
    _start_staleness_thread()

    # Feast online store — graceful degradation if not yet materialized
    try:
        from feast import FeatureStore
        state.feast_store = FeatureStore(repo_path=settings.feature_store.repo_path)
        log.info("startup.feast_connected")
    except Exception as e:
        log.warning("startup.feast_failed", error=str(e))
        state.feast_store = None

    log.info("startup.complete", elapsed_sec=round(time.time() - t0, 2))
    yield
    log.info("shutdown")


def _init_db(engine):
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS recommendation_log (
                request_id TEXT PRIMARY KEY,
                user_id INTEGER,
                model_used TEXT,
                is_cold_start BOOLEAN,
                items_recommended TEXT,
                total_latency_ms REAL,
                stage_latencies TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS click_log (
                click_id TEXT PRIMARY KEY,
                request_id TEXT,
                user_id INTEGER,
                item_id INTEGER,
                model_used TEXT,
                rank_shown INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        )
        conn.commit()


def _start_staleness_thread():
    interval = settings.monitoring.staleness_check_interval_hours * 3600

    def _run():
        while True:
            time.sleep(interval)
            try:
                report = staleness_detector.run_health_check()
                if report["any_alert"]:
                    log.warning("staleness.alert_detected", report=report)
            except Exception as e:
                log.error("staleness.check_failed", error=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RecSys Recommendation Engine",
    description="Netflix-style real-time recommendation system with two-stage retrieval, "
    "Thompson Sampling A/B, cold-start handling, and staleness monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User ID from MovieLens 1M")
    top_n: int = Field(10, ge=1, le=50)
    model_override: Optional[str] = Field(
        None, description="Force 'svd' or 'ncf' (bypasses bandit)"
    )
    exclude_seen: bool = Field(True, description="Exclude already-seen items")


class RecommendItem(BaseModel):
    item_id: int
    title: str
    genre: str
    score: float
    is_fresh: bool


class RecommendResponse(BaseModel):
    request_id: str
    user_id: int
    model_used: str
    is_cold_start: bool
    recommendations: List[RecommendItem]
    latency_ms: Dict[str, float]
    bandit_state: Dict


class ClickRequest(BaseModel):
    request_id: str
    user_id: int
    item_id: int
    model_used: str
    rank_shown: int


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------


async def _recommend_pipeline(req: RecommendRequest) -> RecommendResponse:
    request_id = str(uuid.uuid4())
    t_total_start = time.time()
    stage_latencies = {}
    is_cold_start = False

    user_id = req.user_id
    user_idx = state.user_id_map.get(user_id)

    # ------------------------------------------------------------------
    # Stage 0: Cache Check
    # ------------------------------------------------------------------
    t0 = time.time()
    if state.redis_client:
        cache_key = f"rec:{user_id}:{req.top_n}"
        try:
            cached = state.redis_client.get(cache_key)
            if cached:
                stage_latencies[STAGE_CACHE_CHECK] = (time.time() - t0) * 1000
                CACHE_HITS.inc()
                PIPELINE_LATENCY.labels(stage=STAGE_CACHE_CHECK).observe(
                    stage_latencies[STAGE_CACHE_CHECK]
                )
                data = json.loads(cached)
                data["request_id"] = request_id
                data["latency_ms"] = {STAGE_TOTAL: stage_latencies[STAGE_CACHE_CHECK]}
                return RecommendResponse(**data)
        except Exception:
            pass
    stage_latencies[STAGE_CACHE_CHECK] = (time.time() - t0) * 1000
    CACHE_MISSES.inc()

    # ------------------------------------------------------------------
    # Cold-start detection
    # ------------------------------------------------------------------
    interaction_count = state.user_interaction_counts.get(user_id, 0)
    is_cold_user = cold_start_handler.is_cold_user(user_id, interaction_count)

    # ------------------------------------------------------------------
    # Stage 1: Candidate Generation (FAISS ANN)
    # ------------------------------------------------------------------
    t0 = time.time()
    candidate_item_idxs: List[int] = []

    if is_cold_user or user_idx is None or state.faiss_index is None:
        is_cold_start = True
        COLD_START_COUNTER.labels(type="user").inc()
        # Fallback: use popularity pool item indexes
        seen_item_ids = set()
        pop_items = cold_start_handler.get_popularity_fallback(
            top_k=settings.faiss.top_k_candidates,
            exclude_seen=seen_item_ids,
        )
        candidate_item_idxs = [
            state.item_id_map[iid] for iid, _ in pop_items if iid in state.item_id_map
        ]
    else:
        if state.ncf_model is not None:
            user_emb = state.ncf_model.get_user_embedding(user_idx).astype(np.float32)
            # L2-normalize for cosine similarity
            norm = np.linalg.norm(user_emb)
            if norm > 0:
                user_emb = user_emb / norm
            user_emb = user_emb.reshape(1, -1)

            k = min(settings.faiss.top_k_candidates, state.faiss_index.ntotal)
            distances, indices = state.faiss_index.search(user_emb, k)
            # Map FAISS internal indices → item_idxs
            for faiss_idx in indices[0]:
                if faiss_idx >= 0:
                    item_id = state.faiss_id_map.get(int(faiss_idx))
                    if item_id is not None:
                        item_idx = state.item_id_map.get(item_id)
                        if item_idx is not None:
                            candidate_item_idxs.append(item_idx)
        else:
            # NCF not available — use all items (SVD handles ranking)
            candidate_item_idxs = list(range(min(500, state.n_total_items)))

    stage_latencies[STAGE_CANDIDATE_GEN] = (time.time() - t0) * 1000
    PIPELINE_LATENCY.labels(stage=STAGE_CANDIDATE_GEN).observe(
        stage_latencies[STAGE_CANDIDATE_GEN]
    )

    # Exclude seen items
    seen_idxs: set = set()
    if req.exclude_seen and user_idx is not None:
        seen_idxs = state.user_history.get(user_idx, set())
    candidate_item_idxs = [idx for idx in candidate_item_idxs if idx not in seen_idxs]

    if not candidate_item_idxs:
        is_cold_start = True

    # ------------------------------------------------------------------
    # Stage 2: Feature Fetch (Feast → Redis online store)
    # Fetches user features materialized by feature_store/materialize.py.
    # Degrades gracefully if the online store isn't populated yet — the
    # ranking stages work fine without these features; they're additive signal.
    # Run `make feast-materialize` after training to populate the online store.
    # ------------------------------------------------------------------
    t0 = time.time()
    feast_features: dict = {}
    if state.feast_store is not None:
        try:
            fv = state.feast_store.get_online_features(
                features=[
                    "user_features:interaction_count",
                    "user_features:is_cold_user",
                    "user_features:avg_rating_proxy",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            feast_features = {k: v[0] for k, v in fv.items() if v and v[0] is not None}
            if feast_features:
                log.debug("stage2.features_fetched", user_id=user_id, keys=list(feast_features.keys()))
        except Exception as e:
            log.debug("stage2.feast_miss", error=str(e))
    stage_latencies[STAGE_FEATURE_FETCH] = (time.time() - t0) * 1000
    PIPELINE_LATENCY.labels(stage=STAGE_FEATURE_FETCH).observe(stage_latencies[STAGE_FEATURE_FETCH])

    # ------------------------------------------------------------------
    # Stage 3: Ranking (Thompson Sampling → SVD or NCF)
    # ------------------------------------------------------------------
    t0 = time.time()
    model_name = req.model_override or bandit.select_arm()
    BANDIT_PULLS.labels(model=model_name).inc()

    scored_candidates: List[tuple] = []  # (item_idx, score)

    if is_cold_start or not candidate_item_idxs:
        # Fallback: rank by popularity score
        pop_items = cold_start_handler.get_popularity_fallback(
            top_k=settings.post_ranking.top_n * 5,
            exclude_seen={state.idx_to_item.get(idx, -1) for idx in seen_idxs},
        )
        for item_id, score in pop_items:
            item_idx = state.item_id_map.get(item_id)
            if item_idx is not None:
                scored_candidates.append((item_idx, score))
        model_name = "popularity_fallback"
    elif model_name == "ncf" and state.ncf_model is not None:
        # NCF: full NeuMF forward pass over candidates
        scores = state.ncf_model.score_candidates(user_idx, candidate_item_idxs)
        scored_candidates = list(zip(candidate_item_idxs, scores.tolist()))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
    elif model_name == "svd" and state.svd_model is not None:
        # SVD: dot product between user factor and candidate item factors
        all_scores = state.svd_model.predict_scores(user_idx)
        scored_candidates = [
            (idx, float(all_scores[idx])) for idx in candidate_item_idxs
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
    else:
        # Fallback: popularity
        pop_items = cold_start_handler.get_popularity_fallback(
            top_k=settings.post_ranking.top_n * 5
        )
        for item_id, score in pop_items:
            item_idx = state.item_id_map.get(item_id)
            if item_idx is not None:
                scored_candidates.append((item_idx, score))
        model_name = "popularity_fallback"

    # Take top-50 for post-ranking
    top50 = scored_candidates[:50]
    # Convert item_idx → item_id for post-ranking
    top50_by_id = [
        (state.idx_to_item.get(idx, -1), score)
        for idx, score in top50
        if state.idx_to_item.get(idx, -1) != -1
    ]

    stage_latencies[STAGE_RANKING] = (time.time() - t0) * 1000
    PIPELINE_LATENCY.labels(stage=STAGE_RANKING).observe(stage_latencies[STAGE_RANKING])

    # Log scores for PSI monitoring
    for _, score in top50_by_id[:10]:
        staleness_detector.log_score(model_name, float(score))

    # ------------------------------------------------------------------
    # Stage 4: Post-Ranking (MMR + freshness + genre cap)
    # ------------------------------------------------------------------
    t0 = time.time()
    final = post_ranker.rerank(
        candidates=top50_by_id,
        item_genres=state.item_genres,
        top_n=req.top_n,
    )
    stage_latencies[STAGE_POST_RANKING] = (time.time() - t0) * 1000
    PIPELINE_LATENCY.labels(stage=STAGE_POST_RANKING).observe(
        stage_latencies[STAGE_POST_RANKING]
    )

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    recommendations = []
    for item_id, score, meta in final:
        recommendations.append(
            RecommendItem(
                item_id=item_id,
                title=cold_start_handler.get_item_title(item_id),
                genre=meta.get("genre", "Unknown"),
                score=round(score, 6),
                is_fresh=meta.get("is_fresh", False),
            )
        )

    # Log for coverage tracking
    staleness_detector.log_recommendations([r.item_id for r in recommendations])
    RECOMMENDATION_COUNTER.labels(
        model=model_name,
        cold_start="true" if is_cold_start else "false",
    ).inc()

    total_latency = (time.time() - t_total_start) * 1000
    stage_latencies[STAGE_TOTAL] = total_latency
    PIPELINE_LATENCY.labels(stage=STAGE_TOTAL).observe(total_latency)

    # Check latency budget
    if total_latency > settings.latency_budgets.total_p99_ms:
        log.warning(
            "latency_budget_exceeded",
            total_ms=round(total_latency, 2),
            budget_ms=settings.latency_budgets.total_p99_ms,
        )

    response = RecommendResponse(
        request_id=request_id,
        user_id=user_id,
        model_used=model_name,
        is_cold_start=is_cold_start,
        recommendations=recommendations,
        latency_ms={k: round(v, 2) for k, v in stage_latencies.items()},
        bandit_state=bandit.get_state(),
    )

    # ------------------------------------------------------------------
    # Stage 5: Cache Write (background)
    # ------------------------------------------------------------------
    if state.redis_client and not is_cold_start:
        try:
            cache_key = f"rec:{user_id}:{req.top_n}"
            cache_data = response.model_dump()
            state.redis_client.setex(
                cache_key,
                settings.redis.recommendation_ttl_seconds,
                json.dumps(cache_data),
            )
        except Exception:
            pass

    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, background_tasks: BackgroundTasks):
    """
    Main recommendation endpoint.
    5-stage pipeline: cache check → FAISS retrieval → Feast features →
    Thompson Sampling ranking → MMR post-ranking.
    """
    return await _recommend_pipeline(req)


@app.post("/feedback/click")
async def click_feedback(req: ClickRequest):
    """
    Record a click event. Updates:
    1. Thompson Sampling bandit (reward=1)
    2. Staleness detector click log
    3. PostgreSQL click log table
    """
    # Update bandit
    bandit.update(req.model_used, clicked=True)
    BANDIT_REWARDS.labels(model=req.model_used).inc()

    # Log to staleness detector
    staleness_detector.log_click(req.model_used, clicked=True)

    # PostgreSQL
    if state.db_engine:
        try:
            with state.db_engine.connect() as conn:
                conn.execute(
                    text("""
                    INSERT INTO click_log (click_id, request_id, user_id, item_id, model_used, rank_shown)
                    VALUES (:cid, :rid, :uid, :iid, :model, :rank)
                """),
                    {
                        "cid": str(uuid.uuid4()),
                        "rid": req.request_id,
                        "uid": req.user_id,
                        "iid": req.item_id,
                        "model": req.model_used,
                        "rank": req.rank_shown,
                    },
                )
                conn.commit()
        except Exception as e:
            log.error("click_log_failed", error=str(e))

    # Invalidate cache for this user
    if state.redis_client:
        try:
            state.redis_client.delete(f"rec:{req.user_id}:10")
        except Exception:
            pass

    return {"status": "ok", "bandit_state": bandit.get_state()}


@app.post("/feedback/no_click")
async def no_click_feedback(request_id: str, user_id: int, model_used: str):
    """Record a no-click (impression without click). Updates bandit with reward=0."""
    bandit.update(model_used, clicked=False)
    staleness_detector.log_click(model_used, clicked=False)
    return {"status": "ok"}


@app.get("/bandit/state")
async def get_bandit_state():
    """Current Thompson Sampling state: α/β params, CTRs, traffic split, winner."""
    state_data = bandit.get_state()
    state_data["traffic_split"] = bandit.get_traffic_split()
    return state_data


@app.post("/admin/bandit/reset")
async def reset_bandit():
    """Reset bandit to uniform prior. Use to start a new experiment."""
    bandit.reset()
    return {"status": "reset", "state": bandit.get_state()}


@app.get("/monitoring/health")
async def monitoring_health():
    """Full staleness detection report: CTR, coverage, PSI."""
    return staleness_detector.run_health_check()


@app.get("/monitoring/latency")
async def latency_report():
    """Latency budget status. Returns stage latency percentiles from Prometheus."""
    return {
        "budget_ms": {
            "total_p99": settings.latency_budgets.total_p99_ms,
            "candidate_generation": settings.latency_budgets.candidate_generation_ms,
            "feature_fetch": settings.latency_budgets.feature_fetch_ms,
            "ranking": settings.latency_budgets.ranking_ms,
            "post_ranking": settings.latency_budgets.post_ranking_ms,
            "cache_hit": settings.latency_budgets.cache_serve_ms,
        },
        "note": "Full latency histograms available on Prometheus :9090 and Grafana :3000",
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics scrape endpoint."""
    from fastapi.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "ncf": state.ncf_model is not None,
            "svd": state.svd_model is not None,
            "faiss": state.faiss_index is not None,
        },
        "redis": state.redis_client is not None,
        "postgres": state.db_engine is not None,
        "n_total_items": state.n_total_items,
    }


@app.get("/")
async def root():
    return {
        "service": "RecSys Recommendation Engine",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }
