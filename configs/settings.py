"""
Typed configuration loader.
Loads configs/config.yaml into Python dataclasses.
Usage:
    from configs.settings import settings
    print(settings.ncf.embedding_dim)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    raw_dir: str
    processed_dir: str
    embeddings_dir: str
    indexes_dir: str
    movielens_url: str
    min_interactions: int
    train_ratio: float
    val_ratio: float


@dataclass
class FeatureStoreConfig:
    repo_path: str
    offline_store_path: str
    online_store: str
    redis_host: str
    redis_port: int
    materialization_interval_hours: int


@dataclass
class SVDConfig:
    n_components: int
    n_iter: int
    random_state: int
    model_path: str
    user_factors_path: str
    item_factors_path: str


@dataclass
class NCFConfig:
    embedding_dim: int
    mlp_layers: List[int]
    dropout: float
    learning_rate: float
    batch_size: int
    num_epochs: int
    num_negatives: int
    early_stopping_patience: int
    model_path: str
    user_embedding_path: str
    item_embedding_path: str


@dataclass
class FAISSConfig:
    index_path: str
    docid_map_path: str
    embedding_dim: int
    n_clusters: int
    n_probe: int
    pq_subquantizers: int
    top_k_candidates: int


@dataclass
class LatencyBudgets:
    total_p99_ms: int
    candidate_generation_ms: int
    feature_fetch_ms: int
    ranking_ms: int
    post_ranking_ms: int
    cache_serve_ms: int


@dataclass
class RedisConfig:
    host: str
    port: int
    recommendation_ttl_seconds: int
    feature_ttl_seconds: int


@dataclass
class BanditConfig:
    models: List[str]
    initial_alpha: float
    initial_beta: float
    state_path: str
    min_samples_per_arm: int
    significance_threshold: float


@dataclass
class PostRankingConfig:
    top_n: int
    mmr_lambda: float
    freshness_weight: float
    max_same_genre: int


@dataclass
class ColdStartConfig:
    user_min_interactions: int
    item_min_interactions: int
    popularity_top_n: int
    content_sim_top_k: int


@dataclass
class KafkaTopics:
    user_events: str
    feedback: str


@dataclass
class KafkaConfig:
    bootstrap_servers: str
    topics: KafkaTopics
    consumer_group: str
    auto_offset_reset: str


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    artifact_location: str


@dataclass
class MonitoringConfig:
    ctr_window_hours: int
    ctr_alert_threshold: float
    coverage_alert_threshold: float
    staleness_check_interval_hours: int
    prometheus_port: int


@dataclass
class PostgresConfig:
    host: str
    port: int
    db: str
    user: str
    password: str
    url: str


@dataclass
class ServicesConfig:
    gateway_port: int
    retrieval_port: int
    ranking_port: int
    feedback_port: int
    mlflow_port: int
    gradio_port: int
    airflow_port: int
    prometheus_port: int
    grafana_port: int


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    data: DataConfig
    feature_store: FeatureStoreConfig
    svd: SVDConfig
    ncf: NCFConfig
    faiss: FAISSConfig
    latency_budgets: LatencyBudgets
    redis: RedisConfig
    bandit: BanditConfig
    post_ranking: PostRankingConfig
    cold_start: ColdStartConfig
    kafka: KafkaConfig
    mlflow: MLflowConfig
    monitoring: MonitoringConfig
    postgres: PostgresConfig
    services: ServicesConfig


def _load_settings() -> Settings:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Override sensitive fields from environment variables so secrets never live
    # in committed files. Set these in .env (local) or your cloud provider's
    # secret store (Railway / Render / HF Spaces).
    pg = raw["postgres"]
    pg["password"] = os.getenv("POSTGRES_PASSWORD", "recsys")
    pg["user"] = os.getenv("POSTGRES_USER", pg["user"])
    pg["host"] = os.getenv("POSTGRES_HOST", pg["host"])
    pg["db"] = os.getenv("POSTGRES_DB", pg["db"])
    # DATABASE_URL takes precedence over component-level overrides
    pg["url"] = os.getenv(
        "DATABASE_URL",
        f"postgresql://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['db']}",
    )

    raw["redis"]["host"] = os.getenv("REDIS_HOST", raw["redis"]["host"])
    raw["kafka"]["bootstrap_servers"] = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", raw["kafka"]["bootstrap_servers"]
    )
    raw["mlflow"]["tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI", raw["mlflow"]["tracking_uri"]
    )

    return Settings(
        data=DataConfig(**raw["data"]),
        feature_store=FeatureStoreConfig(**raw["feature_store"]),
        svd=SVDConfig(**raw["svd"]),
        ncf=NCFConfig(**raw["ncf"]),
        faiss=FAISSConfig(**raw["faiss"]),
        latency_budgets=LatencyBudgets(**raw["latency_budgets"]),
        redis=RedisConfig(**raw["redis"]),
        bandit=BanditConfig(**raw["bandit"]),
        post_ranking=PostRankingConfig(**raw["post_ranking"]),
        cold_start=ColdStartConfig(**raw["cold_start"]),
        kafka=KafkaConfig(
            bootstrap_servers=raw["kafka"]["bootstrap_servers"],
            topics=KafkaTopics(**raw["kafka"]["topics"]),
            consumer_group=raw["kafka"]["consumer_group"],
            auto_offset_reset=raw["kafka"]["auto_offset_reset"],
        ),
        mlflow=MLflowConfig(**raw["mlflow"]),
        monitoring=MonitoringConfig(**raw["monitoring"]),
        postgres=PostgresConfig(**raw["postgres"]),
        services=ServicesConfig(**raw["services"]),
    )


settings: Settings = _load_settings()
