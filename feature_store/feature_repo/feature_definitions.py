"""
Feast Feature Definitions for the Recommendation Engine.

Two feature views:
  1. user_features     — aggregated per-user stats (interaction count, recency, cold flag)
  2. item_features     — aggregated per-item stats (popularity, genre, cold flag)

Point-in-time correct joins are used during training to prevent leakage:
  - For each (user, item, timestamp) training pair, only features that existed
    BEFORE the timestamp are joined. This prevents using future interactions
    as features for past predictions — a silent but common production bug.

Online store:
  - At serving time, features are retrieved from Redis in <5ms
  - The same feature computation code is used for both training and serving
    → eliminates training-serving skew

Run after preprocessing:
    cd feature_store/feature_repo && feast apply
    python feature_store/materialize.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import Entity, Feature, FeatureStore, FeatureView, FileSource, ValueType
from feast.data_format import ParquetFormat

# ---------------------------------------------------------------------------
# Entities (primary keys)
# ---------------------------------------------------------------------------

user_entity = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Unique user identifier from MovieLens 1M",
)

item_entity = Entity(
    name="item_id",
    value_type=ValueType.INT64,
    description="Unique item (movie) identifier from MovieLens 1M",
)

# ---------------------------------------------------------------------------
# Data Sources (Parquet files — offline store)
# ---------------------------------------------------------------------------

_base = Path(__file__).parent.parent.parent / "data" / "processed"

user_stats_source = FileSource(
    path=str(_base / "user_stats.parquet"),
    event_timestamp_column="last_interaction_ts",
    created_timestamp_column="first_interaction_ts",
)

item_stats_source = FileSource(
    path=str(_base / "item_stats.parquet"),
    event_timestamp_column="last_interaction_ts",
)

# ---------------------------------------------------------------------------
# Feature Views
# ---------------------------------------------------------------------------

user_feature_view = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=timedelta(days=7),  # features expire after 7 days without refresh
    features=[
        Feature(name="interaction_count", dtype=ValueType.INT64),
        Feature(name="is_cold_user", dtype=ValueType.INT32),
        Feature(name="avg_rating_proxy", dtype=ValueType.DOUBLE),
    ],
    source=user_stats_source,
    tags={"team": "recsys", "tier": "online"},
)

item_feature_view = FeatureView(
    name="item_features",
    entities=["item_id"],
    ttl=timedelta(days=30),  # item features are more stable
    features=[
        Feature(name="interaction_count", dtype=ValueType.INT64),
        Feature(name="is_cold_item", dtype=ValueType.INT32),
        Feature(name="popularity_score", dtype=ValueType.DOUBLE),
        Feature(name="primary_genre", dtype=ValueType.STRING),
    ],
    source=item_stats_source,
    tags={"team": "recsys", "tier": "online"},
)
