"""
Feast Materialization: sync offline store → online store (Redis).

This runs after training data changes or on a schedule (every 24h).
It pushes the latest user and item features from Parquet files into Redis
so the serving layer can retrieve them in <5ms at inference time.

This is the single most important operation for eliminating training-serving skew:
the SAME feature definitions used during training are materialized into
the online store used at serving time.

Run:
    python feature_store/materialize.py

Or via Airflow DAG (scheduled).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feast import FeatureStore


def materialize(hours_back: int = 24):
    repo_path = Path(__file__).parent / "feature_repo"
    store = FeatureStore(repo_path=str(repo_path))

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(hours=hours_back)

    print(
        f"[materialize] Syncing features from {start_date.isoformat()} → {end_date.isoformat()}"
    )
    print(f"[materialize] Online store: Redis")

    store.materialize(
        start_date=start_date,
        end_date=end_date,
    )
    print("[materialize] Done — features are live in Redis online store.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hours-back", type=int, default=24)
    args = parser.parse_args()
    materialize(hours_back=args.hours_back)
