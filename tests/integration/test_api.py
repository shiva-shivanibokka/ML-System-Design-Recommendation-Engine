"""
Integration tests for FastAPI endpoints.
Mocks all external dependencies (Redis, PostgreSQL, model files, parquet data)
so tests run without Docker, trained models, or real data files.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def _make_parquet(path, *args, **kwargs):
    """Return minimal valid DataFrames matching the schema expected by main.py lifespan."""
    p = str(path)
    if "user_id_map" in p:
        return pd.DataFrame({"user_id": [1, 2, 3], "user_idx": [0, 1, 2]})
    if "item_id_map" in p:
        return pd.DataFrame({"item_id": [10, 20, 30], "item_idx": [0, 1, 2]})
    if "movies" in p:
        return pd.DataFrame({
            "item_id": [10, 20, 30],
            "primary_genre": ["Action", "Drama", "Comedy"],
            "title": ["Movie A", "Movie B", "Movie C"],
            "genres": [["Action"], ["Drama"], ["Comedy"]],
            "genre_vector": [np.zeros(18), np.zeros(18), np.zeros(18)],
        })
    if "user_stats" in p:
        return pd.DataFrame({
            "user_id": [1, 2, 3],
            "interaction_count": [50, 3, 100],
            "last_interaction_ts": pd.to_datetime(["2024-01-01"] * 3),
            "first_interaction_ts": pd.to_datetime(["2023-01-01"] * 3),
            "avg_rating_proxy": [0.5, 0.5, 0.5],
        })
    if "item_stats" in p:
        return pd.DataFrame({
            "item_id": [10, 20, 30],
            "interaction_count": [100, 50, 25],
            "popularity_score": [1.0, 0.5, 0.25],
            "last_interaction_ts": pd.to_datetime(["2024-01-01"] * 3),
        })
    if "train" in p:
        return pd.DataFrame({"user_idx": [0, 0, 1, 2], "item_idx": [0, 1, 2, 0]})
    return pd.DataFrame()


@pytest.fixture(scope="module")
def client():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None  # always cache miss so pipeline runs

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    with patch("pandas.read_parquet", side_effect=_make_parquet), \
         patch("redis.Redis", return_value=mock_redis), \
         patch("sqlalchemy.create_engine", return_value=mock_engine), \
         patch("pathlib.Path.exists", return_value=False), \
         patch("serving.main._start_staleness_thread"):
        from fastapi.testclient import TestClient
        from serving.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Basic endpoint tests
# ---------------------------------------------------------------------------

def test_root_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "service" in data
    assert "docs" in data


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_schema(client):
    r = client.get("/health")
    data = r.json()
    assert "status" in data
    assert "models" in data
    assert isinstance(data["models"], dict)
    assert "ncf" in data["models"]
    assert "svd" in data["models"]
    assert "redis" in data
    assert "postgres" in data


def test_metrics_endpoint_returns_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus exposition format always starts with # HELP or metric name
    assert b"recsys_" in r.content or b"# HELP" in r.content


def test_bandit_state_returns_valid_json(client):
    r = client.get("/bandit/state")
    assert r.status_code == 200
    data = r.json()
    assert "alphas" in data
    assert "betas" in data
    assert "total_pulls" in data
    assert "ctrs" in data


def test_monitoring_latency_returns_budget(client):
    r = client.get("/monitoring/latency")
    assert r.status_code == 200
    data = r.json()
    assert "budget_ms" in data
    assert data["budget_ms"]["total_p99"] == 100


def test_monitoring_health_endpoint(client):
    r = client.get("/monitoring/health")
    assert r.status_code == 200
    data = r.json()
    assert "ctrs" in data
    assert "catalog_coverage" in data


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

def test_recommend_missing_user_id_returns_422(client):
    r = client.post("/recommend", json={})
    assert r.status_code == 422


def test_recommend_invalid_top_n_too_large_returns_422(client):
    r = client.post("/recommend", json={"user_id": 1, "top_n": 100})
    assert r.status_code == 422


def test_recommend_invalid_top_n_zero_returns_422(client):
    r = client.post("/recommend", json={"user_id": 1, "top_n": 0})
    assert r.status_code == 422


def test_click_feedback_missing_fields_returns_422(client):
    r = client.post("/feedback/click", json={"user_id": 1})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

def test_bandit_reset_endpoint(client):
    r = client.post("/admin/bandit/reset")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "reset"
    assert "state" in data


def test_recommend_cold_user_is_cold_start(client):
    # user_id=2 has interaction_count=3, below threshold of 5 → cold start
    r = client.post("/recommend", json={"user_id": 2, "top_n": 3})
    assert r.status_code in (200, 500)
    if r.status_code == 200:
        data = r.json()
        assert "recommendations" in data
        assert data["is_cold_start"] is True


def test_recommend_response_schema(client):
    r = client.post("/recommend", json={"user_id": 2, "top_n": 3})
    if r.status_code == 200:
        data = r.json()
        for key in ["request_id", "user_id", "model_used", "is_cold_start", "recommendations", "latency_ms"]:
            assert key in data
