import os
import pytest
from unittest.mock import patch


def test_postgres_password_from_env():
    with patch.dict(os.environ, {"POSTGRES_PASSWORD": "supersecret"}):
        from configs.settings import _load_settings
        s = _load_settings()
        assert s.postgres.password == "supersecret"


def test_database_url_override_takes_precedence():
    custom_url = "postgresql://user:pass@myhost:5432/mydb"
    with patch.dict(os.environ, {"DATABASE_URL": custom_url, "POSTGRES_PASSWORD": "other"}):
        from configs.settings import _load_settings
        s = _load_settings()
        assert s.postgres.url == custom_url


def test_redis_host_from_env():
    with patch.dict(os.environ, {"REDIS_HOST": "my-redis-host"}):
        from configs.settings import _load_settings
        s = _load_settings()
        assert s.redis.host == "my-redis-host"


def test_kafka_servers_from_env():
    with patch.dict(os.environ, {"KAFKA_BOOTSTRAP_SERVERS": "broker1:9092,broker2:9092"}):
        from configs.settings import _load_settings
        s = _load_settings()
        assert s.kafka.bootstrap_servers == "broker1:9092,broker2:9092"


def test_mlflow_uri_from_env():
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://my-mlflow:5001"}):
        from configs.settings import _load_settings
        s = _load_settings()
        assert s.mlflow.tracking_uri == "http://my-mlflow:5001"


def test_postgres_url_constructed_from_components():
    env = {
        "POSTGRES_PASSWORD": "mypass",
        "POSTGRES_USER": "myuser",
        "POSTGRES_HOST": "myhost",
        "POSTGRES_DB": "mydb",
    }
    # Clear DATABASE_URL so it's built from components
    with patch.dict(os.environ, env):
        os.environ.pop("DATABASE_URL", None)
        from configs.settings import _load_settings
        s = _load_settings()
        assert "mypass" in s.postgres.url
        assert "myuser" in s.postgres.url
        assert "myhost" in s.postgres.url


def test_postgres_port_default():
    from configs.settings import _load_settings
    s = _load_settings()
    assert s.postgres.port == 5432
