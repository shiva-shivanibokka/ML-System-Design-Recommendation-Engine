"""
Grafana Cloud metrics push.

The HF Space cannot be reliably scraped by Prometheus (it sleeps and has no
stable inbound scrape target), so instead the gateway pushes its metrics to
Grafana Cloud's hosted Prometheus via the Prometheus Pushgateway protocol on a
background thread. No-op unless GRAFANA_PUSH_URL is set.
"""
from __future__ import annotations

import threading
import time

import structlog
from prometheus_client import push_to_gateway, REGISTRY
from prometheus_client.exposition import basic_auth_handler

log = structlog.get_logger()


def _auth_handler_factory(user: str, key: str):
    def handler(url, method, timeout, headers, data):
        return basic_auth_handler(url, method, timeout, headers, data, user, key)
    return handler


def start_metrics_push(
    url: str | None,
    user: str | None,
    key: str | None,
    interval: int = 30,
) -> bool:
    """Start a daemon thread pushing metrics to Grafana Cloud. No-op if url unset."""
    if not url:
        log.info("metrics_push.skip_no_url")
        return False

    handler = _auth_handler_factory(user or "", key or "") if user else None

    def _loop():
        while True:
            try:
                push_to_gateway(url, job="recsys-gateway", registry=REGISTRY, handler=handler)
            except Exception as e:
                log.debug("metrics_push.failed", error=str(e))
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    log.info("metrics_push.started", url=url)
    return True
