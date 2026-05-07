"""
Model Staleness Detection and Monitoring.

Three signals tracked continuously:

1. CTR (Click-Through Rate) per model arm
   - Rolling 7-day CTR computed from click log
   - Alert if current CTR < 80% of 7-day baseline
   - Triggers Airflow retraining DAG if sustained for > 6 hours

2. Catalog Coverage
   - What % of the total item catalog is being recommended?
   - Coverage collapse: model recommends same 50 items to everyone
   - This happens when embeddings collapse or popularity bias dominates
   - Alert if < 10% of catalog recommended in a 24h window

3. Score Distribution Shift (PSI — Population Stability Index)
   - Monitors the distribution of recommendation scores over time
   - PSI > 0.2 indicates significant distribution shift → model drift
   - The same metric used in credit risk models (Evidently AI, WhyLabs)

Prometheus Metrics exposed:
  - recsys_ctr_by_model{model="svd/ncf"}     (Gauge)
  - recsys_catalog_coverage                   (Gauge)
  - recsys_psi_score                          (Gauge)
  - recsys_recommendation_count_total         (Counter)
  - recsys_pipeline_latency_ms{stage="..."}   (Histogram)
  - recsys_cache_hits_total                   (Counter)
  - recsys_cache_misses_total                 (Counter)
  - recsys_cold_start_total{type="user/item"} (Counter)
  - recsys_bandit_pulls_total{model="..."}    (Counter)

All metrics follow Prometheus naming conventions and are scraped every 15s.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings


# ---------------------------------------------------------------------------
# Prometheus Metric Definitions
# ---------------------------------------------------------------------------

# Latency histograms per pipeline stage
# Latency budget: total p99 < 100ms (Netflix/YouTube standard)
LATENCY_BUCKETS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]  # ms

PIPELINE_LATENCY = Histogram(
    "recsys_pipeline_latency_ms",
    "Latency per pipeline stage in milliseconds",
    ["stage"],
    buckets=LATENCY_BUCKETS,
)

# CTR per model
CTR_GAUGE = Gauge(
    "recsys_ctr_by_model",
    "Rolling 7-day click-through rate by model",
    ["model"],
)

# Catalog coverage
COVERAGE_GAUGE = Gauge(
    "recsys_catalog_coverage",
    "Fraction of item catalog recommended in last 24 hours",
)

# PSI score (score distribution shift)
PSI_GAUGE = Gauge(
    "recsys_psi_score",
    "Population Stability Index of recommendation scores (drift signal)",
    ["model"],
)

# Staleness alert (1 = stale, 0 = healthy)
STALENESS_ALERT = Gauge(
    "recsys_staleness_alert",
    "1 if model staleness detected, 0 otherwise",
    ["signal"],  # "ctr", "coverage", "psi"
)

# Throughput counters
RECOMMENDATION_COUNTER = Counter(
    "recsys_recommendation_count_total",
    "Total recommendation requests served",
    ["model", "cold_start"],
)

CACHE_HITS = Counter("recsys_cache_hits_total", "Redis cache hits")
CACHE_MISSES = Counter("recsys_cache_misses_total", "Redis cache misses")

COLD_START_COUNTER = Counter(
    "recsys_cold_start_total",
    "Cold-start fallback invocations",
    ["type"],  # "user" or "item"
)

BANDIT_PULLS = Counter(
    "recsys_bandit_pulls_total",
    "Total Thompson Sampling arm pulls",
    ["model"],
)

BANDIT_REWARDS = Counter(
    "recsys_bandit_rewards_total",
    "Total Thompson Sampling arm rewards (clicks)",
    ["model"],
)

# ------------------------------------------------------------------
# Stage names for latency tracking
# ------------------------------------------------------------------
STAGE_CACHE_CHECK = "cache_check"
STAGE_FEATURE_FETCH = "feature_fetch"
STAGE_CANDIDATE_GEN = "candidate_generation"  # Stage 1: FAISS ANN
STAGE_RANKING = "ranking"  # Stage 2: NCF/SVD rerank
STAGE_POST_RANKING = "post_ranking"  # MMR + freshness + rules
STAGE_TOTAL = "total"


# ---------------------------------------------------------------------------
# Staleness Detector
# ---------------------------------------------------------------------------


class StalenessDetector:
    """
    Continuously monitors three staleness signals.
    Runs as a background thread in the serving process.
    """

    def __init__(self):
        self.click_log: Deque[Tuple[float, str, bool]] = deque(
            maxlen=100_000
        )  # (timestamp, model_name, clicked)
        self.score_log: Deque[Tuple[float, str, float]] = deque(
            maxlen=50_000
        )  # (timestamp, model_name, score)
        self.recommended_items: Deque[Tuple[float, int]] = deque(
            maxlen=100_000
        )  # (timestamp, item_id) — for coverage tracking
        self._n_total_items: int = 0
        self._baseline_ctrs: Dict[str, float] = {}  # computed at startup

    def set_total_items(self, n: int):
        self._n_total_items = n

    # ------------------------------------------------------------------
    # Log events
    # ------------------------------------------------------------------

    def log_click(self, model_name: str, clicked: bool):
        self.click_log.append((time.time(), model_name, clicked))

    def log_score(self, model_name: str, score: float):
        self.score_log.append((time.time(), model_name, score))

    def log_recommendations(self, item_ids: List[int]):
        ts = time.time()
        for iid in item_ids:
            self.recommended_items.append((ts, iid))

    # ------------------------------------------------------------------
    # CTR monitoring
    # ------------------------------------------------------------------

    def compute_rolling_ctr(self, model_name: str, window_hours: int = None) -> float:
        window_hours = window_hours or settings.monitoring.ctr_window_hours
        cutoff = time.time() - (window_hours * 3600)
        relevant = [
            (ts, m, c) for ts, m, c in self.click_log if ts > cutoff and m == model_name
        ]
        if not relevant:
            return 0.0
        total = len(relevant)
        clicks = sum(1 for _, _, c in relevant if c)
        return clicks / total

    def check_ctr_staleness(self) -> Dict[str, bool]:
        """Returns {model: is_stale} for each model."""
        results = {}
        for model in settings.bandit.models:
            ctr = self.compute_rolling_ctr(model)
            CTR_GAUGE.labels(model=model).set(ctr)

            baseline = self._baseline_ctrs.get(model, ctr)  # first reading is baseline
            if not self._baseline_ctrs.get(model):
                self._baseline_ctrs[model] = ctr

            is_stale = (
                baseline > 0.01
                and ctr < baseline * settings.monitoring.ctr_alert_threshold
            )
            STALENESS_ALERT.labels(signal=f"ctr_{model}").set(1 if is_stale else 0)
            results[model] = is_stale

            if is_stale:
                print(
                    f"[Monitor] STALENESS ALERT: {model} CTR dropped "
                    f"({ctr:.4f} < {baseline * settings.monitoring.ctr_alert_threshold:.4f})"
                )
        return results

    # ------------------------------------------------------------------
    # Coverage monitoring
    # ------------------------------------------------------------------

    def compute_catalog_coverage(self, window_hours: int = 24) -> float:
        """
        Fraction of total catalog recommended in the last window_hours.
        Coverage < 10% = model is recommending the same items to everyone.
        """
        if self._n_total_items == 0:
            return 1.0
        cutoff = time.time() - (window_hours * 3600)
        recent_ids = {iid for ts, iid in self.recommended_items if ts > cutoff}
        coverage = len(recent_ids) / self._n_total_items
        COVERAGE_GAUGE.set(coverage)
        is_stale = coverage < settings.monitoring.coverage_alert_threshold
        STALENESS_ALERT.labels(signal="coverage").set(1 if is_stale else 0)
        if is_stale:
            print(
                f"[Monitor] COVERAGE COLLAPSE: only {coverage:.2%} of catalog "
                f"recommended in last {window_hours}h"
            )
        return coverage

    # ------------------------------------------------------------------
    # PSI (score distribution shift)
    # ------------------------------------------------------------------

    def compute_psi(
        self,
        model_name: str,
        reference_window_hours: int = 48,
        current_window_hours: int = 6,
    ) -> float:
        """
        Population Stability Index (PSI).
        Compares current score distribution vs. reference (48h ago).
        PSI < 0.1 = stable, 0.1-0.2 = moderate shift, > 0.2 = significant drift.
        """
        now = time.time()
        ref_cutoff = now - (reference_window_hours * 3600)
        cur_cutoff = now - (current_window_hours * 3600)

        reference = [
            s
            for ts, m, s in self.score_log
            if ref_cutoff < ts <= cur_cutoff and m == model_name
        ]
        current = [
            s for ts, m, s in self.score_log if ts > cur_cutoff and m == model_name
        ]

        if len(reference) < 100 or len(current) < 50:
            return 0.0  # insufficient data

        bins = np.linspace(0, 1, 11)  # 10 equal-width bins over [0,1] scores
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)

        # Clip to avoid log(0)
        ref_hist = np.clip(ref_hist, 1e-10, None)
        cur_hist = np.clip(cur_hist, 1e-10, None)

        # Normalize to proportions
        ref_prop = ref_hist / ref_hist.sum()
        cur_prop = cur_hist / cur_hist.sum()

        psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
        PSI_GAUGE.labels(model=model_name).set(psi)
        is_stale = psi > 0.2
        STALENESS_ALERT.labels(signal=f"psi_{model_name}").set(1 if is_stale else 0)
        if is_stale:
            print(
                f"[Monitor] PSI DRIFT ALERT: {model_name} PSI={psi:.4f} (threshold=0.2)"
            )
        return psi

    # ------------------------------------------------------------------
    # Full health check
    # ------------------------------------------------------------------

    def run_health_check(self) -> dict:
        """Called every `staleness_check_interval_hours` by background thread."""
        ctrs = {m: self.compute_rolling_ctr(m) for m in settings.bandit.models}
        ctr_alerts = self.check_ctr_staleness()
        coverage = self.compute_catalog_coverage()
        psi_scores = {m: self.compute_psi(m) for m in settings.bandit.models}

        return {
            "ctrs": ctrs,
            "ctr_alerts": ctr_alerts,
            "catalog_coverage": coverage,
            "coverage_alert": coverage < settings.monitoring.coverage_alert_threshold,
            "psi_scores": psi_scores,
            "psi_alerts": {m: s > 0.2 for m, s in psi_scores.items()},
            "any_alert": (
                any(ctr_alerts.values())
                or coverage < settings.monitoring.coverage_alert_threshold
                or any(s > 0.2 for s in psi_scores.values())
            ),
        }


# Singleton
staleness_detector = StalenessDetector()
