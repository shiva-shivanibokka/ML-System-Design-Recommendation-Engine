"""
Kafka Event Consumer — buffers interaction events for offline retraining.

Consumes events from the `user-interaction-events` topic and buffers them,
flushing to the Feast offline store as new training rows (on a size or time
threshold, and on shutdown). Offsets are committed only after a successful
flush (at-least-once) so events aren't lost on restart.

(The real-time bandit update + PostgreSQL analytics write happen in the serving
`/feedback/click` endpoint, not here — this consumer only feeds the nightly retrain.)

This closes the data flywheel:
  User interacts → Kafka event → consumer → Feast offline store
  → nightly Airflow DAG picks up new data → retrains model
  → promotes if better → serving layer hot-reloads

In production at Netflix/Spotify:
  - This consumer runs as a separate service (Flink or Spark Structured Streaming)
  - Events are aggregated into 5-minute micro-batches
  - User feature vectors are updated in near-real-time (not nightly)
  - We simulate the simpler nightly batch pattern here for reproducibility

Run:
    python streaming/consumer.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

try:
    from kafka import KafkaConsumer

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

FLUSH_INTERVAL_EVENTS = 500
FLUSH_INTERVAL_SECONDS = 60


def flush_to_feast_offline(events: List[dict]):
    """
    Write buffered click events to the Feast offline store.
    These become new positive training samples for the next retraining run.
    """
    if not events:
        return

    proc = Path(settings.data.processed_dir)
    new_interactions_path = proc / "new_interactions_buffer.parquet"

    # Convert to DataFrame with same schema as train.parquet
    rows = []
    for ev in events:
        if ev.get("event_type") in ("click", "rating"):
            rows.append(
                {
                    "user_id": ev["user_id"],
                    "item_id": ev["item_id"],
                    "timestamp": pd.to_datetime(ev["timestamp"]),
                    "label": 1,
                    "source": "kafka_feedback",
                }
            )

    if not rows:
        return

    new_df = pd.DataFrame(rows)

    # Append to existing buffer (Airflow reads this nightly)
    if new_interactions_path.exists():
        existing = pd.read_parquet(new_interactions_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(new_interactions_path, index=False)
    print(
        f"[Consumer] Flushed {len(rows)} events to Feast offline buffer "
        f"(total={len(combined):,} rows)"
    )


def trigger_nightly_retrain():
    """
    Nightly retraining trigger — fires at 2 AM daily.
    Replaces the Airflow DAG referenced in the architecture docs.
    Only retrains if enough new interactions have buffered since the last run.
    """
    buffer_path = Path(settings.data.processed_dir) / "new_interactions_buffer.parquet"
    if not buffer_path.exists():
        print("[Scheduler] No interaction buffer found — skipping retrain")
        return

    # Snapshot the buffer up-front with an atomic rename so events that arrive
    # from the consumer during the multi-minute retrain accumulate in a fresh
    # buffer instead of being deleted by the old read-then-unlink race.
    snapshot = buffer_path.with_name("new_interactions_retraining.parquet")

    def _restore(src: Path) -> None:
        """Fold a snapshot back into the live buffer so events are never lost."""
        if not src.exists():
            return
        if buffer_path.exists():
            pd.concat(
                [pd.read_parquet(buffer_path), pd.read_parquet(src)], ignore_index=True
            ).to_parquet(buffer_path, index=False)
            src.unlink()
        else:
            src.rename(buffer_path)

    try:
        buffer_path.rename(snapshot)
    except OSError as e:
        print(f"[Scheduler] Could not snapshot buffer ({e}) — skipping")
        return

    buf = pd.read_parquet(snapshot)
    min_new = 100
    if len(buf) < min_new:
        print(f"[Scheduler] Only {len(buf)} new interactions (need {min_new}) — skipping")
        _restore(snapshot)
        return

    print(f"[Scheduler] Triggering retrain on {len(buf)} new interactions")
    result = subprocess.run(
        ["python", "training/train.py", "--model", "all"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("[Scheduler] Retrain complete — clearing snapshot")
        snapshot.unlink()
    else:
        print(f"[Scheduler] Retrain failed:\n{result.stderr}")
        _restore(snapshot)  # keep the events for the next run


def consume():
    if not KAFKA_AVAILABLE:
        print("[Consumer] kafka-python not installed. Exiting.")
        return

    try:
        consumer = KafkaConsumer(
            settings.kafka.topics.user_events,
            bootstrap_servers=settings.kafka.bootstrap_servers,
            group_id=settings.kafka.consumer_group,
            auto_offset_reset=settings.kafka.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            # Commit offsets manually, only after a successful flush, so a crash
            # between poll and flush can't silently drop buffered events.
            enable_auto_commit=False,
        )
        print(f"[Consumer] Listening on {settings.kafka.topics.user_events}")
    except Exception as e:
        print(f"[Consumer] Cannot connect to Kafka: {e}")
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(trigger_nightly_retrain, "cron", hour=2, minute=0)
    scheduler.start()
    print("[Consumer] Nightly retraining scheduler started (fires at 02:00 daily)")

    buffer: list = []
    last_flush = time.time()

    def _flush() -> None:
        nonlocal last_flush
        if buffer:
            flush_to_feast_offline(list(buffer))
            consumer.commit()  # advance offsets only after the buffer is persisted
            buffer.clear()
        last_flush = time.time()

    try:
        # poll() with a timeout returns control on idle, so the time-based flush
        # fires even when no new events arrive (a `for message in consumer:` loop
        # would block indefinitely on a quiet topic).
        while True:
            for records in consumer.poll(timeout_ms=1000).values():
                buffer.extend(m.value for m in records)
            if (
                len(buffer) >= FLUSH_INTERVAL_EVENTS
                or (time.time() - last_flush) >= FLUSH_INTERVAL_SECONDS
            ):
                _flush()
    except KeyboardInterrupt:
        print("[Consumer] Shutting down…")
    finally:
        _flush()  # never lose buffered events on shutdown
        scheduler.shutdown()
        consumer.close()


if __name__ == "__main__":
    consume()
