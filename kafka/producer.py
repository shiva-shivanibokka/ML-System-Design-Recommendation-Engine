"""
Kafka Event Producer — Simulates User Interaction Events.

Produces synthetic user events to the Kafka topic `user-interaction-events`.
In production: your frontend/app sends these events directly to Kafka
via a client SDK whenever a user clicks, rates, or watches.

Event schema:
{
  "event_id":    "uuid",
  "event_type":  "click" | "rating" | "watch",
  "user_id":     int,
  "item_id":     int,
  "timestamp":   ISO-8601 string,
  "context": {
    "model_used":     "svd" | "ncf" | "popularity_fallback",
    "rank_shown":     int,        # position in recommendation list (1-indexed)
    "session_id":     "uuid",
    "device_type":    "mobile" | "desktop" | "tv",
  }
}

Usage:
  # Simulate 1000 events at 10 events/second:
  python kafka/producer.py --n-events 1000 --rate 10

  # Replay historical interactions from MovieLens:
  python kafka/producer.py --mode historical
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

try:
    from kafka import KafkaProducer

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("[Producer] kafka-python not installed — using mock producer (print only)")


class MockProducer:
    """Fallback when Kafka is not running — prints events to stdout."""

    def send(self, topic, value):
        msg = json.loads(value.decode())
        print(
            f"[MOCK] topic={topic} user={msg['user_id']} item={msg['item_id']} "
            f"type={msg['event_type']} model={msg['context'].get('model_used', '?')}"
        )
        return self

    def flush(self):
        pass

    def close(self):
        pass


def create_producer():
    if not KAFKA_AVAILABLE:
        return MockProducer()
    try:
        producer = KafkaProducer(
            bootstrap_servers=settings.kafka.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,
            acks="all",
        )
        print(f"[Producer] Connected to Kafka: {settings.kafka.bootstrap_servers}")
        return producer
    except Exception as e:
        print(f"[Producer] Kafka not available ({e}), using mock producer")
        return MockProducer()


def make_event(
    user_id: int,
    item_id: int,
    event_type: str = "click",
    model_used: str = "ncf",
    rank_shown: int = 1,
    session_id: str = None,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "user_id": user_id,
        "item_id": item_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": {
            "model_used": model_used,
            "rank_shown": rank_shown,
            "session_id": session_id or str(uuid.uuid4()),
            "device_type": random.choice(["mobile", "desktop", "tv"]),
        },
    }


def simulate_realtime(n_events: int = 1000, rate: float = 10.0):
    """
    Simulate user interaction events at `rate` events per second.
    Uses random users and items from the MovieLens 1M ID range.
    """
    proc = Path(settings.data.processed_dir)
    user_map = pd.read_parquet(proc / "user_id_map.parquet")
    item_map = pd.read_parquet(proc / "item_id_map.parquet")

    user_ids = user_map["user_id"].tolist()
    item_ids = item_map["item_id"].tolist()
    models = ["svd", "ncf"]

    producer = create_producer()
    topic = settings.kafka.topics.user_events
    interval = 1.0 / rate

    print(f"[Producer] Simulating {n_events} events at {rate}/sec → topic={topic}")
    sent = 0
    try:
        for i in range(n_events):
            user_id = random.choice(user_ids)
            item_id = random.choice(item_ids)
            model = random.choice(models)
            rank = random.randint(1, 10)
            # Click probability: higher-ranked items clicked more often
            event_type = "click" if random.random() < (0.15 / rank) else "impression"
            event = make_event(user_id, item_id, event_type, model, rank)
            producer.send(topic, value=event)
            sent += 1
            if sent % 100 == 0:
                print(f"[Producer] Sent {sent}/{n_events} events")
            time.sleep(interval)
    finally:
        producer.flush()
        producer.close()

    print(f"[Producer] Done. Sent {sent} events to {topic}")


def replay_historical():
    """
    Replay actual MovieLens 1M interactions as Kafka events.
    Useful for testing the full pipeline with real data.
    """
    proc = Path(settings.data.processed_dir)
    train = pd.read_parquet(proc / "train.parquet")
    user_map = pd.read_parquet(proc / "user_id_map.parquet")
    item_map = pd.read_parquet(proc / "item_id_map.parquet")

    idx_to_user = dict(zip(user_map["user_idx"], user_map["user_id"]))
    idx_to_item = dict(zip(item_map["item_idx"], item_map["item_id"]))

    producer = create_producer()
    topic = settings.kafka.topics.user_events

    print(
        f"[Producer] Replaying {len(train):,} historical interactions → topic={topic}"
    )
    for i, row in train.iterrows():
        user_id = idx_to_user.get(int(row["user_idx"]))
        item_id = idx_to_item.get(int(row["item_idx"]))
        if user_id is None or item_id is None:
            continue
        event = make_event(
            user_id=int(user_id),
            item_id=int(item_id),
            event_type="rating",
            model_used="historical",
            rank_shown=0,
        )
        producer.send(topic, value=event)
        if i % 10_000 == 0:
            print(f"[Producer] Replayed {i:,}/{len(train):,}")
            producer.flush()

    producer.flush()
    producer.close()
    print("[Producer] Historical replay complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["realtime", "historical"], default="realtime"
    )
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    if args.mode == "historical":
        replay_historical()
    else:
        simulate_realtime(n_events=args.n_events, rate=args.rate)
