#!/usr/bin/env bash
set -euo pipefail

# 1. Start Redpanda (Kafka-compatible) in the background, single-node dev mode
redpanda start \
  --overprovisioned --smp 1 --memory 512M --reserve-memory 0M \
  --node-id 0 --check=false \
  --kafka-addr PLAINTEXT://0.0.0.0:9092 \
  --advertise-kafka-addr PLAINTEXT://localhost:9092 &

# 2. Wait for the broker port to accept connections
echo "[entrypoint] waiting for Kafka broker on :9092"
for i in $(seq 1 30); do
  if (echo > /dev/tcp/localhost/9092) >/dev/null 2>&1; then break; fi
  sleep 1
done

# 3. Start the Kafka consumer in the background.
# NOTE: invoke as a SCRIPT (python kafka/consumer.py), not `python -m kafka.consumer`.
# The local kafka/ dir shadows the installed kafka-python package under `-m`, which
# would break `from kafka import KafkaConsumer`. Running the script keeps the package importable.
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
python kafka/consumer.py &

# 4. Foreground: the FastAPI gateway
exec uvicorn serving.main:app --host 0.0.0.0 --port 8000
