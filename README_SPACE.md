---
title: RecSys Gateway
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Real-Time Recommendation Engine — Gateway

FastAPI serving pipeline (FAISS retrieval → Thompson-Sampling bandit → ranking →
post-ranking) with an embedded Kafka (Redpanda) broker + feedback consumer.
Models are downloaded from HF Hub on startup.

See the source repo for architecture and ADRs.
