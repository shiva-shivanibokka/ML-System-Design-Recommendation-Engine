# Deployment Runbook — Free-Tier Live Deployment

This runbook reproduces the full live deployment of the recommendation engine using
only free tiers. Replace every `<...>` placeholder with your own values.

## Topology

| Component | Host | Free tier |
|---|---|---|
| Next.js frontend | Vercel | Permanent free |
| FastAPI gateway (+ embedded Redpanda/Kafka + consumer) | Hugging Face Spaces (Docker) | Permanent free, 2 vCPU / 16 GB, sleeps after 48h idle |
| Trained models + processed data | Hugging Face Hub (model repo) | Permanent free |
| Redis | Upstash | Permanent free (serverless) |
| Postgres | Neon | Permanent free (0.5 GB) |
| MLflow tracking | DagsHub | Permanent free hosted MLflow |
| Prometheus + Grafana | Grafana Cloud | Permanent free (metrics pushed) |

> Supabase and Render are intentionally not used (free tiers exhausted).

## Secret / env-var reference

Set these as **HF Space secrets** (gateway) unless noted otherwise.

| Name | Provider | Example / notes |
|---|---|---|
| `HF_MODEL_REPO` | HF Hub | `<user>/recsys-artifacts` |
| `HF_TOKEN` | HF | read-scope token (needed only for a private artifact repo) |
| `REDIS_URL` | Upstash | `rediss://default:<pw>@<host>.upstash.io:6379` |
| `DATABASE_URL` | Neon | `postgresql://<user>:<pw>@<host>/<db>?sslmode=require` |
| `MLFLOW_TRACKING_URI` | DagsHub | `https://dagshub.com/<user>/<repo>.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | DagsHub | your DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub | DagsHub token |
| `GRAFANA_PUSH_URL` | Grafana Cloud | Pushgateway/remote endpoint |
| `GRAFANA_PUSH_USER` | Grafana Cloud | numeric instance id |
| `GRAFANA_PUSH_KEY` | Grafana Cloud | API key |
| `CORS_ORIGINS` | — | set to the Vercel URL once known (start `*`) |
| `NEXT_PUBLIC_API_URL` | — | **Vercel** env var = the Space URL |

All are read via env vars only; nothing is committed. Unset values degrade gracefully
(app still boots; the relevant feature is disabled).

## Step 1 — Train models locally (GPU laptop)

```bash
python scripts/download_movielens.py     # -> data/raw
python training/train.py --model all     # -> models/svd, models/ncf, data/indexes, data/processed
ls -lh models/svd models/ncf data/indexes data/processed   # verify non-empty
```

## Step 2 — Upload the artifact bundle to HF Hub

```bash
huggingface-cli login                    # paste HF_TOKEN
huggingface-cli upload <user>/recsys-artifacts models/          models/          --repo-type model
huggingface-cli upload <user>/recsys-artifacts data/indexes/    data/indexes/    --repo-type model
huggingface-cli upload <user>/recsys-artifacts data/processed/  data/processed/  --repo-type model
```

The gateway's `serving/artifacts.py` downloads `models/**`, `data/indexes/**`, and
`data/processed/**` from this repo on startup.

## Step 3 — Provision managed data services

- **Upstash**: create a Redis database → copy `rediss://` URL → `REDIS_URL`.
- **Neon**: create project/db → copy pooled connection string → `DATABASE_URL`.
- **DagsHub**: create/connect repo → copy MLflow URI + token → `MLFLOW_TRACKING_*`.
- **Grafana Cloud**: create a stack → copy Prometheus push URL + instance id + API key
  → `GRAFANA_PUSH_*`. Import `monitoring/grafana/dashboards/recsys_overview.json`.

## Step 4 — Deploy the gateway to Hugging Face Spaces

1. Create a new Space at huggingface.co/new-space, SDK = **Docker**, name `recsys-gateway`.
2. Add all gateway secrets from the table above.
3. The Space expects its metadata in `README.md`; use the contents of `README_SPACE.md`
   (copy `README_SPACE.md` → `README.md` on the Space, or push and rename there).
4. Push the repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/recsys-gateway
   git push space main
   ```
5. Verify: open `https://<user>-recsys-gateway.hf.space/health` → `200` with
   `"models": {"ncf": true, "svd": true}` (confirms the HF Hub download worked).

The container runs `entrypoint.sh`: Redpanda (Kafka-compatible) → `kafka/consumer.py`
→ `uvicorn serving.main:app`.

## Step 5 — Deploy the frontend to Vercel

1. Import the GitHub repo at vercel.com; set **Root Directory** = `frontend`.
2. Add env var `NEXT_PUBLIC_API_URL` = `https://<user>-recsys-gateway.hf.space`.
3. Deploy; open the Vercel URL and confirm recommendations render.
4. Set the Space secret `CORS_ORIGINS` to the exact Vercel URL and restart the Space.

## Step 6 — End-to-end verification

```bash
curl https://<user>-recsys-gateway.hf.space/health
curl -X POST https://<user>-recsys-gateway.hf.space/recommend \
  -H 'content-type: application/json' -d '{"user_id":1,"top_n":5}'
curl "https://<user>-recsys-gateway.hf.space/recommend/1/explain?top_n=3"
```

- Vercel URL serves live recommendations.
- Grafana Cloud dashboard shows request rate + latency after a few calls.
- DagsHub MLflow UI shows training runs.
- GitHub Actions CI is green.

## Known tradeoffs

- **Cold start:** the Space sleeps after 48h idle; first request takes ~30–60s to wake.
- **Embedded Kafka is ephemeral:** the broker + consumer reset when the Space
  sleeps/rebuilds. The local `docker-compose` stack remains the "true distributed"
  reference for the event-streaming path.
- **16 GB RAM** is shared across Redpanda + consumer + gateway + FAISS; fine for
  MovieLens-1M, the main resource to watch if the catalog grows.
