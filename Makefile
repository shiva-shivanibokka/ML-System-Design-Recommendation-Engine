.PHONY: help setup data train feast-apply feast-materialize serve test \
        docker-up docker-down bootstrap lint \
        frontend-install frontend-dev frontend-build

PYTHON := python
PIP := pip

help:
	@echo "RecSys Recommendation Engine — available targets:"
	@echo ""
	@echo "  bootstrap         Full first-time setup (setup + data + train + feast-apply + feast-materialize)"
	@echo "  setup             Install Python dependencies"
	@echo "  data              Download and preprocess MovieLens-1M dataset"
	@echo "  train             Train SVD + NeuMF, build FAISS index & t-SNE embedding map"
	@echo "  feast-apply       Register feature definitions in Feast"
	@echo "  feast-materialize Sync offline features → Redis online store"
	@echo "  serve             Start the FastAPI gateway on :8000"
	@echo "  test              Run all unit + integration tests"
	@echo "  docker-up         Start all services (Kafka, Redis, Postgres, Grafana...)"
	@echo "  docker-down       Stop and remove all containers"
	@echo "  lint              Run ruff linter"
	@echo ""
	@echo "  frontend-install  Install Next.js frontend dependencies"
	@echo "  frontend-dev      Start Next.js dev server on :3001"
	@echo "  frontend-build    Build Next.js for production"

setup:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/download_movielens.py

train:
	$(PYTHON) training/train.py --model all
	$(PYTHON) scripts/build_embedding_map.py

feast-apply:
	cd feature_store/feature_repo && feast apply

feast-materialize:
	cd feature_store && $(PYTHON) materialize.py

serve:
	uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short
	$(PYTHON) -m pytest tests/integration/ -v --tb=short

docker-up:
	@test -f .env || (echo "ERROR: .env not found. Copy .env.example to .env and set secrets." && exit 1)
	docker compose up -d --build
	@echo "Services starting. Access points:"
	@echo "  API:      http://localhost:8000"
	@echo "  MLflow:   http://localhost:5001"
	@echo "  Grafana:  http://localhost:3000"
	@echo "  Prometheus: http://localhost:9090"

docker-down:
	docker compose down

bootstrap: setup data train feast-apply feast-materialize
	@echo "Bootstrap complete. Run 'make serve' or 'make docker-up' to start."

lint:
	ruff check .

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev -- --port 3001

frontend-build:
	cd frontend && npm run build
