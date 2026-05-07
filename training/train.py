"""
Training Pipeline: SVD + NCF with MLflow experiment tracking.

Trains both models, evaluates them, registers in MLflow Model Registry,
and builds the FAISS ANN index from NCF item embeddings.

Run:
    python training/train.py --model svd
    python training/train.py --model ncf
    python training/train.py --model all   (default)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings
from training.ncf_model import InteractionDataset, NeuMF, evaluate_ncf
from training.svd_model import SVDRecommender, evaluate


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------


def load_data():
    proc = Path(settings.data.processed_dir)
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    movies = pd.read_parquet(proc / "movies.parquet")
    user_map = pd.read_parquet(proc / "user_id_map.parquet")
    item_map = pd.read_parquet(proc / "item_id_map.parquet")

    n_users = user_map["user_idx"].max() + 1
    n_items = item_map["item_idx"].max() + 1

    # Build user history (for negative sampling + evaluation)
    user_history = train.groupby("user_idx")["item_idx"].apply(set).to_dict()

    print(f"[data] n_users={n_users:,}  n_items={n_items:,}")
    print(f"[data] train={len(train):,}  val={len(val):,}")
    return train, val, movies, user_history, n_users, n_items


# ---------------------------------------------------------------------------
# SVD Training
# ---------------------------------------------------------------------------


def train_svd(train, val, user_history, n_users, n_items):
    print("\n" + "=" * 60)
    print("TRAINING: SVD Baseline")
    print("=" * 60)

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    with mlflow.start_run(run_name="svd-baseline") as run:
        mlflow.log_params(
            {
                "model_type": "SVD",
                "n_components": settings.svd.n_components,
                "n_iter": settings.svd.n_iter,
                "n_users": n_users,
                "n_items": n_items,
                "train_size": len(train),
            }
        )

        model = SVDRecommender()
        t0 = time.time()
        model.fit(train, n_users, n_items)
        train_time = time.time() - t0

        metrics = evaluate(model, val, train, top_k=10)
        mlflow.log_metrics({**metrics, "train_time_sec": train_time})
        model.save()

        # Register in MLflow Model Registry
        mlflow.log_artifact(settings.svd.model_path, artifact_path="model")
        mlflow.register_model(f"runs:/{run.info.run_id}/model", "recsys-svd")
        print(f"[SVD] Training complete in {train_time:.1f}s")
        print(f"[SVD] MLflow run: {run.info.run_id}")
        return model, metrics


# ---------------------------------------------------------------------------
# NCF Training
# ---------------------------------------------------------------------------


def train_ncf(train, val, user_history, n_users, n_items):
    print("\n" + "=" * 60)
    print("TRAINING: NeuMF (Neural Collaborative Filtering)")
    print("=" * 60)

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NCF] Device: {device}")

    train_interactions = train[["user_idx", "item_idx"]].values

    dataset = InteractionDataset(
        interactions=train_interactions,
        n_items=n_items,
        user_history=user_history,
        num_negatives=settings.ncf.num_negatives,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.ncf.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = NeuMF(n_users=n_users, n_items=n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.ncf.learning_rate)
    criterion = nn.BCELoss()

    # Val data for early stopping
    val_interactions = val[["user_idx", "item_idx"]].values

    best_ndcg = 0.0
    patience_count = 0
    best_state = None

    with mlflow.start_run(run_name="ncf-neumf") as run:
        mlflow.log_params(
            {
                "model_type": "NeuMF",
                "embedding_dim": settings.ncf.embedding_dim,
                "mlp_layers": str(settings.ncf.mlp_layers),
                "dropout": settings.ncf.dropout,
                "lr": settings.ncf.learning_rate,
                "batch_size": settings.ncf.batch_size,
                "num_negatives": settings.ncf.num_negatives,
                "n_users": n_users,
                "n_items": n_items,
                "train_size": len(train),
            }
        )

        t_total = time.time()

        for epoch in range(1, settings.ncf.num_epochs + 1):
            model.train()
            total_loss = 0.0
            t_epoch = time.time()

            for users, items, labels in loader:
                users = users.to(device)
                items = items.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                preds = model(users, items)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            epoch_time = time.time() - t_epoch

            # Evaluation (CPU)
            model_cpu = model.cpu()
            metrics = evaluate_ncf(
                model_cpu,
                val_interactions,
                user_history,
                n_items=n_items,
                top_k=10,
                n_eval_users=500,
            )
            model = model_cpu.to(device)

            ndcg = metrics["ndcg_at_10"]
            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "val_hr_at_10": metrics["hr_at_10"],
                    "val_ndcg_at_10": ndcg,
                    "epoch_time_sec": epoch_time,
                },
                step=epoch,
            )

            print(
                f"  Epoch {epoch:02d}/{settings.ncf.num_epochs} | "
                f"loss={avg_loss:.4f} | "
                f"HR@10={metrics['hr_at_10']:.4f} | "
                f"NDCG@10={ndcg:.4f} | "
                f"time={epoch_time:.1f}s"
            )

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= settings.ncf.early_stopping_patience:
                    print(
                        f"[NCF] Early stopping at epoch {epoch} (patience={patience_count})"
                    )
                    break

            # Re-sample negatives each epoch (important for generalization)
            dataset._build_samples()

        total_time = time.time() - t_total

        # Restore best model
        model.load_state_dict(best_state)
        model = model.cpu()
        model.eval()

        mlflow.log_metrics(
            {"best_ndcg_at_10": best_ndcg, "total_train_time_sec": total_time}
        )
        model.save()

        # Save item embeddings for FAISS
        item_embs = model.get_all_item_embeddings()
        Path(settings.ncf.item_embedding_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(settings.ncf.item_embedding_path, item_embs)
        print(
            f"[NCF] Item embeddings saved → {settings.ncf.item_embedding_path}  shape={item_embs.shape}"
        )

        mlflow.log_artifact(settings.ncf.model_path, artifact_path="model")
        mlflow.register_model(f"runs:/{run.info.run_id}/model", "recsys-ncf")
        print(f"[NCF] Best NDCG@10={best_ndcg:.4f}")
        print(f"[NCF] Total training time: {total_time:.1f}s")
        print(f"[NCF] MLflow run: {run.info.run_id}")
        return model, {"ndcg_at_10": best_ndcg}


# ---------------------------------------------------------------------------
# Build FAISS Index
# ---------------------------------------------------------------------------


def build_faiss_index():
    """
    Build FAISS IVF+PQ index from NCF item GMF embeddings.

    This is the Stage 1 retrieval index:
    - At serving time, user's GMF embedding vector is used to query this index
    - Returns top-500 nearest item embeddings in ~20ms
    - IVF partitions the space into n_clusters Voronoi cells
    - PQ compresses each vector to reduce memory (16MB vs 512MB for flat L2)
    """
    import faiss
    import pickle

    print("\n" + "=" * 60)
    print("BUILDING: FAISS IVF+PQ Index (Stage 1 Retrieval)")
    print("=" * 60)

    emb_path = Path(settings.ncf.item_embedding_path)
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Item embeddings not found: {emb_path}. Train NCF first."
        )

    item_embs = np.load(emb_path).astype(np.float32)
    n_items, dim = item_embs.shape
    print(f"[FAISS] Item embeddings: {n_items:,} × {dim} dims")

    # L2-normalize for cosine similarity via inner product
    norms = np.linalg.norm(item_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    item_embs_norm = item_embs / norms

    # Build IVF+PQ index
    n_clusters = settings.faiss.n_clusters
    n_pq = settings.faiss.pq_subquantizers

    print(f"[FAISS] Building IVF{n_clusters},PQ{n_pq} index ...")
    quantizer = faiss.IndexFlatIP(dim)  # Inner product for cosine sim
    index = faiss.IndexIVFPQ(quantizer, dim, n_clusters, n_pq, 8)

    print(f"[FAISS] Training index on {n_items:,} vectors ...")
    index.train(item_embs_norm)
    index.add(item_embs_norm)
    index.nprobe = settings.faiss.n_probe

    # Save
    idx_path = Path(settings.faiss.index_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))

    # Save item_idx → item_id mapping (FAISS returns 0-based indices)
    item_map = pd.read_parquet(
        Path(settings.data.processed_dir) / "item_id_map.parquet"
    )
    id_map = dict(zip(item_map["item_idx"].tolist(), item_map["item_id"].tolist()))
    with open(settings.faiss.docid_map_path, "wb") as f:
        pickle.dump(id_map, f)

    print(f"[FAISS] Index saved → {idx_path}  ({idx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[FAISS] ID map saved → {settings.faiss.docid_map_path}")
    print(
        f"[FAISS] Index contains {index.ntotal:,} vectors, nprobe={settings.faiss.n_probe}"
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["svd", "ncf", "all"], default="all")
    args = parser.parse_args()

    train_df, val_df, movies, user_history, n_users, n_items = load_data()

    if args.model in ("svd", "all"):
        train_svd(train_df, val_df, user_history, n_users, n_items)

    if args.model in ("ncf", "all"):
        train_ncf(train_df, val_df, user_history, n_users, n_items)
        build_faiss_index()

    print("\n[done] Training complete. Models registered in MLflow.")
    print(f"  MLflow UI: {settings.mlflow.tracking_uri}")


if __name__ == "__main__":
    main()
