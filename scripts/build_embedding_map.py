"""
Build a 2D projection of the NCF item embeddings for the frontend "embedding
galaxy" visualization.

Runs t-SNE (via scikit-learn, no extra deps) over the trained item embeddings,
joins genre + title metadata, and writes data/processed/embedding_map.parquet
with columns [item_id, x, y, genre, title, popularity].

The gateway serves this file at GET /catalog/embedding_map. Precomputing here
keeps the Space startup fast and the projection deterministic.

Run: python scripts/build_embedding_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings  # noqa: E402


def main() -> None:
    proc = Path(settings.data.processed_dir)
    emb = np.load(settings.ncf.item_embedding_path)  # (n_items, 64), row = item_idx
    id_map = pd.read_parquet(proc / "item_id_map.parquet")  # item_id, item_idx
    movies = pd.read_parquet(proc / "movies.parquet")[["item_id", "title", "primary_genre"]]
    item_stats = pd.read_parquet(proc / "item_stats.parquet")[["item_id", "popularity_score"]]

    print(f"[map] embeddings={emb.shape}  items={len(id_map)}")

    # PCA(50) -> t-SNE(2) is the standard, stable recipe.
    n_pca = min(50, emb.shape[1])
    reduced = PCA(n_components=n_pca, random_state=42).fit_transform(emb)
    coords = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        init="pca",
        random_state=42,
    ).fit_transform(reduced)
    print("[map] t-SNE done")

    df = id_map.sort_values("item_idx").reset_index(drop=True)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df = df.merge(movies, on="item_id", how="left").merge(item_stats, on="item_id", how="left")
    df = df.rename(columns={"primary_genre": "genre", "popularity_score": "popularity"})
    df["genre"] = df["genre"].fillna("Unknown")
    df["title"] = df["title"].fillna("Item " + df["item_id"].astype(str))
    df["popularity"] = df["popularity"].fillna(0.0)

    # Normalize coords to a stable [-1, 1] range so the frontend can scale easily.
    for c in ("x", "y"):
        lo, hi = df[c].min(), df[c].max()
        df[c] = 2 * (df[c] - lo) / (hi - lo) - 1

    out = df[["item_id", "x", "y", "genre", "title", "popularity"]]
    dest = proc / "embedding_map.parquet"
    out.to_parquet(dest, index=False)
    print(f"[map] wrote {len(out)} points -> {dest}")


if __name__ == "__main__":
    main()
