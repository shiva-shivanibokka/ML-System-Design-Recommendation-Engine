"""
Download and preprocess MovieLens 1M dataset.

MovieLens 1M:
  - 1,000,209 ratings from 6,040 users on 3,706 movies
  - Ratings: 1-5 stars
  - Files: ratings.dat, movies.dat, users.dat

Outputs (all Parquet):
  data/processed/ratings.parquet       — user_id, item_id, rating, timestamp
  data/processed/movies.parquet        — item_id, title, genres (list)
  data/processed/users.parquet         — user_id, gender, age, occupation, zip
  data/processed/interactions.parquet  — implicit feedback (rating >= 4 = positive)
  data/processed/item_content.parquet  — item_id, genre_vector (multi-hot)
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet

Run: python scripts/download_movielens.py
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

RAW_DIR = Path(settings.data.raw_dir)
PROC_DIR = Path(settings.data.processed_dir)
URL = settings.data.movielens_url

ALL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(ALL_GENRES)}


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ml-1m.zip"
    if zip_path.exists():
        print(f"[download] Already exists: {zip_path}")
        return zip_path
    print(f"[download] Fetching {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)
    print(f"[download] Saved → {zip_path}")
    return zip_path


def extract(zip_path: Path) -> Path:
    extract_dir = RAW_DIR / "ml-1m"
    if extract_dir.exists():
        print(f"[extract] Already extracted: {extract_dir}")
        return extract_dir
    print(f"[extract] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    print(f"[extract] Done → {extract_dir}")
    return extract_dir


def load_ratings(ml_dir: Path) -> pd.DataFrame:
    path = ml_dir / "ratings.dat"
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    print(
        f"[ratings] {len(df):,} rows, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items"
    )
    return df


def load_movies(ml_dir: Path) -> pd.DataFrame:
    path = ml_dir / "movies.dat"
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres_raw"],
        encoding="latin-1",
    )
    df["item_id"] = df["item_id"].astype(int)
    df["genres"] = df["genres_raw"].apply(lambda x: x.split("|") if pd.notna(x) else [])
    df["primary_genre"] = df["genres"].apply(lambda g: g[0] if g else "Unknown")

    # Multi-hot genre vector
    def genre_vector(genres):
        vec = np.zeros(len(ALL_GENRES), dtype=np.float32)
        for g in genres:
            if g in GENRE_TO_IDX:
                vec[GENRE_TO_IDX[g]] = 1.0
        return vec

    df["genre_vector"] = df["genres"].apply(genre_vector)
    df = df.drop(columns=["genres_raw"])
    print(f"[movies] {len(df):,} movies, {len(ALL_GENRES)} genre dims")
    return df


def load_users(ml_dir: Path) -> pd.DataFrame:
    path = ml_dir / "users.dat"
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    df["user_id"] = df["user_id"].astype(int)
    print(f"[users] {len(df):,} users")
    return df


def build_implicit_feedback(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit positive interactions.
    Rating >= 4  → positive interaction (label=1)
    Rating < 4   → ignored (not used as explicit negatives — NCF samples negatives randomly)

    This mirrors Netflix/Spotify treatment: watch/listen = positive signal,
    not-watch ≠ explicit negative.
    """
    positive = ratings[ratings["rating"] >= 4.0].copy()
    positive["label"] = 1
    positive = positive[["user_id", "item_id", "timestamp", "label"]]
    positive = positive.sort_values("timestamp").reset_index(drop=True)
    print(f"[implicit] {len(positive):,} positive interactions (rating >= 4)")
    return positive


def chronological_split(
    interactions: pd.DataFrame, train_r: float, val_r: float
) -> tuple:
    """
    Chronological leave-k-out split per user.
    Last interaction  → test
    Second-to-last    → val
    Everything else   → train

    This is the standard evaluation protocol for sequential recommendation
    (used in BERT4Rec, SASRec, and most RecSys papers).
    """
    interactions = interactions.sort_values(["user_id", "timestamp"])
    train_rows, val_rows, test_rows = [], [], []

    for uid, group in interactions.groupby("user_id"):
        n = len(group)
        if n < 3:
            train_rows.append(group)
            continue
        test_rows.append(group.iloc[[-1]])
        val_rows.append(group.iloc[[-2]])
        train_rows.append(group.iloc[:-2])

    train = pd.concat(train_rows).reset_index(drop=True)
    val = pd.concat(val_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)
    print(f"[split] train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def compute_user_stats(interactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate user-level features for Feast."""
    stats = (
        interactions.groupby("user_id")
        .agg(
            interaction_count=("item_id", "count"),
            avg_rating_proxy=("label", "mean"),  # all 1s here, kept for schema
            last_interaction_ts=("timestamp", "max"),
            first_interaction_ts=("timestamp", "min"),
        )
        .reset_index()
    )
    stats["is_cold_user"] = (
        stats["interaction_count"] < settings.cold_start.user_min_interactions
    ).astype(int)
    return stats


def compute_item_stats(
    interactions: pd.DataFrame, movies: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate item-level features for Feast."""
    stats = (
        interactions.groupby("item_id")
        .agg(
            interaction_count=("user_id", "count"),
            last_interaction_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    stats = stats.merge(movies[["item_id", "primary_genre"]], on="item_id", how="left")
    stats["is_cold_item"] = (
        stats["interaction_count"] < settings.cold_start.item_min_interactions
    ).astype(int)
    # Popularity score: log-normalized interaction count
    stats["popularity_score"] = np.log1p(stats["interaction_count"])
    stats["popularity_score"] = (
        stats["popularity_score"] / stats["popularity_score"].max()
    ).round(4)
    return stats


def remap_ids(
    ratings: pd.DataFrame,
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple:
    """
    Remap user_id and item_id to contiguous 0-based integers.
    Required by NCF embedding layers.
    Returns remapped DataFrames + id mapping dicts.
    """
    unique_users = sorted(interactions["user_id"].unique())
    unique_items = sorted(interactions["item_id"].unique())

    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}

    for df in [ratings, interactions]:
        df["user_idx"] = df["user_id"].map(user2idx)
        df["item_idx"] = df["item_id"].map(item2idx)

    users["user_idx"] = users["user_id"].map(user2idx)
    movies["item_idx"] = movies["item_id"].map(item2idx)

    n_users = len(unique_users)
    n_items = len(unique_items)
    print(f"[remap] {n_users:,} users → idx 0..{n_users - 1}")
    print(f"[remap] {n_items:,} items → idx 0..{n_items - 1}")
    return ratings, interactions, users, movies, user2idx, item2idx


def save_all(
    ratings,
    movies,
    users,
    interactions,
    train,
    val,
    test,
    user_stats,
    item_stats,
    user2idx,
    item2idx,
):
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    ratings.to_parquet(PROC_DIR / "ratings.parquet", index=False)
    movies.to_parquet(PROC_DIR / "movies.parquet", index=False)
    users.to_parquet(PROC_DIR / "users.parquet", index=False)
    interactions.to_parquet(PROC_DIR / "interactions.parquet", index=False)
    train.to_parquet(PROC_DIR / "train.parquet", index=False)
    val.to_parquet(PROC_DIR / "val.parquet", index=False)
    test.to_parquet(PROC_DIR / "test.parquet", index=False)
    user_stats.to_parquet(PROC_DIR / "user_stats.parquet", index=False)
    item_stats.to_parquet(PROC_DIR / "item_stats.parquet", index=False)

    # Save id maps as parquet for easy lookup
    pd.DataFrame(list(user2idx.items()), columns=["user_id", "user_idx"]).to_parquet(
        PROC_DIR / "user_id_map.parquet", index=False
    )
    pd.DataFrame(list(item2idx.items()), columns=["item_id", "item_idx"]).to_parquet(
        PROC_DIR / "item_id_map.parquet", index=False
    )

    print(f"[save] All files written to {PROC_DIR}/")
    for f in sorted(PROC_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {size_mb:.1f} MB")


def main():
    zip_path = download()
    ml_dir = extract(zip_path)

    ratings = load_ratings(ml_dir)
    movies = load_movies(ml_dir)
    users = load_users(ml_dir)

    interactions = build_implicit_feedback(ratings)
    ratings, interactions, users, movies, user2idx, item2idx = remap_ids(
        ratings, interactions, users, movies
    )

    train, val, test = chronological_split(
        interactions,
        settings.data.train_ratio,
        settings.data.val_ratio,
    )

    user_stats = compute_user_stats(interactions)
    item_stats = compute_item_stats(interactions, movies)

    save_all(
        ratings,
        movies,
        users,
        interactions,
        train,
        val,
        test,
        user_stats,
        item_stats,
        user2idx,
        item2idx,
    )
    print("\n[done] Preprocessing complete.")


if __name__ == "__main__":
    main()
