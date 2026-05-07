"""
Cold-Start Handler.

Two cold-start problems and their solutions:

1. NEW USER cold-start (no interaction history):
   Solution: Popularity-based fallback
   - Serve the top-N globally most-popular items
   - Optionally stratified by genre if user provides a preference
   - After N interactions, switch to collaborative filtering
   - "N" = settings.cold_start.user_min_interactions (default: 5)

   Real-world: Netflix serves "Top 10 in your country" to new users.
               Spotify serves trending tracks until enough listens.

2. NEW ITEM cold-start (item has no interaction data):
   Solution: Content-based fallback using item genre embeddings
   - Embed the item's genre vector (multi-hot, 18 dims) → find similar
     items by cosine similarity → use those similar items' collaborative
     signals as a proxy
   - At serving time, new items are blended into recommendations
     using their content similarity to the user's known preferences
   - "New" = settings.cold_start.item_min_interactions (default: 10)

   Real-world: YouTube uses video metadata (title, description, tags)
               to cold-start new videos before view data accumulates.
               Airbnb uses listing features for new property recommendations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings


class ColdStartHandler:
    """
    Handles cold-start routing for both new users and new items.
    Loaded once at serving startup and kept in memory.
    """

    def __init__(self):
        self._popularity_pool: List[Tuple[int, float]] = []  # (item_id, score)
        self._item_genre_vectors: Dict[int, np.ndarray] = {}  # item_id → 18-dim vector
        self._genre_to_items: Dict[str, List[int]] = {}  # genre → [item_ids]
        self._item_meta: Dict[int, dict] = {}  # item_id → {title, genres, ...}
        self._loaded = False

    def load(self):
        proc = Path(settings.data.processed_dir)

        # Popularity pool: top-N items by interaction count
        item_stats = pd.read_parquet(proc / "item_stats.parquet")
        top_items = item_stats.nlargest(
            settings.cold_start.popularity_top_n, "interaction_count"
        )
        self._popularity_pool = list(
            zip(top_items["item_id"].tolist(), top_items["popularity_score"].tolist())
        )

        # Genre vectors for content-based fallback
        movies = pd.read_parquet(proc / "movies.parquet")
        for _, row in movies.iterrows():
            item_id = int(row["item_id"])
            # genre_vector is stored as numpy array in parquet (object dtype)
            vec = row["genre_vector"]
            if isinstance(vec, np.ndarray):
                self._item_genre_vectors[item_id] = vec
            self._item_meta[item_id] = {
                "title": row.get("title", ""),
                "primary_genre": row.get("primary_genre", "Unknown"),
                "genres": row.get("genres", []),
            }
            genre = row.get("primary_genre", "Unknown")
            self._genre_to_items.setdefault(genre, []).append(item_id)

        self._loaded = True
        print(f"[ColdStart] Loaded {len(self._popularity_pool)} popularity items")
        print(f"[ColdStart] Loaded {len(self._item_genre_vectors)} genre vectors")

    def is_cold_user(self, user_id: int, interaction_count: int) -> bool:
        return interaction_count < settings.cold_start.user_min_interactions

    def is_cold_item(self, item_id: int, interaction_count: int) -> bool:
        return interaction_count < settings.cold_start.item_min_interactions

    def get_popularity_fallback(
        self,
        top_k: int = None,
        exclude_seen: set = None,
        genre_filter: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """
        Returns top-k popular items, optionally filtered by genre.
        Used for new user cold-start.
        """
        top_k = top_k or settings.post_ranking.top_n
        exclude_seen = exclude_seen or set()

        pool = self._popularity_pool
        if genre_filter and genre_filter in self._genre_to_items:
            genre_item_set = set(self._genre_to_items[genre_filter])
            pool = [(iid, score) for iid, score in pool if iid in genre_item_set]

        filtered = [(iid, score) for iid, score in pool if iid not in exclude_seen]
        return filtered[:top_k]

    def get_content_based_fallback(
        self,
        liked_item_ids: List[int],
        top_k: int = None,
        exclude_seen: set = None,
    ) -> List[Tuple[int, float]]:
        """
        Content-based fallback for new items.
        Finds items similar to the user's liked items using genre vectors.
        Used to fill recommendation slots when collaborative models
        haven't seen an item yet.
        """
        top_k = top_k or settings.cold_start.content_sim_top_k
        exclude_seen = exclude_seen or set()

        if not liked_item_ids or not self._item_genre_vectors:
            return self.get_popularity_fallback(top_k, exclude_seen)

        # Average genre vector of liked items
        liked_vecs = [
            self._item_genre_vectors[iid]
            for iid in liked_item_ids
            if iid in self._item_genre_vectors
        ]
        if not liked_vecs:
            return self.get_popularity_fallback(top_k, exclude_seen)

        user_profile = np.mean(liked_vecs, axis=0)
        norm = np.linalg.norm(user_profile)
        if norm > 0:
            user_profile = user_profile / norm

        # Cosine similarity against all items
        similarities = []
        for item_id, vec in self._item_genre_vectors.items():
            if item_id in exclude_seen:
                continue
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            cos_sim = float(user_profile @ (vec / vec_norm))
            similarities.append((item_id, cos_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_item_title(self, item_id: int) -> str:
        return self._item_meta.get(item_id, {}).get("title", f"Item {item_id}")

    def get_item_genre(self, item_id: int) -> str:
        return self._item_meta.get(item_id, {}).get("primary_genre", "Unknown")


# Singleton — loaded once at startup
cold_start_handler = ColdStartHandler()
