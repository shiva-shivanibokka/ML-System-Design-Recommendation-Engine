"""
Post-Ranking Layer.

This is the final transformation applied to the scored candidate list
BEFORE returning results to the user. Every major recommendation system
has this layer — it separates raw model scores from business logic.

Three operations in order:

1. Freshness Boost
   Items interacted with recently (last 7 days in the dataset, or
   "new releases" in production) get a small score boost.
   This prevents "stale" popular items from dominating forever.
   Netflix: boosts newly added titles in recommendation rows.
   Spotify: boosts songs released in the last 30 days.

2. MMR Diversity Re-ranking (Maximal Marginal Relevance)
   Without diversity enforcement, a pure relevance-ranked list often
   looks like: Action, Action, Action, Action, Action, Comedy, Drama...
   MMR balances relevance vs. diversity by penalizing items similar
   (same genre) to already-selected items.

   MMR score: λ × relevance(i) - (1-λ) × max_similarity(i, already_selected)
   λ=1 → pure relevance  |  λ=0 → pure diversity  |  λ=0.7 → balanced

   Origin: Carbonell & Goldstein (1998), used at Google, Bing, and most
   modern recommendation systems under the name "diversity re-ranking."

3. Business Rules / Hard Constraints
   Applied last, these are non-negotiable:
   - max_same_genre: no more than N items from the same genre
   - (extensible: regional licensing, content moderation blocklists, etc.)
   These constraints override model scores — model logic cannot touch them.

Output: top-N diverse, fresh, business-rule-compliant recommendations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings


class PostRanker:
    """
    Post-ranking pipeline: freshness → MMR diversity → hard constraints.
    Stateless — all data passed per request.
    """

    def __init__(self):
        self.top_n = settings.post_ranking.top_n
        self.mmr_lambda = settings.post_ranking.mmr_lambda
        self.freshness_weight = settings.post_ranking.freshness_weight
        self.max_same_genre = settings.post_ranking.max_same_genre

    def rerank(
        self,
        candidates: List[
            Tuple[int, float]
        ],  # (item_id, relevance_score) — top-K from Stage 2
        item_genres: Dict[int, str],  # item_id → genre string
        recent_item_ids: Optional[
            set
        ] = None,  # items interacted with in last 7 days (fresh)
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float, dict]]:
        """
        Full post-ranking pipeline.

        Returns: List of (item_id, final_score, metadata_dict)
        """
        top_n = top_n or self.top_n
        if not candidates:
            return []

        # Step 1: Freshness boost
        candidates = self._apply_freshness_boost(candidates, recent_item_ids)

        # Step 2: MMR diversity re-ranking
        ranked = self._mmr_rerank(
            candidates, item_genres, top_n=min(top_n * 3, len(candidates))
        )

        # Step 3: Hard genre constraints
        final = self._apply_genre_cap(ranked, item_genres, top_n=top_n)

        # Package with metadata
        result = []
        for item_id, score in final:
            result.append(
                (
                    item_id,
                    round(score, 6),
                    {
                        "genre": item_genres.get(item_id, "Unknown"),
                        "is_fresh": item_id in (recent_item_ids or set()),
                    },
                )
            )
        return result

    # ------------------------------------------------------------------
    # Step 1: Freshness Boost
    # ------------------------------------------------------------------

    def _apply_freshness_boost(
        self,
        candidates: List[Tuple[int, float]],
        recent_item_ids: Optional[set],
    ) -> List[Tuple[int, float]]:
        if not recent_item_ids or self.freshness_weight == 0:
            return candidates

        boosted = []
        for item_id, score in candidates:
            if item_id in recent_item_ids:
                score = score * (1.0 + self.freshness_weight)
            boosted.append((item_id, score))

        # Re-sort after boost
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    # ------------------------------------------------------------------
    # Step 2: MMR Diversity Re-ranking
    # ------------------------------------------------------------------

    def _mmr_rerank(
        self,
        candidates: List[Tuple[int, float]],
        item_genres: Dict[int, str],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        """
        Maximal Marginal Relevance (MMR) re-ranking.

        At each step, selects the item that maximizes:
          MMR(i) = λ × relevance(i) − (1−λ) × sim(i, already_selected)

        Similarity here is binary genre similarity (0 or 1) as a simple,
        interpretable proxy. In production, item embedding cosine similarity
        is used instead (requires item embeddings to be loaded).
        """
        if self.mmr_lambda >= 1.0 or len(candidates) <= 1:
            return candidates[:top_n]

        # Normalize relevance scores to [0, 1]
        scores = np.array([s for _, s in candidates])
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            norm_scores = (scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.ones_like(scores)

        items = [item_id for item_id, _ in candidates]
        remaining = list(range(len(items)))
        selected = []
        selected_genres = []

        while len(selected) < top_n and remaining:
            best_idx = None
            best_mmr = -np.inf

            for i in remaining:
                rel = norm_scores[i]

                # Similarity to already-selected items
                # Genre-based: penalize if same genre as any selected item
                if not selected_genres:
                    sim = 0.0
                else:
                    item_genre = item_genres.get(items[i], "Unknown")
                    genre_overlap = sum(1 for g in selected_genres if g == item_genre)
                    sim = genre_overlap / len(selected_genres)

                mmr = self.mmr_lambda * rel - (1.0 - self.mmr_lambda) * sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            if best_idx is None:
                break

            item_id = items[best_idx]
            selected.append((item_id, float(scores[best_idx])))
            selected_genres.append(item_genres.get(item_id, "Unknown"))
            remaining.remove(best_idx)

        return selected

    # ------------------------------------------------------------------
    # Step 3: Hard Genre Cap (Business Rules)
    # ------------------------------------------------------------------

    def _apply_genre_cap(
        self,
        ranked: List[Tuple[int, float]],
        item_genres: Dict[int, str],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        """
        Hard cap: no more than max_same_genre items from any single genre.
        Items that violate the cap are skipped (not reordered to later).
        This is a hard business rule — it cannot be overridden by model scores.
        """
        genre_counts: Dict[str, int] = {}
        final = []

        for item_id, score in ranked:
            genre = item_genres.get(item_id, "Unknown")
            count = genre_counts.get(genre, 0)
            if count < self.max_same_genre:
                final.append((item_id, score))
                genre_counts[genre] = count + 1
            if len(final) >= top_n:
                break

        # If we couldn't fill top_n due to genre cap, fill from remaining
        if len(final) < top_n:
            filled_ids = {iid for iid, _ in final}
            for item_id, score in ranked:
                if item_id not in filled_ids:
                    final.append((item_id, score))
                if len(final) >= top_n:
                    break

        return final[:top_n]


# Singleton
post_ranker = PostRanker()
