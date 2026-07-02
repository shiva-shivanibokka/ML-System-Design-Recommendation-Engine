"""
SVD Baseline Model — Matrix Factorization via Truncated SVD.

Why SVD:
  - Matrix factorization is the original collaborative filtering technique
    (Netflix Prize 2009). Fast to train, interpretable, strong baseline.
  - In production: used as the lightweight arm in the A/B bandit test.
    When the NCF arm wins decisively, SVD is retired. When they're tied,
    SVD is preferred (lower latency, lower compute cost).

Architecture:
  - Build a user-item interaction matrix (users × items)
  - Decompose: M ≈ U × Σ × Vᵀ  via sklearn TruncatedSVD
  - User factors U: shape (n_users, k)
  - Item factors Vᵀ: shape (k, n_items) → we store V: (n_items, k)
  - Prediction: score(u, i) = U[u] · V[i]
  - Ranking: for user u, sort all items by score descending

Serving:
  - User factors loaded in memory (6040 users × 128 dims = ~3MB, trivial)
  - Dot product over all items: ~0.5ms on CPU
  - No GPU needed

Evaluation metrics (same as NCF for fair comparison):
  - HR@10 (Hit Rate): was the held-out item in the top-10?
  - NDCG@10: normalized discounted cumulative gain at 10
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings


class SVDRecommender:
    """
    SVD-based collaborative filtering recommender.
    Serves as the baseline model (arm A in the Thompson Sampling bandit).
    """

    def __init__(self, n_components: int = None):
        self.n_components = n_components or settings.svd.n_components
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            n_iter=settings.svd.n_iter,
            random_state=settings.svd.random_state,
        )
        self.user_factors: np.ndarray = None  # (n_users, k)
        self.item_factors: np.ndarray = None  # (n_items, k)
        self.n_users: int = 0
        self.n_items: int = 0
        self._user2idx: Dict = {}
        self._item2idx: Dict = {}
        self._idx2item: Dict = {}

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "SVDRecommender":
        """
        Fit on training interactions.
        train: DataFrame with columns [user_idx, item_idx, label]
        """
        self.n_users = n_users
        self.n_items = n_items

        print(f"[SVD] Building interaction matrix ({n_users} × {n_items}) ...")
        rows = train["user_idx"].values
        cols = train["item_idx"].values
        data = train["label"].values.astype(np.float32)

        matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        print(f"[SVD] Matrix density: {matrix.nnz / (n_users * n_items):.4%}")

        print(f"[SVD] Fitting TruncatedSVD (k={self.n_components}) ...")
        self.user_factors = self.svd.fit_transform(matrix)  # (n_users, k)
        self.item_factors = self.svd.components_.T  # (n_items, k)

        explained = self.svd.explained_variance_ratio_.sum()
        print(f"[SVD] Explained variance: {explained:.4f}")
        return self

    def predict_scores(self, user_idx: int) -> np.ndarray:
        """
        Return raw dot-product scores for all items for a given user.
        Shape: (n_items,)
        """
        if self.user_factors is None:
            raise RuntimeError("Model not fitted yet.")
        return self.user_factors[user_idx] @ self.item_factors.T

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        exclude_seen: set = None,
    ) -> List[Tuple[int, float]]:
        """
        Return top-k (item_idx, score) pairs for a user, excluding seen items.
        """
        scores = self.predict_scores(user_idx)
        if exclude_seen:
            scores[list(exclude_seen)] = -np.inf
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Return item factor vector — used for content-based cold-start fallback."""
        return self.item_factors[item_idx]

    def save(self):
        Path(settings.svd.model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(settings.svd.model_path, "wb") as f:
            pickle.dump(self, f)
        np.save(settings.svd.user_factors_path, self.user_factors)
        np.save(settings.svd.item_factors_path, self.item_factors)
        print(f"[SVD] Saved → {settings.svd.model_path}")

    @classmethod
    def load(cls) -> "SVDRecommender":
        with open(settings.svd.model_path, "rb") as f:
            model = pickle.load(f)
        print(f"[SVD] Loaded from {settings.svd.model_path}")
        return model


def evaluate(
    model: SVDRecommender,
    val: pd.DataFrame,
    train: pd.DataFrame,
    top_k: int = 10,
    n_eval_users: int = 1000,
    n_negatives: int = 99,
) -> Dict[str, float]:
    """
    Leave-one-out evaluation using the 100-item sampling protocol — the SAME
    protocol as training.ncf_model.evaluate_ncf (rank the held-out positive
    against 99 sampled negatives), so SVD and NCF HR@K/NDCG@K are directly
    comparable (the earlier full-catalog ranking made them apples-to-oranges).
    """
    seen = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    n_items = model.n_items

    # One held-out positive per user
    test_pairs, seen_users = [], set()
    for user_idx, item_idx in zip(val["user_idx"].tolist(), val["item_idx"].tolist()):
        u = int(user_idx)
        if u not in seen_users:
            test_pairs.append((u, int(item_idx)))
            seen_users.add(u)
    np.random.shuffle(test_pairs)
    test_pairs = test_pairs[:n_eval_users]

    hits, ndcgs = [], []
    for user_idx, pos_item in test_pairs:
        history = seen.get(user_idx, set())
        all_scores = model.predict_scores(user_idx)

        negatives = []
        while len(negatives) < n_negatives:
            neg = int(np.random.randint(0, n_items))
            if neg != pos_item and neg not in history:
                negatives.append(neg)
        candidates = [pos_item] + negatives
        cand_scores = all_scores[candidates]

        ranked_indices = np.argsort(-cand_scores, kind="stable")[:top_k]
        ranked_items = [candidates[i] for i in ranked_indices]

        if pos_item in ranked_items:
            rank = ranked_items.index(pos_item) + 1
            hits.append(1)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            hits.append(0)
            ndcgs.append(0.0)

    hr = float(np.mean(hits)) if hits else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    print(f"[SVD Eval] HR@{top_k}={hr:.4f}  NDCG@{top_k}={ndcg:.4f}  (n={len(hits)})")
    return {f"hr_at_{top_k}": hr, f"ndcg_at_{top_k}": ndcg}
