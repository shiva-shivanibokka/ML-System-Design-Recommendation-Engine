"""
Neural Collaborative Filtering (NCF) — PyTorch Implementation.

Architecture (NeuMF — Neural Matrix Factorization, He et al. 2017):
  NeuMF = GMF path + MLP path, outputs merged via a final linear layer.

  GMF Path (Generalized Matrix Factorization):
    user_gmf_emb(u) ⊙ item_gmf_emb(i)    → element-wise product → linear
    Captures the same signal as SVD but with learned embeddings.

  MLP Path (Multi-Layer Perceptron):
    [user_mlp_emb(u) ∥ item_mlp_emb(i)]  → concat → MLP → hidden layers
    Captures non-linear user-item interactions SVD cannot.

  Final:
    [gmf_output ∥ mlp_output] → Linear(output_dim → 1) → Sigmoid → score

Why NeuMF over plain NCF:
  - Combines linear (GMF) + non-linear (MLP) collaborative signals
  - The GMF path's item embeddings are extracted for FAISS Stage 1 retrieval
  - Published result: NeuMF outperforms GMF and MLP alone on MovieLens 1M

Training:
  - Binary Cross-Entropy on (user, pos_item, label=1) and (user, neg_item, label=0)
  - Negative sampling: for each positive, sample `num_negatives` random items
    not in the user's interaction history (in-batch negatives)
  - Optimizer: Adam with lr=0.001
  - Evaluation: HR@10, NDCG@10 on validation set after each epoch
  - Early stopping on validation NDCG@10

Item Embeddings for FAISS:
  After training, we extract the GMF item embedding matrix (n_items × emb_dim)
  and build a FAISS IVF+PQ index over it. At serving time, the user's GMF
  embedding is used as the query vector to retrieve the top-500 nearest items
  (Stage 1 candidate generation — ANN retrieval).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class InteractionDataset(Dataset):
    """
    Negative-sampled interaction dataset for NCF training.
    For each positive (user, item) pair, samples `num_negatives` random
    items not in the user's history. This is the standard NCF training protocol.
    """

    def __init__(
        self,
        interactions: np.ndarray,  # shape (N, 2): [[user_idx, item_idx], ...]
        n_items: int,
        user_history: Dict[int, set],
        num_negatives: int = 4,
    ):
        self.interactions = interactions
        self.n_items = n_items
        self.user_history = user_history
        self.num_negatives = num_negatives
        self._build_samples()

    def _build_samples(self):
        """Pre-generate negative samples once per epoch."""
        users, items, labels = [], [], []
        for user_idx, item_idx in self.interactions:
            # Positive
            users.append(user_idx)
            items.append(item_idx)
            labels.append(1.0)
            # Negatives
            seen = self.user_history.get(int(user_idx), set())
            neg_count = 0
            while neg_count < self.num_negatives:
                neg = np.random.randint(0, self.n_items)
                if neg not in seen:
                    users.append(user_idx)
                    items.append(neg)
                    labels.append(0.0)
                    neg_count += 1

        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF).

    Two separate embedding spaces:
      - GMF embeddings: captured via element-wise product (linear interactions)
      - MLP embeddings: captured via concatenation + MLP (non-linear interactions)

    Final prediction: sigmoid(Linear([gmf_out ∥ mlp_out]))
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = None,
        mlp_layers: List[int] = None,
        dropout: float = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim or settings.ncf.embedding_dim
        self.mlp_layers = mlp_layers or settings.ncf.mlp_layers
        self.dropout_rate = dropout or settings.ncf.dropout

        # GMF path — separate embedding tables
        self.user_gmf_emb = nn.Embedding(n_users, self.embedding_dim)
        self.item_gmf_emb = nn.Embedding(n_items, self.embedding_dim)

        # MLP path — separate embedding tables (standard NeuMF practice)
        self.user_mlp_emb = nn.Embedding(n_users, self.embedding_dim)
        self.item_mlp_emb = nn.Embedding(n_items, self.embedding_dim)

        # MLP tower
        mlp_input_dim = self.embedding_dim * 2
        mlp_modules = []
        in_dim = mlp_input_dim
        for out_dim in self.mlp_layers:
            mlp_modules.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction layer: GMF output (emb_dim) + MLP last layer output
        final_input_dim = self.embedding_dim + self.mlp_layers[-1]
        self.prediction = nn.Linear(final_input_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [
            self.user_gmf_emb,
            self.item_gmf_emb,
            self.user_mlp_emb,
            self.item_mlp_emb,
        ]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF path
        u_gmf = self.user_gmf_emb(user_ids)
        i_gmf = self.item_gmf_emb(item_ids)
        gmf_out = u_gmf * i_gmf  # element-wise product

        # MLP path
        u_mlp = self.user_mlp_emb(user_ids)
        i_mlp = self.item_mlp_emb(item_ids)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Concatenate and predict
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        logit = self.prediction(combined).squeeze(-1)
        return torch.sigmoid(logit)

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Returns the user's GMF embedding vector.
        Used as the ANN query vector in Stage 1 retrieval.
        """
        with torch.no_grad():
            uid = torch.LongTensor([user_id])
            emb = self.user_gmf_emb(uid).squeeze(0).numpy()
        return emb

    def get_all_item_embeddings(self) -> np.ndarray:
        """
        Returns all item GMF embeddings. Shape: (n_items, embedding_dim).
        These are indexed by FAISS for Stage 1 ANN retrieval.
        """
        with torch.no_grad():
            all_ids = torch.arange(self.item_gmf_emb.num_embeddings)
            embs = self.item_gmf_emb(all_ids).numpy()
        return embs

    def score_candidates(
        self, user_id: int, candidate_item_ids: List[int]
    ) -> np.ndarray:
        """
        Run full NeuMF forward pass for a user vs a list of candidate items.
        Used in Stage 2 reranking: score top-500 candidates from FAISS.
        Shape returns: (len(candidate_item_ids),)
        """
        with torch.no_grad():
            n = len(candidate_item_ids)
            users = torch.LongTensor([user_id] * n)
            items = torch.LongTensor(candidate_item_ids)
            scores = self(users, items).numpy()
        return scores

    def save(self, path: str = None):
        path = path or settings.ncf.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_users": self.user_gmf_emb.num_embeddings,
                "n_items": self.item_gmf_emb.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "mlp_layers": self.mlp_layers,
                "dropout": self.dropout_rate,
            },
            path,
        )
        print(f"[NCF] Model saved → {path}")

    @classmethod
    def load(cls, path: str = None) -> "NeuMF":
        path = path or settings.ncf.model_path
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            n_users=checkpoint["n_users"],
            n_items=checkpoint["n_items"],
            embedding_dim=checkpoint["embedding_dim"],
            mlp_layers=checkpoint["mlp_layers"],
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"[NCF] Model loaded from {path}")
        return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_ncf(
    model: NeuMF,
    val_interactions: np.ndarray,
    train_history: Dict[int, set],
    n_items: int,
    top_k: int = 10,
    n_eval_users: int = 1000,
) -> Dict[str, float]:
    """
    Leave-one-out evaluation using 100-item sampling protocol (NCF paper standard):
    For each user, rank the 1 positive item against 99 randomly sampled negatives.
    Report HR@K and NDCG@K.

    Note: For full evaluation, use all items (set n_eval_users high).
    The 100-item protocol is standard in the NCF paper for reproducibility.
    """
    model.eval()
    hits, ndcgs = [], []

    # Deduplicate: one test positive per user
    seen_users = set()
    test_pairs = []
    for user_idx, item_idx in val_interactions:
        if user_idx not in seen_users:
            test_pairs.append((int(user_idx), int(item_idx)))
            seen_users.add(user_idx)

    np.random.shuffle(test_pairs)
    test_pairs = test_pairs[:n_eval_users]

    with torch.no_grad():
        for user_idx, pos_item in test_pairs:
            # Sample 99 negatives not in user history
            history = train_history.get(user_idx, set())
            negatives = []
            while len(negatives) < 99:
                neg = np.random.randint(0, n_items)
                if neg != pos_item and neg not in history:
                    negatives.append(neg)
            candidates = [pos_item] + negatives

            users_t = torch.LongTensor([user_idx] * 100)
            items_t = torch.LongTensor(candidates)
            scores = model(users_t, items_t).numpy()

            # Rank
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            ranked_items = [candidates[i] for i in ranked_indices]

            if pos_item in ranked_items:
                rank = ranked_items.index(pos_item) + 1
                hits.append(1)
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                hits.append(0)
                ndcgs.append(0.0)

    hr = float(np.mean(hits))
    ndcg = float(np.mean(ndcgs))
    print(f"[NCF Eval] HR@{top_k}={hr:.4f}  NDCG@{top_k}={ndcg:.4f}  (n={len(hits)})")
    return {f"hr_at_{top_k}": hr, f"ndcg_at_{top_k}": ndcg}
