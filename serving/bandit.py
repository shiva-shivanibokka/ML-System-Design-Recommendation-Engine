"""
Thompson Sampling Multi-Armed Bandit Router.

Replaces the naive 50/50 hard A/B split with an adaptive traffic allocator.

Why Thompson Sampling over fixed A/B:
  - Fixed 50/50: even after 10,000 requests clearly show NCF wins, you still
    send 50% of traffic to the worse model. This wastes opportunity.
  - Thompson Sampling: maintains a Beta(α, β) distribution per model arm.
    At each request, samples θ ~ Beta(α, β) per arm, serves the arm with
    the highest sample. Over time, winning arms get more traffic automatically.
  - After N successful samples, Bayesian credible interval is computed to
    detect a statistically significant winner.

How it works:
  - Each arm starts at Beta(1, 1) = Uniform[0,1] (maximum uncertainty)
  - A click (reward=1) increments α: α += 1
  - A no-click (reward=0) increments β: β += 1
  - At request time: θ_arm = sample from Beta(α_arm, β_arm)
  - Route to arm with max(θ_arm)

Statistical Winner Detection:
  - Uses Monte Carlo simulation (10,000 samples) to compute the probability
    that arm A CTR > arm B CTR: P(θ_A > θ_B)
  - If P > significance_threshold (default 0.95) → declare winner
  - Identical to Bayesian A/B testing used at Booking.com, Netflix, and Lyft

State Persistence:
  - Bandit state saved to JSON file and updated after every feedback event
  - Reloaded at startup so the bandit doesn't reset between deployments
  - In production: this state would live in Redis for multi-instance serving
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings


class ThompsonSamplingBandit:
    """
    Thompson Sampling multi-armed bandit for model selection.
    Each model (SVD, NCF) is one arm with its own Beta distribution.
    """

    def __init__(self, model_names: list = None):
        self.model_names = model_names or settings.bandit.models
        # Beta distribution parameters per arm: {model_name: [alpha, beta]}
        self.alphas: Dict[str, float] = {
            m: settings.bandit.initial_alpha for m in self.model_names
        }
        self.betas: Dict[str, float] = {
            m: settings.bandit.initial_beta for m in self.model_names
        }
        self.total_pulls: Dict[str, int] = {m: 0 for m in self.model_names}
        self.total_rewards: Dict[str, int] = {m: 0 for m in self.model_names}
        self._winner: Optional[str] = None
        self._winner_declared_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Core: select arm
    # ------------------------------------------------------------------

    def select_arm(self) -> str:
        """
        Sample θ ~ Beta(α, β) for each arm, return arm with max θ.
        This is pure Thompson Sampling — O(k) where k = number of arms.

        Exploration vs exploitation is automatic:
        - Low sample count → wide Beta → high variance → more exploration
        - High sample count → narrow Beta → low variance → more exploitation
        """
        # If a winner has been declared, always serve winner
        if self._winner:
            return self._winner

        # Force exploration if any arm has too few samples
        for model in self.model_names:
            if self.total_pulls[model] < settings.bandit.min_samples_per_arm:
                return model  # round-robin exploration for cold arms

        # Thompson Sampling: sample from each Beta and take argmax
        samples = {
            model: float(np.random.beta(self.alphas[model], self.betas[model]))
            for model in self.model_names
        }
        chosen = max(samples, key=samples.get)
        return chosen

    # ------------------------------------------------------------------
    # Feedback: update arm
    # ------------------------------------------------------------------

    def update(self, model_name: str, clicked: bool):
        """
        Update the Beta distribution for the given arm based on observed reward.
        clicked=True  → reward=1 → α += 1
        clicked=False → reward=0 → β += 1
        """
        if model_name not in self.alphas:
            return
        if clicked:
            self.alphas[model_name] += 1
            self.total_rewards[model_name] += 1
        else:
            self.betas[model_name] += 1
        self.total_pulls[model_name] += 1

        # Check for winner after every update
        self._check_for_winner()
        self.save()

    # ------------------------------------------------------------------
    # Winner detection (Bayesian credible interval)
    # ------------------------------------------------------------------

    def _check_for_winner(self, n_mc_samples: int = 10_000):
        """
        Monte Carlo estimate of P(arm_i_CTR > arm_j_CTR) for all pairs.
        If any arm has P > significance_threshold against all others → winner.
        """
        if self._winner:
            return  # already decided

        # Need minimum samples before declaring winner
        min_pulls = min(self.total_pulls.values())
        if min_pulls < settings.bandit.min_samples_per_arm:
            return

        n = len(self.model_names)
        if n < 2:
            return

        # Sample from Beta distributions
        samples = {
            m: np.random.beta(self.alphas[m], self.betas[m], size=n_mc_samples)
            for m in self.model_names
        }

        # For each arm: P(this arm > all other arms)
        for model in self.model_names:
            my_samples = samples[model]
            beats_all = np.ones(n_mc_samples, dtype=bool)
            for other in self.model_names:
                if other == model:
                    continue
                beats_all &= my_samples > samples[other]
            prob_best = float(beats_all.mean())

            if prob_best >= settings.bandit.significance_threshold:
                self._winner = model
                self._winner_declared_at = time.time()
                print(
                    f"[Bandit] WINNER DECLARED: {model} "
                    f"(P(best)={prob_best:.4f} >= {settings.bandit.significance_threshold})"
                )
                print(f"[Bandit] Final CTRs: {self.get_ctrs()}")
                return

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_ctrs(self) -> Dict[str, float]:
        """Empirical CTR per arm: total_rewards / total_pulls."""
        return {
            m: (self.total_rewards[m] / max(self.total_pulls[m], 1))
            for m in self.model_names
        }

    def get_state(self) -> dict:
        """Full bandit state for dashboard/API."""
        ctrs = self.get_ctrs()
        return {
            "alphas": dict(self.alphas),
            "betas": dict(self.betas),
            "total_pulls": dict(self.total_pulls),
            "total_rewards": dict(self.total_rewards),
            "ctrs": ctrs,
            "winner": self._winner,
            "winner_declared_at": self._winner_declared_at,
            "mean_ctrs": {
                m: self.alphas[m] / (self.alphas[m] + self.betas[m])
                for m in self.model_names
            },
            "uncertainty": {
                m: float(
                    np.sqrt(
                        (self.alphas[m] * self.betas[m])
                        / (
                            (self.alphas[m] + self.betas[m]) ** 2
                            * (self.alphas[m] + self.betas[m] + 1)
                        )
                    )
                )
                for m in self.model_names
            },
        }

    def get_traffic_split(self) -> Dict[str, float]:
        """
        Estimated effective traffic split based on Thompson Sampling.
        Approximated by simulating 10,000 arm selections.
        """
        counts = {m: 0 for m in self.model_names}
        for _ in range(10_000):
            arm = self.select_arm()
            counts[arm] += 1
        return {m: counts[m] / 10_000 for m in self.model_names}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        state_path = Path(settings.bandit.state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(
                {
                    "alphas": self.alphas,
                    "betas": self.betas,
                    "total_pulls": self.total_pulls,
                    "total_rewards": self.total_rewards,
                    "winner": self._winner,
                    "winner_declared_at": self._winner_declared_at,
                },
                f,
                indent=2,
            )

    def load(self) -> bool:
        state_path = Path(settings.bandit.state_path)
        if not state_path.exists():
            return False
        try:
            with open(state_path) as f:
                state = json.load(f)
            self.alphas = state["alphas"]
            self.betas = state["betas"]
            self.total_pulls = state["total_pulls"]
            self.total_rewards = state["total_rewards"]
            self._winner = state.get("winner")
            self._winner_declared_at = state.get("winner_declared_at")
            print(f"[Bandit] State restored from {state_path}")
            print(f"[Bandit] Pulls: {self.total_pulls}  CTRs: {self.get_ctrs()}")
            return True
        except Exception as e:
            print(f"[Bandit] Could not load state: {e}. Starting fresh.")
            return False

    def reset(self):
        """Reset bandit to initial state (for experiments)."""
        self.alphas = {m: settings.bandit.initial_alpha for m in self.model_names}
        self.betas = {m: settings.bandit.initial_beta for m in self.model_names}
        self.total_pulls = {m: 0 for m in self.model_names}
        self.total_rewards = {m: 0 for m in self.model_names}
        self._winner = None
        self._winner_declared_at = None
        state_path = Path(settings.bandit.state_path)
        if state_path.exists():
            state_path.unlink()
        print("[Bandit] State reset.")


# Singleton
bandit = ThompsonSamplingBandit()
bandit.load()
