"""Batch parameter samplers for optimization.

All samplers produce (N, P) int64 index matrices where each value is
an index into the parameter's values list. The engine converts these
to actual values via encoding.indices_to_values().

Samplers:
- RandomSampler: uniform random sampling (baseline)
- SobolSampler: quasi-random Sobol sequence for uniform coverage
- EDASampler: Cross-Entropy/EDA with probability tables
"""

from __future__ import annotations

import numpy as np

from backtester.core.encoding import EncodingSpec


class RandomSampler:
    """Uniform random sampling — baseline sampler."""

    def __init__(self, spec: EncodingSpec, seed: int | None = None):
        self.spec = spec
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        n: int,
        mask: np.ndarray | None = None,
        locked: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate (N, P) index matrix.

        Args:
            n: Number of parameter sets to generate.
            mask: (P,) bool — only sample True columns; False columns get 0.
            locked: (P,) int64 — locked column values override sampling.
                Set to -1 for unlocked columns.
        """
        p = self.spec.num_params
        matrix = np.zeros((n, p), dtype=np.int64)

        for col in self.spec.columns:
            if mask is not None and not mask[col.index]:
                continue
            if locked is not None and locked[col.index] >= 0:
                matrix[:, col.index] = locked[col.index]
                continue
            num_vals = len(col.values)
            matrix[:, col.index] = self.rng.integers(0, num_vals, size=n)

        return matrix


class SobolSampler:
    """Sobol quasi-random sequence mapped to discrete parameter indices.

    Better coverage than pure random — reduces clumping, covers corners.
    Uses scipy's Sobol if available, falls back to scrambled random.
    """

    def __init__(self, spec: EncodingSpec, seed: int | None = None):
        self.spec = spec
        self.seed = seed or 0
        self._sobol = None
        self._init_sobol()

    def _init_sobol(self) -> None:
        try:
            from scipy.stats.qmc import Sobol
            self._sobol = Sobol(d=self.spec.num_params, scramble=True, seed=self.seed)
        except ImportError:
            self._sobol = None

    def sample(
        self,
        n: int,
        mask: np.ndarray | None = None,
        locked: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate (N, P) index matrix using Sobol sequence."""
        p = self.spec.num_params

        if self._sobol is not None and p > 0:
            # Sobol generates [0,1)^P — map to indices
            # Round up to power of 2 for Sobol
            n_sobol = max(n, 1)
            # Sobol requires powers of 2
            n_pow2 = 1
            while n_pow2 < n_sobol:
                n_pow2 *= 2
            raw = self._sobol.random(n_pow2)[:n]  # Take first n
            matrix = np.zeros((n, p), dtype=np.int64)
            for col in self.spec.columns:
                if mask is not None and not mask[col.index]:
                    continue
                if locked is not None and locked[col.index] >= 0:
                    matrix[:, col.index] = locked[col.index]
                    continue
                num_vals = len(col.values)
                matrix[:, col.index] = np.clip(
                    (raw[:, col.index] * num_vals).astype(np.int64),
                    0, num_vals - 1,
                )
        else:
            # Fallback to random
            rng = np.random.default_rng(self.seed)
            matrix = np.zeros((n, p), dtype=np.int64)
            for col in self.spec.columns:
                if mask is not None and not mask[col.index]:
                    continue
                if locked is not None and locked[col.index] >= 0:
                    matrix[:, col.index] = locked[col.index]
                    continue
                num_vals = len(col.values)
                matrix[:, col.index] = rng.integers(0, num_vals, size=n)

        return matrix


class EDASampler:
    """Cross-Entropy / Estimation of Distribution Algorithm sampler.

    Maintains per-parameter probability tables. Samples from the current
    distribution, then updates toward the elite subset.

    Natively discrete, trivially batchable, ~microseconds per batch.
    """

    def __init__(
        self,
        spec: EncodingSpec,
        learning_rate: float = 0.3,
        min_prob: float = 0.01,
        seed: int | None = None,
    ):
        self.spec = spec
        self.lr = learning_rate
        self.min_prob = min_prob
        self.rng = np.random.default_rng(seed)

        # Initialize uniform probability tables
        self.prob_tables: list[np.ndarray] = []
        for col in spec.columns:
            n_vals = len(col.values)
            self.prob_tables.append(np.full(n_vals, 1.0 / n_vals))

    def sample(
        self,
        n: int,
        mask: np.ndarray | None = None,
        locked: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample from current probability distribution."""
        p = self.spec.num_params
        matrix = np.zeros((n, p), dtype=np.int64)

        for col in self.spec.columns:
            if mask is not None and not mask[col.index]:
                continue
            if locked is not None and locked[col.index] >= 0:
                matrix[:, col.index] = locked[col.index]
                continue

            probs = self.prob_tables[col.index]
            # Sample from categorical distribution
            matrix[:, col.index] = self.rng.choice(
                len(probs), size=n, p=probs,
            )

        return matrix

    def update(
        self,
        elite_indices: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        """Update probability tables toward elite distribution.

        Args:
            elite_indices: (K, P) int64 — elite parameter index sets.
            mask: (P,) bool — only update True columns.
        """
        k = elite_indices.shape[0]
        if k == 0:
            return

        for col in self.spec.columns:
            if mask is not None and not mask[col.index]:
                continue

            n_vals = len(col.values)
            # Compute empirical distribution from elites
            counts = np.zeros(n_vals, dtype=np.float64)
            for i in range(k):
                idx = int(elite_indices[i, col.index])
                if 0 <= idx < n_vals:
                    counts[idx] += 1.0
            elite_dist = counts / k

            # Blend toward elite distribution
            old_prob = self.prob_tables[col.index]
            new_prob = (1.0 - self.lr) * old_prob + self.lr * elite_dist

            # Apply probability floor
            new_prob = np.maximum(new_prob, self.min_prob)
            new_prob /= new_prob.sum()  # Renormalize

            self.prob_tables[col.index] = new_prob

    def reset(self) -> None:
        """Reset to uniform distribution."""
        for col in self.spec.columns:
            n_vals = len(col.values)
            self.prob_tables[col.index] = np.full(n_vals, 1.0 / n_vals)
