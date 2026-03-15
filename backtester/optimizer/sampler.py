"""Batch parameter samplers for optimization.

All samplers produce (N, P) int64 index matrices where each value is
an index into the parameter's values list. The engine converts these
to actual values via encoding.indices_to_values().

Samplers:
- RandomSampler: uniform random sampling (baseline)
- SobolSampler: quasi-random Sobol sequence for uniform coverage
- EDASampler: Cross-Entropy/EDA with probability tables
- CMAESSampler: CMA-ES with Margin for exploitation via cmaes library
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtester.core.encoding import EncodingSpec


@dataclass
class NeighborhoodSpec:
    """Defines per-parameter index bounds for neighborhood sampling.

    Used in refinement to constrain sampling to a local region around
    the locked best values from previous stages.
    """
    min_bounds: np.ndarray   # (P,) int — min index per param
    max_bounds: np.ndarray   # (P,) int — max index per param


def build_neighborhood(
    spec: EncodingSpec,
    locked: np.ndarray,
    radius: int,
) -> NeighborhoodSpec:
    """Build neighborhood bounds centered on locked values.

    Args:
        spec: Encoding spec with parameter value lists.
        locked: (P,) int64 — locked index per param (-1 = unlocked, full range).
        radius: Number of index steps to explore in each direction.

    Returns:
        NeighborhoodSpec with per-param [min, max] index bounds.
    """
    p = spec.num_params
    min_bounds = np.zeros(p, dtype=np.int64)
    max_bounds = np.zeros(p, dtype=np.int64)

    for col in spec.columns:
        n_vals = len(col.values)
        if locked[col.index] >= 0:
            center = int(locked[col.index])
            min_bounds[col.index] = max(0, center - radius)
            max_bounds[col.index] = min(n_vals - 1, center + radius)
        else:
            # Unlocked param: full range
            min_bounds[col.index] = 0
            max_bounds[col.index] = n_vals - 1

    return NeighborhoodSpec(min_bounds=min_bounds, max_bounds=max_bounds)


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
        neighborhood: NeighborhoodSpec | None = None,
    ) -> np.ndarray:
        """Generate (N, P) index matrix.

        Args:
            n: Number of parameter sets to generate.
            mask: (P,) bool — active columns use this sampler's strategy;
                inactive+unlocked columns get uniform random (noise averaging).
            locked: (P,) int64 — locked column values override sampling.
                Set to -1 for unlocked columns. Checked BEFORE mask.
            neighborhood: When set, constrains sampling to [min, max] per param.
        """
        p = self.spec.num_params
        matrix = np.zeros((n, p), dtype=np.int64)

        for col in self.spec.columns:
            # Locked always wins — preserves results from earlier stages
            if locked is not None and locked[col.index] >= 0:
                matrix[:, col.index] = locked[col.index]
                continue
            # Apply neighborhood bounds if provided
            if neighborhood is not None:
                lo = int(neighborhood.min_bounds[col.index])
                hi = int(neighborhood.max_bounds[col.index])
                matrix[:, col.index] = self.rng.integers(lo, hi + 1, size=n)
            else:
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
        neighborhood: NeighborhoodSpec | None = None,
    ) -> np.ndarray:
        """Generate (N, P) index matrix using Sobol sequence.

        Locked params always use locked value (checked before mask).
        Inactive+unlocked params get Sobol coverage for noise averaging.
        """
        p = self.spec.num_params

        if self._sobol is not None and p > 0:
            # Sobol generates [0,1)^P — map to indices
            n_sobol = max(n, 1)
            n_pow2 = 1
            while n_pow2 < n_sobol:
                n_pow2 *= 2
            raw = self._sobol.random(n_pow2)[:n]  # Take first n
            matrix = np.zeros((n, p), dtype=np.int64)
            for col in self.spec.columns:
                # Locked always wins — preserves earlier stage results
                if locked is not None and locked[col.index] >= 0:
                    matrix[:, col.index] = locked[col.index]
                    continue
                if neighborhood is not None:
                    # Scale Sobol [0,1) to [lo, hi] integer range
                    lo = int(neighborhood.min_bounds[col.index])
                    hi = int(neighborhood.max_bounds[col.index])
                    span = hi - lo + 1
                    matrix[:, col.index] = np.clip(
                        (raw[:, col.index] * span).astype(np.int64) + lo,
                        lo, hi,
                    )
                else:
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
                if locked is not None and locked[col.index] >= 0:
                    matrix[:, col.index] = locked[col.index]
                    continue
                if neighborhood is not None:
                    lo = int(neighborhood.min_bounds[col.index])
                    hi = int(neighborhood.max_bounds[col.index])
                    matrix[:, col.index] = rng.integers(lo, hi + 1, size=n)
                else:
                    num_vals = len(col.values)
                    matrix[:, col.index] = rng.integers(0, num_vals, size=n)

        return matrix


class EDASampler:
    """Cross-Entropy / Estimation of Distribution Algorithm sampler.

    Maintains per-parameter probability tables. Samples from the current
    distribution, then updates toward the elite subset.

    Features:
    - Adaptive learning rate: decays from lr_initial toward lr_floor over
      successive updates. Early exploitation is aggressive, late is conservative.
    - Entropy monitoring: Shannon entropy per parameter for convergence diagnostics.

    Natively discrete, trivially batchable, ~microseconds per batch.
    """

    def __init__(
        self,
        spec: EncodingSpec,
        learning_rate: float = 0.3,
        lr_decay: float = 0.95,
        lr_floor: float = 0.05,
        min_prob: float = 0.01,
        seed: int | None = None,
    ):
        self.spec = spec
        self.lr_initial = learning_rate
        self.lr_decay = lr_decay
        self.lr_floor = lr_floor
        self.min_prob = min_prob
        self.rng = np.random.default_rng(seed)
        self._update_count = 0

        # Initialize uniform probability tables
        self.prob_tables: list[np.ndarray] = []
        for col in spec.columns:
            n_vals = len(col.values)
            self.prob_tables.append(np.full(n_vals, 1.0 / n_vals))

    @property
    def effective_lr(self) -> float:
        """Current learning rate after decay."""
        return self.lr_floor + (self.lr_initial - self.lr_floor) * (
            self.lr_decay ** self._update_count
        )

    @property
    def update_count(self) -> int:
        return self._update_count

    def sample(
        self,
        n: int,
        mask: np.ndarray | None = None,
        locked: np.ndarray | None = None,
        neighborhood: NeighborhoodSpec | None = None,
    ) -> np.ndarray:
        """Sample from current probability distribution.

        Locked params always use locked value (checked before mask).
        Active params use EDA probabilities. Inactive+unlocked params
        get uniform random for noise averaging.
        """
        p = self.spec.num_params
        matrix = np.zeros((n, p), dtype=np.int64)

        for col in self.spec.columns:
            # Locked always wins — preserves earlier stage results
            if locked is not None and locked[col.index] >= 0:
                matrix[:, col.index] = locked[col.index]
                continue

            if mask is not None and not mask[col.index]:
                # Inactive + unlocked: uniform random for noise averaging
                if neighborhood is not None:
                    lo = int(neighborhood.min_bounds[col.index])
                    hi = int(neighborhood.max_bounds[col.index])
                    matrix[:, col.index] = self.rng.integers(lo, hi + 1, size=n)
                else:
                    num_vals = len(col.values)
                    matrix[:, col.index] = self.rng.integers(0, num_vals, size=n)
                continue

            # Active + unlocked: sample from EDA probability distribution
            probs = self.prob_tables[col.index]
            if neighborhood is not None:
                # Slice probability table to neighborhood and renormalize
                lo = int(neighborhood.min_bounds[col.index])
                hi = int(neighborhood.max_bounds[col.index])
                sliced = probs[lo:hi + 1].copy()
                sliced_sum = sliced.sum()
                if sliced_sum > 0:
                    sliced /= sliced_sum
                else:
                    sliced = np.full(len(sliced), 1.0 / len(sliced))
                matrix[:, col.index] = self.rng.choice(
                    len(sliced), size=n, p=sliced,
                ) + lo
            else:
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

        Uses adaptive learning rate that decays over successive updates.

        Args:
            elite_indices: (K, P) int64 — elite parameter index sets.
            mask: (P,) bool — only update True columns.
        """
        k = elite_indices.shape[0]
        if k == 0:
            return

        lr = self.effective_lr

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

            # Blend toward elite distribution with adaptive LR
            old_prob = self.prob_tables[col.index]
            new_prob = (1.0 - lr) * old_prob + lr * elite_dist

            # Apply probability floor
            new_prob = np.maximum(new_prob, self.min_prob)
            new_prob /= new_prob.sum()  # Renormalize

            self.prob_tables[col.index] = new_prob

        self._update_count += 1

    def entropy(self, mask: np.ndarray | None = None) -> float:
        """Mean normalized entropy across active parameters.

        Returns a value in [0, 1]:
        - 1.0 = all parameters uniformly distributed (max exploration)
        - 0.0 = all parameters fully converged to a single value

        Args:
            mask: (P,) bool — only include True columns. None = all.
        """
        entropies = self.entropy_per_param(mask)
        if len(entropies) == 0:
            return 1.0
        return float(np.mean(entropies))

    def entropy_per_param(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Normalized Shannon entropy for each active parameter.

        Returns (K,) array where K = number of active params, values in [0, 1].
        """
        result = []
        for col in self.spec.columns:
            if mask is not None and not mask[col.index]:
                continue
            probs = self.prob_tables[col.index]
            n_vals = len(probs)
            if n_vals <= 1:
                result.append(0.0)
                continue
            # Shannon entropy: H = -sum(p * log2(p)), skip zero probs
            nonzero = probs[probs > 0]
            h = -np.sum(nonzero * np.log2(nonzero))
            h_max = np.log2(n_vals)
            result.append(h / h_max)
        return np.array(result, dtype=np.float64)

    def reset(self) -> None:
        """Reset to uniform distribution and restart LR decay."""
        for col in self.spec.columns:
            n_vals = len(col.values)
            self.prob_tables[col.index] = np.full(n_vals, 1.0 / n_vals)
        self._update_count = 0


class CMAESSampler:
    """CMA-ES with Margin sampler for exploitation phase.

    Uses CMAwM from the cmaes library. All parameters are treated as
    integer indices into discrete value lists, matching our encoding system.

    CMAwM handles mixed-integer optimization natively via discretization
    margins, avoiding the need for manual rounding heuristics.
    """

    def __init__(
        self,
        spec: EncodingSpec,
        sigma0: float = 0.3,
        population_size: int | None = None,
        seed: int | None = None,
    ):
        self._spec = spec
        self._sigma0 = sigma0
        self._pop_size = population_size
        self._seed = seed
        self._cma: object | None = None  # CMAwM instance
        self._active_dims: list[int] = []  # indices into full P-dim space
        self._pending_tell: list[tuple[np.ndarray, int]] = []  # (x_tell, row_index)
        self._generation_count = 0
        self._initial_sigma: float = 0.0
        self._current_pop_size: int | None = population_size
        self._rng = np.random.default_rng(seed)
        self._last_mask_key: tuple | None = None  # track mask changes

    def _init_cma(
        self,
        mask: np.ndarray | None,
        locked: np.ndarray | None,
        neighborhood: NeighborhoodSpec | None,
    ) -> None:
        """Lazy-init CMAwM instance when we know active dimensions."""
        from cmaes import CMAwM

        # Determine active dims: mask=True AND not locked
        self._active_dims = []
        for col in self._spec.columns:
            if locked is not None and locked[col.index] >= 0:
                continue  # locked — skip
            if mask is not None and not mask[col.index]:
                continue  # inactive — skip
            self._active_dims.append(col.index)

        n_active = len(self._active_dims)
        if n_active == 0:
            self._cma = None
            return

        mean = np.zeros(n_active, dtype=np.float64)
        bounds = np.zeros((n_active, 2), dtype=np.float64)
        steps = np.ones(n_active, dtype=np.float64)  # integer discretization

        for j, dim_idx in enumerate(self._active_dims):
            col = self._spec.columns[dim_idx]
            n_vals = len(col.values)
            lo = 0.0
            hi = float(n_vals - 1)

            if neighborhood is not None:
                lo = float(neighborhood.min_bounds[dim_idx])
                hi = float(neighborhood.max_bounds[dim_idx])

            bounds[j, 0] = lo
            bounds[j, 1] = hi
            mean[j] = (lo + hi) / 2.0

        # sigma = sigma0 * max range across active dims
        max_range = max((bounds[j, 1] - bounds[j, 0]) for j in range(n_active))
        sigma = self._sigma0 * max(max_range, 1.0)
        self._initial_sigma = sigma

        self._cma = CMAwM(
            mean=mean,
            sigma=sigma,
            bounds=bounds,
            steps=steps,
            n_max_resampling=100,
            seed=int(self._rng.integers(0, 2**31)) if self._seed is None else self._seed,
            population_size=self._current_pop_size,
        )
        self._pending_tell = []

    def _mask_key(self, mask: np.ndarray | None, locked: np.ndarray | None) -> tuple:
        """Hash key to detect when mask/locked change and we need reinit."""
        m = tuple(mask.astype(int)) if mask is not None else ()
        l_ = tuple(locked) if locked is not None else ()
        return (m, l_)

    def sample(
        self,
        n: int,
        mask: np.ndarray | None = None,
        locked: np.ndarray | None = None,
        neighborhood: NeighborhoodSpec | None = None,
    ) -> np.ndarray:
        """Generate n parameter sets using CMA-ES.

        Returns (n, P) int64 index matrix.
        """
        p = self._spec.num_params
        out = np.zeros((n, p), dtype=np.int64)

        # Check if we need to (re)initialize
        mk = self._mask_key(mask, locked)
        if self._cma is None or mk != self._last_mask_key:
            self._init_cma(mask, locked, neighborhood)
            self._last_mask_key = mk

        # Fill locked params
        for col in self._spec.columns:
            if locked is not None and locked[col.index] >= 0:
                out[:, col.index] = locked[col.index]

        # Fill inactive (mask=False, not locked) with uniform random
        for col in self._spec.columns:
            if locked is not None and locked[col.index] >= 0:
                continue
            if mask is not None and not mask[col.index]:
                n_vals = len(col.values)
                if neighborhood is not None:
                    lo = int(neighborhood.min_bounds[col.index])
                    hi = int(neighborhood.max_bounds[col.index])
                    out[:, col.index] = self._rng.integers(lo, hi + 1, size=n)
                else:
                    out[:, col.index] = self._rng.integers(0, n_vals, size=n)

        # If no active dims, return early
        if not self._active_dims or self._cma is None:
            return out

        # Generate samples via CMA-ES ask()
        # CMA-ES requires exactly population_size ask() calls per generation
        # before a tell(). We fill rows in batches of population_size.
        pop_size = self._cma.population_size
        filled = 0
        self._pending_tell = []

        while filled < n:
            # IPOP restart if converged
            if self._cma.should_stop():
                self._ipop_restart(mask, locked, neighborhood)
                if self._cma is None:
                    # Fallback to uniform random for remaining rows
                    for row in range(filled, n):
                        for j, dim_idx in enumerate(self._active_dims):
                            col = self._spec.columns[dim_idx]
                            n_vals = len(col.values)
                            if neighborhood is not None:
                                lo = int(neighborhood.min_bounds[dim_idx])
                                hi = int(neighborhood.max_bounds[dim_idx])
                            else:
                                lo, hi = 0, n_vals - 1
                            out[row, dim_idx] = self._rng.integers(lo, hi + 1)
                    filled = n
                    break

            # Ask for one generation worth of samples
            batch_x_eval = []
            batch_x_tell = []
            for _ in range(pop_size):
                x_eval, x_tell = self._cma.ask()
                batch_x_eval.append(x_eval)
                batch_x_tell.append(x_tell)

            # Fill output rows from this generation
            for k in range(pop_size):
                if filled >= n:
                    break
                x_eval = batch_x_eval[k]
                x_tell = batch_x_tell[k]

                for j, dim_idx in enumerate(self._active_dims):
                    col = self._spec.columns[dim_idx]
                    n_vals = len(col.values)
                    if neighborhood is not None:
                        lo = int(neighborhood.min_bounds[dim_idx])
                        hi = int(neighborhood.max_bounds[dim_idx])
                    else:
                        lo, hi = 0, n_vals - 1
                    val = int(np.clip(np.round(x_eval[j]), lo, hi))
                    out[filled, dim_idx] = val

                self._pending_tell.append((x_tell, filled))
                filled += 1

            # If we used fewer than pop_size, we still need to store the rest
            # for tell() padding
            for k in range(filled - (filled % pop_size if filled % pop_size else pop_size), pop_size):
                if k < len(batch_x_tell) and (batch_x_tell[k] is not None):
                    idx_in_pending = len(self._pending_tell) - 1
                    # Already added above or excess — handled in update()
                    pass

        return out

    def _ipop_restart(
        self,
        mask: np.ndarray | None,
        locked: np.ndarray | None,
        neighborhood: NeighborhoodSpec | None,
    ) -> None:
        """IPOP restart: reinitialize with doubled sigma and population size."""
        self._current_pop_size = (self._current_pop_size or self._cma.population_size) * 2
        new_sigma = self._sigma0 * 2
        self._sigma0_restart = new_sigma
        # Save old sigma0, init with wider
        old_sigma0 = self._sigma0
        self._sigma0 = min(new_sigma, 1.0)  # cap at 1.0
        self._init_cma(mask, locked, neighborhood)
        self._sigma0 = old_sigma0  # restore for future use

    def update(
        self,
        index_batch: np.ndarray,
        qualities: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        """Update CMA-ES with evaluation results.

        Args:
            index_batch: (K, P) int64 -- the indices that were sampled
            qualities: (K,) float64 -- quality scores for each row
            mask: (P,) bool -- active params (same as sample())

        CMA-ES minimizes, we maximize quality, so negate.
        """
        if self._cma is None or not self._pending_tell:
            return

        pop_size = self._cma.population_size

        # Build a map from row_index -> quality for evaluated rows
        # Match pending_tell entries to their qualities by comparing
        # the active-dim indices in index_batch
        quality_by_row: dict[int, float] = {}
        for pt_x_tell, pt_row_idx in self._pending_tell:
            # Find this row in index_batch
            for k in range(index_batch.shape[0]):
                match = True
                for dim_idx in self._active_dims:
                    if index_batch.shape[1] <= dim_idx:
                        match = False
                        break
                    # Compare the rounded x_eval stored in out vs index_batch
                    # We stored pt_row_idx as the row in the output matrix
                    # index_batch[k] may be a subset, match by active dim values
                    pass
                break
            # Simpler approach: use row index position directly
            quality_by_row[pt_row_idx] = float("-inf")

        # Better approach: pending_tell stores (x_tell, row_index).
        # index_batch rows correspond to rows from sample() output.
        # Match by finding which pending rows appear in index_batch.
        #
        # Since index_batch is typically a filtered subset of the sample() output,
        # we match by active-dim values.
        active_set = set(self._active_dims)

        # Build lookup from active-dim tuple -> list of (x_tell, row_idx)
        pending_by_active = {}
        for x_tell, row_idx in self._pending_tell:
            # We don't have the original out matrix, but we can reconstruct
            # the active-dim values from x_tell by rounding
            key_vals = []
            for j, dim_idx in enumerate(self._active_dims):
                col = self._spec.columns[dim_idx]
                n_vals = len(col.values)
                val = int(np.clip(np.round(x_tell[j]), 0, n_vals - 1))
                key_vals.append(val)
            key = tuple(key_vals)
            if key not in pending_by_active:
                pending_by_active[key] = []
            pending_by_active[key].append((x_tell, row_idx))

        # Process in generation-sized chunks
        # Collect all (x_tell, fitness) pairs
        tell_pairs: list[tuple[np.ndarray, float]] = []
        used_pending = set()

        for k in range(index_batch.shape[0]):
            key_vals = []
            for dim_idx in self._active_dims:
                key_vals.append(int(index_batch[k, dim_idx]))
            key = tuple(key_vals)

            if key in pending_by_active and pending_by_active[key]:
                x_tell, row_idx = pending_by_active[key].pop(0)
                tell_pairs.append((x_tell, -float(qualities[k])))  # negate for minimization
                used_pending.add(row_idx)

        # Fill remaining slots with worst fitness (large positive = bad for minimization)
        worst_fitness = 1e18
        for x_tell, row_idx in self._pending_tell:
            if row_idx not in used_pending:
                tell_pairs.append((x_tell, worst_fitness))

        # Tell in generation-sized chunks
        while len(tell_pairs) >= pop_size:
            chunk = tell_pairs[:pop_size]
            tell_pairs = tell_pairs[pop_size:]
            try:
                self._cma.tell(chunk)
                self._generation_count += 1
            except Exception:
                # CMA-ES can raise on degenerate inputs; just skip
                pass

        self._pending_tell = []

    def reset(self) -> None:
        """Reset CMA-ES state for a new stage."""
        self._cma = None
        self._active_dims = []
        self._pending_tell = []
        self._generation_count = 0
        self._current_pop_size = self._pop_size
        self._last_mask_key = None

    @property
    def converged(self) -> bool:
        """Check if CMA-ES has converged."""
        if self._cma is None:
            return False
        return self._cma.should_stop()

    def entropy(self, mask: np.ndarray | None = None) -> float:
        """Proxy entropy from CMA-ES state for dashboard compatibility.

        Returns 1.0 (initial/uniform) to 0.0 (converged).
        Uses sigma relative to initial sigma as proxy.
        """
        if self._cma is None or self._initial_sigma <= 0:
            return 1.0
        try:
            # Access internal sigma from CMAwM
            current_sigma = self._cma._sigma
            ratio = current_sigma / self._initial_sigma
            return float(np.clip(ratio, 0.0, 1.0))
        except AttributeError:
            return 1.0
