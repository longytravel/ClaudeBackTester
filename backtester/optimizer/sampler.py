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

    def update(self, *args, **kwargs) -> None:
        """No-op: Sobol has no learning state."""
        pass

    def reset(self) -> None:
        """Reset Sobol sequence."""
        self._init_sobol()

    def entropy(self, mask: np.ndarray | None = None) -> float:
        """Always 1.0 — Sobol provides maximum coverage."""
        return 1.0


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
        # CMA-ES population_size may be much smaller than n (batch_size).
        # We run multiple generations to fill the batch, then tell() in
        # generation-sized chunks during update().
        pop_size = self._cma.population_size
        filled = 0
        self._pending_tell = []

        # Precompute per-active-dim bounds (vectorized clip targets)
        n_active = len(self._active_dims)
        lo_arr = np.zeros(n_active, dtype=np.float64)
        hi_arr = np.zeros(n_active, dtype=np.float64)
        for j, dim_idx in enumerate(self._active_dims):
            col = self._spec.columns[dim_idx]
            if neighborhood is not None:
                lo_arr[j] = float(neighborhood.min_bounds[dim_idx])
                hi_arr[j] = float(neighborhood.max_bounds[dim_idx])
            else:
                lo_arr[j] = 0.0
                hi_arr[j] = float(len(col.values) - 1)

        while filled < n:
            # IPOP restart if converged
            if self._cma.should_stop():
                self._ipop_restart(mask, locked, neighborhood)
                if self._cma is None:
                    # Fallback to uniform random for remaining rows
                    for j, dim_idx in enumerate(self._active_dims):
                        remaining = n - filled
                        lo, hi = int(lo_arr[j]), int(hi_arr[j])
                        out[filled:n, dim_idx] = self._rng.integers(lo, hi + 1, size=remaining)
                    filled = n
                    break
                pop_size = self._cma.population_size  # may have doubled

            # Ask for one generation worth of samples
            batch_x_eval = []
            batch_x_tell = []
            for _ in range(pop_size):
                x_eval, x_tell = self._cma.ask()
                batch_x_eval.append(x_eval)
                batch_x_tell.append(x_tell)

            # Vectorized: stack into matrix, clip, round, fill output
            use_count = min(pop_size, n - filled)
            x_matrix = np.array(batch_x_eval[:use_count])  # (use_count, n_active)
            rounded = np.clip(np.round(x_matrix), lo_arr, hi_arr).astype(np.int64)

            for j, dim_idx in enumerate(self._active_dims):
                out[filled:filled + use_count, dim_idx] = rounded[:, j]

            # Store x_tell for update() — indexed by output row
            for k in range(use_count):
                self._pending_tell.append((batch_x_tell[k], filled + k))
            # Store excess (beyond n) for tell() padding
            for k in range(use_count, pop_size):
                self._pending_tell.append((batch_x_tell[k], -1))  # -1 = not evaluated

            filled += use_count

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
        original_indices: np.ndarray | None = None,
    ) -> None:
        """Update CMA-ES with evaluation results.

        Args:
            index_batch: (K, P) int64 -- the indices that were evaluated
                (may be a pre-filtered subset of sample() output)
            qualities: (K,) float64 -- quality scores for each row
            mask: (P,) bool -- active params (same as sample())
            original_indices: (K,) int64 -- row positions in the original
                sample() output. Used for exact x_tell pairing (no key
                reconstruction needed). If None, falls back to positional.

        CMA-ES minimizes, we maximize quality, so negate.
        """
        if self._cma is None or not self._pending_tell:
            return

        pop_size = self._cma.population_size
        worst_fitness = 1e18

        # Build row_index -> quality map using exact row positions
        quality_by_row: dict[int, float] = {}
        if original_indices is not None:
            for i, row_idx in enumerate(original_indices):
                quality_by_row[int(row_idx)] = float(qualities[i])
        else:
            # Fallback: assume sequential rows 0..K-1
            for i in range(len(qualities)):
                quality_by_row[i] = float(qualities[i])

        # Pair each pending x_tell with its fitness via exact row index
        tell_pairs: list[tuple[np.ndarray, float]] = []

        for x_tell, row_idx in self._pending_tell:
            if row_idx < 0:
                # Excess sample beyond batch — assign worst fitness
                tell_pairs.append((x_tell, worst_fitness))
            elif row_idx in quality_by_row:
                tell_pairs.append((x_tell, -quality_by_row[row_idx]))  # negate for minimization
            else:
                # Row was pre-filtered out — assign worst fitness
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


class GASampler:
    """Genetic Algorithm sampler for exploitation phase.

    Natively discrete: tournament selection, uniform crossover, per-param mutation.
    Maintains a population across batches with elitism. Near-zero suggestion overhead.
    """

    def __init__(
        self,
        spec: EncodingSpec,
        population_size: int = 200,
        mutation_rate: float = 0.08,
        crossover_rate: float = 0.8,
        elite_pct: float = 0.2,
        tournament_size: int = 3,
        seed: int | None = None,
    ):
        self._spec = spec
        self._pop_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._elite_pct = elite_pct
        self._tournament_size = tournament_size
        self._rng = np.random.default_rng(seed)

        # Population: (pop_size, P) int64 indices + (pop_size,) fitness
        self._population: np.ndarray | None = None
        self._fitness: np.ndarray | None = None
        self._generation_count = 0

    def _init_population(
        self,
        mask: np.ndarray | None,
        locked: np.ndarray | None,
        neighborhood: NeighborhoodSpec | None,
    ) -> None:
        """Initialize random population respecting constraints."""
        p = self._spec.num_params
        pop = np.zeros((self._pop_size, p), dtype=np.int64)

        for col in self._spec.columns:
            if locked is not None and locked[col.index] >= 0:
                pop[:, col.index] = locked[col.index]
                continue
            n_vals = len(col.values)
            if neighborhood is not None:
                lo = int(neighborhood.min_bounds[col.index])
                hi = int(neighborhood.max_bounds[col.index])
            else:
                lo, hi = 0, n_vals - 1
            pop[:, col.index] = self._rng.integers(lo, hi + 1, size=self._pop_size)

        self._population = pop
        self._fitness = np.full(self._pop_size, -np.inf, dtype=np.float64)

    def _tournament_select(self, n_parents: int) -> np.ndarray:
        """Select parents via tournament selection. Returns indices into population."""
        parents = np.zeros(n_parents, dtype=np.int64)
        for i in range(n_parents):
            competitors = self._rng.integers(0, self._pop_size, size=self._tournament_size)
            best = competitors[np.argmax(self._fitness[competitors])]
            parents[i] = best
        return parents

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        mask: np.ndarray | None,
        locked: np.ndarray | None,
    ) -> np.ndarray:
        """Uniform crossover: each param independently from parent1 or parent2."""
        child = parent1.copy()
        if self._rng.random() < self._crossover_rate:
            swap = self._rng.random(len(child)) < 0.5
            # Only swap active, unlocked params
            for i in range(len(child)):
                if locked is not None and locked[i] >= 0:
                    continue
                if mask is not None and not mask[i]:
                    continue
                if swap[i]:
                    child[i] = parent2[i]
        return child

    def _mutate(
        self,
        individual: np.ndarray,
        mask: np.ndarray | None,
        locked: np.ndarray | None,
        neighborhood: NeighborhoodSpec | None,
    ) -> np.ndarray:
        """Per-param mutation: ordinal ±1-2 steps, categorical uniform random."""
        result = individual.copy()
        for col in self._spec.columns:
            if locked is not None and locked[col.index] >= 0:
                continue
            if mask is not None and not mask[col.index]:
                continue
            if self._rng.random() >= self._mutation_rate:
                continue

            n_vals = len(col.values)
            if neighborhood is not None:
                lo = int(neighborhood.min_bounds[col.index])
                hi = int(neighborhood.max_bounds[col.index])
            else:
                lo, hi = 0, n_vals - 1

            if col.is_categorical:
                # Categorical: uniform random from valid range
                result[col.index] = self._rng.integers(lo, hi + 1)
            else:
                # Ordinal: ±1 or ±2 steps
                delta = self._rng.choice([-2, -1, 1, 2])
                result[col.index] = np.clip(
                    int(result[col.index]) + delta, lo, hi,
                )
        return result

    def sample(
        self,
        n: int,
        mask: np.ndarray | None = None,
        locked: np.ndarray | None = None,
        neighborhood: NeighborhoodSpec | None = None,
    ) -> np.ndarray:
        """Generate n parameter sets using genetic operators.

        Returns (n, P) int64 index matrix.
        """
        p = self._spec.num_params

        # Initialize population on first call
        if self._population is None:
            self._init_population(mask, locked, neighborhood)

        out = np.zeros((n, p), dtype=np.int64)
        filled = 0

        while filled < n:
            # Generate one generation of offspring
            n_elite = max(1, int(self._pop_size * self._elite_pct))
            n_children = self._pop_size - n_elite

            # Elites survive unchanged
            elite_order = np.argsort(-self._fitness)[:n_elite]
            offspring = [self._population[i].copy() for i in elite_order]

            # Generate children via crossover + mutation
            if n_children > 0 and np.any(self._fitness > -np.inf):
                parents = self._tournament_select(n_children * 2)
                for c in range(n_children):
                    p1 = self._population[parents[c * 2]]
                    p2 = self._population[parents[c * 2 + 1]]
                    child = self._crossover(p1, p2, mask, locked)
                    child = self._mutate(child, mask, locked, neighborhood)
                    offspring.append(child)
            else:
                # No fitness info yet — generate random individuals
                for _ in range(n_children):
                    ind = np.zeros(p, dtype=np.int64)
                    for col in self._spec.columns:
                        if locked is not None and locked[col.index] >= 0:
                            ind[col.index] = locked[col.index]
                            continue
                        n_vals = len(col.values)
                        if neighborhood is not None:
                            lo = int(neighborhood.min_bounds[col.index])
                            hi = int(neighborhood.max_bounds[col.index])
                        else:
                            lo, hi = 0, n_vals - 1
                        ind[col.index] = self._rng.integers(lo, hi + 1)
                    offspring.append(ind)

            # Fill output from this generation
            gen_array = np.array(offspring, dtype=np.int64)
            use_count = min(len(gen_array), n - filled)
            out[filled:filled + use_count] = gen_array[:use_count]

            # Fill inactive params with uniform random (noise averaging)
            for col in self._spec.columns:
                if locked is not None and locked[col.index] >= 0:
                    continue
                if mask is not None and not mask[col.index]:
                    n_vals = len(col.values)
                    if neighborhood is not None:
                        lo = int(neighborhood.min_bounds[col.index])
                        hi = int(neighborhood.max_bounds[col.index])
                    else:
                        lo, hi = 0, n_vals - 1
                    out[filled:filled + use_count, col.index] = self._rng.integers(
                        lo, hi + 1, size=use_count,
                    )

            filled += use_count
            self._generation_count += 1

        return out

    def update(
        self,
        index_batch: np.ndarray,
        qualities: np.ndarray,
        mask: np.ndarray | None = None,
        original_indices: np.ndarray | None = None,
    ) -> None:
        """Update GA population with evaluation results.

        Replaces population with the best individuals seen so far (steady-state).
        """
        if self._population is None or len(qualities) == 0:
            return

        # Merge current population + new evaluated individuals
        # Keep the best pop_size individuals by fitness
        new_pop = index_batch.copy()
        new_fit = qualities.copy()

        combined_pop = np.vstack([self._population, new_pop])
        combined_fit = np.concatenate([self._fitness, new_fit])

        # Select top pop_size by fitness
        top_order = np.argsort(-combined_fit)[:self._pop_size]
        self._population = combined_pop[top_order].copy()
        self._fitness = combined_fit[top_order].copy()

    def reset(self) -> None:
        """Reset population for a new stage."""
        self._population = None
        self._fitness = None
        self._generation_count = 0

    def entropy(self, mask: np.ndarray | None = None) -> float:
        """Population diversity: normalized Shannon entropy of allele frequencies."""
        if self._population is None:
            return 1.0

        entropies = []
        for col in self._spec.columns:
            if mask is not None and not mask[col.index]:
                continue
            n_vals = len(col.values)
            if n_vals <= 1:
                entropies.append(0.0)
                continue
            # Count allele frequencies
            values = self._population[:, col.index]
            counts = np.bincount(values, minlength=n_vals).astype(np.float64)
            freqs = counts / counts.sum()
            nonzero = freqs[freqs > 0]
            h = -np.sum(nonzero * np.log2(nonzero))
            h_max = np.log2(n_vals)
            entropies.append(h / h_max if h_max > 0 else 0.0)

        return float(np.mean(entropies)) if entropies else 1.0

    @property
    def effective_lr(self) -> float:
        """Compatibility property for dashboard — returns mutation rate."""
        return self._mutation_rate

    @property
    def converged(self) -> bool:
        """Check if population has lost diversity."""
        return self.entropy() < 0.05
