"""MAP-Elites diversity archive for optimization.

Maintains a grid of elite solutions indexed by behavioral descriptors.
Ensures the final candidate set is diverse, not just N copies of the
same strategy with slightly different parameters.

Descriptors:
- Trade frequency bucket (trades/year)
- Average hold time bucket
- SL mode
- Management complexity (how many management features enabled)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtester.core.dtypes import (
    M_QUALITY,
    M_TRADES,
    NUM_METRICS,
)


@dataclass
class ArchiveEntry:
    """A single entry in the MAP-Elites archive."""
    trial_index: int           # Original index in the metrics array
    quality: float             # Quality score
    descriptors: tuple         # Hashable descriptor tuple (grid cell key)
    metrics: np.ndarray        # Full metrics row
    params: np.ndarray | None = None  # Parameter values (optional)


class DiversityArchive:
    """MAP-Elites diversity archive.

    Descriptor dimensions:
    - trade_freq: [low, medium, high] based on trade count
    - quality_tier: [poor, okay, good, excellent] based on quality quartiles

    Each grid cell holds the best entry (by quality score).
    """

    def __init__(
        self,
        trade_freq_bins: list[float] | None = None,
        quality_bins: list[float] | None = None,
    ):
        # Trade frequency bins: boundaries for [low, medium, high]
        self.trade_freq_bins = trade_freq_bins or [30.0, 100.0]
        # Quality bins: boundaries for [poor, okay, good, excellent]
        self.quality_bins = quality_bins or [5.0, 15.0, 30.0]
        self.grid: dict[tuple, ArchiveEntry] = {}

    def _compute_descriptors(self, metrics_row: np.ndarray) -> tuple:
        """Extract descriptor tuple from a metrics row."""
        trades = metrics_row[M_TRADES]
        quality = metrics_row[M_QUALITY]

        # Bucket trade frequency
        trade_bucket = 0
        for b in self.trade_freq_bins:
            if trades >= b:
                trade_bucket += 1

        # Bucket quality
        quality_bucket = 0
        for b in self.quality_bins:
            if quality >= b:
                quality_bucket += 1

        return (trade_bucket, quality_bucket)

    def add(
        self,
        trial_index: int,
        metrics_row: np.ndarray,
        params: np.ndarray | None = None,
    ) -> bool:
        """Add a candidate to the archive. Returns True if it was inserted/replaced."""
        quality = float(metrics_row[M_QUALITY])
        desc = self._compute_descriptors(metrics_row)

        entry = ArchiveEntry(
            trial_index=trial_index,
            quality=quality,
            descriptors=desc,
            metrics=metrics_row.copy(),
            params=params.copy() if params is not None else None,
        )

        if desc not in self.grid or quality > self.grid[desc].quality:
            self.grid[desc] = entry
            return True
        return False

    def add_batch(
        self,
        metrics: np.ndarray,
        params: np.ndarray | None = None,
        offset: int = 0,
    ) -> int:
        """Add a batch of candidates. Returns number inserted/replaced."""
        count = 0
        for i in range(metrics.shape[0]):
            p = params[i] if params is not None else None
            if self.add(offset + i, metrics[i], p):
                count += 1
        return count

    def get_top_n(self, n: int = 50) -> list[ArchiveEntry]:
        """Get top N entries by quality, ensuring diversity across grid cells."""
        entries = sorted(self.grid.values(), key=lambda e: -e.quality)
        return entries[:n]

    def get_all(self) -> list[ArchiveEntry]:
        """Get all archive entries sorted by quality."""
        return sorted(self.grid.values(), key=lambda e: -e.quality)

    @property
    def size(self) -> int:
        return len(self.grid)

    def clear(self) -> None:
        self.grid.clear()


def select_top_n_diverse(
    metrics: np.ndarray,
    n: int = 50,
    valid_mask: np.ndarray | None = None,
    params: np.ndarray | None = None,
) -> list[int]:
    """Select top N diverse candidates using MAP-Elites archive.

    Returns list of trial indices ensuring diversity across
    trade frequency and quality tiers.
    """
    archive = DiversityArchive()

    candidates = np.arange(metrics.shape[0])
    if valid_mask is not None:
        candidates = candidates[valid_mask]

    for idx in candidates:
        p = params[idx] if params is not None else None
        archive.add(int(idx), metrics[idx], p)

    entries = archive.get_top_n(n)
    return [e.trial_index for e in entries]
