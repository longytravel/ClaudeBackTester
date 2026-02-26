"""Progress callback protocol for optimization and pipeline stages.

Defines dataclasses for progress events and a Protocol class that
consumers (e.g., the dashboard WebSocket server) can implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class BatchProgress:
    """Fired after each optimizer batch evaluation."""

    stage_name: str
    stage_index: int
    total_stages: int
    trials_done: int
    trials_total: int
    best_quality: float
    best_sharpe: float
    best_trades: int
    valid_count: int
    valid_rate: float
    batch_best_quality: float
    batch_mean_quality: float
    phase: str  # "exploration" | "exploitation"
    entropy: float | None
    effective_lr: float | None
    evals_per_sec: float
    elapsed_secs: float


@dataclass
class StageComplete:
    """Fired when an optimization stage finishes."""

    stage_name: str
    stage_index: int
    total_stages: int
    best_quality: float
    best_metrics: dict[str, float]
    trials_evaluated: int
    valid_count: int
    elapsed_secs: float


@dataclass
class PipelineProgress:
    """Fired after each pipeline validation stage."""

    stage_name: str
    candidates_total: int
    candidates_surviving: int
    detail: str


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for receiving optimization/pipeline progress updates."""

    def on_batch(self, p: BatchProgress) -> None: ...
    def on_stage(self, p: StageComplete) -> None: ...
    def on_pipeline(self, p: PipelineProgress) -> None: ...
