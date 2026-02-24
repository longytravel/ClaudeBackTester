"""Parameter encoding: converts ParamSpace + param dicts to/from (N,P) float64 matrices.

Handles:
- Numeric params: stored as actual float64 values
- Categorical params (strings): stored as integer indices into the values list
- Booleans: stored as 0.0 / 1.0
- List params (allowed_days): encoded as bitmask (Mon=bit0..Sun=bit6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backtester.core.dtypes import DAYS_BITMASK
from backtester.strategies.base import ParamDef, ParamSpace


@dataclass
class ParamColumn:
    """Describes one column in the encoded parameter matrix."""
    name: str
    index: int            # Column index in the (N,P) matrix
    group: str            # Parameter group name
    values: list[Any]     # Original allowed values (for decode)
    is_categorical: bool  # True if values are strings
    is_boolean: bool      # True if values are [False, True]
    is_bitmask: bool      # True if values are lists (e.g., allowed_days)


@dataclass
class EncodingSpec:
    """Full encoding specification for a ParamSpace.

    Maps between Python param dicts and (N, P) float64 matrices.
    """
    columns: list[ParamColumn] = field(default_factory=list)
    name_to_index: dict[str, int] = field(default_factory=dict)
    num_params: int = 0

    def column(self, name: str) -> ParamColumn:
        return self.columns[self.name_to_index[name]]

    def group_indices(self, group: str) -> list[int]:
        """Get column indices for all params in a group."""
        return [c.index for c in self.columns if c.group == group]

    def group_names(self, group: str) -> list[str]:
        """Get param names for a group."""
        return [c.name for c in self.columns if c.group == group]

    @property
    def groups(self) -> dict[str, list[int]]:
        """Map group name -> list of column indices."""
        out: dict[str, list[int]] = {}
        for c in self.columns:
            out.setdefault(c.group, []).append(c.index)
        return out

    def num_values(self, name: str) -> int:
        """Number of allowed values for a parameter."""
        return len(self.columns[self.name_to_index[name]].values)


def _is_list_of_lists(values: list[Any]) -> bool:
    """Check if values are lists (like allowed_days)."""
    return len(values) > 0 and isinstance(values[0], list)


def _is_boolean_param(values: list[Any]) -> bool:
    """Check if values are [False, True] or [True, False]."""
    return set(values) == {True, False} and len(values) == 2


def _is_categorical(values: list[Any]) -> bool:
    """Check if values are strings (categorical)."""
    return len(values) > 0 and isinstance(values[0], str)


def _days_list_to_bitmask(days: list[int]) -> float:
    """Convert a list of day numbers to bitmask float."""
    if not isinstance(days, (list, tuple)):
        raise TypeError(
            f"_days_list_to_bitmask expected list, got {type(days).__name__}: {days!r}"
        )
    mask = 0
    for d in days:
        mask |= DAYS_BITMASK[int(d)]
    return float(mask)


def _bitmask_to_days_list(mask: float) -> list[int]:
    """Convert bitmask float back to list of day numbers."""
    mask_int = int(mask)
    return [d for d, bit in sorted(DAYS_BITMASK.items()) if mask_int & bit]


def build_encoding_spec(param_space: ParamSpace) -> EncodingSpec:
    """Build an EncodingSpec from a ParamSpace.

    Each parameter becomes one column in the (N, P) matrix.
    Column order matches iteration order of the ParamSpace.
    """
    spec = EncodingSpec()
    for idx, pdef in enumerate(param_space):
        is_bitmask = _is_list_of_lists(pdef.values)
        # Must check bitmask BEFORE boolean — list values can't be hashed
        is_bool = False if is_bitmask else _is_boolean_param(pdef.values)
        is_cat = False if is_bitmask else _is_categorical(pdef.values)

        col = ParamColumn(
            name=pdef.name,
            index=idx,
            group=pdef.group,
            values=pdef.values,
            is_categorical=is_cat,
            is_boolean=is_bool,
            is_bitmask=is_bitmask,
        )
        spec.columns.append(col)
        spec.name_to_index[pdef.name] = idx

    spec.num_params = len(spec.columns)
    return spec


def encode_value(col: ParamColumn, value: Any) -> float:
    """Encode a single Python value to float64 for JIT."""
    if col.is_bitmask:
        return _days_list_to_bitmask(value)
    elif col.is_boolean:
        return 1.0 if value else 0.0
    elif col.is_categorical:
        return float(col.values.index(value))
    else:
        return float(value)


def decode_value(col: ParamColumn, encoded: float) -> Any:
    """Decode a single float64 back to Python value."""
    if col.is_bitmask:
        return _bitmask_to_days_list(encoded)
    elif col.is_boolean:
        return bool(encoded >= 0.5)
    elif col.is_categorical:
        idx = int(round(encoded))
        return col.values[idx]
    else:
        # Return as the same type as the original values
        if col.values and isinstance(col.values[0], int):
            return int(round(encoded))
        return encoded


def encode_params(spec: EncodingSpec, params: dict[str, Any]) -> np.ndarray:
    """Encode a single param dict to a 1D float64 array of shape (P,)."""
    row = np.zeros(spec.num_params, dtype=np.float64)
    for col in spec.columns:
        if col.name in params:
            row[col.index] = encode_value(col, params[col.name])
    return row


def decode_params(spec: EncodingSpec, row: np.ndarray) -> dict[str, Any]:
    """Decode a 1D float64 array back to a param dict."""
    params: dict[str, Any] = {}
    for col in spec.columns:
        params[col.name] = decode_value(col, row[col.index])
    return params


def encode_batch(spec: EncodingSpec, param_dicts: list[dict[str, Any]]) -> np.ndarray:
    """Encode N param dicts to an (N, P) float64 matrix."""
    n = len(param_dicts)
    matrix = np.zeros((n, spec.num_params), dtype=np.float64)
    for i, params in enumerate(param_dicts):
        matrix[i] = encode_params(spec, params)
    return matrix


def decode_batch(spec: EncodingSpec, matrix: np.ndarray) -> list[dict[str, Any]]:
    """Decode an (N, P) float64 matrix back to N param dicts."""
    return [decode_params(spec, matrix[i]) for i in range(matrix.shape[0])]


def indices_to_values(spec: EncodingSpec, index_matrix: np.ndarray) -> np.ndarray:
    """Convert an (N, P) index matrix to an (N, P) value matrix.

    The optimizer works in index space (0..len(values)-1 per param).
    This converts to actual float64 values for the JIT loop.
    Categoricals remain as integer indices (0, 1, 2...).
    Booleans remain as 0.0/1.0.
    Bitmask params: index into values list, then convert to bitmask.
    Numeric params: index into values list, return actual value.
    """
    n, p = index_matrix.shape
    value_matrix = np.zeros((n, p), dtype=np.float64)

    for col in spec.columns:
        col_indices = index_matrix[:, col.index].astype(np.int64)
        if col.is_bitmask:
            for i in range(n):
                idx = col_indices[i]
                value_matrix[i, col.index] = _days_list_to_bitmask(col.values[idx])
        elif col.is_boolean:
            # Boolean values list is [False, True] — index 0=False, 1=True
            for i in range(n):
                value_matrix[i, col.index] = float(col_indices[i])
        elif col.is_categorical:
            # Categorical stays as index (the JIT uses integer codes)
            value_matrix[:, col.index] = col_indices.astype(np.float64)
        else:
            # Numeric: look up actual value from values list
            vals_array = np.array(col.values, dtype=np.float64)
            value_matrix[:, col.index] = vals_array[col_indices]

    return value_matrix


def random_index_matrix(
    spec: EncodingSpec,
    n: int,
    rng: np.random.Generator | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Generate (N, P) random index matrix.

    Each column i gets random integers in [0, len(values_i)).
    If mask is provided (shape (P,) bool), only generate for True columns;
    False columns get index 0.
    """
    rng = rng or np.random.default_rng()
    matrix = np.zeros((n, spec.num_params), dtype=np.int64)

    for col in spec.columns:
        if mask is not None and not mask[col.index]:
            continue
        num_vals = len(col.values)
        matrix[:, col.index] = rng.integers(0, num_vals, size=n)

    return matrix
