"""Tests for parameter encoding: round-trip, categorical mapping, bitmask."""

import numpy as np
import pytest

from backtester.core.encoding import (
    EncodingSpec,
    build_encoding_spec,
    decode_params,
    encode_params,
    encode_batch,
    decode_batch,
    indices_to_values,
    random_index_matrix,
)
from backtester.strategies.base import ParamDef, ParamSpace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_param_space() -> ParamSpace:
    """ParamSpace with all param types: numeric, categorical, boolean, bitmask."""
    return ParamSpace([
        ParamDef("rsi_period", [7, 14, 21], group="signal"),
        ParamDef("sl_mode", ["fixed_pips", "atr_based", "swing"], group="risk"),
        ParamDef("sl_fixed_pips", [10, 20, 30, 50], group="risk"),
        ParamDef("breakeven_enabled", [False, True], group="management"),
        ParamDef("allowed_days", [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3],
            [1, 2, 3],
        ], group="time"),
    ])


# ---------------------------------------------------------------------------
# build_encoding_spec
# ---------------------------------------------------------------------------

class TestBuildEncodingSpec:
    def test_column_count(self):
        spec = build_encoding_spec(_make_param_space())
        assert spec.num_params == 5

    def test_column_indices(self):
        spec = build_encoding_spec(_make_param_space())
        assert spec.name_to_index["rsi_period"] == 0
        assert spec.name_to_index["sl_mode"] == 1
        assert spec.name_to_index["breakeven_enabled"] == 3
        assert spec.name_to_index["allowed_days"] == 4

    def test_categorical_detection(self):
        spec = build_encoding_spec(_make_param_space())
        assert spec.column("sl_mode").is_categorical is True
        assert spec.column("rsi_period").is_categorical is False

    def test_boolean_detection(self):
        spec = build_encoding_spec(_make_param_space())
        assert spec.column("breakeven_enabled").is_boolean is True
        assert spec.column("rsi_period").is_boolean is False

    def test_bitmask_detection(self):
        spec = build_encoding_spec(_make_param_space())
        assert spec.column("allowed_days").is_bitmask is True
        assert spec.column("rsi_period").is_bitmask is False

    def test_group_indices(self):
        spec = build_encoding_spec(_make_param_space())
        risk_idx = spec.group_indices("risk")
        assert len(risk_idx) == 2
        assert spec.columns[risk_idx[0]].name == "sl_mode"
        assert spec.columns[risk_idx[1]].name == "sl_fixed_pips"

    def test_groups_property(self):
        spec = build_encoding_spec(_make_param_space())
        groups = spec.groups
        assert "signal" in groups
        assert "risk" in groups
        assert "management" in groups
        assert "time" in groups

    def test_num_values(self):
        spec = build_encoding_spec(_make_param_space())
        assert spec.num_values("rsi_period") == 3
        assert spec.num_values("sl_fixed_pips") == 4
        assert spec.num_values("breakeven_enabled") == 2


# ---------------------------------------------------------------------------
# Round-trip encode/decode
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_numeric_roundtrip(self):
        spec = build_encoding_spec(_make_param_space())
        params = {"rsi_period": 14, "sl_mode": "atr_based", "sl_fixed_pips": 30,
                  "breakeven_enabled": True, "allowed_days": [0, 1, 2, 3, 4]}
        encoded = encode_params(spec, params)
        decoded = decode_params(spec, encoded)
        assert decoded["rsi_period"] == 14
        assert decoded["sl_fixed_pips"] == 30

    def test_categorical_roundtrip(self):
        spec = build_encoding_spec(_make_param_space())
        for mode in ["fixed_pips", "atr_based", "swing"]:
            params = {"rsi_period": 7, "sl_mode": mode, "sl_fixed_pips": 10,
                      "breakeven_enabled": False, "allowed_days": [0, 1, 2, 3, 4]}
            encoded = encode_params(spec, params)
            decoded = decode_params(spec, encoded)
            assert decoded["sl_mode"] == mode

    def test_boolean_roundtrip(self):
        spec = build_encoding_spec(_make_param_space())
        for val in [True, False]:
            params = {"rsi_period": 7, "sl_mode": "fixed_pips", "sl_fixed_pips": 10,
                      "breakeven_enabled": val, "allowed_days": [0, 1, 2, 3, 4]}
            encoded = encode_params(spec, params)
            decoded = decode_params(spec, encoded)
            assert decoded["breakeven_enabled"] is val

    def test_bitmask_roundtrip(self):
        spec = build_encoding_spec(_make_param_space())
        days_options = [[0, 1, 2, 3, 4], [0, 1, 2, 3], [1, 2, 3]]
        for days in days_options:
            params = {"rsi_period": 7, "sl_mode": "fixed_pips", "sl_fixed_pips": 10,
                      "breakeven_enabled": False, "allowed_days": days}
            encoded = encode_params(spec, params)
            decoded = decode_params(spec, encoded)
            assert decoded["allowed_days"] == days

    def test_full_roundtrip(self):
        spec = build_encoding_spec(_make_param_space())
        params = {"rsi_period": 21, "sl_mode": "swing", "sl_fixed_pips": 50,
                  "breakeven_enabled": True, "allowed_days": [1, 2, 3]}
        encoded = encode_params(spec, params)
        decoded = decode_params(spec, encoded)
        assert decoded == params


# ---------------------------------------------------------------------------
# Batch encode/decode
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_shape(self):
        spec = build_encoding_spec(_make_param_space())
        dicts = [
            {"rsi_period": 7, "sl_mode": "fixed_pips", "sl_fixed_pips": 10,
             "breakeven_enabled": False, "allowed_days": [0, 1, 2, 3, 4]},
            {"rsi_period": 21, "sl_mode": "swing", "sl_fixed_pips": 50,
             "breakeven_enabled": True, "allowed_days": [1, 2, 3]},
        ]
        matrix = encode_batch(spec, dicts)
        assert matrix.shape == (2, 5)
        assert matrix.dtype == np.float64

    def test_batch_roundtrip(self):
        spec = build_encoding_spec(_make_param_space())
        original = [
            {"rsi_period": 7, "sl_mode": "fixed_pips", "sl_fixed_pips": 10,
             "breakeven_enabled": False, "allowed_days": [0, 1, 2, 3, 4]},
            {"rsi_period": 21, "sl_mode": "swing", "sl_fixed_pips": 50,
             "breakeven_enabled": True, "allowed_days": [1, 2, 3]},
        ]
        matrix = encode_batch(spec, original)
        decoded = decode_batch(spec, matrix)
        assert decoded == original


# ---------------------------------------------------------------------------
# Bitmask encoding specifics
# ---------------------------------------------------------------------------

class TestBitmask:
    def test_mon_fri_bitmask(self):
        """Mon-Fri should be 31 (0b11111)."""
        spec = build_encoding_spec(_make_param_space())
        params = {"rsi_period": 7, "sl_mode": "fixed_pips", "sl_fixed_pips": 10,
                  "breakeven_enabled": False, "allowed_days": [0, 1, 2, 3, 4]}
        encoded = encode_params(spec, params)
        assert encoded[spec.name_to_index["allowed_days"]] == 31.0

    def test_mon_thu_bitmask(self):
        """Mon-Thu should be 15 (0b01111)."""
        spec = build_encoding_spec(_make_param_space())
        params = {"rsi_period": 7, "sl_mode": "fixed_pips", "sl_fixed_pips": 10,
                  "breakeven_enabled": False, "allowed_days": [0, 1, 2, 3]}
        encoded = encode_params(spec, params)
        assert encoded[spec.name_to_index["allowed_days"]] == 15.0

    def test_tue_thu_bitmask(self):
        """Tue-Thu should be 14 (0b01110)."""
        spec = build_encoding_spec(_make_param_space())
        params = {"rsi_period": 7, "sl_mode": "fixed_pips", "sl_fixed_pips": 10,
                  "breakeven_enabled": False, "allowed_days": [1, 2, 3]}
        encoded = encode_params(spec, params)
        assert encoded[spec.name_to_index["allowed_days"]] == 14.0


# ---------------------------------------------------------------------------
# Index matrix → value matrix conversion
# ---------------------------------------------------------------------------

class TestIndicesConversion:
    def test_indices_to_values_numeric(self):
        spec = build_encoding_spec(_make_param_space())
        # rsi_period values: [7, 14, 21] — index 1 → value 14
        idx_matrix = np.array([[1, 0, 2, 0, 0]], dtype=np.int64)
        val_matrix = indices_to_values(spec, idx_matrix)
        assert val_matrix[0, 0] == 14.0  # rsi_period

    def test_indices_to_values_categorical(self):
        spec = build_encoding_spec(_make_param_space())
        # sl_mode values: ["fixed_pips", "atr_based", "swing"] — index stays as index
        idx_matrix = np.array([[0, 2, 0, 0, 0]], dtype=np.int64)
        val_matrix = indices_to_values(spec, idx_matrix)
        assert val_matrix[0, 1] == 2.0  # sl_mode index 2 = "swing"

    def test_indices_to_values_boolean(self):
        spec = build_encoding_spec(_make_param_space())
        idx_matrix = np.array([[0, 0, 0, 1, 0]], dtype=np.int64)
        val_matrix = indices_to_values(spec, idx_matrix)
        assert val_matrix[0, 3] == 1.0  # breakeven_enabled = True

    def test_indices_to_values_bitmask(self):
        spec = build_encoding_spec(_make_param_space())
        # allowed_days values: [[0,1,2,3,4], [0,1,2,3], [1,2,3]]
        # index 0 → Mon-Fri → bitmask 31
        idx_matrix = np.array([[0, 0, 0, 0, 0]], dtype=np.int64)
        val_matrix = indices_to_values(spec, idx_matrix)
        assert val_matrix[0, 4] == 31.0

    def test_indices_to_values_batch(self):
        spec = build_encoding_spec(_make_param_space())
        idx_matrix = np.array([
            [0, 0, 0, 0, 0],  # rsi=7, sl_mode=fixed, sl_pips=10, be=False, days=Mon-Fri
            [2, 1, 3, 1, 2],  # rsi=21, sl_mode=atr, sl_pips=50, be=True, days=Tue-Thu
        ], dtype=np.int64)
        val_matrix = indices_to_values(spec, idx_matrix)
        assert val_matrix.shape == (2, 5)
        assert val_matrix[0, 0] == 7.0
        assert val_matrix[1, 0] == 21.0
        assert val_matrix[1, 2] == 50.0
        assert val_matrix[1, 3] == 1.0
        assert val_matrix[1, 4] == 14.0  # Tue-Thu bitmask


# ---------------------------------------------------------------------------
# Random index matrix
# ---------------------------------------------------------------------------

class TestRandomIndexMatrix:
    def test_shape(self):
        spec = build_encoding_spec(_make_param_space())
        matrix = random_index_matrix(spec, 100)
        assert matrix.shape == (100, 5)

    def test_values_in_range(self):
        spec = build_encoding_spec(_make_param_space())
        matrix = random_index_matrix(spec, 1000)
        for col in spec.columns:
            col_vals = matrix[:, col.index]
            assert col_vals.min() >= 0
            assert col_vals.max() < len(col.values)

    def test_mask_zeroes_unmasked(self):
        spec = build_encoding_spec(_make_param_space())
        mask = np.array([True, False, True, False, True], dtype=bool)
        matrix = random_index_matrix(spec, 100, mask=mask)
        # Unmasked columns should all be 0
        assert np.all(matrix[:, 1] == 0)
        assert np.all(matrix[:, 3] == 0)

    def test_reproducible(self):
        spec = build_encoding_spec(_make_param_space())
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        m1 = random_index_matrix(spec, 50, rng=rng1)
        m2 = random_index_matrix(spec, 50, rng=rng2)
        np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_param_space(self):
        spec = build_encoding_spec(ParamSpace())
        assert spec.num_params == 0
        row = encode_params(spec, {})
        assert row.shape == (0,)
        decoded = decode_params(spec, row)
        assert decoded == {}

    def test_single_value_param(self):
        ps = ParamSpace([ParamDef("fixed", [42], group="signal")])
        spec = build_encoding_spec(ps)
        encoded = encode_params(spec, {"fixed": 42})
        decoded = decode_params(spec, encoded)
        assert decoded["fixed"] == 42

    def test_float_numeric_values(self):
        ps = ParamSpace([ParamDef("mult", [0.5, 1.0, 1.5, 2.0], group="risk")])
        spec = build_encoding_spec(ps)
        encoded = encode_params(spec, {"mult": 1.5})
        decoded = decode_params(spec, encoded)
        assert abs(decoded["mult"] - 1.5) < 1e-10
