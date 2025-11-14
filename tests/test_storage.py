from __future__ import annotations
import numpy as np
import pytest
from sawnergy import sawnergy_util

def test_write_accepts_stacked_ndarray(tmp_path):
    p = tmp_path / "arrs"
    with sawnergy_util.ArrayStorage(p, mode="w") as st:
        X = np.arange(12, dtype=np.float32).reshape(3, 2, 2)  # (k, *item_shape)
        st.write(X, to_block_named="BLOCK")                   # stacked ndarray ok
        Y = st.read("BLOCK", slice(None))
        assert Y.shape == X.shape and Y.dtype == X.dtype

def test_arrays_per_chunk_defaults_warning(tmp_path):
    p = tmp_path / "arrs2"
    with sawnergy_util.ArrayStorage(p, mode="w") as st:
        A = [np.zeros((2, 2), dtype=np.float32)] * 2
        with pytest.warns(RuntimeWarning):
            st.write(A, to_block_named="B")                  # no arrays_per_chunk provided
        chunk_meta = dict(st.root.attrs.get("array_chunk_size_in_block", {}))
        assert chunk_meta["B"] == 10

def test_mismatched_shape_or_dtype_rejected(tmp_path):
    p = tmp_path / "arrs3"
    with sawnergy_util.ArrayStorage(p, mode="w") as st:
        good = [np.zeros((2, 2), dtype=np.float32)] * 2
        bad_shape = [np.zeros((2, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float32)]
        bad_dtype = [np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float64)]
        st.write(good, to_block_named="C")
        with pytest.raises(ValueError):
            st.write(bad_shape, to_block_named="D")
        with pytest.raises(ValueError):
            st.write(bad_dtype, to_block_named="E")

def test_delete_block_errors_in_readonly(tmp_path):
    p = tmp_path / "arrs4"
    with sawnergy_util.ArrayStorage(p, mode="w") as st:
        st.write([np.zeros((1,), dtype=np.int16)], to_block_named="X")
    # reopen read-only
    with sawnergy_util.ArrayStorage(p, mode="r") as st:
        with pytest.raises(RuntimeError):
            st.delete_block("X")

def test_repr_lists_blocks_sorted(tmp_path):
    p = tmp_path / "arrs5"
    with sawnergy_util.ArrayStorage(p, mode="w") as st:
        st.write([np.zeros((1,), dtype=np.int32)], to_block_named="B")
        st.write([np.zeros((1,), dtype=np.int32)], to_block_named="A")
        blocks = st.list_blocks()
        assert blocks == ["A", "B"]
        rep = repr(st)
        assert "ArrayStorage" in rep and "blocks=2" in rep
