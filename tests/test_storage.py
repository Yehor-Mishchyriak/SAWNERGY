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


def test_compress_and_cleanup_success(tmp_path, monkeypatch):
    output = tmp_path / "archive.zip"
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    closed = []
    close = sawnergy_util.ArrayStorage.close

    def record_close(self):
        closed.append(self.store_path)
        close(self)

    monkeypatch.setattr(sawnergy_util.ArrayStorage, "close", record_close)
    with sawnergy_util.ArrayStorage.compress_and_cleanup(output, 0) as storage:
        working_path = storage.store_path
        storage.write(data, "BLOCK", arrays_per_chunk=1)
        storage.add_attr("description", "round trip")
        assert working_path.is_dir()
        assert not output.exists()

    assert working_path in closed
    assert not working_path.exists()
    with sawnergy_util.ArrayStorage(output, "r") as archive:
        np.testing.assert_array_equal(archive.read("BLOCK"), data)
        assert archive.get_attr("description") == "round trip"


@pytest.mark.parametrize("existing_output", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_compress_and_cleanup_body_failure(tmp_path, monkeypatch, existing_output, error_type):
    output = tmp_path / "archive.zip"
    if existing_output:
        output.write_bytes(b"previous output")
    error = error_type("extraction failed")
    closed = []

    def unexpected_compress(*args, **kwargs):
        pytest.fail("Compression must not run after a body failure")

    monkeypatch.setattr(sawnergy_util.ArrayStorage, "compress", unexpected_compress)
    with pytest.raises(error_type) as caught:
        with sawnergy_util.ArrayStorage.compress_and_cleanup(output, 0) as storage:
            working_path = storage.store_path

            def failing_close():
                closed.append(True)
                raise OSError("store close failed")

            # The public close() already logs ordinary underlying close errors.
            monkeypatch.setattr(storage.store, "close", failing_close)
            raise error

    assert caught.value is error
    assert closed
    assert not working_path.exists()
    if existing_output:
        assert output.read_bytes() == b"previous output"
    else:
        assert not output.exists()


def test_compress_and_cleanup_compression_failure(tmp_path, monkeypatch):
    error = OSError("compression failed")
    closed = []
    close = sawnergy_util.ArrayStorage.close

    def record_close(self):
        closed.append(self.store_path)
        close(self)

    def failing_compress(self, into, *, compression_level):
        assert self.store_path.is_dir()
        raise error

    monkeypatch.setattr(sawnergy_util.ArrayStorage, "close", record_close)
    monkeypatch.setattr(sawnergy_util.ArrayStorage, "compress", failing_compress)
    with pytest.raises(OSError) as caught:
        with sawnergy_util.ArrayStorage.compress_and_cleanup(tmp_path / "archive.zip", 0) as storage:
            working_path = storage.store_path

    assert caught.value is error
    assert working_path in closed
    assert not working_path.exists()


def test_compress_and_cleanup_initialization_failure(tmp_path, monkeypatch):
    error = OSError("store initialization failed")
    paths = []

    def failing_init(self, pth, mode):
        path = sawnergy_util.Path(pth)
        assert path.is_dir()
        paths.append(path)
        raise error

    monkeypatch.setattr(sawnergy_util.ArrayStorage, "__init__", failing_init)
    with pytest.raises(OSError) as caught:
        with sawnergy_util.ArrayStorage.compress_and_cleanup(tmp_path / "archive.zip", 0):
            pytest.fail("A failed store must not be yielded")

    assert caught.value is error
    assert len(paths) == 1
    assert not paths[0].exists()


@pytest.mark.parametrize("failure_stage", ["body", "compress", None])
def test_compress_and_cleanup_cleanup_failure(tmp_path, monkeypatch, failure_stage):
    error = RuntimeError("work failed")
    cleanup_error = OSError("directory cleanup failed")
    cleanup = sawnergy_util.tempfile.TemporaryDirectory.cleanup

    def failing_cleanup(self):
        cleanup(self)
        raise cleanup_error

    def compress(self, into, *, compression_level):
        if failure_stage == "compress":
            raise error

    monkeypatch.setattr(sawnergy_util.tempfile.TemporaryDirectory, "cleanup", failing_cleanup)
    monkeypatch.setattr(sawnergy_util.ArrayStorage, "compress", compress)
    expected = cleanup_error if failure_stage is None else error
    with pytest.raises(type(expected)) as caught:
        with sawnergy_util.ArrayStorage.compress_and_cleanup(tmp_path / "archive.zip", 0) as storage:
            working_path = storage.store_path
            if failure_stage == "body":
                raise error

    assert caught.value is expected
    assert not working_path.exists()
