from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pytest
from sawnergy import sawnergy_util


def _storage_state(storage):
    return (
        json.dumps(dict(storage.root.attrs), sort_keys=True),
        {p.relative_to(storage.store_path): p.read_bytes() if p.is_file() else None
         for p in storage.store_path.rglob("*")},
    )


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_write_rejected_batch_leaves_storage_unchanged(tmp_path, existing, mismatch):
    good = np.ones((2, 2), dtype=np.float32)
    bad = (np.ones((3, 2), dtype=np.float32) if mismatch == "shape"
           else np.ones((2, 2), dtype=np.float64))
    with sawnergy_util.ArrayStorage(tmp_path / "rejected_batch", "w") as storage:
        if existing:
            storage.write([good], "BLOCK", arrays_per_chunk=2)
        before = _storage_state(storage)
        with pytest.raises(ValueError):
            storage.write([good, bad], "BLOCK", arrays_per_chunk=2)
        assert _storage_state(storage) == before


@pytest.mark.parametrize("chunk,error", [(0, ValueError), (-1, ValueError),
                                       (0.5, ValueError), (float("nan"), ValueError),
                                       ("bad", TypeError)])
def test_write_invalid_chunk_leaves_storage_unchanged(tmp_path, chunk, error):
    with sawnergy_util.ArrayStorage(tmp_path / "invalid_chunk", "w") as storage:
        before = _storage_state(storage)
        with pytest.raises(error):
            storage.write(np.ones((1, 2, 2), dtype=np.float32), "BLOCK", arrays_per_chunk=chunk)
        assert _storage_state(storage) == before


@pytest.mark.parametrize("conflict", ["shape", "dtype", "chunk"])
def test_write_cached_conflict_does_not_fill_other_metadata(tmp_path, conflict):
    with sawnergy_util.ArrayStorage(tmp_path / "cached_conflict", "w") as storage:
        storage.write(np.ones((1, 2, 2), dtype=np.float32), "BLOCK", arrays_per_chunk=2)
        storage.add_attr("array_shape_in_block", {"BLOCK": [3, 2]} if conflict == "shape" else {})
        storage.add_attr("array_dtype_in_block", {"BLOCK": np.dtype("float64").str} if conflict == "dtype" else {})
        before = _storage_state(storage)
        with pytest.raises(RuntimeError):
            storage.write(np.ones((1, 2, 2), dtype=np.float32), "BLOCK",
                          arrays_per_chunk=3 if conflict == "chunk" else 2)
        assert _storage_state(storage) == before


@pytest.mark.parametrize("conflict,error", [("shape", TypeError), ("dtype", TypeError),
                                          ("subgroup", TypeError), ("scalar", TypeError),
                                          ("chunk", RuntimeError)])
def test_write_existing_member_conflict_leaves_storage_unchanged(tmp_path, conflict, error):
    with sawnergy_util.ArrayStorage(tmp_path / "member_conflict", "w") as storage:
        data = np.ones((1, 2, 2), dtype=np.float32)
        if conflict == "subgroup":
            storage.root.create_group("BLOCK")
        elif conflict == "scalar":
            storage.root.create_array("BLOCK", data=np.array(1, dtype=np.float32))
            data = np.array([2], dtype=np.float32)
        else:
            existing = (np.ones((1, 3, 2), dtype=np.float32) if conflict == "shape"
                        else data.astype(np.float64) if conflict == "dtype" else data)
            storage.root.create_array("BLOCK", data=existing,
                                      chunks=(4 if conflict == "chunk" else 2,) + existing.shape[1:])
        before = _storage_state(storage)
        with pytest.raises(error):
            storage.write(data, "BLOCK", arrays_per_chunk=2)
        assert _storage_state(storage) == before


def test_write_batch_conversion_failure_leaves_storage_unchanged(tmp_path):
    class UnconvertibleBatch(list):
        def __array__(self, dtype=None, copy=None):
            raise ValueError("batch conversion failed")

    with sawnergy_util.ArrayStorage(tmp_path / "conversion", "w") as storage:
        before = _storage_state(storage)
        data = UnconvertibleBatch([np.ones((2, 2), dtype=np.float32)])
        with pytest.raises(ValueError, match="batch conversion failed"):
            storage.write(data, "BLOCK", arrays_per_chunk=2)
        assert _storage_state(storage) == before


@pytest.mark.parametrize("converted_shape", [(), (2, 2, 2)])
def test_write_inconsistent_batch_conversion_leaves_storage_unchanged(tmp_path, converted_shape):
    class InconsistentBatch(list):
        def __array__(self, dtype=None, copy=None):
            return np.ones(converted_shape, dtype=dtype)

    with sawnergy_util.ArrayStorage(tmp_path / "converted_shape", "w") as storage:
        before = _storage_state(storage)
        data = InconsistentBatch([np.ones((2, 2), dtype=np.float32)])
        with pytest.raises(ValueError):
            storage.write(data, "BLOCK", arrays_per_chunk=2)
        assert _storage_state(storage) == before


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_write_warning_error_leaves_storage_unchanged(tmp_path):
    with sawnergy_util.ArrayStorage(tmp_path / "warning", "w") as storage:
        before = _storage_state(storage)
        with pytest.raises(RuntimeWarning):
            storage.write(np.ones((1, 2, 2), dtype=np.float32), "BLOCK")
        assert _storage_state(storage) == before


@pytest.mark.parametrize("container", [list, tuple, np.asarray, iter])
def test_write_valid_append_preserves_values_and_metadata(tmp_path, container):
    first = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    second = np.full((1, 2, 2), 10, dtype=np.float32)
    with sawnergy_util.ArrayStorage(tmp_path / "valid_append", "w") as storage:
        storage.write(container(first), "BLOCK", arrays_per_chunk=2)
        before = json.dumps(dict(storage.root.attrs), sort_keys=True)
        storage.write(container(second), "BLOCK")
        np.testing.assert_array_equal(storage.read("BLOCK"), np.concatenate([first, second]))
        assert json.dumps(dict(storage.root.attrs), sort_keys=True) == before


@pytest.mark.parametrize("container", [list, tuple, np.asarray, iter])
def test_write_empty_batch_remains_noop(tmp_path, container):
    with sawnergy_util.ArrayStorage(tmp_path / "empty_batch", "w") as storage:
        before = _storage_state(storage)
        storage.write(container([]), "BLOCK", arrays_per_chunk=0)
        assert _storage_state(storage) == before


@pytest.fixture(params=["local", "zip"])
def unmarked_store(tmp_path, request):
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    path = tmp_path / ("unmarked.zip" if request.param == "zip" else "unmarked.zarr")
    store = (sawnergy_util.ZipStore(path, mode="w") if request.param == "zip"
             else sawnergy_util.LocalStore(path))
    with store:
        root = sawnergy_util.zarr.group(store=store)
        root.attrs["description"] = "external group"
        root.create_array("BLOCK", data=data)
    return path, data


def test_init_readonly_does_not_write_metadata(unmarked_store, monkeypatch):
    path, data = unmarked_store
    before = ({path: path.read_bytes()} if path.is_file() else
              {p: p.read_bytes() for p in path.rglob("*") if p.is_file()})

    def unexpected_write(*args, **kwargs):
        pytest.fail("Read-only construction must not attempt attribute writes")

    monkeypatch.setattr(sawnergy_util.zarr.core.attributes.Attributes, "__setitem__", unexpected_write)
    with sawnergy_util.ArrayStorage(path, "r") as storage:
        assert storage.store.read_only
        assert storage.root.store.read_only
        assert dict(storage.root.attrs) == {"description": "external group"}
        np.testing.assert_array_equal(storage.read("BLOCK"), data)
        for key in ("array_chunk_size_in_block", "array_shape_in_block", "array_dtype_in_block"):
            with pytest.raises(KeyError):
                storage.get_attr(key)

    after = ({path: path.read_bytes()} if path.is_file() else
             {p: p.read_bytes() for p in path.rglob("*") if p.is_file()})
    assert after == before


@pytest.mark.parametrize("mode", ["r", "r+"])
def test_init_missing_read_store_is_not_created(tmp_path, mode):
    path = tmp_path / "missing" / "archive.zarr"
    with pytest.raises(FileNotFoundError):
        sawnergy_util.ArrayStorage(path, mode)
    assert not path.parent.exists()


@pytest.mark.parametrize("mode", ["r", "r+"])
def test_init_missing_group_is_not_created(tmp_path, mode):
    path = tmp_path / "empty.zarr"
    path.mkdir()
    with pytest.raises(FileNotFoundError):
        sawnergy_util.ArrayStorage(path, mode)
    assert list(path.iterdir()) == []


@pytest.mark.parametrize("mode", ["a", "w", "w-"])
def test_init_creation_modes(tmp_path, mode):
    path = tmp_path / "created.zarr"
    with sawnergy_util.ArrayStorage(path, mode) as storage:
        assert storage.store_path == path
        assert isinstance(storage.root, sawnergy_util.zarr.Group)
        for key in ("array_chunk_size_in_block", "array_shape_in_block", "array_dtype_in_block"):
            assert storage.get_attr(key) == {}


@pytest.mark.parametrize("mode", ["a", "r+", "w", "w-"])
def test_init_existing_store_respects_mode(tmp_path, mode):
    path = tmp_path / "existing.zarr"
    data = np.array([[1, 2]], dtype=np.int16)
    with sawnergy_util.ArrayStorage(path, "w") as storage:
        storage.write(data, "BLOCK", arrays_per_chunk=1)
        storage.add_attr("description", "preserve")
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}

    if mode == "w-":
        with pytest.raises(FileExistsError):
            sawnergy_util.ArrayStorage(path, mode)
        assert {p: p.read_bytes() for p in path.rglob("*") if p.is_file()} == before
    else:
        with sawnergy_util.ArrayStorage(path, mode) as storage:
            if mode == "w":
                assert storage.list_blocks() == []
                with pytest.raises(KeyError):
                    storage.get_attr("description")
            else:
                np.testing.assert_array_equal(storage.read("BLOCK"), data)
                assert storage.get_attr("description") == "preserve"
                assert storage.get_attr("array_chunk_size_in_block") == {"BLOCK": 1}
                storage.write(data, "BLOCK")
                np.testing.assert_array_equal(storage.read("BLOCK"), np.concatenate([data, data]))


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_init_writable_initializes_missing_metadata(tmp_path, mode):
    path = tmp_path / "external.zarr"
    with sawnergy_util.LocalStore(path) as store:
        root = sawnergy_util.zarr.group(store=store)
        root.attrs["description"] = "preserve"
        root.attrs["array_chunk_size_in_block"] = {"BLOCK": 4}
    with sawnergy_util.ArrayStorage(path, mode) as storage:
        assert storage.get_attr("description") == "preserve"
        assert storage.get_attr("array_chunk_size_in_block") == {"BLOCK": 4}
        assert storage.get_attr("array_shape_in_block") == {}
        assert storage.get_attr("array_dtype_in_block") == {}


@pytest.mark.parametrize("mode", ["invalid", "", None])
def test_init_invalid_mode_does_not_create_store(tmp_path, mode):
    path = tmp_path / "invalid.zarr"
    with pytest.raises(ValueError):
        sawnergy_util.ArrayStorage(path, mode)
    assert not path.exists()


@pytest.mark.parametrize("mode", ["r", "r+", "a"])
def test_init_rejects_array_root_without_mutation(tmp_path, monkeypatch, mode):
    path = tmp_path / "array.zarr"
    with sawnergy_util.LocalStore(path) as store:
        sawnergy_util.zarr.create_array(store=store, data=np.arange(3))
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    closed = []
    close = sawnergy_util.LocalStore.close

    def record_close(self):
        closed.append(self.root)
        close(self)

    monkeypatch.setattr(sawnergy_util.LocalStore, "close", record_close)
    with pytest.raises(TypeError, match="must be a group"):
        sawnergy_util.ArrayStorage(path, mode)
    assert path in closed
    assert {p: p.read_bytes() for p in path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("stage,error_type", [("open", OSError), ("open", KeyboardInterrupt),
                                             ("attributes", OSError)])
@pytest.mark.parametrize("close_fails", [False, True])
def test_init_failure_closes_store_and_preserves_error(unmarked_store, monkeypatch, stage, error_type, close_fails):
    path, _ = unmarked_store
    error = error_type("initialization failed")
    backend = sawnergy_util.ZipStore if path.suffix == ".zip" else sawnergy_util.LocalStore
    mode = "r" if path.suffix == ".zip" else "a"
    closed = []
    close = backend.close

    def record_close(self):
        is_target = getattr(self, "path", getattr(self, "root", None)) == path
        if is_target:
            closed.append(self)
        close(self)
        if close_fails and is_target:
            raise OSError("closing failed")

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Initialization must not retry group creation")

    if stage == "open":
        open_store = sawnergy_util.zarr.open

        def fail_open(*args, **kwargs):
            open_store(*args, **kwargs)
            raise error

        monkeypatch.setattr(sawnergy_util.zarr, "open", fail_open)
    else:
        def fail_attributes(self):
            raise error

        monkeypatch.setattr(sawnergy_util.zarr.Group, "attrs", property(fail_attributes))

    monkeypatch.setattr(backend, "close", record_close)
    monkeypatch.setattr(sawnergy_util.zarr, "group", unexpected_fallback)
    with pytest.raises(error_type) as caught:
        sawnergy_util.ArrayStorage(path, mode)
    assert caught.value is error
    assert closed


def test_init_metadata_write_failure_closes_store(tmp_path, monkeypatch):
    path = tmp_path / "metadata.zarr"
    error = OSError("metadata write failed")
    closed = []
    close = sawnergy_util.LocalStore.close
    attributes = sawnergy_util.zarr.core.attributes.Attributes
    assign = attributes.__setitem__

    def fail_metadata(self, key, value):
        if key == "array_shape_in_block":
            raise error
        assign(self, key, value)

    def record_close(self):
        if self.root == path:
            closed.append(self)
        close(self)

    monkeypatch.setattr(attributes, "__setitem__", fail_metadata)
    monkeypatch.setattr(sawnergy_util.LocalStore, "close", record_close)
    with pytest.raises(OSError) as caught:
        sawnergy_util.ArrayStorage(path, "a")
    assert caught.value is error
    assert closed


@pytest.mark.parametrize("mode", ["w", "w-"])
def test_init_creation_failure_closes_store(tmp_path, monkeypatch, mode):
    path = tmp_path / "creation.zarr"
    error = OSError("group creation failed")
    closed = []
    open_group = sawnergy_util.zarr.open_group
    close = sawnergy_util.LocalStore.close

    def fail_creation(*args, **kwargs):
        open_group(*args, **kwargs)
        raise error

    def record_close(self):
        if self.root == path:
            closed.append(self)
        close(self)

    monkeypatch.setattr(sawnergy_util.zarr, "open_group", fail_creation)
    monkeypatch.setattr(sawnergy_util.LocalStore, "close", record_close)
    with pytest.raises(OSError) as caught:
        sawnergy_util.ArrayStorage(path, mode)
    assert caught.value is error
    assert closed


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


@pytest.mark.parametrize("destination", ["default", "file", "directory"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_compress_publishes_complete_archive(tmp_path, monkeypatch, destination, existing_output):
    output = tmp_path / "source.zip" if destination == "default" else tmp_path / "exports" / "source.zip"
    if existing_output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with sawnergy_util.ArrayStorage(tmp_path / "previous", "w") as previous:
            previous.write(np.array([[99]], dtype=np.int16), "OLD", arrays_per_chunk=1)
            previous.compress(output, compression_level=0)
    before = output.read_bytes() if existing_output else None
    replace = sawnergy_util.os.replace
    replacements = []
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    def publish(src, dst):
        src, dst = Path(src), Path(dst)
        assert src != dst and src.parent == output.parent
        assert dst == output
        assert (output.read_bytes() if output.exists() else None) == before
        # Opening a separate reader verifies the ZIP was finalized before publication.
        with sawnergy_util.ZipStore(src, mode="r") as archive:
            root = sawnergy_util.zarr.open_group(store=archive, mode="r")
            np.testing.assert_array_equal(root["BLOCK"][:], data)
            np.testing.assert_array_equal(root["OTHER"][:], np.array([[7]], dtype=np.int16))
            assert root["BLOCK"].dtype == data.dtype
            assert root["BLOCK"].chunks == (2, 2, 2)
            assert dict(root.attrs) == root_attrs
            assert dict(root["BLOCK"].attrs) == {"unit": "test unit"}
        replacements.append(src)
        replace(src, dst)

    with sawnergy_util.ArrayStorage(tmp_path / "source", "w") as storage:
        storage.write(data, "BLOCK", arrays_per_chunk=2)
        storage.write(np.array([[7]], dtype=np.int16), "OTHER", arrays_per_chunk=1)
        storage.add_attr("description", "replacement")
        storage.root["BLOCK"].attrs["unit"] = "test unit"
        root_attrs = dict(storage.root.attrs)
        monkeypatch.setattr(sawnergy_util.os, "replace", publish)
        into = None if destination == "default" else output if destination == "file" else output.parent
        assert storage.compress(into, compression_level=3) == str(output)
        np.testing.assert_array_equal(storage.read("BLOCK"), data)

    assert len(replacements) == 1
    assert not replacements[0].exists()
    assert {p for p in output.parent.iterdir() if p.is_file()} == {output}
    with sawnergy_util.ArrayStorage(output, "r") as archive:
        np.testing.assert_array_equal(archive.read("BLOCK"), data)
        assert archive.list_blocks() == ["BLOCK", "OTHER"]


@pytest.mark.parametrize("existing_output", [False, True])
@pytest.mark.parametrize("stage,error_type", [
    ("temporary file", OSError), ("metadata", OSError), ("copy", OSError),
    ("close", OSError), ("replace", OSError),
    ("copy", KeyboardInterrupt),
])
def test_compress_failure_preserves_destination(tmp_path, monkeypatch, existing_output, stage, error_type):
    output = tmp_path / "archive.zip"
    error = error_type("compression failed")
    with sawnergy_util.ArrayStorage(tmp_path / "source", "w") as storage:
        storage.write(np.arange(4, dtype=np.float32).reshape(2, 2), "BLOCK", arrays_per_chunk=1)
        if existing_output:
            storage.compress(output, compression_level=0)
        before = output.read_bytes() if existing_output else None
        entries = set(tmp_path.iterdir())

        if stage == "temporary file":
            def fail_temporary_file(*args, **kwargs):
                raise error

            monkeypatch.setattr(sawnergy_util.tempfile, "NamedTemporaryFile", fail_temporary_file)
        elif stage == "metadata":
            group = sawnergy_util.zarr.group

            def fail_metadata(*args, **kwargs):
                group(*args, **kwargs)
                raise error

            monkeypatch.setattr(sawnergy_util.zarr, "group", fail_metadata)
        elif stage == "copy":
            assign = sawnergy_util.zarr.Array.__setitem__

            def fail_copy(self, key, value):
                assign(self, key, value)
                raise error

            monkeypatch.setattr(sawnergy_util.zarr.Array, "__setitem__", fail_copy)
        elif stage == "close":
            close = sawnergy_util.ZipStore.close

            def fail_close(self):
                close(self)
                raise error

            monkeypatch.setattr(sawnergy_util.ZipStore, "close", fail_close)
        else:
            def fail_replace(src, dst):
                raise error

            monkeypatch.setattr(sawnergy_util.os, "replace", fail_replace)

        with pytest.raises(error_type) as caught:
            storage.compress(output, compression_level=3)

        assert caught.value is error
        assert (output.read_bytes() if output.exists() else None) == before
        assert set(tmp_path.iterdir()) == entries


def test_compress_cleanup_error_preserves_original_failure(tmp_path, monkeypatch, caplog):
    error = OSError("copy failed")
    output = tmp_path / "archive.zip"
    leftovers = []
    unlink = Path.unlink

    def fail_copy(self, key, value):
        raise error

    def fail_unlink(self, *args, **kwargs):
        leftovers.append(self)
        raise OSError("temporary file removal failed")

    with sawnergy_util.ArrayStorage(tmp_path / "source", "w") as storage:
        storage.write(np.array([[1]], dtype=np.int16), "BLOCK", arrays_per_chunk=1)
        with monkeypatch.context() as patch:
            patch.setattr(sawnergy_util.zarr.Array, "__setitem__", fail_copy)
            patch.setattr(Path, "unlink", fail_unlink)
            with pytest.raises(OSError) as caught:
                storage.compress(output, compression_level=0)
        assert caught.value is error
        assert not output.exists()
        assert "Failed to remove temporary archive" in caplog.text
        assert len(leftovers) == 1 and leftovers[0].is_file()
        unlink(leftovers[0])


def test_compress_zip_source_remains_noop(tmp_path):
    output = tmp_path / "archive.zip"
    with sawnergy_util.ArrayStorage(tmp_path / "source", "w") as storage:
        storage.write(np.array([[1]], dtype=np.int16), "BLOCK", arrays_per_chunk=1)
        storage.compress(output, compression_level=0)
    before = output.read_bytes()
    with sawnergy_util.ArrayStorage(output, "r") as archive:
        assert archive.compress(tmp_path / "other.zip", compression_level=-1) == str(output)
    assert output.read_bytes() == before
    assert not (tmp_path / "other.zip").exists()


@pytest.mark.parametrize("step", [0, -1, -4])
@pytest.mark.parametrize("shape", [(0, 2), (5, 2), ()])
def test_block_iter_rejects_nonpositive_steps(tmp_path, step, shape):
    with sawnergy_util.ArrayStorage(tmp_path / "invalid_step", mode="w") as st:
        st.root.create_array("BLOCK", data=np.zeros(shape, dtype=np.float32))
        with pytest.raises(ValueError, match="step must be positive"):
            list(st.block_iter("BLOCK", step=step))


@pytest.mark.parametrize("step", [1, 2, 5, 8])
def test_block_iter_preserves_order_shapes_and_detached_copies(tmp_path, step):
    data = np.arange(20, dtype=np.float32).reshape(5, 2, 2)
    with sawnergy_util.ArrayStorage(tmp_path / "chunks", mode="w") as st:
        st.write(data, to_block_named="BLOCK", arrays_per_chunk=2)
        chunks = list(st.block_iter("BLOCK", step=step))
        assert [chunk.shape for chunk in chunks] == [
            (min(step, len(data) - start), 2, 2) for start in range(0, len(data), step)
        ]
        assert all(chunk.dtype == data.dtype for chunk in chunks)
        np.testing.assert_array_equal(np.concatenate(chunks), data)
        chunks[0][...] = -1
        np.testing.assert_array_equal(st.read("BLOCK", slice(None)), data)


def test_block_iter_positive_step_on_empty_and_scalar_arrays(tmp_path):
    with sawnergy_util.ArrayStorage(tmp_path / "edge_shapes", mode="w") as st:
        st.root.create_array("EMPTY", data=np.zeros((0, 2), dtype=np.int32))
        st.root.create_array("SCALAR", data=np.array(7, dtype=np.int32))
        assert list(st.block_iter("EMPTY", step=2)) == []
        chunks = list(st.block_iter("SCALAR", step=2))
        assert len(chunks) == 1 and chunks[0].shape == ()
        assert chunks[0] == 7


@pytest.mark.parametrize("item_shape", [(0, 2), (2, 0), (2, 2), ()])
def test_write_and_append_preserve_item_shape_with_positive_chunks(tmp_path, item_shape):
    first = np.ones((2,) + item_shape, dtype=np.float32)
    second = np.full((1,) + item_shape, 2, dtype=np.float32)
    with sawnergy_util.ArrayStorage(tmp_path / "empty_dimensions", mode="w") as st:
        st.write(first, to_block_named="BLOCK", arrays_per_chunk=2)
        st.write(second, to_block_named="BLOCK")
        stored = st.read("BLOCK", slice(None))
        assert stored.shape == (3,) + item_shape
        assert stored.dtype == first.dtype
        np.testing.assert_array_equal(stored, np.concatenate([first, second]))
        assert st.root["BLOCK"].chunks == (2,) + tuple(max(1, dim) for dim in item_shape)
        assert tuple(st.root.attrs["array_shape_in_block"]["BLOCK"]) == item_shape
