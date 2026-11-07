from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import sawnergy
from sawnergy.sawnergy_util import ArrayStorage
from sawnergy.walks import Walker
from sawnergy.embedding import Embedder, Visualizer as EmbeddingVisualizer
from sawnergy.visual import Visualizer as RINVisualizer
from sawnergy.visual import visualizer_util


@pytest.mark.parametrize("fixture_name,schema_name", [
    ("rin_archive_path", "rin"),
    ("rin_archive_path_both", "rin"),
    ("walks_archive_path", "rin_walks"),
    ("walks_archive_path_full", "rin_walks"),
    ("timeaware_walks_archive_path", "rin_walks"),
    ("embeddings_archive_path", "shallow_residue_embeddings"),
])
def test_writers_persist_own_schema_and_package_version(request, fixture_name, schema_name):
    path = request.getfixturevalue(fixture_name)
    with ArrayStorage(path, "r") as storage:
        producer_version = storage.get_attr("producer_version")
        assert isinstance(producer_version, str)
        assert producer_version == sawnergy.__version__
        assert storage.get_attr("schema_name") == schema_name
        schema_version = storage.get_attr("schema_version")
        assert type(schema_version) is int
        assert schema_version == 1


def test_generic_storage_does_not_add_semantic_schema_markers(tmp_path):
    with ArrayStorage(tmp_path / "generic", "w") as storage:
        assert {"producer_version", "schema_name", "schema_version"}.isdisjoint(storage.root.attrs)


READERS = [(Walker, "rin"), (Embedder, "rin_walks"),
           (RINVisualizer, "rin"), (EmbeddingVisualizer, "shallow_residue_embeddings")]
INVALID_MARKERS = [
    ({}, ("producer_version", "schema_name", "schema_version"), "legacy"),
    ({}, ("producer_version",), "producer_version"),
    ({}, ("schema_name",), "schema_name"),
    ({}, ("schema_version",), "schema_version"),
    ({"producer_version": None}, (), "producer_version"),
    ({"producer_version": 1}, (), "producer_version"),
    ({"schema_name": None}, (), "schema_name"),
    ({"schema_name": 1}, (), "schema_name"),
    ({"schema_name": "unknown_schema"}, (), "schema_name"),
    ({"schema_version": None}, (), "schema_version"),
    ({"schema_version": "1"}, (), "schema_version"),
    ({"schema_version": 1.0}, (), "schema_version"),
    ({"schema_version": True}, (), "schema_version"),
    ({"schema_version": False}, (), "schema_version"),
    ({"schema_version": 0}, (), "schema_version"),
    ({"schema_version": -1}, (), "schema_version"),
    ({"schema_version": 2}, (), "schema_version"),
]


@pytest.fixture(params=READERS, ids=["walker", "embedder", "rin_visualizer", "embedding_visualizer"])
def reader_case(request, monkeypatch, patched_shared_ndarray):
    # Exercise the real constructors; existing visualizer fixtures replace them.
    monkeypatch.setattr(visualizer_util, "warm_start_matplotlib", lambda: None)
    monkeypatch.setattr(RINVisualizer, "no_instances", False)
    monkeypatch.setattr(EmbeddingVisualizer, "no_instances", False)
    return request.param


def _reader_archive(tmp_path, schema, markers, zipped):
    path = tmp_path / "input.zarr"
    with ArrayStorage(path, "w") as storage:
        if schema == "rin":
            attrs = dict(com_name="COM", attractive_transitions_name="MATRIX",
                         repulsive_transitions_name=None, attractive_energies_name="MATRIX",
                         repulsive_energies_name=None)
            arrays = {"COM": np.arange(6, dtype=np.float32).reshape(1, 2, 3),
                      "MATRIX": np.array([[[0, 1], [1, 0]]], dtype=np.float32)}
        elif schema == "rin_walks":
            attrs = dict(attractive_RWs_name="WALKS", repulsive_RWs_name=None,
                         attractive_SAWs_name=None, repulsive_SAWs_name=None,
                         num_RWs=1, num_SAWs=0, node_count=2, time_stamp_count=1, walk_length=1)
            arrays = {"WALKS": np.array([[[1, 2], [2, 1]]], dtype=np.uint16)}
        else:
            attrs = dict(frame_embeddings_name="EMBEDDINGS")
            arrays = {"EMBEDDINGS": np.arange(6, dtype=np.float32).reshape(1, 2, 3)}
        for name, data in arrays.items():
            storage.write(data, name, arrays_per_chunk=1)
        for key, value in {**attrs, **markers}.items():
            storage.add_attr(key, value)
        if zipped:
            path = Path(storage.compress(tmp_path / "input.zip", compression_level=0))
    return path


def _archive_bytes(path):
    return (path.read_bytes() if path.is_file() else
            {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()})


@pytest.mark.parametrize("zipped", [False, True], ids=["local", "zip"])
@pytest.mark.parametrize("changes,missing,diagnostic", INVALID_MARKERS)
def test_readers_reject_incompatible_markers_before_dataset_access(
    tmp_path, monkeypatch, reader_case, patched_shared_ndarray, zipped, changes, missing, diagnostic,
):
    reader, schema = reader_case
    markers = dict(producer_version="older-producer", schema_name=schema, schema_version=1)
    markers.update(changes)
    for key in missing:
        del markers[key]
    path = _reader_archive(tmp_path, schema, markers, zipped)
    before = _archive_bytes(path)
    get_attr, exit_storage = ArrayStorage.get_attr, ArrayStorage.__exit__
    exited = []

    def guarded_get_attr(storage, key):
        if key not in ("producer_version", "schema_name", "schema_version"):
            pytest.fail("Dataset metadata was accessed before rejecting the archive")
        return get_attr(storage, key)

    def unexpected_read(*args, **kwargs):
        pytest.fail("Dataset values were accessed before rejecting the archive")

    def record_exit(storage, *args):
        exited.append(storage.store_path)
        return exit_storage(storage, *args)

    monkeypatch.setattr(ArrayStorage, "get_attr", guarded_get_attr)
    monkeypatch.setattr(ArrayStorage, "read", unexpected_read)
    monkeypatch.setattr(ArrayStorage, "__exit__", record_exit)
    monkeypatch.setattr(patched_shared_ndarray, "create", unexpected_read)
    with pytest.raises(ValueError) as caught:
        reader(path)
    assert diagnostic in str(caught.value).lower()
    for key in missing:
        assert key in str(caught.value)
    if (not missing and isinstance(markers["schema_name"], str)
            and type(markers["schema_version"]) is int
            and (markers["schema_name"] != schema or markers["schema_version"] != 1)):
        assert schema in str(caught.value) and "1" in str(caught.value)
        assert repr(markers["schema_name"]) in str(caught.value)
        assert repr(markers["schema_version"]) in str(caught.value)
    assert exited == [path.resolve()]
    assert _archive_bytes(path) == before


@pytest.mark.parametrize("wrong_case", [False, True], ids=["other_known_schema", "uppercase_schema"])
def test_readers_match_their_specific_schema_exactly(tmp_path, monkeypatch, reader_case, wrong_case):
    reader, schema = reader_case
    other = schema.upper() if wrong_case else "rin_walks" if schema == "rin" else "rin"
    markers = dict(producer_version="other", schema_name=other, schema_version=1)
    path = _reader_archive(tmp_path, schema, markers, zipped=False)

    def unexpected_read(*args, **kwargs):
        pytest.fail("Wrong-schema dataset was loaded")

    monkeypatch.setattr(ArrayStorage, "read", unexpected_read)
    with pytest.raises(ValueError) as caught:
        reader(path)
    assert repr(schema) in str(caught.value) and repr(other) in str(caught.value)


@pytest.mark.parametrize("zipped", [False, True], ids=["local", "zip"])
@pytest.mark.parametrize("producer", ["0.0.0", "future-development-build"])
def test_readers_accept_supported_schema_from_other_producers(tmp_path, reader_case, zipped, producer):
    import matplotlib.pyplot as plt

    reader, schema = reader_case
    markers = dict(producer_version=producer, schema_name=schema, schema_version=1)
    path = _reader_archive(tmp_path, schema, markers, zipped)
    before = _archive_bytes(path)
    instance = reader(path)
    try:
        if reader is Walker:
            np.testing.assert_array_equal(instance.attr_matrices.array, [[[0, 1], [1, 0]]])
        elif reader is Embedder:
            np.testing.assert_array_equal(instance.attractive_RWs, [[[1, 2], [2, 1]]])
        else:
            values = instance.COM_coords if reader is RINVisualizer else instance.E
            np.testing.assert_array_equal(values, np.arange(6).reshape(1, 2, 3))
        assert _archive_bytes(path) == before
    finally:
        if reader is Walker:
            instance.close()
        elif reader in (RINVisualizer, EmbeddingVisualizer):
            plt.close(instance._fig)


def test_reader_schema_checks_remain_active_under_optimization():
    # Zarr 3.1.2 cannot reopen valid groups under -O. Isolate its metadata loader
    # while executing the real constructors and compatibility gate in a subprocess.
    script = textwrap.dedent('''
        from types import SimpleNamespace
        from tests.test_archive_schemas import READERS, INVALID_MARKERS
        from sawnergy import sawnergy_util
        from sawnergy.visual import visualizer_util

        class DatasetAccessed(RuntimeError):
            pass

        class Storage:
            def __init__(self, *args, **kwargs):
                self.root = SimpleNamespace(attrs=markers)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                exits.append(True)
            def get_attr(self, key):
                if key in ("producer_version", "schema_name", "schema_version"):
                    return markers[key]
                raise DatasetAccessed()
            def read(self, *args):
                raise DatasetAccessed()

        sawnergy_util.ArrayStorage = Storage
        visualizer_util.warm_start_matplotlib = lambda: None
        for reader, schema in READERS:
            for changes, missing, diagnostic in INVALID_MARKERS:
                markers = dict(producer_version="other", schema_name=schema, schema_version=1)
                markers.update(changes)
                for key in missing:
                    del markers[key]
                exits = []
                try:
                    reader("input.zip")
                except ValueError as error:
                    if diagnostic not in str(error).lower() or exits != [True]:
                        raise RuntimeError("Invalid diagnostic or archive not closed")
                else:
                    raise RuntimeError("Incompatible archive accepted under -O")
            markers = dict(producer_version="other", schema_name=schema, schema_version=1)
            exits = []
            try:
                reader("input.zip")
            except DatasetAccessed:
                if exits != [True]:
                    raise RuntimeError("Archive not closed at dataset boundary")
            else:
                raise RuntimeError("Supported schema did not reach dataset loading")
    ''')
    result = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
