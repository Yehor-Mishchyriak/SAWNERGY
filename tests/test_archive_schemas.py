from __future__ import annotations

import pytest

import sawnergy
from sawnergy.sawnergy_util import ArrayStorage


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
