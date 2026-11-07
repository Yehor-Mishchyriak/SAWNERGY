from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest


def test_ci_selection_main_always_runs_everything():
    from CICD.select_tests import select_groups, workflow_outputs

    for paths in ([], ["README.md"], ["src/sawnergy/visual/visualizer.py"]):
        assert select_groups(paths, full=True) == {"all"}
        outputs = workflow_outputs(select_groups(paths, full=True))
        assert json.loads(outputs["groups"]) == ["all"]
        assert outputs["run_tests"] == outputs["run_torch"] == "true"
        assert outputs["markexpr"] == ""


@pytest.mark.parametrize("path,expected", [
    ("src/sawnergy/visual/visualizer.py", {"visual", "schema", "api"}),
    ("src/sawnergy/embedding/visualizer.py", {"embedding_visual", "schema", "api"}),
    ("src/sawnergy/visual/visualizer_util.py", {"visual", "embedding_visual", "schema", "api"}),
    ("src/sawnergy/embedding/SGNS_torch.py", {"torch", "api"}),
    ("src/sawnergy/embedding/SGNS_pml.py", {"pureml", "api"}),
    ("tests/test_storage.py", {"storage"}),
    ("tests/test_embedding.py", {"embedding", "pureml", "torch"}),
    ("README.md", set()),
])
def test_ci_selection_limits_unrelated_tests(path, expected):
    from CICD.select_tests import select_groups

    assert select_groups([path]) == expected


@pytest.mark.parametrize("path,required,excluded", [
    ("src/sawnergy/sawnergy_util.py", {"storage", "utils", "rin", "walks", "embedding", "visual", "embedding_visual", "schema", "api"}, {"torch", "pureml"}),
    ("src/sawnergy/rin/rin_util.py", {"rin", "walks", "embedding", "visual", "embedding_visual", "schema", "api"}, {"torch", "pureml"}),
    ("src/sawnergy/walks/walker.py", {"walks", "embedding", "embedding_visual", "schema", "api"}, {"torch", "pureml", "visual"}),
    ("src/sawnergy/embedding/embedder.py", {"embedding", "embedding_visual", "schema", "api", "torch", "pureml"}, {"visual"}),
    ("src/sawnergy/_archive_schema.py", {"walks", "embedding", "visual", "embedding_visual", "schema", "api"}, {"torch", "pureml"}),
])
def test_ci_selection_includes_downstream_dependencies(path, required, excluded):
    from CICD.select_tests import select_groups

    groups = select_groups([path])
    assert required <= groups
    assert groups.isdisjoint(excluded)


@pytest.mark.parametrize("path", [
    "pyproject.toml", "requirements.txt", "tests/conftest.py", "src/sawnergy/__init__.py",
    ".github/workflows/test.yml", "CICD/select_tests.py", "src/sawnergy/new_component.py",
    "tests/test_new_component.py",
])
def test_ci_selection_falls_back_to_full_suite(path):
    from CICD.select_tests import select_groups

    assert select_groups([path]) == {"all"}


def test_ci_selection_unions_changes_and_skips_docs_only():
    from CICD.select_tests import select_groups, workflow_outputs

    assert select_groups(["tests/test_storage.py", "tests/test_rin.py"]) == {"storage", "rin"}
    assert select_groups(["docs/specifications.md", "CONTRIBUTING.md"]) == set()
    assert workflow_outputs(set())["run_tests"] == "false"
    assert workflow_outputs(set())["run_torch"] == "false"
    assert workflow_outputs({"storage"})["run_torch"] == "false"
    assert workflow_outputs({"torch", "api"})["run_torch"] == "true"
    assert workflow_outputs({"storage"})["markexpr"] == "not torch"


@pytest.mark.parametrize("module,expected", [
    ("sawnergy.walks.walker", "walks"), ("sawnergy.embedding.embedder", "embedding"),
    ("sawnergy.visual.visualizer", "visual"), ("sawnergy.embedding.visualizer", "embedding_visual"),
])
def test_ci_selection_classifies_mixed_reader_cases(module, expected):
    from CICD.select_tests import group_for_test

    reader = SimpleNamespace(__module__=module)
    assert group_for_test("tests/test_archive_schemas.py", {"reader_case": (reader, "schema")}) == expected


@pytest.mark.parametrize("fixture,expected", [
    ("rin_archive_path_both", "rin"), ("walks_archive_path", "walks"),
    ("timeaware_walks_archive_path", "walks"), ("embeddings_archive_path", "embedding"),
])
def test_ci_selection_classifies_mixed_writer_cases(fixture, expected):
    from CICD.select_tests import group_for_test

    assert group_for_test("tests/test_archive_schemas.py", {"fixture_name": fixture}) == expected


def test_ci_selection_separates_backends_and_keeps_unknown_tests():
    from CICD.select_tests import group_for_test

    assert group_for_test("tests/test_embedding.py", {}, torch=True) == "torch"
    assert group_for_test("tests/test_embedding.py", {}, pureml=True) == "pureml"
    assert group_for_test("tests/test_embedding.py", {}) == "embedding"
    assert group_for_test("tests/test_archive_schemas.py", {}) == "schema"
    assert group_for_test("tests/test_archive_schemas.py", {},
                          name="test_generic_storage_does_not_add_semantic_schema_markers") == "storage"
    assert group_for_test("tests/new/test_unknown.py", {}) == "always"


@pytest.mark.parametrize("pull_request", [False, True])
def test_ci_diff_uses_push_endpoints_or_pr_merge_base(monkeypatch, pull_request):
    from CICD.select_tests import changed_files

    calls = []

    def git_diff(command):
        calls.append(command)
        return b"src/sawnergy/old.py\0src/sawnergy/new.py\0"

    monkeypatch.setattr(subprocess, "check_output", git_diff)
    assert changed_files("base", "head", pull_request=pull_request) == [
        "src/sawnergy/old.py", "src/sawnergy/new.py",
    ]
    assert "--no-renames" in calls[0]
    assert ("base...head" in calls[0]) is pull_request
    if not pull_request:
        assert calls[0][-2:] == ["base", "head"]


def test_ci_missing_history_triggers_full_selection(monkeypatch):
    from CICD.select_tests import changed_files, select_groups

    def unavailable(command):
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(subprocess, "check_output", unavailable)
    assert changed_files("base", "head") is None
    assert changed_files("0" * 40, "head") is None
    assert changed_files("", "head") is None
    assert select_groups(None) == {"all"}
