from __future__ import annotations

import json
import os
import subprocess


DOWNSTREAM = {"rin", "walks", "embedding", "visual", "embedding_visual", "schema", "api"}
SOURCE_GROUPS = {
    "src/sawnergy/sawnergy_util.py": DOWNSTREAM | {"storage", "utils"},
    "src/sawnergy/_archive_schema.py": DOWNSTREAM,
    "src/sawnergy/rin/": DOWNSTREAM,
    "src/sawnergy/walks/": {"walks", "embedding", "embedding_visual", "schema", "api"},
    "src/sawnergy/embedding/embedder.py": {"embedding", "embedding_visual", "schema", "api", "pureml", "torch"},
    "src/sawnergy/embedding/SGNS_torch.py": {"torch", "api"},
    "src/sawnergy/embedding/SGNS_pml.py": {"pureml", "api"},
    "src/sawnergy/embedding/visualizer.py": {"embedding_visual", "schema", "api"},
    "src/sawnergy/embedding/__init__.py": {"embedding", "embedding_visual", "schema", "api", "pureml", "torch"},
    "src/sawnergy/visual/visualizer.py": {"visual", "schema", "api"},
    "src/sawnergy/visual/": {"visual", "embedding_visual", "schema", "api"},
    "src/sawnergy/logging_util.py": {"api"},
}
TEST_GROUPS = {
    "tests/test_storage.py": "storage",
    "tests/test_sawnergy_util.py": "utils",
    "tests/test_rin.py": "rin",
    "tests/test_walks.py": "walks",
    "tests/test_embedding.py": "embedding",
    "tests/test_visual.py": "visual",
    "tests/test_embedding_visualizer.py": "embedding_visual",
    "tests/test_public_api.py": "api",
    "tests/test_archive_schemas.py": "schema",
    "tests/test_ci_selection.py": "ci",
}
GLOBAL_FILES = {
    "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "pytest.ini", "tox.ini",
    "conftest.py", "tests/conftest.py", "tests/__init__.py", "src/sawnergy/__init__.py",
}


def select_groups(paths: list[str] | None, *, full: bool = False) -> set[str]:
    """Select affected development tests; unknown code/history requests a full run."""
    if full or paths is None:
        return {"all"}
    groups = set()
    for path in paths:
        if path in GLOBAL_FILES or path.startswith((".github/workflows/", "CICD/")):
            return {"all"}
        if path in TEST_GROUPS:
            groups.add(TEST_GROUPS[path])
            if path == "tests/test_embedding.py":
                groups.update({"pureml", "torch"})
            elif path == "tests/test_archive_schemas.py":
                groups.update(DOWNSTREAM | {"storage"})
            continue
        for source, affected in SOURCE_GROUPS.items():
            if path == source or (source.endswith("/") and path.startswith(source)):
                groups.update(affected)
                break
        else:
            if path.startswith(("src/", "tests/")):
                return {"all"}
    return groups


def group_for_test(path: str, params: dict, *, name: str = "", torch: bool = False, pureml: bool = False) -> str:
    if torch:
        return "torch"
    if pureml:
        return "pureml"
    if path == "tests/test_archive_schemas.py":
        if name == "test_generic_storage_does_not_add_semantic_schema_markers":
            return "storage"
        if "reader_case" in params:
            module = params["reader_case"][0].__module__
            return {"sawnergy.walks.walker": "walks", "sawnergy.embedding.embedder": "embedding",
                    "sawnergy.visual.visualizer": "visual",
                    "sawnergy.embedding.visualizer": "embedding_visual"}.get(module, "always")
        fixture = params.get("fixture_name", "")
        if "walks" in fixture:
            return "walks"
        if fixture.startswith("rin_"):
            return "rin"
        if fixture.startswith("embeddings_"):
            return "embedding"
    return TEST_GROUPS.get(path, "always")


def workflow_outputs(groups: set[str]) -> dict[str, str]:
    full = "all" in groups
    return {
        "groups": json.dumps(sorted(groups)),
        "run_tests": str(bool(groups - {"torch"})).lower(),
        "run_torch": str(full or "torch" in groups).lower(),
        "full": str(full).lower(),
        "markexpr": "" if full else "not torch",
    }


def changed_files(base: str, head: str, *, pull_request: bool = False) -> list[str] | None:
    if not base or not base.strip("0"):
        return None
    revisions = [f"{base}...{head}"] if pull_request else [base, head]
    try:
        data = subprocess.check_output(["git", "diff", "--name-only", "--no-renames", "-z", *revisions])
    except (OSError, subprocess.CalledProcessError):
        return None
    return [os.fsdecode(path) for path in data.split(b"\0") if path]


if __name__ == "__main__":
    full = os.environ.get("SAWNERGY_FULL_SUITE") == "true"
    paths = [] if full else changed_files(
        os.environ.get("SAWNERGY_BASE_SHA", ""), os.environ.get("SAWNERGY_HEAD_SHA", "HEAD"),
        pull_request=os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
    )
    outputs = workflow_outputs(select_groups(paths, full=full))
    text = "".join(f"{key}={value}\n" for key, value in outputs.items())
    print(text, end="")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(text)
