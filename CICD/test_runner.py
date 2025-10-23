from __future__ import annotations
from pathlib import Path
import sys

import pytest


def run_all_tests() -> int:
    """Invoke pytest on the repository's ``tests`` package."""
    repo_root = Path(__file__).resolve().parent.parent
    tests_dir = repo_root / "tests"

    if not tests_dir.exists():
        raise FileNotFoundError(f"Tests directory not found: {tests_dir}")

    # Ensure local package imports pick up the repo root first
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return pytest.main([str(tests_dir)])


if __name__ == "__main__":
    sys.exit(run_all_tests())
