import importlib.util
import sys
from types import ModuleType
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def ensure_package(fullname: str) -> ModuleType:
    """
    Ensure a (possibly nested) package object exists in sys.modules.
    Example: ensure_package("sawnergy.rin") creates 'sawnergy' and 'sawnergy.rin'.
    """
    parts = fullname.split(".")
    cur = ""
    pkg = None
    for p in parts:
        cur = p if not cur else f"{cur}.{p}"
        if cur not in sys.modules:
            m = ModuleType(cur)
            m.__path__ = [str(PROJECT_ROOT)]
            sys.modules[cur] = m
        pkg = sys.modules[cur]
    return pkg

def load_module(fullname: str, filename: str) -> ModuleType:
    """
    Load a module from a file on disk under a desired 'fullname'.
    """
    path = (PROJECT_ROOT / filename).resolve()
    spec = importlib.util.spec_from_file_location(fullname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {fullname} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def load_core_sawnergy():
    """
    Create pseudo-package tree and load core modules so relative imports resolve.
    """
    ensure_package("sawnergy")
    ensure_package("sawnergy.rin")
    ensure_package("sawnergy.visualizer")

    load_module("sawnergy.sawnergy_util", "sawnergy_util.py")
    load_module("sawnergy.rin.rin_util", "rin_util.py")
    load_module("sawnergy.visualizer.visualizer_util", "visualizer_util.py")
