# CICD/build_whl.py
from __future__ import annotations
from setuptools import setup, find_packages
from pathlib import Path
from decimal import Decimal, InvalidOperation
import json

ROOT = Path(__file__).resolve().parent.parent

def read_readme():
    for name in ("README.md", "README.MD"):
        p = ROOT / name
        if p.exists():
            return p.read_text(encoding="utf-8")
    return ""

def _bump_version_str(v: str) -> str:
    """
    Keep current behavior (add +0.1) but avoid float rounding.
    Always format with one decimal place (e.g., 0.2 -> 0.3, 1.0 -> 1.1).
    If the value isn't a simple decimal, bump the last dot-part with base-10 carry
    (e.g., 1.2.9 -> 1.3.0).
    """
    try:
        return str((Decimal(v) + Decimal("0.1")).quantize(Decimal("0.1")))
    except (InvalidOperation, ValueError):
        parts = v.split(".")
        if parts and all(p.isdigit() for p in parts):
            nums = [int(p) for p in parts]
            nums[-1] += 1
            # propagate base-10 carry from right to left
            for i in range(len(nums) - 1, 0, -1):
                if nums[i] >= 10:
                    nums[i] = 0
                    nums[i - 1] += 1
            return ".".join(str(n) for n in nums)
        # last resort: return original to avoid breaking build
        return v

def load_cfg():
    cfg_path = ROOT / "pkg_meta.json"
    raw: dict = json.loads(cfg_path.read_text(encoding="utf-8"))

    old_version = str(raw["version"])
    new_version = _bump_version_str(old_version)

    # persist bumped version back to pkg_meta.json
    persisted = dict(raw)
    persisted["version"] = new_version
    cfg_path.write_text(json.dumps(persisted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # runtime cfg mirrors persisted, with possible README substitution
    data = dict(raw)
    data["version"] = new_version

    if data.get("long_description") == "[PLACE_HOLDER]":
        data["long_description"] = read_readme()

    data.setdefault("long_description_content_type", "text/markdown")

    pkgs_cfg = (data.pop("packages", {}) or {}).get("find", {})
    return data, pkgs_cfg

if __name__ == "__main__":
    cfg, find_kwargs = load_cfg()
    setup(
        packages=find_packages(**find_kwargs),
        **cfg
    )