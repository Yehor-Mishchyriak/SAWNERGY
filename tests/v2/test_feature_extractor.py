import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sawnergy.v2.traj_features.feature_extractor import FeatureExtractor, feature


@pytest.mark.parametrize("kind", ["pairwise", "global"])
@pytest.mark.parametrize("scope", ["window", "trajectory"])
def test_feature_marks_original_method_without_executing_it(kind, scope):
    def method(self, ctx):
        raise AssertionError("Decoration must not run the method")

    signature = inspect.signature(method)
    decorated = feature(kind=kind, scope=scope)(method)

    assert decorated is method
    assert decorated.feature_kind == kind
    assert decorated.feature_scope == scope
    assert inspect.signature(decorated) == signature


def test_feature_preserves_instance_binding_and_return_value():
    class Extractor(FeatureExtractor):
        @feature(kind="global", scope="window")
        def values(self, ctx):
            return self, ctx

    first, second = Extractor(), Extractor()
    ctx = object()
    assert first.values(ctx) == (first, ctx)
    assert second.values(ctx) == (second, ctx)
    assert first.values.feature_kind == "global"
    assert first.values.feature_scope == "window"


def test_feature_preserves_method_exception():
    error = RuntimeError("extraction failed")

    class Extractor(FeatureExtractor):
        @feature(kind="pairwise", scope="trajectory")
        def values(self, ctx):
            raise error

    with pytest.raises(RuntimeError) as caught:
        Extractor().values(object())
    assert caught.value is error


@pytest.mark.parametrize("parameter", ["kind", "scope"])
@pytest.mark.parametrize("invalid", ["unknown", "", None, 1, [], {}])
def test_feature_rejects_invalid_metadata(parameter, invalid):
    arguments = {"kind": "pairwise", "scope": "window", parameter: invalid}
    with pytest.raises(ValueError):
        feature(**arguments)


def test_feature_metadata_validation_remains_active_under_optimization():
    source = """
from sawnergy.v2.traj_features.feature_extractor import feature
for arguments in ({'kind': 'invalid', 'scope': 'window'},
                  {'kind': 'global', 'scope': 'invalid'}):
    try:
        feature(**arguments)
    except ValueError:
        continue
    raise RuntimeError('Invalid feature metadata was accepted')
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run([sys.executable, "-O", "-c", source], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
