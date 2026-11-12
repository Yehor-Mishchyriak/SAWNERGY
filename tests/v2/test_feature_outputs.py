from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sawnergy.v2.traj_features.feature_extractor import ExtractionContext, FeatureExtractor, feature


@pytest.fixture
def ctx(tmp_path):
    return ExtractionContext(cpptraj=tmp_path / "cpptraj", topology_file=tmp_path / "topology",
                             trajectory_file=tmp_path / "trajectory", selection_mask=":1,3",
                             residue_ids=((1, 1), (2, 3)), atom_ids=((1, 2), (5,)),
                             frame_ids=(1, 3, 5), windows=((1, 3), (5,)))


def test_feature_snapshots_output_declarations():
    def method(self, ctx):
        raise AssertionError("Decoration executed the feature")

    decorated = feature(kind="pairwise", scope="window", shape=("N", "N"),
                        dtype=np.float32, units="angstrom", statistic="mean per-frame distance")(method)
    assert decorated is method
    assert decorated.feature_shape == ("N", "N")
    assert decorated.feature_dtype == np.dtype("float32")
    assert isinstance(decorated.feature_dtype, np.dtype)
    assert decorated.feature_units == "angstrom"
    assert decorated.feature_statistic == "mean per-frame distance"


@pytest.mark.parametrize("missing", ["shape", "dtype", "units", "statistic"])
def test_output_declarations_are_required(missing):
    arguments = dict(kind="global", scope="trajectory", shape=(), dtype="float64",
                     units="1", statistic="mean")
    del arguments[missing]
    with pytest.raises(TypeError):
        feature(**arguments)


@pytest.mark.parametrize("kind,shape", [
    ("pairwise", ()), ("pairwise", (2, 2)), ("pairwise", ("N",)),
    ("pairwise", ["N", "N"]), ("global", ("N",)), ("global", (True,)),
    ("global", (-1,)), ("global", (1.5,)), ("global", [2]), ("global", None),
])
def test_feature_rejects_invalid_shapes(kind, shape):
    with pytest.raises(ValueError):
        feature(kind=kind, scope="window", shape=shape, dtype="float64", units="1", statistic="mean")


@pytest.mark.parametrize("dtype", [None, "bad-dtype", object, complex, "U4", "datetime64[ns]",
                                    [("field", "f8", -1)]])
def test_feature_rejects_unsupported_dtypes(dtype):
    with pytest.raises(TypeError):
        feature(kind="global", scope="trajectory", shape=(), dtype=dtype, units="1", statistic="mean")


@pytest.mark.parametrize("field", ["units", "statistic"])
@pytest.mark.parametrize("value", ["", "  ", None, 1])
def test_feature_rejects_blank_or_nonstring_descriptions(field, value):
    arguments = dict(kind="global", scope="trajectory", shape=(), dtype="float64",
                     units="1", statistic="mean")
    arguments[field] = value
    with pytest.raises(ValueError):
        feature(**arguments)


@pytest.mark.parametrize("kind,scope,shape,output_shape", [
    ("pairwise", "window", ("N", "N"), (2, 2, 2)),
    ("pairwise", "trajectory", ("N", "N"), (2, 2)),
    ("global", "window", (3,), (2, 3)),
    ("global", "window", (), (2,)),
    ("global", "trajectory", (), ()),
    ("global", "trajectory", (2, 3), (2, 3)),
    ("global", "window", (0,), (2, 0)),
    ("global", "trajectory", (0,), (0,)),
])
@pytest.mark.parametrize("dtype", [bool, "int32", "uint16", "float32", "float64"])
def test_output_validation_accepts_all_kinds_scopes_and_real_dtypes(ctx, kind, scope, shape, output_shape, dtype):
    values = np.ones(output_shape, dtype=dtype)
    values.setflags(write=False)

    class Extractor(FeatureExtractor):
        @feature(kind=kind, scope=scope, shape=shape, dtype=dtype, units="1", statistic="mean")
        def values_for_call(self, ctx):
            return values

    extractor = Extractor()
    actual = extractor.values_for_call(ctx)
    assert actual is values
    assert extractor._validate_output(extractor.values_for_call, actual, ctx) is None
    np.testing.assert_array_equal(actual, np.ones(output_shape, dtype=dtype))
    assert not actual.flags.writeable


@pytest.mark.parametrize("values,error", [
    (None, TypeError), ([[[1.0]]], TypeError), (1.0, TypeError),
    (np.ones((2, 2, 2), dtype="float32"), TypeError),
    (np.ones((2, 2, 2), dtype=np.dtype("float64").newbyteorder("S")), TypeError),
    (np.ones((2, 2, 2), dtype=object), TypeError),
    (np.ma.array(np.ones((2, 2, 2)), mask=False), TypeError),
    (np.ma.array(np.ones((2, 2, 2)), mask=True), TypeError),
    (np.full((2, 2, 2), np.nan), ValueError),
    (np.full((2, 2, 2), np.inf), ValueError),
    (np.full((2, 2, 2), -np.inf), ValueError),
    (np.ones((2, 2)), ValueError), (np.ones((1, 2, 2)), ValueError),
    (np.ones((3, 2, 2)), ValueError), (np.ones((2, 1, 2)), ValueError),
    (np.ones((2, 2, 3)), ValueError), (np.ones((0, 2, 2)), ValueError),
])
def test_output_validation_rejects_invalid_arrays(ctx, values, error):
    class Extractor(FeatureExtractor):
        @feature(kind="pairwise", scope="window", shape=("N", "N"),
                 dtype="float64", units="angstrom", statistic="mean per-frame distance")
        def distances(self, ctx):
            return values

    extractor = Extractor()
    with pytest.raises(error):
        extractor._validate_output(extractor.distances, extractor.distances(ctx), ctx)


def test_window_feature_requires_windows(ctx):
    class Extractor(FeatureExtractor):
        @feature(kind="global", scope="window", shape=(), dtype="float64", units="1", statistic="mean")
        def values(self, ctx):
            return np.ones(0)

    extractor = Extractor()
    with pytest.raises(ValueError, match="window"):
        extractor._validate_output(extractor.values, extractor.values(ctx), replace(ctx, windows=()))


def test_validation_preserves_noncontiguous_output(ctx):
    class Extractor(FeatureExtractor):
        @feature(kind="global", scope="window", shape=(2, 3), dtype="float64", units="1", statistic="mean")
        def values(self, ctx):
            return np.arange(24, dtype="float64").reshape(4, 2, 3)[::2]

    extractor = Extractor()
    values = extractor.values(ctx)
    before = values.copy()
    extractor._validate_output(extractor.values, values, ctx)
    np.testing.assert_array_equal(values, before)
    assert not values.flags.c_contiguous


def test_output_validation_under_optimization():
    source = """
import numpy as np
from sawnergy.v2.traj_features.feature_extractor import ExtractionContext, FeatureExtractor, feature
class Extractor(FeatureExtractor):
    @feature(kind='pairwise', scope='window', shape=('N', 'N'), dtype='float64',
             units='angstrom', statistic='mean per-frame distance')
    def distances(self, ctx):
        return np.ones((1, 1, 1))
ctx = ExtractionContext(cpptraj='cpptraj', topology_file='topology', trajectory_file='trajectory',
                        selection_mask=':1', residue_ids=((1, 1),), atom_ids=((1,),),
                        frame_ids=(1,), windows=((1,),))
extractor = Extractor()
extractor._validate_output(extractor.distances, extractor.distances(ctx), ctx)
for values, error in ((None, TypeError), (np.ones((1, 1, 1), dtype='float32'), TypeError),
                      (np.ones((1, 1)), ValueError), (np.full((1, 1, 1), np.nan), ValueError),
                      (np.ma.array(np.ones((1, 1, 1)), mask=True), TypeError)):
    try:
        extractor._validate_output(extractor.distances, values, ctx)
    except error:
        continue
    raise RuntimeError('Invalid output was accepted')
try:
    feature(kind='global', scope='trajectory', shape=(-1,), dtype='float64', units='1', statistic='mean')
except ValueError:
    pass
else:
    raise RuntimeError('Invalid shape was accepted')
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run([sys.executable, "-O", "-c", source], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
