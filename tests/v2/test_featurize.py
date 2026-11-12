import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from sawnergy import __version__
from sawnergy.sawnergy_util import ArrayStorage
from sawnergy.v2.traj_features.feature_extractor import ExtractionContext, FeatureExtractor, feature


@pytest.fixture
def run(tmp_path, monkeypatch):
    ctx = ExtractionContext(cpptraj=tmp_path / "cpptraj", topology_file=tmp_path / "topology",
                            trajectory_file=tmp_path / "trajectory", residue_ids=((1, 1), (2, 3)),
                            atom_ids=((1, 2), (5,)), frame_ids=(2, 4, 6, 8, 10),
                            windows=((2, 4), (6, 8), (10,)))
    monkeypatch.setattr(FeatureExtractor, "_prepare_inputs", lambda self, *args, **kwargs: ctx)

    def extract(extractor, output=None, **changes):
        options = dict(frame_range=(2, 10), frame_stride=2, window_size=2)
        options.update(changes)
        return extractor.featurize(ctx.topology_file, ctx.trajectory_file, ctx.cpptraj,
                                   {1: [1], 2: [3]}, output or tmp_path / "features.zip", **options)

    return extract


def _extractor(events):
    class Extractor(FeatureExtractor):
        @feature(kind="pairwise", scope="window", shape=("N", "N"), dtype=">f8", units="frame", statistic="mean frame ID")
        def pairs(self, ctx):
            assert not ctx.dependencies
            events.append(("pairs", ctx.windows))
            means = np.array([np.mean(window) for window in ctx.windows], dtype=">f8")
            return np.broadcast_to(means[:, None, None], (len(means), 2, 2))

        @feature(kind="global", scope="window", shape=(), dtype="float32", units="frame", statistic="pair score", requires=("pairs",))
        def scores(self, ctx):
            assert set(ctx.dependencies) == {"pairs"}
            events.append(("scores", ctx.windows))
            values = ctx.dependencies["pairs"].read()
            assert len(values) == len(ctx.windows)
            return values[:, 0, 0].astype("float32")

        @feature(kind="global", scope="trajectory", shape=(), dtype="float64", units="frame", statistic="frame-weighted mean", requires=("scores",))
        def overall(self, ctx):
            events.append(("overall", ctx.windows))
            scores = np.concatenate(list(ctx.dependencies["scores"].block_iter(step=2)))
            return np.asarray(np.average(scores, weights=[len(window) for window in ctx.windows]))

        @feature(kind="pairwise", scope="trajectory", shape=("N", "N"), dtype="float64", units="frame", statistic="maximum window score", requires=("pairs",))
        def pair_summary(self, ctx):
            events.append(("pair_summary", ctx.windows))
            return np.maximum.reduce([batch[0] for batch in ctx.dependencies["pairs"].block_iter()])

        @feature(kind="global", scope="window", shape=(), dtype=bool, units="1", statistic="above mean", requires=("scores", "overall"))
        def above_mean(self, ctx):
            events.append(("above_mean", ctx.windows))
            assert ctx.dependencies["overall"].read().shape == (1,)
            return ctx.dependencies["scores"].read() > ctx.dependencies["overall"].read(0)

        @feature(kind="global", scope="trajectory", shape=(0,), dtype="float32", units="1", statistic="empty vector")
        def empty(self, ctx):
            events.append(("empty", ctx.windows))
            return np.empty(0, dtype="float32")

    return Extractor()


@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_featurize_publishes_exact_layout_and_aligned_dependencies(run, batch_size):
    events = []
    output = run(_extractor(events), batch_size=batch_size)
    assert Path(output).is_absolute() and Path(output).suffix == ".zip"
    names = [name for name, _ in events]
    assert names == sorted(names, key=["pairs", "scores", "overall", "pair_summary", "above_mean", "empty"].index)
    assert names.count("overall") == names.count("pair_summary") == names.count("empty") == 1
    with ArrayStorage(output, "r") as storage:
        assert storage.get_attr("schema_name") == "traj_features"
        assert storage.get_attr("schema_version") == 1
        assert storage.get_attr("producer_version") == __version__
        assert storage.get_attr("frame_range") == [2, 10]
        assert storage.get_attr("frame_stride") == storage.get_attr("window_size") == 2
        np.testing.assert_array_equal(storage.read("residue_ids"), [[1, 1], [2, 3]])
        np.testing.assert_array_equal(storage.read("atom_ids"), [1, 2, 5])
        np.testing.assert_array_equal(storage.read("atom_offsets"), [0, 2, 3])
        np.testing.assert_array_equal(storage.read("frame_ids"), [2, 4, 6, 8, 10])
        np.testing.assert_array_equal(storage.read("window_offsets"), [0, 2, 4, 5])
        np.testing.assert_array_equal(storage.read("feature_0")[:, 0, 0], [3, 7, 10])
        np.testing.assert_array_equal(storage.read("feature_1"), [3, 7, 10])
        np.testing.assert_array_equal(storage.read("feature_2"), [6])
        np.testing.assert_array_equal(storage.read("feature_3"), np.full((1, 2, 2), 10.0))
        np.testing.assert_array_equal(storage.read("feature_4"), [False, True, True])
        assert storage.read("feature_5").shape == (1, 0)
        manifest = storage.get_attr("features")
        assert [entry["name"] for entry in manifest] == ["pairs", "scores", "overall", "pair_summary", "above_mean", "empty"]
        assert [entry["block"] for entry in manifest] == [f"feature_{i}" for i in range(6)]
        assert manifest[0] == dict(name="pairs", block="feature_0", kind="pairwise", scope="window",
                                   shape=["N", "N"], dtype="float64", units="frame", statistic="mean frame ID", requires=[])
        assert manifest[4]["requires"] == ["scores", "overall"]
        assert set(storage.list_blocks()) == {"residue_ids", "atom_ids", "atom_offsets", "frame_ids", "window_offsets"} | {entry["block"] for entry in manifest}


@pytest.mark.parametrize("changes", [{"batch_size": 0}, {"batch_size": True},
                                     {"compression_level": -1}, {"compression_level": True},
                                     {"compression_level": 10}, {"compression_level": 1.5},
                                     {"compression_level": None}, {"compression_level": "3"}])
def test_featurize_rejects_invalid_options(run, changes):
    with pytest.raises(ValueError):
        run(_extractor([]), **changes)


def test_featurize_requires_zip_destination_and_features(run, tmp_path):
    with pytest.raises(ValueError):
        run(_extractor([]), tmp_path / "directory")
    with pytest.raises(ValueError):
        run(FeatureExtractor())


@pytest.mark.parametrize("values,error", [(None, TypeError), (1.0, TypeError),
    (np.ones(1, dtype="float32"), TypeError), (np.ma.array([1.0], mask=False), TypeError),
    (np.ones((1, 1)), ValueError), (np.asarray([np.nan]), ValueError)])
def test_featurize_rejects_bad_outputs_before_writing_them(run, monkeypatch, values, error):
    write = ArrayStorage.write

    def checked_write(self, values, to_block_named, **kwargs):
        assert to_block_named != "feature_0"
        return write(self, values, to_block_named, **kwargs)

    monkeypatch.setattr(ArrayStorage, "write", checked_write)

    class Invalid(FeatureExtractor):
        @feature(kind="global", scope="window", shape=(), dtype="float64", units="1", statistic="test")
        def value(self, ctx):
            return values

    with pytest.raises(error):
        run(Invalid())


def test_featurize_preserves_compression_failure_and_existing_destination(run, tmp_path, monkeypatch):
    error = OSError("compression failed")
    paths = []

    def fail(self, into, *, compression_level):
        paths.append(self.store_path)
        raise error

    monkeypatch.setattr(ArrayStorage, "compress", fail)
    output = tmp_path / "features.zip"
    output.write_bytes(b"previous archive")
    with pytest.raises(OSError) as caught:
        run(_extractor([]), output)
    assert caught.value is error
    assert output.read_bytes() == b"previous archive"
    assert paths and all(not path.exists() for path in paths)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error", [RuntimeError("failed"), KeyboardInterrupt("interrupted")])
def test_featurize_preserves_feature_errors_and_destination(run, tmp_path, monkeypatch, existing, error):
    output = tmp_path / "features.zip"
    if existing:
        output.write_bytes(b"previous archive")
    paths = []
    initialize = ArrayStorage.__init__

    def track(self, path, mode):
        initialize(self, path, mode)
        if mode == "w":
            paths.append(self.store_path)

    monkeypatch.setattr(ArrayStorage, "__init__", track)

    class Failing(FeatureExtractor):
        @feature(kind="global", scope="window", shape=(), dtype="float64", units="1", statistic="test")
        def values(self, ctx):
            if ctx.windows[0][0] > 2:
                raise error
            return np.ones(len(ctx.windows))

    with pytest.raises(type(error)) as caught:
        run(Failing(), output)
    assert caught.value is error
    assert paths and all(not path.exists() for path in paths)
    assert output.read_bytes() == b"previous archive" if existing else not output.exists()


@pytest.mark.parametrize("corruption,error", [
    ("schema", ValueError), ("identity", ValueError), ("offset", ValueError), ("frame", ValueError),
    ("window", ValueError), ("manifest", ValueError), ("metadata_type", ValueError),
    ("extra_array", ValueError), ("extra_group", ValueError), ("missing", ValueError),
    ("nan", ValueError), ("shape", ValueError), ("cache_shape", ValueError),
    ("cache_chunk", ValueError), ("cache_dtype", TypeError), ("dtype", TypeError),
])
def test_finished_archive_is_revalidated_before_publication(run, tmp_path, monkeypatch, corruption, error):
    write = ArrayStorage.write
    paths = []

    def corrupt(self, values, to_block_named, **kwargs):
        write(self, values, to_block_named, **kwargs)
        if to_block_named != "feature_0":
            return
        paths.append(self.store_path)
        if corruption == "schema":
            self.add_attr("schema_version", 2)
        elif corruption == "identity":
            self.root["residue_ids"][0, 1] = 99
        elif corruption == "offset":
            self.root["atom_offsets"][1] = 0
        elif corruption == "frame":
            self.root["frame_ids"][0] = 3
        elif corruption == "window":
            self.root["window_offsets"][1] = 1
        elif corruption == "manifest":
            manifest = self.get_attr("features")
            manifest[0]["statistic"] = "changed"
            self.add_attr("features", manifest)
        elif corruption == "metadata_type":
            self.add_attr("frame_stride", 2.0)
        elif corruption == "extra_array":
            self.root.create_array("extra", data=np.ones(1))
        elif corruption == "extra_group":
            self.root.create_group("extra")
        elif corruption == "missing":
            self.root.__delitem__("atom_ids")
        elif corruption == "nan":
            self.root["feature_0"][0] = np.nan
        elif corruption == "shape":
            self.root["feature_0"].resize((0,))
        elif corruption.startswith("cache_"):
            field, value = {"cache_shape": ("array_shape_in_block", [2]),
                            "cache_chunk": ("array_chunk_size_in_block", True),
                            "cache_dtype": ("array_dtype_in_block", None)}[corruption]
            cached = self.get_attr(field)
            cached["feature_0"] = value
            self.add_attr(field, cached)
        else:
            self.root.__delitem__("feature_0")
            self.root.create_array("feature_0", data=np.ones(1, dtype="int16"))

    monkeypatch.setattr(ArrayStorage, "write", corrupt)

    class Single(FeatureExtractor):
        @feature(kind="global", scope="trajectory", shape=(), dtype="float64", units="1", statistic="one")
        def value(self, ctx):
            return np.asarray(1.0)

    output = tmp_path / "features.zip"
    output.write_bytes(b"previous archive")
    with pytest.raises(error):
        run(Single(), output)
    assert output.read_bytes() == b"previous archive"
    assert paths and all(not path.exists() for path in paths)


def test_native_featurize_runs_from_input_preparation_to_archive(tmp_path):
    cpptraj = os.environ.get("CPPTRAJ") or shutil.which("cpptraj")
    if not cpptraj:
        pytest.skip("Set CPPTRAJ to run the native pipeline test")
    example = Path(__file__).resolve().parents[2] / "example_MD_for_quick_start"

    class FrameMean(FeatureExtractor):
        @feature(kind="global", scope="window", shape=(), dtype="float64", units="frame", statistic="mean source-frame ID")
        def mean_frame(self, ctx):
            return np.asarray([np.mean(window) for window in ctx.windows])

    output = FrameMean().featurize(example / "p53_DBD.prmtop", example / "p53_DBD.nc", cpptraj,
                                  {1: [3, 1]}, tmp_path / "native.zip", frame_range=(2, 11),
                                  frame_stride=3, window_size=3, batch_size=2)
    with ArrayStorage(output, "r") as storage:
        np.testing.assert_array_equal(storage.read("feature_0"), [5, 11])
        np.testing.assert_array_equal(storage.read("frame_ids"), [2, 5, 8, 11])


def test_featurize_preserves_additional_inert_metadata(run, monkeypatch):
    write = ArrayStorage.write

    def annotate(self, values, to_block_named, **kwargs):
        write(self, values, to_block_named, **kwargs)
        self.add_attr("experiment", "example")
        manifest = self.get_attr("features")
        manifest[0]["description"] = "extra provenance"
        self.add_attr("features", manifest)

    monkeypatch.setattr(ArrayStorage, "write", annotate)
    output = run(_extractor([]))
    with ArrayStorage(output, "r") as storage:
        assert storage.get_attr("experiment") == "example"
        assert storage.get_attr("features")[0]["description"] == "extra provenance"


def test_featurize_and_final_validation_under_optimization(tmp_path):
    source = """
import sys
import numpy as np
from sawnergy.sawnergy_util import ArrayStorage
from sawnergy.v2.traj_features.feature_extractor import ExtractionContext, FeatureExtractor, feature
class Extractor(FeatureExtractor):
    def _prepare_inputs(self, *args, **kwargs):
        return ExtractionContext(cpptraj='cpptraj', topology_file='topology', trajectory_file='trajectory',
                                 residue_ids=((1, 1),), atom_ids=((1,),), frame_ids=(1, 2), windows=((1,), (2,)))
    @feature(kind='global', scope='window', shape=(), dtype='float64', units='1', statistic='one')
    def values(self, ctx):
        return np.ones(len(ctx.windows))
extractor = Extractor()
for value in (None, '3', True):
    try:
        extractor.featurize('top', 'traj', 'cpptraj', {1: [1]}, sys.argv[1], compression_level=value)
    except ValueError:
        continue
    raise RuntimeError('Invalid compression level was accepted')
extractor.featurize('top', 'traj', 'cpptraj', {1: [1]}, sys.argv[1])
write = ArrayStorage.write
def corrupt(self, values, to_block_named, **kwargs):
    write(self, values, to_block_named, **kwargs)
    if to_block_named == 'feature_0':
        self.root['frame_ids'][0] = 99
ArrayStorage.write = corrupt
try:
    extractor.featurize('top', 'traj', 'cpptraj', {1: [1]}, sys.argv[1])
except ValueError:
    pass
else:
    raise RuntimeError('Corrupted archive was published')
"""
    output = tmp_path / "optimized.zip"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run([sys.executable, "-O", "-c", source, str(output)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with ArrayStorage(output, "r") as storage:
        np.testing.assert_array_equal(storage.read("frame_ids"), [1, 2])
