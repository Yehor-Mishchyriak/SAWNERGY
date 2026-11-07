from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sawnergy import sawnergy_util


INPUT_FACTORIES = [list, tuple, np.asarray, iter, lambda values: (v for v in values)]
INPUT_IDS = ["list", "tuple", "ndarray", "iterator", "generator"]


@pytest.mark.parametrize("make_input", INPUT_FACTORIES, ids=INPUT_IDS)
@pytest.mark.parametrize("batch_size", [-1, 0, 2])
@pytest.mark.parametrize("ranges,inclusive_end", [(False, False), (True, False), (True, True)])
def test_batches_of_empty_input_yields_nothing(make_input, batch_size, ranges, inclusive_end):
    assert list(sawnergy_util.batches_of(
        make_input([]), batch_size=batch_size, ranges=ranges, inclusive_end=inclusive_end,
    )) == []


@pytest.mark.parametrize("make_input", [list, iter], ids=["sequence", "iterator"])
def test_batches_of_empty_default_does_not_call_wrapper(make_input):
    def unexpected_batch(values):
        pytest.fail("Empty input must not produce a batch or call out_as")

    assert list(sawnergy_util.batches_of(make_input([]))) == []
    assert list(sawnergy_util.batches_of(
        make_input([]), ranges=True, inclusive_end=True, out_as=unexpected_batch,
    )) == []


@pytest.mark.parametrize("make_input", INPUT_FACTORIES, ids=INPUT_IDS)
@pytest.mark.parametrize("batch_size,bounds", [
    (-1, [(0, 5)]),
    (0, [(0, 5)]),
    (1, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
    (2, [(0, 2), (2, 4), (4, 5)]),
    (3, [(0, 3), (3, 5)]),
    (8, [(0, 5)]),
])
def test_batches_of_preserves_nonempty_values_ranges_and_wrappers(make_input, batch_size, bounds):
    values = [10, 20, 30, 40, 50]
    expected = [values[start:stop] for start, stop in bounds]

    assert list(sawnergy_util.batches_of(make_input(values), batch_size)) == expected
    assert list(sawnergy_util.batches_of(
        make_input(values), batch_size, ranges=True,
    )) == [list(pair) for pair in bounds]
    assert list(sawnergy_util.batches_of(
        make_input(values), batch_size, ranges=True, inclusive_end=True, out_as=tuple,
    )) == [(start, stop - 1) for start, stop in bounds]
    assert list(sawnergy_util.batches_of(
        make_input(values), batch_size, inclusive_end=True, out_as=tuple,
    )) == [tuple(batch) for batch in expected]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("values", [[], [3, 1, 2]])
def test_elementwise_processor_preserves_order_arguments_and_completion(monkeypatch, parallel, capture, values):
    # Force collection in reverse submission order, independently of scheduling.
    monkeypatch.setattr(sawnergy_util, "as_completed", lambda futures: reversed(list(futures)))
    seen = []

    def transform(value, factor, *, offset):
        seen.append(value)
        return value * factor + offset

    runner = sawnergy_util.elementwise_processor(
        in_parallel=parallel, Executor=ThreadPoolExecutor, max_workers=2, capture_output=capture,
    )
    result = runner(iter(values), transform, 2, offset=5)
    assert result == ([value * 2 + 5 for value in values] if capture else None)
    assert sorted(seen) == sorted(values)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_elementwise_processor_propagates_task_failure(parallel, capture):
    error = ValueError("task failed")

    def fail(value):
        raise error

    runner = sawnergy_util.elementwise_processor(
        in_parallel=parallel, Executor=ThreadPoolExecutor, max_workers=1, capture_output=capture,
    )
    with pytest.raises(ValueError) as caught:
        runner([1], fail)
    assert caught.value is error


def test_elementwise_processor_requires_executor_at_invocation():
    runner = sawnergy_util.elementwise_processor(in_parallel=True)
    with pytest.raises(ValueError, match="Executor"):
        runner([], abs)


@pytest.mark.parametrize("workers", [None, 0])
@pytest.mark.parametrize("cpus", [None, 3])
def test_elementwise_processor_resolves_falsy_worker_count(monkeypatch, workers, cpus):
    seen = []

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, *, max_workers):
            seen.append(max_workers)
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(sawnergy_util.os, "cpu_count", lambda: cpus)
    runner = sawnergy_util.elementwise_processor(
        in_parallel=True, Executor=RecordingExecutor, max_workers=workers,
    )
    assert runner([-1, -2], abs) == [1, 2]
    assert seen == [cpus or 1]


def test_create_updated_subprocess_env_copies_converts_and_removes(monkeypatch):
    monkeypatch.setenv("SAWNERGY_KEEP", "inherited")
    monkeypatch.setenv("SAWNERGY_REMOVE", "remove me")
    before = dict(os.environ)
    env = sawnergy_util.create_updated_subprocess_env(
        SAWNERGY_REMOVE=None, SAWNERGY_ABSENT=None, SAWNERGY_TRUE=True,
        SAWNERGY_FALSE=False, SAWNERGY_INT=3, SAWNERGY_PATH=Path("some directory/file"),
    )
    assert env == {
        **{key: value for key, value in before.items()
           if key not in ("SAWNERGY_REMOVE", "SAWNERGY_ABSENT")},
        "SAWNERGY_TRUE": "TRUE", "SAWNERGY_FALSE": "FALSE",
        "SAWNERGY_INT": "3", "SAWNERGY_PATH": "some directory/file",
    }
    env["SAWNERGY_KEEP"] = "child only"
    assert dict(os.environ) == before


@pytest.mark.parametrize("text,expected", [
    ("", []), ("header\n", ["header\n"]),
    ("header\r\nαβ\r\nlast", ["header\n", "αβ\n", "last"]),
])
def test_read_lines_preserves_text_and_skips_exactly_one_header(tmp_path, text, expected):
    path = tmp_path / "lines.txt"
    path.write_bytes(text.encode("utf-8"))
    assert sawnergy_util.read_lines(str(path), skip_header=False) == expected
    assert sawnergy_util.read_lines(str(path)) == expected[1:]


def test_read_lines_propagates_file_and_decoding_errors(tmp_path):
    path = tmp_path / "lines.txt"
    with pytest.raises(FileNotFoundError):
        sawnergy_util.read_lines(str(path))
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        sawnergy_util.read_lines(str(path))


def test_temporary_file_returns_unique_closed_empty_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sawnergy_util.tempfile, "tempdir", str(tmp_path))
    paths = []
    try:
        for _ in range(2):
            path = sawnergy_util.temporary_file("sawnergy-", ".dat")
            paths.append(path)
            assert isinstance(path, Path) and path.parent == tmp_path
            assert path.name.startswith("sawnergy-") and path.suffix == ".dat"
            assert path.read_bytes() == b""
            path.write_text("caller-owned")
            assert path.read_text() == "caller-owned"
        assert paths[0] != paths[1]
    finally:
        for path in paths:
            path.unlink()


def test_compose_steps_order_identity_kwargs_and_none_results():
    value = object()
    assert sawnergy_util.compose_steps()(value) is value
    kwargs = {"factor": 2}
    pipeline = sawnergy_util.compose_steps(
        (lambda value, *, factor: value * factor, kwargs), (lambda value: value + 3, None),
    )
    assert pipeline(5) == 13
    kwargs["factor"] = 4  # Options are read at invocation, not snapshotted.
    assert pipeline(5) == 23
    seen = []
    pipeline = sawnergy_util.compose_steps(
        (lambda value, **options: options["seen"].append(value), {"seen": seen}),
        (lambda value: (value, seen.copy()), None),
    )
    assert pipeline("input") == (None, ["input"])


@pytest.mark.parametrize("step", [None, (), (abs,), (abs, {}, 1), [abs, {}], (None, {})])
def test_compose_steps_rejects_invalid_steps_at_construction(step):
    with pytest.raises(TypeError):
        sawnergy_util.compose_steps(step)


def test_compose_steps_defers_kwargs_conversion_and_preserves_step_errors():
    pipeline = sawnergy_util.compose_steps((abs, object()))
    with pytest.raises(TypeError):
        pipeline(-1)
    error = RuntimeError("step failed")
    seen = []

    def fail(value):
        raise error

    pipeline = sawnergy_util.compose_steps((fail, None), (seen.append, None))
    with pytest.raises(RuntimeError) as caught:
        pipeline(1)
    assert caught.value is error and seen == []


def test_current_time_uses_local_formatted_clock(monkeypatch):
    monkeypatch.setattr(sawnergy_util, "datetime", SimpleNamespace(
        now=lambda: datetime(2026, 11, 6, 7, 8, 9),
    ))
    assert sawnergy_util.current_time() == "2026-11-06_070809"


@pytest.mark.parametrize("has_parent,name,expected", [
    (False, "MainProcess", True), (True, "MainProcess", False), (False, "Worker", False),
])
def test_is_main_process_requires_name_and_no_parent(monkeypatch, has_parent, name, expected):
    monkeypatch.setattr(sawnergy_util.mp, "parent_process", lambda: object() if has_parent else None)
    monkeypatch.setattr(sawnergy_util.mp, "current_process", lambda: SimpleNamespace(name=name))
    assert sawnergy_util.is_main_process() is expected
