from __future__ import annotations

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
