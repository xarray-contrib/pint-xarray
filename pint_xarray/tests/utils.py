import re
from contextlib import contextmanager

import numpy as np
import pytest
from pint import Quantity
from xarray.testing import assert_equal, assert_identical  # noqa: F401

from ..conversion import array_strip_units, extract_indexer_units, extract_units


@contextmanager
def raises_regex(error, pattern):
    __tracebackhide__ = True
    with pytest.raises(error) as excinfo:
        yield
    message = str(excinfo.value)
    if not re.search(pattern, message):
        raise AssertionError(
            f"exception {excinfo.value!r} did not match pattern {pattern!r}"
        )


def assert_array_units_equal(a, b):
    __tracebackhide__ = True

    units_a = getattr(a, "units", None)
    units_b = getattr(b, "units", None)

    assert units_a == units_b


def assert_array_equal(a, b):
    __tracebackhide__ = True

    a_ = getattr(a, "magnitude", a)
    b_ = getattr(b, "magnitude", b)

    np.testing.assert_array_equal(a_, b_)


def assert_slice_equal(a, b):
    attrs = ("start", "stop", "step")
    values_a = tuple(getattr(a, name) for name in attrs)
    values_b = tuple(getattr(b, name) for name in attrs)
    stripped_a = tuple(array_strip_units(v) for v in values_a)
    stripped_b = tuple(array_strip_units(v) for v in values_b)

    assert (
        stripped_a == stripped_b
    ), f"different values: {stripped_a!r} ←→ {stripped_b!r}"


def assert_indexer_equal(a, b):
    __tracebackhide__ = True

    assert type(a) == type(b)
    if isinstance(a, slice):
        assert_slice_equal(a, b)
    elif isinstance(a, (Quantity, np.ndarray)):
        assert_array_equal(a, b)
    else:
        a_ = array_strip_units(a)
        b_ = array_strip_units(b)
        assert a_ == b_, f"different values: {a_!r} ←→ {b_!r}"


def assert_indexer_units_equal(a, b):
    __tracebackhide__ = True

    units_a = extract_indexer_units(a)
    units_b = extract_indexer_units(b)

    assert units_a == units_b, f"different units: {units_a!r} ←→ {units_b!r}"


def assert_units_equal(a, b):
    __tracebackhide__ = True
    assert extract_units(a) == extract_units(b)
