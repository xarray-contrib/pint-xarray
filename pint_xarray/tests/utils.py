import re
from contextlib import contextmanager

import numpy as np
import pytest
from xarray.testing import assert_equal, assert_identical  # noqa: F401

from ..conversion import extract_units


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


def assert_units_equal(a, b):
    __tracebackhide__ = True
    assert extract_units(a) == extract_units(b)
