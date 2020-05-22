from contextlib import contextmanager
import re

import pytest

import xarray as xr

from pint.quantity import Quantity


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


def array_extract_units(obj):
    if isinstance(obj, (xr.Variable, xr.DataArray, xr.Dataset)):
        obj = obj.data

    try:
        return obj.units
    except AttributeError:
        return None


def extract_units(obj):
    if isinstance(obj, xr.Dataset):
        vars_units = {
            name: array_extract_units(value) for name, value in obj.data_vars.items()
        }
        coords_units = {
            name: array_extract_units(value) for name, value in obj.coords.items()
        }

        units = {**vars_units, **coords_units}
    elif isinstance(obj, xr.DataArray):
        vars_units = {obj.name: array_extract_units(obj)}
        coords_units = {
            name: array_extract_units(value) for name, value in obj.coords.items()
        }

        units = {**vars_units, **coords_units}
    elif isinstance(obj, xr.Variable):
        vars_units = {None: array_extract_units(obj.data)}

        units = {**vars_units}
    elif isinstance(obj, Quantity):
        vars_units = {None: array_extract_units(obj)}

        units = {**vars_units}
    else:
        units = {}

    return units


def assert_units_equal(a, b):
    __tracebackhide__ = True
    assert extract_units(a) == extract_units(b)


def assert_equal_with_units(a, b):
    # works like xr.testing.assert_equal, but also explicitly checks units
    # so, it is more like assert_identical
    __tracebackhide__ = True

    if isinstance(a, xr.Dataset) or isinstance(b, xr.Dataset):
        a_units = extract_units(a)
        b_units = extract_units(b)

        a_without_units = strip_units(a)
        b_without_units = strip_units(b)

        assert a_without_units.equals(b_without_units), formatting.diff_dataset_repr(
            a, b, "equals"
        )
        assert a_units == b_units
    else:
        a = a if not isinstance(a, (xr.DataArray, xr.Variable)) else a.data
        b = b if not isinstance(b, (xr.DataArray, xr.Variable)) else b.data

        assert type(a) == type(b) or (
            isinstance(a, Quantity) and isinstance(b, Quantity)
        )

        # workaround until pint implements allclose in __array_function__
        if isinstance(a, Quantity) or isinstance(b, Quantity):
            assert (
                hasattr(a, "magnitude") and hasattr(b, "magnitude")
            ) and np.allclose(a.magnitude, b.magnitude, equal_nan=True)
            assert (hasattr(a, "units") and hasattr(b, "units")) and a.units == b.units
        else:
            assert np.allclose(a, b, equal_nan=True)
