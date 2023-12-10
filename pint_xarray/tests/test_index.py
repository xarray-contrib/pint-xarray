import pandas as pd
import pytest
import xarray as xr
from xarray.core.indexes import IndexSelResult, PandasIndex

from pint_xarray import unit_registry as ureg
from pint_xarray.index import PintIndex


@pytest.mark.parametrize(
    "base_index",
    [
        PandasIndex(pd.Index([1, 2, 3]), dim="x"),
        PandasIndex(pd.Index([0.1, 0.2, 0.3]), dim="x"),
        PandasIndex(pd.Index([1j, 2j, 3j]), dim="y"),
    ],
)
@pytest.mark.parametrize("units", [ureg.Unit("m"), ureg.Unit("s")])
def test_init(base_index, units):
    index = PintIndex(index=base_index, units=units)

    assert index.index.equals(base_index)
    assert index.units == units


def test_replace():
    old_index = PandasIndex([1, 2, 3], dim="y")
    new_index = PandasIndex([0.1, 0.2, 0.3], dim="x")

    old = PintIndex(index=old_index, units=ureg.Unit("m"))
    new = old._replace(new_index)

    assert new.index.equals(new_index)
    assert new.units == old.units
    # no mutation
    assert old.index.equals(old_index)


@pytest.mark.parametrize(
    ["wrapped_index", "units", "expected"],
    (
        pytest.param(
            PandasIndex(pd.Index([1, 2, 3]), dim="x"),
            {"x": ureg.Unit("m")},
            {"x": xr.Variable("x", ureg.Quantity([1, 2, 3], "m"))},
        ),
        pytest.param(
            PandasIndex(pd.Index([1j, 2j, 3j]), dim="y"),
            {"y": ureg.Unit("ms")},
            {"y": xr.Variable("y", ureg.Quantity([1j, 2j, 3j], "ms"))},
        ),
    ),
)
def test_create_variables(wrapped_index, units, expected):
    index = PintIndex(index=wrapped_index, units=units)

    actual = index.create_variables()

    assert list(actual.keys()) == list(expected.keys())
    assert all([actual[k].equals(expected[k]) for k in expected.keys()])


@pytest.mark.parametrize(
    ["labels", "expected"],
    (
        ({"x": ureg.Quantity(1, "m")}, IndexSelResult(dim_indexers={"x": 0})),
        ({"x": ureg.Quantity(3000, "mm")}, IndexSelResult(dim_indexers={"x": 2})),
        ({"x": ureg.Quantity(0.002, "km")}, IndexSelResult(dim_indexers={"x": 1})),
    ),
)
def test_sel(labels, expected):
    index = PintIndex(
        index=PandasIndex(pd.Index([1, 2, 3, 4]), dim="x"), units={"x": ureg.Unit("m")}
    )

    actual = index.sel(labels)

    assert actual == expected
