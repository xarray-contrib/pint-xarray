import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.indexes import IndexSelResult, PandasIndex

from pint_xarray import unit_registry as ureg
from pint_xarray.index import PintIndex


def indexer_equal(first, second):
    if type(first) is not type(second):
        return False

    if isinstance(first, np.ndarray):
        return np.all(first == second)
    else:
        return first == second


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
        (
            {"x": ureg.Quantity([0.002, 0.004], "km")},
            IndexSelResult(dim_indexers={"x": np.array([1, 3])}),
        ),
        (
            {"x": slice(ureg.Quantity(2, "m"), ureg.Quantity(3000, "mm"))},
            IndexSelResult(dim_indexers={"x": slice(1, 3)}),
        ),
    ),
)
def test_sel(labels, expected):
    index = PintIndex(
        index=PandasIndex(pd.Index([1, 2, 3, 4]), dim="x"), units={"x": ureg.Unit("m")}
    )

    actual = index.sel(labels)

    assert isinstance(actual, IndexSelResult)
    assert list(actual.dim_indexers.keys()) == list(expected.dim_indexers.keys())
    assert all(
        [
            indexer_equal(actual.dim_indexers[k], expected.dim_indexers[k])
            for k in expected.dim_indexers.keys()
        ]
    )


@pytest.mark.parametrize(
    "indexers",
    ({"y": 0}, {"y": [1, 2]}, {"y": slice(0, None, 2)}, {"y": xr.Variable("y", [1])}),
)
def test_isel(indexers):
    wrapped_index = PandasIndex(pd.Index([1, 2, 3, 4]), dim="y")
    index = PintIndex(index=wrapped_index, units={"y": ureg.Unit("s")})

    actual = index.isel(indexers)

    wrapped_ = wrapped_index.isel(indexers)
    if wrapped_ is not None:
        expected = PintIndex(
            index=wrapped_index.isel(indexers), units={"y": ureg.Unit("s")}
        )
    else:
        expected = None

    assert (actual is None and expected is None) or actual.equals(expected)


@pytest.mark.parametrize(
    ["other", "expected"],
    (
        (
            PintIndex(
                index=PandasIndex(pd.Index([1, 2, 3, 4]), dim="x"),
                units={"x": ureg.Unit("cm")},
            ),
            True,
        ),
        (PandasIndex(pd.Index([1, 2, 3, 4]), dim="x"), False),
        (
            PintIndex(
                index=PandasIndex(pd.Index([1, 2, 3, 4]), dim="x"),
                units={"x": ureg.Unit("m")},
            ),
            False,
        ),
        (
            PintIndex(
                index=PandasIndex(pd.Index([1, 2, 3, 4]), dim="y"),
                units={"y": ureg.Unit("cm")},
            ),
            False,
        ),
        (
            PintIndex(
                index=PandasIndex(pd.Index([1, 3, 3, 4]), dim="x"),
                units={"x": ureg.Unit("cm")},
            ),
            False,
        ),
    ),
)
def test_equals(other, expected):
    index = PintIndex(
        index=PandasIndex(pd.Index([1, 2, 3, 4]), dim="x"), units={"x": ureg.Unit("cm")}
    )

    actual = index.equals(other)

    assert actual == expected


@pytest.mark.parametrize(
    ["shifts", "expected_index"],
    (
        ({"x": 0}, PandasIndex(pd.Index([-2, -1, 0, 1, 2]), dim="x")),
        ({"x": 1}, PandasIndex(pd.Index([2, -2, -1, 0, 1]), dim="x")),
        ({"x": 2}, PandasIndex(pd.Index([1, 2, -2, -1, 0]), dim="x")),
        ({"x": -1}, PandasIndex(pd.Index([-1, 0, 1, 2, -2]), dim="x")),
        ({"x": -2}, PandasIndex(pd.Index([0, 1, 2, -2, -1]), dim="x")),
    ),
)
def test_roll(shifts, expected_index):
    index = PintIndex(
        index=PandasIndex(pd.Index([-2, -1, 0, 1, 2]), dim="x"),
        units={"x": ureg.Unit("m")},
    )

    actual = index.roll(shifts)
    expected = index._replace(expected_index)

    assert actual.equals(expected)


@pytest.mark.parametrize("dims_dict", ({"y": "x"}, {"y": "z"}))
@pytest.mark.parametrize("name_dict", ({"y2": "y3"}, {"y2": "y1"}))
def test_rename(name_dict, dims_dict):
    wrapped_index = PandasIndex(pd.Index([1, 2], name="y2"), dim="y")
    index = PintIndex(index=wrapped_index, units={"y": ureg.Unit("m")})

    actual = index.rename(name_dict, dims_dict)
    expected = PintIndex(
        index=wrapped_index.rename(name_dict, dims_dict), units=index.units
    )

    assert actual.equals(expected)


@pytest.mark.parametrize("indexer", ([0], slice(0, 2)))
def test_getitem(indexer):
    wrapped_index = PandasIndex(pd.Index([1, 2], name="y2"), dim="y")
    index = PintIndex(index=wrapped_index, units={"y": ureg.Unit("m")})

    actual = index[indexer]
    expected = PintIndex(index=wrapped_index[indexer], units=index.units)

    assert actual.equals(expected)


@pytest.mark.parametrize("wrapped_index", (PandasIndex(pd.Index([1, 2]), dim="x"),))
def test_repr_inline(wrapped_index):
    index = PintIndex(index=wrapped_index, units=ureg.Unit("m"))

    # TODO: parametrize
    actual = index._repr_inline_(90)

    assert "PintIndex" in actual
    assert wrapped_index.__class__.__name__ in actual
