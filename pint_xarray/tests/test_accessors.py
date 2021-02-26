import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from pint import Unit, UnitRegistry
from pint.errors import UndefinedUnitError

from .. import accessors, conversion
from ..errors import DimensionalityError
from .utils import assert_equal, assert_identical, assert_units_equal, raises_regex

pytestmark = [
    pytest.mark.filterwarnings("error::pint.UnitStrippedWarning"),
]

# make sure scalars are converted to 0d arrays so quantities can
# always be treated like ndarrays
unit_registry = UnitRegistry(force_ndarray=True)
Quantity = unit_registry.Quantity


def assert_all_str_or_none(mapping):
    __tracebackhide__ = True

    compared = {
        key: isinstance(value, str) or value is None for key, value in mapping.items()
    }
    not_passing = {key: value for key, value in mapping.items() if not compared[key]}
    check = all(compared.values())

    assert check, f"Not all values are str or None: {not_passing}"


@pytest.fixture
def example_unitless_da():
    array = np.linspace(0, 10, 20)
    x = np.arange(20)
    u = np.linspace(0, 1, 20)
    da = xr.DataArray(
        data=array,
        dims="x",
        coords={"x": ("x", x), "u": ("x", u, {"units": "hour"})},
        attrs={"units": "m"},
    )
    return da


@pytest.fixture()
def example_quantity_da():
    array = np.linspace(0, 10, 20) * unit_registry.m
    x = np.arange(20)
    u = np.linspace(0, 1, 20) * unit_registry.hour
    return xr.DataArray(data=array, dims="x", coords={"x": ("x", x), "u": ("x", u)})


class TestQuantifyDataArray:
    def test_attach_units_from_str(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.pint.quantify("m")
        assert_array_equal(result.data.magnitude, orig.data)
        # TODO better comparisons for when you can't access the unit_registry?
        assert str(result.data.units) == "meter"

    def test_attach_units_given_registry(self, example_unitless_da):
        orig = example_unitless_da
        ureg = UnitRegistry(force_ndarray=True)
        result = orig.pint.quantify("m", unit_registry=ureg)
        assert_array_equal(result.data.magnitude, orig.data)
        assert result.data.units == ureg.Unit("m")

    def test_attach_units_from_attrs(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.pint.quantify()
        assert_array_equal(result.data.magnitude, orig.data)
        assert str(result.data.units) == "meter"

        remaining_attrs = conversion.extract_unit_attributes(result)
        assert {k: v for k, v in remaining_attrs.items() if v is not None} == {}

    def test_attach_units_given_unit_objs(self, example_unitless_da):
        orig = example_unitless_da
        ureg = UnitRegistry(force_ndarray=True)
        result = orig.pint.quantify(ureg.Unit("m"), unit_registry=ureg)
        assert_array_equal(result.data.magnitude, orig.data)
        assert result.data.units == ureg.Unit("m")

    def test_error_when_already_units(self, example_quantity_da):
        da = example_quantity_da
        with raises_regex(ValueError, "already has units"):
            da.pint.quantify()

    def test_error_on_nonsense_units(self, example_unitless_da):
        da = example_unitless_da
        with pytest.raises(UndefinedUnitError):
            da.pint.quantify(units="aecjhbav")

    def test_parse_integer_inverse(self):
        # Regression test for issue #40
        da = xr.DataArray([10], attrs={"units": "m^-1"})
        result = da.pint.quantify()
        assert result.pint.units == Unit("1 / meter")


@pytest.mark.parametrize("formatter", ("", "P", "C"))
@pytest.mark.parametrize("flags", ("", "~", "#", "~#"))
def test_units_to_str_or_none(formatter, flags):
    unit_format = f"{{:{flags}{formatter}}}"
    unit_attrs = {None: "m", "a": "s", "b": "degC", "c": "degF", "d": "degK"}
    units = {key: unit_registry.Unit(value) for key, value in unit_attrs.items()}

    expected = {key: unit_format.format(value) for key, value in units.items()}
    actual = accessors.units_to_str_or_none(units, unit_format)

    assert expected == actual
    assert units == {key: unit_registry.Unit(value) for key, value in actual.items()}

    expected = {None: None}
    assert expected == accessors.units_to_str_or_none(expected, unit_format)


class TestDequantifyDataArray:
    def test_strip_units(self, example_quantity_da):
        result = example_quantity_da.pint.dequantify()
        assert isinstance(result.data, np.ndarray)
        assert isinstance(result.coords["x"].data, np.ndarray)

    def test_attrs_reinstated(self, example_quantity_da):
        da = example_quantity_da
        result = da.pint.dequantify()

        units = conversion.extract_units(da)
        attrs = conversion.extract_unit_attributes(result)

        assert units == attrs
        assert_all_str_or_none(attrs)

    def test_roundtrip_data(self, example_unitless_da):
        orig = example_unitless_da
        quantified = orig.pint.quantify()
        result = quantified.pint.dequantify()
        assert_equal(result, orig)


class TestPropertiesDataArray:
    def test_magnitude_getattr(self, example_quantity_da):
        da = example_quantity_da
        actual = da.pint.magnitude
        assert not isinstance(actual, Quantity)

    def test_magnitude_getattr_unitless(self, example_unitless_da):
        da = example_unitless_da
        xr.testing.assert_duckarray_equal(da.pint.magnitude, da.data)

    def test_units_getattr(self, example_quantity_da):
        da = example_quantity_da
        actual = da.pint.units
        assert isinstance(actual, Unit)
        assert actual == unit_registry.m

    def test_units_setattr(self, example_quantity_da):
        da = example_quantity_da
        with pytest.raises(ValueError):
            da.pint.units = "s"

    def test_units_getattr_unitless(self, example_unitless_da):
        da = example_unitless_da
        assert da.pint.units is None

    def test_units_setattr_unitless(self, example_unitless_da):
        da = example_unitless_da
        da.pint.units = unit_registry.s
        assert da.pint.units == unit_registry.s


@pytest.fixture()
def example_unitless_ds():
    users = np.linspace(0, 10, 20)
    funds = np.logspace(0, 10, 20)
    t = np.arange(20)
    ds = xr.Dataset(
        data_vars={"users": (["t"], users), "funds": (["t"], funds)}, coords={"t": t}
    )
    ds["users"].attrs["units"] = ""
    ds["funds"].attrs["units"] = "pound"
    return ds


@pytest.fixture()
def example_quantity_ds():
    users = np.linspace(0, 10, 20) * unit_registry.dimensionless
    funds = np.logspace(0, 10, 20) * unit_registry.pound
    t = np.arange(20)
    ds = xr.Dataset(
        data_vars={"users": (["t"], users), "funds": (["t"], funds)}, coords={"t": t}
    )
    return ds


class TestQuantifyDataSet:
    def test_attach_units_from_str(self, example_unitless_ds):
        orig = example_unitless_ds
        result = orig.pint.quantify()
        assert_array_equal(result["users"].data.magnitude, orig["users"].data)
        assert str(result["users"].data.units) == "dimensionless"

    def test_attach_units_given_registry(self, example_unitless_ds):
        orig = example_unitless_ds
        orig["users"].attrs.clear()
        result = orig.pint.quantify(
            {"users": "dimensionless"}, unit_registry=unit_registry
        )
        assert_array_equal(result["users"].data.magnitude, orig["users"].data)
        assert str(result["users"].data.units) == "dimensionless"

    def test_attach_units_from_attrs(self, example_unitless_ds):
        orig = example_unitless_ds
        orig["users"].attrs.clear()
        result = orig.pint.quantify({"users": "dimensionless"})
        assert_array_equal(result["users"].data.magnitude, orig["users"].data)
        assert str(result["users"].data.units) == "dimensionless"

        remaining_attrs = conversion.extract_unit_attributes(result)
        assert {k: v for k, v in remaining_attrs.items() if v is not None} == {}

    def test_attach_units_given_unit_objs(self, example_unitless_ds):
        orig = example_unitless_ds
        orig["users"].attrs.clear()
        dimensionless = unit_registry.Unit("dimensionless")
        result = orig.pint.quantify({"users": dimensionless})
        assert_array_equal(result["users"].data.magnitude, orig["users"].data)
        assert str(result["users"].data.units) == "dimensionless"

    def test_error_when_already_units(self, example_quantity_ds):
        with raises_regex(ValueError, "already has units"):
            example_quantity_ds.pint.quantify({"funds": "pounds"})

    def test_error_on_nonsense_units(self, example_unitless_ds):
        ds = example_unitless_ds
        with pytest.raises(UndefinedUnitError):
            ds.pint.quantify(units={"users": "aecjhbav"})


class TestDequantifyDataSet:
    def test_strip_units(self, example_quantity_ds):
        result = example_quantity_ds.pint.dequantify()

        assert all(
            isinstance(var.data, np.ndarray) for var in result.variables.values()
        )

    def test_attrs_reinstated(self, example_quantity_ds):
        ds = example_quantity_ds
        result = ds.pint.dequantify()

        units = conversion.extract_units(ds)
        # workaround for Unit("dimensionless") != str(Unit("dimensionless"))
        units = {
            key: str(value) if isinstance(value, Unit) else value
            for key, value in units.items()
        }

        attrs = conversion.extract_unit_attributes(result)

        assert units == attrs
        assert_all_str_or_none(attrs)

    def test_roundtrip_data(self, example_unitless_ds):
        orig = example_unitless_ds
        quantified = orig.pint.quantify()

        result = quantified.pint.dequantify()
        assert_equal(result, orig)

        result = quantified.pint.dequantify().pint.quantify()
        assert_equal(quantified, result)


@pytest.mark.parametrize(
    ["units_arg", "units_kwargs"],
    [
        ({"a": "g", "b": unit_registry.g}, {}),
        ("g", {}),
        ("g", {"a": "g"}),
        (None, {"a": "g", "b": unit_registry.g}),
    ],
)
def test_to_dataset(units_arg, units_kwargs):
    a = np.linspace(0, 10, 21) * unit_registry.kg
    b = np.linspace(0, 20, 21) * unit_registry.mg
    t = np.arange(21)
    ds = xr.Dataset(data_vars={"a": (["t"], a), "b": (["t"], b)}, coords={"t": t})

    ae = np.linspace(0, 10_000, 21) * unit_registry.g
    be = np.linspace(0, 20.0 / 1000, 21) * unit_registry.g
    expected = xr.Dataset(
        data_vars={"a": (["t"], ae), "b": (["t"], be)}, coords={"t": t}
    )

    actual = ds.pint.to(units_arg, **units_kwargs)
    assert_units_equal(actual, expected)
    assert_equal(expected, actual)


@pytest.mark.parametrize(
    ["obj", "indexers", "expected", "error"],
    (
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            xr.Dataset(
                {
                    "x": ("x", [10, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60], {"units": unit_registry.Unit("s")}),
                }
            ),
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            xr.Dataset(
                {
                    "x": ("x", [1, 3], {"units": unit_registry.Unit("m")}),
                    "y": ("y", [1], {"units": unit_registry.Unit("min")}),
                }
            ),
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            {"x": Quantity([1, 3], "s"), "y": Quantity([1], "m")},
            None,
            DimensionalityError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            xr.DataArray(
                [[0], [4]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60], {"units": unit_registry.Unit("s")}),
                },
            ),
            None,
            id="DataArray-identical units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            xr.DataArray(
                [[0], [4]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [1, 3], {"units": unit_registry.Unit("m")}),
                    "y": ("y", [1], {"units": unit_registry.Unit("min")}),
                },
            ),
            None,
            id="DataArray-compatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "s"), "y": Quantity([60], "m")},
            None,
            DimensionalityError,
            id="DataArray-incompatible units",
        ),
    ),
)
def test_sel(obj, indexers, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.sel(indexers)
    else:
        actual = obj.pint.sel(indexers)
        assert_units_equal(actual, expected)
        assert_identical(actual, expected)


@pytest.mark.parametrize(
    ["obj", "indexers", "expected", "error"],
    (
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            {"x": Quantity([10, 30, 50], "dm"), "y": Quantity([0, 120, 240], "s")},
            xr.Dataset(
                {
                    "x": ("x", [10, 30, 50], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [0, 120, 240], {"units": unit_registry.Unit("s")}),
                }
            ),
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            {"x": Quantity([0, 1, 3, 5], "m"), "y": Quantity([0, 2, 4], "min")},
            xr.Dataset(
                {
                    "x": ("x", [0, 1, 3, 5], {"units": unit_registry.Unit("m")}),
                    "y": ("y", [0, 2, 4], {"units": unit_registry.Unit("min")}),
                }
            ),
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            {"x": Quantity([1, 3], "s"), "y": Quantity([1], "m")},
            None,
            DimensionalityError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30, 50], "dm"), "y": Quantity([0, 240], "s")},
            xr.DataArray(
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 30, 50], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [0, 240], {"units": unit_registry.Unit("s")}),
                },
            ),
            None,
            id="DataArray-identical units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([1, 3, 5], "m"), "y": Quantity([0, 2], "min")},
            xr.DataArray(
                [[np.nan, 1], [np.nan, 5], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [1, 3, 5], {"units": unit_registry.Unit("m")}),
                    "y": ("y", [0, 2], {"units": unit_registry.Unit("min")}),
                },
            ),
            None,
            id="DataArray-compatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "s"), "y": Quantity([60], "m")},
            None,
            DimensionalityError,
            id="DataArray-incompatible units",
        ),
    ),
)
def test_reindex(obj, indexers, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.reindex(indexers)
    else:
        actual = obj.pint.reindex(indexers)
        assert_units_equal(actual, expected)
        assert_identical(actual, expected)


@pytest.mark.parametrize(
    ["obj", "other", "expected", "error"],
    (
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            xr.Dataset(
                {
                    "x": ("x", [10, 30, 50], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [0, 120, 240], {"units": unit_registry.Unit("s")}),
                }
            ),
            xr.Dataset(
                {
                    "x": ("x", [10, 30, 50], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [0, 120, 240], {"units": unit_registry.Unit("s")}),
                }
            ),
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            xr.Dataset(
                {
                    "x": ("x", [0, 1, 3, 5], {"units": unit_registry.Unit("m")}),
                    "y": ("y", [0, 2, 4], {"units": unit_registry.Unit("min")}),
                }
            ),
            xr.Dataset(
                {
                    "x": ("x", [0, 1, 3, 5], {"units": unit_registry.Unit("m")}),
                    "y": ("y", [0, 2, 4], {"units": unit_registry.Unit("min")}),
                }
            ),
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            xr.Dataset(
                {
                    "x": ("x", [1, 3], {"units": unit_registry.Unit("s")}),
                    "y": ("y", [1], {"units": unit_registry.Unit("m")}),
                }
            ),
            None,
            DimensionalityError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            xr.Dataset(
                {
                    "x": ("x", [10, 30, 50], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [0, 240], {"units": unit_registry.Unit("s")}),
                }
            ),
            xr.DataArray(
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 30, 50], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [0, 240], {"units": unit_registry.Unit("s")}),
                },
            ),
            None,
            id="DataArray-identical units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            xr.Dataset(
                {
                    "x": ("x", [1, 3, 5], {"units": unit_registry.Unit("m")}),
                    "y": ("y", [0, 2], {"units": unit_registry.Unit("min")}),
                }
            ),
            xr.DataArray(
                [[np.nan, 1], [np.nan, 5], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [1, 3, 5], {"units": unit_registry.Unit("m")}),
                    "y": ("y", [0, 2], {"units": unit_registry.Unit("min")}),
                },
            ),
            None,
            id="DataArray-compatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            xr.Dataset(
                {
                    "x": ("x", [10, 30], {"units": unit_registry.Unit("s")}),
                    "y": ("y", [60], {"units": unit_registry.Unit("m")}),
                }
            ),
            None,
            DimensionalityError,
            id="DataArray-incompatible units",
        ),
    ),
)
def test_reindex_like(obj, other, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.reindex_like(other)
    else:
        actual = obj.pint.reindex_like(other)
        assert_units_equal(actual, expected)
        assert_identical(actual, expected)
