import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from pint import Unit, UnitRegistry

from .. import accessors, conversion
from .utils import (
    assert_equal,
    assert_identical,
    assert_units_equal,
    requires_bottleneck,
    requires_dask_array,
    requires_scipy,
)

pytestmark = [
    pytest.mark.filterwarnings("error::pint.UnitStrippedWarning"),
]

# make sure scalars are converted to 0d arrays so quantities can
# always be treated like ndarrays
unit_registry = UnitRegistry(force_ndarray=True)
Quantity = unit_registry.Quantity

nan = np.nan


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
        result = orig.pint.quantify("s")
        assert_array_equal(result.data.magnitude, orig.data)
        # TODO better comparisons for when you can't access the unit_registry?
        assert str(result.data.units) == "second"

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

    def test_attach_units_from_str_attr_no_unit(self, example_unitless_da):
        orig = example_unitless_da
        orig.attrs["units"] = "none"
        result = orig.pint.quantify("m")
        assert_array_equal(result.data.magnitude, orig.data)
        assert str(result.data.units) == "meter"

    def test_attach_units_given_unit_objs(self, example_unitless_da):
        orig = example_unitless_da
        ureg = UnitRegistry(force_ndarray=True)
        result = orig.pint.quantify(ureg.Unit("m"), unit_registry=ureg)
        assert_array_equal(result.data.magnitude, orig.data)
        assert result.data.units == ureg.Unit("m")

    @pytest.mark.parametrize("no_unit_value", conversion.no_unit_values)
    def test_override_units(self, example_unitless_da, no_unit_value):
        orig = example_unitless_da
        result = orig.pint.quantify(no_unit_value, u=no_unit_value)

        with pytest.raises(AttributeError):
            result.data.units
        with pytest.raises(AttributeError):
            result["u"].data.units

    def test_error_when_changing_units(self, example_quantity_da):
        da = example_quantity_da
        with pytest.raises(ValueError, match="already has units"):
            da.pint.quantify("s")

    def test_attach_no_units(self):
        arr = xr.DataArray([1, 2, 3], dims="x")
        quantified = arr.pint.quantify()
        assert_identical(quantified, arr)
        assert_units_equal(quantified, arr)

    def test_attach_no_new_units(self):
        da = xr.DataArray(unit_registry.Quantity([1, 2, 3], "m"), dims="x")
        quantified = da.pint.quantify()
        assert_identical(quantified, da)
        assert_units_equal(quantified, da)

    def test_attach_same_units(self):
        da = xr.DataArray(unit_registry.Quantity([1, 2, 3], "m"), dims="x")
        quantified = da.pint.quantify("m")
        assert_identical(quantified, da)
        assert_units_equal(quantified, da)

    def test_error_when_changing_units_dimension_coordinates(self):
        arr = xr.DataArray(
            [1, 2, 3],
            dims="x",
            coords={"x": ("x", [-1, 0, 1], {"units": unit_registry.Unit("m")})},
        )
        with pytest.raises(ValueError, match="already has units"):
            arr.pint.quantify({"x": "s"})

    def test_dimension_coordinate_array(self):
        ds = xr.Dataset(coords={"x": ("x", [10], {"units": "m"})})
        arr = ds.x

        # does not actually quantify because `arr` wraps a IndexVariable
        # but we still get a `Unit` in the attrs
        q = arr.pint.quantify()
        assert isinstance(q.attrs["units"], Unit)

    def test_dimension_coordinate_array_already_quantified(self):
        ds = xr.Dataset(coords={"x": ("x", [10], {"units": unit_registry.Unit("m")})})
        arr = ds.x

        with pytest.raises(ValueError):
            arr.pint.quantify({"x": "s"})

    def test_dimension_coordinate_array_already_quantified_same_units(self):
        ds = xr.Dataset(coords={"x": ("x", [10], {"units": unit_registry.Unit("m")})})
        arr = ds.x

        quantified = arr.pint.quantify({"x": "m"})

        assert_identical(quantified, arr)
        assert_units_equal(quantified, arr)

    def test_error_on_nonsense_units(self, example_unitless_da):
        da = example_unitless_da
        with pytest.raises(ValueError, match=str(da.name)):
            da.pint.quantify(units="aecjhbav")

    def test_error_on_nonsense_units_attrs(self, example_unitless_da):
        da = example_unitless_da
        da.attrs["units"] = "aecjhbav"
        with pytest.raises(
            ValueError, match=rf"{da.name}: {da.attrs['units']} \(attribute\)"
        ):
            da.pint.quantify()

    def test_parse_integer_inverse(self):
        # Regression test for issue #40
        da = xr.DataArray([10], attrs={"units": "m^-1"})
        result = da.pint.quantify()
        assert result.pint.units == Unit("1 / meter")


@pytest.mark.parametrize("formatter", ("", "P", "C"))
@pytest.mark.parametrize("modifier", ("", "~"))
def test_units_to_str_or_none(formatter, modifier):
    unit_format = f"{{:{modifier}{formatter}}}"
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

    def test_multiindex(self):
        mindex = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("lat", "lon"))

        da = xr.DataArray(
            np.arange(len(mindex)), dims="multi", coords={"multi": mindex}
        )
        result = da.pint.dequantify()

        xr.testing.assert_identical(da, result)
        assert isinstance(result.indexes["multi"], pd.MultiIndex)


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

    def test_attach_units_from_str_attr_no_unit(self, example_unitless_ds):
        orig = example_unitless_ds
        orig["users"].attrs["units"] = "none"
        result = orig.pint.quantify({"users": "m"})
        assert_array_equal(result["users"].data.magnitude, orig["users"].data)
        assert str(result["users"].data.units) == "meter"

    @pytest.mark.parametrize("no_unit_value", conversion.no_unit_values)
    def test_override_units(self, example_unitless_ds, no_unit_value):
        orig = example_unitless_ds
        result = orig.pint.quantify({"users": no_unit_value})
        assert (
            getattr(result["users"].data, "units", "not a quantity") == "not a quantity"
        )

    def test_error_when_already_units(self, example_quantity_ds):
        with pytest.raises(ValueError, match="already has units"):
            example_quantity_ds.pint.quantify({"funds": "kg"})

    def test_attach_no_units(self):
        ds = xr.Dataset({"a": ("x", [1, 2, 3])})
        quantified = ds.pint.quantify()
        assert_identical(quantified, ds)
        assert_units_equal(quantified, ds)

    def test_attach_no_new_units(self):
        ds = xr.Dataset({"a": ("x", unit_registry.Quantity([1, 2, 3], "m"))})
        quantified = ds.pint.quantify()

        assert_identical(quantified, ds)
        assert_units_equal(quantified, ds)

    def test_attach_same_units(self):
        ds = xr.Dataset({"a": ("x", unit_registry.Quantity([1, 2, 3], "m"))})
        quantified = ds.pint.quantify({"a": "m"})

        assert_identical(quantified, ds)
        assert_units_equal(quantified, ds)

    def test_error_when_changing_units_dimension_coordinates(self):
        ds = xr.Dataset(
            coords={"x": ("x", [-1, 0, 1], {"units": unit_registry.Unit("m")})},
        )
        with pytest.raises(ValueError, match="already has units"):
            ds.pint.quantify({"x": "s"})

    def test_error_on_nonsense_units(self, example_unitless_ds):
        ds = example_unitless_ds
        with pytest.raises(ValueError):
            ds.pint.quantify(units={"users": "aecjhbav"})

    def test_error_on_nonsense_units_attrs(self, example_unitless_ds):
        ds = example_unitless_ds
        ds.users.attrs["units"] = "aecjhbav"
        with pytest.raises(
            ValueError, match=rf"'users': {ds.users.attrs['units']} \(attribute\)"
        ):
            ds.pint.quantify()

    def test_error_indicates_problematic_variable(self, example_unitless_ds):
        ds = example_unitless_ds
        with pytest.raises(ValueError, match="'users'"):
            ds.pint.quantify(units={"users": "aecjhbav"})

    def test_existing_units(self, example_quantity_ds):
        ds = example_quantity_ds.copy()
        ds.t.attrs["units"] = unit_registry.Unit("m")

        with pytest.raises(ValueError, match="Cannot attach"):
            ds.pint.quantify({"funds": "kg"})

    def test_existing_units_dimension(self, example_quantity_ds):
        ds = example_quantity_ds.copy()
        ds.t.attrs["units"] = unit_registry.Unit("m")

        with pytest.raises(ValueError, match="Cannot attach"):
            ds.pint.quantify({"t": "s"})


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
    ["obj", "units", "expected", "error"],
    (
        pytest.param(
            xr.Dataset(
                {"a": ("x", Quantity([0, 1], "m")), "b": ("x", Quantity([2, 4], "s"))}
            ),
            {"a": "mm", "b": "ms"},
            xr.Dataset(
                {
                    "a": ("x", Quantity([0, 1000], "mm")),
                    "b": ("x", Quantity([2000, 4000], "ms")),
                }
            ),
            None,
            id="Dataset-compatible units-data",
        ),
        pytest.param(
            xr.Dataset(
                {"a": ("x", Quantity([0, 1], "km")), "b": ("x", Quantity([2, 4], "cm"))}
            ),
            "m",
            xr.Dataset(
                {
                    "a": ("x", Quantity([0, 1000], "m")),
                    "b": ("x", Quantity([0.02, 0.04], "m")),
                }
            ),
            None,
            id="Dataset-compatible units-data-str",
        ),
        pytest.param(
            xr.Dataset(
                {"a": ("x", Quantity([0, 1], "m")), "b": ("x", Quantity([2, 4], "s"))}
            ),
            {"a": "ms", "b": "mm"},
            None,
            ValueError,
            id="Dataset-incompatible units-data",
        ),
        pytest.param(
            xr.Dataset(coords={"x": ("x", [2, 4], {"units": Unit("s")})}),
            {"x": "ms"},
            xr.Dataset(coords={"x": ("x", [2000, 4000], {"units": Unit("ms")})}),
            None,
            id="Dataset-compatible units-dims",
        ),
        pytest.param(
            xr.Dataset(coords={"x": ("x", [2, 4], {"units": Unit("s")})}),
            {"x": "mm"},
            None,
            ValueError,
            id="Dataset-incompatible units-dims",
        ),
        pytest.param(
            xr.DataArray(Quantity([0, 1], "m"), dims="x"),
            {None: "mm"},
            xr.DataArray(Quantity([0, 1000], "mm"), dims="x"),
            None,
            id="DataArray-compatible units-data",
        ),
        pytest.param(
            xr.DataArray(Quantity([0, 1], "m"), dims="x"),
            "mm",
            xr.DataArray(Quantity([0, 1000], "mm"), dims="x"),
            None,
            id="DataArray-compatible units-data-str",
        ),
        pytest.param(
            xr.DataArray(Quantity([0, 1], "m"), dims="x", name="a"),
            {"a": "mm"},
            xr.DataArray(Quantity([0, 1000], "mm"), dims="x", name="a"),
            None,
            id="DataArray-compatible units-data-by name",
        ),
        pytest.param(
            xr.DataArray(Quantity([0, 1], "m"), dims="x"),
            {None: "ms"},
            None,
            ValueError,
            id="DataArray-incompatible units-data",
        ),
        pytest.param(
            xr.DataArray(
                [0, 1], dims="x", coords={"x": ("x", [2, 4], {"units": Unit("s")})}
            ),
            {"x": "ms"},
            xr.DataArray(
                [0, 1],
                dims="x",
                coords={"x": ("x", [2000, 4000], {"units": Unit("ms")})},
            ),
            None,
            id="DataArray-compatible units-dims",
        ),
        pytest.param(
            xr.DataArray(
                [0, 1], dims="x", coords={"x": ("x", [2, 4], {"units": Unit("s")})}
            ),
            {"x": "mm"},
            None,
            ValueError,
            id="DataArray-incompatible units-dims",
        ),
    ),
)
def test_to(obj, units, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.to(units)
    else:
        actual = obj.pint.to(units)

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
            KeyError,
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
            KeyError,
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
            KeyError,
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
            KeyError,
            id="DataArray-incompatible units",
        ),
    ),
)
def test_loc(obj, indexers, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.loc[indexers]
    else:
        actual = obj.pint.loc[indexers]
        assert_units_equal(actual, expected)
        assert_identical(actual, expected)


@pytest.mark.parametrize(
    ["obj", "indexers", "values", "expected", "error"],
    (
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
            [[-1], [-2]],
            xr.DataArray(
                [[-1, 1], [2, 3], [-2, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            None,
            id="coords-identical units",
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
            [[-1], [-2]],
            xr.DataArray(
                [[-1, 1], [2, 3], [-2, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            None,
            id="coords-compatible units",
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
            {"x": Quantity([1, 3], "s"), "y": Quantity([1], "m")},
            [[-1], [-2]],
            None,
            KeyError,
            id="coords-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity([[0, 1], [2, 3], [4, 5]], "m"),
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            Quantity([[-1], [-2]], "m"),
            xr.DataArray(
                Quantity([[-1, 1], [2, 3], [-2, 5]], "m"),
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            None,
            id="data-identical units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity([[0, 1], [2, 3], [4, 5]], "m"),
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            Quantity([[-1], [-2]], "km"),
            xr.DataArray(
                Quantity([[-1000, 1], [2, 3], [-2000, 5]], "m"),
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            None,
            id="data-compatible units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity([[0, 1], [2, 3], [4, 5]], "m"),
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            Quantity([[-1], [-2]], "s"),
            None,
            pint.DimensionalityError,
            id="data-incompatible units",
        ),
    ),
)
def test_loc_setitem(obj, indexers, values, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.loc[indexers] = values
    else:
        obj.pint.loc[indexers] = values
        assert_units_equal(obj, expected)
        assert_identical(obj, expected)


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
                    "x": ("x", [20], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [120], {"units": unit_registry.Unit("s")}),
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
                    "x": ("x", [20], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [120], {"units": unit_registry.Unit("s")}),
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
            KeyError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": unit_registry.Unit("s")}),
                }
            ),
            {"x": Quantity([10, 30], "m"), "y": Quantity([60], "min")},
            None,
            KeyError,
            id="Dataset-compatible units-not found",
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
                [[3]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [20], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [120], {"units": unit_registry.Unit("s")}),
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
                [[3]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [20], {"units": unit_registry.Unit("dm")}),
                    "y": ("y", [120], {"units": unit_registry.Unit("s")}),
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
            KeyError,
            id="DataArray-incompatible units",
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
            {"x": Quantity([10, 30], "m"), "y": Quantity([60], "min")},
            None,
            KeyError,
            id="DataArray-compatible units-not found",
        ),
    ),
)
def test_drop_sel(obj, indexers, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.drop_sel(indexers)
    else:
        actual = obj.pint.drop_sel(indexers)
        assert_units_equal(actual, expected)
        assert_identical(actual, expected)


@requires_dask_array
@pytest.mark.parametrize(
    "obj",
    (
        pytest.param(
            xr.Dataset(
                {"a": ("x", np.linspace(0, 1, 11))},
                coords={"u": ("x", np.arange(11))},
            ),
            id="Dataset-no units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (
                        "x",
                        Quantity(np.linspace(0, 1, 11), "m"),
                    )
                },
                coords={
                    "u": (
                        "x",
                        Quantity(np.arange(11), "m"),
                    )
                },
            ),
            id="Dataset-units",
        ),
        pytest.param(
            xr.DataArray(
                np.linspace(0, 1, 11),
                coords={
                    "u": (
                        "x",
                        np.arange(11),
                    )
                },
                dims="x",
            ),
            id="DataArray-no units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity(np.linspace(0, 1, 11), "m"),
                coords={
                    "u": (
                        "x",
                        Quantity(np.arange(11), "m"),
                    )
                },
                dims="x",
            ),
            id="DataArray-units",
        ),
    ),
)
def test_chunk(obj):
    actual = obj.pint.chunk({"x": 2})

    expected = (
        obj.pint.dequantify().chunk({"x": 2}).pint.quantify(unit_registry=unit_registry)
    )

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
            ValueError,
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
            ValueError,
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
            ValueError,
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
            ValueError,
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


@requires_scipy
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
            ValueError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (("x", "y"), Quantity([[0, 1], [2, 3], [4, 5]], "kg")),
                    "x": [10, 20, 30],
                    "y": [60, 120],
                }
            ),
            {
                "x": [15, 25],
                "y": [75, 105],
            },
            xr.Dataset(
                {
                    "a": (("x", "y"), Quantity([[1.25, 1.75], [3.25, 3.75]], "kg")),
                    "x": [15, 25],
                    "y": [75, 105],
                }
            ),
            None,
            id="Dataset-data units",
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
            ValueError,
            id="DataArray-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity([[0, 1], [2, 3], [4, 5]], "kg"),
                dims=("x", "y"),
                coords={
                    "x": [10, 20, 30],
                    "y": [60, 120],
                },
            ),
            {
                "x": [15, 25],
                "y": [75, 105],
            },
            xr.DataArray(
                Quantity([[1.25, 1.75], [3.25, 3.75]], "kg"),
                dims=("x", "y"),
                coords={
                    "x": [15, 25],
                    "y": [75, 105],
                },
            ),
            None,
            id="DataArray-data units",
        ),
    ),
)
def test_interp(obj, indexers, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.interp(indexers)
    else:
        actual = obj.pint.interp(indexers)
        assert_units_equal(actual, expected)
        assert_identical(actual, expected)


@requires_scipy
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
            ValueError,
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
            xr.Dataset(
                {
                    "a": (("x", "y"), Quantity([[0, 1], [2, 3], [4, 5]], "kg")),
                    "x": [10, 20, 30],
                    "y": [60, 120],
                }
            ),
            xr.Dataset(
                {
                    "x": [15, 25],
                    "y": [75, 105],
                }
            ),
            xr.Dataset(
                {
                    "a": (("x", "y"), Quantity([[1.25, 1.75], [3.25, 3.75]], "kg")),
                    "x": [15, 25],
                    "y": [75, 105],
                }
            ),
            None,
            id="Dataset-data units",
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
            ValueError,
            id="DataArray-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity([[0, 1], [2, 3], [4, 5]], "kg"),
                dims=("x", "y"),
                coords={
                    "x": [10, 20, 30],
                    "y": [60, 120],
                },
            ),
            xr.Dataset(
                {
                    "x": [15, 25],
                    "y": [75, 105],
                }
            ),
            xr.DataArray(
                Quantity([[1.25, 1.75], [3.25, 3.75]], "kg"),
                dims=("x", "y"),
                coords={
                    "x": [15, 25],
                    "y": [75, 105],
                },
            ),
            None,
            id="DataArray-data units",
        ),
    ),
)
def test_interp_like(obj, other, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.pint.interp_like(other)
    else:
        actual = obj.pint.interp_like(other)
        assert_units_equal(actual, expected)
        assert_identical(actual, expected)


@requires_bottleneck
@pytest.mark.parametrize(
    ["obj", "expected"],
    (
        pytest.param(
            xr.Dataset(
                {"a": ("x", [nan, 0, nan, 1, nan, nan, 2, nan])},
                coords={"u": ("x", [nan, 0, nan, 1, nan, nan, 2, nan])},
            ),
            xr.Dataset(
                {"a": ("x", [nan, 0, 0, 1, 1, 1, 2, 2])},
                coords={"u": ("x", [nan, 0, nan, 1, nan, nan, 2, nan])},
            ),
            id="Dataset-no units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
            ),
            xr.Dataset(
                {"a": ("x", Quantity([nan, 0, 0, 1, 1, 1, 2, 2], "m"))},
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
            ),
            id="Dataset-units",
        ),
        pytest.param(
            xr.DataArray(
                [nan, 0, nan, 1, nan, nan, 2, nan],
                coords={
                    "u": (
                        "x",
                        [nan, 0, nan, 1, nan, nan, 2, nan],
                    )
                },
                dims="x",
            ),
            xr.DataArray(
                [nan, 0, 0, 1, 1, 1, 2, 2],
                coords={
                    "u": (
                        "x",
                        [nan, 0, nan, 1, nan, nan, 2, nan],
                    )
                },
                dims="x",
            ),
            id="DataArray-no units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
                dims="x",
            ),
            xr.DataArray(
                Quantity([nan, 0, 0, 1, 1, 1, 2, 2], "m"),
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
                dims="x",
            ),
            id="DataArray-units",
        ),
    ),
)
def test_ffill(obj, expected):
    actual = obj.pint.ffill(dim="x")
    assert_identical(actual, expected)
    assert_units_equal(actual, expected)


@requires_bottleneck
@pytest.mark.parametrize(
    ["obj", "expected"],
    (
        pytest.param(
            xr.Dataset(
                {"a": ("x", [nan, 0, nan, 1, nan, nan, 2, nan])},
                coords={"u": ("x", [nan, 0, nan, 1, nan, nan, 2, nan])},
            ),
            xr.Dataset(
                {"a": ("x", [0, 0, 1, 1, 2, 2, 2, nan])},
                coords={"u": ("x", [nan, 0, nan, 1, nan, nan, 2, nan])},
            ),
            id="Dataset-no units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
            ),
            xr.Dataset(
                {"a": ("x", Quantity([0, 0, 1, 1, 2, 2, 2, nan], "m"))},
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
            ),
            id="Dataset-units",
        ),
        pytest.param(
            xr.DataArray(
                [nan, 0, nan, 1, nan, nan, 2, nan],
                coords={
                    "u": (
                        "x",
                        [nan, 0, nan, 1, nan, nan, 2, nan],
                    )
                },
                dims="x",
            ),
            xr.DataArray(
                [0, 0, 1, 1, 2, 2, 2, nan],
                coords={
                    "u": (
                        "x",
                        [nan, 0, nan, 1, nan, nan, 2, nan],
                    )
                },
                dims="x",
            ),
            id="DataArray-no units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
                dims="x",
            ),
            xr.DataArray(
                Quantity([0, 0, 1, 1, 2, 2, 2, nan], "m"),
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, 2, nan], "m"),
                    )
                },
                dims="x",
            ),
            id="DataArray-units",
        ),
    ),
)
def test_bfill(obj, expected):
    actual = obj.pint.bfill(dim="x")
    assert_identical(actual, expected)
    assert_units_equal(actual, expected)


@pytest.mark.parametrize(
    ["obj", "expected"],
    (
        pytest.param(
            xr.Dataset(
                {"a": ("x", [nan, 0, nan, 1, nan, nan, nan, 2, nan])},
                coords={"u": ("x", [nan, 0, nan, 1, nan, nan, nan, 2, nan])},
            ),
            xr.Dataset(
                {"a": ("x", [nan, 0, 0.5, 1, 1.25, 1.5, 1.75, 2, nan])},
                coords={"u": ("x", [nan, 0, nan, 1, nan, nan, nan, 2, nan])},
            ),
            id="Dataset-no units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, nan, 2, nan], "m"),
                    )
                },
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, nan, 2, nan], "m"),
                    )
                },
            ),
            xr.Dataset(
                {"a": ("x", Quantity([nan, 0, 0.5, 1, 1.25, 1.5, 1.75, 2, nan], "m"))},
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, nan, 2, nan], "m"),
                    )
                },
            ),
            id="Dataset-units",
        ),
        pytest.param(
            xr.DataArray(
                [nan, 0, nan, 1, nan, nan, nan, 2, nan],
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, nan, 2, nan], "m"),
                    )
                },
                dims="x",
            ),
            xr.DataArray(
                [nan, 0, 0.5, 1, 1.25, 1.5, 1.75, 2, nan],
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, nan, 2, nan], "m"),
                    )
                },
                dims="x",
            ),
            id="DataArray-units",
        ),
        pytest.param(
            xr.DataArray(
                Quantity([nan, 0, nan, 1, nan, nan, nan, 2, nan], "m"),
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, nan, 2, nan], "m"),
                    )
                },
                dims="x",
            ),
            xr.DataArray(
                Quantity([nan, 0, 0.5, 1, 1.25, 1.5, 1.75, 2, nan], "m"),
                coords={
                    "u": (
                        "x",
                        Quantity([nan, 0, nan, 1, nan, nan, nan, 2, nan], "m"),
                    )
                },
                dims="x",
            ),
            id="DataArray-units",
        ),
    ),
)
def test_interpolate_na(obj, expected):
    actual = obj.pint.interpolate_na(dim="x")
    assert_identical(actual, expected)
    assert_units_equal(actual, expected)
