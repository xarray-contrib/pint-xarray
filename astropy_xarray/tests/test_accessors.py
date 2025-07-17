# make sure scalars are converted to 0d arrays so quantities can
# always be treated like ndarrays
import astropy.units as u
import astropy.units.core
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy.units import Quantity, Unit, UnitBase
from numpy.testing import assert_array_equal

from astropy_xarray import accessors, conversion
from astropy_xarray.index import AstropyIndex
from astropy_xarray.tests.utils import (
    assert_equal,
    assert_identical,
    assert_units_equal,
    requires_bottleneck,
    requires_dask_array,
    requires_scipy,
)

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
    y = np.linspace(0, 1, 20)
    da = xr.DataArray(
        data=array,
        dims="x",
        coords={"x": ("x", x), "u": ("x", y, {"units": "hour"})},
        attrs={"units": "m"},
    )
    return da


@pytest.fixture()
def example_quantity_da():
    array = np.linspace(0, 10, 20) * u.m
    x = np.arange(20)
    y = np.linspace(0, 1, 20) * u.hour
    return xr.DataArray(data=array, dims="x", coords={"x": ("x", x), "u": ("x", y)})


class TestQuantifyDataArray:
    def test_attach_units_from_str(self, example_unitless_da: xr.DataArray):
        orig = example_unitless_da
        result = orig.astropy.quantify("s")
        assert_array_equal(result.data.value, orig.data)
        # TODO better comparisons for when you can't access the u?
        assert str(result.data.unit) == "s"

    def test_attach_units_given_registry(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.astropy.quantify("m")
        assert_array_equal(result.data.value, orig.data)
        assert result.data.unit == u.Unit("m")

    def test_attach_units_from_attrs(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.astropy.quantify()
        assert_array_equal(result.data.value, orig.data)
        assert str(result.data.unit) == "m"

        remaining_attrs = conversion.extract_unit_attributes(result)
        assert {k: v for k, v in remaining_attrs.items() if v is not None} == {}

    def test_attach_units_from_str_attr_no_unit(self, example_unitless_da):
        orig = example_unitless_da
        orig.attrs["units"] = "none"
        result = orig.astropy.quantify("m")
        assert_array_equal(result.data.value, orig.data)
        assert str(result.data.unit) == "m"

    def test_attach_units_given_unit_objs(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.astropy.quantify(u.Unit("m"))
        assert_array_equal(result.data.value, orig.data)
        assert result.data.unit == u.Unit("m")

    @pytest.mark.parametrize("no_unit_value", conversion.no_unit_values)
    def test_override_units(self, example_unitless_da, no_unit_value):
        orig = example_unitless_da
        result = orig.astropy.quantify(no_unit_value, u=no_unit_value)

        with pytest.raises(AttributeError):
            result.data.unit
        with pytest.raises(AttributeError):
            result["u"].data.unit

    def test_error_when_changing_units(self, example_quantity_da):
        da = example_quantity_da
        with pytest.raises(ValueError, match="already has units"):
            da.astropy.quantify("s")

    def test_attach_no_units(self):
        arr = xr.DataArray([1, 2, 3], dims="x")
        quantified = arr.astropy.quantify()
        assert_identical(quantified, arr)
        assert_units_equal(quantified, arr)

    def test_attach_no_new_units(self):
        da = xr.DataArray(u.Quantity([1, 2, 3], "m"), dims="x")
        quantified = da.astropy.quantify()
        assert_identical(quantified, da)
        assert_units_equal(quantified, da)

    def test_attach_same_units(self):
        da = xr.DataArray(u.Quantity([1, 2, 3], "m"), dims="x")
        quantified = da.astropy.quantify("m")
        assert_identical(quantified, da)
        assert_units_equal(quantified, da)

    def test_error_when_changing_units_dimension_coordinates(self):
        arr = xr.DataArray(
            [1, 2, 3],
            dims="x",
            coords={"x": ("x", [-1, 0, 1], {"units": u.Unit("m")})},
        )
        with pytest.raises(ValueError, match="already has units"):
            arr.astropy.quantify({"x": "s"})

    def test_dimension_coordinate_array(self):
        ds = xr.Dataset(coords={"x": ("x", [10], {"units": "m"})})
        arr = ds.x

        # does not actually quantify because `arr` wraps a IndexVariable
        # but we still get a `Unit` in the attrs
        q = arr.astropy.quantify()
        assert isinstance(q.attrs["units"], UnitBase)

    def test_dimension_coordinate_array_already_quantified(self):
        ds = xr.Dataset(coords={"x": ("x", [10], {"units": u.Unit("m")})})
        arr = ds.x

        with pytest.raises(ValueError):
            arr.astropy.quantify({"x": "s"})

    def test_dimension_coordinate_array_already_quantified_same_units(self):
        x = u.Quantity([10], "m")
        coords = xr.Coordinates(
            {"x": x},
            indexes={
                "x": AstropyIndex.from_variables(
                    {"x": xr.Variable("x", x.value)},
                    options={"units": x.unit},
                ),
            },
        )
        ds = xr.Dataset(coords=coords)
        arr = ds.x

        quantified = arr.astropy.quantify({"x": "m"})

        assert_identical(quantified, arr)
        assert_units_equal(quantified, arr)

    def test_error_on_nonsense_units(self, example_unitless_da):
        da = example_unitless_da
        with pytest.raises(ValueError, match=str(da.name)):
            da.astropy.quantify(units="aecjhbav")

    def test_error_on_nonsense_units_attrs(self, example_unitless_da):
        da = example_unitless_da
        da.attrs["units"] = "aecjhbav"
        with pytest.raises(
            ValueError, match=rf"{da.name}: {da.attrs['units']} \(attribute\)"
        ):
            da.astropy.quantify()

    def test_parse_integer_inverse(self):
        # Regression test for issue #40
        da = xr.DataArray([10], attrs={"units": "m^-1"})
        result = da.astropy.quantify()
        assert result.astropy.unit == Unit("1 / meter")


@pytest.mark.parametrize(
    "unit_attrs,formatters",
    [
        pytest.param(
            {None: "m", "a": "s", "d": "Kelvin"},
            ("", "fits", "vounit", "cds", "ogip", "generic", "unicode", "console"),
            id="si",
        ),
        pytest.param(
            {None: "m", "a": "s", "b": "Celsius", "d": "Kelvin"},
            ("", "fits", "generic", "unicode", "console"),
            id="metric",
        ),
        pytest.param(
            {None: "m", "a": "s", "b": "Celsius", "c": "Fahrenheit", "d": "Kelvin"},
            ("", "generic", "unicode", "console"),
            id="mixed",
        ),
    ],
)
def test_units_to_str_or_none(unit_attrs, formatters):
    import astropy.units.imperial

    astropy.units.imperial.enable()

    units = {key: u.Unit(value) for key, value in unit_attrs.items()}
    for formatter in formatters:
        unit_format = f"{{:{formatter}}}"

        expected = {key: unit_format.format(value) for key, value in units.items()}
        actual = accessors.units_to_str_or_none(units, unit_format)

        assert expected == actual
        assert units == {key: u.Unit(value) for key, value in actual.items()}

        expected = {None: None}
        assert expected == accessors.units_to_str_or_none(expected, unit_format)


class TestDequantifyDataArray:
    def test_strip_units(self, example_quantity_da):
        result = example_quantity_da.astropy.dequantify()
        assert isinstance(result.data, np.ndarray)
        assert isinstance(result.coords["x"].data, np.ndarray)

    def test_attrs_reinstated(self, example_quantity_da):
        da = example_quantity_da
        result = da.astropy.dequantify()

        units = conversion.extract_units(da)
        attrs = conversion.extract_unit_attributes(result)

        assert units == attrs
        assert_all_str_or_none(attrs)

    def test_roundtrip_data(self, example_unitless_da):
        orig = example_unitless_da
        quantified = orig.astropy.quantify()
        result = quantified.astropy.dequantify()
        assert_equal(result, orig)

    def test_multiindex(self):
        mindex = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("lat", "lon"))

        da = xr.DataArray(
            np.arange(len(mindex)), dims="multi", coords={"multi": mindex}
        )
        result = da.astropy.dequantify()

        xr.testing.assert_identical(da, result)
        assert isinstance(result.indexes["multi"], pd.MultiIndex)


class TestPropertiesDataArray:
    def test_value_getattr(self, example_quantity_da):
        da = example_quantity_da
        actual = da.astropy.value
        assert not isinstance(actual, Quantity)

    def test_value_getattr_unitless(self, example_unitless_da):
        da = example_unitless_da
        xr.testing.assert_duckarray_equal(da.astropy.value, da.data)

    def test_units_getattr(self, example_quantity_da):
        da = example_quantity_da
        actual = da.astropy.unit
        assert isinstance(actual, u.UnitBase)
        assert actual == u.m

    def test_units_setattr(self, example_quantity_da):
        da = example_quantity_da
        with pytest.raises(ValueError):
            da.astropy.unit = "s"

    def test_units_getattr_unitless(self, example_unitless_da):
        da = example_unitless_da
        assert da.astropy.unit is None

    def test_units_setattr_unitless(self, example_unitless_da):
        da = example_unitless_da
        da.astropy.unit = u.s
        assert da.astropy.unit == u.s


@pytest.fixture()
def example_unitless_ds():
    users = np.linspace(0, 10, 20)
    funds = np.logspace(0, 10, 20)
    t = np.arange(20)
    ds = xr.Dataset(
        data_vars={"users": (["t"], users), "funds": (["t"], funds)}, coords={"t": t}
    )
    ds["users"].attrs["units"] = ""
    ds["funds"].attrs["units"] = "kilogram"
    return ds


@pytest.fixture()
def example_quantity_ds():
    users = np.linspace(0, 10, 20) * u.dimensionless_unscaled
    funds = np.logspace(0, 10, 20) * u.gram
    t = np.arange(20)
    ds = xr.Dataset(
        data_vars={"users": (["t"], users), "funds": (["t"], funds)}, coords={"t": t}
    )
    return ds


class TestQuantifyDataSet:
    def test_attach_units_from_str(self, example_unitless_ds):
        orig = example_unitless_ds
        result = orig.astropy.quantify()
        assert_array_equal(result["users"].data.value, orig["users"].data)
        assert str(result["users"].data.unit) == ""

    def test_attach_units_given_registry(self, example_unitless_ds):
        orig = example_unitless_ds
        orig["users"].attrs.clear()
        result = orig.astropy.quantify(
            {"users": ""},
        )
        assert_array_equal(result["users"].data.value, orig["users"].data)
        assert str(result["users"].data.unit) == ""

    def test_attach_units_from_attrs(self, example_unitless_ds):
        orig = example_unitless_ds
        orig["users"].attrs.clear()
        result = orig.astropy.quantify({"users": ""})
        assert_array_equal(result["users"].data.value, orig["users"].data)
        assert str(result["users"].data.unit) == ""

        remaining_attrs = conversion.extract_unit_attributes(result)
        assert {k: v for k, v in remaining_attrs.items() if v is not None} == {}

    def test_attach_units_given_unit_objs(self, example_unitless_ds):
        orig = example_unitless_ds
        orig["users"].attrs.clear()
        dimensionless = u.Unit("")
        result = orig.astropy.quantify({"users": dimensionless})
        assert_array_equal(result["users"].data.value, orig["users"].data)
        assert str(result["users"].data.unit) == ""

    def test_attach_units_from_str_attr_no_unit(self, example_unitless_ds):
        orig = example_unitless_ds
        orig["users"].attrs["units"] = "none"
        result = orig.astropy.quantify({"users": "m"})
        assert_array_equal(result["users"].data.value, orig["users"].data)
        assert str(result["users"].data.unit) == "m"

    @pytest.mark.parametrize("no_unit_value", conversion.no_unit_values)
    def test_override_units(self, example_unitless_ds, no_unit_value):
        orig = example_unitless_ds
        result = orig.astropy.quantify({"users": no_unit_value})
        assert (
            getattr(result["users"].data, "units", "not a quantity") == "not a quantity"
        )

    def test_error_when_already_units(self, example_quantity_ds):
        with pytest.raises(ValueError, match="already has units"):
            example_quantity_ds.astropy.quantify({"funds": "kg"})

    def test_attach_no_units(self):
        ds = xr.Dataset({"a": ("x", [1, 2, 3])})
        quantified = ds.astropy.quantify()
        assert_identical(quantified, ds)
        assert_units_equal(quantified, ds)

    def test_attach_no_new_units(self):
        ds = xr.Dataset({"a": ("x", u.Quantity([1, 2, 3], "m"))})
        quantified = ds.astropy.quantify()

        assert_identical(quantified, ds)
        assert_units_equal(quantified, ds)

    def test_attach_same_units(self):
        ds = xr.Dataset({"a": ("x", u.Quantity([1, 2, 3], "m"))})
        quantified = ds.astropy.quantify({"a": "m"})

        assert_identical(quantified, ds)
        assert_units_equal(quantified, ds)

    def test_error_when_changing_units_dimension_coordinates(self):
        ds = xr.Dataset(
            coords={"x": ("x", [-1, 0, 1], {"units": u.Unit("m")})},
        )
        with pytest.raises(ValueError, match="already has units"):
            ds.astropy.quantify({"x": "s"})

    def test_error_on_nonsense_units(self, example_unitless_ds):
        ds = example_unitless_ds
        with pytest.raises(ValueError):
            ds.astropy.quantify(units={"users": "aecjhbav"})

    def test_error_on_nonsense_units_attrs(self, example_unitless_ds):
        ds = example_unitless_ds
        ds.users.attrs["units"] = "aecjhbav"
        with pytest.raises(
            ValueError, match=rf"'users': {ds.users.attrs['units']} \(attribute\)"
        ):
            ds.astropy.quantify()

    def test_error_indicates_problematic_variable(self, example_unitless_ds):
        ds = example_unitless_ds
        with pytest.raises(ValueError, match="'users'"):
            ds.astropy.quantify(units={"users": "aecjhbav"})

    def test_existing_units(self, example_quantity_ds):
        ds = example_quantity_ds.copy()
        ds.t.attrs["units"] = u.Unit("m")

        with pytest.raises(ValueError, match="Cannot attach"):
            ds.astropy.quantify({"funds": "kg"})

    def test_existing_units_dimension(self, example_quantity_ds):
        ds = example_quantity_ds.copy()
        ds.t.attrs["units"] = u.Unit("m")

        with pytest.raises(ValueError, match="Cannot attach"):
            ds.astropy.quantify({"t": "s"})


class TestDequantifyDataSet:
    def test_strip_units(self, example_quantity_ds):
        result = example_quantity_ds.astropy.dequantify()

        assert all(
            isinstance(var.data, np.ndarray) for var in result.variables.values()
        )

    def test_attrs_reinstated(self, example_quantity_ds):
        ds = example_quantity_ds
        result = ds.astropy.dequantify()

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
        quantified = orig.astropy.quantify()

        result = quantified.astropy.dequantify()
        assert_equal(result, orig)

        result = quantified.astropy.dequantify().astropy.quantify()
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
            xr.Dataset(coords=xr.Coordinates({"x": Quantity([2, 4], "s")}, indexes={})),
            {"x": "ms"},
            xr.Dataset(
                coords=xr.Coordinates({"x": Quantity([2000, 4000], "ms")}, indexes={})
            ),
            None,
            id="Dataset-compatible units-dims-no index",
        ),
        pytest.param(
            xr.Dataset(coords=xr.Coordinates({"x": Quantity([2, 4], "s")}, indexes={})),
            {"x": "mm"},
            None,
            ValueError,
            id="Dataset-incompatible units-dims-no index",
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
                [0, 1],
                dims="x",
                coords=xr.Coordinates({"x": Quantity([2, 4], "s")}, indexes={}),
            ),
            {"x": "ms"},
            xr.DataArray(
                [0, 1],
                dims="x",
                coords=xr.Coordinates({"x": Quantity([2000, 4000], "ms")}, indexes={}),
            ),
            None,
            id="DataArray-compatible units-dims-no index",
        ),
        # pytest.param(
        #     xr.DataArray(
        #         [0, 1],
        #         dims="x",
        #         coords=xr.Coordinates({"x": Quantity([2, 4], "s")}, indexes={}),
        #     ),
        #     {"x": "ms"},
        #     xr.DataArray(
        #         [0, 1],
        #         dims="x",
        #         coords=xr.Coordinates({"x": Quantity([2000, 4000], "ms")}, indexes={}),
        #     ),
        #     None,
        #     id="DataArray-compatible units-dims-no index",
        # ),
        pytest.param(
            xr.DataArray(
                [0, 1],
                dims="x",
                coords=xr.Coordinates({"x": Quantity([2, 4], "s")}, indexes={}),
            ),
            {"x": "mm"},
            None,
            ValueError,
            id="DataArray-incompatible units-dims-no index",
        ),
    ),
)
def test_to(obj, units, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.astropy.to(units)
    else:
        actual = obj.astropy.to(units)

        assert_units_equal(actual, expected)
        assert_identical(actual, expected)


@pytest.mark.parametrize(
    ["obj", "indexers", "equivalencies", "expected", "error"],
    (
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                }
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            None,
            xr.Dataset(
                {
                    "x": ("x", [10, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60], {"units": u.Unit("s")}),
                }
            ),
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                }
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            None,
            xr.Dataset(
                {
                    "x": ("x", [1, 3], {"units": u.Unit("m")}),
                    "y": ("y", [1], {"units": u.Unit("min")}),
                }
            ),
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [1, 2, 3], {"units": u.Unit("arcsec")}),
                    "y": ("y", [2, 3], {"units": u.Unit("k")}),
                }
            ),
            {"x": Quantity([1, 0.5], "pc"), "y": Quantity([300], "1/m")},
            u.parallax(),
            xr.Dataset(
                {
                    "x": ("x", [1, 0.5], {"units": u.Unit("pc")}),
                    "y": ("y", [300], {"units": u.Unit("1/m")}),
                }
            ),
            None,
            id="Dataset-equivalent units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                }
            ),
            {"x": Quantity([1, 3], "s"), "y": Quantity([1], "m")},
            None,
            None,
            KeyError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            None,
            xr.DataArray(
                [[0], [4]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            None,
            xr.DataArray(
                [[0], [4]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [1, 3], {"units": u.Unit("m")}),
                    "y": ("y", [1], {"units": u.Unit("min")}),
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
                    "x": ("x", [1, 2, 3], {"units": u.Unit("arcsec")}),
                    "y": ("y", [2, 3], {"units": u.Unit("k")}),
                },
            ),
            {"x": Quantity([1, 0.5], "pc"), "y": Quantity([300], "1/m")},
            u.parallax(),
            xr.DataArray(
                [[1], [3]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [1, 0.5], {"units": u.Unit("pc")}),
                    "y": ("y", [300], {"units": u.Unit("1/m")}),
                },
            ),
            None,
            id="DataArray-equivalent units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "s"), "y": Quantity([60], "m")},
            None,
            None,
            KeyError,
            id="DataArray-incompatible units",
        ),
    ),
)
def test_sel(obj, indexers, equivalencies, expected, error):
    obj_ = obj.astropy.quantify()

    if error is not None:
        with pytest.raises(error):
            obj_.astropy.sel(indexers, equivalencies=equivalencies)
    else:
        expected_ = expected.astropy.quantify()

        actual = obj_.astropy.sel(indexers, equivalencies=equivalencies)
        assert_units_equal(actual, expected_)
        assert_identical(actual, expected_)


@pytest.mark.parametrize(
    ["obj", "indexers", "expected", "error"],
    (
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                }
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            xr.Dataset(
                {
                    "x": ("x", [10, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60], {"units": u.Unit("s")}),
                }
            ),
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                }
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            xr.Dataset(
                {
                    "x": ("x", [1, 3], {"units": u.Unit("m")}),
                    "y": ("y", [1], {"units": u.Unit("min")}),
                }
            ),
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            xr.DataArray(
                [[0], [4]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            xr.DataArray(
                [[0], [4]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [1, 3], {"units": u.Unit("m")}),
                    "y": ("y", [1], {"units": u.Unit("min")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
    obj_ = obj.astropy.quantify()

    if error is not None:
        with pytest.raises(error):
            obj_.astropy.loc[indexers]
    else:
        expected_ = expected.astropy.quantify()

        actual = obj_.astropy.loc[indexers]
        assert_units_equal(actual, expected_)
        assert_identical(actual, expected_)


@pytest.mark.parametrize(
    ["obj", "indexers", "values", "expected", "error"],
    (
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            [[-1], [-2]],
            xr.DataArray(
                [[-1, 1], [2, 3], [-2, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            [[-1], [-2]],
            xr.DataArray(
                [[-1, 1], [2, 3], [-2, 5]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            Quantity([[-1], [-2]], "m"),
            xr.DataArray(
                Quantity([[-1, 1], [2, 3], [-2, 5]], "m"),
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            Quantity([[-1], [-2]], "km"),
            xr.DataArray(
                Quantity([[-1000, 1], [2, 3], [-2000, 5]], "m"),
                dims=("x", "y"),
                coords={
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            Quantity([[-1], [-2]], "s"),
            None,
            astropy.units.core.UnitConversionError,
            id="data-incompatible units",
        ),
    ),
)
def test_loc_setitem(obj, indexers, values, expected, error):
    if error is not None:
        with pytest.raises(error):
            obj.astropy.loc[indexers] = values
    else:
        obj.astropy.loc[indexers] = values
        assert_units_equal(obj, expected)
        assert_identical(obj, expected)


@pytest.mark.parametrize(
    ["obj", "indexers", "expected", "error"],
    (
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                }
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            xr.Dataset(
                {
                    "x": ("x", [20], {"units": u.Unit("dm")}),
                    "y": ("y", [120], {"units": u.Unit("s")}),
                }
            ),
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                }
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            xr.Dataset(
                {
                    "x": ("x", [20], {"units": u.Unit("dm")}),
                    "y": ("y", [120], {"units": u.Unit("s")}),
                }
            ),
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([10, 30], "dm"), "y": Quantity([60], "s")},
            xr.DataArray(
                [[3]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [20], {"units": u.Unit("dm")}),
                    "y": ("y", [120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
                },
            ),
            {"x": Quantity([1, 3], "m"), "y": Quantity([1], "min")},
            xr.DataArray(
                [[3]],
                dims=("x", "y"),
                coords={
                    "x": ("x", [20], {"units": u.Unit("dm")}),
                    "y": ("y", [120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
                    "x": ("x", [10, 20, 30], {"units": u.Unit("dm")}),
                    "y": ("y", [60, 120], {"units": u.Unit("s")}),
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
            obj.astropy.drop_sel(indexers)
    else:
        actual = obj.astropy.drop_sel(indexers)
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
    actual = obj.astropy.chunk({"x": 2})

    expected = obj.astropy.dequantify().chunk({"x": 2}).astropy.quantify()

    assert_units_equal(actual, expected)
    assert_identical(actual, expected)


@pytest.mark.parametrize(
    [
        "obj",
        "units",
        "equivalencies",
        "indexers",
        "expected",
        "expected_units",
        "error",
    ],
    (
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            None,
            {"x": Quantity([10, 30, 50], "dm"), "y": Quantity([0, 120, 240], "s")},
            xr.Dataset({"x": ("x", [10, 30, 50]), "y": ("y", [0, 120, 240])}),
            {"x": "dm", "y": "s"},
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            None,
            {"x": Quantity([0, 1, 3, 5], "m"), "y": Quantity([0, 2, 4], "min")},
            xr.Dataset({"x": ("x", [0, 1, 3, 5]), "y": ("y", [0, 2, 4])}),
            {"x": "m", "y": "min"},
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [1, 2, 3]), "y": ("y", [60, 120])}),
            {"x": "k", "y": "s"},
            u.parallax(),
            {"x": Quantity([0, 100, 300, 500], "1/m"), "y": Quantity([0, 2, 4], "min")},
            xr.Dataset({"x": ("x", [0, 100, 300, 500]), "y": ("y", [0, 2, 4])}),
            {"x": "1/m", "y": "min"},
            None,
            id="Dataset-equivalent units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            None,
            {"x": Quantity([1, 3], "s"), "y": Quantity([1], "m")},
            None,
            {},
            ValueError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (("x", "y"), np.array([[0, 1], [2, 3], [4, 5]])),
                    "x": [10, 20, 30],
                    "y": [60, 120],
                }
            ),
            {"a": "kg"},
            None,
            {
                "x": [15, 25],
                "y": [75, 105],
            },
            xr.Dataset(
                {
                    "a": (("x", "y"), np.array([[np.nan, np.nan], [np.nan, np.nan]])),
                    "x": [15, 25],
                    "y": [75, 105],
                }
            ),
            {"a": "kg"},
            None,
            id="Dataset-data units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            None,
            {"x": Quantity([10, 30, 50], "dm"), "y": Quantity([0, 240], "s")},
            xr.DataArray(
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 30, 50]), "y": ("y", [0, 240])},
            ),
            {"x": "dm", "y": "s"},
            None,
            id="DataArray-identical units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            None,
            {"x": Quantity([1, 3, 5], "m"), "y": Quantity([0, 2], "min")},
            xr.DataArray(
                [[np.nan, 1], [np.nan, 5], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [1, 3, 5]), "y": ("y", [0, 2])},
            ),
            {"x": "m", "y": "min"},
            None,
            id="DataArray-compatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [1, 2, 3]), "y": ("y", [60, 120])},
            ),
            {"x": "k", "y": "s"},
            u.parallax(),
            {"x": Quantity([100, 300, 500], "1/m"), "y": Quantity([0, 2], "min")},
            xr.DataArray(
                [[np.nan, 1], [np.nan, 5], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [100, 300, 500]), "y": ("y", [0, 2])},
            ),
            {"x": "1/m", "y": "min"},
            None,
            id="DataArray-equivalent units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            None,
            {"x": Quantity([10, 30], "s"), "y": Quantity([60], "m")},
            None,
            {},
            ValueError,
            id="DataArray-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                np.array([[0, 1], [2, 3], [4, 5]]),
                dims=("x", "y"),
                coords={"x": [10, 20, 30], "y": [60, 120]},
            ),
            {None: "kg"},
            None,
            {"x": [15, 25], "y": [75, 105]},
            xr.DataArray(
                [[np.nan, np.nan], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": [15, 25], "y": [75, 105]},
            ),
            {None: "kg"},
            None,
            id="DataArray-data units",
        ),
    ),
)
def test_reindex(obj, units, equivalencies, indexers, expected, expected_units, error):
    obj_ = obj.astropy.quantify(units)

    if error is not None:
        with pytest.raises(error):
            obj.astropy.reindex(indexers)
    else:
        expected_ = expected.astropy.quantify(units=expected_units)

        actual = obj_.astropy.reindex(indexers, equivalencies=equivalencies)
        assert_units_equal(actual, expected_)
        assert_identical(actual, expected_)


@pytest.mark.parametrize(
    [
        "obj",
        "units",
        "equivalencies",
        "other",
        "other_units",
        "expected",
        "expected_units",
        "error",
    ],
    (
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            None,
            xr.Dataset({"x": ("x", [10, 30, 50]), "y": ("y", [0, 120, 240])}),
            {"x": "dm", "y": "s"},
            xr.Dataset({"x": ("x", [10, 30, 50]), "y": ("y", [0, 120, 240])}),
            {"x": "dm", "y": "s"},
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            None,
            xr.Dataset({"x": ("x", [0, 1, 3, 5]), "y": ("y", [0, 2, 4])}),
            {"x": "m", "y": "min"},
            xr.Dataset({"x": ("x", [0, 1, 3, 5]), "y": ("y", [0, 2, 4])}),
            {"x": "m", "y": "min"},
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "k", "y": "s"},
            u.parallax(),
            xr.Dataset({"x": ("x", [0, 1000, 3000, 5000]), "y": ("y", [0, 2, 4])}),
            {"x": "1/m", "y": "min"},
            xr.Dataset({"x": ("x", [0, 1000, 3000, 5000]), "y": ("y", [0, 2, 4])}),
            {"x": "1/m", "y": "min"},
            None,
            id="Dataset-equivalent units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            None,
            xr.Dataset({"x": ("x", [1, 3]), "y": ("y", [1])}),
            {"x": "s", "y": "m"},
            None,
            {},
            ValueError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            None,
            xr.Dataset({"x": ("x", [10, 30, 50]), "y": ("y", [0, 240])}),
            {"x": "dm", "y": "s"},
            xr.DataArray(
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 30, 50]), "y": ("y", [0, 240])},
            ),
            {"x": "dm", "y": "s"},
            None,
            id="DataArray-identical units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (("x", "y"), [[0, 1], [2, 3], [4, 5]]),
                    "x": [10, 20, 30],
                    "y": [60, 120],
                }
            ),
            {"a": "kg"},
            None,
            xr.Dataset({"x": [15, 25], "y": [75, 105]}),
            {},
            xr.Dataset(
                {
                    "a": (("x", "y"), [[np.nan, np.nan], [np.nan, np.nan]]),
                    "x": [15, 25],
                    "y": [75, 105],
                }
            ),
            {"a": "kg"},
            None,
            id="Dataset-data units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            None,
            xr.Dataset({"x": ("x", [1, 3, 5]), "y": ("y", [0, 2])}),
            {"x": "m", "y": "min"},
            xr.DataArray(
                [[np.nan, 1], [np.nan, 5], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [1, 3, 5]), "y": ("y", [0, 2])},
            ),
            {"x": "m", "y": "min"},
            None,
            id="DataArray-compatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "k", "y": "s"},
            u.parallax(),
            xr.Dataset({"x": ("x", [1000, 3000, 5000]), "y": ("y", [0, 2])}),
            {"x": "1/m", "y": "min"},
            xr.DataArray(
                [[np.nan, 1], [np.nan, 5], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [1000, 3000, 5000]), "y": ("y", [0, 2])},
            ),
            {"x": "1/m", "y": "min"},
            None,
            id="DataArray-equivalent units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            None,
            xr.Dataset({"x": ("x", [10, 30]), "y": ("y", [60])}),
            {"x": "s", "y": "m"},
            None,
            {},
            ValueError,
            id="DataArray-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": [10, 20, 30], "y": [60, 120]},
            ),
            {"a": "kg"},
            None,
            xr.Dataset({"x": [15, 25], "y": [75, 105]}),
            {},
            xr.DataArray(
                [[np.nan, np.nan], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": [15, 25], "y": [75, 105]},
            ),
            {"a": "kg"},
            None,
            id="DataArray-data units",
        ),
    ),
)
def test_reindex_like(
    obj, units, equivalencies, other, other_units, expected, expected_units, error
):
    obj_ = obj.astropy.quantify(units)
    other_ = other.astropy.quantify(other_units)

    if error is not None:
        with pytest.raises(error):
            obj_.astropy.reindex_like(other_, equivalencies=equivalencies)
    else:
        expected_ = expected.astropy.quantify(expected_units)

        actual = obj_.astropy.reindex_like(other_)
        assert_units_equal(actual, expected_)
        assert_identical(actual, expected_)


@requires_scipy
@pytest.mark.parametrize(
    ["obj", "units", "indexers", "expected", "expected_units", "error", "kwargs"],
    (
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            {"x": Quantity([10, 30, 50], "dm"), "y": Quantity([0, 120, 240], "s")},
            xr.Dataset({"x": ("x", [10, 30, 50]), "y": ("y", [0, 120, 240])}),
            {"x": "dm", "y": "s"},
            None,
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            {"x": Quantity([0, 1, 3, 5], "m"), "y": Quantity([0, 2, 4], "min")},
            xr.Dataset({"x": ("x", [0, 1, 3, 5]), "y": ("y", [0, 2, 4])}),
            {"x": "m", "y": "min"},
            None,
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            {"x": Quantity([1, 3], "s"), "y": Quantity([1], "m")},
            None,
            {},
            ValueError,
            None,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (("x", "y"), np.array([[0, 1], [2, 3], [4, 5]])),
                    "x": [10, 20, 30],
                    "y": [60, 120],
                }
            ),
            {"a": "kg"},
            {
                "x": [15, 25],
                "y": [75, 105],
            },
            xr.Dataset(
                {
                    "a": (("x", "y"), np.array([[1.25, 1.75], [3.25, 3.75]])),
                    "x": [15, 25],
                    "y": [75, 105],
                }
            ),
            {"a": "kg"},
            None,
            None,
            id="Dataset-data units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            {"x": Quantity([10, 30, 50], "dm"), "y": Quantity([0, 240], "s")},
            xr.DataArray(
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 30, 50]), "y": ("y", [0, 240])},
            ),
            {"x": "dm", "y": "s"},
            None,
            None,
            id="DataArray-identical units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            {"x": Quantity([1, 3, 5], "m"), "y": Quantity([0, 2], "min")},
            xr.DataArray(
                [[np.nan, 1], [np.nan, 5], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [1, 3, 5]), "y": ("y", [0, 2])},
            ),
            {"x": "m", "y": "min"},
            None,
            None,
            id="DataArray-compatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            {"x": Quantity([10, 30], "s"), "y": Quantity([60], "m")},
            None,
            {},
            ValueError,
            None,
            id="DataArray-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                np.array([[0, 1], [2, 3], [4, 5]]),
                dims=("x", "y"),
                coords={"x": [10, 20, 30], "y": [60, 120]},
            ),
            {None: "kg"},
            {"x": [15, 25], "y": [75, 105]},
            xr.DataArray(
                [[1.25, 1.75], [3.25, 3.75]],
                dims=("x", "y"),
                coords={"x": [15, 25], "y": [75, 105]},
            ),
            {None: "kg"},
            None,
            None,
            id="DataArray-data units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            {"x": Quantity([1, 3, 5], "m"), "y": Quantity([0, 2], "min")},
            xr.DataArray(
                [[0, 1], [0, 5], [0, 0]],
                dims=("x", "y"),
                coords={"x": ("x", [1, 3, 5]), "y": ("y", [0, 2])},
            ),
            {"x": "m", "y": "min"},
            None,
            {"bounds_error": False, "fill_value": 0},
            id="DataArray-other parameters",
        ),
    ),
)
def test_interp(obj, units, indexers, expected, expected_units, error, kwargs):
    obj_ = obj.astropy.quantify(units)

    if error is not None:
        with pytest.raises(error):
            obj_.astropy.interp(indexers, kwargs=kwargs)
    else:
        expected_ = expected.astropy.quantify(expected_units)

        actual = obj_.astropy.interp(indexers, kwargs=kwargs)
        assert_units_equal(actual, expected_)
        assert_identical(actual, expected_)


@requires_scipy
@pytest.mark.parametrize(
    ["obj", "units", "other", "other_units", "expected", "expected_units", "error"],
    (
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            xr.Dataset({"x": ("x", [10, 30, 50]), "y": ("y", [0, 120, 240])}),
            {"x": "dm", "y": "s"},
            xr.Dataset({"x": ("x", [10, 30, 50]), "y": ("y", [0, 120, 240])}),
            {"x": "dm", "y": "s"},
            None,
            id="Dataset-identical units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            xr.Dataset({"x": ("x", [0, 1, 3, 5]), "y": ("y", [0, 2, 4])}),
            {"x": "m", "y": "min"},
            xr.Dataset({"x": ("x", [0, 1, 3, 5]), "y": ("y", [0, 2, 4])}),
            {"x": "m", "y": "min"},
            None,
            id="Dataset-compatible units",
        ),
        pytest.param(
            xr.Dataset({"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])}),
            {"x": "dm", "y": "s"},
            xr.Dataset({"x": ("x", [1, 3]), "y": ("y", [1])}),
            {"x": "s", "y": "m"},
            None,
            {},
            ValueError,
            id="Dataset-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            xr.Dataset({"x": ("x", [10, 30, 50]), "y": ("y", [0, 240])}),
            {"x": "dm", "y": "s"},
            xr.DataArray(
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 30, 50]), "y": ("y", [0, 240])},
            ),
            {"x": "dm", "y": "s"},
            None,
            id="DataArray-identical units",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "a": (("x", "y"), [[0, 1], [2, 3], [4, 5]]),
                    "x": [10, 20, 30],
                    "y": [60, 120],
                }
            ),
            {"a": "kg"},
            xr.Dataset({"x": [15, 25], "y": [75, 105]}),
            {},
            xr.Dataset(
                {
                    "a": (("x", "y"), [[1.25, 1.75], [3.25, 3.75]]),
                    "x": [15, 25],
                    "y": [75, 105],
                }
            ),
            {"a": "kg"},
            None,
            id="Dataset-data units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            xr.Dataset({"x": ("x", [1, 3, 5]), "y": ("y", [0, 2])}),
            {"x": "m", "y": "min"},
            xr.DataArray(
                [[np.nan, 1], [np.nan, 5], [np.nan, np.nan]],
                dims=("x", "y"),
                coords={"x": ("x", [1, 3, 5]), "y": ("y", [0, 2])},
            ),
            {"x": "m", "y": "min"},
            None,
            id="DataArray-compatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": ("x", [10, 20, 30]), "y": ("y", [60, 120])},
            ),
            {"x": "dm", "y": "s"},
            xr.Dataset({"x": ("x", [10, 30]), "y": ("y", [60])}),
            {"x": "s", "y": "m"},
            None,
            {},
            ValueError,
            id="DataArray-incompatible units",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3], [4, 5]],
                dims=("x", "y"),
                coords={"x": [10, 20, 30], "y": [60, 120]},
            ),
            {"a": "kg"},
            xr.Dataset({"x": [15, 25], "y": [75, 105]}),
            {},
            xr.DataArray(
                [[1.25, 1.75], [3.25, 3.75]],
                dims=("x", "y"),
                coords={"x": [15, 25], "y": [75, 105]},
            ),
            {"a": "kg"},
            None,
            id="DataArray-data units",
        ),
    ),
)
def test_interp_like(obj, units, other, other_units, expected, expected_units, error):
    obj_ = obj.astropy.quantify(units)
    other_ = other.astropy.quantify(other_units)

    if error is not None:
        with pytest.raises(error):
            obj_.astropy.interp_like(other_)
    else:
        expected_ = expected.astropy.quantify(expected_units)

        actual = obj_.astropy.interp_like(other_)
        assert_units_equal(actual, expected_)
        assert_identical(actual, expected_)


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
    actual = obj.astropy.ffill(dim="x")
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
    actual = obj.astropy.bfill(dim="x")
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
    actual = obj.astropy.interpolate_na(dim="x")
    assert_identical(actual, expected)
    assert_units_equal(actual, expected)
