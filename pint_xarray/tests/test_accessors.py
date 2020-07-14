import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from pint import UnitRegistry
from pint.errors import UndefinedUnitError
from xarray.testing import assert_equal

from .. import conversion
from .utils import raises_regex

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

    def test_registry_kwargs(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.pint.quantify(registry_kwargs={"auto_reduce_dimensions": True})
        assert result.data._REGISTRY.auto_reduce_dimensions is True


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


@pytest.mark.skip(reason="Not yet implemented")
class TestPropertiesDataArray:
    def test_units(self):
        ...


@pytest.mark.skip(reason="Not yet implemented")
class TestConversionDataArray:
    def test_units(self):
        ...


@pytest.mark.skip(reason="Not yet implemented")
class TestUncertainties:
    def test_units(self):
        ...


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


@pytest.mark.skip(reason="Not yet implemented")
class TestDequantifyDataSet:
    ...


@pytest.mark.skip(reason="Not yet implemented")
class TestIndexing:
    ...
