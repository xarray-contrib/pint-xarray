import pytest

import xarray as xr
from xarray.testing import assert_equal

import numpy as np
from numpy.testing import assert_array_equal

from pint import UnitRegistry
from pint.unit import Unit
from pint.errors import UndefinedUnitError, DimensionalityError

from pintxarray.accessors import PintDataArrayAccessor, PintDatasetAccessor
from .utils import extract_units, raises_regex


# make sure scalars are converted to 0d arrays so quantities can
# always be treated like ndarrays
unit_registry = UnitRegistry(force_ndarray=True)
Quantity = unit_registry.Quantity


@pytest.fixture()
def example_unitless_da():
    array = np.linspace(0, 10, 20)
    x = np.arange(20)
    da = xr.DataArray(data=array, dims="x", coords={"x": x})
    da.attrs['units'] = 'm'
    da.coords['x'].attrs['units'] = 's'
    return da

@pytest.fixture()
def example_quantity_da():
    array = np.linspace(0, 10, 20) * unit_registry.m
    x = np.arange(20) * unit_registry.s
    return xr.DataArray(data=array, dims="x", coords={"x": x})


class TestQuantifyDataArray:
    def test_attach_units_given_registry(self, example_unitless_da):
        orig = example_unitless_da
        ureg = UnitRegistry(force_ndarray=True)
        result = orig.pint.quantify('m', unit_registry=ureg)
        assert_array_equal(result.data.magnitude, orig.data)
        assert result.data.units == ureg.Unit('m')

    def test_attach_units(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.pint.quantify('m')
        assert_array_equal(result.data.magnitude, orig.data)
        # TODO better comparisons for when you can't access the unit_registry?
        assert str(result.data.units) == 'meter'

    def test_attach_units_from_attrs(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.pint.quantify()
        assert_array_equal(result.data.magnitude, orig.data)
        assert str(result.data.units) == 'meter'

    def test_attach_unit_class(self, example_unitless_da):
        orig = example_unitless_da
        ureg = UnitRegistry(force_ndarray=True)
        result = orig.pint.quantify(ureg.Unit('m'), unit_registry=ureg)
        assert_array_equal(result.data.magnitude, orig.data)
        assert result.data.units == ureg.Unit('m')

    def test_error_when_already_units(self, example_quantity_da):
        da = example_quantity_da
        with raises_regex(ValueError, "already has units"):
            da.pint.quantify()

    def test_error_on_nonsense_units(self, example_unitless_da):
        da = example_unitless_da
        with pytest.raises(UndefinedUnitError):
            da.pint.quantify(units='aecjhbav')

    def test_registry_kwargs(self, example_unitless_da):
        orig = example_unitless_da
        result = orig.pint.quantify(registry_kwargs=
                                    {'auto_reduce_dimensions': True})
        assert(result.data._REGISTRY.auto_reduce_dimensions) == True


class TestDequantifyDataArray:
    def test_strip_units(self, example_quantity_da):
        result = example_quantity_da.pint.dequantify()
        assert isinstance(result.data, np.ndarray)
        assert isinstance(result.coords['x'].data, np.ndarray)

    def test_error_if_no_units(self, example_unitless_da):
        with raises_regex(ValueError, "does not have units"):
            example_unitless_da.pint.dequantify()

    def test_attrs_reinstated(self, example_quantity_da):
        da = example_quantity_da
        result = da.pint.dequantify()
        assert result.attrs['units'] == 'meter'

    def test_roundtrip_data(self, example_unitless_da):
        orig = example_unitless_da
        quantified = orig.pint.quantify()
        result = quantified.pint.dequantify()
        assert_equal(result, orig)


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


@pytest.mark.skip(reason="Not yet implemented")
class TestQuantifyDataSet:
    def test_attach_units(self):
        ...

    def test_attach_str_units(self):
        ...

    def test_attach_units_from_registry(self):
        ...


@pytest.mark.skip(reason="Not yet implemented")
class TestIndexing:
    ...
