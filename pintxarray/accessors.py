# TODO is it possible to import pint-xarray from within xarray if pint is present?
from xarray import (register_dataarray_accessor, register_dataset_accessor,
                    DataArray, Dataset)
from xarray.core.npcompat import IS_NEP18_ACTIVE

import numpy as np

import pint
from pint.quantity import Quantity
from pint.unit import Unit


if not hasattr(Quantity, "__array_function__"):
    raise ImportError("Imported version of pint does not implement "
                      "__array_function__")

if not IS_NEP18_ACTIVE:
    raise ImportError("NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled")


# TODO could/should we overwrite xr.open_dataset and xr.open_mfdataset to make
# them apply units upon loading???
# TODO could even override the decode_cf kwarg?

# TODO docstrings
# TODO type hints
# TODO f-strings


def _array_attach_units(data, unit, convert_from=None):
    """
    Internal utility function for attaching units to a numpy-like array,
    converting them, or throwing the correct error.
    """

    if isinstance(data, Quantity):
        if not convert_from:
            raise ValueError(f"Cannot attach unit {unit} to quantity: data "
                             f"already has units {data.units}")
        elif isinstance(convert_from, Unit):
            data = data.magnitude
        elif convert_from is True:  # intentionally accept exactly true
            if data.check(unit):
                convert_from = data.units
                data = data.magnitude
            else:
                raise ValueError("Cannot convert quantity from {data.units} "
                                 "to {unit}")
        else:
            raise ValueError("Cannot convert from invalid unit {convert_from}")

    # to make sure we also encounter the case of "equal if converted"
    if convert_from is not None:
        quantity = (data * convert_from).to(
            unit
            if isinstance(unit, Unit)
            else unit.dimensionless
        )
    else:
        try:
            quantity = data * unit
        except np.core._exceptions.UFuncTypeError:
            # from @keewis in xarray.tests.test_units - unsure what this checks?
            if unit != 1:
                raise

            quantity = data

    return quantity


@register_dataarray_accessor("pint")
class PintDataArrayAccessor:
    """
    Access methods for DataArrays with units using Pint.

    Methods and attributes can be accessed through the `.pint` attribute.
    """

    def __init__(self, da):
        self.da = da

    def quantify(self, units=None, unit_registry=None, registry_kwargs=None):
        """
        Attaches units to the DataArray.

        Units can be specified as a pint.Unit or as a string, which will will
        be parsed by the given unit registry. If no units are specified then
        the units will be parsed from the `'units'` entry of the DataArray's
        `.attrs`. Will raise a ValueError if the DataArray already contains a
        unit-aware array.

        Parameters
        ----------
        units : pint.Unit or str, optional
            Physical units to use for this DataArray. If not provided, will try
            to read them from `DataArray.attrs['units']` using pint's parser.
        unit_registry : `pint.UnitRegistry`, optional
            Unit registry to be used for the units attached to this DataArray.
            If not given then a default registry will be created.
        registry_kwargs : dict, optional
            Keyword arguments to be passed to `pint.UnitRegistry`.

        Returns
        -------
        quantified - DataArray whose wrapped array data will now be a Quantity
        array with the specified units.

        Examples
        --------
        >>> da.pint.quantify(units='Hz')
        <xarray.DataArray (frequency: 6)>
        Quantity([ 0.4,  0.9,  1.7,  4.8,  3.2,  9.1], 'Hz')
        Coordinates:
        * wavelength  (wavelength) np.array 1e-4, 2e-4, 4e-4, 6e-4, 1e-3, 2e-3
        """

        if isinstance(self.da.data, Quantity):
            raise ValueError

        if unit_registry is None:
            if registry_kwargs is None:
                registry_kwargs = {}
            unit_registry = pint.UnitRegistry(**registry_kwargs)
        else:
            if registry_kwargs is not None:
                raise ValueError("Cannot supply registry kwargs without "
                                 "supplying a registry")

        if units is None:
            # TODO option to read and decode units according to CF conventions (see MetPy)?
            attr_units = self.da.attrs['units']
            units = unit_registry.parse_expression(attr_units)
        elif isinstance(units, Unit):
            # TODO do we have to check what happens if someone passes a Unit instance
            # without creating a unit registry?
            pass
        else:
            units = unit_registry.Unit(units)

        quantity = _array_attach_units(self.da.data, units, convert_from=None)

        # TODO should we (temporarily) remove the attrs here so that they don't become inconsistent?
        return DataArray(dims=self.da.dims, data=quantity,
                         coords=self.da.coords, attrs=self.da.attrs )

    def dequantify(self, encode_cf=True):
        da = DataArray(dim=self.da.dims, data=self.da.pint.magnitude,
                       coords=self.da.coords, attrs=self.da.attrs,
                       encoding=self.da.encoding)
        da.attrs['units'] = self.da.pint.units
        return da

    @property
    def magnitude(self):
        return self.da.data.magnitude

    @magnitude.setter
    def magnitude(self, da):
        self.da = DataArray(dim=self.da.dims, data=da.data,
                            coords=self.da.coords, attrs=self.da.attrs)
    @property
    def units(self):
        return self.da.data.units

    @units.setter
    def units(self, units):
        quantity = _array_attach_units(self.da.data, units)
        self.da = DataArray(dim=self.da.dims, data=quantity,
                            coords=self.da.coords, attrs=self.da.attrs)

    @property
    def dimensionality(self):
        return self.da.data.dimensionality

    def to(self, units):
        quantity = self.da.data.to(units)
        return DataArray(dim=self.da.dims, data=quantity,
                         coords=self.da.coords, attrs=self.da.attrs,
                         encoding=self.da.encoding)

    def to_base_units(self):
        quantity = self.da.data.to_base_units()
        return DataArray(dim=self.da.dims, data=quantity,
                         coords=self.da.coords, attrs=self.da.attrs,
                         encoding=self.da.encoding)

    # TODO integrate with the uncertainties package here...?
    def plus_minus(self, value, error, relative=False):
        quantity = self.da.data.plus_minus(value, error, relative)
        return DataArray(dim=self.da.dims, data=quantity,
                         coords=self.da.coords, attrs=self.da.attrs,
                         encoding=self.da.encoding)

    def sel(self, indexers=None, method=None, tolerance=None, drop=False,
            **indexers_kwargs):
        ...

    @property
    def loc(self):
        ...


@register_dataset_accessor("pint")
class PintDatasetAccessor:
    def __init__(self, ds):
        self.ds = ds

    def quantify(self, unit_registry=None, decode_cf=False):
        quantified_vars = {name: da.pint.quantify(unit_registry=unit_registry,
                                                  decode_cf=decode_cf)
                           for name, da in self.ds.items()}
        return Dataset(quantified_vars, attrs=self.ds.attrs,
                       encoding=self.ds.encoding)

    def dequantify(self):
        dequantified_vars = {name: da.pint.to_base_units()
                             for name, da in self.ds.items()}
        return Dataset(dequantified_vars, attrs=self.ds.attrs,
                       encoding=self.ds.encoding)

    def to_base_units(self):
        base_vars = {name: da.pint.to_base_units()
                     for name, da in self.ds.items()}
        return Dataset(base_vars, attrs=self.ds.attrs, encoding=self.ds.encoding)

    # TODO way to change every variable in ds to be expressed in a new units system?

    def sel(self, indexers=None, method=None, tolerance=None, drop=False,
            **indexers_kwargs):
        ...

    @property
    def loc(self):
        ...
