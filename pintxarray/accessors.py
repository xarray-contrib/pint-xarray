from importlib import import_module

import pint
from pint.quantity import Quantity
from pint.unit import Unit
# TODO is it possible to import pint-xarray from within xarray if pint is present?
import xarray as xr
import numpy as np
from xarray.core.npcompat import IS_NEP18_ACTIVE


if not hasattr(Quantity, "__array_function__"):
    raise ImportError("Imported version of pint does not implement "
                      "__array_function__ yet")

if not IS_NEP18_ACTIVE:
    raise ImportError("NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled")

# TODO do we need this? xarray chooses it's own labels anyway
try:
    mpl = import_module("matplotlib")
    has_mpl = True
except ImportError:
    has_mpl = False
else:
    # TODO can we do this without initialising a Unit Registry?
    unit_registry = pint.UnitRegistry()
    unit_registry.setup_matplotlib(True)


# TODO could/should we overwrite xr.open_dataset and xr.open_mfdataset to make
# them apply units upon loading???
# TODO could even override the decode_cf kwarg?

# TODO docstrings
# TODO type hints
# TODO f-strings

def array_attach_units(data, unit, convert_from=None):
    if isinstance(data, Quantity):
        if not convert_from:
            raise ValueError(
                "cannot attach unit {unit} to quantity ({data.units})".format(
                    unit=unit, data=data
                )
            )
        elif isinstance(convert_from, Unit):
            data = data.magnitude
        elif convert_from is True:  # intentionally accept exactly true
            if data.check(unit):
                convert_from = data.units
                data = data.magnitude
            else:
                raise ValueError(
                    "cannot convert quantity ({data.units}) to {unit}".format(
                        unit=unit, data=data
                    )
                )
        else:
            raise ValueError(
                "cannot convert from invalid unit {convert_from}".format(
                    convert_from=convert_from
                )
            )

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
            if unit != 1:
                raise

            quantity = data

    return quantity


# TODO Error checking (that data is actually a quantity etc)

# TODO refactor with an apply_to(da, data_method) function?


@xr.register_dataarray_accessor("pint")
class PintDataArrayAccessor:
    def __init__(self, da):
        self.da = da

    def quantify(self, units=None, unit_registry=None, decode_cf=False):
        # TODO read and decode units according to CF conventions (see MetPy)
        if not units:
            if decode_cf:
                # TODO unit = Unit(_decode_cf(self.da.attrs['units']))
                raise NotImplementedError
            else:
                units = Unit(self.da.attrs['units'])

        quantity = array_attach_units(self.da.data, units)
        # TODO should we (temporarily) remove the attrs here?
        return xr.DataArray(dim=self.da.dims, data=quantity,
                            coords=self.da.coords, attrs=self.da.attrs,
                            encoding=self.da.encoding)

    def dequantify(self, encode_cf=True):
        da = xr.DataArray(dim=self.da.dims, data=self.da.pint.magnitude,
                          coords=self.da.coords, attrs=self.da.attrs,
                          encoding=self.da.encoding)
        da.attrs['units'] = self.da.pint.units
        return da

    @property
    def units(self):
        return self.da.data.units

    @units.setter
    def units(self, units):
        quantity = array_attach_units(self.da.data, units)
        self.da = xr.DataArray(dim=self.da.dims, data=quantity,
                               coords=self.da.coords, attrs=self.da.attrs,
                               encoding=self.da.encoding)

    @property
    def magnitude(self):
        return self.da.data.magnitude

    @magnitude.setter
    def magnitude(self, da):
        self.da = xr.DataArray(dim=self.da.dims, data=da.data,
                               coords=self.da.coords, attrs=self.da.attrs,
                               encoding=self.da.encoding)

    def to(self, units):
        quantity = self.da.data.to(units)
        return xr.DataArray(dim=self.da.dims, data=quantity,
                            coords=self.da.coords, attrs=self.da.attrs,
                            encoding=self.da.encoding)

    def to_base_units(self):
        quantity = self.da.data.to_base_units()
        return xr.DataArray(dim=self.da.dims, data=quantity,
                            coords=self.da.coords, attrs=self.da.attrs,
                            encoding=self.da.encoding)

    # TODO integrate with the uncertainties package here...?
    def plus_minus(self, value, error, relative=False):
        quantity = self.da.data.plus_minus(value, error, relative)
        return xr.DataArray(dim=self.da.dims, data=quantity,
                            coords=self.da.coords, attrs=self.da.attrs,
                            encoding=self.da.encoding)

    def sel(self, indexers=None, method=None, tolerance=None, drop=False,
            **indexers_kwargs):
        ...

    @property
    def loc(self):
        ...


@xr.register_dataset_accessor("pint")
class PintDatasetAccessor:
    def __init__(self, ds):
        self.ds = ds

    def quantify(self, unit_registry=None, decode_cf=False):
        quantified_vars = {name: da.pint.quantify(unit_registry=unit_registry,
                                                  decode_cf=decode_cf)
                           for name, da in self.ds.items()}
        return xr.Dataset(quantified_vars, attrs=self.ds.attrs,
                          encoding=self.ds.encoding)

    def dequantify(self):
        dequantified_vars = {name: da.pint.to_base_units()
                             for name, da in self.ds.items()}
        return xr.Dataset(dequantified_vars, attrs=self.ds.attrs,
                          encoding=self.ds.encoding)

    def to_base_units(self):
        base_vars = {name: da.pint.to_base_units()
                     for name, da in self.ds.items()}
        return xr.Dataset(base_vars, attrs=self.ds.attrs, encoding=self.ds.encoding)

    # TODO way to change every variable in ds to be expressed in a new units system?

    def sel(self, indexers=None, method=None, tolerance=None, drop=False,
            **indexers_kwargs):
        ...

    @property
    def loc(self):
        ...
