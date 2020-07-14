# TODO is it possible to import pint-xarray from within xarray if pint is present?
import itertools

import numpy as np
import pint
from pint.quantity import Quantity
from pint.unit import Unit
from xarray import (
    DataArray,
    Dataset,
    Variable,
    register_dataarray_accessor,
    register_dataset_accessor,
)
from xarray.core.npcompat import IS_NEP18_ACTIVE

from . import conversion

if not hasattr(Quantity, "__array_function__"):
    raise ImportError(
        "Imported version of pint does not implement " "__array_function__"
    )

if not IS_NEP18_ACTIVE:
    raise ImportError("NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled")


# TODO could/should we overwrite xr.open_dataset and xr.open_mfdataset to make
# them apply units upon loading???
# TODO could even override the decode_cf kwarg?

# TODO docstrings
# TODO type hints


def is_dict_like(obj):
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def zip_mappings(*mappings, fill_value=None):
    """ zip mappings by combining values for common keys into a tuple

    Works like itertools.zip_longest, so if a key is missing from a
    mapping, it is replaced by ``fill_value``.

    Parameters
    ----------
    *mappings : dict-like
        The mappings to zip
    fill_value
        The value to use if a key is missing from a mapping.

    Returns
    -------
    zipped : dict-like
        The zipped mapping
    """
    keys = set(itertools.chain.from_iterable(mapping.keys() for mapping in mappings))

    # TODO: could this be made more efficient using itertools.groupby?
    zipped = {
        key: tuple(mapping.get(key, fill_value) for mapping in mappings) for key in keys
    }
    return zipped


def units_to_str_or_none(mapping):
    return {
        key: str(value) if isinstance(value, Unit) else value
        for key, value in mapping.items()
    }


# based on xarray.core.utils.either_dict_or_kwargs
# https://github.com/pydata/xarray/blob/v0.15.1/xarray/core/utils.py#L249-L268
def either_dict_or_kwargs(positional, keywords, method_name):
    if positional is not None:
        if not is_dict_like(positional):
            raise ValueError(
                f"the first argument to .{method_name} must be a dictionary"
            )
        if keywords:
            raise ValueError(
                "cannot specify both keyword and positional "
                f"arguments to .{method_name}"
            )
        return positional
    else:
        return keywords


def _array_attach_units(data, unit, convert_from=None):
    """
    Internal utility function for attaching units to a numpy-like array,
    converting them, or throwing the correct error.
    """

    if isinstance(data, Quantity):
        if not convert_from:
            raise ValueError(
                f"Cannot attach unit {unit} to quantity: data "
                f"already has units {data.units}"
            )
        elif isinstance(convert_from, Unit):
            data = data.magnitude
        elif convert_from is True:  # intentionally accept exactly true
            if data.check(unit):
                convert_from = data.units
                data = data.magnitude
            else:
                raise ValueError(
                    "Cannot convert quantity from {data.units} " "to {unit}"
                )
        else:
            raise ValueError("Cannot convert from invalid unit {convert_from}")

    # to make sure we also encounter the case of "equal if converted"
    if convert_from is not None:
        quantity = (data * convert_from).to(
            unit if isinstance(unit, Unit) else unit.dimensionless
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


def _get_registry(unit_registry, registry_kwargs):
    if unit_registry is None:
        if registry_kwargs is None:
            registry_kwargs = {}
        registry_kwargs.update(force_ndarray=True)
        # TODO should this registry object then be stored somewhere global?
        unit_registry = pint.UnitRegistry(**registry_kwargs)
    return unit_registry


def _decide_units(units, registry, unit_attribute):
    if units is None and unit_attribute is None:
        # or warn and return None?
        raise ValueError("no units given")
    elif units is None:
        # TODO option to read and decode units according to CF conventions (see MetPy)?
        units = registry.parse_expression(unit_attribute).units
    elif isinstance(units, Unit):
        # TODO do we have to check what happens if someone passes a Unit instance
        # without creating a unit registry?
        # TODO and what happens if they pass in a Unit from a different registry
        pass
    else:
        units = registry.Unit(units)
    return units


def _quantify_variable(var, units):
    new_data = _array_attach_units(var.data, units, convert_from=None)
    new_var = Variable(dims=var.dims, data=new_data, attrs=var.attrs)
    return new_var


def _dequantify_variable(var):
    new_var = Variable(dims=var.dims, data=var.data.magnitude, attrs=var.attrs)
    new_var.attrs["units"] = str(var.data.units)
    return new_var


@register_dataarray_accessor("pint")
class PintDataArrayAccessor:
    """
    Access methods for DataArrays with units using Pint.

    Methods and attributes can be accessed through the `.pint` attribute.
    """

    def __init__(self, da):
        self.da = da

    def quantify(
        self, units=None, unit_registry=None, registry_kwargs=None, **unit_kwargs
    ):
        """
        Attaches units to the DataArray.

        Units can be specified as a pint.Unit or as a string, which will be
        parsed by the given unit registry. If no units are specified then the
        units will be parsed from the `'units'` entry of the DataArray's
        `.attrs`. Will raise a ValueError if the DataArray already contains a
        unit-aware array.

        Parameters
        ----------
        units : pint.Unit or str or mapping of hashable to , optional
            Physical units to use for this DataArray. If not provided, will try
            to read them from ``DataArray.attrs['units']`` using pint's parser.
        unit_registry : pint.UnitRegistry, optional
            Unit registry to be used for the units attached to this DataArray.
            If not given then a default registry will be created.
        registry_kwargs : dict, optional
            Keyword arguments to be passed to `pint.UnitRegistry`.

        Returns
        -------
        quantified : DataArray
            DataArray whose wrapped array data will now be a Quantity
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
            raise ValueError(
                f"Cannot attach unit {units} to quantity: data "
                f"already has units {self.da.data.units}"
            )

        if isinstance(units, (str, pint.Unit)):
            if self.da.name in unit_kwargs:
                raise ValueError(
                    f"ambiguous values given for {repr(self.da.name)}:"
                    f" {repr(units)} and {repr(unit_kwargs[self.da.name])}"
                )
            unit_kwargs[self.da.name] = units
            units = None

        units = either_dict_or_kwargs(units, unit_kwargs, ".quantify")

        registry = _get_registry(unit_registry, registry_kwargs)

        # TODO should we (temporarily) remove the attrs here so that they don't become inconsistent?
        unit_attrs = conversion.extract_unit_attributes(self.da, delete=False)

        units = {
            name: _decide_units(unit, registry, unit_attribute)
            for name, (unit, unit_attribute) in zip_mappings(units, unit_attrs).items()
            if unit is not None or unit_attribute is not None
        }

        return conversion.attach_units(self.da, units)

    def dequantify(self):
        """
        Removes units from the DataArray and it's coordinates.

        Will replace `.attrs['units']` on each variable with a string
        representation of the `pint.Unit` instance.

        Returns
        -------
        dequantified : DataArray
            DataArray whose array data is unitless, and of the type
            that was previously wrapped by `pint.Quantity`.
        """

        units = units_to_str_or_none(conversion.extract_units(self.da))
        new_obj = conversion.attach_unit_attributes(
            conversion.strip_units(self.da), units,
        )

        return new_obj

    @property
    def magnitude(self):
        return self.da.data.magnitude

    @property
    def units(self):
        return self.da.data.units

    @units.setter
    def units(self, units):
        quantity = _array_attach_units(self.da.data, units)
        self.da = DataArray(
            dim=self.da.dims, data=quantity, coords=self.da.coords, attrs=self.da.attrs
        )

    @property
    def dimensionality(self):
        return self.da.data.dimensionality

    @property
    def registry(self):
        # TODO is this a bad idea? (see GH issue #1071 in pint)
        return self.data._REGISTRY

    @registry.setter
    def registry(self, _):
        raise AttributeError("Don't try to change the registry once created")

    def to(self, units=None, **unit_kwargs):
        """ convert the quantities in a DataArray

        Parameters
        ----------
        units : str or pint.Unit or mapping of hashable to str or pint.Unit, optional
            The units to convert to. If a unit name or
            :py:class`pint.Unit` object, convert the DataArray's
            data. If a dict-like, it has to map a variable name to a
            unit name or :py:class:`pint.Unit` object.
        **unit_kwargs
            The kwargs form of ``units``. Can only be used for
            variable names that are strings and valid python identifiers.

        Returns
        -------
        object : DataArray
            A new object with converted units.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     data=np.linspace(0, 1, 5) * ureg.m,
        ...     coords={"u": ("x", np.arange(5) * ureg.s)},
        ...     dims="x",
        ...     name="arr",
        ... )
        >>> da
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
        Coordinates:
            u        (x) int64 <Quantity([0 1 2 3 4], 'second')>
        Dimensions without coordinates: x

        Convert the data

        >>> da.pint.to("mm")
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 <Quantity([0 1 2 3 4], 'second')>
        Dimensions without coordinates: x
        >>> da.pint.to(ureg.mm)
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 <Quantity([0 1 2 3 4], 'second')>
        Dimensions without coordinates: x
        >>> da.pint.to({da.name: "mm"})
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 <Quantity([0 1 2 3 4], 'second')>
        Dimensions without coordinates: x

        Convert coordinates

        >>> da.pint.to({"u": ureg.ms})
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x
        >>> da.pint.to(u="ms")
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x

        Convert both simultaneously

        >>> da.pint.to("mm", u="ms")
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x
        >>> da.pint.to({"arr": ureg.mm, "u": ureg.ms})
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x
        >>> da.pint.to(arr="mm", u="ms")
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x
        """
        if isinstance(units, (str, pint.Unit)):
            unit_kwargs[self.da.name] = units
            units = None
        elif units is not None and not is_dict_like(units):
            raise ValueError(
                "units must be either a string, a pint.Unit object or a dict-like,"
                f" but got {units!r}"
            )

        units = either_dict_or_kwargs(units, unit_kwargs, "to")

        return conversion.convert_units(self.da, units)

    def to_base_units(self):
        quantity = self.da.data.to_base_units()
        return DataArray(
            dim=self.da.dims,
            data=quantity,
            coords=self.da.coords,
            attrs=self.da.attrs,
            encoding=self.da.encoding,
        )

    # TODO integrate with the uncertainties package here...?
    def plus_minus(self, value, error, relative=False):
        quantity = self.da.data.plus_minus(value, error, relative)
        return DataArray(
            dim=self.da.dims,
            data=quantity,
            coords=self.da.coords,
            attrs=self.da.attrs,
            encoding=self.da.encoding,
        )

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        ...

    @property
    def loc(self):
        ...


@register_dataset_accessor("pint")
class PintDatasetAccessor:
    """
    Access methods for DataArrays with units using Pint.

    Methods and attributes can be accessed through the `.pint` attribute.
    """

    def __init__(self, ds):
        self.ds = ds

    def quantify(
        self, units=None, unit_registry=None, registry_kwargs=None, **unit_kwargs
    ):
        """
        Attaches units to each variable in the Dataset.

        Units can be specified as a pint.Unit or as a string, which will
        be parsed by the given unit registry. If no units are specified then
        the units will be parsed from the `'units'` entry of the DataArray's
        `.attrs`. Will raise a ValueError if any of the DataArrays already
        contain a unit-aware array.

        Parameters
        ----------
        units : mapping from variable names to pint.Unit or str, optional
            Physical units to use for particular DataArrays in this Dataset. If
            not provided, will try to read them from
            `Dataset[var].attrs['units']` using pint's parser.
        unit_registry : `pint.UnitRegistry`, optional
            Unit registry to be used for the units attached to each DataArray
            in this Dataset. If not given then a default registry will be
            created.
        registry_kwargs : dict, optional
            Keyword arguments to be passed to `pint.UnitRegistry`.

        Returns
        -------
        quantified - Dataset whose variables will now contain Quantity
        arrays with units.
        """
        units = either_dict_or_kwargs(units, unit_kwargs, ".quantify")
        registry = _get_registry(unit_registry, registry_kwargs)

        # TODO should we (temporarily) remove the attrs here so that they don't become inconsistent?
        unit_attrs = conversion.extract_unit_attributes(self.ds, delete=False)
        units = {
            name: _decide_units(unit, registry, attr)
            for name, (unit, attr) in zip_mappings(units, unit_attrs).items()
            if unit is not None or attr is not None
        }

        return conversion.attach_units(self.ds, units)

    def dequantify(self):
        units = units_to_str_or_none(conversion.extract_units(self.ds))
        new_obj = conversion.attach_unit_attributes(
            conversion.strip_units(self.ds), units
        )
        return new_obj

    def to(self, units=None, **unit_kwargs):
        """ convert the quantities in a DataArray

        Parameters
        ----------
        units : mapping of hashable to str or pint.Unit, optional
            Maps variable names to the unit to convert to.
        **unit_kwargs
            The kwargs form of ``units``. Can only be used for
            variable names that are strings and valid python identifiers.

        Returns
        -------
        object : DataArray
            A new object with converted units.

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={
        ...         "a": ("x", np.linspace(0, 1, 5) * ureg.m),
        ...         "b": ("x", np.linspace(-1, 0, 5) * ureg.kg),
        ...     },
        ...     coords={"u": ("x", np.arange(5) * ureg.s)},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 <Quantity([0 1 2 3 4], 'second')>
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
            b        (x) float64 <Quantity([-1.   -0.75 -0.5  -0.25  0.  ], 'kilogram')>

        Convert the data

        >>> ds.pint.to({"a": "mm", "b": ureg.g})
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 <Quantity([0 1 2 3 4], 'second')>
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 <Quantity([   0.  250.  500.  750. 1000.], 'millimet...
            b        (x) float64 <Quantity([-1000.  -750.  -500.  -250.     0.], 'gra...
        >>> ds.pint.to(a=ureg.mm, b="g")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 <Quantity([0 1 2 3 4], 'second')>
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 <Quantity([   0.  250.  500.  750. 1000.], 'millimet...
            b        (x) float64 <Quantity([-1000.  -750.  -500.  -250.     0.], 'gra...

        Convert coordinates

        >>> ds.pint.to({"u": ureg.ms})
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
            b        (x) float64 <Quantity([-1.   -0.75 -0.5  -0.25  0.  ], 'kilogram')>
        >>> ds.pint.to(u="ms")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
            b        (x) float64 <Quantity([-1.   -0.75 -0.5  -0.25  0.  ], 'kilogram')>

        Convert both simultaneously

        >>> ds.pint.to(a=ureg.mm, b=ureg.g, u="ms")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 <Quantity([   0.  250.  500.  750. 1000.], 'millimet...
            b        (x) float64 <Quantity([-1000.  -750.  -500.  -250.     0.], 'gra...
        >>> ds.pint.to({"a": "mm", "b": "g", "u": ureg.ms})
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 <Quantity([   0. 1000. 2000. 3000. 4000.], 'millisec...
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 <Quantity([   0.  250.  500.  750. 1000.], 'millimet...
            b        (x) float64 <Quantity([-1000.  -750.  -500.  -250.     0.], 'gra...
        """
        units = either_dict_or_kwargs(units, unit_kwargs, "to")

        return conversion.convert_units(self.ds, units)

    def to_base_units(self):
        base_vars = {name: da.pint.to_base_units() for name, da in self.ds.items()}
        return Dataset(base_vars, coords=self.ds.coords, attrs=self.ds.attrs)

    # TODO unsure if the upstream capability exists in pint for this yet.
    def to_system(self, system):
        raise NotImplementedError

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        ...

    @property
    def loc(self):
        ...