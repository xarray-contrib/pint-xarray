# TODO is it possible to import pint-xarray from within xarray if pint is present?
import itertools

import pint
from pint.quantity import Quantity
from pint.unit import Unit
from xarray import register_dataarray_accessor, register_dataset_accessor
from xarray.core.dtypes import NA

from . import conversion
from .errors import DimensionalityError


def setup_registry(registry):
    """set up the given registry for use with pint_xarray

    Namely, it enables ``force_ndarray_like`` to make sure results are always
    duck arrays.

    Parameters
    ----------
    registry : pint.UnitRegistry
        The registry to modify
    """
    if not registry.force_ndarray and not registry.force_ndarray_like:
        registry.force_ndarray_like = True

    return registry


default_registry = setup_registry(pint.get_application_registry())

# TODO could/should we overwrite xr.open_dataset and xr.open_mfdataset to make
# them apply units upon loading???
# TODO could even override the decode_cf kwarg?

# TODO docstrings
# TODO type hints


def is_dict_like(obj):
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def zip_mappings(*mappings, fill_value=None):
    """zip mappings by combining values for common keys into a tuple

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


def merge_mappings(first, *mappings):
    result = first.copy()
    for mapping in mappings:
        result.update(
            {key: value for key, value in mapping.items() if value is not None}
        )

    return result


def units_to_str_or_none(mapping, unit_format):
    formatter = str if not unit_format else lambda v: unit_format.format(v)

    return {
        key: formatter(value) if isinstance(value, Unit) else value
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


def get_registry(unit_registry, new_units, existing_units):
    units = merge_mappings(existing_units, new_units)
    registries = {unit._REGISTRY for unit in units.values() if isinstance(unit, Unit)}

    if unit_registry is None:
        if not registries:
            unit_registry = default_registry
        elif len(registries) == 1:
            (unit_registry,) = registries
    registries.add(unit_registry)

    if len(registries) > 1 or unit_registry not in registries:
        raise ValueError(
            "using multiple unit registries in the same object is not supported"
        )

    if not unit_registry.force_ndarray_like and not unit_registry.force_ndarray:
        raise ValueError(
            "invalid registry. Please enable 'force_ndarray_like' or 'force_ndarray'."
        )

    return unit_registry


def _decide_units(units, registry, unit_attribute):
    if units is None and unit_attribute is None:
        # or warn and return None?
        raise ValueError("no units given")
    elif units is None:
        # TODO option to read and decode units according to CF conventions (see MetPy)?
        units = registry.parse_units(unit_attribute)
    elif isinstance(units, Unit):
        # TODO do we have to check what happens if someone passes a Unit instance
        # without creating a unit registry?
        # TODO and what happens if they pass in a Unit from a different registry
        pass
    else:
        units = registry.Unit(units)
    return units


@register_dataarray_accessor("pint")
class PintDataArrayAccessor:
    """
    Access methods for DataArrays with units using Pint.

    Methods and attributes can be accessed through the `.pint` attribute.
    """

    def __init__(self, da):
        self.da = da

    def quantify(self, units=None, unit_registry=None, **unit_kwargs):
        """
        Attach units to the DataArray.

        Units can be specified as a pint.Unit or as a string, which will be
        parsed by the given unit registry. If no units are specified then the
        units will be parsed from the `'units'` entry of the DataArray's
        `.attrs`. Will raise a ValueError if the DataArray already contains a
        unit-aware array.

        .. note::
            Be aware that unless you're using ``dask`` this will load
            the data into memory. To avoid that, consider converting
            to ``dask`` first (e.g. using ``chunk``).

            As units in dimension coordinates are not supported until
            ``xarray`` changes the way it implements indexes, these
            units will be set as attributes.

        Parameters
        ----------
        units : unit-like or mapping of hashable to unit-like, optional
            Physical units to use for this DataArray. If a str or
            pint.Unit, will be used as the DataArray's units. If a
            dict-like, it should map a variable name to the desired
            unit (use the DataArray's name to refer to its data). If
            not provided, will try to read them from
            ``DataArray.attrs['units']`` using pint's parser. The
            ``"units"`` attribute will be removed from all variables
            except from dimension coordinates.
        unit_registry : pint.UnitRegistry, optional
            Unit registry to be used for the units attached to this DataArray.
            If not given then a default registry will be created.
        **unit_kwargs
            Keyword argument form of units.

        Returns
        -------
        quantified : DataArray
            DataArray whose wrapped array data will now be a Quantity
            array with the specified units.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     data=[0.4, 0.9, 1.7, 4.8, 3.2, 9.1],
        ...     dims=["wavelength"],
        ...     coords={"wavelength": [1e-4, 2e-4, 4e-4, 6e-4, 1e-3, 2e-3]},
        ... )
        >>> da.pint.quantify(units="Hz")
        <xarray.DataArray (wavelength: 6)>
        <Quantity([0.4 0.9 1.7 4.8 3.2 9.1], 'hertz')>
        Coordinates:
          * wavelength  (wavelength) float64 0.0001 0.0002 0.0004 0.0006 0.001 0.002
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

        units = either_dict_or_kwargs(units, unit_kwargs, "quantify")

        registry = get_registry(unit_registry, units, conversion.extract_units(self.da))

        unit_attrs = conversion.extract_unit_attributes(self.da)

        units = {
            name: _decide_units(unit, registry, unit_attribute)
            for name, (unit, unit_attribute) in zip_mappings(units, unit_attrs).items()
            if unit is not None or unit_attribute is not None
        }

        return self.da.pipe(conversion.strip_unit_attributes).pipe(
            conversion.attach_units, units
        )

    def dequantify(self, format=None):
        """
        Convert the units of the DataArray to string attributes.

        Will replace ``.attrs['units']`` on each variable with a string
        representation of the ``pint.Unit`` instance.

        Parameters
        ----------
        format : str, optional
            The format used for the string representations.

        Returns
        -------
        dequantified : DataArray
            DataArray whose array data is unitless, and of the type
            that was previously wrapped by `pint.Quantity`.
        """
        units = conversion.extract_unit_attributes(self.da)
        units.update(conversion.extract_units(self.da))

        unit_format = f"{{:{format}}}" if isinstance(format, str) else format

        units = units_to_str_or_none(units, unit_format)
        return (
            self.da.pipe(conversion.strip_units)
            .pipe(conversion.strip_unit_attributes)
            .pipe(conversion.attach_unit_attributes, units)
        )

    @property
    def magnitude(self):
        """the magnitude of the data or the data itself if not a quantity."""
        data = self.da.data
        return getattr(data, "magnitude", data)

    @property
    def units(self):
        """the units of the data or :py:obj:`None` if not a quantity.

        Setting the units is possible, but only if the data is not already a quantity.
        """
        return getattr(self.da.data, "units", None)

    @units.setter
    def units(self, units):
        self.da.data = conversion.array_attach_units(self.da.data, units)

    @property
    def dimensionality(self):
        """get the dimensionality of the data or :py:obj:`None` if not a quantity."""
        return getattr(self.da.data, "dimensionality", None)

    @property
    def registry(self):
        # TODO is this a bad idea? (see GH issue #1071 in pint)
        return getattr(self.da.data, "_REGISTRY", None)

    @registry.setter
    def registry(self, _):
        raise AttributeError("Don't try to change the registry once created")

    def to(self, units=None, **unit_kwargs):
        """convert the quantities in a DataArray

        Parameters
        ----------
        units : unit-like or mapping of hashable to unit-like, optional
            The units to convert to. If a unit name or ``pint.Unit``
            object, convert the DataArray's data. If a dict-like, it
            has to map a variable name to a unit name or ``pint.Unit``
            object.
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
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x

        Convert the data

        >>> da.pint.to("mm")
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x
        >>> da.pint.to(ureg.mm)
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x
        >>> da.pint.to({da.name: "mm"})
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x

        Convert coordinates

        >>> da.pint.to({"u": ureg.ms})
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        >>> da.pint.to(u="ms")
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x

        Convert both simultaneously

        >>> da.pint.to("mm", u="ms")
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        >>> da.pint.to({"arr": ureg.mm, "u": ureg.ms})
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        >>> da.pint.to(arr="mm", u="ms")
        <xarray.DataArray 'arr' (x: 5)>
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
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

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        """unit-aware version of sel

        Just like :py:meth:`DataArray.sel`, except the dataset's indexes are converted
        to the units of the indexers first.

        .. note::
            ``tolerance`` is not supported, yet. It will be passed through to
            ``DataArray.sel`` unmodified.

        See Also
        --------
        xarray.Dataset.pint.sel
        xarray.DataArray.sel
        xarray.Dataset.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")

        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
        }

        # TODO: handle tolerance

        # make sure we only have compatible units
        dims = self.da.dims
        unit_attrs = conversion.extract_unit_attributes(self.da)
        index_units = {
            name: units for name, units in unit_attrs.items() if name in dims
        }

        registry = get_registry(None, index_units, indexer_units)

        units = zip_mappings(indexer_units, index_units)
        incompatible_units = [
            key
            for key, (indexer_unit, index_unit) in units.items()
            if (
                None not in (indexer_unit, index_unit)
                and not registry.is_compatible_with(indexer_unit, index_unit)
            )
        ]
        if incompatible_units:
            units1 = {key: indexer_units[key] for key in incompatible_units}
            units2 = {key: index_units[key] for key in incompatible_units}
            raise DimensionalityError(units1, units2)

        # convert the indexes to the indexer's units
        converted = conversion.convert_units(self.da, indexer_units)

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in indexers.items()
        }
        indexed = converted.sel(
            stripped_indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
        )

        return indexed

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

    def quantify(self, units=None, unit_registry=None, **unit_kwargs):
        """
        Attach units to the variables of the Dataset.

        Units can be specified as a ``pint.Unit`` or as a
        string, which will be parsed by the given unit registry. If no
        units are specified then the units will be parsed from the
        ``"units"`` entry of the Dataset variable's ``.attrs``. Will
        raise a ValueError if any of the variables already contain a
        unit-aware array.

        .. note::
            Be aware that unless you're using ``dask`` this will load
            the data into memory. To avoid that, consider converting
            to ``dask`` first (e.g. using ``chunk``).

            As units in dimension coordinates are not supported until
            ``xarray`` changes the way it implements indexes, these
            units will be set as attributes.

        Parameters
        ----------
        units : mapping of hashable to unit-like, optional
            Physical units to use for particular DataArrays in this
            Dataset. It should map variable names to units (unit names
            or ``pint.Unit`` objects). If not provided, will try to
            read them from ``Dataset[var].attrs['units']`` using
            pint's parser. The ``"units"`` attribute will be removed
            from all variables except from dimension coordinates.
        unit_registry : pint.UnitRegistry, optional
            Unit registry to be used for the units attached to each
            DataArray in this Dataset. If not given then a default
            registry will be created.
        **unit_kwargs
            Keyword argument form of ``units``.

        Returns
        -------
        quantified : Dataset
            Dataset whose variables will now contain Quantity arrays
            with units.

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {"a": ("x", [0, 3, 2], {"units": "m"}), "b": ("x", [5, -2, 1])},
        ...     coords={"x": [0, 1, 2], "u": ("x", [-1, 0, 1], {"units": "s"})},
        ... )

        >>> ds.pint.quantify()
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 0 1 2
            u        (x) int64 [s] -1 0 1
        Data variables:
            a        (x) int64 [m] 0 3 2
            b        (x) int64 5 -2 1
        >>> ds.pint.quantify({"b": "dm"})
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 0 1 2
            u        (x) int64 [s] -1 0 1
        Data variables:
            a        (x) int64 [m] 0 3 2
            b        (x) int64 [dm] 5 -2 1
        """
        units = either_dict_or_kwargs(units, unit_kwargs, "quantify")
        registry = get_registry(unit_registry, units, conversion.extract_units(self.ds))

        unit_attrs = conversion.extract_unit_attributes(self.ds)

        units = {
            name: _decide_units(unit, registry, attr)
            for name, (unit, attr) in zip_mappings(units, unit_attrs).items()
            if unit is not None or attr is not None
        }

        return self.ds.pipe(conversion.strip_unit_attributes).pipe(
            conversion.attach_units, units
        )

    def dequantify(self, format=None):
        """
        Convert units from the Dataset to string attributes.

        Will replace ``.attrs['units']`` on each variable with a string
        representation of the ``pint.Unit`` instance.

        Parameters
        ----------
        format : str, optional
            The format used for the string representations.

        Returns
        -------
        dequantified : Dataset
            Dataset whose data variables are unitless, and of the type
            that was previously wrapped by ``pint.Quantity``.
        """
        units = conversion.extract_unit_attributes(self.ds)
        units.update(conversion.extract_units(self.ds))

        unit_format = f"{{:{format}}}" if isinstance(format, str) else format

        units = units_to_str_or_none(units, unit_format)
        return (
            self.ds.pipe(conversion.strip_units)
            .pipe(conversion.strip_unit_attributes)
            .pipe(conversion.attach_unit_attributes, units)
        )

    def to(self, units=None, **unit_kwargs):
        """convert the quantities in a Dataset

        Parameters
        ----------
        units : unit-like or mapping of hashable to unit-like, optional
            The units to convert to. If a unit name or ``pint.Unit``
            object, convert all the Dataset's data variables. If a dict-like, it
            maps variable names to unit names or ``pint.Unit``
            objects.
        **unit_kwargs
            The kwargs form of ``units``. Can only be used for
            variable names that are strings and valid python identifiers.

        Returns
        -------
        object : Dataset
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
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [m] 0.0 0.25 0.5 0.75 1.0
            b        (x) float64 [kg] -1.0 -0.75 -0.5 -0.25 0.0

        Convert the data

        >>> ds.pint.to({"a": "mm", "b": ureg.g})
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [mm] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 [g] -1e+03 -750.0 -500.0 -250.0 0.0
        >>> ds.pint.to(a=ureg.mm, b="g")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [mm] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 [g] -1e+03 -750.0 -500.0 -250.0 0.0

        Convert coordinates

        >>> ds.pint.to({"u": ureg.ms})
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [m] 0.0 0.25 0.5 0.75 1.0
            b        (x) float64 [kg] -1.0 -0.75 -0.5 -0.25 0.0
        >>> ds.pint.to(u="ms")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [m] 0.0 0.25 0.5 0.75 1.0
            b        (x) float64 [kg] -1.0 -0.75 -0.5 -0.25 0.0

        Convert both simultaneously

        >>> ds.pint.to(a=ureg.mm, b=ureg.g, u="ms")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [mm] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 [g] -1e+03 -750.0 -500.0 -250.0 0.0
        >>> ds.pint.to({"a": "mm", "b": "g", "u": ureg.ms})
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [mm] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 [g] -1e+03 -750.0 -500.0 -250.0 0.0

        Convert homogeneous data

        >>> ds = xr.Dataset(
        ...     data_vars={
        ...         "a": ("x", np.linspace(0, 1, 5) * ureg.kg),
        ...         "b": ("x", np.linspace(-1, 0, 5) * ureg.mg),
        ...     },
        ...     coords={"u": ("x", np.arange(5) * ureg.s)},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [kg] 0.0 0.25 0.5 0.75 1.0
            b        (x) float64 [mg] -1.0 -0.75 -0.5 -0.25 0.0
        >>> ds.pint.to("g")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 [g] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 [g] -0.001 -0.00075 -0.0005 -0.00025 0.0
        """
        if isinstance(units, (str, pint.Unit)):
            unit_kwargs.update(
                {name: units for name in self.ds.keys() if name not in unit_kwargs}
            )
            units = None
        elif units is not None and not is_dict_like(units):
            raise ValueError(
                "units must be either a string, a pint.Unit object or a dict-like,"
                f" but got {units!r}"
            )

        units = either_dict_or_kwargs(units, unit_kwargs, "to")

        return conversion.convert_units(self.ds, units)

    def reindex(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        copy=True,
        fill_value=NA,
        **indexers_kwargs,
    ):
        """unit-aware version of reindex"""
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "reindex")

        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
        }

        # TODO: handle tolerance
        # TODO: handle fill_value

        # make sure we only have compatible units
        dims = self.ds.dims
        unit_attrs = conversion.extract_unit_attributes(self.ds)
        index_units = {
            name: units for name, units in unit_attrs.items() if name in dims
        }

        registry = get_registry(None, index_units, indexer_units)

        units = zip_mappings(indexer_units, index_units)
        incompatible_units = [
            key
            for key, (indexer_unit, index_unit) in units.items()
            if (
                None not in (indexer_unit, index_unit)
                and not registry.is_compatible_with(indexer_unit, index_unit)
            )
        ]
        if incompatible_units:
            units1 = {key: indexer_units[key] for key in incompatible_units}
            units2 = {key: index_units[key] for key in incompatible_units}
            raise DimensionalityError(units1, units2)

        # convert the indexes to the indexer's units
        converted = conversion.convert_units(self.ds, indexer_units)

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in indexers.items()
        }
        indexed = converted.reindex(
            stripped_indexers,
            method=method,
            tolerance=tolerance,
            copy=copy,
            fill_value=fill_value,
        )
        return indexed

    def reindex_like(
        self, other, method=None, tolerance=None, copy=True, fill_value=NA
    ):
        """unit-aware version of reindex_like"""
        from xarray.core import alignment

        indexers = alignment.reindex_like_indexers(self, other)
        return self.reindex(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            copy=copy,
            fill_value=fill_value,
        )

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        """unit-aware version of sel

        Just like :py:meth:`xarray.Dataset.sel`, except the dataset's indexes are converted to the units
        of the indexers first.

        .. note::
            ``tolerance`` is not supported, yet. It will be passed through to
            ``Dataset.sel`` unmodified.

        See Also
        --------
        xarray.DataArray.pint.sel
        xarray.Dataset.sel
        xarray.DataArray.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")

        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
        }

        # TODO: handle tolerance

        # make sure we only have compatible units
        dims = self.ds.dims
        unit_attrs = conversion.extract_unit_attributes(self.ds)
        index_units = {
            name: units for name, units in unit_attrs.items() if name in dims
        }

        registry = get_registry(None, index_units, indexer_units)

        units = zip_mappings(indexer_units, index_units)
        incompatible_units = [
            key
            for key, (indexer_unit, index_unit) in units.items()
            if (
                None not in (indexer_unit, index_unit)
                and not registry.is_compatible_with(indexer_unit, index_unit)
            )
        ]
        if incompatible_units:
            units1 = {key: indexer_units[key] for key in incompatible_units}
            units2 = {key: index_units[key] for key in incompatible_units}
            raise DimensionalityError(units1, units2)

        # convert the indexes to the indexer's units
        converted = conversion.convert_units(self.ds, indexer_units)

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in indexers.items()
        }
        indexed = converted.sel(
            stripped_indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
        )

        return indexed

    @property
    def loc(self):
        ...
