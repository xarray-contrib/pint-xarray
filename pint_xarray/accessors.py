# TODO is it possible to import pint-xarray from within xarray if pint is present?
import itertools

import pint
from pint import Unit
from xarray import register_dataarray_accessor, register_dataset_accessor
from xarray.core.dtypes import NA

from . import conversion
from .conversion import no_unit_values
from .errors import format_error_message

_default = object()


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


def units_to_str_or_none(mapping, unit_format):
    formatter = str if not unit_format else lambda v: unit_format.format(v)

    return {
        key: formatter(value) if isinstance(value, Unit) else value
        for key, value in mapping.items()
    }


# based on xarray.core.utils.either_dict_or_kwargs
# https://github.com/pydata/xarray/blob/v0.15.1/xarray/core/utils.py#L249-L268
def either_dict_or_kwargs(positional, keywords, method_name):
    if positional not in (_default, None):
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
    units = itertools.chain(new_units.values(), existing_units.values())
    registries = {unit._REGISTRY for unit in units if isinstance(unit, Unit)}

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
    if units is _default and unit_attribute in (None, _default):
        # or warn and return None?
        raise ValueError("no units given")
    elif units in no_unit_values or isinstance(units, Unit):
        # TODO what happens if they pass in a Unit from a different registry
        return units
    elif units is _default:
        if unit_attribute in no_unit_values:
            return unit_attribute
        if isinstance(unit_attribute, Unit):
            units = unit_attribute
        else:
            units = registry.parse_units(unit_attribute)
    else:
        units = registry.parse_units(units)
    return units


class DatasetLocIndexer:
    __slots__ = ("ds",)

    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, indexers):
        if not is_dict_like(indexers):
            raise NotImplementedError("pandas-style indexing is not supported, yet")

        dims = self.ds.dims
        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
            if name in dims
        }

        # convert the indexes to the indexer's units
        try:
            converted = conversion.convert_units(self.ds, indexer_units)
        except ValueError as e:
            raise KeyError(*e.args) from e

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in indexers.items()
        }
        return converted.loc[stripped_indexers]


class DataArrayLocIndexer:
    __slots__ = ("da",)

    def __init__(self, da):
        self.da = da

    def __getitem__(self, indexers):
        if not is_dict_like(indexers):
            raise NotImplementedError("pandas-style indexing is not supported, yet")

        dims = self.da.dims
        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
            if name in dims
        }

        # convert the indexes to the indexer's units
        try:
            converted = conversion.convert_units(self.da, indexer_units)
        except ValueError as e:
            raise KeyError(*e.args) from e

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in indexers.items()
        }
        return converted.loc[stripped_indexers]

    def __setitem__(self, indexers, values):
        if not is_dict_like(indexers):
            raise NotImplementedError("pandas-style indexing is not supported, yet")

        dims = self.da.dims
        unit_attrs = conversion.extract_unit_attributes(self.da)
        index_units = {
            name: units for name, units in unit_attrs.items() if name in dims
        }

        # convert the indexers to the index units
        try:
            converted = conversion.convert_indexer_units(indexers, index_units)
        except ValueError as e:
            raise KeyError(*e.args) from e

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in converted.items()
        }
        self.da.loc[stripped_indexers] = values


@register_dataarray_accessor("pint")
class PintDataArrayAccessor:
    """
    Access methods for DataArrays with units using Pint.

    Methods and attributes can be accessed through the `.pint` attribute.
    """

    def __init__(self, da):
        self.da = da

    def quantify(self, units=_default, unit_registry=None, **unit_kwargs):
        """
        Attach units to the DataArray.

        Units can be specified as a pint.Unit or as a string, which will be
        parsed by the given unit registry. If no units are specified then the
        units will be parsed from the `'units'` entry of the DataArray's
        `.attrs`. Will raise a ValueError if the DataArray already contains a
        unit-aware array with a different unit.

        .. note::
            Be aware that unless you're using ``dask`` this will load
            the data into memory. To avoid that, consider converting
            to ``dask`` first (e.g. using ``chunk``).

        .. warning::

            As units in dimension coordinates are not supported until
            ``xarray`` changes the way it implements indexes, these
            units will be set as attributes.

        .. note::
            Also note that datetime units (i.e. ones that match
            ``{units} since {date}``) in unit attributes will be
            ignored, to avoid interfering with ``xarray``'s datetime
            encoding / decoding.

        Parameters
        ----------
        units : unit-like or mapping of hashable to unit-like, optional
            Physical units to use for this DataArray. If a str or
            pint.Unit, will be used as the DataArray's units. If a
            dict-like, it should map a variable name to the desired
            unit (use the DataArray's name to refer to its data). If
            not provided, ``quantify`` will try to read them from
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

        Notes
        -----
        ``"none"`` and ``None`` can be used to mark variables that should not
        be quantified.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     data=[0.4, 0.9, 1.7, 4.8, 3.2, 9.1],
        ...     dims=["wavelength"],
        ...     coords={"wavelength": [1e-4, 2e-4, 4e-4, 6e-4, 1e-3, 2e-3]},
        ... )
        >>> da.pint.quantify(units="Hz")
        <xarray.DataArray (wavelength: 6)> Size: 48B
        <Quantity([0.4 0.9 1.7 4.8 3.2 9.1], 'hertz')>
        Coordinates:
          * wavelength  (wavelength) float64 48B 0.0001 0.0002 0.0004 0.0006 0.001 0.002

        Don't quantify the data:

        >>> da = xr.DataArray(
        ...     data=[0.4, 0.9],
        ...     dims=["wavelength"],
        ...     attrs={"units": "Hz"},
        ... )
        >>> da.pint.quantify(units=None)
        <xarray.DataArray (wavelength: 2)> Size: 16B
        array([0.4, 0.9])
        Dimensions without coordinates: wavelength

        Quantify with the same unit:

        >>> q = da.pint.quantify()
        >>> q
        <xarray.DataArray (wavelength: 2)> Size: 16B
        <Quantity([0.4 0.9], 'hertz')>
        Dimensions without coordinates: wavelength
        >>> q.pint.quantify("Hz")
        <xarray.DataArray (wavelength: 2)> Size: 16B
        <Quantity([0.4 0.9], 'hertz')>
        Dimensions without coordinates: wavelength
        """
        if units is None or isinstance(units, (str, pint.Unit)):
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

        possible_new_units = zip_mappings(units, unit_attrs, fill_value=_default)
        new_units = {}
        invalid_units = {}
        for name, (unit, attr) in possible_new_units.items():
            if unit not in (_default, None) or attr not in (_default, None):
                try:
                    new_units[name] = _decide_units(unit, registry, attr)
                except (ValueError, pint.UndefinedUnitError) as e:
                    if unit not in (_default, None):
                        type = "parameter"
                        reported_unit = unit
                    else:
                        type = "attribute"
                        reported_unit = attr

                    invalid_units[name] = (reported_unit, type, e)

        if invalid_units:
            raise ValueError(format_error_message(invalid_units, "parse"))

        existing_units = {
            name: unit
            for name, unit in conversion.extract_units(self.da).items()
            if isinstance(unit, Unit)
        }
        overwritten_units = {
            name: (old, new)
            for name, (old, new) in zip_mappings(
                existing_units, new_units, fill_value=_default
            ).items()
            if old is not _default and new is not _default and old != new
        }
        if overwritten_units:
            errors = {
                name: (
                    new,
                    ValueError(
                        f"Cannot attach unit {repr(new)} to quantity: data "
                        f"already has units {repr(old)}"
                    ),
                )
                for name, (old, new) in overwritten_units.items()
            }
            raise ValueError(format_error_message(errors, "attach"))

        return self.da.pipe(conversion.strip_unit_attributes).pipe(
            conversion.attach_units, new_units
        )

    def dequantify(self, format=None):
        r"""
        Convert the units of the DataArray to string attributes.

        Will replace ``.attrs['units']`` on each variable with a string
        representation of the ``pint.Unit`` instance.

        Parameters
        ----------
        format : str, default: None
            The format specification (as accepted by pint) used for the string
            representations. If ``None``, the registry's default
            (:py:attr:`pint.UnitRegistry.default_format`) is used instead.

        Returns
        -------
        dequantified : DataArray
            DataArray whose array data is unitless, and of the type
            that was previously wrapped by `pint.Quantity`.

        See Also
        --------
        :doc:`pint:user/formatting`
            pint's string formatting guide

        Examples
        --------
        >>> da = xr.DataArray([0, 1], dims="x")
        >>> q = da.pint.quantify("m / s")
        >>> q
        <xarray.DataArray (x: 2)> Size: 16B
        <Quantity([0 1], 'meter / second')>
        Dimensions without coordinates: x

        >>> q.pint.dequantify(format="P")
        <xarray.DataArray (x: 2)> Size: 16B
        array([0, 1])
        Dimensions without coordinates: x
        Attributes:
            units:    meter/second
        >>> q.pint.dequantify(format="~P")
        <xarray.DataArray (x: 2)> Size: 16B
        array([0, 1])
        Dimensions without coordinates: x
        Attributes:
            units:    m/s

        Use the registry's default format

        >>> pint_xarray.unit_registry.default_format = "~L"
        >>> q.pint.dequantify()
        <xarray.DataArray (x: 2)> Size: 16B
        array([0, 1])
        Dimensions without coordinates: x
        Attributes:
            units:    \frac{\mathrm{m}}{\mathrm{s}}
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
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x

        Convert the data

        >>> da.pint.to("mm")
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x
        >>> da.pint.to(ureg.mm)
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x
        >>> da.pint.to({da.name: "mm"})
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x

        Convert coordinates

        >>> da.pint.to({"u": ureg.ms})
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        >>> da.pint.to(u="ms")
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([0.   0.25 0.5  0.75 1.  ], 'meter')>
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x

        Convert both simultaneously

        >>> da.pint.to("mm", u="ms")
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        >>> da.pint.to({"arr": ureg.mm, "u": ureg.ms})
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        >>> da.pint.to(arr="mm", u="ms")
        <xarray.DataArray 'arr' (x: 5)> Size: 40B
        <Quantity([   0.  250.  500.  750. 1000.], 'millimeter')>
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
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

    def chunk(self, chunks, name_prefix="xarray-", token=None, lock=False):
        """unit-aware version of chunk

        Like :py:meth:`xarray.DataArray.chunk`, but chunking a quantity will change the
        wrapped type to ``dask``.

        .. note::
            It is recommended to only use this when chunking in-memory arrays. To
            rechunk please use :py:meth:`xarray.DataArray.chunk`.

        See Also
        --------
        xarray.DataArray.chunk
        xarray.Dataset.pint.chunk
        """
        units = conversion.extract_units(self.da)
        stripped = conversion.strip_units(self.da)

        chunked = stripped.chunk(
            chunks, name_prefix=name_prefix, token=token, lock=lock
        )
        return conversion.attach_units(chunked, units)

    def reindex(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        copy=True,
        fill_value=NA,
        **indexers_kwargs,
    ):
        """unit-aware version of reindex

        Like :py:meth:`xarray.DataArray.reindex`, except the object's indexes are
        converted to the units of the indexers first.

        .. note::
            ``tolerance`` and ``fill_value`` are not supported, yet. They will be passed
            through to ``DataArray.reindex`` unmodified.

        See Also
        --------
        xarray.Dataset.pint.reindex
        xarray.DataArray.pint.reindex_like
        xarray.DataArray.reindex
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "reindex")

        dims = self.da.dims
        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
            if name in dims
        }

        # TODO: handle tolerance
        # TODO: handle fill_value

        # convert the indexes to the indexer's units
        converted = conversion.convert_units(self.da, indexer_units)

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
        """unit-aware version of reindex_like

        Like :py:meth:`xarray.DataArray.reindex_like`, except the object's indexes
        are converted to the units of the indexers first.

        .. note::
            ``tolerance`` and ``fill_value`` are not supported, yet. They will be passed
            through to ``DataArray.reindex_like`` unmodified.

        See Also
        --------
        xarray.Dataset.pint.reindex_like
        xarray.DataArray.pint.reindex
        xarray.DataArray.reindex_like
        """
        indexer_units = conversion.extract_unit_attributes(other)

        # TODO: handle tolerance
        # TODO: handle fill_value

        converted = conversion.convert_units(self.da, indexer_units)
        return converted.reindex_like(
            other,
            method=method,
            tolerance=tolerance,
            copy=copy,
            fill_value=fill_value,
        )

    def interp(
        self,
        coords=None,
        method="linear",
        assume_sorted=False,
        kwargs=None,
        **coords_kwargs,
    ):
        """unit-aware version of interp

        Like :py:meth:`xarray.DataArray.interp`, except the object's indexes are
        converted to the units of the indexers first.

        .. note::
            ``kwargs`` is passed unmodified to ``DataArray.interp``

        See Also
        --------
        xarray.Dataset.pint.interp
        xarray.DataArray.pint.interp_like
        xarray.DataArray.interp
        """
        indexers = either_dict_or_kwargs(coords, coords_kwargs, "interp")

        dims = self.da.dims
        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
            if name in dims
        }

        # convert the indexes to the indexer's units
        converted = conversion.convert_units(self.da, indexer_units)
        units = conversion.extract_units(converted)
        stripped = conversion.strip_units(converted)

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in indexers.items()
        }
        interpolated = stripped.interp(
            stripped_indexers,
            method=method,
            assume_sorted=False,
            kwargs=None,
        )
        return conversion.attach_units(interpolated, units)

    def interp_like(self, other, method="linear", assume_sorted=False, kwargs=None):
        """unit-aware version of interp_like

        Like :py:meth:`xarray.DataArray.interp_like`, except the object's indexes are converted
        to the units of the indexers first.

        .. note::
            ``kwargs`` is passed unmodified to ``DataArray.interp``

        See Also
        --------
        xarray.Dataset.pint.interp_like
        xarray.DataArray.pint.interp
        xarray.DataArray.interp_like
        """
        indexer_units = conversion.extract_unit_attributes(other)

        converted = conversion.convert_units(self.da, indexer_units)
        units = conversion.extract_units(converted)
        stripped = conversion.strip_units(converted)
        interpolated = stripped.interp_like(
            other,
            method=method,
            assume_sorted=assume_sorted,
            kwargs=kwargs,
        )
        return conversion.attach_units(interpolated, units)

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        """unit-aware version of sel

        Like :py:meth:`xarray.DataArray.sel`, except the object's indexes are converted
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

        dims = self.da.dims
        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
            if name in dims
        }

        # TODO: handle tolerance

        # convert the indexes to the indexer's units
        try:
            converted = conversion.convert_units(self.da, indexer_units)
        except ValueError as e:
            raise KeyError(*e.args) from e

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
        """Unit-aware attribute for indexing

        .. note::
           Position based indexing (e.g. ``ds.loc[1, 2:]``) is not supported, yet

        See Also
        --------
        xarray.DataArray.loc
        """
        return DataArrayLocIndexer(self.da)

    def drop_sel(self, labels=None, *, errors="raise", **labels_kwargs):
        """unit-aware version of drop_sel

        Just like :py:meth:`xarray.DataArray.drop_sel`, except the indexers are converted
        to the units of the object's indexes first.

        See Also
        --------
        xarray.Dataset.pint.drop_sel
        xarray.DataArray.drop_sel
        xarray.Dataset.drop_sel
        """
        indexers = either_dict_or_kwargs(labels, labels_kwargs, "drop_sel")

        dims = self.da.dims
        unit_attrs = conversion.extract_unit_attributes(self.da)
        index_units = {
            name: units for name, units in unit_attrs.items() if name in dims
        }

        # convert the indexers to the indexes units
        try:
            converted_indexers = conversion.convert_indexer_units(indexers, index_units)
        except ValueError as e:
            raise KeyError(*e.args) from e

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in converted_indexers.items()
        }
        indexed = self.da.drop_sel(
            stripped_indexers,
            errors=errors,
        )

        return indexed

    def ffill(self, dim, limit=None):
        """unit-aware version of ffill

        Like :py:meth:`xarray.DataArray.ffill` but without stripping the data units.

        See Also
        --------
        xarray.DataArray.ffill
        xarray.DataArray.pint.bfill
        """
        units = conversion.extract_units(self.da)
        stripped = conversion.strip_units(self.da)

        filled = stripped.ffill(dim=dim, limit=limit)

        return conversion.attach_units(filled, units)

    def bfill(self, dim, limit=None):
        """unit-aware version of bfill

        Like :py:meth:`xarray.DataArray.bfill` but without stripping the data units.

        See Also
        --------
        xarray.DataArray.bfill
        xarray.DataArray.pint.ffill
        """
        units = conversion.extract_units(self.da)
        stripped = conversion.strip_units(self.da)

        filled = stripped.bfill(dim=dim, limit=limit)

        return conversion.attach_units(filled, units)

    def interpolate_na(
        self,
        dim=None,
        method="linear",
        limit=None,
        use_coordinate=True,
        max_gap=None,
        keep_attrs=None,
        **kwargs,
    ):
        """unit-aware version of interpolate_na

        Like :py:meth:`xarray.DataArray.interpolate_na` but without stripping the units
        on data or coordinates.

        .. note::
            ``max_gap`` is not supported, yet, and will be passed through to
            ``DataArray.interpolate_na`` unmodified.

        See Also
        --------
        xarray.Dataset.pint.interpolate_na
        xarray.DataArray.interpolate_na
        """
        units = conversion.extract_units(self.da)
        stripped = conversion.strip_units(self.da)

        interpolated = stripped.interpolate_na(
            dim=dim,
            method=method,
            limit=limit,
            use_coordinate=use_coordinate,
            max_gap=max_gap,
            keep_attrs=keep_attrs,
            **kwargs,
        )

        return conversion.attach_units(interpolated, units)


@register_dataset_accessor("pint")
class PintDatasetAccessor:
    """
    Access methods for DataArrays with units using Pint.

    Methods and attributes can be accessed through the `.pint` attribute.
    """

    def __init__(self, ds):
        self.ds = ds

    def quantify(self, units=_default, unit_registry=None, **unit_kwargs):
        """
        Attach units to the variables of the Dataset.

        Units can be specified as a ``pint.Unit`` or as a
        string, which will be parsed by the given unit registry. If no
        units are specified then the units will be parsed from the
        ``"units"`` entry of the Dataset variable's ``.attrs``. Will
        raise a ValueError if any of the variables already contain a
        unit-aware array with a different unit.

        .. note::
            Be aware that unless you're using ``dask`` this will load
            the data into memory. To avoid that, consider converting
            to ``dask`` first (e.g. using ``chunk``).

        .. warning::

            As units in dimension coordinates are not supported until
            ``xarray`` changes the way it implements indexes, these
            units will be set as attributes.

        .. note::
            Also note that datetime units (i.e. ones that match
            ``{units} since {date}``) in unit attributes will be
            ignored, to avoid interfering with ``xarray``'s datetime
            encoding / decoding.

        Parameters
        ----------
        units : mapping of hashable to unit-like, optional
            Physical units to use for particular DataArrays in this
            Dataset. It should map variable names to units (unit names
            or ``pint.Unit`` objects). If not provided, ``quantify``
            will try to read them from ``Dataset[var].attrs['units']``
            using pint's parser. The ``"units"`` attribute will be
            removed from all variables except from dimension coordinates.
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

        Notes
        -----
        ``"none"`` and ``None`` can be used to mark variables
        that should not be quantified.

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {"a": ("x", [0, 3, 2], {"units": "m"}), "b": ("x", [5, -2, 1])},
        ...     coords={"x": [0, 1, 2], "u": ("x", [-1, 0, 1], {"units": "s"})},
        ... )

        >>> ds.pint.quantify()
        <xarray.Dataset> Size: 96B
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 24B 0 1 2
            u        (x) int64 24B [s] -1 0 1
        Data variables:
            a        (x) int64 24B [m] 0 3 2
            b        (x) int64 24B 5 -2 1
        >>> ds.pint.quantify({"b": "dm"})
        <xarray.Dataset> Size: 96B
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 24B 0 1 2
            u        (x) int64 24B [s] -1 0 1
        Data variables:
            a        (x) int64 24B [m] 0 3 2
            b        (x) int64 24B [dm] 5 -2 1

        Don't quantify specific variables:

        >>> ds.pint.quantify({"a": None})
        <xarray.Dataset> Size: 96B
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 24B 0 1 2
            u        (x) int64 24B [s] -1 0 1
        Data variables:
            a        (x) int64 24B 0 3 2
            b        (x) int64 24B 5 -2 1

        Quantify with the same unit:

        >>> q = ds.pint.quantify()
        >>> q
        <xarray.Dataset> Size: 96B
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 24B 0 1 2
            u        (x) int64 24B [s] -1 0 1
        Data variables:
            a        (x) int64 24B [m] 0 3 2
            b        (x) int64 24B 5 -2 1
        >>> q.pint.quantify({"a": "m"})
        <xarray.Dataset> Size: 96B
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 24B 0 1 2
            u        (x) int64 24B [s] -1 0 1
        Data variables:
            a        (x) int64 24B [m] 0 3 2
            b        (x) int64 24B 5 -2 1
        """
        units = either_dict_or_kwargs(units, unit_kwargs, "quantify")
        registry = get_registry(unit_registry, units, conversion.extract_units(self.ds))

        unit_attrs = conversion.extract_unit_attributes(self.ds)

        possible_new_units = zip_mappings(units, unit_attrs, fill_value=_default)
        new_units = {}
        invalid_units = {}
        for name, (unit, attr) in possible_new_units.items():
            if unit is not _default or attr not in (None, _default):
                try:
                    new_units[name] = _decide_units(unit, registry, attr)
                except (ValueError, pint.UndefinedUnitError) as e:
                    if unit is not _default:
                        type = "parameter"
                        reported_unit = unit
                    else:
                        type = "attribute"
                        reported_unit = attr

                    invalid_units[name] = (reported_unit, type, e)

        if invalid_units:
            raise ValueError(format_error_message(invalid_units, "parse"))

        existing_units = {
            name: unit
            for name, unit in conversion.extract_units(self.ds).items()
            if isinstance(unit, Unit)
        }
        overwritten_units = {
            name: (old, new)
            for name, (old, new) in zip_mappings(
                existing_units, new_units, fill_value=_default
            ).items()
            if old is not _default and new is not _default and old != new
        }
        if overwritten_units:
            errors = {
                name: (
                    new,
                    ValueError(
                        f"Cannot attach unit {repr(new)} to quantity: data "
                        f"already has units {repr(old)}"
                    ),
                )
                for name, (old, new) in overwritten_units.items()
            }
            raise ValueError(format_error_message(errors, "attach"))

        return self.ds.pipe(conversion.strip_unit_attributes).pipe(
            conversion.attach_units, new_units
        )

    def dequantify(self, format=None):
        r"""
        Convert units from the Dataset to string attributes.

        Will replace ``.attrs['units']`` on each variable with a string
        representation of the ``pint.Unit`` instance.

        Parameters
        ----------
        format : str, default: None
            The format specification (as accepted by pint's unit formatter) used for the
            string representations. If ``None``, the registry's default
            (:py:attr:`pint.UnitRegistry.default_format`) is used instead.

        Returns
        -------
        dequantified : Dataset
            Dataset whose data variables are unitless, and of the type
            that was previously wrapped by ``pint.Quantity``.

        See Also
        --------
        :doc:`pint:user/formatting`
            pint's string formatting guide

        Examples
        --------
        >>> ds = xr.Dataset({"a": ("x", [0, 1]), "b": ("y", [2, 3, 4])})
        >>> q = ds.pint.quantify({"a": "m / s", "b": "s"})
        >>> q
        <xarray.Dataset> Size: 40B
        Dimensions:  (x: 2, y: 3)
        Dimensions without coordinates: x, y
        Data variables:
            a        (x) int64 16B [m/s] 0 1
            b        (y) int64 24B [s] 2 3 4

        >>> d = q.pint.dequantify(format="P")
        >>> d.a
        <xarray.DataArray 'a' (x: 2)> Size: 16B
        array([0, 1])
        Dimensions without coordinates: x
        Attributes:
            units:    meter/second
        >>> d.b
        <xarray.DataArray 'b' (y: 3)> Size: 24B
        array([2, 3, 4])
        Dimensions without coordinates: y
        Attributes:
            units:    second

        >>> d = q.pint.dequantify(format="~P")
        >>> d.a
        <xarray.DataArray 'a' (x: 2)> Size: 16B
        array([0, 1])
        Dimensions without coordinates: x
        Attributes:
            units:    m/s
        >>> d.b
        <xarray.DataArray 'b' (y: 3)> Size: 24B
        array([2, 3, 4])
        Dimensions without coordinates: y
        Attributes:
            units:    s

        Use the registry's default format

        >>> pint_xarray.unit_registry.default_format = "~L"
        >>> d = q.pint.dequantify()
        >>> d.a
        <xarray.DataArray 'a' (x: 2)> Size: 16B
        array([0, 1])
        Dimensions without coordinates: x
        Attributes:
            units:    \frac{\mathrm{m}}{\mathrm{s}}
        >>> d.b
        <xarray.DataArray 'b' (y: 3)> Size: 24B
        array([2, 3, 4])
        Dimensions without coordinates: y
        Attributes:
            units:    \mathrm{s}
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
            object, convert all the object's data variables. If a dict-like, it
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
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [m] 0.0 0.25 0.5 0.75 1.0
            b        (x) float64 40B [kg] -1.0 -0.75 -0.5 -0.25 0.0

        Convert the data

        >>> ds.pint.to({"a": "mm", "b": ureg.g})
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [mm] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 40B [g] -1e+03 -750.0 -500.0 -250.0 0.0
        >>> ds.pint.to(a=ureg.mm, b="g")
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [mm] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 40B [g] -1e+03 -750.0 -500.0 -250.0 0.0

        Convert coordinates

        >>> ds.pint.to({"u": ureg.ms})
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [m] 0.0 0.25 0.5 0.75 1.0
            b        (x) float64 40B [kg] -1.0 -0.75 -0.5 -0.25 0.0
        >>> ds.pint.to(u="ms")
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [m] 0.0 0.25 0.5 0.75 1.0
            b        (x) float64 40B [kg] -1.0 -0.75 -0.5 -0.25 0.0

        Convert both simultaneously

        >>> ds.pint.to(a=ureg.mm, b=ureg.g, u="ms")
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [mm] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 40B [g] -1e+03 -750.0 -500.0 -250.0 0.0
        >>> ds.pint.to({"a": "mm", "b": "g", "u": ureg.ms})
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) float64 40B [ms] 0.0 1e+03 2e+03 3e+03 4e+03
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [mm] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 40B [g] -1e+03 -750.0 -500.0 -250.0 0.0

        Convert homogeneous data

        >>> ds = xr.Dataset(
        ...     data_vars={
        ...         "a": ("x", np.linspace(0, 1, 5) * ureg.kg),
        ...         "b": ("x", np.linspace(-1, 0, 5) * ureg.mg),
        ...     },
        ...     coords={"u": ("x", np.arange(5) * ureg.s)},
        ... )
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [kg] 0.0 0.25 0.5 0.75 1.0
            b        (x) float64 40B [mg] -1.0 -0.75 -0.5 -0.25 0.0
        >>> ds.pint.to("g")
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Coordinates:
            u        (x) int64 40B [s] 0 1 2 3 4
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 40B [g] 0.0 250.0 500.0 750.0 1e+03
            b        (x) float64 40B [g] -0.001 -0.00075 -0.0005 -0.00025 0.0
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

    def chunk(self, chunks, name_prefix="xarray-", token=None, lock=False):
        """unit-aware version of chunk

        Like :py:meth:`xarray.Dataset.chunk`, but chunking a quantity will change the
        wrapped type to ``dask``.

        .. note::
            It is recommended to only use this when chunking in-memory arrays. To
            rechunk please use :py:meth:`xarray.Dataset.chunk`.

        See Also
        --------
        xarray.Dataset.chunk
        xarray.DataArray.pint.chunk
        """
        units = conversion.extract_units(self.ds)
        stripped = conversion.strip_units(self.ds)

        chunked = stripped.chunk(
            chunks, name_prefix=name_prefix, token=token, lock=lock
        )
        return conversion.attach_units(chunked, units)

    def reindex(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        copy=True,
        fill_value=NA,
        **indexers_kwargs,
    ):
        """unit-aware version of reindex

        Like :py:meth:`xarray.Dataset.reindex`, except the object's indexes are converted
        to the units of the indexers first.

        .. note::
            ``tolerance`` and ``fill_value`` are not supported, yet. They will be passed through to
            ``Dataset.reindex`` unmodified.

        See Also
        --------
        xarray.DataArray.pint.reindex
        xarray.Dataset.pint.reindex_like
        xarray.Dataset.reindex
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "reindex")

        dims = self.ds.dims
        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
            if name in dims
        }

        # TODO: handle tolerance
        # TODO: handle fill_value

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
        """unit-aware version of reindex_like

        Like :py:meth:`xarray.Dataset.reindex_like`, except the object's indexes are converted
        to the units of the indexers first.

        .. note::
            ``tolerance`` and ``fill_value`` are not supported, yet. They will be passed through to
            ``Dataset.reindex_like`` unmodified.

        See Also
        --------
        xarray.DataArray.pint.reindex_like
        xarray.Dataset.pint.reindex
        xarray.Dataset.reindex_like
        """
        indexer_units = conversion.extract_unit_attributes(other)

        # TODO: handle tolerance
        # TODO: handle fill_value

        converted = conversion.convert_units(self.ds, indexer_units)
        return converted.reindex_like(
            other,
            method=method,
            tolerance=tolerance,
            copy=copy,
            fill_value=fill_value,
        )

    def interp(
        self,
        coords=None,
        method="linear",
        assume_sorted=False,
        kwargs=None,
        **coords_kwargs,
    ):
        """unit-aware version of interp

        Like :py:meth:`xarray.Dataset.interp`, except the object's indexes are converted
        to the units of the indexers first.

        .. note::
            ``kwargs`` is passed unmodified to ``Dataset.interp``

        See Also
        --------
        xarray.DataArray.pint.interp
        xarray.Dataset.pint.interp_like
        xarray.Dataset.interp
        """
        indexers = either_dict_or_kwargs(coords, coords_kwargs, "interp")

        dims = self.ds.dims
        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
            if name in dims
        }

        # convert the indexes to the indexer's units
        converted = conversion.convert_units(self.ds, indexer_units)
        units = conversion.extract_units(converted)
        stripped = conversion.strip_units(converted)

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in indexers.items()
        }
        interpolated = stripped.interp(
            stripped_indexers,
            method=method,
            assume_sorted=False,
            kwargs=None,
        )
        return conversion.attach_units(interpolated, units)

    def interp_like(self, other, method="linear", assume_sorted=False, kwargs=None):
        """unit-aware version of interp_like

        Like :py:meth:`xarray.Dataset.interp_like`, except the object's indexes are
        converted to the units of the indexers first.

        .. note::
            ``kwargs`` is passed unmodified to ``Dataset.interp``

        See Also
        --------
        xarray.DataArray.pint.interp_like
        xarray.Dataset.pint.interp
        xarray.Dataset.interp_like
        """
        indexer_units = conversion.extract_unit_attributes(other)

        converted = conversion.convert_units(self.ds, indexer_units)
        units = conversion.extract_units(converted)
        stripped = conversion.strip_units(converted)
        interpolated = stripped.interp_like(
            other,
            method=method,
            assume_sorted=assume_sorted,
            kwargs=kwargs,
        )
        return conversion.attach_units(interpolated, units)

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ):
        """unit-aware version of sel

        Like :py:meth:`xarray.Dataset.sel`, except the object's indexes are converted to
        the units of the indexers first.

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

        dims = self.ds.dims
        indexer_units = {
            name: conversion.extract_indexer_units(indexer)
            for name, indexer in indexers.items()
            if name in dims
        }

        # TODO: handle tolerance

        # convert the indexes to the indexer's units
        try:
            converted = conversion.convert_units(self.ds, indexer_units)
        except ValueError as e:
            raise KeyError(*e.args) from e

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
        """Unit-aware attribute for indexing

        Only supports ``__getitem__``.

        .. note::
           Position based indexing (e.g. ``ds.loc[1, 2:]``) is not supported, yet

        See Also
        --------
        xarray.Dataset.loc
        """
        return DatasetLocIndexer(self.ds)

    def drop_sel(self, labels=None, *, errors="raise", **labels_kwargs):
        """unit-aware version of drop_sel

        Just like :py:meth:`xarray.Dataset.drop_sel`, except the indexers are converted
        to the units of the object's indexes first.

        See Also
        --------
        xarray.DataArray.pint.drop_sel
        xarray.Dataset.drop_sel
        xarray.DataArray.drop_sel
        """
        indexers = either_dict_or_kwargs(labels, labels_kwargs, "drop_sel")

        dims = self.ds.dims
        unit_attrs = conversion.extract_unit_attributes(self.ds)
        index_units = {
            name: units for name, units in unit_attrs.items() if name in dims
        }

        # convert the indexers to the indexes units
        try:
            converted_indexers = conversion.convert_indexer_units(indexers, index_units)
        except ValueError as e:
            raise KeyError(*e.args) from e

        # index
        stripped_indexers = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in converted_indexers.items()
        }
        indexed = self.ds.drop_sel(
            stripped_indexers,
            errors=errors,
        )

        return indexed

    def ffill(self, dim, limit=None):
        """unit-aware version of ffill

        Like :py:meth:`xarray.Dataset.ffill` but without stripping the data units.

        See Also
        --------
        xarray.Dataset.ffill
        xarray.Dataset.pint.bfill
        """
        units = conversion.extract_units(self.ds)
        stripped = conversion.strip_units(self.ds)

        filled = stripped.ffill(dim=dim, limit=limit)

        return conversion.attach_units(filled, units)

    def bfill(self, dim, limit=None):
        """unit-aware version of bfill

        Like :py:meth:`xarray.Dataset.bfill` but without stripping the data units.

        See Also
        --------
        xarray.Dataset.bfill
        xarray.Dataset.pint.ffill
        """
        units = conversion.extract_units(self.ds)
        stripped = conversion.strip_units(self.ds)

        filled = stripped.bfill(dim=dim, limit=limit)

        return conversion.attach_units(filled, units)

    def interpolate_na(
        self,
        dim=None,
        method="linear",
        limit=None,
        use_coordinate=True,
        max_gap=None,
        keep_attrs=None,
        **kwargs,
    ):
        """unit-aware version of interpolate_na

        Like :py:meth:`xarray.Dataset.interpolate_na` but without stripping the units on
        data or coordinates.

        .. note::
            ``max_gap`` is not supported, yet, and will be passed through to
            ``Dataset.interpolate_na`` unmodified.

        See Also
        --------
        xarray.DataArray.pint.interpolate_na
        xarray.Dataset.interpolate_na
        """
        units = conversion.extract_units(self.ds)
        stripped = conversion.strip_units(self.ds)

        interpolated = stripped.interpolate_na(
            dim=dim,
            method=method,
            limit=limit,
            use_coordinate=use_coordinate,
            max_gap=max_gap,
            keep_attrs=keep_attrs,
            **kwargs,
        )

        return conversion.attach_units(interpolated, units)
