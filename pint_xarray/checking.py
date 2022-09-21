import functools
import inspect
from inspect import Parameter

from pint import Quantity
from xarray import DataArray, Dataset, Variable

from . import conversion
from .accessors import PintDataArrayAccessor  # noqa


def detect_missing_params(params, units):
    """detect parameters for which no units were specified"""
    variable_params = {
        Parameter.VAR_POSITIONAL,
        Parameter.VAR_KEYWORD,
    }

    return {
        name
        for name, param in params.items()
        if name not in units.arguments and param.kind not in variable_params
    }


def convert_and_strip(obj, units):
    if isinstance(obj, (DataArray, Dataset, Variable)):
        if not isinstance(units, dict):
            units = {None: units}
        return conversion.strip_units(conversion.convert_units(obj, units))
    elif isinstance(obj, Quantity):
        return obj.m_as(units)
    elif units is None:
        return obj
    else:
        raise ValueError(f"unknown type: {type(obj)}")


def convert_and_strip_args(args, units):
    return [convert_and_strip(obj, units_) for obj, units_ in zip(args, units)]


def convert_and_strip_kwargs(kwargs, units):
    return {name: convert_and_strip(kwargs[name], units[name]) for name in kwargs}


def always_iterable(obj, base_type=(str, bytes)):
    """
    If *obj* is iterable, return an iterator over its items,
    If *obj* is not iterable, return a one-item iterable containing *obj*,
    If *obj* is ``None``, return an empty iterable.
    If *base_type* is set, objects for which ``isinstance(obj, base_type)``
    returns ``True`` won't be considered iterable.

    Copied from more_itertools.
    """

    if obj is None:
        return iter(())

    if (base_type is not None) and isinstance(obj, base_type):
        return iter((obj,))

    try:
        return iter(obj)
    except TypeError:
        return iter((obj,))


def attach_return_units(results, units):
    if units is None:
        # ignore types and units of return values
        return results
    elif results is None:
        raise TypeError(
            "Expected function to return something, but function returned None"
        )
    else:
        # handle case of function returning only one result by promoting to 1-element tuple
        return_units_iterable = tuple(always_iterable(units, base_type=(str, dict)))
        results_iterable = tuple(always_iterable(results, base_type=(str, Dataset)))

        # check same number of things were returned as expected
        if len(results_iterable) != len(return_units_iterable):
            raise TypeError(
                f"{len(results_iterable)} return values were received, but {len(return_units_iterable)} "
                "return values were expected"
            )

        converted_results = _attach_multiple_units(
            results_iterable, return_units_iterable
        )

        if isinstance(results, tuple) or len(converted_results) != 1:
            return converted_results
        else:
            return converted_results[0]


def _check_or_convert_to_then_strip(obj, units):
    """
    Checks the object is of a valid type (Quantity or DataArray), then attempts to convert it to the specified units,
    then strips the units from it.
    """

    if units is None:
        # allow for passing through non-numerical arguments
        return obj
    elif isinstance(obj, Quantity):
        converted = obj.to(units)
        return converted.magnitude
    elif isinstance(obj, (DataArray, Dataset)):
        converted = obj.pint.to(units)
        return converted.pint.dequantify()
    else:
        raise TypeError(
            "Can only expect units for arguments of type xarray.DataArray,"
            f" xarray.Dataset, or pint.Quantity, not {type(obj)}"
        )


def _attach_units(obj, units):
    """Attaches units, but can also create pint.Quantity objects from numpy scalars"""
    if isinstance(obj, (DataArray, Dataset)):
        return obj.pint.quantify(units)
    else:
        return Quantity(obj, units=units)


def _attach_multiple_units(objects, units):
    """Attaches list of units to list of objects elementwise"""
    converted_objects = [_attach_units(obj, unit) for obj, unit in zip(objects, units)]
    return converted_objects


def expects(*args_units, return_units=None, **kwargs_units):
    """
    Decorator which ensures the inputs and outputs of the decorated
    function are expressed in the expected units.

    Arguments to the decorated function are checked for the specified
    units, converting to those units if necessary, and then stripped
    of their units before being passed into the undecorated
    function. Therefore the undecorated function should expect
    unquantified DataArrays, Datasets, or numpy-like arrays, but with
    the values expressed in specific units.

    Parameters
    ----------
    func : callable
        Function to decorate, which accepts zero or more
        xarray.DataArrays or numpy-like arrays as inputs, and may
        optionally return one or more xarray.DataArrays or numpy-like
        arrays.
    *args_units : unit-like or mapping of hashable to unit-like, optional
        Units to expect for each positional argument given to func.

        The decorator will first check that arguments passed to the
        decorated function possess these specific units (or will
        attempt to convert the argument to these units), then will
        strip the units before passing the magnitude to the wrapped
        function.

        A value of None indicates not to check that argument for units
        (suitable for flags and other non-data arguments).
    return_units : unit-like or list of unit-like or mapping of hashable to unit-like \
                   or list of mapping of hashable to unit-like, optional
        The expected units of the returned value(s), either as a
        single unit or as a list of units. The decorator will attach
        these units to the variables returned from the function.

        A value of None indicates not to attach any units to that
        return value (suitable for flags and other non-data results).
    kwargs_units : mapping of hashable to unit-like, optional
        Unit to expect for each keyword argument given to func.

        The decorator will first check that arguments passed to the
        decorated function possess these specific units (or will
        attempt to convert the argument to these units), then will
        strip the units before passing the magnitude to the wrapped
        function.

        A value of None indicates not to check that argument for units
        (suitable for flags and other non-data arguments).

    Returns
    -------
    return_values : Any
        Return values of the wrapped function, either a single value
        or a tuple of values. These will be given units according to
        return_units.

    Raises
    ------
    TypeError
        If an argument or return value has a specified unit, but is
        not an xarray.DataArray or pint.Quantity.  Also thrown if any
        of the units are not a valid type, or if the number of
        arguments or return values does not match the number of units
        specified.

    Examples
    --------

    Decorating a function which takes one quantified input, but
    returns a non-data value (in this case a boolean).

    >>> @expects("deg C")
    ... def above_freezing(temp):
    ...     return temp > 0

    Decorating a function which allows any dimensions for the array, but also
    accepts an optional `weights` keyword argument, which must be dimensionless.

    >>> @expects(None, weights="dimensionless")
    ... def mean(da, weights=None):
    ...     if weights:
    ...         return da.weighted(weights=weights).mean()
    ...     else:
    ...         return da.mean()

    """

    def _expects_decorator(func):

        # check same number of arguments were passed as expected
        sig = inspect.signature(func)

        params = sig.parameters

        bound_units = sig.bind_partial(*args_units, **kwargs_units)

        missing_params = detect_missing_params(params, bound_units)
        if missing_params:
            raise TypeError(
                "Some parameters of the decorated function are missing units:"
                f" {', '.join(sorted(missing_params))}"
            )

        @functools.wraps(func)
        def _unit_checking_wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)

            converted_args = convert_and_strip_args(bound.args, bound_units.args)
            converted_kwargs = convert_and_strip_kwargs(
                bound.kwargs, bound_units.kwargs
            )

            results = func(*converted_args, **converted_kwargs)

            return attach_return_units(results, return_units)

        return _unit_checking_wrapper

    return _expects_decorator
