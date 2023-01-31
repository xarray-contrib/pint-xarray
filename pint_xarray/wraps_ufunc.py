import functools
import warnings
from inspect import Parameter, signature
from typing import Union

import numpy as np
import xarray as xr
from pint import Quantity, UnitStrippedWarning

import pint_xarray

from .accessors import default_registry

_handled_types = (float, int, np.ndarray)


def _check_wrapper_args(unit_iterable):
    """
    Convert the "unit_iterable" to a tuple if it isn't already an iterable.
    Make sure that the elements in the unit_iterable are either Units or strings.
    """
    for arg in unit_iterable:
        if arg is not None and not isinstance(arg, (default_registry.Unit, str)):
            raise TypeError(
                f"wraps arguments must by of type str or Unit, not {type(arg)} ({arg})"
            )


def _convert_units(key, value, unit, debug=False):
    """Convert "value" such that it has units of "unit".

    If unit=None: don't convert the value.
    If unit="": allow non-dimensional inputs like floats, ints, arrays
    """
    if debug:
        print(f"Converting {key} with value {value} to unit {unit}")

    if unit is None:
        # Do nothing
        return value

    dimensionless = (unit == "") or (unit == default_registry.dimensionless)

    if not dimensionless and not isinstance(value, (xr.DataArray, Quantity)):
        raise TypeError(
            f"Input for {key} is of type {type(value)} which does not contain units, but units of {unit} are required for {key}."
        )

    if isinstance(value, xr.DataArray):
        if value.pint.units is None:
            value = value.pint.quantify("")
        return value.pint.to(unit)
    elif isinstance(value, Quantity):
        return value.to(unit).magnitude
    elif dimensionless and isinstance(value, _handled_types):
        return value

    # Catch unhandled case
    raise NotImplementedError(
        f"Could not process input for {key} with type={type(value)}."
    )


def _set_units(value, unit, key):
    """Set the units of "value" to "unit" (or leave as is if unit=None)."""
    if not default_registry.force_ndarray_like and not default_registry.force_ndarray:
        raise ValueError(
            "Set 'force_ndarray_like' or 'force_ndarray' when defining default_registry=UnitRegistry(...)."
        )

    if unit is None:
        return value

    dimensionless = (unit == "") or (unit == default_registry.dimensionless)

    if dimensionless and isinstance(value, _handled_types):
        return value

    if isinstance(value, xr.DataArray):
        if value.pint.units is None:
            return value.pint.quantify(unit, unit_registry=default_registry)
        else:
            return value.pint.to(unit)

    elif isinstance(value, Quantity):
        return value.to(unit)

    elif isinstance(value, _handled_types):
        return Quantity(value, unit)

    # Catch unhandled case
    raise NotImplementedError(
        f"Could not process output with type={type(value)} for output {key}."
    )


def _get_default_values(func):
    """Get default values for arguments."""
    func_signature = signature(func)

    return {
        k: v.default
        for k, v in func_signature.parameters.items()
        if v.default is not Parameter.empty
    }


def wraps_ufunc(
    _func=None,
    *,
    return_units: dict[str, str],
    input_units: dict[str, str],
    ufunc_kwargs_from_wrapper=dict(),
    auto_none: dict = False,
):
    """Wraps a function to accept pint-xarray inputs and then uses xarrays apply_ufunc to iterate over dimensions.

    You can pass arguments to apply_ufunc (see https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html)
    by supplying an additional ufunc_kwargs key-word argument to the function, or by setting
    ufunc_kwargs_from_wrapper when calling @wraps_ufunc.

    For each element in return_units and input_units, you can set
    * None: do nothing with this argument
    * "" or default_registry.dimensionless: allow untyped floats, ints and np.arrays. Ensure Quantity and xr.DataArrays units
                                reduce to dimensionless.
    * "<unit>" or default_registry.<unit>: require Quantity or xr.DataArray, convert their units to <unit>.

    Closely based on pint.registry_helpers.wraps.
    """
    _check_wrapper_args(input_units)
    _check_wrapper_args(return_units)

    if "vectorize" not in ufunc_kwargs_from_wrapper.keys():
        # Unless explicitly set to False, assume that the user wants vectorize=True.
        ufunc_kwargs_from_wrapper["vectorize"] = True
    if "output_core_dims" not in ufunc_kwargs_from_wrapper.keys():
        # Unless explicitly set, assume that the user doesn't want to add new dimensions to the output.
        ufunc_kwargs_from_wrapper["output_core_dims"] = [
            () for i in range(len(return_units.keys()))
        ]

    def decorator(func):

        # Work out what the function is expecting as inputs
        func_parameters = signature(func).parameters

        if auto_none:
            for parameter in func_parameters:
                if not parameter in input_units.keys():
                    # Only need to include units which you want to perform unit-checking on.
                    input_units[parameter] = None

        # Make sure that there are as many units in "input_units" as there are arguments
        count_params = len(func_parameters)
        if len(input_units) != count_params:
            raise TypeError(
                f"{func.__name__} takes {count_params} parameters, but {len(input_units)} units were passed"
            )

        for parameter in input_units.keys():
            if not parameter in func_parameters:
                raise TypeError(
                    f"{func.__name__} does not have a parameter {parameter}"
                )

        @functools.wraps(func)
        def wrapper(*positional_args, ufunc_kwargs=dict(), debug=False, **keyword_args):

            # Should we copy kwargs? Safer since no global modification, but
            # also requires more memory.

            # Convert all positional arguments into keyword arguments
            for arg, param in zip(positional_args, func_parameters):
                if param in keyword_args:
                    raise KeyError(f"Repeated argument for {param}")
                keyword_args[param] = arg

            # Add default values if they aren't already set
            keyword_args = {**_get_default_values(func), **keyword_args}

            if debug:
                print(f"{func} called with {keyword_args}")

            # Convert the input into the desired units
            for key in func_parameters:
                value = keyword_args[key]
                unit = input_units[key]

                keyword_args[key] = _convert_units(key, value, unit, debug=debug)

            ufunc_kwargs = {**ufunc_kwargs_from_wrapper, **ufunc_kwargs}

            with warnings.catch_warnings():
                # Suppress the UnitStrippedWarning — we want to drop units and down-cast to unitless arrays since
                # this is what the pint.registry_helpers.wraps decorator does.
                warnings.simplefilter("ignore", category=UnitStrippedWarning)

                if debug:
                    print(
                        f"apply_ufunc called with {keyword_args}, with kwargs {ufunc_kwargs}"
                    )

                function_return = xr.apply_ufunc(
                    func,
                    *[keyword_args[param] for param in func_parameters],
                    **ufunc_kwargs,
                )

                if debug:
                    print(f"apply_ufunc returned {function_return}")

            # Convert the output into the desired units
            if isinstance(function_return, tuple):
                # Multiple return
                if len(return_units) != len(function_return):
                    raise TypeError(
                        f"{func.__name__} returned {len(function_return)} values(s), but {len(return_units)} units were passed"
                    )

                new_function_return = list(function_return)

                for i, (returned, unit, key) in enumerate(
                    zip(function_return, return_units.values(), return_units.keys())
                ):
                    new_function_return[i] = _set_units(returned, unit, key)

                return tuple(new_function_return)

            else:
                if (
                    len(return_units) == 0
                    and isinstance(function_return, xr.DataArray)
                    and np.all(function_return == None)
                ):
                    return None
                elif len(return_units) != 1:
                    raise TypeError(
                        f"{func.__name__} returned {len(function_return)} value(s), but {len(return_units)} units were passed"
                    )

                return _set_units(
                    function_return,
                    unit=list(return_units.values())[0],
                    key=list(return_units.keys())[0],
                )

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
