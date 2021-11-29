import functools

from pint import Quantity
from xarray import DataArray

from .accessors import PintDataArrayAccessor  # noqa


def expects(*args_units, return_units=None, **kwargs_units):
    """
    Decorator which checks the inputs and outputs of the decorated function have certain units.

    Arguments

    Note that the coordinates of input DataArrays are not checked, only the data.
    So if your decorated function uses coordinates and you wish to check their units,
    you should pass the coordinates of interest as separate arguments.

    Parameters
    ----------
    func: function
        Function to decorate. which accepts zero or more xarray.DataArrays or numpy-like arrays as inputs,
        and may optionally return one or more xarray.DataArrays or numpy-like arrays.
    args_units : Union[str, pint.Unit, None]
        Units to expect for each positional argument given to func.

        The decorator will check that arguments passed to the decorated function possess these specific units, or will
        attempt to convert the argument to these units.

        A value of None indicates not to check that argument for units (suitable for flags
        and other non-data arguments).
    return_units : Union[Union[str, pint.Unit, None, False], Sequence[Union[str, pint.Unit, None]], Optional
        The expected units of the returned value(s), either as a single unit or as an iterable of units.

        The decorator will check that results returned from the decorated function possess these specific units, or will
        attempt to convert the results to these units.

        A value of None indicates not to check that return value for units (suitable for flags and other
        non-data arguments). Passing False means that no return value is expected from the function at all,
        and an error will be raised if a return value is found.
    kwargs_units : Dict[str, Union[str, pint.Unit, None]], Optional
        Unit to expect for each keyword argument given to func.

        The decorator will check that arguments passed to the decorated function possess these specific units, or will
        attempt to convert the argument to these units.

        A value of None indicates not to check that argument for units (suitable for flags
        and other non-data arguments).

    Returns
    -------
    return_values : Any
        Return values of the wrapped function, either a single value or a tuple of values.

    Raises
    ------
    TypeError
        If an argument or return value has a specified unit, but is not an xarray.DataArray or pint.Quantity.

    Examples
    --------

    Decorating a function which takes one quantified input, but returns a non-data value (in this case a boolean).

    >>> @expects("deg C")
    ... def above_freezing(temp):
    ...     return temp > 0


    TODO: example where we check units of an optional weighted kwarg
    """

    # TODO: Check args_units, kwargs_units, and return_units types
    # TODO: Check number of arguments line up

    def _expects_decorator(func):
        @functools.wraps(func)
        def _unit_checking_wrapper(*args, **kwargs):

            converted_args = []
            for arg, arg_unit in zip(args, args_units):
                converted_arg = _check_then_convert_to(arg, arg_unit)
                converted_args.append(converted_arg)

            converted_kwargs = {}
            for key, val in kwargs.items():
                kwarg_unit = kwargs_units[key]
                converted_kwargs[key] = _check_then_convert_to(val, kwarg_unit)

            results = func(*converted_args, **converted_kwargs)

            if results is not None:
                if return_units is False:
                    raise ValueError("Did not expect function to return anything")
                elif return_units is not None:
                    # TODO check something was actually returned
                    # TODO check same number of things were returned as expected
                    # TODO handle single return value vs tuple of return values

                    converted_results = []
                    for return_unit, return_value in zip(return_units, results):
                        converted_result = _check_then_convert_to(
                            return_value, return_unit
                        )
                        converted_results.append(converted_result)

                    return tuple(converted_results)
                else:
                    return results
            else:
                if return_units:
                    raise ValueError("Expected function to return something")

        return _unit_checking_wrapper

    return _expects_decorator


def _check_then_convert_to(obj, units):
    if isinstance(obj, Quantity):
        converted = obj.to(units)
        return converted.magnitude
    elif isinstance(obj, DataArray):
        converted = obj.pint.to(units)
        return converted.pint.magnitude
    else:
        raise TypeError(
            "Can only expect units for arguments of type xarray.DataArray or pint.Quantity,"
            f"not {type(obj)}"
        )
