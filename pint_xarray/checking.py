import functools

from pint import Quantity, Unit
from xarray import DataArray

from .accessors import PintDataArrayAccessor  # noqa


def expects(*args_units, return_units=None, **kwargs_units):
    """
    Decorator which ensures the inputs and outputs of the decorated function are expressed in the expected units.

    Arguments to the decorated function are checked for the specified units, converting to those units if necessary, and
    then stripped of their units before being passed into the undecorated function. Therefore the undecorated function
    should expect unquantified DataArrays or numpy-like arrays, but with the values expressed in specific units.

    Note that the coordinates of input DataArrays are not checked, only the data.
    So if your decorated function uses coordinates and you wish to check their units,
    you should pass the coordinates of interest as separate arguments.

    Parameters
    ----------
    func : callable
        Function to decorate. which accepts zero or more xarray.DataArrays or numpy-like arrays as inputs,
        and may optionally return one or more xarray.DataArrays or numpy-like arrays.
    *args_units : None or unit-like or mapping of None or unit-like, optional
        Units to expect for each positional argument given to func.

        The decorator will first check that arguments passed to the decorated function possess these specific units
        (or will attempt to convert the argument to these units), then will strip the units before passing the magnitude
        to the wrapped function.

        A value of None indicates not to check that argument for units (suitable for flags and other non-data
        arguments).
    return_units : Union[Union[str, pint.Unit, None], List[Union[str, pint.Unit, None]], Optional
        The expected units of the returned value(s), either as a single unit or as a list of units. The decorator
        will attach these units to the variables returned from the function.

        A value of None indicates not to attach any units to that return value (suitable for flags and other
        non-data results).
    kwargs_units : Dict[str, Union[str, pint.Unit, None]], Optional
        Unit to expect for each keyword argument given to func.

        The decorator will first check that arguments passed to the decorated function possess these specific units
        (or will attempt to convert the argument to these units), then will strip the units before passing the magnitude
        to the wrapped function.

        A value of None indicates not to check that argument for units (suitable for flags and other non-data
        arguments).

    Returns
    -------
    return_values : Any
        Return values of the wrapped function, either a single value or a tuple of values. These will be given units
        according to return_units.

    Raises
    ------
    TypeError
        If an argument or return value has a specified unit, but is not an xarray.DataArray or pint.Quantity.
        Also thrown if any of the units are not a valid type, or if the number of arguments or return values does not
        match the number of units specified.

    Examples
    --------

    Decorating a function which takes one quantified input, but returns a non-data value (in this case a boolean).

    >>> @expects("deg C")
    ... def above_freezing(temp):
    ...     return temp > 0


    TODO: example where we check units of an optional weighted kwarg
    """

    # TODO generalise to allow for dictionaries of units for DataArray coordinates / Datasets

    # Check types of args_units, kwargs_units, and return_units
    all_units = list(args_units) + list(kwargs_units.values())
    if isinstance(return_units, list):
        all_units = all_units + return_units
    elif return_units:
        all_units = all_units + [return_units]
    for a in all_units:
        if not isinstance(a, (Unit, str)) and a is not None:
            raise TypeError(
                f"{a} is not a valid type for a unit, it is of type {type(a)}"
            )

    def _expects_decorator(func):
        @functools.wraps(func)
        def _unit_checking_wrapper(*args, **kwargs):

            # check same number of arguments were passed as expected
            if len(args) != len(args_units):
                raise TypeError(
                    f"{len(args)} arguments were passed, but {len(args_units)} arguments were expected"
                )

            converted_args = []
            for arg, arg_unit in zip(args, args_units):
                converted_arg = _check_or_convert_to_then_strip(arg, arg_unit)
                converted_args.append(converted_arg)

            converted_kwargs = {}
            for key, val in kwargs.items():
                kwarg_unit = kwargs_units.get(key, None)
                converted_kwargs[key] = _check_or_convert_to_then_strip(val, kwarg_unit)

            results = func(*converted_args, **converted_kwargs)

            if results is None:
                if return_units:
                    raise TypeError(
                        "Expected function to return something, but function returned None"
                    )
            else:
                if return_units is None:
                    # ignore types and units of return values
                    return results
                else:
                    # handle case of function returning only one result by promoting to 1-element tuple
                    if not isinstance(results, tuple):
                        results = (results,)
                    if isinstance(return_units, (tuple, list)):
                        # avoid mutating return_units because that would change the variables' scope
                        return_units_list = return_units
                    else:
                        return_units_list = [return_units]

                    # check same number of things were returned as expected
                    if len(results) != len(return_units_list):
                        raise TypeError(
                            f"{len(results)} return values were received, but {len(return_units_list)} "
                            "return values were expected"
                        )

                    converted_results = []
                    for return_unit, return_value in zip(return_units_list, results):
                        converted_result = _attach_units(return_value, return_unit)
                        converted_results.append(converted_result)

                    if len(converted_results) == 1:
                        return tuple(converted_results)[0]
                    else:
                        return tuple(converted_results)

        return _unit_checking_wrapper

    return _expects_decorator


def _check_or_convert_to_then_strip(obj, units):
    """
    Checks the object is of a valid type (Quantity or DataArray), then attempts to convert it to the specified units,
    then strips the units from it.
    """

    if units is None:
        # allow for passing through non-numerical arguments
        return obj
    else:
        if isinstance(obj, Quantity):
            converted = obj.to(units)
            return converted.magnitude
        elif isinstance(obj, DataArray):
            converted = obj.pint.to(units)
            return converted.pint.dequantify()
        else:
            raise TypeError(
                "Can only expect units for arguments of type xarray.DataArray or pint.Quantity,"
                f"not {type(obj)}"
            )


def _attach_units(obj, units):
    """Attaches units, but can also create pint.Quantity objects from numpy scalars"""
    if isinstance(obj, DataArray):
        return obj.pint.quantify(units)
    else:
        return Quantity(obj, units=units)
