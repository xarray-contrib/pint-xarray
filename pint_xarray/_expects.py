import functools
import inspect

import pint
import xarray as xr

from pint_xarray.itertools import zip_mappings


def expects(*args_units, return_value=None, **kwargs_units):
    def outer(func):
        signature = inspect.signature(func)

        params_units = signature.bind_partial(*args_units, **kwargs_units)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal return_value

            params = signature.bind(*args, **kwargs)
            # don't apply defaults, as those have to already be in the correct units

            errors = []
            for name, (value, units) in zip_mappings(
                params.arguments, params_units.arguments
            ):
                try:
                    if units is None:
                        if isinstance(value, pint.Quantity) or (
                            isinstance(value, (xr.DataArray, xr.Dataset))
                            and value.pint.units
                        ):
                            raise ValueError(
                                "Passed in a quantity where none was expected"
                            )
                        continue
                    if isinstance(value, pint.Quantity):
                        params.arguments[name] = value.m_as(units)
                    elif isinstance(value, (xr.DataArray, xr.Dataset)):
                        params.arguments[name] = value.pint.to(units).pint.dequantify()
                    else:
                        raise ValueError(
                            f"Attempting to convert non-quantity {value} to {units}."
                        )
                except Exception as e:
                    e.add_note(
                        f"expects: raised while trying to convert parameter {name}"
                    )
                    errors.append(e)

            if errors:
                raise ExceptionGroup("Errors while converting parameters", errors)

            result = func(*params.args, **params.kwargs)

            if (isinstance(result, tuple) ^ isinstance(return_value, tuple)) or (
                isinstance(result, tuple) and len(result) != len(return_value)
            ):
                raise ValueError("mismatched number of return values")

            if result is None:
                return

            n_results = len(result) if isinstance(result, tuple) else 1

            if not isinstance(result, tuple):
                result = (result,)
            if not isinstance(return_value, tuple):
                return_value = (return_value,)

            final_result = []
            errors = []
            for index, (value, units) in enumerate(zip(result, return_value)):
                if units is not None:
                    try:
                        if isinstance(value, (xr.Dataset, xr.DataArray)):
                            value = value.pint.quantify(units)
                        else:
                            value = units.m_from(value)
                    except Exception as e:
                        e.add_note(
                            f"expects: raised while trying to convert return value {index}"
                        )
                        errors.append(e)

                final_result.append(value)

            if errors:
                raise ExceptionGroup("Errors while converting return values", errors)

            if n_results == 1:
                return final_result[0]
            return tuple(final_result)

        return wrapper

    return outer
