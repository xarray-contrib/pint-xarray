import functools
import inspect
import itertools
from inspect import Parameter

import pint
import pint.testing
import xarray as xr

from pint_xarray.accessors import get_registry
from pint_xarray.conversion import extract_units
from pint_xarray.itertools import zip_mappings

variable_parameters = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)


def expects(*args_units, return_value=None, **kwargs_units):
    def outer(func):
        signature = inspect.signature(func)

        params_units = signature.bind(*args_units, **kwargs_units)

        missing_params = [
            name
            for name, p in signature.parameters.items()
            if p.kind not in variable_parameters and name not in params_units.arguments
        ]
        if missing_params:
            raise ValueError(
                "Missing units for the following parameters: "
                + ", ".join(map(repr, missing_params))
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal return_value

            params = signature.bind(*args, **kwargs)
            # don't apply defaults, as those can't be quantities and thus must
            # already be in the correct units

            spec_units = dict(
                enumerate(
                    itertools.chain.from_iterable(
                        spec.values() if isinstance(spec, dict) else (spec,)
                        for spec in params_units.arguments.values()
                        if spec is not None
                    )
                )
            )
            params_units_ = dict(
                enumerate(
                    itertools.chain.from_iterable(
                        (
                            extract_units(param)
                            if isinstance(param, (xr.DataArray, xr.Dataset))
                            else (param.units,)
                        )
                        for name, param in params.arguments.items()
                        if isinstance(param, (xr.DataArray, xr.Dataset, pint.Quantity))
                    )
                )
            )

            ureg = get_registry(
                None,
                dict(spec_units) if spec_units else {},
                dict(params_units_) if params_units else {},
            )

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
                            value = ureg.Quantity(value, units)
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
