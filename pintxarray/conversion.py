import pint
from xarray import DataArray, Dataset, Variable


def array_attach_units(data, unit, registry=None):
    """ attach a unit to the data

    Parameters
    ----------
    data : array-like
        The data to attach units to.
    unit : str or pint.Unit
        The desired unit.
    registry : pint.UnitRegistry, optional
        The registry to use if ``unit`` is a string. Must not be
        specified otherwise.

    Returns
    -------
    quantity : pint.Quantity
    """

    if unit is None:
        return data

    if not isinstance(unit, (pint.Unit, str)):
        raise ValueError(f"cannot use {unit!r} as a unit")

    if isinstance(data, pint.Quantity):
        raise ValueError(
            f"Cannot attach unit {unit!r} to quantity: data "
            f"already has units {data.units}"
        )

    if registry is None:
        if isinstance(unit, str):
            raise ValueError(
                "cannot use a string as unit without specifying a registry"
            )

        registry = unit._REGISTRY

    return registry.Quantity(data, unit)


def array_convert_units(data, unit):
    """ convert the units of an array

    This is roughly the same as ``data.to(unit)``.

    Parameters
    ----------
    data : quantity or array-like
        The data to convert. If it is not a quantity, it is assumed to be
        dimensionless.
    unit : str or pint.Unit
        The unit to convert to. If a string ``data`` has to be a quantity.

    Returns
    -------
    result : pint.Quantity
        The converted data
    """
    if unit is None:
        return data

    if not isinstance(unit, (str, pint.Unit)):
        raise ValueError(f"cannot use {unit!r} as a unit")
    elif isinstance(unit, str) and not isinstance(data, pint.Quantity):
        raise ValueError(f"cannot convert a non-quantity using {unit!r} as unit")

    registry = data._REGISTRY if isinstance(unit, str) else unit._REGISTRY

    if not isinstance(data, pint.Quantity):
        data = registry.Quantity(data, "dimensionless")

    return data.to(unit)


def array_extract_units(data):
    """ extract the units of an array

    If ``data`` is not a quantity, the units are ``None``
    """
    try:
        return data.units
    except AttributeError:
        return None


def array_strip_units(data):
    """ strip the units of a quantity """
    try:
        return data.magnitude
    except AttributeError:
        return data


def extract_units(obj):
    if isinstance(obj, Dataset):
        vars_units = {
            name: array_extract_units(value.data)
            for name, value in obj.data_vars.items()
        }
        coords_units = {
            name: array_extract_units(value.data) for name, value in obj.coords.items()
        }

        units = {**vars_units, **coords_units}
    elif isinstance(obj, DataArray):
        vars_units = {obj.name: array_extract_units(obj.data)}
        coords_units = {
            name: array_extract_units(value.data) for name, value in obj.coords.items()
        }

        units = {**vars_units, **coords_units}
    elif isinstance(obj, Variable):
        vars_units = {None: array_extract_units(obj.data)}

        units = {**vars_units}
    else:
        raise ValueError(f"unknown type: {type(obj)}")

    return units
