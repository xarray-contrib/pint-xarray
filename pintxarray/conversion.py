import pint


def array_convert_units(data, unit):
    """ convert the units of an array

    This is roughly the same as ``data.to(unit)``.
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
