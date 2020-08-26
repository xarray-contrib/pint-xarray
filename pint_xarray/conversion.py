import itertools

import pint
from xarray import DataArray, Dataset, Variable


def array_attach_units(data, unit, registry=None):
    """attach a unit to the data

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
    """convert the units of an array

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
    """extract the units of an array

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


def attach_units(obj, units, registry=None):
    if isinstance(obj, DataArray):
        old_name = obj.name
        new_name = old_name if old_name is not None else "<this-array>"
        ds = obj.rename(new_name).to_dataset()
        units = units.copy()
        units[new_name] = units.get(old_name)

        new_ds = attach_units(ds, units, registry=registry)
        new_obj = new_ds.get(new_name).rename(old_name)
    elif isinstance(obj, Dataset):
        data_vars = {
            name: attach_units(
                array.variable, {None: units.get(name)}, registry=registry
            )
            for name, array in obj.data_vars.items()
        }
        coords = {
            name: attach_units(
                array.variable, {None: units.get(name)}, registry=registry
            )
            for name, array in obj.coords.items()
        }

        new_obj = Dataset(data_vars=data_vars, coords=coords, attrs=obj.attrs)
    elif isinstance(obj, Variable):
        new_data = array_attach_units(obj.data, units.get(None), registry=registry)
        new_obj = obj.copy(data=new_data)
    else:
        raise ValueError(f"cannot attach units to {obj!r}: unknown type")

    return new_obj


def attach_unit_attributes(obj, units, attr="units"):
    new_obj = obj.copy()
    if isinstance(obj, DataArray):
        for name, var in itertools.chain([(obj.name, new_obj)], new_obj.coords.items()):
            unit = units.get(name)
            if unit is None:
                continue

            var.attrs[attr] = unit
    elif isinstance(obj, Dataset):
        for name, var in new_obj.variables.items():
            unit = units.get(name)
            if unit is None:
                continue

            var.attrs[attr] = unit
    elif isinstance(obj, Variable):
        new_obj.attrs[attr] = units.get(None)
    else:
        raise ValueError(f"cannot attach unit attributes to {obj!r}: unknown type")

    return new_obj


def convert_units(obj, units):
    if not isinstance(units, dict):
        units = {None: units}

    if isinstance(obj, Variable):
        new_data = array_convert_units(obj.data, units.get(None))
        new_obj = obj.copy(data=new_data)
    elif isinstance(obj, DataArray):
        original_name = obj.name
        name = obj.name if obj.name is not None else "<this-array>"

        units_ = units.copy()
        if obj.name in units_:
            units_[name] = units_[obj.name]

        ds = obj.rename(name).to_dataset()
        converted = convert_units(ds, units_)

        new_obj = converted[name].rename(original_name)
    elif isinstance(obj, Dataset):
        coords = {
            name: convert_units(data.variable, units.get(name))
            if name not in obj.dims
            else data
            for name, data in obj.coords.items()
        }
        data_vars = {
            name: convert_units(data.variable, units.get(name))
            for name, data in obj.items()
        }

        new_obj = Dataset(coords=coords, data_vars=data_vars, attrs=obj.attrs)
    else:
        raise ValueError("cannot convert non-xarray objects")

    return new_obj


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


def extract_unit_attributes(obj, attr="units"):
    if isinstance(obj, DataArray):
        variables = itertools.chain([(obj.name, obj)], obj.coords.items())
        units = {name: var.attrs.get(attr, None) for name, var in variables}
    elif isinstance(obj, Dataset):
        units = {name: var.attrs.get(attr, None) for name, var in obj.variables.items()}
    elif isinstance(obj, Variable):
        units = {None: obj.attrs.get(attr, None)}
    else:
        raise ValueError(
            f"cannot retrieve unit attributes from unknown type: {type(obj)}"
        )

    return units


def strip_units(obj):
    if isinstance(obj, Variable):
        data = array_strip_units(obj.data)
        new_obj = obj.copy(data=data)
    elif isinstance(obj, DataArray):
        original_name = obj.name
        name = obj.name if obj.name is not None else "<this-array>"
        ds = obj.rename(name).to_dataset()
        stripped = strip_units(ds)

        new_obj = stripped[name].rename(original_name)
    elif isinstance(obj, Dataset):
        data_vars = {
            name: strip_units(array.variable) for name, array in obj.data_vars.items()
        }
        coords = {
            name: strip_units(array.variable) for name, array in obj.coords.items()
        }

        new_obj = Dataset(data_vars=data_vars, coords=coords, attrs=obj.attrs)
    else:
        raise ValueError("cannot strip units from {obj!r}: unknown type")

    return new_obj


def strip_unit_attributes(obj, attr="units"):
    new_obj = obj.copy()
    if isinstance(obj, DataArray):
        variables = itertools.chain([(new_obj.name, new_obj)], new_obj.coords.items())
        for _, var in variables:
            var.attrs.pop(attr, None)
    elif isinstance(obj, Dataset):
        for var in new_obj.variables.values():
            var.attrs.pop(attr, None)
    elif isinstance(obj, Variable):
        new_obj.attrs.pop(attr, None)
    else:
        raise ValueError(
            f"cannot retrieve unit attributes from unknown type: {type(obj)}"
        )

    return new_obj
