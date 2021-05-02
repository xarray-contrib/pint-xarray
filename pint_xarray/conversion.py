import itertools

import pint
from xarray import DataArray, Dataset, IndexVariable, Variable

from .errors import format_error_message

unit_attribute_name = "units"
slice_attributes = ("start", "stop", "step")


def array_attach_units(data, unit):
    """attach a unit to the data

    Parameters
    ----------
    data : array-like
        The data to attach units to.
    unit : pint.Unit
        The desired unit.

    Returns
    -------
    quantity : pint.Quantity
    """

    if unit is None:
        return data

    if not isinstance(unit, pint.Unit):
        raise ValueError(f"cannot use {unit!r} as a unit")

    if isinstance(data, pint.Quantity):
        raise ValueError(
            f"Cannot attach unit {unit!r} to quantity: data "
            f"already has units {data.units}"
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
    """strip the units of a quantity"""
    try:
        return data.magnitude
    except AttributeError:
        return data


def attach_units_variable(variable, units):
    if isinstance(variable, IndexVariable):
        new_obj = variable.copy()
        if units is not None:
            new_obj.attrs[unit_attribute_name] = units
    elif isinstance(variable, Variable):
        new_data = array_attach_units(variable.data, units)
        new_obj = variable.copy(data=new_data)
    else:
        raise ValueError(f"invalid type: {variable!r}")

    return new_obj


def attach_units(obj, units):
    if isinstance(obj, DataArray):
        old_name = obj.name
        new_name = old_name if old_name is not None else "<this-array>"
        ds = obj.rename(new_name).to_dataset()
        units = units.copy()
        units[new_name] = units.get(old_name)

        new_ds = attach_units(ds, units)
        new_obj = new_ds.get(new_name).rename(old_name)
    elif isinstance(obj, Dataset):
        attached = {}
        rejected_vars = {}
        for name, var in obj.variables.items():
            unit = units.get(name)
            try:
                converted = attach_units_variable(var, unit)
                attached[name] = converted
            except ValueError as e:
                rejected_vars[name] = (unit, e)

        if rejected_vars:
            raise ValueError(format_error_message(rejected_vars, "attach"))

        data_vars = {
            name: var for name, var in attached.items() if name not in obj._coord_names
        }
        coords = {
            name: var for name, var in attached.items() if name in obj._coord_names
        }

        new_obj = Dataset(data_vars=data_vars, coords=coords, attrs=obj.attrs)
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
    else:
        raise ValueError(f"cannot attach unit attributes to {obj!r}: unknown type")

    return new_obj


def convert_units_variable(variable, units):
    if isinstance(variable, IndexVariable):
        if variable.level_names:
            # don't try to convert MultiIndexes
            return variable

        if units is not None:
            quantity = array_attach_units(
                variable.data, variable.attrs.get(unit_attribute_name)
            )
            converted = array_convert_units(quantity, units)
            new_obj = variable.copy(data=array_strip_units(converted))

            new_obj.attrs[unit_attribute_name] = array_extract_units(converted)
        else:
            new_obj = variable
    elif isinstance(variable, Variable):
        converted = array_convert_units(variable.data, units)
        new_obj = variable.copy(data=converted)
    else:
        raise ValueError(f"unknown type: {variable}")

    return new_obj


def convert_units(obj, units):
    if isinstance(obj, DataArray):
        original_name = obj.name
        name = obj.name if obj.name is not None else "<this-array>"

        units_ = units.copy()
        if obj.name in units_:
            units_[name] = units_[obj.name]

        ds = obj.rename(name).to_dataset()
        converted = convert_units(ds, units_)

        new_obj = converted[name].rename(original_name)
    elif isinstance(obj, Dataset):
        converted = {}
        failed = {}
        for name, var in obj.variables.items():
            unit = units.get(name)
            try:
                converted[name] = convert_units_variable(var, unit)
            except (ValueError, pint.errors.PintTypeError) as e:
                failed[name] = e

        if failed:
            raise ValueError(format_error_message(failed, "convert"))

        coords = {
            name: var for name, var in converted.items() if name in obj._coord_names
        }
        data_vars = {
            name: var for name, var in converted.items() if name not in obj._coord_names
        }

        new_obj = Dataset(data_vars=data_vars, coords=coords, attrs=obj.attrs)
    else:
        raise ValueError(f"cannot convert object: {obj}")

    return new_obj


def extract_units(obj):
    if isinstance(obj, Dataset):
        units = extract_unit_attributes(obj)
        dims = obj.dims
        units.update(
            {
                name: array_extract_units(value.data)
                for name, value in obj.variables.items()
                if name not in dims
            }
        )
    elif isinstance(obj, DataArray):
        original_name = obj.name
        name = obj.name if obj.name is not None else "<this-array>"

        ds = obj.rename(name).to_dataset()

        units = extract_units(ds)
        units[original_name] = units.pop(name)
    else:
        raise ValueError(f"unknown type: {type(obj)}")

    return units


def extract_unit_attributes(obj, attr="units"):
    if isinstance(obj, DataArray):
        original_name = obj.name
        name = obj.name if obj.name is not None else "<this-array>"

        ds = obj.rename(name).to_dataset()

        units = extract_unit_attributes(ds)
        units[original_name] = units.pop(name)
    elif isinstance(obj, Dataset):
        units = {name: var.attrs.get(attr, None) for name, var in obj.variables.items()}
    else:
        raise ValueError(
            f"cannot retrieve unit attributes from unknown type: {type(obj)}"
        )

    return units


def strip_units_variable(var):
    data = array_strip_units(var.data)
    return var.copy(data=data)


def strip_units(obj):
    if isinstance(obj, DataArray):
        original_name = obj.name
        name = obj.name if obj.name is not None else "<this-array>"
        ds = obj.rename(name).to_dataset()
        stripped = strip_units(ds)

        new_obj = stripped[name].rename(original_name)
    elif isinstance(obj, Dataset):
        data_vars = {
            name: strip_units_variable(variable)
            for name, variable in obj.variables.items()
            if name not in obj._coord_names
        }
        coords = {
            name: strip_units_variable(variable)
            for name, variable in obj.variables.items()
            if name in obj._coord_names
        }

        new_obj = Dataset(data_vars=data_vars, coords=coords, attrs=obj.attrs)
    else:
        raise ValueError("cannot strip units from {obj!r}: unknown type")

    return new_obj


def strip_unit_attributes(obj, attr="units"):
    if isinstance(obj, DataArray):
        original_name = obj.name
        name = obj.name if obj.name is not None else "<this-array>"

        ds = obj.rename(name).to_dataset()

        stripped = strip_unit_attributes(ds)

        new_obj = stripped[name].rename(original_name)
    elif isinstance(obj, Dataset):
        new_obj = obj.copy()
        for var in new_obj.variables.values():
            var.attrs.pop(attr, None)
    else:
        raise ValueError(f"cannot strip unit attributes from unknown type: {type(obj)}")

    return new_obj


def slice_extract_units(indexer):
    elements = {name: getattr(indexer, name) for name in slice_attributes}
    extracted_units = [
        array_extract_units(value)
        for name, value in elements.items()
        if value is not None
    ]
    none_values = [_ is None for _ in extracted_units]
    if not extracted_units or all(none_values):
        # empty slice (slice(None)) or slice without units
        return None

    dimensionalities = {
        str(getattr(units, "dimensionality", "dimensionless"))
        for units in extracted_units
    }
    if len(dimensionalities) > 1:
        raise ValueError(f"incompatible units in {indexer}: {dimensionalities}")

    units = [_ for _ in extracted_units if _ is not None]
    if len(set(units)) == 1:
        return units[0]
    else:
        units_ = units[0]
        registry = units_._REGISTRY
        return registry.Quantity(1, units_).to_base_units().units


def convert_units_slice(indexer, units):
    attrs = {name: getattr(indexer, name) for name in slice_attributes}
    converted = {
        name: array_convert_units(value, units) if value is not None else None
        for name, value in attrs.items()
    }
    args = [converted[name] for name in slice_attributes]

    return slice(*args)


def convert_indexer_units(indexers, units):
    def convert(indexer, units):
        if isinstance(indexer, slice):
            return convert_units_slice(indexer, units)
        elif isinstance(indexer, DataArray):
            return convert_units(indexer, {None: units})
        elif isinstance(indexer, Variable):
            return convert_units_variable(indexer, units)
        else:
            return array_convert_units(indexer, units)

    converted = {}
    invalid = {}
    for name, indexer in indexers.items():
        indexer_units = units.get(name)
        try:
            converted[name] = convert(indexer, indexer_units)
        except (ValueError, pint.errors.PintTypeError) as e:
            invalid[name] = e

    if invalid:
        raise ValueError(format_error_message(invalid, "convert_indexers"))

    return converted


def extract_indexer_units(indexer):
    if isinstance(indexer, slice):
        return slice_extract_units(indexer)
    elif isinstance(indexer, (DataArray, Variable)):
        return array_extract_units(indexer.data)
    else:
        return array_extract_units(indexer)


def strip_indexer_units(indexer):
    if isinstance(indexer, slice):
        return slice(
            array_strip_units(indexer.start),
            array_strip_units(indexer.stop),
            array_strip_units(indexer.step),
        )
    elif isinstance(indexer, DataArray):
        return strip_units(indexer)
    elif isinstance(indexer, Variable):
        return strip_units_variable(indexer)
    else:
        return array_strip_units(indexer)
