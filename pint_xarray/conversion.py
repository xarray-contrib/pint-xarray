import itertools
import re

import pint
from xarray import DataArray, Dataset, IndexVariable, Variable

from .compat import call_on_dataset
from .errors import format_error_message

no_unit_values = ("none", None)
unit_attribute_name = "units"
slice_attributes = ("start", "stop", "step")
temporary_name = "<this-array>"

time_units_re = r"\w+"
datetime_re = r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?"
datetime_units_re = re.compile(rf"{time_units_re} since {datetime_re}")


def is_datetime_unit(unit):
    return isinstance(unit, str) and datetime_units_re.match(unit) is not None


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
    if unit in no_unit_values:
        return data

    if not isinstance(unit, pint.Unit):
        raise ValueError(f"cannot use {unit!r} as a unit")

    if isinstance(data, pint.Quantity):
        if data.units == unit:
            return data

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


def dataset_from_variables(variables, coords, attrs):
    data_vars = {name: var for name, var in variables.items() if name not in coords}
    coords = {name: var for name, var in variables.items() if name in coords}

    return Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def attach_units_dataset(obj, units):
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
        raise ValueError(rejected_vars)

    return dataset_from_variables(attached, obj._coord_names, obj.attrs)


def attach_units(obj, units):
    if not isinstance(obj, (DataArray, Dataset)):
        raise ValueError(f"cannot attach units to {obj!r}: unknown type")

    if isinstance(obj, DataArray):
        units = units.copy()
        if obj.name in units:
            units[temporary_name] = units.get(obj.name)

    try:
        new_obj = call_on_dataset(
            attach_units_dataset, obj, name=temporary_name, units=units
        )
    except ValueError as e:
        (rejected_vars,) = e.args
        if temporary_name in rejected_vars:
            rejected_vars[obj.name] = rejected_vars.pop(temporary_name)

        raise ValueError(format_error_message(rejected_vars, "attach")) from e

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


def convert_units_dataset(obj, units):
    converted = {}
    failed = {}
    for name, var in obj.variables.items():
        unit = units.get(name)
        try:
            converted[name] = convert_units_variable(var, unit)
        except (ValueError, pint.errors.PintTypeError) as e:
            failed[name] = e

    if failed:
        raise ValueError(failed)

    return dataset_from_variables(converted, obj._coord_names, obj.attrs)


def convert_units(obj, units):
    if not isinstance(obj, (DataArray, Dataset)):
        raise ValueError(f"cannot convert object: {obj!r}: unknown type")

    if isinstance(obj, DataArray):
        units = units.copy()
        if obj.name in units:
            units[temporary_name] = units.pop(obj.name)

    try:
        new_obj = call_on_dataset(
            convert_units_dataset, obj, name=temporary_name, units=units
        )
    except ValueError as e:
        (failed,) = e.args
        if temporary_name in failed:
            failed[obj.name] = failed.pop(temporary_name)

        raise ValueError(format_error_message(failed, "convert")) from e

    return new_obj


def extract_units_dataset(obj):
    return {name: array_extract_units(var.data) for name, var in obj.variables.items()}


def extract_units(obj):
    if not isinstance(obj, (DataArray, Dataset)):
        raise ValueError(f"unknown type: {type(obj)}")

    unit_attributes = extract_unit_attributes(obj)

    units = call_on_dataset(extract_units_dataset, obj, name=temporary_name)
    if temporary_name in units:
        units[obj.name] = units.pop(temporary_name)

    units_ = unit_attributes.copy()
    units_.update({k: v for k, v in units.items() if v is not None})

    return units_


def extract_unit_attributes_dataset(obj, attr="units"):
    all_units = {name: var.attrs.get(attr, None) for name, var in obj.variables.items()}

    return {
        name: unit for name, unit in all_units.items() if not is_datetime_unit(unit)
    }


def extract_unit_attributes(obj, attr="units"):
    if not isinstance(obj, (DataArray, Dataset)):
        raise ValueError(
            f"cannot retrieve unit attributes from unknown type: {type(obj)}"
        )

    units = call_on_dataset(
        extract_unit_attributes_dataset, obj, name=temporary_name, attr=attr
    )
    if temporary_name in units:
        units[obj.name] = units.pop(temporary_name)

    return units


def strip_units_variable(var):
    if not isinstance(var.data, pint.Quantity):
        return var

    data = array_strip_units(var.data)
    return var.copy(data=data)


def strip_units_dataset(obj):
    variables = {name: strip_units_variable(var) for name, var in obj.variables.items()}

    return dataset_from_variables(variables, obj._coord_names, obj.attrs)


def strip_units(obj):
    if not isinstance(obj, (DataArray, Dataset)):
        raise ValueError("cannot strip units from {obj!r}: unknown type")

    return call_on_dataset(strip_units_dataset, obj, name=temporary_name)


def strip_unit_attributes_dataset(obj, attr="units"):
    new_obj = obj.copy()
    for var in new_obj.variables.values():
        if is_datetime_unit(var.attrs.get(attr, "")):
            continue

        var.attrs.pop(attr, None)

    return new_obj


def strip_unit_attributes(obj, attr="units"):
    if not isinstance(obj, (DataArray, Dataset)):
        raise ValueError(f"cannot strip unit attributes from unknown type: {type(obj)}")

    return call_on_dataset(
        strip_unit_attributes_dataset, obj, name=temporary_name, attr=attr
    )


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
