from xarray.core.formatting import format_array_flat


def inline_repr(quantity, max_width):
    magnitude = quantity.magnitude
    units = quantity.units

    units_repr = str(units)
    data_repr = format_array_flat(magnitude, max_width - len(units_repr) - 3)

    return f"[{units_repr}] {data_repr}"
