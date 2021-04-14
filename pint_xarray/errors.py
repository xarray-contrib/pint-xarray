import pint


def format_error_message(mapping, op):
    sep = "\n    " if len(mapping) == 1 else "\n -- "
    if op == "attach":
        message = "Cannot attach units:"
        message = sep.join(
            [message]
            + [
                f"cannot attach units to variable {key!r}: {unit} (reason: {str(e)})"
                for key, (unit, e) in mapping.items()
            ]
        )
    elif op == "parse":
        message = "Cannot parse units:"
        message = sep.join(
            [message]
            + [
                f"invalid units for variable {key!r}: {unit} ({type}) (reason: {str(e)})"
                for key, (unit, type, e) in mapping.items()
            ]
        )
    elif op == "convert":
        message = "Cannot convert variables:"
        message = sep.join(
            [message]
            + [
                f"incompatible units for variable {key!r}: {error}"
                for key, error in mapping.items()
            ]
        )
    else:
        raise ValueError("invalid op")

    return message


class DimensionalityError(pint.DimensionalityError):
    """Raised when trying to convert between incompatible units

    Parameters
    ----------
    units1 : mapping of hashable to unit-like
        The units of the existing object which are incompatible with the new units.
    units2 : mapping of hashable to unit-like
        The units to convert the object to which are incompatible with the existing
        units.
    """

    def __init__(self, units1, units2):
        if not units1:
            raise ValueError("no units given")
        elif units1.keys() != units2.keys():
            raise ValueError("units1 and units2 must have the same keys")

        self.incompatible_units = {
            key: (units1[key], units2[key]) for key in units1.keys()
        }

    def __str__(self):
        incompatible_units = self.incompatible_units

        message = "Cannot convert some variables:"
        sep = " " if len(incompatible_units) == 1 else "\n -- "

        message = sep.join(
            [message]
            + [
                f"incompatible units for {key}: {u1.dimensionality} != {u2.dimensionality}"
                for key, (u1, u2) in incompatible_units.items()
            ]
        )
        return message
