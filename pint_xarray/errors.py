import pint


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
        if len(incompatible_units) == 1:
            sep = ""
            message += " "
        else:
            sep = "\n -- "

        message = sep.join(
            [message]
            + [
                f"incompatible units for {key}: {u1.dimensionality} != {u2.dimensionality}"
                for key, (u1, u2) in incompatible_units.items()
            ]
        )
        return message
