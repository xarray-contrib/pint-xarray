from typing import TypeVar

import astropy.units as u
from astropy.time import Time


# Time
def dump_time(time: Time):
    return {
        "val": time.value,
        "format": time.format,
        "precision": time.precision,
        "scale": time.scale,
    }


# Quantity
def dump_quantity(q: u.Quantity):
    return {"value": float(q.value), "unit": str(q.unit)}


_T = TypeVar("_T", bound=u.Quantity | Time)


def load_optional_object(cls: type[_T], kwargs: dict | None) -> _T | None:
    return cls(**kwargs) if kwargs is not None else None
