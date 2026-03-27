import numpy as np
from astropy.time import Time, TimeDelta

import astropy_xarray
import astropy_xarray.formatting


def time_inline_repr(time: Time, max_width: int):
    value = time.datetime64
    scale_repr = f"{time.format} {time.scale}"
    if isinstance(value, np.ndarray):
        data_repr = astropy_xarray.formatting.format_array_flat(
            value, max_width - len(scale_repr) - 3
        )
    else:
        data_repr = astropy_xarray.formatting.maybe_truncate(
            repr(value), max_width - len(scale_repr) - 3
        )

    return f"[{scale_repr}] {data_repr}"


def time_delta_inline_repr(time: TimeDelta, max_width: int):
    value = time.value
    scale_repr = f"{time.format} {time.scale}"
    if isinstance(value, np.ndarray):
        data_repr = astropy_xarray.formatting.format_array_flat(
            value, max_width - len(scale_repr) - 3
        )
    else:
        data_repr = astropy_xarray.formatting.maybe_truncate(
            repr(value), max_width - len(scale_repr) - 3
        )

    return f"[{scale_repr}] {data_repr}"


Time.dtype = property(lambda self: self.value.dtype)
Time.nbytes = property(lambda self: self.value.nbytes)
Time.__array__ = lambda obj: obj.value
Time.__array_ufunc__ = lambda obj, *args, **kwargs: obj.value.__array_ufunc__(
    *args, **kwargs
)
Time._repr_inline_ = time_inline_repr
Time.__array_namespace = np


TimeDelta.dtype = property(lambda self: self.value.dtype)
TimeDelta.nbytes = property(lambda self: self.value.nbytes)
TimeDelta.__array__ = lambda obj: obj.value
TimeDelta.__array_ufunc__ = lambda obj, *args, **kwargs: obj.value.__array_ufunc__(
    *args, **kwargs
)
TimeDelta._repr_inline_ = time_delta_inline_repr
TimeDelta.__array_namespace = np
