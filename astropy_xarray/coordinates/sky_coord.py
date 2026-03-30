import itertools
from collections.abc import Generator

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.utils import ShapedLikeNDArray

from astropy_xarray.coordinates.frame import dump_frame, load_frame, load_representation

_ArrayLike = list | np.ndarray | ShapedLikeNDArray


def _skycoord_representation_component_names(
    skycoord: SkyCoord,
) -> tuple[str, ...]:
    frame = skycoord.frame.replicate_without_data(copy=True)
    frame.representation_type = type(skycoord.frame.data)
    if skycoord.frame.data.differentials:
        frame.differential_type = type(skycoord.frame.data.differentials.get("s"))

    return tuple(name for name in frame.get_representation_component_names())


def _skycoord_differential_component_names(
    skycoord: SkyCoord,
) -> tuple[str, ...]:
    if skycoord.data.differentials.get("s") is None:
        return ()

    frame = skycoord.frame.replicate_without_data(copy=True)
    frame.representation_type = type(skycoord.frame.data)
    if skycoord.frame.data.differentials:
        frame.differential_type = type(skycoord.frame.data.differentials.get("s"))

    return tuple(name for name in frame.get_representation_component_names("s"))


def _skycoord_component_names(
    skycoord: SkyCoord,
) -> Generator[str, None, None]:
    yield from itertools.chain(
        _skycoord_representation_component_names(skycoord),
        _skycoord_differential_component_names(skycoord),
    )
    # use default frame components


def _skycoord_components(skycoord: SkyCoord) -> dict[str, u.Quantity]:
    frame = skycoord.frame.replicate_without_data(copy=True)
    frame.representation_type = type(skycoord.frame.data)
    if skycoord.frame.data.differentials:
        frame.differential_type = type(skycoord.frame.data.differentials.get("s"))
    data = skycoord.frame.data

    # need to access via data components to avoid rounding
    frame_to_rep = (
        frame.get_representation_component_names()
        | frame.get_representation_component_names("s")
    )
    return {
        name: (
            getattr(data, frame_to_rep[name])
            if hasattr(data, frame_to_rep[name])
            else getattr(data.differentials["s"], frame_to_rep[name])
        )
        for name in _skycoord_component_names(skycoord)
    }


def _skycoord_to_dataarrays(
    skycoord: SkyCoord,
    coords: dict[str, tuple[str, _ArrayLike] | _ArrayLike] | None = None,
) -> dict[str, xr.DataArray]:

    dims = (
        [
            coords[key][0] if isinstance(coords[key], tuple) else key
            for key in coords.keys()
        ]
        if coords is not None
        else None
    )
    return {
        name: xr.DataArray(data=component, coords=coords, dims=dims)
        for name, component in _skycoord_components(skycoord).items()
    }


def skycoord_to_dataset(
    skycoord: SkyCoord,
    coords: dict[str, tuple[str, _ArrayLike] | _ArrayLike] | None = None,
) -> xr.Dataset:
    """Convert a SkyCoord object to an xarray Dataset.

    Args:
        skycoord: SkyCoord object.
        coords: coordinates to assign to multidimensional skycoords. Defaults to None.

    Returns:
        quantified dataset.
    """
    return xr.Dataset(
        coords=coords if coords is not None else None,
        data_vars=_skycoord_to_dataarrays(skycoord, coords),
        attrs=dict(
            frame=dump_frame(skycoord.frame),
        ),
    )


def dataset_to_skycoord(ds: xr.Dataset) -> SkyCoord:
    """Convert a SkyCoord-based Dataset with metadata attributes to a SkyCoord.

    Args:
        ds: skycoord dataset.

    Returns:
        SkyCoord object.
    """
    dsq = ds.astropy.quantify()
    frame = load_frame(ds.attrs["frame"])
    frame._data = load_representation(
        ds.attrs["frame"]["data"]["representation_type"],
        ds.attrs["frame"]["data"].get("differential_type"),
        ds.attrs["frame"]["name"],
        {k: v.data for k, v in dsq.data_vars.items()},
    )
    frame.representation_type = ds.attrs["frame"]["representation_type"]
    frame.differential_type = ds.attrs["frame"]["differential_type"]
    return SkyCoord(frame)
