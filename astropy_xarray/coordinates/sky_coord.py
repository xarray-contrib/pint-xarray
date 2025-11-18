import itertools
from collections.abc import Generator
from enum import Enum, auto

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord

from astropy_xarray.coordinates.frame import dump_frame, load_frame, load_representation


class DatasetRepresentation(Enum):
    FRAME = auto()
    """
    Frame selected repesentation and differential component names, simplifying for any `unitspherical` types in data.

    Same as `get_representation_component_names()` and `get_differential_component_names()`
    """

    FRAME_FULL = auto()
    """
    Frame selected repesentation and differential component names.

    Same as `get_representation_component_names()` and `get_differential_component_names()`
    """

    FRAME_DEFAULT = auto()
    """Frame default components names."""

    FRAME_DATA = auto()
    """
    Frame component names from resetting the display representation and differential to match the data.

    Guarentees roundtrip accuracy.
    """

    DATA = auto()
    """Data representation components. Guarentees roundtrip accuracy."""


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
    coords: list[tuple[str, list]] | None = None,
) -> dict[str, xr.DataArray]:
    return {
        name: xr.DataArray(
            data=component, coords=dict(coords) if coords is not None else None
        )
        for name, component in _skycoord_components(skycoord).items()
    }


def skycoord_to_dataset(
    skycoord: SkyCoord,
    coords: list[tuple[str, np.ndarray]] | None = None,
) -> xr.Dataset:
    """Convert a SkyCoord object to an xarray Dataset.

    Args:
        skycoord: SkyCoord object.
        coords: coordinates to assign to multidimensional skycoords. Defaults to None.

    Returns:
        quantified dataset.
    """
    return xr.Dataset(
        coords=dict(coords) if coords is not None else None,
        data_vars=_skycoord_to_dataarrays(skycoord, coords),
        attrs=dict(
            frame=dump_frame(skycoord.frame),
        ),
    )


def dataset_to_skycoord(ds: xr.Dataset, use_frame_names: bool = True) -> SkyCoord:
    """Convert a SkyCoord-based Dataset with metadata attributes to a SkyCoord.

    Args:
        ds: skycoord dataset.
        use_frame_names: True if frame component names are used a data_vars. Defaults to True.

    Returns:
        SkyCoord object.
    """
    dsq = ds.astropy.quantify()
    frame = load_frame(ds.attrs["frame"])
    frame._data = load_representation(
        ds.attrs["frame"]["data"]["representation_type"],
        ds.attrs["frame"]["data"].get("differential_type"),
        ds.attrs["frame"]["name"] if use_frame_names else None,
        {k: v.data for k, v in dsq.data_vars.items()},
    )
    frame.representation_type = ds.attrs["frame"]["representation_type"]
    frame.differential_type = ds.attrs["frame"]["differential_type"]
    return SkyCoord(frame)
