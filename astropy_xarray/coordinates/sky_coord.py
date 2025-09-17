import itertools
from collections.abc import Generator

import astropy.units as u
import xarray as xr
from astropy.coordinates import CartesianRepresentation, SkyCoord

from astropy_xarray.coordinates.frame import dump_frame, load_frame, load_representation


def _skycoord_representation_component_names(
    skycoord: SkyCoord,
    skip_implicit=True,
    use_data_components=True,
) -> tuple[str, ...]:
    if use_data_components:
        return tuple(
            name
            for name in skycoord.data.components
            if not skip_implicit
            or skycoord.representation_type == CartesianRepresentation
            or getattr(skycoord.data, name).unit is not u.dimensionless_unscaled
        )
    else:
        return tuple(
            name
            for name in skycoord.get_representation_component_names()
            if not skip_implicit
            or skycoord.representation_type == CartesianRepresentation
            or getattr(skycoord, name).unit is not u.dimensionless_unscaled
        )


def _skycoord_differential_component_names(
    skycoord: SkyCoord,
    skip_implicit=True,
    use_data_components=True,
) -> tuple[str, ...]:
    diff = skycoord.data.differentials.get("s")
    if diff is None:
        return ()
    if use_data_components:
        return tuple(
            name
            for name in diff.components
            if not skip_implicit
            or getattr(diff, name).unit is not u.dimensionless_unscaled
        )
    else:
        return tuple(skycoord.get_representation_component_names("s"))


def _skycoord_component_names(
    skycoord: SkyCoord,
    skip_implicit=True,
    use_data_components=True,
) -> Generator[str, None, None]:
    if use_data_components:
        yield from itertools.chain(
            _skycoord_representation_component_names(skycoord, skip_implicit),
            _skycoord_differential_component_names(skycoord, skip_implicit),
        )
    else:
        yield from itertools.chain(
            _skycoord_representation_component_names(skycoord, skip_implicit),
            _skycoord_differential_component_names(skycoord, skip_implicit),
        )


def _skycoord_components(
    skycoord: SkyCoord, use_frame_names=True, skip_dimensionless=True
) -> dict[str, u.Quantity]:
    return {
        name: (
            getattr(skycoord.data, name)
            if hasattr(skycoord.data, name)
            else getattr(skycoord.data.differentials["s"], name)
        )
        for name in _skycoord_component_names(
            skycoord, use_frame_names, skip_dimensionless
        )
    }


def _skycoord_to_dataarrays(
    skycoord: SkyCoord,
    coords: list[tuple[str, list]] | None = None,
    skip_dimensionless=True,
) -> dict[str, xr.DataArray]:
    return {
        name: xr.DataArray(
            data=component, coords=dict(coords) if coords is not None else None
        )
        for name, component in _skycoord_components(
            skycoord, skip_dimensionless
        ).items()
    }


def skycoord_to_dataset(
    skycoord: SkyCoord,
    coords: list[tuple[str, list[int]]] | None = None,
) -> xr.Dataset:
    return xr.Dataset(
        coords=dict(coords) if coords is not None else None,
        data_vars=_skycoord_to_dataarrays(skycoord, coords),
        attrs=dict(
            frame=dump_frame(skycoord.frame),
        ),
    )


def dataset_to_skycoord(ds: xr.Dataset) -> SkyCoord:
    dsq = ds.astropy.quantify()
    frame = load_frame(ds.attrs["frame"]).realize_frame(
        representation_type=ds.attrs["frame"]["representation_type"],
        differential_type=ds.attrs["frame"]["differential_type"],
        data=load_representation(
            ds.attrs["frame"]["data"]["representation_type"],
            ds.attrs["frame"]["data"].get("differential_type"),
            {k: v.data for k, v in dsq.data_vars.items()},
        ),
    )
    return SkyCoord(frame)
