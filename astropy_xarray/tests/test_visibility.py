import json

import astropy.units as u
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy.coordinates import (
    GCRS,
    ICRS,
    AltAz,
    SkyCoord,
)
from xarray.tests import requires_zarr

from astropy_xarray.coordinates import (
    dataset_to_skycoord,
    skycoord_to_dataset,
)
from astropy_xarray.tests.multi_index_utils import (
    open_datatree_decompress_multi_index,
    save_datatree_compress_multi_index,
)

FIELD_PHASE_CENTER_ICRS = [
    SkyCoord(
        ra=[2, 0.2, 0.02, 0.002] * u.deg,
        dec=[4, 0.4, 0.04, 0.004] * u.deg,
        frame="icrs",
    ),
    SkyCoord(
        ra=[5, 0.5, 0.05, 0.005] * u.deg,
        dec=[6, 0.6, 0.06, 0.006] * u.deg,
        frame="icrs",
    ),
]

FIELD_PHASE_CENTER_FK5 = [
    SkyCoord(
        ra=[0.2, 0.2, 0.02, 0.002] * u.rad,
        dec=[0.4, 0.4, 0.04, 0.004] * u.rad,
        frame="fk5",
    ),
    SkyCoord(
        ra=[0.5, 0.5, 0.05, 0.005] * u.rad,
        dec=[0.6, 0.6, 0.06, 0.006] * u.rad,
        frame="fk5",
    ),
]


# topocentric
FIELD_PHASE_CENTER_ALTAZ = [
    SkyCoord(
        alt=[0.2, 0.2, 0.02, 0.002] * u.rad,
        az=[0.4, 0.4, 0.04, 0.004] * u.rad,
        frame=AltAz(),
    ),
    SkyCoord(
        alt=[0.5, 0.5, 0.05, 0.005] * u.rad,
        az=[0.6, 0.6, 0.06, 0.006] * u.rad,
        frame=AltAz(),
    ),
]

# geocentric
FIELD_PHASE_CENTER_GEO = [
    SkyCoord(
        x=[0.2, 0.2] * u.km,
        y=[0.4, 0.4] * u.km,
        z=[0.4, 0.4] * u.km,
        frame=GCRS(),
        representation_type="cartesian",
    ),
    SkyCoord(
        x=[0.5, 0.5] * u.km,
        y=[0.6, 0.6] * u.km,
        z=[0.6, 0.6] * u.km,
        representation_type="cartesian",
        frame=GCRS(
            obstime="J2000.000",
            obsgeoloc=(0.0, 0.0, 0.0) * u.m,
            obsgeovel=(0.0, 0.0, 0.0) * u.m / u.s,
        ),
    ),
]


def astropy_encode_json(data: xr.DataArray | xr.Dataset | xr.DataTree) -> bytes:
    if isinstance(data, xr.DataTree):
        dict_data = {
            group: ds.to_dict("array")
            for group, ds in data.astropy.dequantify().to_dict().items()
        }
    else:
        dict_data = data.astropy.dequantify().to_dict("array")
    return json.dumps(dict_data)  # type: ignore


def astropy_encode_msgpack(data: xr.DataArray | xr.Dataset | xr.DataTree) -> bytes:
    if isinstance(data, xr.DataTree):
        dict_data = {
            group: ds.to_dict("array")
            for group, ds in data.astropy.dequantify().to_dict().items()
        }
    else:
        dict_data = data.astropy.dequantify().to_dict("array")
    return msgpack.packb(dict_data, default=msgpack_numpy.encode)  # type: ignore


def astropy_decode_msgpack(data: bytes | memoryview):
    data_dict: dict = msgpack.unpackb(data, object_hook=msgpack_numpy.decode)
    if "/" in data_dict:
        dt_dict = {
            group: xr.Dataset.from_dict(ds_dict).astropy.quantify()
            for group, ds_dict in data_dict.items()
        }
        return xr.DataTree.from_dict(dt_dict)
    if "data_vars" in data_dict:
        return xr.Dataset.from_dict(data_dict).astropy.quantify()
    else:
        return xr.DataArray.from_dict(data_dict).astropy.quantify()


def generate_baselines(antenna_count) -> list[tuple[int, int]]:
    baselines = []
    for i in range(antenna_count):
        for j in range(i + 1, antenna_count):
            baselines.append((i, j))
    return baselines


@requires_zarr
def test_simple_visibility_dataset():
    a = 4  # antennas
    b = int(a * (a - 1) / 2)  # baselines
    p = 4  # polarisations
    f = 7  # frequencies
    t = 2  # timestamps

    # import cf_xarray as cfxr
    baselines = generate_baselines(a)
    antenna1, antenna2 = map(np.array, zip(*baselines))

    VISIBILITY = xr.DataTree(
        xr.Dataset(
            coords=dict(
                time=xr.DataArray(
                    [1.0, 2.0] * u.s,
                    dims=["time"],
                ),
                frequency=xr.DataArray(
                    np.arange(1000.0, 1007.0) * u.Hz,
                    dims=["frequency"],
                ),
                polarisation=xr.DataArray(
                    ["XX", "XY", "YX", "YY"], dims=["polarisation"]
                ),
                # simple index
                baseline=xr.DataArray(np.arange(len(baselines)), dims=["baseline"]),
                antenna1=xr.DataArray(antenna1, dims=["baseline"]),
                antenna2=xr.DataArray(antenna2, dims=["baseline"]),
                uvw_label=xr.DataArray(["u", "v", "w"], dims=["uvw_label"]),
            ),
            data_vars=dict(
                uvw=xr.DataArray(
                    data=[
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [0.0, 0.0, 0.0],
                        ],
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [0.0, 0.0, 0.0],
                        ],
                    ]
                    * u.m,
                    dims=["time", "baseline", "uvw_label"],
                ),
                vis=xr.DataArray(
                    data=np.ones([t, b, f, p]),
                    dims=["time", "baseline", "frequency", "polarisation"],
                ),
            ),
        ),
        children=dict(
            field_phase_center=xr.DataTree(
                skycoord_to_dataset(
                    SkyCoord(ra=[0.1] * u.deg, dec=[0.5] * u.deg, frame=ICRS()),
                    coords=[
                        ("time_poly", [0]),
                    ],
                )
            ),
            # extensions
            calibrator=xr.DataTree(
                skycoord_to_dataset(
                    SkyCoord(
                        ra=[[0.5], [1.5], [0.2], [0.9], [1.1]] * u.deg,
                        dec=[[1.2], [0.8], [0.6], [1.0], [1.5]] * u.deg,
                    ),
                    coords=[
                        ("calibrator_label", ["A", "B", "C", "D", "E"]),
                        ("time_poly", [0]),
                    ],
                )
            ),
        ),
    )

    m = astropy_encode_msgpack(VISIBILITY)
    ds_out = astropy_decode_msgpack(m)
    xr.testing.assert_identical(VISIBILITY, ds_out)

    import zarr.storage as zs

    store = zs.MemoryStore()

    VISIBILITY.astropy.dequantify().to_zarr(store, mode="w", consolidated=True)
    dt_store = xr.open_datatree(
        store, engine="zarr", consolidated=True
    ).astropy.quantify()

    xr.testing.assert_identical(VISIBILITY, dt_store)

    expected = dataset_to_skycoord(VISIBILITY.field_phase_center.dataset)
    actual = dataset_to_skycoord(dt_store.field_phase_center.dataset)
    np.testing.assert_array_equal(actual, expected, strict=True)


@requires_zarr
@pytest.mark.parametrize(
    "skycoords,frame,unit",
    [
        (
            FIELD_PHASE_CENTER_ICRS,
            {"name": "icrs"},
            {"ra": "deg", "dec": "deg"},
        ),
        (
            FIELD_PHASE_CENTER_FK5,
            {"name": "fk5", "equinox": 1},
            {"ra": "deg", "dec": "deg"},
        ),
        (FIELD_PHASE_CENTER_GEO, {"name": "gcrs"}, {"x": "km", "y": "km", "z": "km"}),
    ],
)
def test_visibility_dataset(skycoords: list[SkyCoord], frame, unit):
    a = 4  # antennas
    b = int(a * (a - 1) / 2)  # baselines
    p = 4  # polarisations
    f = 7  # frequencies
    t = 2  # timestamps

    VISIBILITY = xr.DataTree(
        xr.Dataset(
            coords=dict(
                time=xr.DataArray(
                    [1.0, 2.0] * u.s,
                    dims=["time"],
                ),
                frequency=xr.DataArray(
                    np.arange(1000.0, 1007.0) * u.Hz,
                    dims=["frequency"],
                ),
                polarisation=xr.DataArray(
                    ["XX", "XY", "YX", "YY"], dims=["polarisation"]
                ),
                **xr.Coordinates.from_pandas_multiindex(
                    pd.MultiIndex.from_tuples(
                        generate_baselines(a), names=("antenna1", "antenna2")
                    ),
                    dim="baseline",
                ),
                uvw_label=xr.DataArray(["u", "v", "w"], dims=["uvw_label"]),
            ),
            data_vars=dict(
                uvw=xr.DataArray(
                    data=[
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [0.0, 0.0, 0.0],
                        ],
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [0.0, 0.0, 0.0],
                        ],
                    ]
                    * u.m,
                    dims=["time", "baseline", "uvw_label"],
                ),
                vis=xr.DataArray(
                    data=np.ones([t, b, f, p]),
                    dims=["time", "baseline", "frequency", "polarisation"],
                ),
            ),
        ),
        children=dict(
            field_phase_center=xr.DataTree(
                skycoord_to_dataset(
                    # SkyCoord(
                    #     SphericalRepresentation(
                    #         lon=[[0.1, 0.2]] * u.deg,
                    #         lat=[[0.5, 0.2]] * u.deg,
                    #         distance=[[0, 0]] * u.pc,
                    #     ).with_differentials(
                    #         SphericalDifferential(
                    #             d_lon=[[1, 1]] * u.deg / u.s,
                    #             d_lat=[[1, 1]] * u.deg / u.s,
                    #             d_distance=[[1, 1]] * u.pc / u.yr,
                    #         )
                    #     ),
                    #     frame=ICRS()
                    # ),
                    SkyCoord(
                        ra=[[0.1, 0.2]] * u.deg,
                        dec=[[0.5, 0.2]] * u.deg,
                        distance=[[0.0, 0.0]] * u.km,
                        pm_ra_cosdec=[[1.0, 1.0]] * u.deg / u.s,
                        pm_dec=[[0.5, 0.2]] * u.deg / u.s,
                        radial_velocity=[[1, 1]] * u.pc / u.yr,
                        frame=ICRS(),
                    ),
                    coords=[
                        ("phase_center_label", [0]),
                        ("time_poly", [0, 1]),
                    ],
                )
            ),
            # extensions
            calibrator=xr.DataTree(
                skycoord_to_dataset(
                    SkyCoord(
                        ra=[[0.5], [1.5], [0.2], [0.9], [1.1]] * u.deg,
                        dec=[[1.2], [0.8], [0.6], [1.0], [1.5]] * u.deg,
                    ),
                    coords=[
                        ("calibrator_label", ["A", "B", "C", "D", "E"]),
                        ("time_poly", [0]),
                    ],
                )
            ),
        ),
    )

    m = astropy_encode_msgpack(VISIBILITY)
    ds_out = astropy_decode_msgpack(m)
    xr.testing.assert_identical(VISIBILITY, ds_out)

    import zarr.storage as zs

    store = zs.MemoryStore()

    save_datatree_compress_multi_index(
        VISIBILITY.astropy.dequantify(), store, mode="w", consolidated=True
    )
    dt_store = open_datatree_decompress_multi_index(
        store, engine="zarr"
    ).astropy.quantify()

    xr.testing.assert_identical(VISIBILITY, dt_store)

    expected = dataset_to_skycoord(VISIBILITY.field_phase_center.dataset)
    actual = dataset_to_skycoord(dt_store.field_phase_center.dataset)

    np.testing.assert_array_equal(actual, expected, strict=True)
