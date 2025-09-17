from collections.abc import Generator

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
import zarr.storage as zs
from astropy.coordinates import (
    Angle,
    BaseCoordinateFrame,
    BaseRepresentation,
    CartesianDifferential,
    CartesianRepresentation,
    CylindricalDifferential,
    CylindricalRepresentation,
    Distance,
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
    SphericalCosLatDifferential,
    SphericalRepresentation,
    UnitSphericalCosLatDifferential,
    UnitSphericalDifferential,
    UnitSphericalRepresentation,
)
from astropy.coordinates.builtin_frames import (
    CIRS,
    FK4,
    FK5,
    GCRS,
    HCRS,
    ICRS,
    ITRS,
    AltAz,
    CustomBarycentricEcliptic,
    Galactic,
    GalacticLSR,
    Galactocentric,
    GeocentricMeanEcliptic,
    GeocentricTrueEcliptic,
    HADec,
    HeliocentricEclipticIAU76,
    HeliocentricMeanEcliptic,
    HeliocentricTrueEcliptic,
    PrecessedGeocentric,
    Supergalactic,
)
from astropy.time import Time
from zarr.abc.store import Store

from astropy_xarray.coordinates import (
    dataset_to_skycoord,
    skycoord_to_dataset,
)
from astropy_xarray.coordinates.sky_coord import (
    _skycoord_differential_component_names,
    _skycoord_representation_component_names,
)


def generate_baselines(antenna_count) -> list[tuple[int, int]]:
    baselines = []
    for i in range(antenna_count):
        for j in range(i + 1, antenna_count):
            baselines.append((i, j))
    return baselines


def _skycoord_representation_components(skycoord: SkyCoord) -> tuple[str, ...]:
    return tuple(name for name in skycoord._representation)


@pytest.mark.parametrize(
    ("frame", "expected_base", "expected_s"),
    [
        (ICRS(), ("ra", "dec"), ("pm_ra_cosdec", "pm_dec", "radial_velocity")),
        (AltAz(), ("az", "alt"), ("pm_az_cosalt", "pm_alt", "radial_velocity")),
        (ITRS(), ("x", "y", "z"), ("v_x", "v_y", "v_z")),
    ],
)
def test_direction_get_components(frame, expected_base, expected_s):

    rep = UnitSphericalRepresentation(
        [[0.1], [2], [0.2]] * u.deg,
        [[0.5], [7], [0.7]] * u.deg,
        differentials=UnitSphericalDifferential(
            d_lon=[[0.002], [0.002], [0.002]] * u.deg / u.yr,
            d_lat=[[0.002], [0.002], [0.002]] * u.deg / u.yr,
        ),
    )
    sc = SkyCoord(frame.realize_frame(rep))
    assert _skycoord_representation_component_names(sc, True, False) == expected_base
    assert _skycoord_differential_component_names(sc, True, False) == expected_s


@pytest.fixture(name="store", scope="session")
def store_fixture() -> Generator[Store, None, None]:
    with zs.MemoryStore() as store:
        yield store


@pytest.mark.parametrize(
    ("expected_data_classes", "data"),
    [
        pytest.param(
            (Longitude, Latitude),
            UnitSphericalRepresentation(
                [[0.1], [0.2]] * u.deg,
                [[0.5], [0.7]] * u.deg,
                differentials=UnitSphericalDifferential(
                    d_lon=[[0.001], [0.002]] * u.mas / u.yr,
                    d_lat=[[0.001], [0.002]] * u.mas / u.yr,
                ),
            ),
            id="sphericalcoslat-2",
        ),
        pytest.param(
            (Longitude, Latitude),
            UnitSphericalRepresentation(
                [[0.1], [0.2]] * u.deg,
                [[0.5], [0.7]] * u.deg,
                differentials=UnitSphericalCosLatDifferential(
                    d_lon_coslat=[[0.001], [0.002]] * u.mas / u.yr,
                    d_lat=[[0.001], [0.002]] * u.mas / u.yr,
                ),
            ),
            id="spherical-2",
        ),
        pytest.param(
            (Longitude, Latitude, Distance),
            SphericalRepresentation(
                [[0.1], [0.2]] * u.deg,
                [[0.5], [0.7]] * u.deg,
                [[0.3], [0.4]] * u.pc,
                differentials=SphericalCosLatDifferential(
                    d_lon_coslat=[[0.001], [0.002]] * u.mas / u.yr,
                    d_lat=[[0.001], [0.002]] * u.mas / u.yr,
                    d_distance=[[0.0002], [0.0002]] * u.km / u.s,
                ),
            ),
            id="spherical-3",
        ),
        pytest.param(
            (u.Quantity, u.Quantity, u.Quantity),
            CartesianRepresentation(
                [[0.1], [0.2]] * u.km,
                [[0.5], [0.7]] * u.km,
                [[0.3], [0.4]] * u.km,
                differentials=CartesianDifferential(
                    d_x=[[0.1], [0.2]] * u.km / u.s,
                    d_y=[[0.5], [0.7]] * u.km / u.s,
                    d_z=[[0.3], [0.4]] * u.km / u.s,
                ),
            ),
            id="cartesian-3",
        ),
        pytest.param(
            (u.Quantity, Angle, u.Quantity),
            CylindricalRepresentation(
                [[0.1], [0.2]] * u.km,
                [[0.5], [0.7]] * u.deg,
                [[0.3], [0.4]] * u.km,
                differentials=CylindricalDifferential(
                    d_rho=[[0.1], [0.2]] * u.km / u.s,
                    d_phi=[[0.5], [0.7]] * u.deg / u.s,
                    d_z=[[0.3], [0.4]] * u.km / u.s,
                ),
            ),
            id="cylindrical-3",
        ),
    ],
)
@pytest.mark.parametrize(
    "frame",
    [
        ICRS(),
        FK5(),
        FK4(),
        GCRS(),
        GCRS(obstime=Time("2024-01-01")),
        CIRS(),
        HCRS(),
        ITRS(),
        AltAz(),
        AltAz(obstime=Time("2025-06-01"), location=EarthLocation.of_site("greenwich")),
        HADec(),
        HADec(obstime=Time("2025-06-01"), location=EarthLocation.of_site("greenwich")),
        Galactic(),
        GalacticLSR(),
        PrecessedGeocentric(),
        GeocentricMeanEcliptic(),
        GeocentricTrueEcliptic(),
        HeliocentricMeanEcliptic(),
        HeliocentricTrueEcliptic(),
        HeliocentricEclipticIAU76(),
        Galactocentric(),
        Galactocentric(
            galcen_distance=3 * u.kpc,
            galcen_coord=ICRS(
                ra=1 * u.deg,
                dec=1 * u.deg,
                pm_ra_cosdec=0.1 * u.mas / u.yr,
                pm_dec=0.1 * u.mas / u.yr,
            ),
            galcen_v_sun=CartesianDifferential(
                d_x=1 * u.km / u.s, d_y=1 * u.km / u.s, d_z=1 * u.km / u.s
            ),
            z_sun=u.Quantity(30 * u.pc),
            roll=u.Quantity(20 * u.deg),
        ),
        Supergalactic(),
        CustomBarycentricEcliptic(),
        CustomBarycentricEcliptic(obliquity=1 * u.deg),
    ],
    ids=lambda param: param.name,
)
def test_skycoord_roundtrip(
    store,
    frame: BaseCoordinateFrame,
    data: BaseRepresentation,
    expected_data_classes: tuple[type, ...],
):
    # all frames support all representation types
    representation = "cylindrical"

    expected = SkyCoord(frame.realize_frame(data))
    expected.representation_type = representation
    expected.differential_type = representation
    assert expected.frame.data.name == data.name
    assert expected.frame.data.differentials["s"].name == data.differentials["s"].name

    coords = [
        ("calibrator_name", ["a1", "a2"]),
        ("time_polynomial", [0]),
    ]
    ds = skycoord_to_dataset(expected, coords=coords)
    assert len(ds.data_vars) == (
        len(expected.frame.data.components)
        + len(expected.frame.data.differentials["s"].components)
    )

    expected_classes = zip(expected.frame.data.components, expected_data_classes)
    for expected_key, expected_class_type in expected_classes:
        assert isinstance(ds.data_vars[expected_key].data, expected_class_type)
        assert ds.data_vars[expected_key].coords.identical(xr.Coordinates(dict(coords)))

    # sanity checks
    assert ds.attrs["frame"]["representation_type"] == representation
    assert ds.attrs["frame"]["differential_type"] == representation
    assert ds.attrs["frame"]["data"]["representation_type"] == data.name
    assert (
        ds.attrs["frame"]["data"]["differential_type"] == data.differentials["s"].name
    )

    actual = dataset_to_skycoord(ds)

    # check representations
    assert actual.representation_type.name == representation
    assert actual.differential_type.name == representation
    assert actual.frame.data.name == data.name
    assert actual.frame.data.differentials["s"].name == data.differentials["s"].name

    # Check Representation
    for component_name in actual.frame.representation_component_names:
        np.testing.assert_array_equal(
            getattr(actual, component_name), getattr(expected, component_name)
        )

    # Check Differentials
    assert expected.data.differentials["s"] is not None
    np.testing.assert_array_equal(
        actual.data.differentials["s"], expected.data.differentials["s"], strict=True
    )

    # Check SkyCoords equal
    assert actual.representation_type == expected.representation_type
    assert actual.differential_type == expected.differential_type
    print(repr(actual.frame), "\n\n", repr(expected.frame))
    assert actual.is_equivalent_frame(expected)
    assert ds.coords.equals(xr.Coordinates(dict(coords)))
    np.testing.assert_array_equal(actual, expected)

    # Check intermediate dataset is serializable
    ds.astropy.dequantify().to_zarr(store, mode="w", consolidated=True)
    ds_store = xr.open_dataset(
        store, engine="zarr", consolidated=True
    ).astropy.quantify()
    xr.testing.assert_identical(ds, ds_store)
