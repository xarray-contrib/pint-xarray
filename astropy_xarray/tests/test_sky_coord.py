import sys

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
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

from astropy_xarray.coordinates import (
    dataset_to_skycoord,
    skycoord_to_dataset,
)
from astropy_xarray.coordinates.sky_coord import (
    _skycoord_differential_component_names,
    _skycoord_representation_component_names,
)

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="python3.10 or higher required for skycoords support",
)


@pytest.mark.parametrize(
    ("expected_base", "expected_s"),
    [
        (
            ("ra", "dec"),
            ("pm_ra_cosdec", "pm_dec", "radial_velocity"),
        ),
    ],
)
def test_unitspherical_repr_components(expected_base, expected_s):
    sc = SkyCoord(
        ra=[[0.1], [2], [0.2]] * u.deg,
        dec=[[0.5], [7], [0.7]] * u.deg,
        pm_ra_cosdec=[[0.002], [0.002], [0.002]] * u.deg / u.yr,
        pm_dec=[[0.002], [0.002], [0.002]] * u.deg / u.yr,
        radial_velocity=[[0.0], [0.0], [0.0]] * u.deg / u.yr,
    )
    expected_repr = str(sc)
    assert _skycoord_representation_component_names(sc) == expected_base
    assert _skycoord_differential_component_names(sc) == expected_s
    assert str(sc) == expected_repr


@pytest.mark.parametrize(
    ("expected_base", "expected_s"),
    [
        (
            ("ra", "dec", "distance"),
            ("pm_ra_cosdec", "pm_dec"),
        )
    ],
)
def test_unitspherical_diff_components(expected_base, expected_s):
    sc = SkyCoord(
        ra=[[0.1], [2], [0.2]] * u.deg,
        dec=[[0.5], [7], [0.7]] * u.deg,
        distance=[[1.0], [1.0], [1.0]] * u.dimensionless_unscaled,
        pm_ra_cosdec=[[0.002], [0.002], [0.002]] * u.deg / u.yr,
        pm_dec=[[0.002], [0.002], [0.002]] * u.deg / u.yr,
    )
    expected_str = str(sc)
    assert _skycoord_representation_component_names(sc) == expected_base
    assert _skycoord_differential_component_names(sc) == expected_s
    assert str(sc) == expected_str


@pytest.mark.parametrize(
    ("frame", "expected_base", "expected_s"),
    [
        (
            ICRS(),
            ("ra", "dec"),
            ("pm_ra_cosdec", "pm_dec"),
        ),
        (
            AltAz(),
            ("az", "alt"),
            ("pm_az_cosalt", "pm_alt"),
        ),
        (
            ITRS(),
            ("lon", "lat"),
            ("pm_lon_coslat", "pm_lat"),
        ),
    ],
)
def test_spherical_direction_components_realize_frame(frame, expected_base, expected_s):
    rep = UnitSphericalRepresentation(
        [[0.1], [2], [0.2]] * u.deg,
        [[0.5], [7], [0.7]] * u.deg,
        differentials=UnitSphericalCosLatDifferential(
            [[0.002], [0.002], [0.002]] * u.deg / u.yr,
            [[0.002], [0.002], [0.002]] * u.deg / u.yr,
        ),
    )
    sc = SkyCoord(frame.realize_frame(rep))
    assert _skycoord_representation_component_names(sc) == expected_base
    assert _skycoord_differential_component_names(sc) == expected_s


@pytest.mark.parametrize(
    ("expected_base", "expected_s"),
    [
        (("x", "y", "z"), ("v_x", "v_y", "v_z")),
    ],
)
def test_cartesian_direction_components(expected_base, expected_s):
    sc = SkyCoord(
        x=[[0.1], [2], [0.2]] * u.dimensionless_unscaled,
        y=[[0.5], [7], [0.7]] * u.dimensionless_unscaled,
        z=[[0.5], [7], [0.7]] * u.dimensionless_unscaled,
        v_x=[[0.002], [0.002], [0.002]] * u.dimensionless_unscaled / u.yr,
        v_y=[[0.002], [0.002], [0.002]] * u.dimensionless_unscaled / u.yr,
        v_z=[[0.002], [0.002], [0.002]] * u.dimensionless_unscaled / u.yr,
        frame="itrs",
    )
    assert _skycoord_representation_component_names(sc) == expected_base
    assert _skycoord_differential_component_names(sc) == expected_s


@pytest.mark.parametrize(
    "use_default_representation",
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
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
    frame: BaseCoordinateFrame,
    data: BaseRepresentation,
    expected_data_classes: tuple[type, ...],
    use_default_representation: bool,
):
    # all frames support all representation types
    representation_type = type(data)
    differential_type = type(data.differentials["s"])

    if use_default_representation:
        expected = SkyCoord(
            frame.realize_frame(
                data.represent_as(
                    frame.default_representation, frame.default_differential
                )
            )
        )
        expected_frame_data_rep_name = frame.default_representation.name
        expected_frame_data_diff_name = frame.default_differential.name
    else:
        expected = SkyCoord(frame.realize_frame(data))
        expected_frame_data_rep_name = representation_type.name
        expected_frame_data_diff_name = differential_type.name

    assert expected.frame.data.name == expected_frame_data_rep_name
    assert expected.frame.data.differentials["s"].name == expected_frame_data_diff_name

    coords = [
        ("calibrator_name", ["a1", "a2"]),
        ("time_polynomial", [0]),
    ]
    ds = skycoord_to_dataset(expected, coords=coords)

    # data_var keys
    expected_keys_frame = expected.frame.copy()
    expected_keys_frame.representation_type = type(expected.frame.data)
    expected_keys_frame.differential_type = type(
        expected.frame.data.differentials.get("s")
    )
    expected_keys = set(expected_keys_frame.get_representation_component_names()) | set(
        expected_keys_frame.get_representation_component_names("s")
    )
    assert set(ds.data_vars) == expected_keys

    # Convert back
    actual = dataset_to_skycoord(ds)

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
    assert actual.is_equivalent_frame(expected)
    assert ds.coords.equals(xr.Coordinates(dict(coords)))
    np.testing.assert_array_equal(actual, expected)
