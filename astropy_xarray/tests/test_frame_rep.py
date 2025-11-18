import astropy.units as u
import numpy as np
import numpy.testing
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
    DatasetRepresentation,
    _skycoord_component_names,
    _skycoord_differential_component_names,
    _skycoord_representation_component_names,
)


def test_regression():
    sc = SkyCoord(
        ICRS().realize_frame(
            CartesianRepresentation(
                [0.1, 0.2] * u.km,
                [0.5, 0.7] * u.km,
                [0.3, 0.4] * u.km,
                differentials=CartesianDifferential(
                    d_x=[0.1, 0.2] * u.km / u.s,
                    d_y=[0.5, 0.7] * u.km / u.s,
                    d_z=[0.3, 0.4] * u.km / u.s,
                ),
            )
        )
    )
    sc2 = SkyCoord(
        ICRS().realize_frame(
            sc.represent_as(
                sc.frame.default_representation, sc.frame.default_differential
            )
        )
    )
    numpy.testing.assert_array_equal(sc.pm_ra_cosdec, sc2.pm_ra_cosdec)


def test_convert_regression():
    sc = SkyCoord(
        ICRS().realize_frame(
            CartesianRepresentation(
                [[0.1], [0.2]] * u.km,
                [[0.5], [0.7]] * u.km,
                [[0.3], [0.4]] * u.km,
                differentials=CartesianDifferential(
                    d_x=[[0.1], [0.2]] * u.km / u.s,
                    d_y=[[0.5], [0.7]] * u.km / u.s,
                    d_z=[[0.3], [0.4]] * u.km / u.s,
                ),
            )
        )
    )

    skycoord_to_dataset(sc, None, representation=DatasetRepresentation.FRAME)
    skycoord_to_dataset(sc, None, representation=DatasetRepresentation.FRAME_DEFAULT)

    sc2 = SkyCoord(
        ICRS().realize_frame(
            sc.represent_as(
                sc.frame.default_representation, sc.frame.default_differential
            )
        )
    )

    assert list(_skycoord_component_names(sc2, True, True)) == list(
        _skycoord_component_names(sc2, True, False)
    )

    print(skycoord_to_dataset(sc2, None, representation=DatasetRepresentation.FRAME))
    print(
        skycoord_to_dataset(
            sc2, None, representation=DatasetRepresentation.FRAME_DEFAULT
        )
    )
