from importlib.metadata import version

import astropy.units as u
import numpy.testing
import pytest
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.coordinates.builtin_frames import ICRS
from packaging.version import Version

from astropy_xarray.coordinates import (
    skycoord_to_dataset,
)
from astropy_xarray.coordinates.sky_coord import (
    _skycoord_component_names,
)


def test_frame_conversion_aliasing():
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


@pytest.mark.xfail(
    condition=Version(version("astropy")) <= Version("7.0.0"),
    reason="astropy 6.0.0 skycoords not supported",
)
def test_frame_conversion_components():
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

    assert list(_skycoord_component_names(sc)) == [
        "x",
        "y",
        "z",
        "v_x",
        "v_y",
        "v_z",
    ]

    skycoord_to_dataset(sc, None)

    sc2 = SkyCoord(
        ICRS().realize_frame(
            sc.represent_as(
                sc.frame.default_representation, sc.frame.default_differential
            )
        )
    )

    assert list(_skycoord_component_names(sc2)) == [
        "ra",
        "dec",
        "distance",
        "pm_ra_cosdec",
        "pm_dec",
        "radial_velocity",
    ]
