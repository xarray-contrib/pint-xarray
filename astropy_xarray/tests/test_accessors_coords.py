import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import Distance, Latitude, Longitude

import astropy_xarray  # noqa: F401


@pytest.mark.parametrize("unit", ["rad", "deg", "arcmin", "arcsec"])
@pytest.mark.parametrize("class_type", [Latitude, Longitude])
def test_angle_dataarray_dequantify(class_type, unit):
    da: xr.DataArray = xr.DataArray(
        class_type([0, 0.5, 1], unit=unit)
    ).astropy.dequantify()

    assert da.attrs == {"units": {"class": class_type.__name__.lower(), "unit": unit}}
    np.testing.assert_array_equal(
        da.data, class_type([0, 0.5, 1], unit=unit).value, strict=True
    )


@pytest.mark.parametrize("unit", ["m", "km", "Mpc"])
@pytest.mark.parametrize("class_type", [Distance])
def test_distance_dataarray_dequantify(class_type, unit):
    # TODO: support allow_negative=True
    da: xr.DataArray = xr.DataArray(
        class_type([0, 1, 2], unit=unit)
    ).astropy.dequantify()

    assert da.attrs == {"units": {"class": class_type.__name__.lower(), "unit": unit}}
    np.testing.assert_array_equal(
        da.data, class_type([0, 1, 2], unit=unit).value, strict=True
    )


@pytest.mark.parametrize("unit", ["rad", "deg", "arcmin", "arcsec"])
@pytest.mark.parametrize("class_type", [Latitude, Longitude])
def test_coord_dataarray_quantify(class_type, unit):
    da = xr.DataArray(
        class_type([0, 0.5, 1], unit=unit).value,
        attrs={"units": {"class": class_type.__name__.lower(), "unit": unit}},
    ).astropy.quantify()

    assert isinstance(da.data, class_type)
    np.testing.assert_array_equal(
        # da,
        da.data,
        class_type([0, 0.5, 1], unit=unit),
        strict=True,
    )


def test_coord_dataset_dequantify():
    ds = xr.Dataset(
        {
            "lat": ("time", Latitude([0, 1, 2] * u.deg)),
            "long": ("time", Longitude([0, 1, 2] * u.deg)),
            "dist": ("time", Distance([0, 1, 2] * u.Mpc)),
            "temp": ("time", u.Quantity([0, 1, 2], u.C)),
        }
    ).astropy.dequantify()

    assert ds.lat.attrs == {"units": {"class": "latitude", "unit": "deg"}}
    np.testing.assert_array_equal(
        ds.lat, Latitude([0, 1, 2], unit="deg").value, strict=True
    )

    assert ds.long.attrs == {"units": {"class": "longitude", "unit": "deg"}}
    np.testing.assert_array_equal(
        ds.lat, Longitude([0, 1, 2], unit="deg").value, strict=True
    )

    assert ds.dist.attrs == {"units": {"class": "distance", "unit": "Mpc"}}
    np.testing.assert_array_equal(
        ds.lat, Longitude([0, 1, 2], unit="deg").value, strict=True
    )


def test_coord_dataset_quantify():
    ds = xr.Dataset(
        {
            "lat": xr.DataArray(
                Latitude([0, 1, 2] * u.deg).value,
                dims="time",
                attrs={"units": {"class": "latitude", "unit": "deg"}},
            ),
            "long": xr.DataArray(
                Longitude([0, 1, 2] * u.deg).value,
                dims="time",
                attrs={"units": {"class": "longitude", "unit": "deg"}},
            ),
            "dist": xr.DataArray(
                Distance([0, 1, 2], "Mpc").value,
                dims="time",
                attrs={"units": {"class": "distance", "unit": "Mpc"}},
            ),
            "temp": xr.DataArray(
                u.Quantity([0, 1, 2], "K").value, dims="time", attrs={"units": "K"}
            ),
        }
    ).astropy.quantify()

    assert isinstance(ds.lat.data, Latitude)
    np.testing.assert_array_equal(
        ds.lat.data, Latitude([0, 1, 2], unit="deg"), strict=True
    )

    assert isinstance(ds.long.data, Longitude)
    np.testing.assert_array_equal(
        ds.long.data, Longitude([0, 1, 2], unit="deg"), strict=True
    )

    assert isinstance(ds.dist.data, Distance)
    np.testing.assert_array_equal(
        ds.dist.data, Distance([0, 1, 2], unit="Mpc"), strict=True
    )
