import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.time import Time, TimeDelta

import astropy_xarray  # noqa: F401


@pytest.mark.parametrize("class_type", [Time, TimeDelta])
def test_time_dataarray_dequantify(class_type):
    da: xr.DataArray = xr.DataArray(
        class_type([0, 1, 2], format="jd", scale="ut1")
    ).astropy.dequantify()

    assert da.attrs == {
        "units": {
            "class": class_type.__name__.lower(),
            "format": "jd",
            "precision": 3,
            "scale": "ut1",
        }
    }
    np.testing.assert_array_equal(
        da.data, class_type([0, 1, 2], format="jd", scale="ut1").value, strict=True
    )


@pytest.mark.parametrize("class_type", [Time, TimeDelta])
def test_time_dataarray_quantify(class_type):
    da = xr.DataArray(
        class_type([0, 1, 2], format="jd", scale="ut1").value,
        attrs={
            "units": {
                "class": class_type.__name__.lower(),
                "format": "jd",
                "precision": 3,
                "scale": "ut1",
            }
        },
    ).astropy.quantify()

    assert isinstance(da.data, class_type)
    np.testing.assert_array_equal(
        da, class_type([0, 1, 2], format="jd", scale="ut1"), strict=True
    )


def test_time_dataset_dequantify():
    ds = xr.Dataset(
        {
            "timestamps": ("time", Time([0, 1, 2], format="jd", scale="ut1")),
            "timesteps": ("time", TimeDelta([0, 1, 2], format="jd", scale="ut1")),
            "secs": ("time", u.Quantity([0, 1, 2], "s")),
            "temp": ("time", u.Quantity([0, 1, 2], "K")),
        }
    ).astropy.dequantify()

    assert ds.timestamps.attrs == {
        "units": {"class": "time", "format": "jd", "precision": 3, "scale": "ut1"}
    }
    np.testing.assert_array_equal(
        ds.timestamps, Time([0, 1, 2], format="jd", scale="ut1").value, strict=True
    )

    assert ds.timesteps.attrs == {
        "units": {"class": "timedelta", "format": "jd", "precision": 3, "scale": "ut1"}
    }
    np.testing.assert_array_equal(
        ds.timesteps, TimeDelta([0, 1, 2], format="jd", scale="ut1").value, strict=True
    )


def test_time_dataset_quantify():
    ds = xr.Dataset(
        {
            "timestamps": xr.DataArray(
                Time([0, 1, 2], format="jd", scale="ut1").value,
                dims="time",
                attrs={
                    "units": {
                        "class": "time",
                        "format": "jd",
                        "precision": 3,
                        "scale": "ut1",
                    }
                },
            ),
            "timesteps": xr.DataArray(
                TimeDelta([0, 1, 2], format="jd", scale="ut1").value,
                dims="time",
                attrs={
                    "units": {
                        "class": "timedelta",
                        "format": "jd",
                        "precision": 3,
                        "scale": "ut1",
                    }
                },
            ),
            "secs": xr.DataArray(
                u.Quantity([0, 1, 2], "s").value, dims="time", attrs={"units": "s"}
            ),
            "temp": xr.DataArray(
                u.Quantity([0, 1, 2], "K").value, dims="time", attrs={"units": "K"}
            ),
        }
    ).astropy.quantify()

    assert isinstance(ds.timestamps.data, Time)
    np.testing.assert_array_equal(
        ds.timestamps, Time([0, 1, 2], format="jd", scale="ut1"), strict=True
    )

    assert isinstance(ds.timesteps.data, TimeDelta)
    np.testing.assert_array_equal(
        ds.timesteps, TimeDelta([0, 1, 2], format="jd", scale="ut1"), strict=True
    )
