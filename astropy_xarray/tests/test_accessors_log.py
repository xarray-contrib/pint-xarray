import astropy.units as u
import numpy as np
import pytest
import xarray as xr

import astropy_xarray  # noqa: F401

# referenceless
# y(x) dB = 10log_10(x) -> x = 10^(y/10)
# y(x) mag = -2.5log_10(x) -> x = 10^(-y/2.5)
# y(x) dex = log_10(x) -> x = 10^(y)

# with reference
# y dB(2 V) = 10log_10(x/2 V) -> x = 2 V * 10^(y/10)
# y mag(2 V) = -2.5log_10(x/2 V) -> x = 2 V * 10^(-y/2.5)
# y dex(2 V) = log_10(x/2 V) -> x = 2 V * 10^(y)


@pytest.mark.parametrize("unit", ["dex", "dB", "mag", "dex(cd)", "dB(cd)", "mag(cd)"])
def test_log_dataarray_dequantify(unit):
    da: xr.DataArray = xr.DataArray(
        u.LogQuantity([0, 0.5, 1], unit=unit)
    ).astropy.dequantify()

    assert da.attrs == {"units": unit}
    np.testing.assert_array_equal(
        da.data, u.LogQuantity([0, 0.5, 1], unit=unit).value, strict=True
    )


@pytest.mark.parametrize("unit", ["dex(10 cd)", "dB(10 cd)", "mag(10 cd)"])
def test_log_dataarray_quantify(unit):
    da: xr.DataArray = xr.DataArray(
        u.LogQuantity([0, 0.5, 1], unit=unit).value, attrs={"units": unit}
    ).astropy.quantify()

    assert isinstance(da.data, u.LogQuantity)
    np.testing.assert_array_equal(
        da.data, u.LogQuantity([0, 0.5, 1], unit=unit), strict=True
    )


def test_log_dataset_dequantify():
    ds: xr.DataArray = xr.Dataset(
        {
            "volume": ("time", [0, 1, 2] * u.dB("20 uPa")),
            "brightness": ("time", [0, 1, 2] * u.mag("cd")),
            "metallicity": ("time", [0, 1, 2] * u.dex),
            "temp": ("time", u.Quantity([0, 1, 2], u.C)),
        }
    ).astropy.dequantify()

    assert ds.volume.attrs == {"units": "dB(20 uPa)"}
    np.testing.assert_array_equal(ds.volume, np.array([0.0, 1.0, 2.0]), strict=True)

    assert ds.brightness.attrs == {"units": "mag(cd)"}
    np.testing.assert_array_equal(ds.brightness, np.array([0.0, 1.0, 2.0]), strict=True)

    assert ds.metallicity.attrs == {"units": "dex"}
    np.testing.assert_array_equal(
        ds.metallicity, np.array([0.0, 1.0, 2.0]), strict=True
    )


def test_coord_dataset_quantify():
    ds: xr.DataArray = xr.Dataset(
        {
            "volume": xr.DataArray(
                np.array([0.0, 1.0, 2.0]), dims="time", attrs={"units": "dB(20 uPa)"}
            ),
            "brightness": xr.DataArray(
                np.array([0.0, 1.0, 2.0]), dims="time", attrs={"units": "mag(cd)"}
            ),
            "metallicity": xr.DataArray(
                np.array([0.0, 1.0, 2.0]), dims="time", attrs={"units": "dex"}
            ),
            "temp": xr.DataArray(
                np.array([0.0, 1.0, 2.0]), dims="time", attrs={"units": "K"}
            ),
        }
    ).astropy.quantify()

    assert isinstance(ds.volume.data, u.LogQuantity)
    np.testing.assert_array_equal(
        ds.volume.data, [0, 1, 2] * u.dB("20 uPa"), strict=True
    )
    np.testing.assert_array_equal(
        ds.volume.data.to("uPa"),
        [20.0, 25.178508235883346, 31.697863849222273] * u.uPa,
        strict=True,
    )

    assert isinstance(ds.brightness.data, u.LogQuantity)
    np.testing.assert_array_equal(
        ds.brightness.data, [0, 1, 2] * u.mag("cd"), strict=True
    )
    np.testing.assert_array_equal(
        ds.brightness.data.to("cd"),
        [1.0, 0.3981071705534972, 0.15848931924611134] * u.cd,
        strict=True,
    )

    assert isinstance(ds.metallicity.data, u.Quantity)
    np.testing.assert_array_equal(ds.metallicity.data, [0, 1, 2] * u.dex, strict=True)
