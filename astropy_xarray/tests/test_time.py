import numpy as np
import pytest
from astropy.time import Time

import astropy_xarray  # noqa: F401


@pytest.fixture(name="time")
def time_fixture(value, format, scale):
    return Time(value, format=format, scale=scale)


@pytest.mark.parametrize("scale", ["utc", "tai", "ut1"])
@pytest.mark.parametrize(
    "value,format,dtype",
    [
        (np.array([1.0, 2.0, 3.0]), "unix", "float64"),
        (np.array([1.0, 2.0, 3.0]), "unix_tai", "float64"),
        (np.array([1.0, 2.0, 3.0]), "mjd", "float64"),
        (
            np.array(
                [
                    np.datetime64("1970-01-01T00:00:09.000082030"),
                    np.datetime64("1970-01-01T00:00:10.000082060"),
                    np.datetime64("1970-01-01T00:00:11.000082090"),
                ]
            ),
            "datetime64",
            "<M8[ns]",
        ),
        (
            np.array(
                [
                    "1970-01-01T00:00:09.000082030",
                    "1970-01-01T00:00:10.000082060",
                    "1970-01-01T00:00:11.000082090",
                ]
            ),
            "isot",
            "<U23",
        ),
        (
            np.array(
                [
                    "1970-01-01 00:00:09.000082030",
                    "1970-01-01 00:00:10.000082060",
                    "1970-01-01 00:00:11.000082090",
                ]
            ),
            "iso",
            "<U23",
        ),
    ],
)
class TestTime:
    def test_time_dtype(self, time, dtype):
        assert time.dtype == dtype

    def test_time_nbytes(self, time, dtype):
        assert time.nbytes == 3 * np.dtype(dtype).itemsize

    def test_array(self, time, dtype):
        np.testing.assert_array_equal(time.value, time.__array__())

    def test_array_ufunc(self, time, dtype):
        assert 6 == time.__array_ufunc__(np.add, "reduce", [1, 2, 3])


@pytest.mark.parametrize(
    "value,format,scale,expected",
    [
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            "unix",
            "utc",
            "[unix utc] 1970-01-01T00:00:01.000000000 ... 1970-01-01T0...",
        ),
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            "unix_tai",
            "utc",
            "[unix_tai utc] 1969-12-31T23:59:52.999918210 ... 1969-12-...",
        ),
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            "mjd",
            "utc",
            "[mjd utc] 1858-11-18T00:00:00.000000000 ... 1858-11-21T00...",
        ),
        (
            np.array(
                [
                    np.datetime64("1970-01-01T00:00:09.000082030"),
                    np.datetime64("1970-01-01T00:00:10.000082060"),
                    np.datetime64("1970-01-01T00:00:11.000082090"),
                ]
            ),
            "datetime64",
            "utc",
            "[datetime64 utc] 1970-01-01T00:00:09.000082030 ... 1970-0...",
        ),
        (
            np.array(
                [
                    "1970-01-01 00:00:09.000082030",
                    "1970-01-01 00:00:10.000082060",
                    "1970-01-01 00:00:11.000082090",
                ]
            ),
            "iso",
            "utc",
            "[iso utc] 1970-01-01T00:00:09.000082030 ... 1970-01-01T00...",
        ),
    ],
)
def test_time_repr_inline(time, expected):
    assert time._repr_inline_(60) == expected
