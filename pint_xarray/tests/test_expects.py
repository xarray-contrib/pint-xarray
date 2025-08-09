import re

import pint
import pytest
import xarray as xr

import pint_xarray

ureg = pint_xarray.unit_registry


class TestExpects:
    @pytest.mark.parametrize(
        ["values", "units", "expected"],
        (
            ((ureg.Quantity(1, "m"), 2), ("mm", None), 500),
            ((ureg.Quantity(1, "m"), ureg.Quantity(0.5, "s")), ("mm", "ms"), 2),
            ((xr.DataArray(4).pint.quantify("km"), 2), ("m", None), xr.DataArray(2000)),
            (
                (
                    xr.DataArray([4, 2, 0]).pint.quantify("cm"),
                    xr.DataArray([4, 2, 1]).pint.quantify("mg"),
                ),
                ("m", "g"),
                xr.DataArray([10, 10, 0]),
            ),
        ),
    )
    def test_args(self, values, units, expected):
        @pint_xarray.expects(*units)
        def func(a, b):
            return a / b

        actual = func(*values)

        if isinstance(actual, xr.DataArray):
            xr.testing.assert_identical(actual, expected)
        elif isinstance(actual, pint.Quantity):
            pint.testing.assert_equal(actual, expected)
        else:
            assert actual == expected

    @pytest.mark.parametrize(
        ["value", "units", "errors", "message"],
        (
            (
                ureg.Quantity(1, "m"),
                None,
                ValueError,
                "quantity where none was expected",
            ),
            (1, "m", ValueError, "Attempting to convert non-quantity"),
        ),
    )
    def test_args_error(self, value, units, errors, message):
        with pytest.raises(
            ExceptionGroup, match="Errors while converting parameters"
        ) as excinfo:

            @pint_xarray.expects(units)
            def func(a):
                return a

            func(value)
        group = excinfo.value
        assert len(group.exceptions) == 1, f"Found {len(group.exceptions)} exceptions"
        exc = group.exceptions[0]
        if not re.search(message, str(exc)):
            raise AssertionError(f"exception {exc!r} did not match pattern {message!r}")
