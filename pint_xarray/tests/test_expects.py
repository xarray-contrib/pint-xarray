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
            ((ureg.Quantity(1, "m"), 2), ("mm", None, None), 500),
            ((ureg.Quantity(1, "m"), ureg.Quantity(0.5, "s")), ("mm", "ms", None), 2),
            (
                (xr.DataArray(4).pint.quantify("km"), 2),
                ("m", None, None),
                xr.DataArray(2000),
            ),
            (
                (
                    xr.DataArray([4, 2, 0]).pint.quantify("cm"),
                    xr.DataArray([4, 2, 1]).pint.quantify("mg"),
                ),
                ("m", "g", None),
                xr.DataArray([10, 10, 0]),
            ),
            (
                (ureg.Quantity(16, "m"), 2, ureg.Quantity(4, "s")),
                ("mm", None, "ms"),
                2,
            ),
        ),
    )
    def test_args(self, values, units, expected):
        @pint_xarray.expects(*units)
        def func(a, b, c=1):
            return a / (b * c)

        actual = func(*values)

        if isinstance(actual, xr.DataArray):
            xr.testing.assert_identical(actual, expected)
        elif isinstance(actual, pint.Quantity):
            pint.testing.assert_equal(actual, expected)
        else:
            assert actual == expected

    @pytest.mark.parametrize(
        ["value", "units", "errors", "message", "multiple"],
        (
            (
                ureg.Quantity(1, "m"),
                (None, None),
                ValueError,
                "quantity where none was expected",
                True,
            ),
            (1, ("m", None), ValueError, "Attempting to convert non-quantity", True),
            (
                1,
                (None,),
                ValueError,
                "Missing units for the following parameters: 'b'",
                False,
            ),
        ),
    )
    def test_args_error(self, value, units, errors, message, multiple):
        if multiple:
            root_error = ExceptionGroup
            root_message = "Errors while converting parameters"
        else:
            root_error = errors
            root_message = message

        with pytest.raises(root_error, match=root_message) as excinfo:

            @pint_xarray.expects(*units)
            def func(a, b=1):
                return a * b

            func(value)

        if not multiple:
            return

        group = excinfo.value
        assert len(group.exceptions) == 1, f"Found {len(group.exceptions)} exceptions"
        exc = group.exceptions[0]
        if not re.search(message, str(exc)):
            raise AssertionError(f"exception {exc!r} did not match pattern {message!r}")
