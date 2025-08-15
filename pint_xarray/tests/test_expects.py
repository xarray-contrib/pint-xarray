import re

import pint
import pytest
import xarray as xr

import pint_xarray
from pint_xarray.testing import assert_units_equal

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
        ["value", "units", "error", "message", "multiple"],
        (
            (
                ureg.Quantity(1, "m"),
                (None, None),
                TypeError,
                "Passed in a quantity where none was expected",
                True,
            ),
            (1, ("m", None), TypeError, "Attempting to convert non-quantity", True),
            (
                1,
                (None,),
                ValueError,
                "Missing units for the following parameters: 'b'",
                False,
            ),
        ),
    )
    def test_args_error(self, value, units, error, message, multiple):
        if multiple:
            root_error = ExceptionGroup
            root_message = "Errors while converting parameters"
        else:
            root_error = error
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
        assert isinstance(
            exc, error
        ), f"Unexpected exception type: {type(exc)}, expected {error}"
        if not re.search(message, str(exc)):
            raise AssertionError(f"exception {exc!r} did not match pattern {message!r}")

    @pytest.mark.parametrize(
        ["values", "units", "expected"],
        (
            (
                {"a": ureg.Quantity(1, "m"), "b": 2},
                {"a": "mm", "b": None, "c": None},
                1000,
            ),
            (
                {"a": 2, "b": ureg.Quantity(100, "cm")},
                {"a": None, "b": "m", "c": None},
                4,
            ),
            (
                {"a": ureg.Quantity(1, "m"), "b": ureg.Quantity(0.5, "s")},
                {"a": "mm", "b": "ms", "c": None},
                4,
            ),
            (
                {"a": xr.DataArray(4).pint.quantify("km"), "b": 2},
                {"a": "m", "b": None, "c": None},
                xr.DataArray(4000),
            ),
            (
                {
                    "a": xr.DataArray([4, 2, 0]).pint.quantify("cm"),
                    "b": xr.DataArray([4, 2, 1]).pint.quantify("mg"),
                },
                {"a": "m", "b": "g", "c": None},
                xr.DataArray([20, 20, 0]),
            ),
        ),
    )
    def test_kwargs(self, values, units, expected):
        @pint_xarray.expects(**units)
        def func(a, b, c=2):
            return a / b * c

        actual = func(**values)

        if isinstance(actual, xr.DataArray):
            xr.testing.assert_identical(actual, expected)
        elif isinstance(actual, pint.Quantity):
            pint.testing.assert_equal(actual, expected)
        else:
            assert actual == expected

    @pytest.mark.parametrize(
        ["values", "return_value_units", "expected"],
        (
            ((1, 2), ("m", "s"), (ureg.Quantity(1, "m"), ureg.Quantity(2, "s"))),
            ((1, 2), "m / s", ureg.Quantity(0.5, "m / s")),
            ((1, 2), None, 0.5),
            (
                (xr.DataArray(2), 2),
                ("m", "s"),
                (xr.DataArray(2).pint.quantify("m"), ureg.Quantity(2, "s")),
            ),
            (
                (xr.DataArray(2), 2),
                "kg / m^2",
                xr.DataArray(1).pint.quantify("kg / m^2"),
            ),
        ),
    )
    def test_return_value(self, values, return_value_units, expected):
        multiple = isinstance(return_value_units, tuple)

        @pint_xarray.expects(a=None, b=None, return_value=return_value_units)
        def func(a, b):
            if multiple:
                return a, b
            else:
                return a / b

        actual = func(*values)
        if isinstance(actual, xr.DataArray):
            xr.testing.assert_identical(actual, expected)
        elif isinstance(actual, pint.Quantity):
            pint.testing.assert_equal(actual, expected)
        else:
            assert actual == expected

    def test_return_value_none(self):
        @pint_xarray.expects(None)
        def func(a):
            return None

        actual = func(1)
        assert actual is None

    def test_return_value_none_error(self):
        @pint_xarray.expects(return_value="Hz")
        def func():
            return None

        with pytest.raises(
            ValueError,
            match="mismatched number of return values: expected 1 but got 0.",
        ):
            func()

    @pytest.mark.parametrize(
        [
            "return_value_units",
            "multiple_units",
            "error",
            "multiple_errors",
            "message",
        ],
        (
            (
                ("m", "s"),
                False,
                ValueError,
                False,
                "mismatched number of return values",
            ),
            (
                "m",
                True,
                ValueError,
                False,
                "mismatched number of return values: expected 1 but got 2",
            ),
            (
                ("m",),
                True,
                ValueError,
                False,
                "mismatched number of return values: expected 1 but got 2",
            ),
            (1, False, TypeError, True, "units must be of type"),
        ),
    )
    def test_return_value_error(
        self, return_value_units, multiple_units, error, multiple_errors, message
    ):
        if multiple_errors:
            root_error = ExceptionGroup
            root_message = "Errors while converting return values"
        else:
            root_error = error
            root_message = message

        with pytest.raises(root_error, match=root_message) as excinfo:

            @pint_xarray.expects(a=None, b=None, return_value=return_value_units)
            def func(a, b):
                if multiple_units:
                    return a, b
                else:
                    return a / b

            func(1, 2)

        if not multiple_errors:
            return

        group = excinfo.value
        assert len(group.exceptions) == 1, f"Found {len(group.exceptions)} exceptions"
        exc = group.exceptions[0]
        assert isinstance(
            exc, error
        ), f"Unexpected exception type: {type(exc)}, expected {error}"
        if not re.search(message, str(exc)):
            raise AssertionError(f"exception {exc!r} did not match pattern {message!r}")

    def test_datasets(self):
        @pint_xarray.expects({"m": "kg", "a": "m / s^2"}, return_value={"f": "newtons"})
        def second_law(ds):
            f_da = ds["m"] * ds["a"]
            return f_da.to_dataset(name="f")

        ds = xr.Dataset({"m": 0.1, "a": 10}).pint.quantify(
            {"m": "tons", "a": "feet / second^2"}
        )

        expected = xr.Dataset({"f": ds["m"] * ds["a"]}).pint.to("newtons")

        actual = second_law(ds)

        assert_units_equal(actual, expected)
        xr.testing.assert_allclose(actual, expected)
