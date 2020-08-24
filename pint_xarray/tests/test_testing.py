import pint
import pytest
import xarray as xr

from pint_xarray import testing

unit_registry = pint.UnitRegistry(force_ndarray_like=True)


@pytest.mark.parametrize(
    ("a", "b", "error"),
    (
        pytest.param(
            xr.DataArray(attrs={"units": "K"}),
            xr.DataArray(attrs={"units": "K"}),
            None,
            id="equal attrs",
        ),
        pytest.param(
            xr.DataArray(attrs={"units": "m"}),
            xr.DataArray(attrs={"units": "K"}),
            AssertionError,
            id="different attrs",
        ),
        pytest.param(
            xr.DataArray([10, 20] * unit_registry.K),
            xr.DataArray([50, 80] * unit_registry.K),
            None,
            id="equal units",
        ),
        pytest.param(
            xr.DataArray([10, 20] * unit_registry.K),
            xr.DataArray([50, 80] * unit_registry.dimensionless),
            AssertionError,
            id="different units",
        ),
        pytest.param(
            xr.Dataset({"a": ("x", [0, 10], {"units": "K"})}),
            xr.Dataset({"a": ("x", [20, 40], {"units": "K"})}),
            None,
            id="matching variables",
        ),
        pytest.param(
            xr.Dataset({"a": ("x", [0, 10], {"units": "K"})}),
            xr.Dataset({"b": ("x", [20, 40], {"units": "K"})}),
            AssertionError,
            id="mismatching variables",
        ),
    ),
)
def test_assert_units_equal(a, b, error):
    if error is not None:
        with pytest.raises(error):
            testing.assert_units_equal(a, b)

        return

    testing.assert_units_equal(a, b)
