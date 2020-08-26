import pint
import pytest

# only need to register _repr_inline_
import pint_xarray  # noqa: F401

unit_registry = pint.UnitRegistry(force_ndarray_like=True)


@pytest.mark.parametrize(
    ("length", "expected"),
    (
        (40, "[N] 7.1 5.4 9.8 21.4 15.3"),
        (20, "[N] 7.1 5.4 ... 15.3"),
        (10, "[N] 7.1..."),
        (7, "[N] ..."),
        (3, "[N] ..."),
    ),
)
def test_inline_repr(length, expected):
    quantity = unit_registry.Quantity([7.1, 5.4, 9.8, 21.4, 15.3], "N")

    assert quantity._repr_inline_(length) == expected
