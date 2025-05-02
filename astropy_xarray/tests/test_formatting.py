import astropy.units as u
import pytest

# only need to register _repr_inline_
import astropy_xarray  # noqa: F401


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
    quantity = u.Quantity([7.1, 5.4, 9.8, 21.4, 15.3], "N")

    assert quantity._repr_inline_(length) == expected
