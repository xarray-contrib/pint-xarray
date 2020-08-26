from . import conversion


def assert_units_equal(a, b):
    """assert that the units of two xarray objects are equal

    Raises an :py:exc:`AssertionError` if the units of both objects are not
    equal. ``units`` attributes and attached unit objects are compared
    separately.

    Parameters
    ----------
    a, b : DataArray or Dataset
        The objects to compare
    """

    __tracebackhide__ = True

    assert conversion.extract_units(a) == conversion.extract_units(b)
    assert conversion.extract_unit_attributes(a) == conversion.extract_unit_attributes(
        b
    )
