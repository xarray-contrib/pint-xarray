from . import conversion, formatting


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

    units_a = conversion.extract_units(a)
    units_b = conversion.extract_units(b)
    assert units_a == units_b, formatting._diff_mapping_repr(
        units_a, units_b, "Units", formatting.summarize_attr
    )

    unit_attrs_a = conversion.extract_unit_attributes(a)
    unit_attrs_b = conversion.extract_unit_attributes(b)
    assert unit_attrs_a == unit_attrs_b, formatting._diff_mapping_repr(
        unit_attrs_a, unit_attrs_b, "Unit attrs", formatting.summarize_attr
    )
