from . import conversion


def assert_units_equal(a, b):
    __tracebackhide__ = True

    assert conversion.extract_units(a) == conversion.extract_units(b)
    assert conversion.extract_unit_attributes(a) == conversion.extract_unit_attributes(
        b
    )
