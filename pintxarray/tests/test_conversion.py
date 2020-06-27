import numpy as np
import pint
import pytest

from pintxarray import conversion

from .utils import assert_array_equal, assert_array_units_equal

unit_registry = pint.UnitRegistry()

pytestmark = pytest.mark.filterwarnings("error::pint.UnitStrippedWarning")


class TestArrayFunctions:
    @pytest.mark.parametrize(
        "registry",
        (
            pytest.param(None, id="without registry"),
            pytest.param(unit_registry, id="with registry"),
        ),
    )
    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="not a unit"),
            pytest.param(None, id="no unit"),
            pytest.param("m", id="string"),
            pytest.param(unit_registry.m, id="unit object"),
        ),
    )
    @pytest.mark.parametrize(
        "data",
        (
            pytest.param(np.array([0, 1]), id="array_like"),
            pytest.param(np.array([1, 2]) * unit_registry.m, id="quantity"),
        ),
    )
    def test_array_attach_units(self, data, unit, registry):
        if unit == 1:
            match = "cannot use .+ as a unit"
        elif isinstance(data, pint.Quantity) and unit is not None:
            match = "already has units"
        elif isinstance(unit, str) and registry is None:
            match = "a string as unit"
        else:
            match = None

        if match is not None:
            with pytest.raises(ValueError, match=match):
                conversion.array_attach_units(data, unit, registry=registry)

            return

        expected = unit_registry.Quantity(data, "m") if unit is not None else data
        actual = conversion.array_attach_units(data, unit, registry=registry)

        assert_array_units_equal(expected, actual)
        assert_array_equal(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="not a unit"),
            pytest.param(None, id="no unit"),
            pytest.param("mm", id="string"),
            pytest.param(unit_registry.mm, id="unit object"),
        ),
    )
    @pytest.mark.parametrize(
        "data",
        (
            pytest.param(np.array([0, 1]), id="array_like"),
            pytest.param(np.array([1, 2]) * unit_registry.m, id="quantity"),
        ),
    )
    def test_array_convert_units(self, data, unit):
        if unit == 1:
            error = ValueError
            match = "cannot use .+ as a unit"
        elif not isinstance(data, pint.Quantity) and isinstance(unit, str):
            error = ValueError
            match = "cannot convert a non-quantity using .+ as unit"
        elif not isinstance(data, pint.Quantity) and unit is not None:
            error = pint.DimensionalityError
            match = None
        else:
            error = None
            match = None

        if error is not None:
            with pytest.raises(error, match=match):
                conversion.array_convert_units(data, unit)

            return

        expected = (
            unit_registry.Quantity(np.array([1000, 2000]), "mm")
            if unit is not None
            else data
        )
        actual = conversion.array_convert_units(data, unit)

        assert_array_equal(expected, actual)

    @pytest.mark.parametrize(
        "data",
        (
            pytest.param(np.array([0, 1]), id="array_like"),
            pytest.param(np.array([1, 2]) * unit_registry.m, id="quantity"),
        ),
    )
    def test_array_extract_units(self, data):
        expected = unit_registry.m if isinstance(data, pint.Quantity) else None
        actual = conversion.array_extract_units(data)

        assert expected == actual

    @pytest.mark.parametrize(
        "data",
        (
            pytest.param(np.array([1, 2]), id="array_like"),
            pytest.param(np.array([1, 2]) * unit_registry.m, id="quantity"),
        ),
    )
    def test_array_strip_units(self, data):
        expected = np.array([1, 2])
        actual = conversion.array_strip_units(data)

        assert_array_equal(expected, actual)
