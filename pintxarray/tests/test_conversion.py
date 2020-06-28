import numpy as np
import pint
import pytest
from xarray import DataArray, Dataset, Variable

from pintxarray import conversion

from .utils import assert_array_equal, assert_array_units_equal, assert_equal

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


class TestXarrayFunctions:
    @pytest.mark.parametrize(
        "coords",
        (
            pytest.param({}, id="no coords"),
            pytest.param(
                {"u": ("x", [10, 3, 4] * unit_registry.m)}, id="non-dimension coord"
            ),
            pytest.param(
                {"x": [0, 1, 2]},
                id="dimension coordinate",
                marks=pytest.mark.xfail(
                    reason="converting indexes not implemented yet"
                ),
            ),
        ),
    )
    @pytest.mark.parametrize("typename", ("Variable", "DataArray", "Dataset"))
    def test_convert_units(self, typename, coords):
        if typename == "Variable":
            if coords:
                pytest.skip("Variable doesn't store coordinates")

            data = np.linspace(0, 1, 3) * unit_registry.m
            obj = Variable(dims="x", data=data)
            units = {None: unit_registry.mm}
        elif typename == "DataArray":
            obj = DataArray(
                dims="x", data=np.linspace(0, 1, 3) * unit_registry.Pa, coords=coords
            )
            units = {None: unit_registry.hPa}
            if "u" in coords:
                units["u"] = unit_registry.mm
        elif typename == "Dataset":
            obj = Dataset(
                data_vars={
                    "a": ("x", np.linspace(-1, 1, 3) * unit_registry.s),
                    "b": ("x", np.linspace(1, 2, 3) * unit_registry.kg),
                },
                coords=coords,
            )
            units = {
                "a": unit_registry.ms,
                "b": unit_registry.gram,
            }
            if "u" in coords:
                units["u"] = unit_registry.mm

        actual = conversion.convert_units(obj, units)

        assert conversion.extract_units(actual) == units
        assert_equal(obj, actual)

    @pytest.mark.parametrize(
        "units",
        (
            pytest.param({None: None, "u": None}, id="no units"),
            pytest.param({None: unit_registry.m, "u": None}, id="data units"),
            pytest.param({None: None, "u": unit_registry.s}, id="coord units"),
            pytest.param(
                {None: unit_registry.m, "u": unit_registry.s}, id="data and coord units"
            ),
        ),
    )
    @pytest.mark.parametrize("typename", ("Variable", "DataArray", "Dataset"))
    def test_extract_units(self, typename, units):
        if typename == "Variable":
            data_units = units.get(None) or 1
            data = np.linspace(0, 1, 2) * data_units

            units = units.copy()
            units.pop("u")

            obj = Variable("x", data)
        elif typename == "DataArray":
            data_units = units.get(None) or 1
            data = np.linspace(0, 1, 2) * data_units

            coord_units = units.get("u") or 1
            coords = {"u": ("x", np.arange(2) * coord_units)}

            obj = DataArray(data, dims="x", coords=coords)
        elif typename == "Dataset":
            data_units = units.get(None)
            data1 = np.linspace(-1, 1, 2) * (data_units or 1)
            data2 = np.linspace(0, 1, 2) * (data_units or 1)

            coord_units = units.get("u") or 1
            coords = {"u": ("x", np.arange(2) * coord_units)}

            units = units.copy()
            units.pop(None)
            units.update({"a": data_units, "b": data_units})

            obj = Dataset({"a": ("x", data1), "b": ("x", data2)}, coords=coords)

        assert conversion.extract_units(obj) == units
