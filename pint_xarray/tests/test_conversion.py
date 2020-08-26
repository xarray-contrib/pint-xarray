import numpy as np
import pint
import pytest
from xarray import DataArray, Dataset, Variable

from pint_xarray import conversion

from .utils import assert_array_equal, assert_array_units_equal, assert_equal

unit_registry = pint.UnitRegistry()

pytestmark = pytest.mark.filterwarnings("error::pint.UnitStrippedWarning")


def filter_none_values(mapping):
    return {k: v for k, v in mapping.items() if v is not None}


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
        "obj",
        (
            pytest.param(Variable("x", np.linspace(0, 1, 5)), id="Variable"),
            pytest.param(
                DataArray(
                    data=np.linspace(0, 1, 5),
                    dims="x",
                    coords={"u": ("x", np.arange(5))},
                ),
                id="DataArray",
            ),
            pytest.param(
                Dataset(
                    {
                        "a": ("x", np.linspace(-1, 1, 5)),
                        "b": ("x", np.linspace(0, 1, 5)),
                    },
                    coords={"u": ("x", np.arange(5))},
                ),
                id="Dataset",
            ),
        ),
    )
    @pytest.mark.parametrize(
        "units",
        (
            pytest.param({None: None, "u": None}, id="no units"),
            pytest.param({None: unit_registry.m, "u": None}, id="data units"),
            pytest.param({None: None, "u": unit_registry.s}, id="coord units"),
        ),
    )
    def test_attach_units(self, obj, units):
        if isinstance(obj, Variable) and "u" in units:
            pytest.skip(msg="variables don't have coordinates")

        if isinstance(obj, Dataset):
            units = units.copy()
            data_units = units.pop(None)
            units.update({"a": data_units, "b": data_units})

        actual = conversion.attach_units(obj, units)

        assert conversion.extract_units(actual) == units

    @pytest.mark.parametrize(
        ["obj", "units"],
        (
            pytest.param(
                DataArray(dims="x", coords={"x": [], "u": ("x", [])}),
                {None: "hPa", "x": "m"},
                id="DataArray",
            ),
            pytest.param(
                Dataset(
                    data_vars={"a": ("x", []), "b": ("x", [])},
                    coords={"x": [], "u": ("x", [])},
                ),
                {"a": "K", "b": "hPa", "u": "m"},
                id="Dataset",
            ),
            pytest.param(
                Variable("x", []),
                {None: "hPa"},
                id="Variable",
            ),
        ),
    )
    def test_attach_unit_attributes(self, obj, units):
        actual = conversion.attach_unit_attributes(obj, units)
        assert units == filter_none_values(conversion.extract_unit_attributes(actual))

    @pytest.mark.parametrize(
        "variant",
        (
            "data",
            pytest.param(
                "dims", marks=pytest.mark.xfail(reason="indexes don't support units")
            ),
            "coords",
        ),
    )
    @pytest.mark.parametrize("typename", ("Variable", "DataArray", "Dataset"))
    def test_convert_units(self, typename, variant):
        if typename == "Variable":
            if variant != "data":
                pytest.skip("Variable doesn't store coordinates")

            data = np.linspace(0, 1, 3) * unit_registry.m
            obj = Variable(dims="x", data=data)
            units = {None: unit_registry.mm}
            expected_units = units
        elif typename == "DataArray":
            unit_variants = {
                "data": (unit_registry.Pa, 1, 1),
                "dims": (1, unit_registry.s, 1),
                "coords": (1, 1, unit_registry.m),
            }
            data_unit, dim_unit, coord_unit = unit_variants.get(variant)

            coords = {
                "data": {},
                "dims": {"x": [0, 1, 2] * dim_unit},
                "coords": {"u": ("x", [10, 3, 4] * coord_unit)},
            }

            obj = DataArray(
                dims="x",
                data=np.linspace(0, 1, 3) * data_unit,
                coords=coords.get(variant),
            )
            template = {
                **{obj.name: None},
                **{name: None for name in obj.coords},
            }
            units = {
                "data": {None: unit_registry.hPa},
                "dims": {"x": unit_registry.ms},
                "coords": {"u": unit_registry.mm},
            }.get(variant)

            expected_units = {**template, **units}
        elif typename == "Dataset":
            unit_variants = {
                "data": ((unit_registry.s, unit_registry.kg), 1, 1),
                "dims": ((1, 1), unit_registry.s, 1),
                "coords": ((1, 1), 1, unit_registry.m),
            }
            (data_unit1, data_unit2), dim_unit, coord_unit = unit_variants.get(variant)

            coords = {
                "data": {},
                "dims": {"x": [0, 1, 2] * dim_unit},
                "coords": {"u": ("x", [10, 3, 4] * coord_unit)},
            }

            obj = Dataset(
                data_vars={
                    "a": ("x", np.linspace(-1, 1, 3) * data_unit1),
                    "b": ("x", np.linspace(1, 2, 3) * data_unit2),
                },
                coords=coords.get(variant),
            )

            template = {
                **{name: None for name in obj.data_vars.keys()},
                **{name: None for name in obj.coords.keys()},
            }
            units = {
                "data": {"a": unit_registry.ms, "b": unit_registry.g},
                "dims": {"x": unit_registry.ms},
                "coords": {"u": unit_registry.mm},
            }.get(variant)
            expected_units = {**template, **units}

        actual = conversion.convert_units(obj, units)

        assert conversion.extract_units(actual) == expected_units
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

    @pytest.mark.parametrize(
        ["obj", "expected"],
        (
            pytest.param(
                DataArray(
                    coords={
                        "x": ("x", [], {"units": "m"}),
                        "u": ("x", [], {"units": "s"}),
                    },
                    attrs={"units": "hPa"},
                    dims="x",
                ),
                {"x": "m", "u": "s", None: "hPa"},
                id="DataArray",
            ),
            pytest.param(
                Dataset(
                    data_vars={
                        "a": ("x", [], {"units": "K"}),
                        "b": ("x", [], {"units": "hPa"}),
                    },
                    coords={
                        "x": ("x", [], {"units": "m"}),
                        "u": ("x", [], {"units": "s"}),
                    },
                ),
                {"a": "K", "b": "hPa", "x": "m", "u": "s"},
                id="Dataset",
            ),
            pytest.param(
                Variable("x", [], {"units": "hPa"}),
                {None: "hPa"},
                id="Variable",
            ),
        ),
    )
    def test_extract_unit_attributes(self, obj, expected):
        actual = conversion.extract_unit_attributes(obj)
        assert expected == actual

    @pytest.mark.parametrize(
        "obj",
        (
            pytest.param(Variable("x", [0, 4, 3] * unit_registry.m), id="Variable"),
            pytest.param(
                DataArray(
                    dims="x",
                    data=[0, 4, 3] * unit_registry.m,
                    coords={"u": ("x", [2, 3, 4] * unit_registry.s)},
                ),
                id="DataArray",
            ),
            pytest.param(
                Dataset(
                    data_vars={
                        "a": ("x", [3, 2, 5] * unit_registry.Pa),
                        "b": ("x", [0, 2, -1] * unit_registry.kg),
                    },
                    coords={"u": ("x", [2, 3, 4] * unit_registry.s)},
                ),
                id="Dataset",
            ),
        ),
    )
    def test_strip_units(self, obj):
        if isinstance(obj, Variable):
            expected_units = {None: None}
        elif isinstance(obj, DataArray):
            expected_units = {None: None}
            expected_units.update({name: None for name in obj.coords.keys()})
        elif isinstance(obj, Dataset):
            expected_units = {name: None for name in obj.variables.keys()}

        actual = conversion.strip_units(obj)
        assert conversion.extract_units(actual) == expected_units

    @pytest.mark.parametrize(
        ["obj", "expected"],
        (
            pytest.param(
                DataArray(
                    coords={
                        "x": ("x", [], {"units": "m"}),
                        "u": ("x", [], {"units": "s"}),
                    },
                    attrs={"units": "hPa"},
                    dims="x",
                ),
                {"x": "m", "u": "s", None: "hPa"},
                id="DataArray",
            ),
            pytest.param(
                Dataset(
                    data_vars={
                        "a": ("x", [], {"units": "K"}),
                        "b": ("x", [], {"units": "hPa"}),
                    },
                    coords={
                        "x": ("x", [], {"units": "m"}),
                        "u": ("x", [], {"units": "s"}),
                    },
                ),
                {"a": "K", "b": "hPa", "x": "m", "u": "s"},
                id="Dataset",
            ),
            pytest.param(
                Variable("x", [], {"units": "hPa"}),
                {None: "hPa"},
                id="Variable",
            ),
        ),
    )
    def test_strip_unit_attributes(self, obj, expected):
        actual = conversion.strip_unit_attributes(obj)
        expected = {}

        assert (
            filter_none_values(conversion.extract_unit_attributes(actual)) == expected
        )
