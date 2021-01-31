import numpy as np
import pint
import pytest
from xarray import DataArray, Dataset, Variable

from pint_xarray import conversion

from .utils import assert_array_equal, assert_array_units_equal, assert_identical

unit_registry = pint.UnitRegistry()
Quantity = unit_registry.Quantity
Unit = unit_registry.Unit

pytestmark = pytest.mark.filterwarnings("error::pint.UnitStrippedWarning")


def filter_none_values(mapping):
    return {k: v for k, v in mapping.items() if v is not None}


def to_quantity(v, u):
    if u is None:
        return v

    return Quantity(v, u)


def convert_quantity(q, u):
    if u is None:
        return q

    return q.to(u)


def strip_quantity(q):
    try:
        return q.magnitude
    except AttributeError:
        return q


class TestArrayFunctions:
    @pytest.mark.parametrize(
        ["unit", "data", "match"],
        (
            pytest.param(
                1.2, np.array([0, 1]), "cannot use .+ as a unit", id="not a unit"
            ),
            pytest.param(
                1, np.array([0, 1]), "cannot use .+ as a unit", id="no unit (1)"
            ),
            pytest.param(None, np.array([0, 1]), None, id="no unit (None)"),
            pytest.param("m", np.array([0, 1]), "cannot use .+ as a unit", id="string"),
            pytest.param(unit_registry.m, np.array([0, 1]), None, id="unit object"),
            pytest.param(
                unit_registry.m,
                Quantity(np.array([0, 1]), "s"),
                "already has units",
                id="unit object on quantity",
            ),
        ),
    )
    def test_array_attach_units(self, data, unit, match):
        if match is not None:
            with pytest.raises(ValueError, match=match):
                conversion.array_attach_units(data, unit)

            return

        expected = unit_registry.Quantity(data, "m") if unit is not None else data
        actual = conversion.array_attach_units(data, unit)

        assert_array_units_equal(expected, actual)
        assert_array_equal(expected, actual)

    @pytest.mark.parametrize(
        ["unit", "data", "expected", "error", "match"],
        (
            pytest.param(
                1.2,
                np.array([0, 1, 2]),
                None,
                ValueError,
                "cannot use .+ as a unit",
                id="not a unit-ndarray",
            ),
            pytest.param(
                1,
                np.array([0, 1, 2]),
                None,
                ValueError,
                "cannot use .+ as a unit",
                id="no unit (1)-ndarray",
            ),
            pytest.param(
                None,
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
                None,
                None,
                id="no unit (None)-ndarray",
            ),
            pytest.param(
                "mm",
                np.array([0, 1, 2]),
                None,
                ValueError,
                "cannot convert a non-quantity using .+ as unit",
                id="string-ndarray",
            ),
            pytest.param(
                "mm",
                Quantity([0, 1, 2], "m"),
                Quantity([0, 1000, 2000], "mm"),
                None,
                None,
                id="string-quantity",
            ),
            pytest.param(
                unit_registry.mm,
                Quantity([0, 1, 2], "m"),
                Quantity([0, 1000, 2000], "mm"),
                None,
                None,
                id="unit object",
            ),
            pytest.param(
                "s",
                Quantity([0, 1, 2], "m"),
                None,
                pint.DimensionalityError,
                None,
                id="quantity-incompatible unit",
            ),
        ),
    )
    def test_array_convert_units(self, data, unit, expected, error, match):
        if error is not None:
            with pytest.raises(error, match=match):
                conversion.array_convert_units(data, unit)

            return

        actual = conversion.array_convert_units(data, unit)
        assert_array_equal(expected, actual)

    @pytest.mark.parametrize(
        "data",
        (
            pytest.param(np.array([0, 1]), id="array_like"),
            pytest.param(Quantity([1, 2], "m"), id="quantity"),
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
            pytest.param(Quantity([1, 2], "m"), id="quantity"),
        ),
    )
    def test_array_strip_units(self, data):
        expected = np.array([1, 2])
        actual = conversion.array_strip_units(data)

        assert_array_equal(expected, actual)


class TestXarrayFunctions:
    @pytest.mark.parametrize("type", ("Dataset", "DataArray"))
    @pytest.mark.parametrize(
        "units",
        (
            pytest.param({"a": None, "b": None, "u": None, "x": None}, id="no units"),
            pytest.param(
                {"a": unit_registry.m, "b": unit_registry.m, "u": None, "x": None},
                id="data units",
            ),
            pytest.param(
                {"a": None, "b": None, "u": unit_registry.s, "x": None},
                id="coord units",
            ),
            pytest.param(
                {"a": None, "b": None, "u": None, "x": unit_registry.m}, id="dim units"
            ),
        ),
    )
    def test_attach_units(self, type, units):
        a = np.linspace(-1, 1, 5)
        b = np.linspace(0, 1, 5)
        x = np.linspace(0, 100, 5)
        u = np.arange(5)

        q_a = to_quantity(a, units.get("a"))
        q_b = to_quantity(b, units.get("b"))
        q_u = to_quantity(u, units.get("u"))

        obj = Dataset({"a": ("x", a), "b": ("x", b)}, coords={"u": ("x", u), "x": x})
        expected = Dataset(
            {"a": ("x", q_a), "b": ("x", q_b)},
            coords={"u": ("x", q_u), "x": ("x", x, {"units": units.get("x")})},
        )
        if type == "DataArray":
            obj = obj["a"]
            expected = expected["a"]

        actual = conversion.attach_units(obj, units)
        assert_identical(actual, expected)

    @pytest.mark.parametrize("type", ("DataArray", "Dataset"))
    def test_attach_unit_attributes(self, type):
        units = {"a": "K", "b": "hPa", "u": "m", "x": "s"}
        obj = Dataset(
            data_vars={"a": ("x", []), "b": ("x", [])},
            coords={"x": [], "u": ("x", [])},
        )
        expected = Dataset(
            {"a": ("x", [], {"units": "K"}), "b": ("x", [], {"units": "hPa"})},
            coords={"x": ("x", [], {"units": "s"}), "u": ("x", [], {"units": "m"})},
        )
        if type == "DataArray":
            obj = obj["a"]
            expected = expected["a"]

        actual = conversion.attach_unit_attributes(obj, units)
        assert_identical(actual, expected)

    @pytest.mark.parametrize("type", ("DataArray", "Dataset"))
    @pytest.mark.parametrize(
        ["variant", "units", "error", "match"],
        (
            pytest.param("none", {}, None, None, id="none-no units"),
            pytest.param(
                "none",
                {"a": Unit("g"), "b": Unit("Pa"), "u": Unit("ms"), "x": Unit("mm")},
                None,
                None,
                id="none-with units",
            ),
            pytest.param("data", {}, None, None, id="data-no units"),
            pytest.param(
                "data",
                {"a": Unit("g"), "b": Unit("Pa")},
                None,
                None,
                id="data-compatible units",
            ),
            pytest.param(
                "data",
                {"a": Unit("s"), "b": Unit("m")},
                None,
                None,
                id="data-incompatible units",
            ),
            pytest.param(
                "dims",
                {},
                None,
                None,
                id="dims-no units",
            ),
            pytest.param(
                "dims",
                {"x": Unit("mm")},
                None,
                None,
                id="dims-compatible units",
            ),
            pytest.param(
                "dims",
                {"x": Unit("ms")},
                None,
                None,
                id="dims-incompatible units",
            ),
            pytest.param(
                "coords",
                {},
                None,
                None,
                id="coords-no units",
            ),
            pytest.param(
                "coords",
                {"u": Unit("ms")},
                None,
                None,
                id="coords-compatible units",
            ),
            pytest.param(
                "coords",
                {"u": Unit("mm")},
                None,
                None,
                id="coords-incompatible units",
            ),
        ),
    )
    def test_convert_units(self, type, variant, units, error, match):
        variants = {
            "none": {"a": None, "b": None, "u": None, "x": None},
            "data": {"a": Unit("kg"), "b": Unit("hPa"), "u": None, "x": None},
            "coords": {"a": None, "b": None, "u": Unit("s"), "x": None},
            "dims": {"a": None, "b": None, "u": None, "x": Unit("m")},
        }

        a = np.linspace(-1, 1, 3)
        b = np.linspace(1, 2, 3)
        u = np.linspace(0, 100, 3)
        x = np.arange(3)

        original_units = variants.get(variant)
        q_a = to_quantity(a, original_units.get("a"))
        q_b = to_quantity(b, original_units.get("b"))
        q_u = to_quantity(u, original_units.get("u"))
        q_x = to_quantity(x, original_units.get("x"))

        obj = Dataset(
            {
                "a": ("x", q_a),
                "b": ("x", q_b),
            },
            coords={
                "u": ("x", q_u),
                "x": ("x", x, {"units": original_units.get("x")}),
            },
        )
        expected = Dataset(
            {
                "a": (
                    "x",
                    convert_quantity(q_a, units.get("a", original_units.get("a"))),
                ),
                "b": (
                    "x",
                    convert_quantity(q_b, units.get("b", original_units.get("b"))),
                ),
            },
            coords={
                "u": (
                    "x",
                    convert_quantity(q_u, units.get("u", original_units.get("u"))),
                ),
                "x": (
                    "x",
                    strip_quantity(convert_quantity(q_x, units.get("x"))),
                    {"units": units.get("x", original_units.get("x"))},
                ),
            },
        )

        if type == "DataArray":
            obj = obj["a"]
            expected = expected["a"]

        actual = conversion.convert_units(obj, units)

        assert conversion.extract_units(actual) == conversion.extract_units(expected)
        assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "units",
        (
            pytest.param({"a": None, "b": None, "u": None, "x": None}, id="none"),
            pytest.param(
                {"a": Unit("kg"), "b": Unit("hPa"), "u": None, "x": None}, id="data"
            ),
            pytest.param({"a": None, "b": None, "u": Unit("s"), "x": None}, id="coord"),
            pytest.param({"a": None, "b": None, "u": None, "x": Unit("m")}, id="dims"),
        ),
    )
    @pytest.mark.parametrize("type", ("DataArray", "Dataset"))
    def test_extract_units(self, type, units):
        a = np.linspace(-1, 1, 2)
        b = np.linspace(0, 1, 2)
        u = np.linspace(0, 100, 2)
        x = np.arange(2)

        obj = Dataset(
            {
                "a": ("x", to_quantity(a, units.get("a"))),
                "b": ("x", to_quantity(b, units.get("b"))),
            },
            coords={
                "u": ("x", to_quantity(u, units.get("u"))),
                "x": ("x", x, {"units": units.get("x")}),
            },
        )
        if type == "DataArray":
            obj = obj["a"]
            units = units.copy()
            del units["b"]

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
                Variable("x", [], {"units": "hPa"}), {None: "hPa"}, id="Variable"
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
                Variable("x", [], {"units": "hPa"}), {None: "hPa"}, id="Variable"
            ),
        ),
    )
    def test_strip_unit_attributes(self, obj, expected):
        actual = conversion.strip_unit_attributes(obj)
        expected = {}

        assert (
            filter_none_values(conversion.extract_unit_attributes(actual)) == expected
        )
