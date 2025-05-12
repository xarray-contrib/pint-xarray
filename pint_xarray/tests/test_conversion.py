import numpy as np
import pandas as pd
import pint
import pytest
from xarray import Coordinates, DataArray, Dataset, Variable
from xarray.core.indexes import PandasIndex

from pint_xarray import conversion
from pint_xarray.index import PintIndex
from pint_xarray.tests.utils import (
    assert_array_equal,
    assert_array_units_equal,
    assert_identical,
    assert_indexer_units_equal,
    assert_indexers_equal,
)

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

    if not isinstance(q, Quantity):
        q = Quantity(q)

    return q.to(u)


def strip_quantity(q):
    try:
        return q.magnitude
    except AttributeError:
        return q


class TestArrayFunctions:
    @pytest.mark.parametrize(
        ["unit", "data", "expected", "match"],
        (
            pytest.param(
                1.2, np.array([0, 1]), None, "cannot use .+ as a unit", id="not a unit"
            ),
            pytest.param(
                1, np.array([0, 1]), None, "cannot use .+ as a unit", id="no unit (1)"
            ),
            pytest.param(
                None, np.array([0, 1]), np.array([0, 1]), None, id="no unit (None)"
            ),
            pytest.param(
                "m", np.array([0, 1]), None, "cannot use .+ as a unit", id="string"
            ),
            pytest.param(
                Unit("m"),
                np.array([0, 1]),
                Quantity([0, 1], "m"),
                None,
                id="unit object",
            ),
            pytest.param(
                Unit("m"),
                Quantity(np.array([0, 1]), "s"),
                None,
                "already has units",
                id="unit object on quantity",
            ),
            pytest.param(
                Unit("m"),
                Quantity(np.array([0, 1]), "m"),
                Quantity(np.array([0, 1]), "m"),
                None,
                id="unit object on quantity with same unit",
            ),
            pytest.param(
                Unit("mm"),
                Quantity(np.array([0, 1]), "m"),
                None,
                "already has units",
                id="unit object on quantity with similar unit",
            ),
        ),
    )
    def test_array_attach_units(self, data, unit, expected, match):
        if match is not None:
            with pytest.raises(ValueError, match=match):
                conversion.array_attach_units(data, unit)

            return

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
                Unit("deg"),
                np.array([0, np.pi / 2, np.pi]),
                Quantity([0, 90, 180], "deg"),
                None,
                None,
                id="dimensionless-ndarray",
            ),
            pytest.param(
                Unit("mm"),
                np.array([0, np.pi / 2, np.pi]),
                None,
                pint.DimensionalityError,
                None,
                id="unit-ndarray",
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
        ["data", "expected"],
        (
            pytest.param(np.array([0, 1]), None, id="array_like"),
            pytest.param(Quantity([1, 2], "m"), Unit("m"), id="quantity"),
        ),
    )
    def test_array_extract_units(self, data, expected):
        actual = conversion.array_extract_units(data)

        assert expected == actual

    @pytest.mark.parametrize(
        ["data", "expected"],
        (
            pytest.param(np.array([1, 2]), np.array([1, 2]), id="array_like"),
            pytest.param(Quantity([1, 2], "m"), np.array([1, 2]), id="quantity"),
        ),
    )
    def test_array_strip_units(self, data, expected):
        actual = conversion.array_strip_units(data)

        assert_array_equal(expected, actual)


class TestXarrayFunctions:
    @pytest.mark.parametrize("type", ("Dataset", "DataArray"))
    @pytest.mark.parametrize(
        "units",
        (
            pytest.param({}, id="empty units"),
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
        q_x = to_quantity(x, units.get("x"))
        q_u = to_quantity(u, units.get("u"))

        index = PandasIndex(x, dim="x")
        if units.get("x") is not None:
            index = PintIndex(index=index, units=units.get("x"))

        obj = Dataset({"a": ("x", a), "b": ("x", b)}, coords={"u": ("x", u), "x": x})
        coords = Coordinates(
            coords={"u": Variable("x", q_u), "x": Variable("x", q_x)},
            indexes={"x": index},
        )
        expected = Dataset(
            {"a": ("x", q_a), "b": ("x", q_b)},
            coords=coords,
        )

        if type == "DataArray":
            obj = obj["a"]
            expected = expected["a"]

        actual = conversion.attach_units(obj, units)
        assert_identical(actual, expected)

        if units.get("x") is None:
            assert not isinstance(actual.xindexes["x"], PintIndex)
        else:
            assert isinstance(actual.xindexes["x"], PintIndex)
            assert actual.xindexes["x"].units == {"x": units.get("x")}

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
                ValueError,
                "(?s)Cannot convert variables:.+'u'",
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
                ValueError,
                "(?s)Cannot convert variables:.+'a'",
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
                ValueError,
                "(?s)Cannot convert variables:.+'x'",
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
                ValueError,
                "(?s)Cannot convert variables:.+'u'",
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

        x_index = PandasIndex(pd.Index(x), "x")
        if original_units.get("x") is not None:
            x_index = PintIndex(index=x_index, units={"x": original_units.get("x")})

        obj = Dataset(
            {
                "a": ("x", q_a),
                "b": ("x", q_b),
            },
            coords=Coordinates(
                {"u": ("x", q_u), "x": ("x", q_x)},
                indexes={"x": x_index},
            ),
        )
        if type == "DataArray":
            obj = obj["a"]

        if error is not None:
            with pytest.raises(error, match=match):
                conversion.convert_units(obj, units)

            return

        expected_a = convert_quantity(q_a, units.get("a", original_units.get("a")))
        expected_b = convert_quantity(q_b, units.get("b", original_units.get("b")))
        expected_u = convert_quantity(q_u, units.get("u", original_units.get("u")))
        expected_x = convert_quantity(q_x, units.get("x"))
        expected_index = PandasIndex(pd.Index(strip_quantity(expected_x)), "x")
        if units.get("x") is not None:
            expected_index = PintIndex(
                index=expected_index, units={"x": units.get("x")}
            )

        expected = Dataset(
            {
                "a": ("x", expected_a),
                "b": ("x", expected_b),
            },
            coords=Coordinates(
                {"u": ("x", expected_u), "x": ("x", expected_x)},
                indexes={"x": expected_index},
            ),
        )

        if type == "DataArray":
            expected = expected["a"]

        actual = conversion.convert_units(obj, units)

        assert conversion.extract_units(actual) == conversion.extract_units(expected)
        assert_identical(actual, expected)

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

        index = PandasIndex(x, "x")
        if units.get("x") is not None:
            index = PintIndex(index=index, units={"x": units.get("x")})

        obj = Dataset(
            {
                "a": ("x", to_quantity(a, units.get("a"))),
                "b": ("x", to_quantity(b, units.get("b"))),
            },
            coords=Coordinates(
                {
                    "u": ("x", to_quantity(u, units.get("u"))),
                    "x": ("x", to_quantity(x, units.get("x"))),
                },
                indexes={"x": index},
            ),
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
                Dataset(coords={"t": ("t", [], {"units": "seconds since 2000-01-01"})}),
                {},
                id="datetime_unit",
            ),
        ),
    )
    def test_extract_unit_attributes(self, obj, expected):
        actual = conversion.extract_unit_attributes(obj)
        assert expected == actual

    @pytest.mark.parametrize(
        ["obj", "expected"],
        (
            pytest.param(
                DataArray(
                    dims="x",
                    data=Quantity([0, 4, 3], "kg"),
                    coords=Coordinates(
                        {
                            "u": ("x", Quantity([2, 3, 4], "s")),
                            "x": Quantity([0, 1, 2], "m"),
                        },
                        indexes={},
                    ),
                ),
                {None: None, "u": None, "x": None},
                id="DataArray",
            ),
            pytest.param(
                Dataset(
                    data_vars={
                        "a": ("x", Quantity([3, 2, 5], "Pa")),
                        "b": ("x", Quantity([0, 2, -1], "kg")),
                    },
                    coords=Coordinates(
                        {
                            "u": ("x", Quantity([2, 3, 4], "s")),
                            "x": Quantity([0, 1, 2], "m"),
                        },
                        indexes={},
                    ),
                ),
                {"a": None, "b": None, "u": None, "x": None},
                id="Dataset",
            ),
        ),
    )
    def test_strip_units(self, obj, expected):
        actual = conversion.strip_units(obj)
        assert conversion.extract_units(actual) == expected

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
                Dataset(coords={"t": ("t", [], {"units": "seconds since 2000-01-01"})}),
                {},
                id="datetime_unit",
            ),
        ),
    )
    def test_strip_unit_attributes(self, obj, expected):
        actual = conversion.strip_unit_attributes(obj)
        expected = {}

        assert (
            filter_none_values(conversion.extract_unit_attributes(actual)) == expected
        )


class TestIndexerFunctions:
    @pytest.mark.parametrize(
        ["indexers", "units", "expected", "error", "match"],
        (
            pytest.param(
                {"x": 1}, {"x": None}, {"x": 1}, None, None, id="scalar-no units"
            ),
            pytest.param(
                {"x": 1},
                {"x": "dimensionless"},
                None,
                ValueError,
                "(?s)Cannot convert indexers:.+'x'",
                id="scalar-dimensionless",
            ),
            pytest.param(
                {"x": Quantity(1, "m")},
                {"x": Unit("dm")},
                {"x": Quantity(10, "dm")},
                None,
                None,
                id="scalar-units",
            ),
            pytest.param(
                {"x": np.array([1, 2])},
                {"x": None},
                {"x": np.array([1, 2])},
                None,
                None,
                id="array-no units",
            ),
            pytest.param(
                {"x": Quantity([1, 2], "m")},
                {"x": Unit("dm")},
                {"x": Quantity([10, 20], "dm")},
                None,
                None,
                id="array-units",
            ),
            pytest.param(
                {"x": Variable("x", [1, 2])},
                {"x": None},
                {"x": Variable("x", [1, 2])},
                None,
                None,
                id="Variable-no units",
            ),
            pytest.param(
                {"x": Variable("x", Quantity([1, 2], "m"))},
                {"x": Unit("dm")},
                {"x": Variable("x", Quantity([10, 20], "dm"))},
                None,
                None,
                id="Variable-units",
            ),
            pytest.param(
                {"x": DataArray([1, 2], dims="x")},
                {"x": None},
                {"x": DataArray([1, 2], dims="x")},
                None,
                None,
                id="DataArray-no units",
            ),
            pytest.param(
                {"x": DataArray(Quantity([1, 2], "m"), dims="x")},
                {"x": Unit("dm")},
                {"x": DataArray(Quantity([10, 20], "dm"), dims="x")},
                None,
                None,
                id="DataArray-units",
            ),
            pytest.param(
                {"x": slice(None)},
                {"x": None},
                {"x": slice(None)},
                None,
                None,
                id="empty slice-no units",
            ),
            pytest.param(
                {"x": slice(1, None)},
                {"x": None},
                {"x": slice(1, None)},
                None,
                None,
                id="slice-no units",
            ),
            pytest.param(
                {"x": slice(Quantity(1, "m"), Quantity(2, "m"))},
                {"x": Unit("m")},
                {"x": slice(Quantity(1, "m"), Quantity(2, "m"))},
                None,
                None,
                id="slice-identical units",
            ),
            pytest.param(
                {"x": slice(Quantity(1, "m"), Quantity(2000, "mm"))},
                {"x": Unit("dm")},
                {"x": slice(Quantity(10, "dm"), Quantity(20, "dm"))},
                None,
                None,
                id="slice-compatible units",
            ),
            pytest.param(
                {"x": slice(Quantity(1, "m"), Quantity(2, "m"))},
                {"x": Unit("ms")},
                None,
                ValueError,
                "(?s)Cannot convert indexers:.+'x'",
                id="slice-incompatible units",
            ),
            pytest.param(
                {"x": slice(1000, Quantity(2000, "ms"))},
                {"x": Unit("s")},
                None,
                ValueError,
                "(?s)Cannot convert indexers:.+'x'",
                id="slice-incompatible units-mixed",
            ),
        ),
    )
    def test_convert_indexer_units(self, indexers, units, expected, error, match):
        if error is not None:
            with pytest.raises(error, match=match):
                conversion.convert_indexer_units(indexers, units)
        else:
            actual = conversion.convert_indexer_units(indexers, units)
            assert_indexers_equal(actual, expected)
            assert_indexer_units_equal(actual, expected)

    @pytest.mark.parametrize(
        ["indexers", "expected"],
        (
            pytest.param({"x": 1}, {"x": None}, id="scalar-no units"),
            pytest.param({"x": Quantity(1, "m")}, {"x": Unit("m")}, id="scalar-units"),
            pytest.param({"x": np.array([1, 2])}, {"x": None}, id="array-no units"),
            pytest.param(
                {"x": Quantity([1, 2], "s")}, {"x": Unit("s")}, id="array-units"
            ),
            pytest.param(
                {"x": Variable("x", [1, 2])}, {"x": None}, id="Variable-no units"
            ),
            pytest.param(
                {"x": Variable("x", Quantity([1, 2], "m"))},
                {"x": Unit("m")},
                id="Variable-units",
            ),
            pytest.param(
                {"x": DataArray([1, 2], dims="x")}, {"x": None}, id="DataArray-no units"
            ),
            pytest.param(
                {"x": DataArray(Quantity([1, 2], "s"), dims="x")},
                {"x": Unit("s")},
                id="DataArray-units",
            ),
            pytest.param({"x": slice(None)}, {"x": None}, id="empty slice-no units"),
            pytest.param({"x": slice(1, None)}, {"x": None}, id="slice-no units"),
            pytest.param(
                {"x": slice(Quantity(1, "m"), Quantity(2, "m"))},
                {"x": Unit("m")},
                id="slice-identical units",
            ),
            pytest.param(
                {"x": slice(Quantity(1, "m"), Quantity(2000, "mm"))},
                {"x": Unit("m")},
                id="slice-compatible units",
            ),
            pytest.param(
                {"x": slice(Quantity(1, "m"), Quantity(2, "ms"))},
                ValueError,
                id="slice-incompatible units",
            ),
            pytest.param(
                {"x": slice(1, Quantity(2, "ms"))},
                ValueError,
                id="slice-incompatible units-mixed",
            ),
            pytest.param(
                {"x": slice(1, Quantity(2, "rad"))},
                {"x": Unit("rad")},
                id="slice-incompatible units-mixed-dimensionless",
            ),
        ),
    )
    def test_extract_indexer_units(self, indexers, expected):
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                conversion.extract_indexer_units(indexers)
        else:
            actual = conversion.extract_indexer_units(indexers)
            assert actual == expected

    @pytest.mark.parametrize(
        ["indexers", "expected"],
        (
            pytest.param({"x": 1}, {"x": 1}, id="scalar-no units"),
            pytest.param({"x": Quantity(1, "m")}, {"x": 1}, id="scalar-units"),
            pytest.param(
                {"x": np.array([1, 2])},
                {"x": np.array([1, 2])},
                id="array-no units",
            ),
            pytest.param(
                {"x": Quantity([1, 2], "s")}, {"x": np.array([1, 2])}, id="array-units"
            ),
            pytest.param(
                {"x": Variable("x", [1, 2])},
                {"x": Variable("x", [1, 2])},
                id="Variable-no units",
            ),
            pytest.param(
                {"x": Variable("x", Quantity([1, 2], "m"))},
                {"x": Variable("x", [1, 2])},
                id="Variable-units",
            ),
            pytest.param(
                {"x": DataArray([1, 2], dims="x")},
                {"x": DataArray([1, 2], dims="x")},
                id="DataArray-no units",
            ),
            pytest.param(
                {"x": DataArray(Quantity([1, 2], "s"), dims="x")},
                {"x": DataArray([1, 2], dims="x")},
                id="DataArray-units",
            ),
            pytest.param(
                {"x": slice(None)}, {"x": slice(None)}, id="empty slice-no units"
            ),
            pytest.param(
                {"x": slice(1, None)}, {"x": slice(1, None)}, id="slice-no units"
            ),
            pytest.param(
                {"x": slice(Quantity(1, "m"), Quantity(2, "m"))},
                {"x": slice(1, 2)},
                id="slice-units",
            ),
        ),
    )
    def test_strip_indexer_units(self, indexers, expected):
        actual = conversion.strip_indexer_units(indexers)

        assert_indexers_equal(actual, expected)
