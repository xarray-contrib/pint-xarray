import pytest

from pint_xarray.itertools import separate, unique, zip_mappings


@pytest.mark.parametrize(
    ["predicate", "iterable"],
    (
        (lambda x: x % 2 == 0, range(10)),
        (lambda x: x in [0, 2, 3, 5], range(10)),
        (lambda x: "s" in x, ["ab", "de", "sf", "fs"]),
    ),
)
def test_separate(predicate, iterable):
    actual_false, actual_true = separate(predicate, iterable)

    expected_true = [el for el in iterable if predicate(el)]
    expected_false = [el for el in iterable if not predicate(el)]

    assert actual_true == expected_true
    assert actual_false == expected_false


@pytest.mark.parametrize(
    ["iterable", "expected"],
    (
        ([5, 4, 4, 1, 2, 3, 2, 1], [5, 4, 1, 2, 3]),
        (list("dadgafffgaefed"), list("dagfe")),
    ),
)
def test_unique(iterable, expected):
    actual = unique(iterable)

    assert actual == expected


@pytest.mark.parametrize(
    ["mappings", "expected"],
    (
        (({"a": 1, "c": 2}, {"a": 2, "b": 0}), [("a", (1, 2))]),
        (({"a": 1, "b": 2}, {"a": 2, "b": 3}), [("a", (1, 2)), ("b", (2, 3))]),
    ),
)
def test_zip_mappings(mappings, expected):
    actual = list(zip_mappings(*mappings))
    assert actual == expected
