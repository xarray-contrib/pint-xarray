import pytest

from pint_xarray.itertools import separate


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
