import pandas as pd
import pytest
from xarray.core.indexes import PandasIndex

from pint_xarray import unit_registry as ureg
from pint_xarray.index import PintIndex


@pytest.mark.parametrize(
    "base_index",
    [
        PandasIndex(pd.Index([1, 2, 3]), dim="x"),
        PandasIndex(pd.Index([0.1, 0.2, 0.3]), dim="x"),
        PandasIndex(pd.Index([1j, 2j, 3j]), dim="y"),
    ],
)
@pytest.mark.parametrize("units", [ureg.Unit("m"), ureg.Unit("s")])
def test_init(base_index, units):
    index = PintIndex(index=base_index, units=units)

    assert index.index.equals(base_index)
    assert index.units == units


def test_replace():
    old_index = PandasIndex([1, 2, 3], dim="y")
    new_index = PandasIndex([0.1, 0.2, 0.3], dim="x")

    old = PintIndex(index=old_index, units=ureg.Unit("m"))
    new = old._replace(new_index)

    assert new.index.equals(new_index)
    assert new.units == old.units
    # no mutation
    assert old.index.equals(old_index)
