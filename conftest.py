import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace, tmpdir):
    import numpy as np
    import pandas as pd
    import pint
    import xarray as xr

    import pint_xarray

    ureg = pint.UnitRegistry(force_ndarray_like=True)

    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["xr"] = xr
    doctest_namespace["pint"] = pint
    doctest_namespace["ureg"] = ureg
    doctest_namespace["pint_xarray"] = pint_xarray

    # always seed numpy.random to make the examples deterministic
    np.random.seed(0)
