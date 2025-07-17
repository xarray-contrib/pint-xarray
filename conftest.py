import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace, tmpdir):
    import astropy
    import astropy.units
    import numpy as np
    import pandas as pd
    import xarray as xr

    import astropy_xarray

    u = astropy.units

    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["xr"] = xr
    doctest_namespace["astropy"] = astropy
    doctest_namespace["u"] = u
    doctest_namespace["astropy_xarray"] = astropy_xarray

    # always seed numpy.random to make the examples deterministic
    np.random.seed(0)
