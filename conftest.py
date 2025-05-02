import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace, tmpdir):
    import numpy as np
    import pandas as pd
    import astropy
    import astropy.units
    import xarray as xr

    import astropy_xarray

    ureg = astropy.units

    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["xr"] = xr
    doctest_namespace["astropy"] = astropy
    doctest_namespace["ureg"] = ureg
    doctest_namespace["astropy_xarray"] = astropy_xarray

    # always seed numpy.random to make the examples deterministic
    np.random.seed(0)
