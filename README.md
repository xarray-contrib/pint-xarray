[![CI](https://github.com/xarray-contrib/astropy-xarray/workflows/CI/badge.svg?branch=main)](https://github.com/xarray-contrib/astropy-xarray/actions?query=branch%3Amain)
[![code coverage](https://codecov.io/gh/xarray-contrib/astropy-xarray/branch/main/graph/badge.svg)](https://codecov.io/gh/xarray-contrib/astropy-xarray)
[![docs](https://readthedocs.org/projects/astropy-xarray/badge/?version=latest)](https://astropy-xarray.readthedocs.io)
[![PyPI version](https://img.shields.io/pypi/v/astropy-xarray.svg)](https://pypi.org/project/astropy-xarray)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/astropy-xarray)](https://github.com/conda-forge/astropy-xarray-feedstock)

# astropy-xarray

A convenience wrapper for using [astropy](https://www.astropy.org) with
[xarray](https://xarray.pydata.org).

## Usage

To convert the variables of a `Dataset` to quantities:

```python
In [1]: import astropy_xarray
   ...: import xarray as xr

In [2]: ds = xr.Dataset({"a": ("x", [0, 1, 2]), "b": ("y", [-3, 5, 1], {"units": "m"})})
   ...: ds
Out[2]:
<xarray.Dataset> Size: 48B
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) int64 24B 0 1 2
    b        (y) int64 24B -3 5 1

In [3]: q = ds.astropy.quantify(a="s")
   ...: q
Out[3]:
<xarray.Dataset> Size: 48B
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) float64 24B [s] 0.0 1.0 2.0
    b        (y) float64 24B [m] -3.0 5.0 1.0
```

to convert to different units:

```python
In [4]: c = q.astropy.to({"a": "ms", "b": "km"})
   ...: c
Out[4]:
<xarray.Dataset> Size: 48B
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) float64 24B [ms] 0.0 1e+03 2e+03
    b        (y) float64 24B [km] -0.003 0.005 0.001
```

to convert back to non-quantities:

```python
In [5]: d = c.astropy.dequantify()
   ...: d
Out[5]:
<xarray.Dataset> Size: 48B
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) float64 24B 0.0 1e+03 2e+03
    b        (y) float64 24B -0.003 0.005 0.001
```

For more, see the [documentation](https://astropy-xarray.readthedocs.io/en/latest/)
