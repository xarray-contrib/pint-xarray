[![CI](https://github.com/xarray-contrib/pint-xarray/workflows/CI/badge.svg?branch=main)](https://github.com/xarray-contrib/pint-xarray/actions?query=branch%3Amain)
[![code coverage](https://codecov.io/gh/xarray-contrib/pint-xarray/branch/main/graph/badge.svg)](https://codecov.io/gh/xarray-contrib/pint-xarray)
[![docs](https://readthedocs.org/projects/pint-xarray/badge/?version=latest)](https://pint-xarray.readthedocs.io)
[![PyPI version](https://img.shields.io/pypi/v/pint-xarray.svg)](https://pypi.org/project/pint-xarray)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/pint-xarray)](https://github.com/conda-forge/pint-xarray-feedstock)

# pint-xarray

A convenience wrapper for using [pint](https://pint.readthedocs.io) with
[xarray](https://xarray.pydata.org).

## Usage

To convert the variables of a `Dataset` to quantities:

```python
In [1]: import pint_xarray
   ...: import xarray as xr

In [2]: ds = xr.Dataset({"a": ("x", [0, 1, 2]), "b": ("y", [-3, 5, 1], {"units": "m"})})
   ...: ds
Out[2]:
<xarray.Dataset>
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) int64 0 1 2
    b        (y) int64 -3 5 1

In [3]: q = ds.pint.quantify(a="s")
   ...: q
Out[3]:
<xarray.Dataset>
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) int64 [s] 0 1 2
    b        (y) int64 [m] -3 5 1
```

to convert to different units:

```python
In [4]: c = q.pint.to({"a": "ms", "b": "km"})
   ...: c
Out[4]:
<xarray.Dataset>
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) float64 [ms] 0.0 1e+03 2e+03
    b        (y) float64 [km] -0.003 0.005 0.001
```

to convert back to non-quantities:

```python
In [5]: d = c.pint.dequantify()
   ...: d
Out[5]:
<xarray.Dataset>
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) float64 0.0 1e+03 2e+03
    b        (y) float64 -0.003 0.005 0.001
```

For more, see the [documentation](https://pint-xarray.readthedocs.io)
