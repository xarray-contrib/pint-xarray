[![CI](https://github.com/calgray/astropy-xarray/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/calgray/astropy-xarray/actions/workflows/ci.yml)
[![code coverage](https://codecov.io/gh/calgray/astropy-xarray/branch/main/graph/badge.svg)](https://codecov.io/gh/calgray/astropy-xarray)
[![docs](https://readthedocs.org/projects/astropy-xarray/badge/?version=latest)](https://astropy-xarray.readthedocs.io)
[![PyPI version](https://img.shields.io/pypi/v/astropy-xarray.svg)](https://pypi.org/project/astropy-xarray)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/astropy-xarray)](https://github.com/conda-forge/astropy-xarray-feedstock)

# astropy-xarray

A convenience wrapper for using [astropy](https://www.astropy.org) with
[xarray](https://xarray.pydata.org).

## Simple Usage

To convert the variables of a `Dataset` to quantities, use accessor `.astropy.quantify()`:

```{code-block} python
import astropy_xarray
import xarray as xr

ds = xr.Dataset({"a": ("x", [0, 1, 2]), "b": ("y", [-3, 5, 1], {"units": "m"})})
ds
```

output:

```
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

to convert to different units, use accessor `.astropy.to()`:

```{code-block} python
c = q.astropy.to({"a": "ms", "b": "km"})
c
```

output:

```
<xarray.Dataset> Size: 48B
Dimensions:  (x: 3, y: 3)
Dimensions without coordinates: x, y
Data variables:
    a        (x) float64 24B [ms] 0.0 1e+03 2e+03
    b        (y) float64 24B [km] -0.003 0.005 0.001
```

to convert back to non-quantities for portability, use accessor `.astropy.dequantify()`:

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

## SkyCoord Usage

To convert a `astropy.skyCoord` to a `Dataset`, use `skycoord_to_dataset()`.

```{code-block} python
import astropy.units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.time import Time

from astropy_xarray.coordinates.sky_coord import (
    skycoord_to_dataset,
)

sc = skycoord_to_dataset(
   SkyCoord(
      ra=[[2, 6, 7, 4]] * u.deg,
      dec=[[4, 7, 4, 3]] * u.deg,
      pm_ra_cosdec=[[1, 1, 1, 1]] * u.mas / u.yr,
      pm_dec=[[1, 1, 1, 1]] * u.mas / u.yr,
      frame="icrs",
   ),
   coords={
      "timestamp": ("time", Time([1.7e9], format='unix')),
      "field_label": ("field", ["a", "b", "c", "d"]),
   }
)
sc
```

output:

```{code-block} python
<xarray.Dataset> Size: 152B
Dimensions:       (time: 1, field: 4)
Coordinates:
    timestamp     (time) float64 8B [utc unix] 2023-11-14T22:13:20.000000000
    field_label   (field) <U1 16B 'a' 'b' 'c' 'd'
Dimensions without coordinates: time, field
Data variables:
    ra            (time, field) float64 32B [°] 2.0 6.0 7.0 4.0
    dec           (time, field) float64 32B [°] 4.0 7.0 4.0 3.0
    pm_ra_cosdec  (time, field) float64 32B [mas yr⁻¹] 1.0 1.0 1.0 1.0
    pm_dec        (time, field) float64 32B [mas yr⁻¹] 1.0 1.0 1.0 1.0
Attributes:
    frame:    {'name': 'icrs', 'representation_type': 'spherical', 'different...
```

To convert SkyCoord-like datasets back to astropy types, use `.astropy.to_skycoord()`:

```python
sc.astropy.to_skycoord()
```

output:

```
<SkyCoord (ICRS): (ra, dec) in deg
    [[(2., 4.), (6., 7.), (7., 4.), (4., 3.)]]
 (pm_ra_cosdec, pm_dec) in mas / yr
    [[(1., 1.), (1., 1.), (1., 1.), (1., 1.)]]>
```

For more, see the [documentation](https://astropy-xarray.readthedocs.io/en/latest/)
