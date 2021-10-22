# Unit-aware arithmetic in xarray, via pint

_TLDR: Xarray now supports unit-aware operations by wrapping [pint arrays](https://pint.readthedocs.io/en/stable/), so your code can automatically track the physical units that your data represents:_

```python
distance = xr.DataArray(10).pint.quantify("metres")
time = xr.DataArray(4).pint.quantify("seconds")

distance / time
```
```
Out: 
<xarray.DataArray ()>
<Quantity(2.5, 'meter / second')>
```

## Units are integral to science

All quantities in science have units, whether explicitly or implicitly. (And even dimensionless quantities like ratios still technically have units.)

Getting our units right is finicky, and can very easily go unnoticed in our code.
Even worse, the consequences of getting units wrong can be huge! 

The most famous example of a units error has to be NASA's $125 million [Mars Climate Orbiter](https://www.simscale.com/blog/2017/12/nasa-mars-climate-orbiter-metric/), which in 1999 burned up in the Martian atmosphere instead of successfully entering orbit around Mars.
A trajectory course correction had gone wrong, and the error was eventually traced back to a units mismatch: the engineers at Lockheed Martin expressed impulse in [pound-force](https://en.wikipedia.org/wiki/Pound_(force)) seconds, whereas the engineers at JPL assumed the impulse value their part of the software received was in SI newton seconds.

<p align = "center">
<img src = "https://clqtg10snjb14i85u49wifbv-wpengine.netdna-ssl.com/wp-content/uploads/2017/12/Customers.jpg">
</p>

<p align = "center">
Newspaper cartoon depicting the incongruence in the units used by NASA and Lockheed Martin scientists that led to the Mars Climate Orbiter disaster.
</p>

We should take stories like this seriously: If we can automatically track units we can potentially eliminate a whole class of possible errors in our scientific work...

## Pint tracks units

There are a few packages for handling units in python (notably [unyt](https://github.com/yt-project/unyt) and [astropy.units](https://docs.astropy.org/en/stable/units/)), but for technical reasons we began units integration in xarray with [Pint](https://pint.readthedocs.io/en/stable/).
These various packages work by providing a numerical array type that acts similarly to a numpy array, and is intended to plug in and replace the raw numpy array (a so-called "duck array type")

Pint provides the `Quantity` object, which is a normal numpy array combined with a `pint.Unit`:

```python
q = np.array([6, 7]) * pint.Unit('metres')
print(repr(q))
```
```
Out: <Quantity([6 7], 'meter')>
```

Pint Quantities act like numpy arrays, except that the units are carried around with the arrays, propagated through operations, and checked during operations involving multiple quantities.


## Xarray now wraps Pint

Thanks to the [tireless work](https://github.com/pydata/xarray/issues/3594) of xarray core developer Justus Magin, you can now enjoy this automatic unit-handling in xarray!

Once you create a unit-aware xarray object (see below for how) you can see the units of the data variables displayed as part of the printable representation.
You also immediately get the key benefits of pint:

1) Units are propagated through arithmetic, and new quantities are built using the units of the inputs:

```python
distance = xr.DataArray(10).pint.quantify("metres")
time = xr.DataArray(4).pint.quantify("seconds")

distance / time
```
```
Out: 
<xarray.DataArray ()>
<Quantity(2.5, 'meter / second')>
```

2) Dimensionally inconsistent units are caught automatically:

```python
apples = xr.DataArray(10).pint.quantify("kg")
oranges = xr.DataArray(200).pint.quantify("cm^3")

apples + oranges
```
```
Out: 
DimensionalityError: Cannot convert from 'kilogram' ([mass]) to 'centimeter ** 3' ([length] ** 3)
```

3) Unit conversions become simple:

```python
walk = xr.DataArray(500).pint.quantify('miles')

walk.pint.to('parsecs')
```
```
Out:
<xarray.DataArray ()>
<Quantity(2.6077643524162074e-11, 'parsec')>
```

4) You can specify that functions should expect certain units, and convert them if needed:

- [ ] TODO this requires pint-xarray #143 to be merged

```python
from pint_xarray import expects

@expects("newton * seconds")
def jpl_trajectory_code(impulse):
    print(f"Received impulse in units of [{impulse.pint.units}]")
 
    # do some rocket science
    ...

lockheed_impulse_value = xr.DataArray(5).pint.quantify("force_pounds * seconds")

jpl_trajectory_code(lockheed_impulse_value)
```
```
Out:
Received impulse in units of [newton * second]
```

In the abstract, tracking units like this is useful in the same way that labelling dimensions with xarray is useful: it helps us avoid errors by relieving us of the burden of remembering arbitrary information about our data.

## Quantifying with pint-xarray

The easiest way to create a unit-aware xarray object is to use the helper package we made: [pint-xarray](https://github.com/xarray-contrib/pint-xarray). 
Once you `import pint_xarray` you can access unit-related functionality via `.pint` on any xarray DataArray or Dataset (this works via [xarray's accessor interface](http://xarray.pydata.org/en/stable/internals/extending-xarray.html)).

Above we have seen examples of quantifying explicitly, where we specify the units in the call to `.quantify()`.
We can do this for multiple variables too, and we can also pass `pint.Unit` instances:
```python
ds = xr.Dataset({'a': 2, 'b': 10})

ds.pint.quantify({'a': 'kg',
                  'b': pint.Unit('moles')})
```
```
Out:
<xarray.Dataset>
Dimensions:  ()
Data variables:
    a        int64 [kg] 2
    b        int64 [mol] 10
```

Alternatively, we can quantify from the object's `.attrs`, automatically reading the metadata which xarray objects carry around.
If nothing is passed to `.quantify()`, it will attempt to parse the `.attrs['units']` entry for each data variable.

This means that for scientific datasets which are stored as files with units in their attributes (which netCDF and Zarr can do for example), using pint with xarray becomes as simple as

```python
import pint_xarray

ds = open_dataset(filepath).pint.quantify()
```

## Dask integration

So xarray can wrap dask arrays, and now it can wrap pint quantities… Can we use both together? Yes!

You can get a unit-aware, dask-backed array either by `.pint.quantify()`-ing a chunked array, or you can `.pint.chunk()` a quantified array.
(If you have dask installed, then `open_dataset(f).pint.quantify()` will already give you a dask-backed, quantified array.)
From there you can `.compute()` the dask-backed objects as normal, and the units will be retained. 

(Under the hood we now have an `xarray.DataArray` wrapping a `pint.Quantity`, which wraps a `dask.array.Array`, which wraps a `numpy.ndarray`.
This "multi-nested duck array" approach can be generalised to include other array libraries (e.g. `scipy.sparse`), but requires [co-ordination](https://github.com/pydata/duck-array-discussion) between the maintainers of the libraries involved.)

## Plotting

- [ ] TODO: Update the plotting page in pint-xarray's docs to not require dequantifying first

## Unit-aware indexes

We would love to be able to promote xarray indexes to pint Quantities, as that would allow you to select data subsets in a unit-aware manner like
```python
da.sel(x=10 * Unit('m'))
```
Unfortunately this will not possible until the ongoing work to extend xarray to support [explicit indexes](https://github.com/pydata/xarray/issues/1603) is complete.

In the meantime pint-xarray offers a workaround. If you tell `.quantify` the units you wish an index to have, it will store those in `.attrs.units` instead.

```python
time = xr.DataArray([0.1, 0.2, 0.3], dims='time')
distance = xr.DataArray(name='distance', 
                        data=[10, 20, 25], 
                        dims=['time'], 
                        coords={'time': time})
distance = distance.pint.quantify({'distance': 'metres', 
                                   'time': 'seconds'})
print(distance.coords['time'].attrs)
```
```
Out: {'units': <Unit('second')>}
```

This allows us to provide conveniently wrapped versions of common xarray methods like `.sel`, so that you can still select subsets of data in a unit-aware fashion like this:

```python
distance.pint.sel(time=200 * pint.Unit('milliseconds'))
```
```
Out: 
<xarray.DataArray 'distance' ()>
<Quantity(20, 'meter')>
Coordinates:
    time     float64 200.0
```
Observe how the `.pint.sel` operation has first converted 200 milliseconds to 0.2 seconds, before finding the distance value that occurs at a time position of 0.2 seconds.

[This wrapping is currently necessary](https://xarray.pydata.org/en/stable/user-guide/duckarrays.html#missing-features) for any operation which needs to be aware of the units of a dimension coordinate of the dataarray, or any xarray operation which relies on an external library (such as calling `scipy` in `.integrate`).

## CF-compliant units for geosciences with cf-xarray

- Different fields tend to have different niche conventions about how certain units are defined.
- By default, pint doesn't understand all the unusual units we use in geosciences
But [pint is customisable](https://pint.readthedocs.io/en/stable/defining.html), and with the help of [cf-xarray](https://github.com/xarray-contrib/cf-xarray) we can teach it about these geoscience-specific units.
- `import cf_xarray.units` (before `import pint_xarray`)
- Put it all together

- [ ] TODO: Example which automatically interprets units of some data from a real climate data store

## Conclusion

- Have a go
- Questions
- Please tell us about any bugs you find, or documentation suggestions you have
- Watch out for unit-aware coordinates later!