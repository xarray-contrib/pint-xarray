# Unit-aware arithmetic in xarray, via pint

- [ ] TODO: Add some pictures for interest

TLDR: Xarray now supports unit-aware operations by wrapping [pint arrays](https://pint.readthedocs.io/en/stable/), so your code can automatically track the physical units that your numerical arrays represent, like this:

```python
distance = xr.DataArray([10]).pint.quantify("metres")
time = xr.DataArray([4]).pint.quantify("seconds")
velocity = distance / time
print(velocity.units)
```
```
Out: <Unit('meter / second')>
```

## Units are integral to science

- All quantities in science have units, whether explicitly or implicitly. (And even dimensionless quantities like ratios still technically have units.)
- Getting our units right is tricky
- The consequences of getting them wrong can be huge!
- If we can automatically track them we can potentially eliminate a whole class of possible errors in our scientific work…

## Pint tracks units

- There are a few packages for handling units in python (in particular [unyt](https://github.com/yt-project/unyt) and [astropy.units](https://docs.astropy.org/en/stable/units/)), but for technical reasons we started with Pint.
- These various packages work by providing a numerical array type that acts similarly to a numpy array, and is intended to plug an and replace the raw numpy array (a so-called "duck array type")
- Pint provides the Quantity object, which is a normal numpy array combined with a pint.Unit :

```python
q = np.array([6, 7]) * pint.Unit('metres')
print(repr(q))
Out: <Quantity([6 7], 'meter')>
```
- pint Quantities act like numpy arrays, except that the uni

## Xarray now wraps Pint

- Thanks to the [tireless work](https://github.com/pydata/xarray/issues/3594) of xarray core developer Justus Magin, you can now enjoy this automatic unit-handling in xarray!
- Once you create a unit-aware xarray object you can
- Units are propagated through arithmetic
- Invalid units are caught automatically

```python
da1 = xr.DataArray(1000).pint.quantify("kg")
da2 = xr.DataArray(40).pint.quantify("amperes")
da1 + da2
```
```
Out: DimensionalityError: Cannot convert from 'kilogram' ([mass]) to 'ampere' ([current])
```

- Convert units
- In the abstract, tracking units like this is useful in the same way that labelling dimensions with xarray is useful: it avoid errors by relieving us of the burden of remembering arbitrary information about our data.

## Quantifying with pint-xarray

The easiest way to create a unit-aware xarray object is to use the helper package we made: [pint-xarray](https://github.com/xarray-contrib/pint-xarray). 
Once you `import pint_xarray` you can access unit-related functionality via `.pint` on any xarray DataArray or Dataset (this works via [xarray's accessor interface](http://xarray.pydata.org/en/stable/internals/extending-xarray.html)).

- Quantifying explicitly
- Quantifying from `.attrs`
- This means that for scientific datasets which are stored as files with units in their attributes, using pint with xarray becomes as simple as

```python
import pint_xarray

ds = open_dataset(filepath).pint.quantify()
```

## Dask integration

So xarray can wrap dask arrays, and now it can wrap pint quantities… Can we use both together? Yes!

- You can get a unit-aware, dask-backed array either by `.pint.quantify()`-ing a chunked array, or you can `.pint.chunk()` a quantified array.
- (If you have dask installed, then `open_dataset(f).pint.quantity()` will already give you a dask-backed, quantified array.)
- From there you can `.compute()` the dask-backed objects as normal, and the units will be retained. 

(Under the hood we now have an `xarray.DataArray` wrapping a `pint.Quantity`, which wraps a `dask.array.Array`, which wraps a `numpy.ndarray`.
This "multi-nested duck array" approach can be generalised to include other array libraries (e.g. `scipy.sparse`), but requires [co-ordination](https://github.com/pydata/duck-array-discussion) between the maintainers of the libraries involved.)

## Plotting

- [ ] TODO: Update the plotting page in pint-xarray's docs to not require dequantifying first

## Unit-aware coordinates

We would love to be able to promote xarray coordinates to pint Quantities, as that would allow you to select data subsets in a unit-aware manner like
```python
da.sel(x=10 * Unit('m'))
```
Unfortunately this will not possible until the ongoing work to extend xarray to support [explicit indexes](https://github.com/pydata/xarray/issues/1603) is complete.

In the meantime pint-xarray offers a workaround. If you tell `.quantify` the units you wish a coordinate to have, it will store those in `.attrs.units` instead.

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

This allows us to provide conveniently wrapped versions of common xarray methods like .sel, so that you can still select subsets of data in a unit-aware fashion like this:

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
This wrapping is necessary for any operation which needs to be aware of the units of the coordinates of the dataarray, such as `.integrate`:

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