.. currentmodule:: xarray

Creating and saving objects with units
======================================

Attaching units
---------------
.. ipython:: python
    :suppress:

    import xarray as xr

Usually, when loading data from disk we get a :py:class:`Dataset` or
:py:class:`DataArray` with units in attributes:

.. ipython::

    In [1]: ds = xr.Dataset(
       ...:     {
       ...:         "a": (("lon", "lat"), [[11.84, 3.12, 9.7], [7.8, 9.3, 14.72]]),
       ...:         "b": (("lon", "lat"), [[13, 2, 7], [5, 4, 9]], {"units": "m"}),
       ...:     },
       ...:     coords={"lat": [10, 20, 30], "lon": [74, 76]},
       ...: )
       ...: ds

    In [2]: da = ds.b
       ...: da

In order to get :py:class:`pint.Quantity` instances, we can use the
:py:meth:`Dataset.pint.quantify` or :py:meth:`DataArray.pint.quantify` methods:

.. ipython::

    In [3]: ds.pint.quantify()

We can also override the units of a variable:

.. ipython::

    In [4]: ds.pint.quantify(b="km")

    In [5]: da.pint.quantify("degree")

Overriding works even if there is no ``units`` attribute, so we could use this
to attach units to a normal :py:class:`Dataset`:

.. ipython::

    In [6]: temporary_ds = xr.Dataset({"a": ("x", [0, 5, 10])}, coords={"x": [1, 2, 3]})
       ...: temporary_ds.pint.quantify({"a": "m"})

Of course, we could use :py:class:`pint.Unit` instances instead of strings to
specify units, too.

If we wanted to change the units of the data of a :py:class:`DataArray`, we
could do so using the :py:attr:`DataArray.name` attribute:

.. ipython::

    In [7]: da.pint.quantify({da.name: "J", "lat": "degree", "lon": "degree"})

However, `xarray`_ currently doesn't support `units in indexes`_, so the new units were set
as attributes. To really observe the changes the ``quantify`` methods make, we
have to first swap the dimensions:

.. ipython::

    In [8]: ds_with_units = ds.swap_dims({"lon": "x", "lat": "y"}).pint.quantify(
       ...:     {"lat": "degree", "lon": "degree"}
       ...: )
       ...: ds_with_units

    In [9]: da_with_units = da.swap_dims({"lon": "x", "lat": "y"}).pint.quantify(
       ...:     {"lat": "degree", "lon": "degree"}
       ...: )
       ...: da_with_units

Saving with units
-----------------
In order to not lose the units when saving to disk, we first have to call the
:py:meth:`Dataset.pint.dequantify` and :py:meth:`DataArray.pint.dequantify`
methods:

.. ipython::

    In [10]: ds_with_units.pint.dequantify()

    In [11]: da_with_units.pint.dequantify()

This will get the string representation of a :py:class:`pint.Unit` instance and
attach it as a ``units`` attribute. The data of the variable will now be
whatever `pint`_ wrapped.

.. _pint: https://pint.readthedocs.org/en/stable
.. _xarray: https://xarray.pydata.org/en/stable
.. _units in indexes: https://github.com/pydata/xarray/issues/1603
