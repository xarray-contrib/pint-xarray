.. currentmodule:: xarray

Converting units
================
.. ipython:: python
    :suppress:

    import xarray as xr

When working with :py:class:`Dataset` or :py:class:`DataArray` objects with
units, we frequently might want to convert the units. Suppose we have:

.. ipython::

    In [1]: ds = xr.Dataset(
       ...:     {"a": ("x", [4, 8, 12, 16])}, coords={"u": ("x", [10, 20, 30, 40])}
       ...: ).astropy.quantify({"a": "m", "u": "s"})
       ...: ds

    In [2]: da = ds.a
       ...: da

To convert the data to different units, we can use the
:py:meth:`Dataset.astropy.to` and :py:meth:`DataArray.astropy.to` methods:

.. ipython::

    In [3]: ds.astropy.to(a="cm", u="ks")

    In [4]: da.astropy.to({da.name: "km", "u": "ms"})
