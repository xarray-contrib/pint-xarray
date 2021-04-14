API reference
=============
This page contains a auto-generated summary of ``pint-xarray``'s API.

.. autosummary::
   :toctree: generated/

   pint_xarray.unit_registry
   pint_xarray.setup_registry

Dataset
-------
.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

   xarray.Dataset.pint.loc

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   xarray.Dataset.pint.quantify
   xarray.Dataset.pint.dequantify
   xarray.Dataset.pint.interp
   xarray.Dataset.pint.interp_like
   xarray.Dataset.pint.reindex
   xarray.Dataset.pint.reindex_like
   xarray.Dataset.pint.drop_sel
   xarray.Dataset.pint.sel
   xarray.Dataset.pint.to
   xarray.Dataset.pint.chunk
   xarray.Dataset.pint.ffill
   xarray.Dataset.pint.bfill
   xarray.Dataset.pint.interpolate_na

DataArray
---------
.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

   xarray.DataArray.pint.loc

   xarray.DataArray.pint.magnitude
   xarray.DataArray.pint.units
   xarray.DataArray.pint.dimensionality
   xarray.DataArray.pint.registry

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   xarray.DataArray.pint.quantify
   xarray.DataArray.pint.dequantify
   xarray.DataArray.pint.interp
   xarray.DataArray.pint.interp_like
   xarray.DataArray.pint.reindex
   xarray.DataArray.pint.reindex_like
   xarray.DataArray.pint.drop_sel
   xarray.DataArray.pint.sel
   xarray.DataArray.pint.to
   xarray.DataArray.pint.chunk
   xarray.DataArray.pint.ffill
   xarray.DataArray.pint.bfill
   xarray.DataArray.pint.interpolate_na

Testing
-------

.. autosummary::
   :toctree: generated/

   pint_xarray.testing.assert_units_equal
