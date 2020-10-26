What's new
==========

0.1 (October 26 2020)
---------------------
- add initial draft of documentation (:pull:`13`, :pull:`20`)
- implement :py:meth:`DataArray.pint.to` and :py:meth:`Dataset.pint.to`
  (:pull:`11`)
- rewrite :py:meth:`DataArray.pint.quantify`,
  :py:meth:`Dataset.pint.quantify`, :py:meth:`DataArray.pint.dequantify` and
  :py:meth:`Dataset.pint.dequantify` (:pull:`17`)
- expose :py:func:`pint_xarray.testing.assert_units_equal` as public API (:pull:`24`)
- fix the :py:attr:`DataArray.pint.units`, :py:attr:`DataArray.pint.magnitude`
  and :py:attr:`DataArray.pint.dimensionality` properties and add docstrings for
  all three. (:pull:`31`)
- use ``pint``'s application registry as a module-global registry (:pull:`32`)
