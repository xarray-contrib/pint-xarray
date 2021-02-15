.. currentmodule:: xarray

What's new
==========

0.2 (*unreleased*)
------------------
- rewrite :py:meth:`Dataset.pint.quantify` and :py:meth:`DataArray.pint.quantify`,
  to use pint's `parse_units` instead of `parse_expression` (:pull:`40`)
- refactor the internal conversion functions (:pull:`56`)
- allow converting indexes (except :py:class:`pandas.MultiIndex`) (:pull:`56`)
- document the reason for requiring the ``force_ndarray_like`` or ``force_ndarray``
  options on unit registries (:pull:`59`)
- allow passing a format string to :py:meth:`Dataset.pint.dequantify` and
  :py:meth:`DataArray.pint.dequantify` (:pull:`49`)
- allow converting all data variables in a Dataset to the same units using
  :py:meth:`Dataset.pint.to` (:issue:`45`, :pull:`63`).
  By `Mika Pflüger <https://github.com/mikapfl>`_.
- update format of examples in docstrings (:pull:`64`).
  By `Mika Pflüger <https://github.com/mikapfl>`_.

v0.1 (October 26 2020)
----------------------
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
