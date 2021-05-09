.. currentmodule:: xarray

What's new
==========

0.2 (May 10 2021)
-----------------
- rewrite :py:meth:`Dataset.pint.quantify` and :py:meth:`DataArray.pint.quantify`, to
  use pint's ``UnitRegistry.parse_units`` instead of ``UnitRegistry.parse_expression``
  (:pull:`40`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- ensure the variables which causes the error is explicit if an error occurs in
  :py:meth:`Dataset.pint.quantify` and other methods (:pull:`43`, :pull:`91`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_ and `Justus Magin <https://github.com/keewis>`_.
- refactor the internal conversion functions (:pull:`56`)
  By `Justus Magin <https://github.com/keewis>`_.
- allow converting indexes (except :py:class:`pandas.MultiIndex`) (:pull:`56`)
  By `Justus Magin <https://github.com/keewis>`_.
- document the reason for requiring the ``force_ndarray_like`` or ``force_ndarray``
  options on unit registries (:pull:`59`)
  By `Justus Magin <https://github.com/keewis>`_.
- allow passing a format string to :py:meth:`Dataset.pint.dequantify` and
  :py:meth:`DataArray.pint.dequantify` (:pull:`49`)
  By `Justus Magin <https://github.com/keewis>`_.
- allow converting all data variables in a Dataset to the same units using
  :py:meth:`Dataset.pint.to` (:issue:`45`, :pull:`63`).
  By `Mika Pflüger <https://github.com/mikapfl>`_.
- update format of examples in docstrings (:pull:`64`).
  By `Mika Pflüger <https://github.com/mikapfl>`_.
- implement :py:meth:`Dataset.pint.sel` and :py:meth:`DataArray.pint.sel` (:pull:`60`).
  By `Justus Magin <https://github.com/keewis>`_.
- implement :py:attr:`Dataset.pint.loc` and :py:attr:`DataArray.pint.loc` (:pull:`79`).
  By `Justus Magin <https://github.com/keewis>`_.
- implement :py:meth:`Dataset.pint.drop_sel` and :py:meth:`DataArray.pint.drop_sel` (:pull:`73`).
  By `Justus Magin <https://github.com/keewis>`_.
- implement :py:meth:`Dataset.pint.chunk` and :py:meth:`DataArray.pint.chunk` (:pull:`83`).
  By `Justus Magin <https://github.com/keewis>`_.
- implement :py:meth:`Dataset.pint.reindex`, :py:meth:`Dataset.pint.reindex_like`,
  :py:meth:`DataArray.pint.reindex` and :py:meth:`DataArray.pint.reindex_like` (:pull:`69`).
  By `Justus Magin <https://github.com/keewis>`_.
- implement :py:meth:`Dataset.pint.interp`, :py:meth:`Dataset.pint.interp_like`,
  :py:meth:`DataArray.pint.interp` and :py:meth:`DataArray.pint.interp_like`
  (:pull:`72`, :pull:`76`, :pull:`97`).
  By `Justus Magin <https://github.com/keewis>`_.
- implement :py:meth:`Dataset.pint.ffill`, :py:meth:`Dataset.pint.bfill`,
  :py:meth:`DataArray.pint.ffill` and :py:meth:`DataArray.pint.bfill` (:pull:`78`).
  By `Justus Magin <https://github.com/keewis>`_.
- implement :py:meth:`Dataset.pint.interpolate_na` and :py:meth:`DataArray.pint.interpolate_na` (:pull:`82`).
  By `Justus Magin <https://github.com/keewis>`_.
- expose :py:func:`pint_xarray.setup_registry` as public API (:pull:`89`)
  By `Justus Magin <https://github.com/keewis>`_.

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
