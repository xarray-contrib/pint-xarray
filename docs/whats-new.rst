.. currentmodule:: xarray

What's new
==========
0.1 (17 Jul 2025)
-----------------

- Added `equivalencies` parameter to :py:meth:`DataArray.astropy.to` and :py:meth:`Dataset.astropy.to`.
- Removed `registry` parameter from :py:meth:`DataArray.astropy.quantify` and :py:meth:`Dataset.astropy.quantify`.
- Migrated ``pint.Quantity`` usage to :py:class:`astropy.units.Quantity`, ``pint.UnitRegistry`` to :py:mod:`astropy.units`, ``pint.UnitRegistry.formatter`` to :py:mod:`astropy.units.format` (:pull:`1`)

  Notable behavioural differences include:

  * Unit registry instance is managed at module scope instead of local scope.
  * Multiplying by an array by an :py:class:`astropy.units.Unit` coerses to :py:class:`astropy.units.Quantity` of float64. Explicit construction required to use other numpy dtypes.
  * :py:class:`astropy.units.Quantity` uses ``unit`` and ``value`` members instead of ``units`` and ``magnitude``.
  * ``Unit()`` is not an instance of :py:class:`astropy.units.Unit`, only :py:class:`astropy.units.UnitBase`.
  * Different format literals, see `built-in formats <https://docs.astropy.org/en/stable/units/format.html#built-in-formats>`_.

  By `Callan Gray <https://github.com/calgray>`_.
