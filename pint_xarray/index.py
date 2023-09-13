from xarray import Variable
from xarray.core.indexes import Index, PandasIndex

from . import conversion


class PintMetaIndex(Index):
    # TODO: inherit from MetaIndex once that exists
    def __init__(self, *, index, units):
        """create a unit-aware MetaIndex

        Parameters
        ----------
        index : xarray.Index
            The wrapped index object.
        units : mapping of hashable to unit-like
            The units of the indexed coordinates
        """
        self.index = index
        self.units = units

    def create_variables(self, variables=None):
        index_vars = self.index.create_variables(variables)

        index_vars_units = {}
        for name, var in index_vars.items():
            data = conversion.array_attach_units(var.data, self.units[name])
            var_units = Variable(var.dims, data, attrs=var.attrs, encoding=var.encoding)
            index_vars_units[name] = var_units

        return index_vars_units

    @classmethod
    def from_variables(cls, variables, options):
        index = PandasIndex.from_variables(variables)
        units_dict = {index.index.name: options.get("units")}
        return cls(index, units_dict)

    def sel(self, labels):
        converted_labels = conversion.convert_indexer_units(labels, self.units)
        stripped_labels = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in converted_labels.items()
        }

        return self.index.sel(stripped_labels)
