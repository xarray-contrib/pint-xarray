from xarray.core.indexes import Index

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

    # don't need `from_variables`: we're always *wrapping* an existing index

    def sel(self, labels):
        converted_labels = conversion.convert_indexer_units(labels, self.units)
        stripped_labels = {
            name: conversion.strip_indexer_units(indexer)
            for name, indexer in converted_labels.items()
        }

        return self.index.sel(stripped_labels)
