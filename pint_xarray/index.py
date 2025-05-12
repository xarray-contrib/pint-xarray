from xarray import Variable
from xarray.core.indexes import Index, PandasIndex

from pint_xarray import conversion


class PintIndex(Index):
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

    def _replace(self, new_index):
        return self.__class__(index=new_index, units=self.units)

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
        if len(variables) != 1:
            raise ValueError("can only create a default index from single variables")

        units = options.pop("units", None)
        index = PandasIndex.from_variables(variables, options=options)
        return cls(index=index, units={index.index.name: units})

    @classmethod
    def concat(cls, indexes, dim, positions):
        raise NotImplementedError()

    @classmethod
    def stack(cls, variables, dim):
        raise NotImplementedError()

    def unstack(self):
        raise NotImplementedError()

    def sel(self, labels):
        converted_labels = conversion.convert_indexer_units(labels, self.units)
        stripped_labels = conversion.strip_indexer_units(converted_labels)

        return self.index.sel(stripped_labels)

    def isel(self, indexers):
        subset = self.index.isel(indexers)
        if subset is None:
            return None

        return self._replace(subset)

    def join(self, other, how="inner"):
        raise NotImplementedError()

    def reindex_like(self, other):
        raise NotImplementedError()

    def equals(self, other):
        if not isinstance(other, PintIndex):
            return False

        # for now we require exactly matching units to avoid the potentially expensive conversion
        if self.units != other.units:
            return False

        # last to avoid the potentially expensive comparison
        return self.index.equals(other.index)

    def roll(self, shifts):
        return self._replace(self.index.roll(shifts))

    def rename(self, name_dict, dims_dict):
        return self._replace(self.index.rename(name_dict, dims_dict))

    def __getitem__(self, indexer):
        return self._replace(self.index[indexer])

    def _repr_inline_(self, max_width):
        return f"{self.__class__.__name__}({self.index.__class__.__name__})"
