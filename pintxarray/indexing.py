import pandas as pd
from pandas.core.base import IndexOpsMixin

from pint import Quantity, Unit


# TODO does pandas supply some kind of abstract base class for indexer objects?
class QuantityIndex(IndexOpsMixin):
    def __init__(self, index=pd.Index, units=None, unit_registry=None,
                 decode_cf=False):
        self._index = index
        self._units = units

    @property
    def units(self):
        ...

    @property
    def magnitude(self):
        ...

    def dequantify(self):
        ...

    def to(self, units):
        ...

    def to_base_units(self):
        ...
