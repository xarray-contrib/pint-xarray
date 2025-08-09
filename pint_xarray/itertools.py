import itertools
from functools import reduce


def separate(predicate, iterable):
    evaluated = ((predicate(el), el) for el in iterable)

    key = lambda x: x[0]
    grouped = itertools.groupby(sorted(evaluated, key=key), key=key)

    groups = {label: [el for _, el in group] for label, group in grouped}

    return groups[False], groups[True]


def unique(iterable):
    return list(dict.fromkeys(iterable))


def zip_mappings(*mappings):
    keys = list(reduce(lambda x, y: set(x.keys()).intersection(y.keys()), mappings))

    for key in keys:
        yield key, tuple(m[key] for m in mappings)
