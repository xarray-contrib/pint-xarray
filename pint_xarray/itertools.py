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
    def common_keys(a, b):
        all_keys = unique(itertools.chain(a.keys(), b.keys()))
        intersection = set(a.keys()).intersection(b.keys())

        return [key for key in all_keys if key in intersection]

    keys = list(reduce(common_keys, mappings))

    for key in keys:
        yield key, tuple(m[key] for m in mappings)
