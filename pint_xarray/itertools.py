import itertools


def separate(predicate, iterable):
    evaluated = ((predicate(el), el) for el in iterable)

    key = lambda x: x[0]
    grouped = itertools.groupby(sorted(evaluated, key=key), key=key)

    groups = {label: [el for _, el in group] for label, group in grouped}

    return groups[False], groups[True]
