from itertools import zip_longest

import numpy as np
from xarray.core.options import OPTIONS


# vendored from xarray.core.formatting
def maybe_truncate(obj, maxlen=500):
    s = str(obj)
    if len(s) > maxlen:
        s = s[: (maxlen - 3)] + "..."
    return s


# vendored from xarray.core.formatting
def pretty_print(x, numchars: int):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = maybe_truncate(x, numchars)
    return s + " " * max(numchars - len(s), 0)


# vendored from xarray.core.formatting
def _get_indexer_at_least_n_items(shape, n_desired, from_end):
    assert 0 < n_desired <= np.prod(shape)
    cum_items = np.cumprod(shape[::-1])
    n_steps = np.argmax(cum_items >= n_desired)
    stop = int(np.ceil(float(n_desired) / np.r_[1, cum_items][n_steps]))
    indexer = (
        ((-1 if from_end else 0),) * (len(shape) - 1 - n_steps)
        + ((slice(-stop, None) if from_end else slice(stop)),)
        + (slice(None),) * n_steps
    )
    return indexer


# vendored from xarray.core.formatting
def first_n_items(array, n_desired):
    """Returns the first n_desired items of an array"""
    # Unfortunately, we can't just do array.flat[:n_desired] here because it
    # might not be a numpy.ndarray. Moreover, access to elements of the array
    # could be very expensive (e.g. if it's only available over DAP), so go out
    # of our way to get them in a single call to __getitem__ using only slices.
    if n_desired < 1:
        raise ValueError("must request at least one item")

    if array.size == 0:
        # work around for https://github.com/numpy/numpy/issues/5195
        return []

    if n_desired < array.size:
        indexer = _get_indexer_at_least_n_items(array.shape, n_desired, from_end=False)
        array = array[indexer]
    return np.asarray(array).flat[:n_desired]


# vendored from xarray.core.formatting
def last_n_items(array, n_desired):
    """Returns the last n_desired items of an array"""
    # Unfortunately, we can't just do array.flat[-n_desired:] here because it
    # might not be a numpy.ndarray. Moreover, access to elements of the array
    # could be very expensive (e.g. if it's only available over DAP), so go out
    # of our way to get them in a single call to __getitem__ using only slices.
    if (n_desired == 0) or (array.size == 0):
        return []

    if n_desired < array.size:
        indexer = _get_indexer_at_least_n_items(array.shape, n_desired, from_end=True)
        array = array[indexer]
    return np.asarray(array).flat[-n_desired:]


# based on xarray.core.formatting.format_item
def format_item(x, quote_strings=True):
    """Returns a succinct summary of an object as a string"""
    if isinstance(x, (str, bytes)):
        return repr(x) if quote_strings else x
    elif isinstance(x, float):
        return f"{x:.4}"
    elif hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating):
        return f"{x.item():.4}"
    else:
        return str(x)


# based on xarray.core.formatting.format_item
def format_items(x):
    """Returns a succinct summaries of all items in a sequence as strings"""
    x = np.asarray(x)
    formatted = [format_item(xi) for xi in x]
    return formatted


def summarize_attr(key, value, col_width=None):
    """Summary for __repr__ - use ``X.attrs[key]`` for full value."""
    # Indent key and add ':', then right-pad if col_width is not None
    k_str = f"    {key}:"
    if col_width is not None:
        k_str = pretty_print(k_str, col_width)
    # Replace tabs and newlines, so we print on one line in known width
    v_str = str(value).replace("\t", "\\t").replace("\n", "\\n")
    # Finally, truncate to the desired display width
    return maybe_truncate(f"{k_str} {v_str}", OPTIONS["display_width"])


# adapted from xarray.core.formatting
def _diff_mapping_repr(a_mapping, b_mapping, title, summarizer, col_width=None):
    def extra_items_repr(extra_keys, mapping, ab_side):
        extra_repr = [summarizer(k, mapping[k], col_width) for k in extra_keys]
        if extra_repr:
            header = f"{title} only on the {ab_side} object:"
            return [header] + extra_repr
        else:
            return []

    a_keys = set(a_mapping)
    b_keys = set(b_mapping)

    summary = []

    diff_items = []

    for k in a_keys & b_keys:
        compatible = a_mapping[k] == b_mapping[k]
        if not compatible:
            temp = [
                summarizer(k, vars[k], col_width) for vars in (a_mapping, b_mapping)
            ]

            diff_items += [ab_side + s[1:] for ab_side, s in zip(("L", "R"), temp)]

    if diff_items:
        summary += [f"Differing {title.lower()}:"] + diff_items

    summary += extra_items_repr(a_keys - b_keys, a_mapping, "left")
    summary += extra_items_repr(b_keys - a_keys, b_mapping, "right")

    return "\n".join(summary)


# vendored from xarray.core.formatting
def format_array_flat(array, max_width: int):
    """Return a formatted string for as many items in the flattened version of
    array that will fit within max_width characters.
    """
    # every item will take up at least two characters, but we always want to
    # print at least first and last items
    max_possibly_relevant = min(
        max(array.size, 1), max(int(np.ceil(max_width / 2.0)), 2)
    )
    relevant_front_items = format_items(
        first_n_items(array, (max_possibly_relevant + 1) // 2)
    )
    relevant_back_items = format_items(last_n_items(array, max_possibly_relevant // 2))
    # interleave relevant front and back items:
    #     [a, b, c] and [y, z] -> [a, z, b, y, c]
    relevant_items = sum(
        zip_longest(relevant_front_items, reversed(relevant_back_items)), ()
    )[:max_possibly_relevant]

    cum_len = np.cumsum([len(s) + 1 for s in relevant_items]) - 1
    if (array.size > 2) and (
        (max_possibly_relevant < array.size) or (cum_len > max_width).any()
    ):
        padding = " ... "
        count = min(
            array.size, max(np.argmax(cum_len + len(padding) - 1 > max_width), 2)
        )
    else:
        count = array.size
        padding = "" if (count <= 1) else " "

    num_front = (count + 1) // 2
    num_back = count - num_front
    # note that num_back is 0 <--> array.size is 0 or 1
    #                         <--> relevant_back_items is []
    pprint_str = "".join(
        [
            " ".join(relevant_front_items[:num_front]),
            padding,
            " ".join(relevant_back_items[-num_back:]),
        ]
    )

    # As a final check, if it's still too long even with the limit in values,
    # replace the end with an ellipsis
    # NB: this will still returns a full 3-character ellipsis when max_width < 3
    if len(pprint_str) > max_width:
        pprint_str = pprint_str[: max(max_width - 3, 0)] + "..."

    return pprint_str


def inline_repr(quantity, max_width):
    magnitude = quantity.magnitude
    units = quantity.units

    units_repr = f"{units:~P}"
    if isinstance(magnitude, np.ndarray):
        data_repr = format_array_flat(magnitude, max_width - len(units_repr) - 3)
    else:
        data_repr = maybe_truncate(repr(magnitude), max_width - len(units_repr) - 3)

    return f"[{units_repr}] {data_repr}"
