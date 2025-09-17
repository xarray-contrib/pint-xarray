import cf_xarray as cfxr
import pandas as pd
import xarray as xr


def compress_multindex_in_ds(ds: xr.Dataset) -> xr.Dataset:
    multiindex_coords = [
        coord
        for coord in ds.coords
        if isinstance(ds.coords[coord].to_index(), pd.MultiIndex)
    ]
    if multiindex_coords:
        ds = cfxr.encode_multi_index_as_compress(ds, multiindex_coords)
    return ds


def reset_multindex_in_tree(tree: xr.DataTree) -> xr.DataTree:
    # Create a new tree with MultiIndex reset in each node's Dataset
    def process_node(node: xr.DataTree) -> xr.DataTree:
        node_data = (
            compress_multindex_in_ds(node.dataset) if node.dataset is not None else None
        )

        new_node = xr.DataTree(dataset=node_data, name=node.name)
        for child_name, child in node.children.items():
            new_node[child_name] = process_node(child)
        return new_node

    return process_node(tree)


def save_datatree_compress_multi_index(tree: xr.DataTree, path: str, **kwargs):
    reset_multindex_in_tree(tree).to_zarr(path, **kwargs)


def restore_multindex_in_ds(ds: xr.Dataset) -> xr.Dataset:
    # Only restore if all coordinate variables are present
    if any("compress" in coord.attrs for coord in ds.coords.values()):
        ds = cfxr.decode_compress_to_multi_index(ds)
    return ds


def restore_multindex_in_tree(tree: xr.Dataset) -> xr.Dataset:
    # Recursively restore MultiIndexes in each dataset
    def process_node(node):
        if node.dataset is not None:
            node.dataset = restore_multindex_in_ds(node.dataset)
        for child in node.children.values():
            process_node(child)
        return node

    return process_node(tree)


def open_datatree_decompress_multi_index(path: str, **kwargs):
    return restore_multindex_in_tree(
        xr.open_datatree(path, **kwargs),
    )
