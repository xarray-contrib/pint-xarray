import xarray as xr

try:
    from xarray import call_on_dataset
except ImportError:

    def call_on_dataset(func, obj, name, *args, **kwargs):
        if isinstance(obj, xr.DataArray):
            ds = obj.to_dataset(name=name)
        else:
            ds = obj

        result = func(ds, *args, **kwargs)

        if isinstance(obj, xr.DataArray) and isinstance(result, xr.Dataset):
            result = result.get(name).rename(obj.name)

        return result
