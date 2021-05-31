import xarray as xr

try:
    from xarray import call_on_dataset
except AttributeError:

    def call_on_dataset(func, obj, name, *args, **kwargs):
        if isinstance(obj, xr.DataArray):
            ds = obj.to_dataset(name=name)
        else:
            ds = obj

        result = func(ds, *args, **kwargs)

        if isinstance(obj, xr.DataArray):
            result = result.get(name).rename(obj.name)

        return result
