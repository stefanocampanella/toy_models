from typing import Sequence, Union

import jax
import torch.utils.data
import xarray as xr
from graphcast import xarray_jax


def dataset2dataset_jax(data: xr.Dataset, *args, **kwargs) -> xr.Dataset:

    def _data_array(var: xr.DataArray, name=None, jax_coords=None):

        return xarray_jax.DataArray(jax.device_put(var.data, *args, **kwargs),
                                    coords=var.coords,
                                    dims=var.dims,
                                    name=name,
                                    attrs=var.attrs,
                                    jax_coords=jax_coords)

    data_variables_names = set(data.variables.keys()) - set(data.coords.keys())
    variables = {name: _data_array(data[name], name=name) for name in data_variables_names}
    return xarray_jax.Dataset(variables, coords=data.coords, attrs=data.attrs)


def dataset_jax2dataset(data: xr.Dataset) -> xr.Dataset:

    data_unwrapped = {}
    for key in data.data_vars.keys():
        variable = data[key]
        data_unwrapped[key] = (variable.dims, xarray_jax.unwrap_data(variable))
    return xr.Dataset(data_vars=data_unwrapped, coords=data.coords, attrs=data.attrs)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data: xr.Dataset, mask, input_variables: Sequence[str], target_variables: Sequence[str]):

        self.input_variables = input_variables
        self.target_variables = target_variables
        self.mask = mask
        data = data.drop_vars(v for v in data.data_vars.keys() if (v not in input_variables) and (v not in target_variables))
        data = data.expand_dims(dim='batch', axis=0)
        data = data.assign_coords(time=data.time.expand_dims(dim='batch', axis=0))
        self.data = data


    def __len__(self):

        return self.data.sizes['time']


    def __getitem__(self, idx):

        item = self.data.isel(time=idx).compute()
        item = item.where(self.mask, 0.0)
        item = dataset2dataset_jax(item)
        inputs = item[self.input_variables]
        targets = item[self.target_variables]
        return inputs, targets


def default_collate_fn(batch):

    if len(batch) > 1:
        inputs, targets = map(lambda datasets: xr.concat(datasets, dim='batch'), zip(*batch))
    else:
        inputs, targets = batch[0]
    return inputs, targets


DatasetOrSubset = Union[Dataset,torch.utils.data.Subset]

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset: DatasetOrSubset, batch_size=None, num_samples=None, replacement=False, collate_fn=default_collate_fn, **kwargs):

        sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement, num_samples=num_samples, generator=kwargs.get('generator'))
        batch_sampler = torch.utils.data.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        kwargs.update({'shuffle': None, 'drop_last': None, 'sampler': None, 'batch_sampler': batch_sampler})
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)