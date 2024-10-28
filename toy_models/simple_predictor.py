import functools
import operator
from typing import Optional, Dict, Any

import haiku as hk
import jax.numpy as jnp
import numpy as np
import xarray as xr
from graphcast import data_utils, model_utils, xarray_jax


class Predictor:

    def __init__(self, climatology: xr.Dataset, mask : xr.DataArray, weights=None,
                 add_grid_features = True, add_mask_feature = True,
                 add_spatial_features = True, add_progress_features = True,
                 mlp_hyperparams: Optional[Dict[str, Any]] = None):

        self.climatology = climatology
        self.mask = mask
        self.num_water_points = mask.sum(dim=('latitude', 'longitude'))
        self.weights = weights
        self.add_spatial_features = add_spatial_features
        self.add_progress_features = add_progress_features
        self.add_grid_features = add_grid_features
        self.add_mask_feature = add_mask_feature
        self.mlp = hk.nets.MLP(**mlp_hyperparams)


    def __call__(self, inputs: xr.Dataset, dummy_target: xr.Dataset) -> xr.Dataset:

        anomaly = self.predict_anomaly(inputs, dummy_target)
        predictions = self.unnormalize(anomaly)
        return predictions


    def loss(self, inputs: xr.Dataset, targets: xr.Dataset) -> xr.DataArray:

        outputs = self.predict_anomaly(inputs, targets)
        targets = self.normalize(targets)
        loss = ((targets - outputs) ** 2).where(self.mask, 0.0)
        loss = loss.sum([d for d in loss.dims if d != 'batch'], skipna=False) / self.num_water_points
        if self.weights is None:
            return functools.reduce(operator.add, (var for var in loss.data_vars.values()))
        else:
            return functools.reduce(operator.add, (self.weights[name] * var for (name, var) in loss.data_vars.items()))



    def predict_anomaly(self, inputs: xr.Dataset, dummy_targets: xr.Dataset) -> xr.Dataset:

        inputs = self.normalize(inputs)
        inputs = self.preprocess(inputs)
        
        batch_size = inputs.sizes['batch']
        latitude_size = inputs.sizes['latitude']
        longitude_size = inputs.sizes['longitude']
        input_channels = inputs.sizes['channels']

        inputs_data = xarray_jax.jax_data(inputs)
        inputs_data = jnp.reshape(inputs_data, (-1, input_channels))
        # TODO: input data should be checked to avoid nan poisoning
        # inputs_data_nonans = jnp.where(jnp.isnan(inputs_data), 0.0, inputs_data)
        # outputs_data = self.mlp(inputs_data_nonans)
        # mask = jax.vmap(lambda xs: jnp.isnan(xs).any(), in_axes=0)(inputs_data)
        # mask = jnp.expand_dims(mask, axis=-1)
        # outputs_data = jnp.where(mask, jnp.nan, outputs_data)
        # TODO: The following assumes that one is trying to predict a single variable.
        outputs_data = self.mlp(inputs_data)
        outputs_data = jnp.reshape(outputs_data, (batch_size, latitude_size, longitude_size))
        outputs_data_array = xarray_jax.DataArray(data=outputs_data, dims=('batch', 'latitude', 'longitude'))
        target_name = list(dummy_targets.data_vars.keys())[0]
        outputs = xarray_jax.Dataset(data_vars={target_name: outputs_data_array}, coords=dummy_targets.coords)
        # Once the code can handle multiple target variables, maybe one could reuse `model_utils.stacked_to_dataset`? Ex.
        # outputs = model_utils.stacked_to_dataset(outputs_data_array, dummy_targets, preserved_dims=('batch', 'latitude', 'longitude'))

        return outputs


    def preprocess(self, inputs: xr.Dataset) -> xr.DataArray:

        features = []

        if self.add_progress_features:
            spatial_features = self.get_spatial_features(inputs)
            features.append(spatial_features)

        if self.add_progress_features:
            # Note `datetime.astype("datetime64[s]").astype(np.int64)`
            # does not work as xarrays always cast dates into nanoseconds!
            progress_features = self.get_year_progress_features(inputs)
            features.append(progress_features)

        if self.add_grid_features:
            grid_features = model_utils.dataset_to_stacked(dataset=inputs,
                                                           preserved_dims=('batch', 'latitude', 'longitude'))
            features.append(grid_features)

        if self.add_mask_feature:
            features.append(self.mask.astype(np.float32))

        inputs = xr.concat(features, dim='channels')

        return inputs


    def get_spatial_features(self, inputs: xr.Dataset) -> xr.DataArray:

        theta, phi = model_utils.lat_lon_deg_to_spherical(inputs.latitude, inputs.longitude)
        phi_mesh, theta_mesh = np.meshgrid(theta, phi)
        spatial_features = np.stack([np.cos(theta_mesh), np.sin(phi_mesh), np.cos(phi_mesh)], axis=-1)
        spatial_features = np.tile(spatial_features, reps=(inputs.sizes['batch'], 1, 1, 1))
        return xr.DataArray(spatial_features, dims=('batch', 'latitude', 'longitude', 'channels'))

    def get_year_progress_features(self, inputs: xr.Dataset) -> xr.DataArray:

        # Note `time.astype("datetime64[s]").astype(np.int64)`
        # does not work as xarrays always cast dates into nanoseconds!
        time = inputs.time.data
        year_progress_rad = 2 * np.pi * data_utils.get_year_progress(time.astype("datetime64[s]").astype(np.int64))
        year_progress_rad = np.expand_dims(year_progress_rad, axis=(1, 2))
        year_progress_rad = np.tile(year_progress_rad, reps=(1, inputs.sizes['latitude'], inputs.sizes['longitude']))
        progress_features = np.stack([np.cos(year_progress_rad), np.sin(year_progress_rad)], axis=-1)
        return xr.DataArray(progress_features, dims=('batch', 'latitude', 'longitude', 'channels'))


    def get_scaling_factors(self, ds: xr.Dataset) -> xr.Dataset:

        month = ds.time.data.astype('datetime64[M]').astype(int) % 12
        scaling_factors = self.climatology.isel(time=month)
        scaling_factors = scaling_factors.rename_dims(time='batch')
        return scaling_factors


    def normalize(self, ds:xr.Dataset) -> xr.Dataset:

        anomaly = xr.Dataset(data_vars={}, coords=ds.coords)
        scaling_factors = self.get_scaling_factors(ds)
        for variable in ds.data_vars.keys():
            anomaly[variable] = (ds[variable] - scaling_factors[variable + '_avg']) / scaling_factors[variable + '_std']
        anomaly = anomaly.where(self.mask, 0.0)
        return anomaly


    def unnormalize(self, anomaly:xr.Dataset) -> xr.Dataset:

        ds = xr.Dataset(data_vars={}, coords=anomaly.coords)
        scaling_factors = self.get_scaling_factors(anomaly)
        for variable in anomaly.data_vars.keys():
            ds[variable] = scaling_factors[variable + '_avg'] + anomaly[variable] * scaling_factors[variable + '_std']
        ds = ds.where(self.mask, jnp.nan)
        return ds
