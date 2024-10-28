import argparse
import pathlib

import haiku as hk
import jax
import optax
import torch.utils.data
import xarray as xr
from graphcast import xarray_jax
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from toy_models.simple_predictor import Predictor
from toy_models.utils.data import DataLoader, Dataset
from toy_models.utils.plots import make_plots

def drop_suffix(s: str) -> str:

    if s.endswith('_avg'):
        key, _ = s.split('_avg')
    elif s.endswith('_std'):
        key, _ = s.split('_std')
    else:
        key = s
    return key


parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('dataset', type=pathlib.Path)
parser.add_argument('climatology', type=pathlib.Path)
parser.add_argument('bathymetry', type=pathlib.Path)
parser.add_argument('--plot-every-n-iterations', default=100, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--jit', action="store_true")
args = parser.parse_args()

adria = xr.load_dataset(args.dataset, engine='h5netcdf')
clima = xr.load_dataset(args.climatology, engine='h5netcdf')
bathy = xr.load_dataset(args.bathymetry, engine='h5netcdf')
clima = clima[[key for key in clima.data_vars.keys() if drop_suffix(key) in adria.data_vars.keys()]]
adria, clima, bathy = xr.align(adria, clima, bathy, exclude=('depth', 'time'))
mask = bathy.mask.isel(depth=0).astype(bool)
for (name, var) in clima.data_vars.items():
    if name.endswith('_avg'):
        clima[name] = clima[name].where(mask, 0.0)
    else:
        clima[name] = clima[name].where(mask, 1.0)

input_variables = ['po4', 'no3', 'so', 'thetao']
target_variables = ['chl']
torch_generator = torch.Generator().manual_seed(args.seed)
dataset = Dataset(data=adria, mask=mask, input_variables=input_variables, target_variables=target_variables)
train_dataset, validation_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[0.8, 0.2],
                                                                  generator=torch_generator)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, generator=torch_generator)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=32, generator=torch_generator)

mlp_hyperparams = {
    'output_sizes': [64, 64, 64, len(target_variables)],
    'with_bias': True,
    }

@hk.without_apply_rng
@hk.transform
def predictor(inputs: xr.Dataset, dummy_targets: xr.Dataset) -> xr.Dataset:

    predictor = Predictor(climatology=clima, mask=mask, mlp_hyperparams=mlp_hyperparams)
    return predictor(inputs, dummy_targets)

@hk.without_apply_rng
@hk.transform
def loss(inputs: xr.Dataset, dummy_targets: xr.Dataset):

    predictor = Predictor(climatology=clima, mask=mask, mlp_hyperparams=mlp_hyperparams)
    loss = predictor.loss(inputs, dummy_targets)
    return xarray_jax.unwrap_data(loss.mean(), require_jax=True)

loss_fn = loss.apply
grads_fn = jax.value_and_grad(loss_fn)
if args.jit:
    loss_fn = jax.jit(loss_fn)
    grads_fn = jax.jit(grads_fn)

key = jax.random.key(args.seed)
sample_inputs, sample_targets = next(iter(train_dataloader))
params = predictor.init(key, sample_inputs, sample_targets)

schedule = optax.schedules.warmup_cosine_decay_schedule(init_value=0.0, peak_value=0.1, end_value=1e-3, warmup_steps=32, decay_steps=128)
optimizer = optax.chain(optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=0.1),  optax.clip_by_global_norm(1.0))
optimizer_state = optimizer.init(params)
nepochs = 32

# Train the model
writer = SummaryWriter()
global_step = 0
for n in range(nepochs):
    decorated_dataloader = tqdm(train_dataloader)
    for inputs, targets in decorated_dataloader:
        # Compute and log the loss and its gradients
        train_loss, grads = grads_fn(params, inputs, targets)
        writer.add_scalar('Loss/train', train_loss.item(), global_step)
        # Update the model parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        # Compute and log the validation loss, then save the best model
        validation_inputs, _ = next(iter(validation_dataloader))
        validation_loss = loss_fn(params, *next(iter(validation_dataloader)))
        decorated_dataloader.set_description_str(f"Epoch {n}")
        writer.add_scalar('Loss/validation', validation_loss.item(), global_step)
        global_step += 1
        if global_step % args.plot_every_n_iterations == 0:
            predictions = predictor.apply(params, sample_inputs, sample_targets)
            month = sample_targets.time.data.astype('datetime64[M]').astype(int) % 12
            fig = make_plots(target=sample_targets['chl'], predictions=predictions['chl'],
                             fig_title=f"Epoch: {n}", nrows=3)
            writer.add_figure("Samples", fig, global_step)
