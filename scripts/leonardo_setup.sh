#! /usr/bin/env bash

ROOT=$(git rev-parse --show-toplevel)
PACKAGES=(copernicusmarine h5netcdf zarr optax tqdm torch tensorboard)

# Change working directory to project root
cd "${ROOT}" || exit

# Load prerequisite modules, mainly Python and CUDA
# TODO: it should probably be downgraded to cineca-ai/3.0.1 because of this:
# https://github.com/google/jax/issues/15384
# But then one should find a functioning version of jax
# and jaxlib to be used with graphcast...
# Seems more reasonable to not have a working jax.profiler.trace
module load profile/deeplrn cineca-ai/4.1.1

# Create Python venv
python -m venv --system-site-packages --upgrade-deps venv || exit

# Activate Python venv, afterwards download packages (needs internet connection)
source "${ROOT}/venv/bin/activate"

JAX_RELEASE_URL=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip download --dest=pkg_cache --find-links=${JAX_RELEASE_URL} ${PACKAGES[@]} "graphcast[interactive] @ git+https://github.com/stefanocampanella/graphcast.git@develop" || exit

# Install packages on a GPU node
ACCOUNT=OGS23_PRACE_IT_0
PARTITION=boost_usr_prod
TIME=10
COMMAND="python -m pip install ${PACKAGES[@]} graphcast[interactive] --no-build-isolation --no-index --find-links pkg_cache"
srun --account ${ACCOUNT} --partition ${PARTITION} --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=${TIME} \
 ${COMMAND} || exit

deactivate
