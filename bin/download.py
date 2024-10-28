import argparse
import datetime
import pathlib
import tomllib
from typing import Any, Dict, Optional

import copernicusmarine as cm
import numpy as np
import xarray as xr
from tqdm import tqdm

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument("config", help="Path to configuration file", type=pathlib.Path)
parser.add_argument("output", help="Path to output file", type=pathlib.Path)
parser.add_argument("--overwrite", help="Weather to overwrite existing output file", action='store_true')
args = parser.parse_args()


class DateRange:

    def __init__(self, start: datetime.datetime, stop: datetime.datetime, step: Optional[Dict[str, int]] = None):

        self.start = start
        self.stop = stop
        if step is not None:
            self.step = datetime.timedelta(**step)
        else:
            self.step = datetime.timedelta(days=1)

    def __iter__(self):

        def _date_iterator():

            current = self.start
            while current < self.stop:
                yield current
                current += self.step

        return _date_iterator()


def download_dataset(configs: Dict[str, Any], **kwargs):

    start = configs.get('start')
    stop = configs.get('stop')
    if start is not None and stop is not None:
        dates = list(DateRange(start=start, stop=stop, step=configs.get('step')))
        if configs.get('random'):
            dates = np.random.choice(dates, 
                                    size=configs['nsamples'], 
                                    replace=configs['replace'])
        dates = sorted(dates)
    else:
        dates = None

    def _ds_generator():
        for ds_config in tqdm(configs['datasets']):
            ds = cm.open_dataset(**ds_config, **configs['copernicusmarine'], start_datetime=start, end_datetime=stop, **kwargs)
            if configs['surface_only']:
                ds = ds.sel(depth=ds.depth[0])
                ds = ds.drop_vars('depth')
            if dates is not None:
                ds = ds.sel(time=dates)
            yield ds

    dataset = xr.merge(_ds_generator())
    return dataset


if __name__ == '__main__':

    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Output file path already exists.")

    cm.describe()

    with args.config.open('rb') as file:
        configs = tomllib.load(file)

    dataset = download_dataset(configs=configs)
    dataset.to_netcdf(args.output, engine='h5netcdf')