import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def make_plots(target: xr.DataArray, predictions: xr.DataArray, fig_title: str, plot_size: float = 5.0, nrows = 8):

    assert predictions.sizes['batch'] == target.sizes['batch']
    batch = predictions.sizes['batch']
    nrows = min(nrows, batch)
    ncols = 4
    fig = plt.figure(figsize=(plot_size * ncols, plot_size * nrows))
    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout()

    choices = np.random.choice(range(nrows), replace=False, size=nrows)

    for n in range(ncols * nrows):

        def add_subplot(data, title):

            ax = fig.add_subplot(nrows, ncols, n + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
            im = ax.imshow(data.sel(batch=choices[n // ncols]), origin='lower')
            plt.colorbar(mappable=im, ax=ax)

        if n % ncols == 0:
            add_subplot(target, "Target")
        if n % ncols == 1:
            add_subplot(predictions, "Predictions")
        if n % ncols == 2:
            add_subplot(predictions - target, "Diff")

    return fig