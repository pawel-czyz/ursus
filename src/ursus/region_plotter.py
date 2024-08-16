import dataclasses
import pyBigWig

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@dataclasses.dataclass
class RegionSettings:
    chromosome: int | str
    start: int
    end: int
    bin_length: int


def _chromosome_name(settings: RegionSettings, prefix: str) -> str:
    return prefix + str(settings.chromosome)


def get_binned_array(
        fname: str,
        settings: RegionSettings,
        chromosome_prefix: str,
    ):
    if settings.end <= settings.start:
        raise ValueError("End can't be less than start.")

    if settings.bin_length < 1:
        raise ValueError("Bin length can't be strictly less than 1.")
    
    n_intervals = (settings.end - settings.start) // settings.bin_length
    
    bw = pyBigWig.open(fname)
    
    a = np.asarray(
        bw.values(_chromosome_name(settings=settings, prefix=chromosome_prefix), settings.start, settings.end),
        dtype=np.float64,
    )

    binned = []
    for i in range(n_intervals):
        val = a[i * settings.bin_length : (i+1) * settings.bin_length].mean()
        binned.append(val)

    binned = np.asarray(binned, dtype=np.float64)
    return binned


def get_multiarray(files: list[str], settings: RegionSettings, chromosome_prefix: str) -> np.ndarray:
    arrs = [
        get_binned_array(fname, settings=settings, chromosome_prefix=chromosome_prefix)
        for fname in files
    ]
    big_arr = np.stack(arrs)
    
    normalized = big_arr / big_arr.sum(axis=0, keepdims=True)
    return normalized



def plot_arrays(
    arrays: list[np.ndarray],
    labels: list[str],
    settings: RegionSettings,
    cmap: str = "Greys",
    dpi: int = 450,
) -> tuple:
    """Returns the figure and an array with axes."""

    if len(arrays) != len(labels):
        raise ValueError("Arrays and labels have to have equal length.")
    
    n_bins = arrays[0].shape[1]
    for arr in arrays:
        if arr.shape[1] != n_bins:
            raise ValueError(f"Arrays have different number of bins. Expected {n_bins}, obtained {arr.shape[1]}.")

    n = len(arrays)

    fig, axs = plt.subplots(n, 1, sharex=True, sharey=True, dpi=dpi)

    # Plot color maps
    for ax, arr, label in zip(axs, arrays, labels):
        sns.heatmap(arr, cmap=cmap, cbar=False, ax=ax)
        ax.set_ylabel(label)
        ax.set_yticks([])

    # Annotate X axis
    ax = axs[-1]

    start = settings.start
    end = settings.end

    us = np.linspace(start / 1e6, end/1e6, 11)
    us = [str(int(u)) + " Mb" for u in us]
    ax.set_xticks(np.linspace(0, n_bins, 11), us)

    fig.suptitle(f"Chromosome {settings.chromosome}")
    fig.tight_layout()

    return fig, axs


