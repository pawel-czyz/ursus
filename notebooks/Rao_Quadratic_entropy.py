# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import gmean


def rao_quadratic_entropy(p):
    """
    Calculate Rao's Quadratic Entropy for a probability vector p
    with distance metric d_{ij} given by Euclidean distances from a circle.

    Parameters:
        p: probability vector over n classes. Should sum to 1.

    Returns:
        float: Rao's Quadratic Entropy H_Q
    """
    n = p.shape[0]

    indices = jnp.arange(n)

    # xs = jnp.cos(2 * jnp.pi * indices / n)
    # ys = jnp.sin(2 * jnp.pi * indices / n)
    # D = jnp.sqrt(jnp.square(xs[:, None] - xs[None, :]) + jnp.square(ys[:, None] - ys[None, :]))

    angles = 2 * jnp.pi * indices / n
    diff_angle = jnp.abs(angles[:, None] - angles[None, :])

    D = jnp.minimum(diff_angle, 2 * jnp.pi - diff_angle)

    p0 = jnp.full_like(p, fill_value=1.0 / n)

    H_Q = jnp.einsum("i,j,ij->", p, p, D)
    H_0 = jnp.einsum("i,j,ij->", p0, p0, D)

    return H_Q / H_0


def get_quadratic_entropy(arr):
    # Shape (n_timepoints, n_gene_bins)
    return jax.vmap(rao_quadratic_entropy, in_axes=1)(arr)


def get_entropy(arr, jitter=1e-12):
    # disp = arr.max(axis=0) - arr.min(axis=0)
    # return disp

    arr = arr + jitter
    arr = arr / arr.sum(axis=0, keepdims=True)
    return (-arr * jnp.log2(arr)).sum(axis=0)


def get_dispersion(arr, jitter=1e-12):
    return np.std(arr, axis=0)


# +
vecs = [
    [
        jnp.array([1.0, 0, 0, 0, 0]),
    ],
    #
    [
        jnp.array([0.5, 0.5, 0, 0, 0]),
        jnp.array([0, 0.5, 0.5, 0, 0]),
        jnp.array([0, 0, 0.5, 0.5, 0]),
    ],
    #
    [
        jnp.array([0.5, 0, 0.5, 0, 0]),
        jnp.array([0.5, 0, 0, 0.5, 0]),
        jnp.array([0, 0.5, 0, 0, 0.5]),
    ],
    [
        jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    ],
]

for vv in vecs:
    for vec in vv:
        print(vec, "\t\t", rao_quadratic_entropy(vec))
    print("\n")


# -


@jax.jit
def f(p):
    last = 1.0 - jnp.sum(p)
    p_ = jnp.concatenate([p, jnp.asarray([last])])
    return -rao_quadratic_entropy(p_)


from jax.scipy.optimize import minimize

minimize(f, jnp.ones(4) / 10, method="BFGS")

# +
import ursus.region_plotter as rp


def get_multiarrays(
    files: list[str], settings, chromosome_prefix: str
) -> tuple[np.ndarray, np.ndarray]:
    arrs = [
        rp.get_binned_array(
            fname, settings=settings, chromosome_prefix=chromosome_prefix
        )
        for fname in files
    ]
    big_arr = np.stack(arrs)

    valid = big_arr.sum(axis=0) > 0

    # big_arr[-2, :] = 0.5 * (big_arr[-2, :] + big_arr[-1, :])
    # big_arr = big_arr[:-1, :]

    normalized = big_arr / big_arr.sum(axis=0, keepdims=True)
    return normalized, valid


DM_FILES = [
    "1_i701_i501_S1__uniq_rmdup_10000.bw",
    "5_i702_i501_S5__uniq_rmdup_10000.bw",
    "9_i703_i501_S9__uniq_rmdup_10000.bw",
    "13_i704_i501_S13__uniq_rmdup_10000.bw",
    "17_i705_i501_S17_uniq_rmdup_10000.bw",
]
DM_FILES = [f"data/202407/bigwig/DM/{fl}" for fl in DM_FILES]

HSR_FILES = [
    "1_i701_i501_S1__uniq_rmdup_10000.bw",
    "5_i702_i501_S5__uniq_rmdup_10000.bw",
    "9_i703_i501_S9__uniq_rmdup_10000.bw",
    "13_i704_i501_S13__uniq_rmdup_10000.bw",
    "17_i705_i501_S17__uniq_rmdup_10000.bw",
]
HSR_FILES = [f"data/202407/bigwig/HSR/{fl}" for fl in HSR_FILES]

RPE1_FILES = [
    "2_i701_i502_S2__uniq_rmdup_10000.bw",
    "6_i702_i502_S6__uniq_rmdup_10000.bw",
    "10_i703_i502_S10__uniq_rmdup_10000.bw",
    "14_i704_i502_S14__uniq_rmdup_10000.bw",
    "18_i705_i502_S18__uniq_rmdup_10000.bw",
]
RPE1_FILES = [f"data/202407/bigwig/RPE-1/{fl}" for fl in RPE1_FILES]


def get_arrays_from_settings(settings, option):
    if option == "DM":
        array = get_multiarrays(
            files=DM_FILES, settings=settings, chromosome_prefix="chr"
        )
    elif option == "RPE-1":
        array = get_multiarrays(
            files=RPE1_FILES, settings=settings, chromosome_prefix=""
        )
    elif option == "HSR":
        array = get_multiarrays(
            files=HSR_FILES, settings=settings, chromosome_prefix=""
        )
    else:
        raise ValueError(f"Cell line '{option}' not known")
    return array


def get_entropy_from_settings(settings, option, entropy_type: str):
    array, val = get_arrays_from_settings(settings, option)

    if entropy_type.lower() == "rao":
        return get_quadratic_entropy(array), val
    elif entropy_type.lower() == "shannon":
        return get_entropy(array), val
    elif entropy_type.lower() == "geometric":
        return get_dispersion(array), val
    else:
        raise ValueError("Entropy not known")


def create_settings(chromosome, start):
    _frame_length = 1_600_000
    _bin_length = 10_000

    # _frame_length = 10_000_000
    # _bin_length = 50_000

    return rp.RegionSettings(
        chromosome=chromosome,
        start=start,
        end=start + _frame_length,
        bin_length=_bin_length,
    )


all_settings = [
    create_settings("8", 126_400_000),
    #
    create_settings("8", 2_000_000),
    create_settings("8", 15_000_000),
    create_settings("8", 24_000_000),
    create_settings("8", 107_000_000),
    create_settings("8", 120_000_000),
    # create_settings("8", 130_000_000),
    # create_settings("8", 132_000_000),
    # create_settings("8", 140_000_000),
]

spacing = 10
_min = 5
_sub = 3

all_settings = (
    [create_settings("8", 126_400_000)]
    +
    #
    [create_settings("1", i * 1_000_000) for i in range(_min, 245 - _sub, spacing)]
    + [create_settings("2", i * 1_000_000) for i in range(_min, 240 - _sub, spacing)]
    + [create_settings("3", i * 1_000_000) for i in range(_min, 197 - _sub, spacing)]
    + [create_settings("4", i * 1_000_000) for i in range(_min, 190 - _sub, spacing)]
    + [create_settings("5", i * 1_000_000) for i in range(_min, 178 - _sub, spacing)]
    + [create_settings("6", i * 1_000_000) for i in range(_min, 168 - _sub, spacing)]
    + [create_settings("7", i * 1_000_000) for i in range(_min, 156 - _sub, spacing)]
    + [create_settings("8", i * 1_000_000) for i in range(_min, 144 - _sub, spacing)]
    + [create_settings("9", i * 1_000_000) for i in range(_min, 138 - _sub, spacing)]
    + [create_settings("10", i * 1_000_000) for i in range(_min, 133 - _sub, spacing)]
    + [create_settings("11", i * 1_000_000) for i in range(_min, 132 - _sub, spacing)]
    + [create_settings("12", i * 1_000_000) for i in range(_min, 130 - _sub, spacing)]
    + [create_settings("13", i * 1_000_000) for i in range(_min, 112 - _sub, spacing)]
    + [create_settings("14", i * 1_000_000) for i in range(_min, 104 - _sub, spacing)]
    + [create_settings("15", i * 1_000_000) for i in range(_min, 98 - _sub, spacing)]
    + [create_settings("16", i * 1_000_000) for i in range(_min, 86 - _sub, spacing)]
    + [create_settings("17", i * 1_000_000) for i in range(_min, 76 - _sub, spacing)]
    + [create_settings("18", i * 1_000_000) for i in range(_min, 74 - _sub, spacing)]
    + [create_settings("19", i * 1_000_000) for i in range(_min, 61 - _sub, spacing)]
    + [create_settings("20", i * 1_000_000) for i in range(_min, 59 - _sub, spacing)]
    + [create_settings("21", i * 1_000_000) for i in range(_min, 44 - _sub, spacing)]
    + [create_settings("22", i * 1_000_000) for i in range(_min, 47 - _sub, spacing)]
)
# -

fig_design = [
    ("DM", "blue"),
    ("HSR", "green"),
    ("RPE-1", "red"),
]

# +
RECALCULATE = False

arrays = {k: [] for k, _ in fig_design}
valids = {k: [] for k, _ in fig_design}

if RECALCULATE:
    print("Recalculating the arrays... This will take a while...")
    for settings in all_settings:
        print(f"Processing {settings.chromosome} {settings.start}...")
        for cell_line, color in fig_design:
            arr, val = get_arrays_from_settings(settings, cell_line)
            arrays[cell_line].append(arr)
            valids[cell_line].append(val)

    # Save the arrays to the disk
    np.savez("arrays.npz", **arrays)
    np.savez("valids.npz", **valids)

else:
    print("Loading the data from the disk.")
    arrays = dict(np.load("arrays.npz"))
    valids = dict(np.load("valids.npz"))

# +
import matplotlib.pyplot as plt

entropies = {k: [] for k, _ in fig_design}


for i, settings in enumerate(all_settings):
    for cell_line, color in fig_design:
        ent = get_quadratic_entropy(arrays[cell_line][i])
        entropies[cell_line].append(ent)
# -

# ## Plot the entropy of each cell line

# +
make_valid_list = []

for i in range(len(all_settings)):
    decs = []
    for cell_line in ["RPE-1", "DM", "HSR"]:
        dec = bool(jnp.all(valids[cell_line][i]))
        decs.append(dec)
    if sum(decs) == 3:
        make_valid_list.append(i)

print("All genome fragments initially considered:", len(all_settings))
print("Genome fragments marked as valid:", len(make_valid_list))
# -

for i, a in enumerate(entropies["HSR"]):
    if a.mean() < 0.42:
        print(i, all_settings[i])

# +

for width in [2.5, 3.5, 4.5, 5.5]:

    fig, axs = plt.subplots(
        1, 3, figsize=(4 * 3, 2.3), dpi=600, sharex=True, sharey=True
    )

    def plot_entr(ax, line1):
        mus = []
        stds = []
        sts = []

        for i in make_valid_list:
            entrs = entropies[line1][i]
            # vals = valids[line1][i]

            mu = jnp.mean(entrs)
            std = jnp.std(entrs, ddof=1)

            st = jnp.mean(entrs)

            mus.append(mu)
            stds.append(std)
            sts.append(st)

        sts = jnp.stack(sts)
        mus = jnp.stack(mus)

        ax.hist(
            sts[1:],
            bins=jnp.linspace(0.2, 1.0, 20),
            density=False,
            color="darkblue",
            alpha=0.5,
        )
        # ax.axvline(jnp.mean(sts[1:]), linestyle=":", color="blue")
        ax.axvline(sts[0], linestyle="-", color="green", linewidth=width)
        print(f"Line: {line1}, RQE: {sts[0]}")

        ax.set_title(f"{line1}")
        tail_prob = float(jnp.mean(mus >= mus[0]))
        print(tail_prob)

    plot_entr(axs[0], "DM")
    plot_entr(axs[1], "HSR")
    plot_entr(axs[2], "RPE-1")

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Counts")
        ax.set_xlabel("Rao's quadratic entropy")

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(f"/tmp/figure-quadratic_entropy-{width}.{ext}", dpi=450)

# +
hist_colors = {
    "DM": "blue",
    "HSR": "salmon",
    "RPE-1": "gold",
}


for width in [1.5, 2.5, 3.5]:

    fig, ax = plt.subplots(1, 1, figsize=(5, 2.3), dpi=600)

    def plot_entr(ax, line1):
        print(f"Line: {line1}")
        mus = []
        stds = []
        sts = []

        for i in make_valid_list:
            entrs = entropies[line1][i]
            # vals = valids[line1][i]

            mu = jnp.mean(entrs)
            std = jnp.std(entrs, ddof=1)

            st = jnp.mean(entrs)

            mus.append(mu)
            stds.append(std)
            sts.append(st)

        sts = jnp.stack(sts)
        mus = jnp.stack(mus)

        ax.hist(
            sts[1:],
            bins=jnp.linspace(0.2, 1.0, 20),
            density=False,
            color=hist_colors[line1],
            alpha=0.3,
            label=line1,
        )
        # ax.axvline(jnp.mean(sts[1:]), linestyle=":", color="blue")
        ax.axvline(sts[0], linestyle=":", color=hist_colors[line1], linewidth=width)
        print(f"   Rao: {float(sts[0]):.4f}")

        # ax.set_title(f"{line1}")
        tail_prob = float(jnp.mean(mus >= mus[0]))
        print(f"   Tail prob.: {tail_prob:.4f}")

    plot_entr(ax, "DM")
    plot_entr(ax, "HSR")
    plot_entr(ax, "RPE-1")

    ax.legend(frameon=False, bbox_to_anchor=(1, 1))

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Counts")
    ax.set_xlabel("Rao's quadratic entropy")

    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(f"/tmp/figure-merged-{width}.{ext}", dpi=450)

# +
import seaborn as sns

n = 5

indices = jnp.arange(n)
xs = jnp.cos(2 * jnp.pi * indices / n)
ys = jnp.sin(2 * jnp.pi * indices / n)

labels = [f"S{i+1}" for i in indices]


D = jnp.sqrt(
    jnp.square(xs[:, None] - xs[None, :]) + jnp.square(ys[:, None] - ys[None, :])
)


# Using arc length
angles = 2 * jnp.pi * indices / n
diff_angle = jnp.abs(angles[:, None] - angles[None, :])

D = jnp.minimum(diff_angle, 2 * jnp.pi - diff_angle)


fig, axs = plt.subplots(1, 2, dpi=350)

for ax in axs:
    ax.set_aspect(1.0)

ax = axs[0]

ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

ts = jnp.linspace(0, 1, 101)
xs_ = jnp.cos(2 * jnp.pi * ts)
ys_ = jnp.sin(2 * jnp.pi * ts)

ax.plot(xs_, ys_, c="darkblue")
ax.scatter(xs, ys, c="darkblue")

ax = axs[1]
sns.heatmap(
    D / D[0, 1],
    ax=ax,
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=False,
    annot=True,
)

fig.tight_layout()
fig.savefig("/tmp/figure-s-phases.pdf")

# +
from scipy import stats

fig, axs = plt.subplots(1, 3, figsize=(4 * 3, 3), dpi=350, sharex=True)


def plot_diffs(ax, line1, line2):
    mus = []
    stds = []
    sts = []

    for i in range(len(all_settings)):
        diffs = entropies[line1][i] - entropies[line2][i]
        mu = jnp.mean(diffs)
        std = jnp.std(diffs, ddof=1)

        st = jnp.mean(diffs)

        mus.append(mu)
        stds.append(std)
        sts.append(st)

    sts = jnp.stack(sts)
    mus = jnp.stack(mus)

    ax.hist(sts[1:], bins=20, density=True, color="blue", alpha=0.5)
    ax.axvline(jnp.mean(sts[1:]), linestyle=":", color="blue")
    ax.axvline(sts[0], linestyle="-", color="green")

    ax.set_title(f"{line1}–{line2}")
    tail_prob = float(jnp.mean(mus[1:] > mus[0]))
    print(tail_prob)


plot_diffs(axs[0], "DM", "HSR")
plot_diffs(axs[1], "DM", "RPE-1")
plot_diffs(axs[2], "HSR", "RPE-1")

for ax in axs:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel("Rao's quadratic entropy difference")
# -

jnp.arange(len(all_settings))[
    jnp.asarray([jnp.mean(a) for a in entropies["RPE-1"]]) >= 1.24
]

# +
banned_indices = [8, 114, 131, 134]

for i in banned_indices:
    print(all_settings[i])
# -

rao_quadratic_entropy(jnp.asarray([0.25, 0.25, 0.25, 0.25]))

all_settings[131]

# +
# (deprecates) Heatmap

import matplotlib.pyplot as plt
import seaborn as sns

fig_design = [
    ("DM", "blue"),
    ("HSR", "green"),
    ("RPE-1", "red"),
]

fig, axs = plt.subplots(
    len(all_settings),
    len(fig_design),
    figsize=(4 * len(fig_design), 3 * len(all_settings)),
    dpi=200,
    sharex=True,
    sharey=True,
)

for i, settings in enumerate(all_settings):
    print(f"Processing {settings.start}...")
    for j, (cell_line, color) in enumerate(fig_design):
        array = get_array_from_settings(settings, cell_line)
        ax = axs[i, j]
        sns.heatmap(array, ax=ax, cbar=False, cmap="Greys", vmin=0, vmax=1)

        ax.set_ylabel(
            f"chr{settings.chromosome}: {settings.start/1e6:.1f} – {settings.end/1e6:.1f} Mb"
        )

for j, (cell_line, _) in enumerate(fig_design):
    axs[0, j].set_title(cell_line)

fig.tight_layout()
fig.savefig("/tmp/figura-heatmap.png", dpi=350)

# +
# (deprecated) Quadratic entropy histogram for each bin

import matplotlib.pyplot as plt

fig_design = [
    ("DM", "blue"),
    ("HSR", "green"),
    ("RPE-1", "red"),
]

fig, axs = plt.subplots(
    len(all_settings), figsize=(4, 3 * len(all_settings)), dpi=200, sharex=True
)
bins = 20

for settings, ax in zip(all_settings, axs.ravel()):
    print(f"Processing {settings.start}...")
    for cell_line, color in fig_design:
        ent, valid = get_entropy_from_settings(settings, cell_line, "rao")

        ax.hist(ent, bins=bins, color=color, alpha=0.5, density=True, label=cell_line)
        ax.axvline(jnp.mean(ent), color=color)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Rao's quadratic entropy")
    ax.set_title(
        f"chr{settings.chromosome}: {settings.start/1e6:.1f} – {settings.end/1e6:.1f} Mb"
    )

ax = axs[0]
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("/tmp/figura-hist.png", dpi=350)

# +
# (deprecated) entropy vs position

import matplotlib.pyplot as plt

fig_design = [
    ("DM", "blue"),
    ("HSR", "green"),
    ("RPE-1", "red"),
]

fig, axs = plt.subplots(
    len(all_settings),
    figsize=(4, 3 * len(all_settings)),
    dpi=200,
    sharex=True,
    sharey=True,
)
bins = jnp.linspace(0.0, 3)

for settings, ax in zip(all_settings, axs.ravel()):
    print(f"Processing {settings.start}...")
    for cell_line, color in fig_design:
        ent = get_entropy_from_settings(settings, cell_line, "rao")

        ax.plot(ent, color=color, label=cell_line)

    ax.spines[["top", "right"]].set_visible(False)
    # ax.set_xlabel("Rao's quadratic entropy")
    ax.set_title(
        f"chr{settings.chromosome}: {settings.start/1e6:.1f} – {settings.end/1e6:.1f} Mb"
    )

ax = axs[0]
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("/tmp/figura-plot.png", dpi=350)
# -
