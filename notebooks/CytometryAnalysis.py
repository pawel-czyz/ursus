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

import flowkit as fk
import matplotlib.pyplot as plt

# +
fcs_path = "data/20241202-FACS/HL60/Specimen_001_hl60.fcs"

# fcs_path = "data/20241202-FACS/HL60/Specimen_001_hl60-2.fcs"

sample = fk.Sample(fcs_path)
sample.channels
# -

data = sample.as_dataframe(source='raw')
data_subsampled = data.sample(n=5_000, replace=False)

# +
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

def plot_kde(
    ax,
    data,
    x_col,
    y_col,
    x_label,
    y_label,
    x_approved,
    y_approved,
):
    X = data[[x_col, y_col]].values
    X_new = StandardScaler().fit_transform(X)

    kde = gaussian_kde(X_new.T)
    density = kde(X_new.T)

    ax.fill_between(
        x_approved,
        y_approved[0] * np.ones(2),
        y_approved[1] * np.ones(2),
        facecolor="gold",
        edgecolor=None,
        alpha=0.3,
    )

    ax.scatter(X[:, 0], X[:, 1], c=density, cmap="magma", s=0.5, alpha=0.3, marker=".", rasterized=True)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# +
from subplots_from_axsize import subplots_from_axsize
import matplotlib.patches as patches
import numpy as np
import seaborn as sns


fig, axs = subplots_from_axsize(axsize=([1, 1, 1.8], 1), dpi=600, wspace=[0.5, 0.5])

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)

ax = axs[0]
filter1_x = "FSC-A"
filter1_y = "SSC-A"
filter1_x_approved = [30_000, 200_000]
filter1_y_approved = [20_000, 140_000]

plot_kde(
    ax, data_subsampled, x_col=filter1_x, y_col=filter1_y, x_label="FSC", y_label="SSC",
    x_approved=filter1_x_approved, y_approved=filter1_y_approved,
)


ax = axs[1]
filter2_x = 'YG 610/20-A'
filter2_y = 'YG 610/20-H'
filter2_x_approved = [40_000, 180_000]
filter2_y_approved = [40_000, 180_000]

plot_kde(
    ax, data_subsampled, x_col=filter2_x, y_col=filter2_y, x_label="610 nm – A", y_label="610 nm – H",
    x_approved=filter2_x_approved, y_approved=filter2_y_approved,
)

ax = axs[2]
ax.set_ylabel("Counts")
ax.set_xlabel("610 nm – A")
ax.set_xticks([])
ax.set_yticks([])

mask_filter1 = data[filter1_x].between(*filter1_x_approved) & \
               data[filter1_y].between(*filter1_y_approved)

# Apply Filter 2 using 'between'
mask_filter2 = data[filter2_x].between(*filter2_x_approved) & \
               data[filter2_y].between(*filter2_y_approved)

# Combine both filters
combined_mask = mask_filter1 & mask_filter2

# Apply the combined mask to filter the DataFrame
filtered_data = data[combined_mask]

hist_col = "YG 610/20-A"
vals = filtered_data[hist_col].values

left_gate = 80_000
right_gate = 148_000

gates = np.linspace(left_gate, right_gate, 4)

n_bins = 800
d_x = (vals.max() - vals.min()) / n_bins

for i in range(len(gates) + 1):
    _offset = 0.05
    if i == 0:
        bins = np.arange(vals.min(), gates[0], d_x)
    elif i == len(gates):
        bins = np.arange(gates[i-1], vals.max(), d_x)
    else:
        bins = np.arange(gates[i-1], gates[i], d_x)

    index = (vals >= bins.min()) & (vals < bins.max())

    colors = sns.color_palette("Paired", n_colors=len(gates) + 1)

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "goldenrod"]
    ax.hist(vals[index], bins=bins, rasterized=True, color=colors[i], alpha=0.8)

for gate in gates:
    ax.axvline(gate, c="black", linestyle=":", linewidth=1.4)
# -


