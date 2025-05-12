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
file = "GSE137764_HCT_GaussiansGSE137764_mooth_scaled_autosome.mat"

with open(file) as fh:
    lines = fh.readlines()

def lmap(f, x):
    return list(map(f, x))

lines = lmap(lambda x: x.strip().split("\t"), lines)
length = min([len(ll) for ll in lines])
lines = lmap(lambda x: x[:length], lines)

# +
chromosome_names = lines[0]
start = lmap(int, lines[1])
end = lmap(int, lines[2])

def _float(x):
    if x == "":
        return 0.0
    else:
        return float(x)

for value_index in range(3, len(lines)):
    phase_name = value_index - 2
    value = lmap(_float, lines[value_index])

    with open(f"outputs/S{phase_name}.bedgraph", "w") as fh:
        for ch, st, en, val in zip(chromosome_names, start, end, value):
            fh.write(f"{ch} {st} {en} {val}\n")
