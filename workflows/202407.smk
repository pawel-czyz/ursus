import dataclasses
import numpy as np
import matplotlib
matplotlib.use("agg")

import ursus.region_plotter as rp

workdir: "data/202407"


# All cell lines for which the data can be preprocessed:
# see the rules at the bottom of the file
CELL_LINES = ("RPE-1", "HSR", "DM", "HL-60", "PC3-DM", "PC3-HSR")


@dataclasses.dataclass
class Settings:
    region: rp.RegionSettings
    cell_lines: list[str]


SETTINGS = {
    "chr20-all": Settings(
        region=rp.RegionSettings(
            start=40_000_000,
            end=50_000_000,
            bin_length=10_000,
            chromosome="20",
        ),
        cell_lines=list(CELL_LINES),
    )
    
    
}

rule all:
    input: expand("figures/{settings}.pdf", settings=SETTINGS)




def _get_files(wildcards):
    settings = wildcards.settings
    cell_lines = SETTINGS[settings].cell_lines
    return [f"arrays/{settings}/{line}.npz" for line in cell_lines]

rule plot_region:
    input: _get_files
    output: "figures/{settings}.pdf"
    params:
        dpi = 450
    run:
        arrays = []
        labels = []
        for fname in input:
            archive = np.load(fname)
            arrays.append(archive["array"])
            labels.append(archive["label"])
        
        fig, _ = rp.plot_arrays(arrays=arrays, labels=labels, settings=SETTINGS[wildcards.settings].region, dpi=int(params.dpi))
        fig.savefig(str(output))


# ==========================================================
# ===== Rules to assemble BigWig files to NumPy arrays =====
# ===== Each cell line has a separate rule             =====
# ==========================================================              

# ----- HSR -----
HSR_FILES = [
        "1_i701_i501_S1__uniq_rmdup_10000.bw",
        "5_i702_i501_S5__uniq_rmdup_10000.bw",
        "9_i703_i501_S9__uniq_rmdup_10000.bw",
        "13_i704_i501_S13__uniq_rmdup_10000.bw",
        "17_i705_i501_S17__uniq_rmdup_10000.bw",
]

rule:
    output: "arrays/{settings}/HSR.npz"
    params:
        prefix = "",
        label = "HSR"
    input: expand("bigwig/HSR/{fname}", fname=HSR_FILES)
    run:
        array = rp.get_multiarray(files=list(input), settings=SETTINGS[wildcards.settings].region, chromosome_prefix=params.prefix)
        np.savez(file=str(output), array=array, label=params.label)


# ----- DM -----
DM_FILES = [
    "1_i701_i501_S1__uniq_rmdup_10000.bw",
    "5_i702_i501_S5__uniq_rmdup_10000.bw",
    "9_i703_i501_S9__uniq_rmdup_10000.bw",
    "13_i704_i501_S13__uniq_rmdup_10000.bw",
    "17_i705_i501_S17_uniq_rmdup_10000.bw",
]
rule:
    output: "arrays/{settings}/DM.npz"
    params:
        prefix = "chr",
        label= "DM"
    input: expand("bigwig/DM/{fname}", fname=DM_FILES)
    run:
        array = rp.get_multiarray(files=list(input), settings=SETTINGS[wildcards.settings].region, chromosome_prefix=params.prefix)
        np.savez(file=str(output), array=array, label=params.label)


# ----- HL-60 -----
HL60_FILES = [
    "3_i701_i503_S3__uniq_rmdup_10000.bw",
    "7_i702_i503_S7__uniq_rmdup_10000.bw",
    "11_i703_i503_S11__uniq_rmdup_10000.bw",
    "15_i704_i503_S15__uniq_rmdup_10000.bw",
    "19_i705_i503_S19__uniq_rmdup_10000.bw",
]
rule:
    output: "arrays/{settings}/HL-60.npz"
    params:
        prefix = "",
        label= "HL-60"
    input: expand("bigwig/HL-60/{fname}", fname=HL60_FILES)
    run:
        array = rp.get_multiarray(files=list(input), settings=SETTINGS[wildcards.settings].region, chromosome_prefix=params.prefix)
        np.savez(file=str(output), array=array, label=params.label)


# ----- RPE-1 -----
RPE1_FILES = [
    "2_i701_i502_S2__uniq_rmdup_10000.bw",
    "6_i702_i502_S6__uniq_rmdup_10000.bw",
    "10_i703_i502_S10__uniq_rmdup_10000.bw",
    "14_i704_i502_S14__uniq_rmdup_10000.bw",
    "18_i705_i502_S18__uniq_rmdup_10000.bw",
]
rule:
    output: "arrays/{settings}/RPE-1.npz"
    params:
        prefix = "",
        label= "RPE-1"
    input: expand("bigwig/RPE-1/{fname}", fname=RPE1_FILES)
    run:
        array = rp.get_multiarray(files=list(input), settings=SETTINGS[wildcards.settings].region, chromosome_prefix=params.prefix)
        np.savez(file=str(output), array=array, label=params.label)


# ----- PC3-DM -----
PC3_DM_FILES = [
    "PC3DM_2_1_2_i701_i502_S2__uniq_rmdup_10000 (1).bw",
    "PC3DM_2_2_6_i702_i502_S6__uniq_rmdup_10000 (1).bw",
    "PC3DM_2_3_10_i703_i502_S10__uniq_rmdup_10000 (1).bw",
    "PC3DM_2_4_14_i704_i502_S14__uniq_rmdup_10000.bw",
    "PC3DM_2_5_18_i705_i502_S18__uniq_rmdup_10000.bw",
]
rule:
    output: "arrays/{settings}/PC3-DM.npz"
    params:
        prefix = "",
        label= "DM PC3"
    input: expand("bigwig/PC3-DM/{fname}", fname=PC3_DM_FILES)
    run:
        array = rp.get_multiarray(files=list(input), settings=SETTINGS[wildcards.settings].region, chromosome_prefix=params.prefix)
        np.savez(file=str(output), array=array, label=params.label)


# ----- PC3-HSR -----
PC3_HSR_FILES = [
    "PC3HSR_1_1_3_i701_i503_S3__uniq_rmdup_10000 (1).bw",
    "PC3HSR_1_2_7_i702_i503_S7__uniq_rmdup_10000 (1).bw",
    "PC3HSR_1_3_11_i703_i503_S11__uniq_rmdup_10000.bw",
    "PC3HSR_1_4_15_i704_i503_S15__uniq_rmdup_10000.bw",
    "PC3HSR_1_5_19_i705_i503_S19__uniq_rmdup_10000.bw",
]
rule:
    output: "arrays/{settings}/PC3-HSR.npz"
    params:
        prefix = "",
        label= "HSR PC3"
    input: expand("bigwig/PC3-HSR/{fname}", fname=PC3_HSR_FILES)
    run:
        array = rp.get_multiarray(files=list(input), settings=SETTINGS[wildcards.settings].region, chromosome_prefix=params.prefix)
        np.savez(file=str(output), array=array, label=params.label)


# =====================================================================================================
# == Below there is a template allowing one to add a new experiment.                                 ==
# == Suppose we keep the data in the EXPERIMENT directory and in the plot we refer to it as "LABEL". ==
# =====================================================================================================
#
#
# EXPERIMENT_FILES = [  # Todo: Substitute EXPERIMENT for the right name
#     "path1.bw",
#     "path2.bw",
#     "...",   
# ]
# rule:
#     output: "arrays/{settings}/EXPERIMENT.npz"  # Todo: Substitute EXPERIMENT for the right name
#     params:
#         prefix = "",  # Todo: Specify a potential prefix. For example, "chr" if the BigWig file has "chr8", but not "8" chromosome 
#         label= "LABEL"  # Todo: Substitute LABEL for the right plot label
#     input: expand("bigwig/EXPERIMENT/{fname}", fname=EXPERIMENT_FILES)  # Todo: Substitute EXPERIMENT for the right name in both places
#     run:
#         array = rp.get_multiarray(files=list(input), settings=SETTINGS[wildcards.settings].region, chromosome_prefix=params.prefix)
#         np.savez(file=str(output), array=array, label=params.label)
