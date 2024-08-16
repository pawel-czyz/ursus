# ursus

## Installation

Install the package, e.g., by:

```bash
$ pip install -e .
```

or

```bash
$ poetry install
```

## Running the workflows

Place the data in the `data/202407/bigwig` subdirectories, as specified in `data/202407/file_structure.txt`. Now you can run the workflow:

```bash
$ snakemake -s workflows/202407.smk -c4
```

and obtain the figures in `data/202407/figures` directory.

