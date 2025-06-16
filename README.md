# AI CUP

## Getting Started

### 1. Install uv

Tutorial: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

### 2. `uv sync`

Use `uv sync` to install the dependencies.

### 3. `uv run`

Use `uv run <python file>` to run the python file.

## Installing new packages

### 1. `uv add <package name>`

Use `uv add <package name>` to install a new package. It will automatically add the package to the `pyproject.toml` file and install it in the virtual environment.

## Script Descriptions

### `palapapa.py`

This script uses `RandomForestClassifier` to do predictions. To generate the
features required for training and inference, train the model, and output the
predictions all in one go, simply run:

```bash
uv run palapapa.py -gpt
```

```plaintext
usage: palapapa.py [-h] [-g | --generate-features | --no-generate-features] [-t | --train-model | --no-train-model] [-p | --generate-submission-csv | --no-generate-submission-csv] [-m CSV_GENERATION_MODEL_PATH]

You need to specify at least one option; otherwise the script does nothing.

options:
  -h, --help            show this help message and exit
  -g, --generate-features, --no-generate-features
                        Whether to generate the features required for training and prediction and then exit. (Will delete the features directory previously generated.) (default: False)
  -t, --train-model, --no-train-model
                        Whether to train the model and saves it to the disk. (default: False)
  -p, --generate-submission-csv, --no-generate-submission-csv
                        Whether to also use the model produced by supplying the -t flag to produce a CSV ready for submission to AI CUP. If you use this flag without the -t flag, you also need to specify the -m flag. (default: False)
  -m, --csv-generation-model-path CSV_GENERATION_MODEL_PATH
                        The path to the model to use when generating the submission CSV. Used when you use the -p flag without the -t flag. If the -p flag is used with the -t flag, this option does nothing. (default: None)
```
