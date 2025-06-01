from __future__ import annotations
from os import mkdir
from sys import stderr
from typing import Any, Final, Literal, cast
from pandas import DataFrame, Series, read_csv, concat, options # pyright: ignore[reportUnknownVariableType]
from os.path import join, exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
from shutil import rmtree
from pathlib import Path
from dataclasses import dataclass
from numpy.fft import rfft
from numpy import float64, int64, isin, array, vstack, array_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score # pyright: ignore[reportUnknownVariableType]
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tqdm import tqdm
from pickle import dump, load, HIGHEST_PROTOCOL
from type_aliases import Double1D, Double2D, Long1D
from random import randint

TRAINING_DATA_BASE_PATH: Final = "./39_Training_Dataset"
TESTING_DATA_BASE_PATH: Final = "./39_Test_Dataset"
TRAINING_DATA_INFO_CSV_PATH: Final = join(TRAINING_DATA_BASE_PATH, "train_info.csv")
TESTING_DATA_INFO_CSV_PATH: Final = join(TESTING_DATA_BASE_PATH, "test_info.csv")
TRAINING_DATA_PATH: Final = join(TRAINING_DATA_BASE_PATH, "train_data")
TESTING_DATA_PATH: Final = join(TESTING_DATA_BASE_PATH, "test_data")
OUTPUT_BASE_PATH: Final = "./output"
FEATURES_BASE_PATH: Final = join(OUTPUT_BASE_PATH, "random_forest_features")
TRAINING_FEATURES_PATH: Final = join(FEATURES_BASE_PATH, "train")
TESTING_FEATURES_PATH: Final = join(FEATURES_BASE_PATH, "test")
MODEL_NAME: Final = "random_forest"
SUBMISSION_CSV_NAME: Final = "random_forest_submission"

# Opts in to Pandas 3.0's copy on write feature.
options.mode.copy_on_write = True

@dataclass
class FeatureCsvRow:
    """
    Represents the fields contained in a row of the feature CSV generated using
    the -g flag.

    All `fft_*` attributes actually store the magnitudes of the FFT result. For
    example, `fft_acceleration_x_mean` is actually the mean of magnitudes of the
    complex fourier-transformed X acceleration values.
    """
    mode: int
    """
    The mode number in `train_info.csv` or `test_info.csv` for this piece of
    data.
    """
    acceleration_x_mean: float
    acceleration_y_mean: float
    acceleration_z_mean: float
    angular_acceleration_x_mean: float
    angular_acceleration_y_mean: float
    angular_acceleration_z_mean: float
    acceleration_x_standard_deviation: float
    acceleration_y_standard_deviation: float
    acceleration_z_standard_deviation: float
    angular_acceleration_x_standard_deviation: float
    angular_acceleration_y_standard_deviation: float
    angular_acceleration_z_standard_deviation: float
    acceleration_x_root_mean_square: float
    acceleration_y_root_mean_square: float
    acceleration_z_root_mean_square: float
    angular_acceleration_x_root_mean_square: float
    angular_acceleration_y_root_mean_square: float
    angular_acceleration_z_root_mean_square: float
    acceleration_x_max: float
    acceleration_y_max: float
    acceleration_z_max: float
    angular_acceleration_x_max: float
    angular_acceleration_y_max: float
    angular_acceleration_z_max: float
    acceleration_x_min: float
    acceleration_y_min: float
    acceleration_z_min: float
    angular_acceleration_x_min: float
    angular_acceleration_y_min: float
    angular_acceleration_z_min: float
    acceleration_x_median: float
    acceleration_y_median: float
    acceleration_z_median: float
    angular_acceleration_x_median: float
    angular_acceleration_y_median: float
    angular_acceleration_z_median: float
    acceleration_x_kurtosis: float
    acceleration_y_kurtosis: float
    acceleration_z_kurtosis: float
    angular_acceleration_x_kurtosis: float
    angular_acceleration_y_kurtosis: float
    angular_acceleration_z_kurtosis: float
    acceleration_x_skewness: float
    acceleration_y_skewness: float
    acceleration_z_skewness: float
    angular_acceleration_x_skewness: float
    angular_acceleration_y_skewness: float
    angular_acceleration_z_skewness: float
    acceleration_mean: float
    angular_acceleration_mean: float
    acceleration_standard_deviation: float
    angular_acceleration_standard_deviation: float
    acceleration_root_mean_square: float
    angular_acceleration_root_mean_square: float
    acceleration_max: float
    angular_acceleration_max: float
    acceleration_min: float
    angular_acceleration_min: float
    acceleration_median: float
    angular_acceleration_median: float
    acceleration_kurtosis: float
    angular_acceleration_kurtosis: float
    acceleration_skewness: float
    angular_acceleration_skewness: float
    fft_acceleration_x_mean: float
    fft_acceleration_y_mean: float
    fft_acceleration_z_mean: float
    fft_angular_acceleration_x_mean: float
    fft_angular_acceleration_y_mean: float
    fft_angular_acceleration_z_mean: float
    fft_acceleration_x_standard_deviation: float
    fft_acceleration_y_standard_deviation: float
    fft_acceleration_z_standard_deviation: float
    fft_angular_acceleration_x_standard_deviation: float
    fft_angular_acceleration_y_standard_deviation: float
    fft_angular_acceleration_z_standard_deviation: float
    fft_acceleration_x_root_mean_square: float
    fft_acceleration_y_root_mean_square: float
    fft_acceleration_z_root_mean_square: float
    fft_angular_acceleration_x_root_mean_square: float
    fft_angular_acceleration_y_root_mean_square: float
    fft_angular_acceleration_z_root_mean_square: float
    fft_acceleration_x_max: float
    fft_acceleration_y_max: float
    fft_acceleration_z_max: float
    fft_angular_acceleration_x_max: float
    fft_angular_acceleration_y_max: float
    fft_angular_acceleration_z_max: float
    fft_acceleration_x_min: float
    fft_acceleration_y_min: float
    fft_acceleration_z_min: float
    fft_angular_acceleration_x_min: float
    fft_angular_acceleration_y_min: float
    fft_angular_acceleration_z_min: float
    fft_acceleration_x_median: float
    fft_acceleration_y_median: float
    fft_acceleration_z_median: float
    fft_angular_acceleration_x_median: float
    fft_angular_acceleration_y_median: float
    fft_angular_acceleration_z_median: float
    fft_acceleration_x_kurtosis: float
    fft_acceleration_y_kurtosis: float
    fft_acceleration_z_kurtosis: float
    fft_angular_acceleration_x_kurtosis: float
    fft_angular_acceleration_y_kurtosis: float
    fft_angular_acceleration_z_kurtosis: float
    fft_acceleration_x_skewness: float
    fft_acceleration_y_skewness: float
    fft_acceleration_z_skewness: float
    fft_angular_acceleration_x_skewness: float
    fft_angular_acceleration_y_skewness: float
    fft_angular_acceleration_z_skewness: float
    fft_acceleration_mean: float
    fft_angular_acceleration_mean: float
    fft_acceleration_standard_deviation: float
    fft_angular_acceleration_standard_deviation: float
    fft_acceleration_root_mean_square: float
    fft_angular_acceleration_root_mean_square: float
    fft_acceleration_max: float
    fft_angular_acceleration_max: float
    fft_acceleration_min: float
    fft_angular_acceleration_min: float
    fft_acceleration_median: float
    fft_angular_acceleration_median: float
    fft_acceleration_kurtosis: float
    fft_angular_acceleration_kurtosis: float
    fft_acceleration_skewness: float
    fft_angular_acceleration_skewness: float

def generate_features_for_single_data(data: DataFrame, mode: int) -> list[FeatureCsvRow]:
    """
    Used by `generate_features` to generate the features of a single piece of
    training or testing data. Returns a list of `FeatureCsvRow` so that
    `generate_features` can write it to a CSV.

    :param data: The `DataFrame` read from one piece of training or testing data
        txt in the following manner:
    ```
    read_csv(<path>,
        sep=r"\\s+",
        names=["acceleration_x", "acceleration_y", "acceleration_z", "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"])
    ```
    
    :param mode: The mode number for this piece of training or testing data as
        obtained from `train_info.csv` or `test_info.csv`.
    """
    fft_data = data.apply(rfft).abs()
    acceleration = cast("Series[float]", data[["acceleration_x", "acceleration_y", "acceleration_z"]].pow(2).sum(axis=1).pow(0.5)) # pyright: ignore[reportUnknownMemberType]
    angular_acceleration = cast("Series[float]", data[["angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"]].pow(2).sum(axis=1).pow(0.5)) # pyright: ignore[reportUnknownMemberType]
    fft_acceleration = cast("Series[float]", fft_data[["acceleration_x", "acceleration_y", "acceleration_z"]].pow(2).sum(axis=1).pow(0.5)) # pyright: ignore[reportUnknownMemberType]
    fft_angular_acceleration = cast("Series[float]", fft_data[["angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"]].pow(2).sum(axis=1).pow(0.5)) # pyright: ignore[reportUnknownMemberType]
    feature = FeatureCsvRow(
        mode=mode,
        # `mean` doesn't actually return float as the type hint suggests, but
        # returns whatever `dtype` `numpy.mean` returns, so in this case it's
        # actually `float64`. Accurately type hinting the attributes of
        # `FeatureCsvRow` would mean that I need to use `cast` everywhere, so I
        # decided not to.
        acceleration_x_mean=data["acceleration_x"].mean(),
        acceleration_y_mean=data["acceleration_y"].mean(),
        acceleration_z_mean=data["acceleration_z"].mean(),
        angular_acceleration_x_mean=data["angular_acceleration_x"].mean(),
        angular_acceleration_y_mean=data["angular_acceleration_y"].mean(),
        angular_acceleration_z_mean=data["angular_acceleration_z"].mean(),
        acceleration_x_standard_deviation=data["acceleration_x"].std(),
        acceleration_y_standard_deviation=data["acceleration_y"].std(),
        acceleration_z_standard_deviation=data["acceleration_z"].std(),
        angular_acceleration_x_standard_deviation=data["angular_acceleration_x"].std(),
        angular_acceleration_y_standard_deviation=data["angular_acceleration_y"].std(),
        angular_acceleration_z_standard_deviation=data["angular_acceleration_z"].std(),
        acceleration_x_root_mean_square=data["acceleration_x"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        acceleration_y_root_mean_square=data["acceleration_y"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        acceleration_z_root_mean_square=data["acceleration_z"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        angular_acceleration_x_root_mean_square=data["angular_acceleration_x"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        angular_acceleration_y_root_mean_square=data["angular_acceleration_y"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        angular_acceleration_z_root_mean_square=data["angular_acceleration_z"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        acceleration_x_max=data["acceleration_x"].max(), # pyright: ignore[reportUnknownArgumentType]
        acceleration_y_max=data["acceleration_y"].max(), # pyright: ignore[reportUnknownArgumentType]
        acceleration_z_max=data["acceleration_z"].max(), # pyright: ignore[reportUnknownArgumentType]
        angular_acceleration_x_max=data["angular_acceleration_x"].max(), # pyright: ignore[reportUnknownArgumentType]
        angular_acceleration_y_max=data["angular_acceleration_y"].max(), # pyright: ignore[reportUnknownArgumentType]
        angular_acceleration_z_max=data["angular_acceleration_z"].max(), # pyright: ignore[reportUnknownArgumentType]
        acceleration_x_min=data["acceleration_x"].min(), # pyright: ignore[reportUnknownArgumentType]
        acceleration_y_min=data["acceleration_y"].min(), # pyright: ignore[reportUnknownArgumentType]
        acceleration_z_min=data["acceleration_z"].min(), # pyright: ignore[reportUnknownArgumentType]
        angular_acceleration_x_min=data["angular_acceleration_x"].min(), # pyright: ignore[reportUnknownArgumentType]
        angular_acceleration_y_min=data["angular_acceleration_y"].min(), # pyright: ignore[reportUnknownArgumentType]
        angular_acceleration_z_min=data["angular_acceleration_z"].min(), # pyright: ignore[reportUnknownArgumentType]
        acceleration_x_median=data["acceleration_x"].median(),
        acceleration_y_median=data["acceleration_y"].median(),
        acceleration_z_median=data["acceleration_z"].median(),
        angular_acceleration_x_median=data["angular_acceleration_x"].median(),
        angular_acceleration_y_median=data["angular_acceleration_y"].median(),
        angular_acceleration_z_median=data["angular_acceleration_z"].median(),
        acceleration_x_kurtosis=cast(float64, data["acceleration_x"].kurt()),
        acceleration_y_kurtosis=cast(float64, data["acceleration_y"].kurt()),
        acceleration_z_kurtosis=cast(float64, data["acceleration_z"].kurt()),
        angular_acceleration_x_kurtosis=cast(float64, data["angular_acceleration_x"].kurt()),
        angular_acceleration_y_kurtosis=cast(float64, data["angular_acceleration_y"].kurt()),
        angular_acceleration_z_kurtosis=cast(float64, data["angular_acceleration_z"].kurt()),
        acceleration_x_skewness=cast(float64, data["acceleration_x"].skew()),
        acceleration_y_skewness=cast(float64, data["acceleration_y"].skew()),
        acceleration_z_skewness=cast(float64, data["acceleration_z"].skew()),
        angular_acceleration_x_skewness=cast(float64, data["angular_acceleration_x"].skew()),
        angular_acceleration_y_skewness=cast(float64, data["angular_acceleration_y"].skew()),
        angular_acceleration_z_skewness=cast(float64, data["angular_acceleration_z"].skew()),
        acceleration_mean=acceleration.mean(),
        angular_acceleration_mean=angular_acceleration.mean(),
        acceleration_standard_deviation=acceleration.std(),
        angular_acceleration_standard_deviation=angular_acceleration.std(),
        acceleration_root_mean_square=acceleration.pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        angular_acceleration_root_mean_square=angular_acceleration.pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        acceleration_max=acceleration.max(),
        angular_acceleration_max=angular_acceleration.max(),
        acceleration_min=acceleration.min(),
        angular_acceleration_min=angular_acceleration.min(),
        acceleration_median=acceleration.median(),
        angular_acceleration_median=angular_acceleration.median(),
        acceleration_kurtosis=cast(float64, acceleration.kurt()),
        angular_acceleration_kurtosis=cast(float64, angular_acceleration.kurt()),
        acceleration_skewness=cast(float64, acceleration.skew()),
        angular_acceleration_skewness=cast(float64, angular_acceleration.skew()),
        fft_acceleration_x_mean=fft_data["acceleration_x"].mean(),
        fft_acceleration_y_mean=fft_data["acceleration_y"].mean(),
        fft_acceleration_z_mean=fft_data["acceleration_z"].mean(),
        fft_angular_acceleration_x_mean=fft_data["angular_acceleration_x"].mean(),
        fft_angular_acceleration_y_mean=fft_data["angular_acceleration_y"].mean(),
        fft_angular_acceleration_z_mean=fft_data["angular_acceleration_z"].mean(),
        fft_acceleration_x_standard_deviation=fft_data["acceleration_x"].std(),
        fft_acceleration_y_standard_deviation=fft_data["acceleration_y"].std(),
        fft_acceleration_z_standard_deviation=fft_data["acceleration_z"].std(),
        fft_angular_acceleration_x_standard_deviation=fft_data["angular_acceleration_x"].std(),
        fft_angular_acceleration_y_standard_deviation=fft_data["angular_acceleration_y"].std(),
        fft_angular_acceleration_z_standard_deviation=fft_data["angular_acceleration_z"].std(),
        fft_acceleration_x_root_mean_square=fft_data["acceleration_x"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        fft_acceleration_y_root_mean_square=fft_data["acceleration_y"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        fft_acceleration_z_root_mean_square=fft_data["acceleration_z"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        fft_angular_acceleration_x_root_mean_square=fft_data["angular_acceleration_x"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        fft_angular_acceleration_y_root_mean_square=fft_data["angular_acceleration_y"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        fft_angular_acceleration_z_root_mean_square=fft_data["angular_acceleration_z"].pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        fft_acceleration_x_max=fft_data["acceleration_x"].max(), # pyright: ignore[reportUnknownArgumentType]
        fft_acceleration_y_max=fft_data["acceleration_y"].max(), # pyright: ignore[reportUnknownArgumentType]
        fft_acceleration_z_max=fft_data["acceleration_z"].max(), # pyright: ignore[reportUnknownArgumentType]
        fft_angular_acceleration_x_max=fft_data["angular_acceleration_x"].max(), # pyright: ignore[reportUnknownArgumentType]
        fft_angular_acceleration_y_max=fft_data["angular_acceleration_y"].max(), # pyright: ignore[reportUnknownArgumentType]
        fft_angular_acceleration_z_max=fft_data["angular_acceleration_z"].max(), # pyright: ignore[reportUnknownArgumentType]
        fft_acceleration_x_min=fft_data["acceleration_x"].min(), # pyright: ignore[reportUnknownArgumentType]
        fft_acceleration_y_min=fft_data["acceleration_y"].min(), # pyright: ignore[reportUnknownArgumentType]
        fft_acceleration_z_min=fft_data["acceleration_z"].min(), # pyright: ignore[reportUnknownArgumentType]
        fft_angular_acceleration_x_min=fft_data["angular_acceleration_x"].min(), # pyright: ignore[reportUnknownArgumentType]
        fft_angular_acceleration_y_min=fft_data["angular_acceleration_y"].min(), # pyright: ignore[reportUnknownArgumentType]
        fft_angular_acceleration_z_min=fft_data["angular_acceleration_z"].min(), # pyright: ignore[reportUnknownArgumentType]
        fft_acceleration_x_median=fft_data["acceleration_x"].median(),
        fft_acceleration_y_median=fft_data["acceleration_y"].median(),
        fft_acceleration_z_median=fft_data["acceleration_z"].median(),
        fft_angular_acceleration_x_median=fft_data["angular_acceleration_x"].median(),
        fft_angular_acceleration_y_median=fft_data["angular_acceleration_y"].median(),
        fft_angular_acceleration_z_median=fft_data["angular_acceleration_z"].median(),
        fft_acceleration_x_kurtosis=cast(float64, fft_data["acceleration_x"].kurt()),
        fft_acceleration_y_kurtosis=cast(float64, fft_data["acceleration_y"].kurt()),
        fft_acceleration_z_kurtosis=cast(float64, fft_data["acceleration_z"].kurt()),
        fft_angular_acceleration_x_kurtosis=cast(float64, fft_data["angular_acceleration_x"].kurt()),
        fft_angular_acceleration_y_kurtosis=cast(float64, fft_data["angular_acceleration_y"].kurt()),
        fft_angular_acceleration_z_kurtosis=cast(float64, fft_data["angular_acceleration_z"].kurt()),
        fft_acceleration_x_skewness=cast(float64, fft_data["acceleration_x"].skew()),
        fft_acceleration_y_skewness=cast(float64, fft_data["acceleration_y"].skew()),
        fft_acceleration_z_skewness=cast(float64, fft_data["acceleration_z"].skew()),
        fft_angular_acceleration_x_skewness=cast(float64, fft_data["angular_acceleration_x"].skew()),
        fft_angular_acceleration_y_skewness=cast(float64, fft_data["angular_acceleration_y"].skew()),
        fft_angular_acceleration_z_skewness=cast(float64, fft_data["angular_acceleration_z"].skew()),
        fft_acceleration_mean=fft_acceleration.mean(),
        fft_angular_acceleration_mean=fft_angular_acceleration.mean(),
        fft_acceleration_standard_deviation=fft_acceleration.std(),
        fft_angular_acceleration_standard_deviation=fft_angular_acceleration.std(),
        fft_acceleration_root_mean_square=fft_acceleration.pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        fft_angular_acceleration_root_mean_square=fft_angular_acceleration.pow(2).mean() ** 0.5, # pyright: ignore[reportUnknownMemberType]
        fft_acceleration_max=fft_acceleration.max(),
        fft_angular_acceleration_max=fft_angular_acceleration.max(),
        fft_acceleration_min=fft_acceleration.min(),
        fft_angular_acceleration_min=fft_angular_acceleration.min(),
        fft_acceleration_median=fft_acceleration.median(),
        fft_angular_acceleration_median=fft_angular_acceleration.median(),
        fft_acceleration_kurtosis=cast(float64, fft_acceleration.kurt()),
        fft_angular_acceleration_kurtosis=cast(float64, fft_angular_acceleration.kurt()),
        fft_acceleration_skewness=cast(float64, fft_acceleration.skew()),
        fft_angular_acceleration_skewness=cast(float64, fft_angular_acceleration.skew()),
    )
    return [feature]

def generate_features():
    """
    Generates a feature CSV for each piece of training and testing data and puts
    the results under `FEATURES_PATH`.
    """
    if exists(FEATURES_BASE_PATH):
        rmtree(FEATURES_BASE_PATH)
    # Can't use `else` here. We want to recreate the directory if the last `if`
    # deleted it.
    if not exists(FEATURES_BASE_PATH):
        mkdir(FEATURES_BASE_PATH)
        mkdir(TRAINING_FEATURES_PATH)
        mkdir(TESTING_FEATURES_PATH)
    # https://stackoverflow.com/questions/59148830/python-pandas-why-cant-i-use-both-index-col-and-usecols-in-the-same-read-csv
    training_data_unique_id_to_mode = read_csv(TRAINING_DATA_INFO_CSV_PATH, index_col="unique_id", usecols=["unique_id","mode"])
    testing_data_unique_id_to_mode = read_csv(TESTING_DATA_INFO_CSV_PATH, index_col="unique_id", usecols=["unique_id","mode"])
    # Used to know the mode number of each piece of training or testing txt
    data_unique_id_to_mode = concat([training_data_unique_id_to_mode, testing_data_unique_id_to_mode])
    # Converts to lists so we can know the total number of files to generate and
    # make the last line of this function that checks which directory to save
    # the CSV in work.
    training_data_txt_paths = list(Path(TRAINING_DATA_PATH).glob("**/*.txt"))
    testing_data_txt_paths = list(Path(TESTING_DATA_PATH).glob("**/*.txt"))
    data_txt_paths = training_data_txt_paths + testing_data_txt_paths
    # Generates a feature CSV for each piece of training and test data
    for data_txt_path in tqdm(data_txt_paths, desc="Generating features", unit="file"):
        # Reads a training of testing .txt file as a DataFrame
        data = read_csv(data_txt_path,
            sep=r"\s+",
            names=["acceleration_x", "acceleration_y", "acceleration_z", "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"])
        data_id = int(data_txt_path.stem)
        mode = cast(int64, data_unique_id_to_mode.loc[data_id, "mode"]).item() # .item() converts int64 to Python's int
        features = generate_features_for_single_data(data, mode)
        # Converts the list of FeatureCsvRows into a list of dicts so that
        # pandas can write them to a CSV
        features_dict_list = [vars(feature) for feature in features]
        features_data_frame = DataFrame(features_dict_list)
        features_filename = data_txt_path.stem
        # There is probably a better way of checking which directory to save in.
        features_data_frame.to_csv(join(TRAINING_FEATURES_PATH if data_txt_path in training_data_txt_paths else TESTING_FEATURES_PATH, f"{features_filename}.csv"), index=False)

def check_feature_directory_existence(option_name: str):
    if not exists(FEATURES_BASE_PATH):
        print(f"You need to generate the features of the training and testing data using the -g option before using the {option_name} option.", file=stderr)
        exit(1)

def check_model_existence(model_path: str, option_name: str):
    if not exists(model_path):
        print(f"You need to train the model using the -t option before using the {option_name} option.", file=stderr)
        exit(1)

def fix_missing_labels(
    predictions: Double2D,
    targets: Series[int],
    target_to_fix: Literal["gender", "hold racket handed", "play years", "level"]) -> tuple[Double2D, Series[int]]:
    """
    The dataset we are working with doesn't have evenly distributed labels. For
    example, there are a lot more male players than female players, and there
    are very few level 4 players. This causes issues when trying to calculate
    the ROC AUC score of a training run, because with cross validation, some
    folds won't have every possible label. For example, the testing fold for the
    player level prediction task will sometimes only contain level 2, 3, and 5
    players. In this case, `roc_auc_score` will throw because it will think that
    there are only 3 kinds of labels, but we provided predictions that each
    contain 4 probabilities (the probability that a player is of level 2, 3, 4,
    or 5 respectively). This function tries to fix this by detecting which
    labels are missing, and for each missing label, it inserts a piece of fake
    data whose target is that missing label, and whose predicted probabilities
    will be evenly distributed. Using the previous example again, this function
    would insert a row into `predictions` that contains `[0.25, 0.25, 0.25,
    0.25]` and an `4` into `targets`.

    :param predictions: A `float64` `ndarray` with N rows and M columns, where N
        is the number of predictions and M depends on `target_to_fix`:

        - `target_to_fix == "gender"`: 2 columns. The first column contains the
          probability that the player is male. The second column contains the
          probability that the player is female.
        - `target_to_fix == "hold racket handed"`: 2 columns. The first column
          contains the probability that the player is right-handed. The second
          column contains the probability that the player is left-handed.
        - `target_to_fix == "play years"`: 3 columns. The first column contains
          the probability that the player has low experience. The second column
          contains the  probability that the player has medium experience. The
          third column contains the probability that the player has high
          experience.
        - `target_to_fix == "level"`: 4 columns. The first column contains the
          probability that the player is of level 2. The second column contains
          the probability that the player is of level 3. The third column
          contains the probability that the player is of level 4. The fourth
          column contains the probability that the player is of level 5.
    
    :param targets: A `Series[int]` that is N elements long, where N is equal to
        the N in `predictions`, that contains the target (correct anwser) of
        each row in `predictions`. The allowed values depend on `target_to_fix`:

        - `target_to_fix == "gender"`: 1 or 2.
        - `target_to_fix == "hold racket handed"`: 1 or 2.
        - `target_to_fix == "play years"`: 0, 1, or 2.
        - `target_to_fix == "level"`: 2, 3, 4, or 5.

        In all cases, if `targets` doesn't contain at least one of every allowed
        value depending on `target_to_fix`, this function will "fix"
        `predictions` and `targets` according to the description above. If it
        does, then nothing is done.

    :returns: A tuple whose first element is the fixed `predictions` and second
        element is the fixed `targets`. These two will not be the same objects
        as the two passed in if this function inserted fake data, i.e. a copy
        will be made. If there was nothing to fix, these two will be the same
        object as the two passed in, i.e. modifying one will modify another.
    """
    fixed_predictions = predictions
    fixed_targets = targets
    match target_to_fix:
        case "gender" | "hold racket handed":
            allowed_values = [1, 2]
        case "play years":
            allowed_values = [0, 1, 2]
        case "level":
            allowed_values = [2, 3, 4, 5]
    # By doing this, if an element in this is False, that means the
    # corresponding value in allowed_values is missing from targets.
    allowed_values_index_to_whether_exists = isin(allowed_values, targets)
    for allowed_values_index, exists in enumerate(allowed_values_index_to_whether_exists):
        if exists:
            continue
        match target_to_fix:
            case "gender" | "hold racket handed":
                fixed_predictions = cast(Double2D, vstack([fixed_predictions, array([0.5, 0.5])]))
            case "play years":
                fixed_predictions = cast(Double2D, vstack([fixed_predictions, array([1 / 3, 1 / 3, 1 / 3])]))
            case "level":
                fixed_predictions = cast(Double2D, vstack([fixed_predictions, array([0.25, 0.25, 0.25, 0.25])]))
        # Adds the missing label to fixed_targets.
        fixed_targets = concat([fixed_targets, Series([allowed_values[allowed_values_index]])])
    return fixed_predictions, fixed_targets

def calculate_roc_auc_scores(
    predictions: list[Double2D],
    targets: DataFrame) -> tuple[float, float, float, float]:
    """
    Calculates the ROC AUC scores of each of the 4 prediction targets in a
    training run.

    :param predictions: A `list` of 2D `float64` `ndarray`s where:

        1. The first is a `ndarray` that has N rows and 2 columns where N is the
           number of predictions. The first column contains the probability that
           the player is male. The second column contains the probability that
           the player is female.
        2. The second is a `ndarray` that has N rows and 2 columns where N is
           the number of predictions. The first column contains the probability
           that the player is right-handed. The second column contains the
           probability that the player is left-handed.
        3. The third is a `ndarray` that has N rows and 3 columns where N is the
           number of predictions. The first column contains the probability that
           the player has low experience. The second column contains the
           probability that the player has medium experience. The third column
           contains the probability that the player has high experience.
        4. The third is a `ndarray` that has N rows and 4 columns where N is the
           number of predictions. The first column contains the probability that
           the player is of level 2. The second column contains the probability
           that the player is of level 3. The third column contains the
           probability that the player is of level 4. The fourth column contains
           the probability that the player is of level 5.
    
    :param targets: A `DataFrame` with N rows and 4 columns, where N is the
        number of predictions, that contains the prediction targets (correct
        answers) where:

        1. A column named "gender" contains the prediction targets for the first
           `ndarray` in `predictions`. Each cell contains either a 1 or 2. 1
           indicates that the player is male. 2 indicates that the player is
           female.
        2. A column named "hold racket handed" contains the prediction targets
           for the second `ndarray` in `predictions`. Each cell contains either
           a 1 or 2. 1 indicates that the player is right-handed. 2 indicates
           that the player is left-handed.
        3. A column named "play years" contains the prediction targets for the
           third `ndarray` in `predictions`. Each cell contains either a 0, 1,
           or 2. 0 indicates that the player has low experience. 1 indicates
           that the player has medium experience. 2 indicates that the player
           has high experience.
        4. A column named "level" contains the prediction targets for the fourth
           `ndarray` in `predictions`. Each cell contains either a 2, 3, 4, or
           5. 2 indicates that the player is of level 2. 3 indicates that the
           player is of level 3. 4 indicates that the player is of level 4. 5
           indicates that the player is of level 5.
        
    :returns: A tuple that contains 4 `float`s, which are the gender ROC AUC
        score, handedness ROC AUC score, experience ROC AUC score, and level ROC
        AUC score, in that order.
    """
    fixed_gender_predictions, fixed_gender_targets = fix_missing_labels(predictions[0], cast("Series[int]", targets["gender"]), "gender")
    fixed_handedness_predictions, fixed_handedness_targets = fix_missing_labels(predictions[1], cast("Series[int]", targets["hold racket handed"]), "hold racket handed")
    fixed_experience_predictions, fixed_experience_targets = fix_missing_labels(predictions[2], cast("Series[int]", targets["play years"]), "play years")
    fixed_level_predictions, fixed_level_targets = fix_missing_labels(predictions[3], cast("Series[int]", targets["level"]), "level")
    return (float(roc_auc_score(fixed_gender_targets, fixed_gender_predictions[:, 1])),
        float(roc_auc_score(fixed_handedness_targets, fixed_handedness_predictions[:, 1])),
        float(roc_auc_score(fixed_experience_targets, fixed_experience_predictions, average="micro", multi_class="ovr")),
        float(roc_auc_score(fixed_level_targets, fixed_level_predictions, average="micro", multi_class="ovr")))

def print_scores(
    gender_score: float,
    handedness_score: float,
    experience_score: float,
    level_score: float):
    """
    Prints the 4 ROC AUC scores calculated from `calculate_roc_auc_scores`.
    """
    print(f"Gender ROC AUC score: {gender_score}")
    print(f"Handedness ROC AUC score: {handedness_score}")
    print(f"Experience ROC AUC score: {experience_score}")
    print(f"Level ROC AUC score: {level_score}")

def check_if_fold_is_usable(training_targets: DataFrame) -> bool:
    """
    The same problem described in the documentation of `fix_missing_labels` can
    also happen the the training set, which can cause the fitted
    `RandomForestClassifier` to output the wrong number of class probabilities.
    For example, if the training set happens to not contain any sample for a
    level 4 player, then the fitted `RandomForestClassifier` would only output 3
    probabilities for the player level prediction task and cause trouble.

    In summary, there are 4 scenarios regarding missing labels:

    1. No missing label in the training set and no missing label in the
       validation set: Everything is fine.
    2. No missing label in the training set and missing label in the validation
       set: There will only be problems when calling `roc_auc_score`, which
       should be handled by `fix_missing_labels`.
    3. Missing label in the training set and no missing label in the validation
       set: `fix_missing_labels` wouldn't do anything because it thinks that
       there are no missing labels, but it doesn't check that the `predictions`
       parameter actually contains the wrong number of columns because the
       fitted `RandomForestClassifier` produces the wrong number of class
       probabilities, so eventually `roc_auc_score` will error because the
       number of classes and probabilities will be different.
    4. Missing label in the training set and missing label in the validation
       set: `fix_missing_labels` will see that there are missing labels in the
       `targets` parameter and will try to insert a row of fake prediction into
       `predictions`. However, since the fitted `RandomForestClassifier` will
       produce the wrong number of class probabilities, when
       `fix_missing_labels` appends the fake row, it will fail because the
       column size of the row it tries to append will be larger than the column
       size of the `predictions` parameters (because missing labels in the
       training set means that the fitted `RandomForestClassifier` will produce
       less class probabilities than expected). In this case, the program fails
       inside `fix_missing_labels` instead of `roc_auc_score`.

    There is also the case of missing labels in the testing set, but it is
    similar to the case of missing labels in the validation set.

    We can see that the only 2 cases where there will actually be problems is
    when the training set has missing labels, so we would use this function to
    check if that is the case, and skip a fold if it is.

    This problem can also be fixed using `StratifiedGroupKFold` or by inserting
    fake features into the training set, but the former will require us to train
    a separate tree for each prediction task and the latter may undermine the
    performance of the model.
    """
    unique_genders = cast(Long1D, training_targets["gender"].unique()) # pyright: ignore[reportUnknownMemberType]
    unique_genders.sort()
    unique_handednesses = cast(Long1D, training_targets["hold racket handed"].unique()) # pyright: ignore[reportUnknownMemberType]
    unique_handednesses.sort()
    unique_experiences = cast(Long1D, training_targets["play years"].unique()) # pyright: ignore[reportUnknownMemberType]
    unique_experiences.sort()
    unique_levels = cast(Long1D, training_targets["level"].unique()) # pyright: ignore[reportUnknownMemberType]
    unique_levels.sort()
    if (array_equal(unique_genders, array([1, 2])) and 
        array_equal(unique_handednesses, array([1, 2])) and
        array_equal(unique_experiences, array([0, 1, 2])) and
        array_equal(unique_levels, array([2, 3, 4, 5]))) :
        return True
    return False

def train_model() -> str | None:
    """
    :returns: The path to the save location of the just trained model. `None` if
        the training failed.
    """
    check_feature_directory_existence("-t")
    training_info = read_csv(TRAINING_DATA_INFO_CSV_PATH, index_col="unique_id")
    training_feature_csv_paths = Path(TRAINING_FEATURES_PATH).glob("*.csv")
    # Collects all training feature CSVs into one big DataFrame used for
    # training.
    training_feature_data_frames: list[DataFrame] = []
    target_column_names = ["gender", "hold racket handed", "play years", "level"]
    # Uses a loop instead of a map to ensure that input_features and targets
    # line up, so that one row in input_features corresponds to one row at the
    # same position in features. Also builds up the groups list for use with
    # GroupKFold.
    target_rows: list[Series[int]] = []
    groups: list[int] = []
    for training_feature_csv_path in training_feature_csv_paths:
        training_feature_data_frames.append(read_csv(training_feature_csv_path))
        unique_id = int(training_feature_csv_path.stem)
        # Gets the row from training_info where the "unique_id" column equals
        # `unique_id` and gets the 4 columns needed for training. The filtering
        # by target_column_names must happen before iloc; otherwise, the
        # resulting Series will have a dtype of object instead of int64. This is
        # because a row in training_info contains a "cut_point" column, which
        # does not contain ints. If we filter out that column first then call
        # iloc, Pandas magically converts the resulting Series to have the dtype
        # of int64 instead of object.
        target_new_row = cast("Series[int]", training_info.loc[unique_id][target_column_names])
        # No need to use ingore_index=True here because training_info is read
        # from a single CSV.
        target_rows.append(target_new_row)
        groups.append(cast(int64, training_info.loc[unique_id, "player_id"]).item())
    # This is "X". This contains all the data that will be split into a training
    # set, validation set, and testing set below.
    all_input_features = concat(training_feature_data_frames, ignore_index=True)
    # This is "y". Same comment as above.
    all_targets = DataFrame(target_rows)
    group_k_fold = GroupKFold(shuffle=True) # pyright: ignore[reportCallIssue]
    group_shuffle_split = GroupShuffleSplit(1, test_size=0.2)
    # training_and_validation_input_features and training_and_validation_targets
    # will then be further split into the training set and the validation set.
    training_and_validation_indices: Long1D
    testing_indices: Long1D
    # Splits the whole dataset into a testing set, and the rest will be further
    # split into a training set and validation set.
    training_and_validation_indices, testing_indices = next(group_shuffle_split.split(all_input_features, all_targets, groups)) # pyright: ignore[reportUnknownMemberType]
    training_and_validation_input_features = all_input_features.iloc[training_and_validation_indices]
    training_and_validation_targets = all_targets.iloc[training_and_validation_indices]
    # `groups` is for `all_input_features`. After splitting `all_input_features`
    # etc. into a testing set and the training-and-validation set, we also need
    # to adjust the size of `groups` so that it only contains the group numbers
    # for the data in the training-and-validation set.
    training_and_validation_groups = [groups[i] for i in training_and_validation_indices]
    testing_input_features = all_input_features.iloc[testing_indices]
    testing_targets = all_targets.iloc[testing_indices]
    training_indices: Long1D
    validation_indices: Long1D
    # RandomForestClassifier.get_params returns the parameters passed to
    # RandomForestClassifier's constructor, but since the parameters all have
    # different types, it's impossible to not make the value's type Any without
    # generating more type errors when using this to recreate another
    # RandomForestClassifier with the same parameters.
    best_random_forest_classifier_parameters: dict[str, Any] = {}
    best_score = 0.0
    best_fold_index = 0
    # If we are so unlucky that none of the folds are usable, skip the final
    # retraining as sell.
    is_any_fold_usable = False
    for fold_index, (training_indices, validation_indices) in enumerate(group_k_fold.split(training_and_validation_input_features, training_and_validation_targets, training_and_validation_groups)): # pyright: ignore[reportUnknownMemberType]
        # The max random_state is 2^32 - 1 becuase sklearn internally uses
        # numpy.random.RandomState, which accepts a seed that ranges from 0 to
        # 2^32 - 1.
        random_state: int = randint(0, 2 ** 32 - 1)
        random_forest_classifier = RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_features="sqrt", random_state=random_state)
        print(f"Fold {fold_index + 1} with random_state {random_state}:")
        training_input_features = training_and_validation_input_features.iloc[training_indices]
        training_targets = training_and_validation_targets.iloc[training_indices]
        if not check_if_fold_is_usable(training_targets):
            print("The training set of this fold contains missing labels. Skipping.")
            continue
        else:
            is_any_fold_usable = True
        validation_input_features = training_and_validation_input_features.iloc[validation_indices]
        validation_targets = training_and_validation_targets.iloc[validation_indices]
        random_forest_classifier.fit(training_input_features, training_targets) # pyright: ignore[reportUnknownMemberType]
        validation_predictions = cast(list[Double2D], random_forest_classifier.predict_proba(validation_input_features)) # pyright: ignore[reportUnknownMemberType]
        testing_predictions = cast(list[Double2D], random_forest_classifier.predict_proba(testing_input_features)) # pyright: ignore[reportUnknownMemberType]
        print("Validation ROC AUC scores:")
        validation_scores = calculate_roc_auc_scores(validation_predictions, validation_targets)
        print_scores(*validation_scores)
        print("Testing ROC AUC scores")
        testing_scores = calculate_roc_auc_scores(testing_predictions, testing_targets)
        print_scores(*testing_scores)
        score = sum(validation_scores) / len(validation_scores)
        print(f"Overall score: {score}")
        if score > best_score:
            best_score = score
            best_fold_index = fold_index
            best_random_forest_classifier_parameters = cast(dict[str, Any], random_forest_classifier.get_params()) # pyright: ignore[reportUnknownMemberType]
        print() # Spaces out each fold
    if not is_any_fold_usable:
        print("None of the folds were usable. Skipping the final training. No model will be output.")
        return None
    print(f"The k-fold cross validation has ended. Now retraining the model with all training and validation data using the parameters from fold {best_fold_index + 1}:")
    final_random_forest_classifier = RandomForestClassifier(**best_random_forest_classifier_parameters)
    final_random_forest_classifier.fit(training_and_validation_input_features, training_and_validation_targets) # pyright: ignore[reportUnknownMemberType]
    testing_predictions = cast(list[Double2D], final_random_forest_classifier.predict_proba(testing_input_features))  # pyright: ignore[reportUnknownMemberType]
    print("Final ROC AUC scores:")
    testing_scores = calculate_roc_auc_scores(testing_predictions, testing_targets)
    print_scores(*testing_scores)
    final_score = sum(testing_scores) / len(testing_scores)
    print(f"Overall score: {final_score}")
    model_path = join(OUTPUT_BASE_PATH, f"{MODEL_NAME}_{final_score}.pkl")
    with open(model_path, "wb") as model_file:
        dump(final_random_forest_classifier, model_file, protocol=HIGHEST_PROTOCOL)
        print(f"Saved the final trained model to {model_path}.")
    return model_path
        
def generate_submission_csv(model_path: str):
    check_feature_directory_existence("-p")
    check_model_existence(model_path, "-p")
    with open(model_path, "rb") as model_file:
        random_forest_classifier: RandomForestClassifier = load(model_file)
    # Converts to a list so tqdm can know the total number of files.
    testing_feature_csv_paths = list(Path(TESTING_FEATURES_PATH).glob("*.csv"))
    # Collects all testing feature CSVs into a single DataFrame.
    testing_feature_data_frames: list[DataFrame] = []
    # predict_proba shouldn't see the unique_id column when predicting, but the
    # submission CSV needs it, so collect it separately.
    unique_ids: list[int] = []
    for testing_feature_csv_path in tqdm(testing_feature_csv_paths, "Reading the testing feature CSVs", unit="file"):
        testing_feature_data_frames.append(read_csv(testing_feature_csv_path))
        unique_ids.append(int(testing_feature_csv_path.stem))
    testing_features = concat(testing_feature_data_frames, ignore_index=True)
    testing_predictions = cast(list[Double2D], random_forest_classifier.predict_proba(testing_features)) # pyright: ignore[reportUnknownMemberType]
    prediction_dict: dict[str, list[int] | Double1D] = {
        "unique_id": unique_ids,
        # We use [: ,0] here because we want the probability of the player being
        # male, not female.
        "gender": cast(Double1D, testing_predictions[0][:, 0]),
        # Same here, we use [:, 0] because we want the probability of the player
        # being right-handed, not left-handed.
        "hold racket handed": cast(Double1D, testing_predictions[1][:, 0]),
        "play years_0": cast(Double1D, testing_predictions[2][:, 0]),
        "play years_1": cast(Double1D, testing_predictions[2][:, 1]),
        "play years_2": cast(Double1D, testing_predictions[2][:, 2]),
        "level_2": cast(Double1D, testing_predictions[3][:, 0]),
        "level_3": cast(Double1D, testing_predictions[3][:, 1]),
        "level_4": cast(Double1D, testing_predictions[3][:, 2]),
        "level_5": cast(Double1D, testing_predictions[3][:, 3])
    }
    prediction_data_frame = DataFrame(prediction_dict)
    prediction_data_frame.set_index("unique_id", inplace=True) # pyright: ignore[reportUnknownMemberType]
    prediction_data_frame.sort_index(inplace=True) # pyright: ignore[reportUnknownMemberType]
    summission_csv_path = join(OUTPUT_BASE_PATH, f"{SUBMISSION_CSV_NAME}.csv")
    prediction_data_frame.to_csv(summission_csv_path)

def main():
    argument_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="You need to specify at least one option; otherwise the script does nothing.")
    argument_parser.add_argument(
        "-g",
        "--generate-features",
        help="Whether to generate the features required for training and prediction and then exit. (Will delete the features directory previously generated.)",
        default=False,
        action=BooleanOptionalAction)
    argument_parser.add_argument(
        "-t",
        "--train-model",
        help="Whether to train the model and saves it to the disk.",
        default=False,
        action=BooleanOptionalAction)
    argument_parser.add_argument(
        "-p",
        "--generate-submission-csv",
        help="Whether to also use the model produced by supplying the -t flag to produce a CSV ready for submission to AI CUP. If you use this flag without the -t flag, you also need to specify the -m flag.",
        default=False,
        action=BooleanOptionalAction)
    argument_parser.add_argument(
        "-m",
        "--csv-generation-model-path",
        help="The path to the model to use when generating the submission CSV. Used when you use the -p flag without the -t flag. If the -p flag is used with the -t flag, this option does nothing.",
    )
    args = argument_parser.parse_args()

    model_path: str | None = None
    if args.generate_features:
        generate_features()
    if args.train_model:
        model_path = train_model()
    if args.generate_submission_csv:
        if model_path == None:
            if args.csv_generation_model_path != None:
                model_path = args.csv_generation_model_path
            else:
                print("You need to use the -m option to specify which model to generate the CSV with if you don't also use the -t flag.", file=stderr)
                exit(1)
        # Pyright is too dumb to see that model_path is either a str, or the
        # program has exited.
        generate_submission_csv(cast(str, model_path))

if __name__ == "__main__":
    main()
