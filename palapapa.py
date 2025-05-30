from math import isnan
from os import mkdir
from sys import stderr
from typing import Any, Final, cast
from pandas import DataFrame, Series, read_csv, concat # pyright: ignore[reportUnknownVariableType]
from os.path import join, exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
from shutil import rmtree
from pathlib import Path
from dataclasses import dataclass
from numpy.fft import rfft
from numpy import float64, int64, object_
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score # pyright: ignore[reportUnknownVariableType]
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tqdm import tqdm
from pickle import dump, load, HIGHEST_PROTOCOL
from type_aliases import Double2D, Long1D
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
    acceleration_max: float
    angular_acceleration_max: float
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
    fft_acceleration_max: float
    fft_angular_acceleration_max: float

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
        acceleration_max=data[["acceleration_x", "acceleration_y", "acceleration_z"]].pow(2).sum(axis=1).pow(0.5).max(), # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        angular_acceleration_max=data[["angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"]].pow(2).sum(axis=1).pow(0.5).max(), # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
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
        fft_acceleration_max=fft_data[["acceleration_x", "acceleration_y", "acceleration_z"]].pow(2).sum(axis=1).pow(0.5).max(), # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        fft_angular_acceleration_max=fft_data[["angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"]].pow(2).sum(axis=1).pow(0.5).max(), # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
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

def print_scores(
    predictions: list[Double2D],
    targets: DataFrame):
    """
    Prints the ROC AUC scores of each of the 4 prediction targets in a training
    run.

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
           third `ndarray` in `predictions`. Each cell contains either a 1, 2,
           or 3. 1 indicates that the player has low experience. 2 indicates
           that the player has medium experience. 3 indicates that the player
           has high experience.
        4. A column named "level" contains the prediction targets for the fourth
           `ndarray` in `predictions`. Each cell contains either a 2, 3, 4, or
           5. 2 indicates that the player is of level 2. 3 indicates that the
           player is of level 3. 4 indicates that the player is of level 4. 5
           indicates that the player is of level 5.
    """
    print(f"Gender ROC AUC score: {roc_auc_score(targets["gender"], predictions[0][:, 1])}")
    print(f"Handedness ROC AUC score: {roc_auc_score(targets["hold racket handed"], predictions[1][:, 1])}")
    print(f"Experience ROC AUC score: {roc_auc_score(targets["play years"], predictions[2], average="micro", multi_class="ovr")}")
    print(f"Level validation ROC AUC score: {roc_auc_score(targets["level"], predictions[3], average="micro", multi_class="ovr")}")

def calculate_overall_score(
    testing_predictions: list[Double2D],
    testing_targets: DataFrame) -> float:
    scores: list[float] = []
    # There is a chance that some of these will fail to compute and throw
    # ValueError because sometimes the testing set just won't have every
    # possible label in the dataset because some labels appear very rarely. This
    # is only compounded by the fact that the testing set only takes up a small
    # portion of the dataset. So if any of the following fail to compute, we
    # will just ignore it and calculate the average score without them. The
    # conversion to float is needed because the return type of roc_auc_score is
    # actually float | float16 | float32 | float64. The first two scores don't
    # need a try-except statement because for some reason in the binary case,
    # roc_auc_score just prints a warning to the terminal and returns NaN. It
    # only throws for the multi-class case.
    gender_score = float(roc_auc_score(testing_targets["gender"], testing_predictions[0][:, 1]))
    if not isnan(gender_score):
        scores.append(gender_score)
    handedness_score = float(roc_auc_score(testing_targets["hold racket handed"], testing_predictions[1][:, 1]))
    if not isnan(handedness_score):
        scores.append(handedness_score)
    # No need to check for NaNs for the following two cases since they just
    # throw without returning any value.
    try:
        scores.append(float(roc_auc_score(testing_targets["play years"], testing_predictions[2], average="micro", multi_class="ovr")))
    except ValueError:
        pass
    try:
        scores.append(float(roc_auc_score(testing_targets["level"], testing_predictions[3], average="micro", multi_class="ovr")))
    except ValueError:
        pass
    return sum(scores) / len(scores)

def train_model() -> str:
    """
    :returs: The path to the save location of the just trained model.
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
    for fold_index, (training_indices, validation_indices) in enumerate(group_k_fold.split(training_and_validation_input_features, training_and_validation_targets, training_and_validation_groups)): # pyright: ignore[reportUnknownMemberType]
        # The max random_state is 2^32 - 1 becuase sklearn internally uses
        # numpy.random.RandomState, which accepts a seed that ranges from 0 to
        # 2^32 - 1.
        random_state: int = randint(0, 2 ** 32 - 1)
        random_forest_classifier = RandomForestClassifier(max_features="sqrt", random_state=random_state)
        print(f"Fold {fold_index + 1} with random_state {random_state}:")
        training_input_features = training_and_validation_input_features.iloc[training_indices]
        training_targets = training_and_validation_targets.iloc[training_indices]
        validation_input_features = training_and_validation_input_features.iloc[validation_indices]
        validation_targets = training_and_validation_targets.iloc[validation_indices]
        random_forest_classifier.fit(training_input_features, training_targets) # pyright: ignore[reportUnknownMemberType]
        validation_predictions = cast(list[Double2D], random_forest_classifier.predict_proba(validation_input_features)) # pyright: ignore[reportUnknownMemberType]
        testing_predictions = cast(list[Double2D], random_forest_classifier.predict_proba(testing_input_features)) # pyright: ignore[reportUnknownMemberType]
        # See the comment inside calculate_overall_score to know why this try is
        # needed.
        try:
            print("Validation ROC AUC scores:")
            print_scores(validation_predictions, validation_targets)
            print("Testing ROC AUC scores")
            print_scores(testing_predictions, testing_targets)
        except ValueError:
            pass
        score = calculate_overall_score(validation_predictions, validation_targets)
        print(f"Overall score: {score}")
        if score > best_score:
            best_score = score
            best_random_forest_classifier_parameters = cast(dict[str, Any], random_forest_classifier.get_params()) # pyright: ignore[reportUnknownMemberType]
        print() # Spaces out each fold
    print("The k-fold cross validation has ended. Now retraining the model with all training and validation data:")
    final_random_forest_classifier = RandomForestClassifier(**best_random_forest_classifier_parameters)
    final_random_forest_classifier.fit(training_and_validation_input_features, training_and_validation_targets) # pyright: ignore[reportUnknownMemberType]
    testing_predictions = cast(list[Double2D], final_random_forest_classifier.predict_proba(testing_input_features))  # pyright: ignore[reportUnknownMemberType]
    try:
        print("Final ROC AUC scores:")
        print_scores(testing_predictions, testing_targets)
    except ValueError:
        pass
    final_score = calculate_overall_score(testing_predictions, testing_targets)
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
    prediction_csv_rows: list[Series[int | float]] = []
    for testing_feature_csv_path in tqdm(testing_feature_csv_paths, "Generating the submission CSV", unit="row"):
        testing_features = read_csv(testing_feature_csv_path)
        testing_predictions = cast(list[Double2D], random_forest_classifier.predict_proba(testing_features)) # pyright: ignore[reportUnknownMemberType]
        prediction_dict: dict[str, int | float] = {
            "unique_id": int(testing_feature_csv_path.stem),
            # Since we are generating predictions one at a time,
            # testing_predictions contains ndarrays that all have only one row.
            # We use [0 ,0] here because we want the probability of the player
            # being male, not female.
            "gender": testing_predictions[0][0, 0],
            # Same here, we use [0, 0] because we want the probability of the
            # player being right-handed, not left-handed.
            "hold racket handed": testing_predictions[1][0, 0],
            "play years_0": testing_predictions[2][0, 0],
            "play years_1": testing_predictions[2][0, 1],
            "play years_2": testing_predictions[2][0, 2],
            "level_2": testing_predictions[3][0, 0],
            "level_3": testing_predictions[3][0, 1],
            "level_4": testing_predictions[3][0, 2],
            "level_5": testing_predictions[3][0, 3]
        }
        # Uses dtype=object_ so that the unique_id column won't have decimal
        # points.
        prediction_csv_rows.append(Series(prediction_dict, dtype=object_))
    prediction_data_frame = DataFrame(prediction_csv_rows)
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
