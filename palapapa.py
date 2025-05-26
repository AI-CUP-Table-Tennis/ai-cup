from os import mkdir
from typing import Final, cast
from pandas import DataFrame, Series, read_csv, concat # pyright: ignore[reportUnknownVariableType]
from os.path import join, exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
from shutil import rmtree
from pathlib import Path
from dataclasses import dataclass
from numpy.fft import rfft
from numpy import float64, int64
from tqdm import tqdm

TRAINING_DATA_BASE_PATH: Final = "./39_Training_Dataset"
TESTING_DATA_BASE_PATH: Final = "./39_Test_Dataset"
TRAINING_DATA_INFO_CSV_PATH: Final = join(TRAINING_DATA_BASE_PATH, "train_info.csv")
TESTING_DATA_INFO_CSV_PATH: Final = join(TESTING_DATA_BASE_PATH, "test_info.csv")
TRAINING_DATA_PATH: Final = join(TRAINING_DATA_BASE_PATH, "train_data")
TESTING_DATA_PATH: Final = join(TESTING_DATA_BASE_PATH, "test_data")
FEATURES_BASE_PATH: Final = "./output/random_forest_features"
TRAINING_FEATURES_PATH: Final = join(FEATURES_BASE_PATH, "train")
TESTING_FEATURES_PATH: Final = join(FEATURES_BASE_PATH, "test")
MODEL_PATH: Final = "./output/random_forest.pkl"

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
        print(f"You need to generate the features of the training and testing data using the -g option before using the {option_name} option.")
        exit(1)

def check_model_existence(option_name: str):
    if not exists(MODEL_PATH):
        print(f"You need to train the model using the -t option before using the {option_name} option.")
        exit(1)

def train_model():
    check_feature_directory_existence("-t")
    training_info = read_csv(TRAINING_DATA_INFO_CSV_PATH)
    training_feature_csv_paths = Path(TRAINING_FEATURES_PATH).glob("*.csv")
    # Collects all training feature CSVs into one big DataFrame used for
    # training.
    training_feature_data_frames: list[DataFrame] = []
    target_column_names = ["gender", "hold racket handed", "play years", "level"]
    # Uses a loop instead of a map to ensure that input_features and targets
    # line up, so that one row in input_features corresponds to one row at the
    # same position in features.
    target_rows: list[Series[int]] = []
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
        target_new_row = cast("Series[int]", training_info[training_info["unique_id"] == unique_id][target_column_names].iloc[0])
        # No need to use ingore_index=True here because training_info is read
        # from a single CSV.
        target_rows.append(target_new_row)
    # This is "X".
    input_features = concat(training_feature_data_frames, ignore_index=True)
    # This is "y".
    targets = DataFrame(target_rows)
    print(input_features)
    print(targets)

def generate_submission_csv():
    check_feature_directory_existence("-p")
    check_model_existence("-p")

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
        help="Whether to use the model produced by supplying the -t flag to produce a CSV ready for submission to AI CUP.",
        default=False,
        action=BooleanOptionalAction)
    args = argument_parser.parse_args()
    if args.generate_features:
        generate_features()
    if args.train_model:
        train_model()
    if args.generate_submission_csv:
        generate_submission_csv()

if __name__ == "__main__":
    main()
