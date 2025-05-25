from os import mkdir
from typing import Final
from pandas import DataFrame, read_csv # pyright: ignore[reportUnknownVariableType]
from os.path import join, exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
from shutil import rmtree
from pathlib import Path
from itertools import chain
from dataclasses import dataclass
from numpy.fft import rfft
from tqdm import tqdm

TRAINING_DATA_BASE_PATH: Final = "./39_Training_Dataset"
TESTING_DATA_BASE_PATH: Final = "./39_Test_Dataset"
TRAINING_DATA_INFO_CSV_PATH: Final = join(TRAINING_DATA_BASE_PATH, "train_info.csv")
TESTING_DATA_INFO_CSV_PATH: Final = join(TESTING_DATA_BASE_PATH, "test_info.csv")
TRAINING_DATA_PATH: Final = join(TRAINING_DATA_BASE_PATH, "train_data")
TESTING_DATA_PATH: Final = join(TESTING_DATA_BASE_PATH, "test_data")
FEATURES_PATH: Final = "./output/random_forest_features"

@dataclass
class FeatureCsvRow:
    """
    Represents the fields contained in a row of the feature CSV generated using
    the -g flag.

    All `fft_*` attributes actually store the magnitudes of the FFT result. For
    example, `fft_acceleration_x_mean` is actually the mean of magnitudes of the
    complex fourier-transformed X acceleration values.
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
    fft_acceleration_max: float
    fft_angular_acceleration_max: float

def generate_features_for_single_data(data: DataFrame) -> list[FeatureCsvRow]:
    """
    Used by `generate_features` to generate the features of a single piece of
    training or testing data. Returns a list of `FeatureCsvRow` so that
    `generate_features` can write it to a CSV.

    :param data: The `DataFrame` read from one piece of training or testing data txt in the following manner:
    ```
    read_csv(<path>,
        sep=r"\\s+",
        names=["acceleration_x", "acceleration_y", "acceleration_z", "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"])
    ```
    """
    fft_data = data.apply(rfft).abs()
    feature = FeatureCsvRow(
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
        fft_acceleration_max=fft_data[["acceleration_x", "acceleration_y", "acceleration_z"]].pow(2).sum(axis=1).pow(0.5).max(), # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        fft_angular_acceleration_max=fft_data[["angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"]].pow(2).sum(axis=1).pow(0.5).max(), # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    return [feature]

def generate_features():
    """
    Generates a feature CSV for each piece of training and testing data and puts
    the results under `FEATURES_PATH`.
    """
    if exists(FEATURES_PATH):
        rmtree(FEATURES_PATH)
    # Can't use `else` here. We want to recreate the directory if the last `if`
    # deleted it.
    if not exists(FEATURES_PATH):
        mkdir(FEATURES_PATH)
    training_data_txt_paths = Path(TRAINING_DATA_PATH).glob("**/*.txt")
    testing_data_txt_paths = Path(TESTING_DATA_PATH).glob("**/*.txt")
    data_txt_paths = list(chain(training_data_txt_paths, testing_data_txt_paths)) # Uses list so we can know the total number of files to generate
    # Generates a feature CSV for each piece of training and test data
    for data_txt_path in tqdm(data_txt_paths, desc="Generating features", unit="file"):
        # Reads a training of testing .txt file as a DataFrame
        data = read_csv(data_txt_path,
            sep=r"\s+",
            names=["acceleration_x", "acceleration_y", "acceleration_z", "angular_acceleration_x", "angular_acceleration_y", "angular_acceleration_z"])
        features = generate_features_for_single_data(data)
        # Converts the list of FeatureCsvRows into a list of dicts so that
        # pandas can write them to a CSV
        features_dict_list = [vars(feature) for feature in features]
        features_data_frame = DataFrame(features_dict_list)
        features_filename = data_txt_path.stem
        features_data_frame.to_csv(join(FEATURES_PATH, f"{features_filename}.csv"), index=False)


def main():
    argument_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    argument_parser.add_argument("-g",
        "--generate-features",
        help="Whether to generate the features required for training and prediction and then exit. (Will delete the features directory previously generated.)",
        default=False,
        action=BooleanOptionalAction)
    args = argument_parser.parse_args()
    if args.generate_features:
        generate_features()
        exit(0)
    if not exists(FEATURES_PATH):
        print("You need to generate the features of the training and testing data using the -g option before running this.")
        exit(1)

if __name__ == "__main__":
    main()
