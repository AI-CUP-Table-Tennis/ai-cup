import glob
import os
import ast
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fft import apply_fft_to_data_list, apply_rfft_to_data_list
from visualization import plot_overlay_metrics, plot_overlay_metrics_fft, plot_six_metrics, plot_six_metrics_fft

TRAINING_DATA_BASE_PATH = "39_Training_Dataset/train_data/"
TRAINING_DATA_METADATA_PATH = "39_Training_Dataset/train_info.csv"
TESTING_DATA_BASE_PATH = "39_Test_Dataset/test_data/"
TESTING_DATA_METADATA_PATH = "39_Test_Dataset/test_info.csv"


ALL_FILES = glob.glob(os.path.join(TRAINING_DATA_BASE_PATH, "*.txt"))

TITLES = ["X軸加速度(Ax)", "Y軸加速度(Ay)", "Z軸加速度(Az)", "X軸角速度(Gx)", "Y軸角速度(Gy)", "Z軸角速度(Gz)"]


OUTPUT_DIR = "output"
OUTPUT_FOLDER_PREFIX = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():

    file_list = [2, 22]

    data_list = []

    for file in file_list:
        file_path = os.path.join(TRAINING_DATA_BASE_PATH, f"{file}.txt")
        if os.path.exists(file_path):
            data = np.loadtxt(file_path)
            data_list.append(data)
        else:
            print(f"Warning: {file_path} not found.")

    print(f"Loaded {len(data_list)} files.")

    meta_data = pd.read_csv(TRAINING_DATA_METADATA_PATH)
    meta_data["cut_point"] = meta_data["cut_point"].apply(lambda s: np.fromstring(s.strip()[1:-1], sep=' ', dtype=int).tolist())


    # print(meta_data.head(), meta_data.columns)
    # print(meta_data["cut_point"][1])

    # 建立唯一輸出資料夾
    output_dir = f"{OUTPUT_DIR}/{OUTPUT_FOLDER_PREFIX}"

    os.makedirs(output_dir, exist_ok=True)

    # plot_six_metrics
    for i, data in enumerate(data_list):
        plot_six_metrics(
            data,
            TITLES,
            save_path=output_dir,
            prefix=f"{file_list[i]}",
            cut_points=meta_data["cut_point"][file_list[i]-1],
        )

    # plot_overlay_metrics
    plot_overlay_metrics(data_list, file_list, TITLES, output_dir + "/overlay_metrics")

    ######################

    # Apply FFT to data_list
    fftfreq, fft_data_list = apply_rfft_to_data_list(data_list)

    # 建立 FFT 輸出資料夾
    fft_output_dir = output_dir + "/fft"
    os.makedirs(fft_output_dir, exist_ok=True)

    # plot_six_metrics for FFT data
    for i, fft_data in enumerate(fft_data_list):
        plot_six_metrics_fft(
            fft_data,
            fftfreq,
            TITLES,
            save_path=fft_output_dir,
            prefix=f"{file_list[i]}",
        )

    # plot_overlay_metrics for FFT data
    plot_overlay_metrics_fft(
        fft_data_list,
        fftfreq,
        file_list,
        TITLES,
        fft_output_dir + "/overlay_metrics",
    )


if __name__ == "__main__":
    main()
