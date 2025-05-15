import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from fft import apply_fft_to_data_list, apply_rfft_to_data_list
from visualization import plot_overlay_metrics, plot_overlay_metrics_fft, plot_six_metrics, plot_six_metrics_fft

DATA_BASE_PATH = "39_Training_Dataset/train_data/"

ALL_FILES = glob.glob(os.path.join(DATA_BASE_PATH, "*.txt"))

TITLE = ["X軸加速度(Ax)", "Y軸加速度(Ay)", "Z軸加速度(Az)", "X軸角速度(Gx)", "Y軸角速度(Gy)", "Z軸角速度(Gz)"]

OUTPUT_DIR = "output"
OUTPUT_FOLDER_PREFIX = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():

    file_list = [2, 220, 589]

    data_list = []

    for file in file_list:
        file_path = os.path.join(DATA_BASE_PATH, f"{file}.txt")
        if os.path.exists(file_path):
            data = np.loadtxt(file_path)
            data_list.append(data)
        else:
            print(f"Warning: {file_path} not found.")

    print(f"Loaded {len(data_list)} files.")

    # 建立唯一輸出資料夾
    output_dir = f"{OUTPUT_DIR}/{OUTPUT_FOLDER_PREFIX}"
    os.makedirs(output_dir, exist_ok=True)

    # plot_six_metrics
    for i, data in enumerate(data_list):
        plot_six_metrics(data, TITLE, save_path=output_dir, prefix=f"{file_list[i]}")

    # plot_overlay_metrics
    plot_overlay_metrics(data_list, file_list, TITLE, output_dir + "/overlay_metrics")

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
            TITLE,
            save_path=fft_output_dir,
            prefix=f"{file_list[i]}",
        )

    # plot_overlay_metrics for FFT data
    plot_overlay_metrics_fft(
        fft_data_list,
        fftfreq,
        file_list,
        TITLE,
        fft_output_dir + "/overlay_metrics",
    )


if __name__ == "__main__":
    main()
