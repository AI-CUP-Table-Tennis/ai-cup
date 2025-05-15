from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np

LINE_WIDTH = 0.7
FONT_FAMILY = ["SF Pro", "Noto Sans TC"]

plt.rcParams["font.family"] = FONT_FAMILY


# 新增函式：繪製六個維度的資料於一張圖中的六個子圖
def plot_six_metrics(
    data: np.ndarray[Any, np.dtype[Any]],
    titles: List[str],
    save_path: Optional[str] = None,
    prefix: str = "plot",
) -> None:
    """
    Plots all 6 metrics in one figure with subplots.

    Parameters:
    - data: ndarray of shape (T, 6)
    - titles: list of 6 strings for y-axis labels
    - save_path: if specified, save the figure to this path
    - prefix: prefix for the saved file name
    """

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=False)
    axes = axes.flatten()
    for i in range(6):
        ax = axes[i]
        ax.plot(data[:, i], linewidth=LINE_WIDTH)
        ax.set_ylabel(titles[i])
        ax.grid(True)
        ax.set_xlabel("Time Step")
    _ = fig.suptitle(f"File {prefix} Metrics")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(f"{save_path}/{prefix}.png")
    plt.close(fig)


def plot_overlay_metrics(
    data_list: np.ndarray[Any, np.dtype[Any]],
    file_list: List[str],
    titles: List[str],
    save_path: Optional[str],
):
    """
    Plots overlay of same metrics from different data sources.

    Parameters:
    - data_list: list of ndarray, each shape (T, 6)
    - file_list: list of corresponding file labels
    - titles: list of y-axis labels for each metric
    - save_path: where to save the output figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(6):
        ax = axes[i]
        for idx, data in enumerate(data_list):
            ax.plot(data[:, i], label=f"File {file_list[idx]}", linewidth=LINE_WIDTH)
        ax.set_title(titles[i])
        ax.set_xlabel("Time Step")
        ax.set_ylabel(titles[i])
        ax.legend()
    _ = fig.suptitle("Overlay Metrics Comparison")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}")
    plt.close()


def plot_six_metrics_fft(
    data: np.ndarray[Any, np.dtype[Any]],
    fftfreq: np.ndarray[Any, np.dtype[Any]],
    titles: List[str],
    save_path: Optional[str] = None,
    prefix: str = "plot",
) -> None:
    """
    Plots all 6 metrics in one figure with subplots.

    Parameters:
    - data: ndarray of shape (T, 6)
    - titles: list of 6 strings for y-axis labels
    - save_path: if specified, save the figure to this path
    - prefix: prefix for the saved file name
    """

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=False)
    axes = axes.flatten()
    for i in range(6):
        ax = axes[i]
        ax.semilogy(fftfreq, data[:, i], linewidth=LINE_WIDTH)
        ax.set_title(titles[i])
        ax.set_ylabel(titles[i])
        ax.set_yticks([4 * 1e1, 5 * 1e1, 6 * 1e1, 7 * 1e1, 8 * 1e1, 9 * 1e1, 10 * 1e1])
        ax.grid(True)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")

    _ = fig.suptitle(f"File {prefix} Metrics")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(f"{save_path}/{prefix}.png")
    plt.close(fig)


def plot_overlay_metrics_fft(
    data_list: List[np.ndarray[Any, np.dtype[Any]]],
    fftfreq: np.ndarray[Any, np.dtype[Any]],
    file_list: List[str],
    titles: List[str],
    save_path: Optional[str],
):
    """
    Plots overlay of same metrics from different data sources.

    Parameters:
    - data_list: list of ndarray, each shape (T, 6)
    - file_list: list of corresponding file labels
    - titles: list of y-axis labels for each metric
    - save_path: where to save the output figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(6):
        ax = axes[i]
        for idx, data in enumerate(data_list):
            ax.semilogy(fftfreq, data[:, i], label=f"File {file_list[idx]}", linewidth=LINE_WIDTH)
        ax.set_title(titles[i])
        ax.set_ylabel(titles[i])
        ax.set_yticks([4 * 1e1, 5 * 1e1, 6 * 1e1, 7 * 1e1, 8 * 1e1, 9 * 1e1, 10 * 1e1])
        ax.grid(True)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")

    _ = fig.suptitle("Overlay Metrics Comparison")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}")
    plt.close()
