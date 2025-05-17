import matplotlib.pyplot as plt
from type_aliases import Double1D, DoubleNBy6, Axes2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes

LINE_WIDTH = 0.7
FONT_FAMILIES = [
    "SF Pro",
    "Noto Sans TC",
    "Noto Sans CJK HK",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC"]

# Don't set font.family directly. This makes it so that Matplotlib tries to find
# if any of the fonts in font.sans-serif exist before failing.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = FONT_FAMILIES

# 新增函式：繪製六個維度的資料於一張圖中的六個子圖
def plot_six_metrics(
    data: DoubleNBy6,
    titles: list[str],
    save_path: str | None = None,
    prefix: str = "plot",
    cut_points: list[int] | None = None,
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
    fig: Figure
    axes: Axes2D
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=False) # pyright: ignore[reportUnknownMemberType]
    axes_flattened = axes.flatten()
    for i in range(6):
        ax: Axes = axes_flattened[i]
        ax.plot(data[:, i], linewidth=LINE_WIDTH) # pyright: ignore[reportUnknownMemberType]
        ax.set_ylabel(titles[i]) # pyright: ignore[reportUnknownMemberType]
        ax.grid(True) # pyright: ignore[reportUnknownMemberType]
        ax.set_xlabel("Time Step") # pyright: ignore[reportUnknownMemberType]
        for cut_point in cut_points or []:
            ax.axvline(x=cut_point, color='red', linestyle='-', linewidth=0.5) # pyright: ignore[reportUnknownMemberType]
    fig.suptitle(f"File {prefix} Metrics") # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(f"{save_path}/{prefix}.png") # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


def plot_overlay_metrics(
    data_list: list[DoubleNBy6],
    file_list: list[int],
    titles: list[str],
    save_path: str | None,
):
    """
    Plots overlay of same metrics from different data sources.

    Parameters:
    - data_list: list of ndarray, each shape (T, 6)
    - file_list: list of corresponding file labels
    - titles: list of y-axis labels for each metric
    - save_path: where to save the output figure
    """
    fig: Figure
    axes: Axes2D
    fig, axes = plt.subplots(3, 2, figsize=(15, 10)) # pyright: ignore[reportUnknownMemberType]
    axes_flattened = axes.flatten()
    for i in range(6):
        ax = axes_flattened[i]
        for idx, data in enumerate(data_list):
            ax.plot(data[:, i], label=f"File {file_list[idx]}", linewidth=LINE_WIDTH)
        ax.set_title(titles[i])
        ax.set_xlabel("Time Step")
        ax.set_ylabel(titles[i])
        ax.legend()
    fig.suptitle("Overlay Metrics Comparison") # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}") # pyright: ignore[reportUnknownMemberType]
    plt.close()


def plot_six_metrics_fft(
    data: DoubleNBy6,
    fftfreq: Double1D,
    titles: list[str],
    save_path: str | None = None,
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
    fig: Figure
    axes: Axes2D
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=False) # pyright: ignore[reportUnknownMemberType]
    axes_flattened = axes.flatten()
    for i in range(6):
        ax = axes_flattened[i]
        ax.semilogy(fftfreq, data[:, i], linewidth=LINE_WIDTH)
        ax.set_title(titles[i])
        ax.set_ylabel(titles[i])
        ax.set_yticks([4 * 1e1, 5 * 1e1, 6 * 1e1, 7 * 1e1, 8 * 1e1, 9 * 1e1, 10 * 1e1])
        ax.grid(True)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")

    fig.suptitle(f"File {prefix} Metrics") # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(f"{save_path}/{prefix}.png") # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


def plot_overlay_metrics_fft(
    data_list: list[DoubleNBy6],
    fftfreq: Double1D,
    file_list: list[int],
    titles: list[str],
    save_path: str | None,
):
    """
    Plots overlay of same metrics from different data sources.

    Parameters:
    - data_list: list of ndarray, each shape (T, 6)
    - file_list: list of corresponding file labels
    - titles: list of y-axis labels for each metric
    - save_path: where to save the output figure
    """
    fig: Figure
    axes: Axes2D
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=False) # pyright: ignore[reportUnknownMemberType]
    axes_flattened = axes.flatten()
    for i in range(6):
        ax = axes_flattened[i]
        for idx, data in enumerate(data_list):
            ax.semilogy(fftfreq, data[:, i], label=f"File {file_list[idx]}", linewidth=LINE_WIDTH)
        ax.set_title(titles[i])
        ax.set_ylabel(titles[i])
        ax.set_yticks([4 * 1e1, 5 * 1e1, 6 * 1e1, 7 * 1e1, 8 * 1e1, 9 * 1e1, 10 * 1e1])
        ax.grid(True)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")

    fig.suptitle("Overlay Metrics Comparison") # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}") # pyright: ignore[reportUnknownMemberType]
    plt.close()
