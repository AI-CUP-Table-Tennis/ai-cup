from numpy import hamming, pad, log10, abs
from typing import cast
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from type_aliases import Double1D, DoubleNBy6

def apply_rfft_to_data_list(data_list: list[DoubleNBy6], fs: float = 85) -> tuple[Double1D, list[DoubleNBy6]]:
    """
    對 data_list 中的每筆 (T,6) 時域資料：
      1. 去均值 (DC removal)
      2. 套 Hamming 窗
      3. 計算 rFFT 取得正頻率幅值
      4. 轉成 dB (20*log10)

    參數
    -------
    data_list : list[np.ndarray]
        每筆 shape = (T,6) 的時域資料
    fs : float
        取樣頻率 (Hz)，決定頻率刻度

    回傳
    -------
    freqs : 1-D ndarray
        真實頻率刻度 (Hz)，長度 = T//2 + 1
    fft_mag_list : list[np.ndarray]
        每筆 shape = (T//2+1, 6) 的 dB 幅值頻譜
    """
    # Determine the longest sequence length among all files
    max_T = max(d.shape[0] for d in data_list)
    # Pre‑compute a full‑length Hamming window for padding cases
    full_win = hamming(max_T)[:, None]
    fft_mag_list: list[DoubleNBy6] = []

    for data in data_list:
        # 1. 去均值
        data_centered: DoubleNBy6 = data - data.mean(axis=0, keepdims=True)

        # 2. 如有需要，zero‑pad 到 max_T
        T_cur = data_centered.shape[0]
        if T_cur < max_T:
            pad_len = max_T - T_cur
            data_padded = pad(data_centered, ((0, pad_len), (0, 0)), mode="constant")
        else:
            data_padded = data_centered  # already the longest one

        # 3. 套 full‑length Hamming 窗
        windowed = data_padded * full_win

        # 4. rFFT -> (N_freq, 6)
        spec = rfft(windowed, axis=0)
        mag = cast(DoubleNBy6, 20 * log10(abs(spec) + 1e-12))  # dB，避免 log(0)

        fft_mag_list.append(mag)

    # 共用一條以最長序列為基準的頻率軸
    freqs = rfftfreq(max_T, d=1.0 / fs)
    return freqs, fft_mag_list


def apply_fft_to_data_list(data_list: list[DoubleNBy6], fs: float = 85) -> tuple[Double1D, list[DoubleNBy6]]:
    """
    對 data_list 中的每筆 (T,6) 時域資料應用標準FFT處理。
    與 apply_rfft_to_data_list 不同，此函數保留完整頻譜（正負頻率）。

    參數
    -------
    data_list : list[np.ndarray]
        每筆 shape = (T,6) 的時域資料
    fs : float
        取樣頻率 (Hz)，決定頻率刻度

    回傳
    -------
    freqs : 1-D ndarray
        完整頻率刻度 (Hz)，長度 = max_T
    fft_mag_list : list[np.ndarray]
        每筆 shape = (T, 6) 的幅值頻譜
    """

    # 找出最長序列長度
    max_T = max(d.shape[0] for d in data_list)
    # 預先計算完整長度的 Hamming 窗
    full_win = hamming(max_T)[:, None]
    fft_mag_list: list[DoubleNBy6] = []

    for data in data_list:
        # 1. 去均值
        data_centered: DoubleNBy6 = data - data.mean(axis=0, keepdims=True)

        # 2. 如有需要，zero-pad 到 max_T
        T_cur = data_centered.shape[0]
        if T_cur < max_T:
            pad_len = max_T - T_cur
            data_padded = pad(data_centered, ((0, pad_len), (0, 0)), mode="constant")
        else:
            data_padded = data_centered  # 已經是最長序列

        # 3. 套 full-length Hamming 窗
        windowed = data_padded * full_win

        # 4. FFT -> (N_freq, 6)，包含正負頻率
        spec = fft(windowed, axis=0)
        mag = cast(DoubleNBy6, abs(spec))
        fft_mag_list.append(mag)

    # 共用一條以最長序列為基準的頻率軸（包含正負頻率）
    freqs = fftfreq(max_T, d=1.0 / fs)
    return freqs, fft_mag_list
