from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal


# =========================================================
# 기본 경로
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "data"
    / "60sec103bpmrest02.csv"
)


# =========================================================
# Band-pass Filter
# =========================================================

def bandpass_filter(
    data,
    lowcut,
    highcut,
    fs,
    order=4
):
    # Nyquist Frequency
    nyquist = fs / 2

    # Butterworth Filter에서 사용할
    # 정규화된 주파수 값
    low = lowcut / nyquist
    high = highcut / nyquist

    # Band-pass Filter 생성
    b, a = signal.butter(
        order,
        [low, high],
        btype="band"
    )

    # 앞/뒤 방향으로 필터를 적용해서
    # 위상 지연을 최소화
    filtered_data = signal.filtfilt(
        b,
        a,
        data
    )

    return filtered_data


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. CSV 읽기
    # -----------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    print(df.head())
    print()

    print(f"데이터 개수: {len(df)}")


    # -----------------------------------------------------
    # 2. Timestamp / Green 값 가져오기
    # -----------------------------------------------------

    timestamps = df["timestamp"]
    g_values = df["g"]


    # -----------------------------------------------------
    # 3. Sampling Frequency 계산
    # -----------------------------------------------------

    duration = (
        timestamps.iloc[-1]
        - timestamps.iloc[0]
    )

    fs = (
        (len(timestamps) - 1)
        / duration
    )

    print(f"측정 시간: {duration:.2f} sec")
    print(f"Sampling Frequency: {fs:.2f} Hz")


    # -----------------------------------------------------
    # 4. Detrending
    # -----------------------------------------------------

    g_detrended = signal.detrend(
        g_values
    )


    # -----------------------------------------------------
    # 5. Band-pass Filter
    #
    # 0.75 Hz = 45 BPM
    # 3.00 Hz = 180 BPM
    # -----------------------------------------------------

    LOWCUT = 0.75
    HIGHCUT = 3.0

    g_filtered = bandpass_filter(
        g_detrended,
        lowcut=LOWCUT,
        highcut=HIGHCUT,
        fs=fs,
        order=4
    )


    # -----------------------------------------------------
    # 6. Welch PSD
    # -----------------------------------------------------

    frequencies, power = signal.welch(
        g_filtered,
        fs=fs,
        nperseg=min(
            256,
            len(g_filtered)
        )
    )


    # -----------------------------------------------------
    # 7. 심박수 범위만 선택
    # -----------------------------------------------------

    valid_range = (
        (frequencies >= LOWCUT)
        & (frequencies <= HIGHCUT)
    )

    valid_freqs = frequencies[
        valid_range
    ]

    valid_power = power[
        valid_range
    ]


    # -----------------------------------------------------
    # 8. 가장 강한 주파수 찾기
    # -----------------------------------------------------

    peak_index = np.argmax(
        valid_power
    )

    peak_frequency = valid_freqs[
        peak_index
    ]


    # -----------------------------------------------------
    # 9. Hz -> BPM 변환
    # -----------------------------------------------------

    estimated_bpm = (
        peak_frequency * 60
    )

    print()
    print(
        f"Peak Frequency: "
        f"{peak_frequency:.3f} Hz"
    )

    print(
        f"Estimated BPM: "
        f"{estimated_bpm:.1f}"
    )


    # =====================================================
    # 그래프 1
    # Band-pass Filter 결과
    # =====================================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        timestamps,
        g_filtered
    )

    plt.xlabel(
        "Time (sec)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Band-pass Filtered Green Signal"
    )

    plt.grid()

    plt.tight_layout()

    plt.show()


    # =====================================================
    # 그래프 2
    # Welch PSD
    # =====================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        valid_freqs,
        valid_power
    )

    # 가장 강한 주파수 표시
    plt.axvline(
        peak_frequency,
        linestyle="--",
        label=f"{estimated_bpm:.1f} BPM"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Power"
    )

    plt.title(
        "Welch Power Spectral Density"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.show()


# =========================================================
# 프로그램 시작
# =========================================================

if __name__ == "__main__":
    main()