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
    / "60sec75bpmrest01.csv"
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

    # Butterworth Filter용 정규화 주파수
    low = lowcut / nyquist
    high = highcut / nyquist

    # Band-pass Filter 생성
    b, a = signal.butter(
        order,
        [low, high],
        btype="band"
    )

    # 앞/뒤 방향으로 적용해서 위상 지연 최소화
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
    # 8. Local Peak 찾기
    # -----------------------------------------------------

    peak_indices, _ = signal.find_peaks(
        valid_power
    )

    if len(peak_indices) == 0:
        print("Peak를 찾지 못했습니다.")
        return


    # Peak들의 Power
    peak_powers = valid_power[
        peak_indices
    ]


    # -----------------------------------------------------
    # 9. Power가 높은 순으로 정렬
    # -----------------------------------------------------

    sorted_order = np.argsort(
        peak_powers
    )[::-1]


    # -----------------------------------------------------
    # 10. Top 3 Peak 출력
    # -----------------------------------------------------

    top_n = min(
        3,
        len(sorted_order)
    )

    print()
    print("========== Top PSD Peaks ==========")

    for rank in range(top_n):

        local_peak_index = peak_indices[
            sorted_order[rank]
        ]

        frequency = valid_freqs[
            local_peak_index
        ]

        peak_power = valid_power[
            local_peak_index
        ]

        bpm = frequency * 60

        print(
            f"{rank + 1}. "
            f"{frequency:.3f} Hz "
            f"-> {bpm:.1f} BPM "
            f"(Power: {peak_power:.4f})"
        )


    # -----------------------------------------------------
    # 11. 가장 강한 Peak 선택
    # -----------------------------------------------------

    strongest_peak_position = sorted_order[0]

    strongest_peak_index = peak_indices[
        strongest_peak_position
    ]

    peak_frequency = valid_freqs[
        strongest_peak_index
    ]

    estimated_bpm = (
        peak_frequency * 60
    )

    print()
    print(
        f"Selected Peak Frequency: "
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

    # PSD 전체 선
    plt.plot(
        valid_freqs,
        valid_power
    )


    # 검출된 Local Peak 표시
    plt.scatter(
        valid_freqs[peak_indices],
        valid_power[peak_indices],
        marker="o",
        label="Detected Peaks"
    )


    # 최종 선택된 Peak 표시
    plt.axvline(
        peak_frequency,
        linestyle="--",
        label=f"Selected: {estimated_bpm:.1f} BPM"
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