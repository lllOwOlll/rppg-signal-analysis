import csv
import time
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# 기본 경로
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# =========================================================
# 설정
# =========================================================

CAMERA_INDEX = 1

MEASURE_SECONDS = 60

# 화면 중앙에서 사용할 ROI 비율
ROI_WIDTH_RATIO = 0.40
ROI_HEIGHT_RATIO = 0.40


def main():

    # -----------------------------------------------------
    # CSV 파일 경로
    # -----------------------------------------------------

    file_time = time.strftime("%Y%m%d_%H%M%S")

    csv_path = (
        DATA_DIR
        / f"60sec_environment_{file_time}.csv"
    )


    # -----------------------------------------------------
    # 데이터 저장
    # -----------------------------------------------------

    timestamps = []
    r_values = []
    g_values = []
    b_values = []


    # -----------------------------------------------------
    # 카메라 연결
    # -----------------------------------------------------

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return


    start_time = time.perf_counter()


    # -----------------------------------------------------
    # 측정
    # -----------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break


        current_time = (
            time.perf_counter()
            - start_time
        )


        # =================================================
        # 중앙 ROI 계산
        # =================================================

        height, width, _ = frame.shape

        roi_width = int(
            width * ROI_WIDTH_RATIO
        )

        roi_height = int(
            height * ROI_HEIGHT_RATIO
        )


        center_x = width // 2
        center_y = height // 2


        x1 = center_x - roi_width // 2
        x2 = center_x + roi_width // 2

        y1 = center_y - roi_height // 2
        y2 = center_y + roi_height // 2


        # =================================================
        # ROI 추출
        # =================================================

        roi = frame[
            y1:y2,
            x1:x2
        ]


        # =================================================
        # ROI 평균 RGB 계산
        # =================================================

        mean_b, mean_g, mean_r, _ = cv2.mean(
            roi
        )


        # =================================================
        # 데이터 저장
        # =================================================

        timestamps.append(
            current_time
        )

        r_values.append(
            mean_r
        )

        g_values.append(
            mean_g
        )

        b_values.append(
            mean_b
        )


        # =================================================
        # 화면 표시
        # =================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Time: {current_time:.1f} / {MEASURE_SECONDS} sec",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            "Environment Capture",
            frame
        )


        # =================================================
        # Console
        # =================================================

        print(
            f"Time: {current_time:.3f}, "
            f"R: {mean_r:.2f}, "
            f"G: {mean_g:.2f}, "
            f"B: {mean_b:.2f}"
        )


        # =================================================
        # 60초 후 자동 종료
        # =================================================

        if current_time >= MEASURE_SECONDS:
            break


        # q 키로 강제 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # =====================================================
    # 카메라 종료
    # =====================================================

    cap.release()
    cv2.destroyAllWindows()


    # =====================================================
    # CSV 저장
    # =====================================================

    if len(timestamps) == 0:
        print("저장할 데이터가 없습니다.")
        return


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "r",
            "g",
            "b"
        ])

        for t, r, g, b in zip(
            timestamps,
            r_values,
            g_values,
            b_values
        ):
            writer.writerow([
                t,
                r,
                g,
                b
            ])


    # =====================================================
    # 결과 출력
    # =====================================================

    measurement_time = (
        timestamps[-1]
        - timestamps[0]
    )

    fps = (
        (len(timestamps) - 1)
        / measurement_time
    )

    print()
    print("환경 측정 종료")
    print(f"CSV 저장 완료: {csv_path}")
    print(f"데이터 개수: {len(timestamps)}")
    print(f"측정 시간: {measurement_time:.2f} sec")
    print(f"평균 처리 FPS: {fps:.2f}")


if __name__ == "__main__":
    main()