import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 기본 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "face_landmarker.task"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# =========================================================
# rPPG 측정에 사용할 얼굴 Landmark
# =========================================================

FOREHEAD_IDX = [
    10, 109, 67, 103, 54, 21,
    338, 297, 332, 284, 251
]

LEFT_CHEEK_IDX = [
    50, 101, 205, 206, 203, 123, 117
]

RIGHT_CHEEK_IDX = [
    280, 330, 425, 426, 423, 352, 346
]


# =========================================================
# Landmark 번호를 실제 픽셀 좌표로 변환
# =========================================================

def get_polygon_points(
    face_landmarks,
    indices,
    width,
    height
):
    points = []

    for idx in indices:
        landmark = face_landmarks[idx]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append([x, y])

    return np.array(
        points,
        dtype=np.int32
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 측정 데이터 저장용 리스트
    # -----------------------------------------------------

    timestamps = []

    r_values = []
    g_values = []
    b_values = []


    # -----------------------------------------------------
    # CSV 파일 경로 생성
    # -----------------------------------------------------

    file_time = time.strftime("%Y%m%d_%H%M%S")

    csv_path = DATA_DIR / f"rgb_signal_{file_time}.csv"


    # -----------------------------------------------------
    # 카메라 연결
    # -----------------------------------------------------

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return


    # -----------------------------------------------------
    # MediaPipe Face Landmarker 설정
    # -----------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )


    # -----------------------------------------------------
    # Face Landmarker 실행
    # -----------------------------------------------------

    with vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        start_time = time.perf_counter()

        while True:

            # -------------------------------------------------
            # 카메라 Frame 읽기
            # -------------------------------------------------

            ret, frame = cap.read()

            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break


            # -------------------------------------------------
            # 실제 시간 계산
            # -------------------------------------------------

            current_time = (
                time.perf_counter()
                - start_time
            )

            frame_timestamp_ms = int(
                current_time * 1000
            )


            # -------------------------------------------------
            # OpenCV BGR -> RGB
            # -------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # -------------------------------------------------
            # numpy image -> MediaPipe Image
            # -------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # -------------------------------------------------
            # Face Landmark 추론
            # -------------------------------------------------

            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms
            )


            # -------------------------------------------------
            # 얼굴이 검출된 경우
            # -------------------------------------------------

            if result.face_landmarks:

                height, width, _ = frame.shape

                face_landmarks = (
                    result.face_landmarks[0]
                )


                # =============================================
                # 1. ROI Landmark -> 실제 픽셀 좌표
                # =============================================

                forehead_points = get_polygon_points(
                    face_landmarks,
                    FOREHEAD_IDX,
                    width,
                    height
                )

                left_cheek_points = get_polygon_points(
                    face_landmarks,
                    LEFT_CHEEK_IDX,
                    width,
                    height
                )

                right_cheek_points = get_polygon_points(
                    face_landmarks,
                    RIGHT_CHEEK_IDX,
                    width,
                    height
                )


                # =============================================
                # 2. Polygon 표시
                # =============================================

                cv2.polylines(
                    frame,
                    [forehead_points],
                    True,
                    (255, 0, 0),
                    2
                )

                cv2.polylines(
                    frame,
                    [left_cheek_points],
                    True,
                    (0, 255, 0),
                    2
                )

                cv2.polylines(
                    frame,
                    [right_cheek_points],
                    True,
                    (0, 255, 0),
                    2
                )


                # =============================================
                # 3. 빈 Mask 생성
                # =============================================

                mask = np.zeros(
                    frame.shape[:2],
                    dtype=np.uint8
                )


                # =============================================
                # 4. ROI 영역을 Mask에 채움
                # =============================================

                cv2.fillPoly(
                    mask,
                    [forehead_points],
                    255
                )

                cv2.fillPoly(
                    mask,
                    [left_cheek_points],
                    255
                )

                cv2.fillPoly(
                    mask,
                    [right_cheek_points],
                    255
                )


                # =============================================
                # 5. ROI RGB 평균값 계산
                # =============================================

                mean_b, mean_g, mean_r, _ = cv2.mean(
                    frame,
                    mask=mask
                )


                # =============================================
                # 6. 측정 데이터 저장
                # =============================================

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


                # =============================================
                # 7. Console 출력
                # =============================================

                print(
                    f"Time: {current_time:.3f}, "
                    f"R: {mean_r:.2f}, "
                    f"G: {mean_g:.2f}, "
                    f"B: {mean_b:.2f}"
                )


                # =============================================
                # 8. Mask 확인
                # =============================================

                cv2.imshow(
                    "ROI Mask",
                    mask
                )


            # -------------------------------------------------
            # 원본 Frame 표시
            # -------------------------------------------------

            cv2.imshow(
                "rPPG Capture",
                frame
            )


            # -------------------------------------------------
            # q 키를 누르면 종료
            # -------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    # =========================================================
    # 카메라 종료
    # =========================================================

    cap.release()
    cv2.destroyAllWindows()


    # =========================================================
    # CSV 저장
    # =========================================================

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

        # CSV Header
        writer.writerow([
            "timestamp",
            "r",
            "g",
            "b"
        ])


        # CSV Data
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


    # =========================================================
    # 결과 출력
    # =========================================================

    print()
    print("측정 종료")
    print(f"CSV 저장 완료: {csv_path}")
    print(f"측정 데이터 수: {len(timestamps)}")

    measurement_time = (
        timestamps[-1]
        - timestamps[0]
    )

    print(
        f"측정 시간: "
        f"{measurement_time:.2f} sec"
    )

    if measurement_time > 0:

        estimated_fps = (
            len(timestamps)
            / measurement_time
        )

        print(
            f"평균 처리 FPS: "
            f"{estimated_fps:.2f}"
        )


# =========================================================
# 프로그램 시작
# =========================================================

if __name__ == "__main__":
    main()