import cv2
import mediapipe as mp
import time
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "face_landmarker.task"

# rPPG 측정에 사용할 얼굴 ROI Landmark
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

timestamps = []
r_values = []
g_values = []
b_values = []

def get_polygon_points(face_landmarks, indices, width, height):
    points = []

    for idx in indices:
        landmark = face_landmarks[idx]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append([x, y])

    return np.array(points, dtype=np.int32)


def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:

        start_time = time.perf_counter()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            current_time = time.perf_counter() - start_time
            frame_timestamp_ms = int(current_time * 1000)
            
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms
            )


            if result.face_landmarks:
                height, width, _ = frame.shape

                face_landmarks = result.face_landmarks[0]

                # -------------------------
                # 1. 랜드마크 -> 픽셀 좌표 변환
                # -------------------------

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


                # -------------------------
                # 2. Polygon 표시
                # -------------------------

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

                # -------------------------
                # 마스크 생성
                # -------------------------
                mask = np.zeros(
                    frame.shape[:2],
                    dtype=np.uint8
                )

                # -------------------------
                # ROI 부분을 흰색으로 채움
                # -------------------------
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

                # -------------------------
                # ROI RGB 계산
                # -------------------------
                mean_b, mean_g, mean_r, _ = cv2.mean(
                    frame,
                    mask=mask
                )

                timestamps.append(current_time)

                r_values.append(mean_r)
                g_values.append(mean_g)
                b_values.append(mean_b)

                print(
                    f"Time: {current_time:.3f}, "
                    f"R: {mean_r:.2f}, "
                    f"G: {mean_g:.2f}, "
                    f"B: {mean_b:.2f}"
                )

                cv2.imshow("ROI Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()