import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "face_landmarker.task"



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
        frame_timestamp_ms = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms
            )

            frame_timestamp_ms += 33

            if result.face_landmarks:
                height, width, _ = frame.shape

                face_landmarks = result.face_landmarks[0]

                for landmark in face_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            cv2.imshow("rPPG Face Landmarks", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()