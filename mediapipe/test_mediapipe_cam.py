import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks.python import vision


# =========================
# Config
# =========================
MODEL_PATH = Path("mediapipe/models/face_landmarker.task")

LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
MOUTH = [61, 291, 81, 178, 13, 14, 402, 311, 308]

EAR_OPEN_THRESH = 0.21
EAR_CLOSE_THRESH = 0.19

MAR_OPEN_THRESH = 0.43
MAR_CLOSE_THRESH = 0.40


# =========================
# MediaPipe setup
# =========================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode


def landmark_to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def get_region_center(landmarks, indices, w, h):
    pts = [landmark_to_pixel(landmarks[i], w, h) for i in indices]
    cx = int(sum(p[0] for p in pts) / len(pts))
    cy = int(sum(p[1] for p in pts) / len(pts))
    return cx, cy


def get_inter_eye_distance(landmarks, w, h):
    left_outer = landmark_to_pixel(landmarks[33], w, h)
    right_outer = landmark_to_pixel(landmarks[263], w, h)
    dx = right_outer[0] - left_outer[0]
    dy = right_outer[1] - left_outer[1]
    return max(1, int((dx ** 2 + dy ** 2) ** 0.5))


def crop_fixed_box(frame, center, box_w, box_h, out_size):
    h, w = frame.shape[:2]
    cx, cy = center

    x1 = max(0, int(cx - box_w / 2))
    y1 = max(0, int(cy - box_h / 2))
    x2 = min(w, int(cx + box_w / 2))
    y2 = min(h, int(cy + box_h / 2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_CUBIC)
    return crop, (x1, y1, x2, y2)


def draw_box(frame, box, label, color):
    if box is None:
        return

    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_EAR(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))

    A = euclidean(pts[2], pts[6])
    B = euclidean(pts[3], pts[5])
    C = euclidean(pts[0], pts[1])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def compute_MAR(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))

    A = euclidean(pts[2], pts[6])
    B = euclidean(pts[3], pts[5])
    C = euclidean(pts[0], pts[1])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def main():
    print("MODEL_PATH:", MODEL_PATH)
    print("Exists:", MODEL_PATH.exists())

    if not MODEL_PATH.exists():
        print("ERROR: Model file not found.")
        print("Expected at:", MODEL_PATH)
        return

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Try higher resolution for less blurry crops
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Left Eye", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Eye", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mouth", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Main", 1000, 700)
    cv2.resizeWindow("Left Eye", 220, 220)
    cv2.resizeWindow("Right Eye", 220, 220)
    cv2.resizeWindow("Mouth", 260, 260)

    eye_state = "OPEN"
    mouth_state = "CLOSED"

    frame_idx = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = frame_idx * 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                face = result.face_landmarks[0]
                h, w = frame.shape[:2]

                # ---- Metrics ----
                ear_left = compute_EAR(face, LEFT_EYE, w, h)
                ear_right = compute_EAR(face, RIGHT_EYE, w, h)
                ear = (ear_left + ear_right) / 2.0

                mar = compute_MAR(face, MOUTH, w, h)

                # ---- State logic ----
                if ear < EAR_CLOSE_THRESH:
                    eye_state = "CLOSED"
                elif ear > EAR_OPEN_THRESH:
                    eye_state = "OPEN"

                if mar > MAR_OPEN_THRESH:
                    mouth_state = "OPEN"
                elif mar < MAR_CLOSE_THRESH:
                    mouth_state = "CLOSED"

                # ---- Draw landmarks on display frame only ----
                for lm in face:
                    x, y = landmark_to_pixel(lm, w, h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # ---- Region centers ----
                left_eye_center = get_region_center(face, LEFT_EYE, w, h)
                right_eye_center = get_region_center(face, RIGHT_EYE, w, h)
                mouth_center = get_region_center(face, MOUTH, w, h)

                eye_dist = get_inter_eye_distance(face, w, h)

                # ---- Stable eye box sizes ----
                eye_box_w = int(0.50 * eye_dist)
                eye_box_h = int(0.35 * eye_dist)

                # ---- Improved mouth box sizing ----
                mar_clamped = max(0.3, min(mar, 1.2))
                mouth_scale = 1.0 + 1.5 * (mar_clamped - 0.3)

                mouth_box_w = int(1.0 * eye_dist)
                mouth_box_h = int(0.65 * eye_dist * mouth_scale)

                # Prevent mouth box from becoming too small
                mouth_box_h = max(mouth_box_h, int(0.65 * eye_dist))

                # Shift downward because mouth opens mostly downward
                cx, cy = mouth_center
                cy = int(cy + 0.05 * mouth_box_h)
                mouth_center = (cx, cy)

                # ---- Crop from clean frame ----
                left_eye_crop, left_eye_box = crop_fixed_box(
                    clean_frame, left_eye_center, eye_box_w, eye_box_h, (32, 64)
                )
                right_eye_crop, right_eye_box = crop_fixed_box(
                    clean_frame, right_eye_center, eye_box_w, eye_box_h, (32, 64)
                )
                mouth_crop, mouth_box = crop_fixed_box(
                    clean_frame, mouth_center, mouth_box_w, mouth_box_h, (32, 64)
                )

                # ---- Draw boxes ----
                draw_box(frame, left_eye_box, "Left Eye", (255, 0, 0))
                draw_box(frame, right_eye_box, "Right Eye", (0, 0, 255))
                draw_box(frame, mouth_box, "Mouth", (0, 255, 255))

                # ---- Show crops ----
                if left_eye_crop is not None:
                    cv2.imshow("Left Eye", left_eye_crop)
                if right_eye_crop is not None:
                    cv2.imshow("Right Eye", right_eye_crop)
                if mouth_crop is not None:
                    cv2.imshow("Mouth", mouth_crop)

                # ---- Status text ----
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}  Eye: {eye_state}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f}  Mouth: {mouth_state}",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 200, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Mouth box h: {mouth_box_h}",
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Main", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()