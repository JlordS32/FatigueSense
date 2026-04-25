from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


# =========================
# Config
# =========================
MODEL_PATH = Path("mediapipe/models/face_landmarker.task")

LEFT_EYE  = [33,  133, 160, 144, 158, 153, 159, 145]
RIGHT_EYE = [362, 263, 387, 373, 385, 380, 386, 374]
MOUTH     = [61,  291,  81, 178,  13,  14, 402, 311]

EAR_OPEN_THRESH  = 0.26
EAR_CLOSE_THRESH = 0.20

MAR_OPEN_THRESH  = 0.43
MAR_CLOSE_THRESH = 0.40

MIN_HORIZONTAL_PX = 5
MIN_EYE_DIST_PX   = 20


# =========================
# MediaPipe setup
# =========================
BaseOptions          = mp.tasks.BaseOptions
FaceLandmarker       = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode    = vision.RunningMode


# =========================
# Helpers
# =========================
def landmark_to_pixel(lm, w, h) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def get_region_center(landmarks, indices, w, h) -> tuple[int, int]:
    pts = [landmark_to_pixel(landmarks[i], w, h) for i in indices]
    cx = int(sum(p[0] for p in pts) / len(pts))
    cy = int(sum(p[1] for p in pts) / len(pts))
    return cx, cy


def get_inter_eye_distance(landmarks, w, h) -> int:
    left_outer  = landmark_to_pixel(landmarks[33],  w, h)
    right_outer = landmark_to_pixel(landmarks[263], w, h)
    dx = right_outer[0] - left_outer[0]
    dy = right_outer[1] - left_outer[1]
    return max(1, int((dx ** 2 + dy ** 2) ** 0.5))


def compute_ear(landmarks, indices, w, h) -> float:
    """
    indices order: [outer, inner, top1, bot1, top2, bot2, top3, bot3]
    Matches LEFT_EYE  = [33, 133, 160, 144, 158, 153, 159, 145]
            RIGHT_EYE = [362, 263, 387, 373, 385, 380, 386, 374]
    """
    pts = [landmark_to_pixel(landmarks[i], w, h) for i in indices]

    horizontal = euclidean(pts[0], pts[1])
    if horizontal < MIN_HORIZONTAL_PX:
        return 0.0

    v1 = euclidean(pts[2], pts[3])
    v2 = euclidean(pts[4], pts[5])
    v3 = euclidean(pts[6], pts[7])

    return (v1 + v2 + v3) / (3.0 * horizontal)


def compute_mar(landmarks, indices, w, h) -> float:
    """
    indices order: [left_corner, right_corner, top1, bot1, top2, bot2, top3, bot3]
    Matches MOUTH = [61, 291, 81, 178, 13, 14, 402, 311]
    """
    pts = [landmark_to_pixel(landmarks[i], w, h) for i in indices]

    horizontal = euclidean(pts[0], pts[1])
    if horizontal < MIN_HORIZONTAL_PX:
        return 0.0

    v1 = euclidean(pts[2], pts[3])
    v2 = euclidean(pts[4], pts[5])
    v3 = euclidean(pts[6], pts[7])

    return (v1 + v2 + v3) / (3.0 * horizontal)


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
        frame, label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )


# =========================
# Main
# =========================
def main():
    print("MODEL_PATH:", MODEL_PATH)
    print("Exists:", MODEL_PATH.exists())

    if not MODEL_PATH.exists():
        print("ERROR: Model file not found.")
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Main",       cv2.WINDOW_NORMAL)
    cv2.namedWindow("Left Eye",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Eye",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mouth",      cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Main",      1000, 700)
    cv2.resizeWindow("Left Eye",   220, 220)
    cv2.resizeWindow("Right Eye",  220, 220)
    cv2.resizeWindow("Mouth",      260, 260)

    left_eye_state = "OPEN"
    right_eye_state = "OPEN"
    mouth_state    = "CLOSED"
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

                eye_dist = get_inter_eye_distance(face, w, h)

                # Reject bad detections (e.g. dark/occluded frames)
                if eye_dist >= MIN_EYE_DIST_PX:

                    # ---- Metrics ----
                    ear_left  = compute_ear(face, LEFT_EYE,  w, h)
                    ear_right = compute_ear(face, RIGHT_EYE, w, h)
                    ear       = (ear_left + ear_right) / 2.0
                    mar       = compute_mar(face, MOUTH, w, h)

                    # ---- State logic (per-eye, with ambiguous zone) ----
                    if ear_left <= EAR_CLOSE_THRESH:
                        left_eye_state = "CLOSED"
                    elif ear_left >= EAR_OPEN_THRESH:
                        left_eye_state = "OPEN"
                    # else: retain previous state

                    if ear_right <= EAR_CLOSE_THRESH:
                        right_eye_state = "CLOSED"
                    elif ear_right >= EAR_OPEN_THRESH:
                        right_eye_state = "OPEN"

                    if mar > MAR_OPEN_THRESH:
                        mouth_state = "OPEN"
                    elif mar < MAR_CLOSE_THRESH:
                        mouth_state = "CLOSED"

                    # ---- Crop sizes ----
                    eye_box_w = int(0.50 * eye_dist)
                    eye_box_h = int(0.35 * eye_dist)

                    mar_clamped  = max(0.3, min(mar, 1.5))
                    mouth_scale  = 1.0 + 4.0 * (mar_clamped - 0.3)
                    mouth_box_w  = int(1.0 * eye_dist)
                    mouth_box_h  = max(int(0.4 * eye_dist * mouth_scale), int(0.35 * eye_dist))

                    # ---- Region centers ----
                    left_eye_center  = get_region_center(face, LEFT_EYE,  w, h)
                    right_eye_center = get_region_center(face, RIGHT_EYE, w, h)
                    mouth_center     = get_region_center(face, MOUTH,     w, h)

                    cx, cy = mouth_center
                    mouth_center = (cx, int(cy + 0.05 * mouth_box_h))

                    # ---- Crops ----
                    left_eye_crop,  left_eye_box  = crop_fixed_box(clean_frame, left_eye_center,  eye_box_w,   eye_box_h,   (64, 64))
                    right_eye_crop, right_eye_box = crop_fixed_box(clean_frame, right_eye_center, eye_box_w,   eye_box_h,   (64, 64))
                    mouth_crop,     mouth_box      = crop_fixed_box(clean_frame, mouth_center,     mouth_box_w, mouth_box_h, (64, 64))

                    # ---- Draw landmarks ----
                    for lm in face:
                        x, y = landmark_to_pixel(lm, w, h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    # ---- Draw boxes ----
                    draw_box(frame, left_eye_box,  f"L: {left_eye_state}",  (255, 0,   0))
                    draw_box(frame, right_eye_box, f"R: {right_eye_state}", (0,   0, 255))
                    draw_box(frame, mouth_box,     f"Mouth: {mouth_state}", (0, 255, 255))

                    # ---- Show crops ----
                    if left_eye_crop  is not None: cv2.imshow("Left Eye",  left_eye_crop)
                    if right_eye_crop is not None: cv2.imshow("Right Eye", right_eye_crop)
                    if mouth_crop     is not None: cv2.imshow("Mouth",     mouth_crop)

                    # ---- HUD ----
                    cv2.putText(frame, f"EAR L: {ear_left:.3f}  R: {ear_right:.3f}  Avg: {ear:.3f}", (30, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255,   0), 2)
                    cv2.putText(frame, f"Eye  L: {left_eye_state}  R: {right_eye_state}",             (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255,   0), 2)
                    cv2.putText(frame, f"MAR: {mar:.3f}  Mouth: {mouth_state}",                       (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200,   0), 2)
                    cv2.putText(frame, f"Eye dist: {eye_dist}px",                                     (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            cv2.imshow("Main", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()