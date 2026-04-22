import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks.python import vision


class RegionCropper:
    def __init__(self, model_path):
        self.model_path = str(model_path)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)
        self.frame_idx = 0

        # landmark indices
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
        self.MOUTH = [61, 291, 81, 178, 13, 14, 402, 311, 308]

    def landmark_to_pixel(self, lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def get_center(self, landmarks, indices, w, h):
        pts = [self.landmark_to_pixel(landmarks[i], w, h) for i in indices]
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        return cx, cy

    def get_inter_eye_distance(self, landmarks, w, h):
        left = self.landmark_to_pixel(landmarks[33], w, h)
        right = self.landmark_to_pixel(landmarks[263], w, h)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        return max(1, int((dx**2 + dy**2) ** 0.5))

    def crop_box(self, frame, center, box_w, box_h, out_size):
        h, w = frame.shape[:2]
        cx, cy = center

        x1 = max(0, int(cx - box_w / 2))
        y1 = max(0, int(cy - box_h / 2))
        x2 = min(w, int(cx + box_w / 2))
        y2 = min(h, int(cy + box_h / 2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_CUBIC)
        return crop

    def get_crops(self, frame):
        """
        Input:
            frame (BGR image)

        Returns:
            left_eye_crop, right_eye_crop, mouth_crop
            (each can be None if detection fails)
        """

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = self.frame_idx * 33
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        self.frame_idx += 1

        if not result.face_landmarks:
            return None, None, None

        face = result.face_landmarks[0]

        # ---- centers ----
        left_eye_center = self.get_center(face, self.LEFT_EYE, w, h)
        right_eye_center = self.get_center(face, self.RIGHT_EYE, w, h)
        mouth_center = self.get_center(face, self.MOUTH, w, h)

        eye_dist = self.get_inter_eye_distance(face, w, h)

        # ---- stable eye boxes ----
        eye_box_w = int(0.5 * eye_dist)
        eye_box_h = int(0.35 * eye_dist)

        # ---- dynamic mouth box (no MAR, purely geometric) ----
        mouth_pts = [self.landmark_to_pixel(face[i], w, h) for i in self.MOUTH]
        xs = [p[0] for p in mouth_pts]
        ys = [p[1] for p in mouth_pts]

        mouth_w_landmark = max(xs) - min(xs)
        mouth_h_landmark = max(ys) - min(ys)

        # hybrid sizing (stable + dynamic)
        mouth_box_w = max(int(1.0 * eye_dist), int(1.6 * mouth_w_landmark))
        mouth_box_h = max(int(0.7 * eye_dist), int(2.5 * mouth_h_landmark))

        # shift downward
        cx, cy = mouth_center
        cy = int(cy + 0.2 * mouth_box_h)
        mouth_center = (cx, cy)

        # ---- crop ----
        left_eye = self.crop_box(frame, left_eye_center, eye_box_w, eye_box_h, (96, 96))
        right_eye = self.crop_box(frame, right_eye_center, eye_box_w, eye_box_h, (96, 96))
        mouth = self.crop_box(frame, mouth_center, mouth_box_w, mouth_box_h, (128, 128))

        return left_eye, right_eye, mouth

    def close(self):
        self.landmarker.close()