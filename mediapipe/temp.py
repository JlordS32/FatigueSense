from region_cropper import RegionCropper
import cv2

cropper = RegionCropper("models/face_landmarker.task")

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    left_eye, right_eye, mouth = cropper.get_crops(frame)

    if left_eye is not None:
        cv2.imshow("Left Eye", left_eye)

    if right_eye is not None:
        cv2.imshow("Right Eye", right_eye)

    if mouth is not None:
        cv2.imshow("Mouth", mouth)

    if cv2.waitKey(1) == 27:
        break

cropper.close()
cap.release()
cv2.destroyAllWindows()