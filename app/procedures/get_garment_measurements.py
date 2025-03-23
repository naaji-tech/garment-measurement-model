import cv2
from ultralytics import YOLO

from app.procedures.compute_measurements import compute_all_measurements
from app.util.config import GARMENT_DETECTOR_MODEL

garment_model = YOLO(GARMENT_DETECTOR_MODEL)


def get_garment_measurements(image_path: str, cm_per_pixel: float) -> dict:
    """
    description:
        - Function to measure garment measurements from an image
    parameters:
        - image_path: str
        - cm_per_pixel: float
    return:
        - dict with all measurement or error with error message
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (896, 896))

    results = garment_model.predict(image, size=896)

    # Verifying one border box for garment and verifying the keypoints detection
    garment_kpts = None
    for r in results:
        if (
            r.boxes is None
            or len(r.boxes) != 1
            or r.boxes.conf < 0.85
            or r.keypoints is None
        ):
            continue

        kpts = r.keypoints.cpu().numpy().xy[0]
        garment_kpts = kpts

    if garment_kpts is None:
        return {"error": "Garment keypoint not found"}

    measurements = compute_all_measurements(garment_kpts, cm_per_pixel)

    return measurements
