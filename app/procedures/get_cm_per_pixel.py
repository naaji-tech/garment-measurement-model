import cv2
from numpy import imag
from ultralytics import YOLO

from app.util.config import (
    REF_OBJ_DETECTOR_MODEL,
    REF_OBJ_HEIGHT_CM,
    REF_OBJ_TYPE,
    REF_OBJ_WIDHT_CM,
)

ref_model = YOLO(REF_OBJ_DETECTOR_MODEL)


def get_cm_per_pixel(image_path: str) -> float:
    """
    description:
        - This function detect the reference object in the image and calculate the cm per pixel unit.
        - Reference object should be in the upper left part of the image.
    parameter:
        - image_path: str
    return:
        - cm_per_pixel: float
    """
    # Variable declareration
    ref_width = None
    ref_height = None

    # Load image and crop the upper left part of the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (896, 896))
    height, width = image.shape[:2]
    ul_image = image[0 : (height // 2), 0 : (width // 2)]

    # Detect reference object width and height
    results = ref_model.predict(ul_image, size=896)

    for result in results:
        for box in result.boxes:
            if box is None or box.conf < 0.85 or result.names[0] not in REF_OBJ_TYPE:
                continue
            x1, y1, x2, y2 = box.cpu().numpy().xyxy[0]
            ref_width = abs(x1 - x2)
            ref_height = abs(y1 - y2)
            break
        # result.show()

    # Verifying the reference object detection
    if ref_width is None or ref_height is None:
        return -1

    ref_width_pixel = ref_width
    ref_height_pixel = ref_height

    cm_per_pixel_width = REF_OBJ_WIDHT_CM / ref_width_pixel
    cm_per_pixel_height = REF_OBJ_HEIGHT_CM / ref_height_pixel

    return (cm_per_pixel_width + cm_per_pixel_height) / 2
