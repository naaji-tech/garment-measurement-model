import math
from torch import Tensor
from app.util.config import GetKeypoint


def compute_all_measurements(keypoints: Tensor, cm_per_pixel: float) -> dict:
    """
    description:
        - This function compute all the measurements from the garment keypoints
    parameters:
        - parameter_list: list
    return:
        - dict with all measurements
    """

    # Helper function: Euclidean distance in pixel space
    def dist_px(kpt1, kpt2):
        return math.sqrt((kpt1[0] - kpt2[0]) ** 2 + (kpt1[1] - kpt2[1]) ** 2)

    # Helper function: convert px to cm units
    def px2cm(px):
        return px * cm_per_pixel

    get_keypoint = GetKeypoint()

    # 1) Shoulder Width
    shoulder_width_px = dist_px(
        keypoints[get_keypoint.LEFT_SHOULDER],
        keypoints[get_keypoint.RIGHT_SHOULDER],
    )
    shoulder_width_cm = px2cm(shoulder_width_px)

    # 2) Chest Width (armpit-to-armpit)
    chest_width_px = dist_px(
        keypoints[get_keypoint.LEFT_ARM_PIT],
        keypoints[get_keypoint.RIGHT_ARM_PIT],
    )
    chest_width_cm = px2cm(chest_width_px)

    # 3) Waist Width
    waist_width_px = dist_px(
        keypoints[get_keypoint.LEFT_WAIST], keypoints[get_keypoint.RIGHT_WAIST]
    )
    waist_width_cm = px2cm(waist_width_px)

    # 4) Bottom Width
    bottom_width_px = dist_px(
        keypoints[get_keypoint.BOTTOM_LEFT], keypoints[get_keypoint.BOTTOM_RIGHT]
    )
    bottom_width_cm = px2cm(bottom_width_px)

    # 5) Sleeve Length
    sleeve_length_px = dist_px(
        keypoints[get_keypoint.RIGHT_SHOULDER],
        keypoints[get_keypoint.SLEEVE_END_TOP],
    )
    sleeve_length_cm = px2cm(sleeve_length_px)

    # 6) Sleeve width
    sleeve_px = dist_px(
        keypoints[get_keypoint.SLEEVE_END_TOP],
        keypoints[get_keypoint.SLEEVE_END_BOTTOM],
    )
    sleeve_cm = px2cm(sleeve_px)

    # 6) Front Length (collar -> bottom)
    front_length_px = dist_px(
        keypoints[get_keypoint.COLLER_TOP], keypoints[get_keypoint.BOTTOM_POINT]
    )
    front_length_cm = px2cm(front_length_px)

    # Finalizing the all measurements
    measurements = {
        "sleeveLength": round(sleeve_length_cm, 2),
        "shoulderWidth": round(shoulder_width_cm, 2),
        "chest": round(chest_width_cm * 2, 2),
        "waist": round(waist_width_cm * 2, 2),
        "bottomCircumference": round(bottom_width_cm * 2, 2),
        "frontLength": round(front_length_cm, 2),
        "sleeve": round(sleeve_cm * 2, 2),
    }

    return measurements
