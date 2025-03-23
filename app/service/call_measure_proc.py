from app.procedures.get_cm_per_pixel import get_cm_per_pixel
from app.procedures.get_garment_measurements import get_garment_measurements


def call_measure_proc(image_path: str) -> dict:
    """
    Description:
        - This function call the garment measurement procedurece to get teh garment measurements from the image
        - First it will get the cm per pixel ratio using image with reference object
        - Then it will call the measure_garment function from measure_garment procedures
    parameter:
        - image_path: str
    return:
        - dict: garment measurements with JSON string
    """

    # get the cm per pixel ratio using image with reference object
    cm_per_pixel = get_cm_per_pixel(image_path)
    print(f"cm_per_pixel : {cm_per_pixel}")

    # Verifying the reference object detection
    if cm_per_pixel < 0:
        return {"error": "reference object not detected"}

    # calling measure_garment function from measure_garment procedures
    res = get_garment_measurements(image_path, cm_per_pixel)

    return res
