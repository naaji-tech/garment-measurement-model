from pydantic import BaseModel

# Reference object width and height in cm
REF_OBJ_TYPE = ["creditCard"]
REF_OBJ_WIDHT_CM = 8.6
REF_OBJ_HEIGHT_CM = 5.4

# AI Models paths
CARD_DETECTOR_MODEL = "models/CardDetectorV1.pt"
GARMENT_DETECTOR_MODEL = "models/yolo8m-350DS-excellent.pt"


# keypoint configuration
class GetKeypoint(BaseModel):
    LEFT_SHOULDER:          int = 0
    RIGHT_SHOULDER:         int = 1
    SLEEVE_END_TOP:         int = 2
    SLEEVE_END_BOTTOM:      int = 3
    RIGHT_ARM_PIT:          int = 4
    LEFT_ARM_PIT:           int = 5
    LEFT_WAIST:             int = 6
    RIGHT_WAIST:            int = 7
    BOTTOM_RIGHT:           int = 8
    BOTTOM_LEFT:            int = 9
    COLLER_TOP:             int = 10
    BOTTOM_POINT:           int = 11