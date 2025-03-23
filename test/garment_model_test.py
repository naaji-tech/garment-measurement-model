from ultralytics import YOLO
import cv2

garment_detector_model = YOLO("models/yolo8m-350DS-excellent.pt")

image = cv2.imread("./test/test_images/product-test-img-2.jpg")
image = cv2.resize(image, (768, 768))

results = garment_detector_model.predict(image, conf=0.2)

# print("Results: ", results)

for r in results:
    print(f"boxes: ${r.boxes}")
    if (
        r.boxes is None
        or len(r.boxes) != 1
        or r.boxes[0].conf < 0.85
        or r.keypoints is None
    ):
        print("garment not found.")
        continue
    
    print("------------")
    kpts = r.keypoints.cpu().numpy().xy[0]
    print(f"keypoints: ${kpts}")
    print("------------")
    print("keypoint: ", kpts[0])
    r.show()
