import cv2
from ultralytics import YOLO

card_detector_model = YOLO("models/ref-object-model-v2.pt")

image = cv2.imread("test/test_images/oversized-mint-green-tee-M.jpg")
image = cv2.resize(image, (896, 896))
print(f"image shape: {image.shape}")
height, width = image.shape[:2]
ul_image = image[0 : (height // 2), 0 : (width // 2)]

results = card_detector_model.predict(ul_image, imgsz=896)

print()
print("------------------------------")
print(f"results type: ${type(results)}")
print(f"results len: ${len(results)}")
print(f"Results: ${results}")
print("------------------------------")

for r in results:
    print(f"boxes type: ${type(r.boxes)}")
    print(f"boxes len: ${len(r.boxes)}")
    print(f"boxes: ${r.boxes}")
    print("------------------------------")
    print()

    for b in r.boxes:
        if b is None or b.conf < 0.85 or r.names[0] not in ["creditCard"]:
            continue

        print("------------")
        print(r.boxes[0].cpu().numpy().xyxy)

        x1, y1, x2, y2 = r.boxes[0].cpu().numpy().xyxy[0]
        ref_width = abs(x1 - x2)
        ref_height = abs(y1 - y2)

        print(f"width : {ref_width}")
        print(f"heigth : {ref_height}")

        cm_per_pixel_width = 8.6 / ref_width
        cm_per_pixel_height = 5.4 / ref_height

        print(f"cm per pixel width: {cm_per_pixel_width}")
        print(f"cm per pixel height: {cm_per_pixel_height}")

        print(f"cm/pixel: {(cm_per_pixel_width + cm_per_pixel_height) / 2}")

    r.show()
