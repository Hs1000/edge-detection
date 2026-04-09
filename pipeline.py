from ultralytics import YOLO
import cv2
import os
import json
import argparse

# Load trained model (update path after training)
model = YOLO("model_weights/best.pt")


def detect_defects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 400:
            bboxes.append([x, y, x+w, y+h])

    return bboxes


def process_image(img_path, output_dir):
    image = cv2.imread(img_path)
    name = os.path.splitext(os.path.basename(img_path))[0]

    results = model(image)[0]

    annotated = image.copy()
    detections = {}

    idx = 1

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())

        belt_crop = image[y1:y2, x1:x2]

        defect_boxes = detect_defects(belt_crop)

        for dx1, dy1, dx2, dy2 in defect_boxes:
            gx1, gy1 = x1 + dx1, y1 + dy1
            gx2, gy2 = x1 + dx2, y1 + dy2

            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)

            detections[str(idx)] = {
                "bbox_coordinates": [gx1, gy1, gx2, gy2]
            }
            idx += 1

    cv2.imwrite(os.path.join(output_dir, f"{name}.jpg"), annotated)

    with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
        json.dump(detections, f, indent=4)


def main(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(image_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            process_image(os.path.join(image_dir, file), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    main(args.image_dir, args.output_dir)
