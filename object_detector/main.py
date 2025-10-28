from ultralytics import YOLO
from PIL import Image
import argparse

# Load YOLOv11 model (you can also use yolov11n.pt, yolov11s.pt, etc.)
model = YOLO("yolo11n.pt")

def detect_objects(image: Image.Image):
    """Run YOLOv11 inference and return JSON with all detections."""
    results = model(image)[0]

    return results.to_json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Yolov11 Object Detector',
                    description='Detect objects in images using YOLOv11')
    parser.add_argument('-i', '--image_path', type=str, help='Path to the input image')
    args = parser.parse_args()
    image_path = args.image_path
    image = Image.open(image_path).convert("RGB")
    detections = detect_objects(image)
    print(detections)