from ultralytics import YOLO
from PIL import Image
# Load YOLOv11 model (you can also use yolov11n.pt, yolov11s.pt, etc.)
model = YOLO("yolo11n.pt")

def detect_objects(image: Image.Image):
    """Run YOLOv11 inference and return JSON with all detections."""
    results = model(image)[0]

    return results.to_json()

if __name__ == "__main__":
    # Example usage
    image_path = "other.jpg"
    image = Image.open(image_path).convert("RGB")
    detections = detect_objects(image)
    print(detections)