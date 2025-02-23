import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from a file path."""
    return cv2.imread(image_path)

def draw_predictions(image, results):
    """Draws bounding boxes and labels on an image."""
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = f"{result.names[class_id]}: {confidence:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image
