import os
import cv2
import torch
from ultralytics import YOLO
from utils import draw_predictions

# Model and input paths
MODEL_PATH = r"runs\detect\train19\weights\best.pt" # Path to trained YOLOv8 model
INPUT_PATH = r"input"  # Folder containing test images or videos
OUTPUT_PATH = r"runs\detection"  # Output directory

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Video processing generator
def video_stream_generator(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        yield frame, out  # Yield frame and writer

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process images & videos
for file in os.listdir(INPUT_PATH):
    input_file = os.path.join(INPUT_PATH, file)
    output_file = os.path.join(OUTPUT_PATH, file)

    # Check file type (image or video)
    if file.endswith((".jpg", ".png", ".jpeg")):
        # Process image
        img = cv2.imread(input_file)
        results = model(img, device=DEVICE)
        img_with_boxes = draw_predictions(img, results)

        # Save and show image
        cv2.imwrite(output_file, img_with_boxes)
        cv2.imshow("Image Detection", img_with_boxes)
        cv2.waitKey(0)  # Wait for key press

    elif file.endswith((".mp4", ".avi", ".mov")):
        # Process video using generator
        for frame, out in video_stream_generator(input_file, output_file):
            results = model(frame, device=DEVICE)
            frame_with_boxes = draw_predictions(frame, results)

            # Show real-time output
            cv2.imshow("YOLOv8 Real-Time Detection", frame_with_boxes)
            out.write(frame_with_boxes)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
print("Detection complete! 🚀")
