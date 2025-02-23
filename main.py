from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import time
import os
import shutil
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# Load YOLO models
object_model = YOLO("model1.pt")  # Object detection
road_model = YOLO("model2.pt")  # Road segmentation

# Video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

def process_video():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Object detection
        obj_results = object_model(frame)
        # Road segmentation
        road_results = road_model(frame)

        # Overlay for segmentation mask
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        # Draw object detection bounding boxes
        for r in obj_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{r.names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process road segmentation mask
        for r in road_results:
            if r.masks is not None:
                for mask in r.masks.data:
                    mask = mask.cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)

                    # Resize the mask to match the frame dimensions
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Ensure the mask is binary (0 or 255)
                    mask_resized = np.where(mask_resized > 128, 255, 0).astype(np.uint8)

                    # Apply the resized mask to the mask_overlay
                    mask_overlay[mask_resized > 128] = (0, 0, 255)  # Highlight road in red


        # Blend mask with frame
        frame = cv2.addWeighted(frame, 1, mask_overlay, 0.5, 0)

        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

        time.sleep(0.03)  # Control frame rate (~30 FPS)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the homepage with live video feed."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/live_feed")
def live_feed():
    """Stream live video feed with YOLO processing."""
    return StreamingResponse(process_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """Handle video upload and process it."""
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save uploaded file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processed_video_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")

    # Process video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Object detection
        obj_results = object_model(frame)
        # Road segmentation
        road_results = road_model(frame)

        # Mask overlay
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        # Draw objects
        for r in obj_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{r.names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process segmentation masks
        for r in road_results:
            if r.masks is not None:
                for mask in r.masks.data:
                    mask = mask.cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)

                    # ✅ Resize the mask to match the frame's dimensions
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                    # ✅ Apply the resized mask correctly
                    mask_overlay[mask > 128] = (0, 0, 255)  


        # Blend mask with frame
        frame = cv2.addWeighted(frame, 1, mask_overlay, 0.5, 0)

        out.write(frame)  # Save frame to processed video

    cap.release()
    out.release()

    return {"message": "Video processed successfully", "download_url": f"/download_video/{file.filename}"}

@app.get("/download_video/{filename}")
def download_video(filename: str):
    """Download processed video."""
    processed_video_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")

    if os.path.exists(processed_video_path):
        return FileResponse(processed_video_path, media_type="video/mp4", filename=f"processed_{filename}")
    return {"error": "File not found"}

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Serve the upload page."""
    return templates.TemplateResponse("upload.html", {"request": request})

def process_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Object detection
        obj_results = object_model(frame)
        road_results = road_model(frame)

        # Mask overlay
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        # Draw objects
        for r in obj_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{r.names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process segmentation masks
        for r in road_results:
            if r.masks is not None:
                for mask in r.masks.data:
                    mask = mask.cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_overlay[mask_resized > 128] = (0, 0, 255)  # Red overlay for road

        # Blend mask with frame
        frame = cv2.addWeighted(frame, 1, mask_overlay, 0.5, 0)

        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    cap.release()


@app.post("/upload_video_stream/")
async def upload_video_stream(file: UploadFile = File(...)):
    """Upload video and return stream URL."""
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save uploaded file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse({"stream_url": f"/stream_video/{file.filename}"})

@app.get("/stream_video/{filename}")
def stream_video(filename: str):
    """Stream video while processing it."""
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(video_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return StreamingResponse(process_video_stream(video_path), media_type="multipart/x-mixed-replace; boundary=frame")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
