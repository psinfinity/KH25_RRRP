from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import cv2
import torch
import shutil
import uvicorn
from ultralytics import YOLO
import time
from utils import draw_predictions

# Initialize FastAPI app
app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

# Jinja2 template engine
templates = Jinja2Templates(directory="templates")

# Paths
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = "model.pt"  # Change this if needed

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model
model = YOLO(MODEL_PATH)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Perform YOLO object detection
        results = model(frame, device=DEVICE)
        # Draw bounding boxes
        frame_with_boxes = draw_predictions(frame, results)
        # Encode frame as JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame_with_boxes)
        frame_bytes = encoded_frame.tobytes()
        # Yield frame data as MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Upload page
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Handle file upload & process video
streaming_videos = {}

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    processed_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Open video for processing
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

    # Store frames for streaming
    def generate():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, device=DEVICE)
            frame_with_boxes = draw_predictions(frame, results)
            out.write(frame_with_boxes)

            # Convert frame to JPEG
            _, encoded_frame = cv2.imencode(".jpg", frame_with_boxes)
            frame_bytes = encoded_frame.tobytes()
            
            # Yield frame-by-frame as an MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
        out.release()

    # Store the generator in memory for streaming
    streaming_videos[file.filename] = generate()

    return JSONResponse({
        "message": "File processing started",
        "download_url": f"/download/{file.filename}"
    })


@app.get("/stream/{filename}")
async def stream_video(filename: str):
    if filename in streaming_videos:
        return StreamingResponse(streaming_videos[filename], media_type="multipart/x-mixed-replace; boundary=frame")
    return JSONResponse({"error": "Stream not found"}, status_code=404)

# Serve processed video for download
@app.get("/download/{filename}")
async def download_video(filename: str):
    processed_file_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
    if os.path.exists(processed_file_path):
        return FileResponse(processed_file_path, media_type="video/mp4", filename=f"processed_{filename}")
    return JSONResponse({"error": "File not found"}, status_code=404)

def generate_video_stream(file_path):
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    processed_path = os.path.join(PROCESSED_FOLDER, f"processed_{os.path.basename(file_path)}")
    out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=DEVICE)
        frame_with_boxes = draw_predictions(frame, results)

        out.write(frame_with_boxes)  # Save processed frame

        _, encoded_frame = cv2.imencode('.jpg', frame_with_boxes)
        frame_bytes = encoded_frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()




# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
