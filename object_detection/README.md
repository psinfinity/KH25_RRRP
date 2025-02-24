# Traffic Object Detection Project

## Overview

This project implements object detection for traffic analysis using YOLO segmentation. It detects and classifies multiple objects such as cars, persons, trucks, buses, and more while ignoring non-essential classes like drivable areas and lanes. The system supports both **training** a custom model and **detecting objects** in images/videos.

---

## Project Structure

```
📂 traffic_analysis/
├── 📂 dataset/               # Contains your dataset (images & labels)
├── 📂 input/                 # Input directory for images and videos to be processed
├── 📂 runs/                  # Stores output and trained models
│   ├── 📂 detection/         # Processed images/videos output by detect.py
│   ├── 📂 detect/train/      # Trained models saved after training
├── 📂 src/                   # Contains core scripts
│   ├── train.py             # Trains the model using config.yaml
│   ├── detect.py            # Runs inference on images/videos
│   ├── utils.py             # Utility functions for visualization
│   ├── convert_json_to_yolo.py  # Converts dataset from JSON to YOLO format
├── config.yaml               # Configuration file for dataset paths and class definitions
└── README.txt                # Project documentation
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/traffic_analysis.git
   cd traffic_analysis
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare dataset:** Place images and labels in the `dataset/` directory. Use `convert_json_to_yolo.py` if your annotations are in JSON format.

---

## Training the Model

Train a YOLO model using the dataset and `config.yaml` settings:

```bash
python src/train.py --config config.yaml
```

- The trained model is saved in `runs/detect/train/`
- The `config.yaml` file defines:
  - Dataset paths
  - Number of classes
  - Class names

---

## Running Object Detection

Run inference on images/videos stored in `input/`:

```bash
python src/detect.py --source input/image.jpg  # For images
python src/detect.py --source input/video.mp4  # For videos
```

- The output is saved in `runs/detection/`
- Uses `utils.py` for visualization (bounding boxes, masks, etc.)

---

## Utility Scripts

- **`convert_json_to_yolo.py`** → Converts BDD100K JSON annotations to YOLO format.
- **`utils.py`** → Handles drawing detections on images/videos.

---

## Notes

- Ensure `config.yaml` is properly configured before training.
- The project supports only **10 object classes**, excluding drivable areas and lanes.
- The detection output uses **specific colors**:
  - **Car** → 🟩 Green
  - **Person** → 🔵 Blue
  - **Truck & Bus** → 🟡 Yellow
  - **Other Objects** → 🌐 Cyan

---

## License

This project is open-source and free to use. Contributions are welcome!

Contact

Rishabh Raj

Email: [rshabhraj311@gmail.com](mailto\:rshabhraj311@gmail.com)

GitHub: riceu69
