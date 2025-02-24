import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Check CUDA availability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Paths
    DATA_CONFIG = r"config.yaml"  # Path to your dataset config
    PRETRAINED_MODEL = "yolov8n.pt"  # Choose from yolov8n.pt, yolov8s.pt, yolov8m.pt
    SAVE_PATH = r"runs/train/best.pt"  # Path to save the trained model

    # Load YOLO model
    model = YOLO(PRETRAINED_MODEL)

    # Train the model
    try:
        results = model.train(
            data=DATA_CONFIG,   # Dataset configuration file
            epochs=40,          # Number of training epochs
            batch=16,           # Increased batch size for better GPU utilization
            imgsz=640,          # Image size (keep this high for accuracy)
            device=DEVICE,      # Use GPU if available
            workers=4,          # Enable multiprocessing (set to 4 or 8 based on CPU cores)
            optimizer="AdamW",  # Use AdamW optimizer for better performance
            lr0=0.001,          # Initial learning rate
            patience=40,        # Early stopping if no improvement
            dropout=0.2,        # Adds dropout to prevent overfitting
            seed=42,            # Set random seed for reproducibility
            verbose=True,       # Print training progress
            cos_lr=True,        # Use cosine learning rate scheduler
            warmup_epochs=3,    # Warmup epochs for better convergence
            cache="disk",         # Cache dataset in RAM for faster training
        )

        # Save the trained model
        model.save(SAVE_PATH)
        print(f"🎯 Training complete! Model saved at: {SAVE_PATH}")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")