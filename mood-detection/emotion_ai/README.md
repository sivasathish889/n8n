# AI-Based Children's Mood Analysis System

This is a production-ready, real-time AI application designed to detect multiple children's faces using a webcam and classify their facial expressions into one of seven emotional mood classes.

The system uses the **YOLO11** face detection model for highly accurate multi-face detection and a fine-tuned **EfficientNetV2-S** (via PyTorch & `timm`) for facial expression classification.

---

## Features
- **Real-Time Video Processing**: High frame rate capture and overlays using OpenCV.
- **Robust Face Detection**: Employs community-trained YOLO11 face detection model weights to reliably localize faces in varying conditions.
- **Multi-Face Support**: Crops, normalizes, and classifies multiple faces simultaneously in the same frame.
- **State-of-the-Art Classifier**: Employs EfficientNetV2-S pretrained on ImageNet and fine-tuned for facial expression categories.
- **Dynamic Color Overlays**: Bounding boxes are styled and color-coded dynamically based on the predicted mood (e.g., Yellow for Happy, Blue for Sad, Red for Angry).
- **Graceful Fallbacks**:
  - **Self-downloading YOLO weights**: Automatically downloads the face detector weights if they are missing locally.
  - **Pipeline self-verification**: Automatically generates synthetic dummy images if the training dataset folder is empty, letting developers test the training script instantly.
  - **Demo Inference**: If no trained checkpoint is found, the inference engine falls back to demo mode using ImageNet backbone features.

---

## Directory Structure
```
emotion_ai/
├── requirements.txt      # Project library dependencies
├── config.py            # Global hyperparameter, folder, and mapping configurations
├── dataset.py           # Custom PyTorch Dataset loaders and augmentations
├── model.py             # EfficientNetV2-S backbone and classifier head
├── utils.py             # Logging, file downloads, model checkpoints, and plotting
├── train.py             # Model training, validation, early stopping, and scheduler loop
├── inference.py         # Real-time webcam OpenCV application with YOLO face detector
├── README.md            # Setup and user guidelines (this file)
│
├── dataset/             # User training images (created automatically if missing)
│   ├── train/
│   │   ├── Happy/
│   │   ├── Sad/
│   │   └── ... (Angry, Disgust, Fear, Surprise, Neutral)
│   └── val/
│       ├── Happy/
│       ├── Sad/
│       └── ...
│
└── outputs/             # Generated checkpoints, training plots, and logs
    ├── best_model.pth   # Best saved model checkpoint
    ├── training.log     # Detailed logs of training sessions
    └── training_curves.png
```

---

## Installation & Setup

### 1. Requirements
Ensure you are using **Python 3.11** or newer. Install all required dependencies from the root directory:

```bash
pip install -r requirements.txt
```

*Note: For macOS users, Apple Silicon GPU acceleration (MPS) is supported automatically via PyTorch.*

### 2. Dataset Layout
To train on custom data or the FER2013 dataset, place images in the `dataset/` directory with the following naming structure:

```
dataset/
 ├── train/
 │     ├── Angry/
 │     ├── Disgust/
 │     ├── Fear/
 │     ├── Happy/
 │     ├── Neutral/
 │     ├── Sad/
 │     └── Surprise/
 └── val/
       ├── Angry/
       ...
```

*If no dataset folder is detected when running the training pipeline, the script will automatically generate a mock synthetic dataset to verify code functionality.*

---

## Running the Application

### 1. Training the Model
To start training the EfficientNetV2 classifier on your dataset:

```bash
python train.py
```

**What it does:**
1. Verifies if datasets are present (creates synthetic images if they are missing).
2. Initializes the `EfficientNetV2-S` model with ImageNet pre-trained weights.
3. Applies extensive augmentations (Random Resized Crop, Horizontal Flip, Rotations, Color Jitter) to training images.
4. Trains the model using **CrossEntropyLoss**, **AdamW** optimizer, and **ReduceLROnPlateau** learning rate scheduler.
5. Saves the best model weights dynamically based on validation accuracy to `outputs/best_model.pth`.
6. Triggers **Early Stopping** if the validation loss fails to improve for 7 consecutive epochs.
7. Saves loss and accuracy plots to `outputs/training_curves.png`.

---

### 2. Real-Time Camera Inference
To launch the webcam face emotion tracker:

```bash
python inference.py
```

**Options:**
- `--source`: Specify camera index or image/video file path. (e.g. `python inference.py --source 0` or `python inference.py --source demo.mp4`)
- `--weights`: Specify custom model weights file path (default: `outputs/best_model.pth`).
- `--conf`: Confidence threshold for face detection bounding box (default: `0.4`).

**Key Bindings:**
- Press **`Q`** to exit the camera feed window and shut down the system cleanly.

---

## Technical Details

### Mood Categories Configuration
The classification maps input faces to seven key states in standard FER2013 format:
1. `Angry` (Vibrant Red)
2. `Disgust` (Forrest Green)
3. `Fear` (Violet-Purple)
4. `Happy` (Vibrant Yellow)
5. `Sad` (Sky Blue)
6. `Surprise` (Vibrant Orange)
7. `Neutral` (Soft Gray)

### Transfer Learning Pipeline
In `model.py`, the `EmotionClassifier` loads an EfficientNetV2-S model and replaces the final classification layer.
To custom fine-tune:
1. Initialize the model with backbone layers frozen using `model.freeze_backbone()`.
2. Train for initial epochs to tune the classification head.
3. Unfreeze all parameters using `model.unfreeze_backbone()` and train with a lower learning rate ($10^{-5}$) for deep features alignment.
