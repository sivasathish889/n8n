#!/usr/bin/env python3
import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import timm

class Config:
    CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    NUM_CLASSES = len(CLASSES)
    
    # Base directory of the script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
    
    YOLO_FACE_MODEL = os.path.join(BASE_DIR, 'yolo11n-face.pt')
    YOLO_FACE_HF_REPO = "AdamCodd/YOLOv11n-face-detection"
    YOLO_FACE_HF_FILE = "model.pt"
    
    MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'
    IMAGE_SIZE = 224
    
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(EmotionClassifier, self).__init__()
        self.backbone = timm.create_model(
            Config.MODEL_NAME, 
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'fc'):
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise AttributeError("Unable to locate model classification layer in backbone.")

    def forward(self, x):
        return self.backbone(x)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def ensure_yolo_face_model():
    model_path = Config.YOLO_FACE_MODEL
    if not os.path.exists(model_path):
        print(f"YOLO Face detection model not found at {model_path}.")
        print("Downloading YOLO11 Face detection model from Hugging Face...")
        from huggingface_hub import hf_hub_download
        import shutil
        downloaded_path = hf_hub_download(repo_id=Config.YOLO_FACE_HF_REPO, filename=Config.YOLO_FACE_HF_FILE)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.copy(downloaded_path, model_path)
        print(f"Successfully downloaded and copied model to: {model_path}")
    return model_path

def get_transforms():
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])
    return val_transform

# BGR color mapping corresponding to emotions
EMOTION_COLORS_BGR = {
    'Happy': (0, 242, 255),       # BGR Yellow
    'Sad': (235, 131, 0),        # BGR Sky Blue
    'Angry': (46, 20, 240),      # BGR Crimson Red
    'Fear': (180, 50, 130),      # BGR Violet-Purple
    'Surprise': (0, 120, 255),    # BGR Vibrant Orange
    'Disgust': (60, 180, 60),     # BGR Forrest Green
    'Neutral': (200, 200, 200)    # BGR Soft Gray
}

def main():
    parser = argparse.ArgumentParser(description="Real-Time Face Emotion Tracking application")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or file path to video")
    parser.add_argument("--weights", type=str, default=Config.BEST_MODEL_PATH, help="Path to the PyTorch classification weights")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO Face detector confidence threshold")
    args = parser.parse_args()

    # Determine input source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    device = get_device()
    print(f"Using device: {device}")

    # Ensure YOLO Face weights are downloaded
    try:
        yolo_weights_path = ensure_yolo_face_model()
    except Exception as e:
        print(f"Failed to ensure YOLO Face detector model: {e}")
        sys.exit(1)

    print("Loading YOLO Face Detector...")
    face_detector = YOLO(yolo_weights_path)

    # Load Emotion Classifier
    print("Loading Emotion Classifier...")
    classifier = EmotionClassifier(num_classes=Config.NUM_CLASSES, pretrained=True)
    if os.path.exists(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        if 'model_state_dict' in state_dict:
            classifier.load_state_dict(state_dict['model_state_dict'])
        else:
            classifier.load_state_dict(state_dict)
        print(f"Trained weights loaded successfully from: {args.weights}")
    else:
        print(f"WARNING: Weights file not found at: {args.weights}")
        print("Using pretrained ImageNet feature extraction fallback (Demo Mode).")
        
    classifier = classifier.to(device)
    classifier.eval()

    val_transform = get_transforms()

    print(f"Opening video capture from source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        sys.exit(1)

    # Set OpenCV properties for low latency on webcams if applicable
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("--------------------------------------------------")
    print("Real-Time Emotion Tracker Running.")
    print("Press 'q' key in the video window to quit.")
    print("--------------------------------------------------")

    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing stream or read error occurred.")
            break

        # Convert to RGB for YOLOv11
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detector(img_rgb, conf=args.conf, verbose=False)
        h, w, _ = frame.shape

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Check bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                face_bgr = frame[y1:y2, x1:x2]
                if face_bgr.size == 0:
                    continue
                
                try:
                    # Convert crop to RGB for PyTorch model
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_tensor = val_transform(face_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits = classifier(face_tensor)
                        probabilities = torch.softmax(logits, dim=1)[0]
                        max_prob, pred_idx = torch.max(probabilities, dim=0)
                        
                        emotion_label = Config.CLASSES[pred_idx.item()]
                        confidence = max_prob.item() * 100.0
                    
                    # Draw BGR rectangle and label
                    color = EMOTION_COLORS_BGR.get(emotion_label, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Text box background
                    label_text = f"{emotion_label}: {confidence:.1f}%"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    rect_y1 = max(0, y1 - text_h - 10)
                    rect_y2 = y1 if y1 - text_h - 10 >= 0 else y1 + text_h + 10
                    
                    cv2.rectangle(frame, (x1, rect_y1), (x1 + text_w + 10, rect_y2), color, -1)
                    # Text color is black if the label color is bright yellow, otherwise white
                    text_color = (0, 0, 0) if emotion_label == 'Happy' else (255, 255, 255)
                    cv2.putText(frame, label_text, (x1 + 5, rect_y2 - 5 if y1 - text_h - 10 >= 0 else rect_y2 - 7),
                                font, font_scale, text_color, thickness, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error processing detected face: {e}")

        # Compute and overlay FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Real-Time Face Emotion Tracker", frame)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Clean shutdown successful.")

if __name__ == "__main__":
    main()
