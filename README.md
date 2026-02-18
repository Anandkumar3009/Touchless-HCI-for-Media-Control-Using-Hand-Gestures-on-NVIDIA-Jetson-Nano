# Real-Time Hand Gesture Controlled Media Player on Jetson Nano

An edge-AI based real-time gesture recognition system that enables touchless media control using NVIDIA Jetson Nano.

The system detects hand landmarks using MediaPipe, extracts normalized features, classifies gestures using a trained ML model (Random Forest / MLP), and controls media playback via MPV using IPC socket communication.

---

# Table of Contents

- Project Overview
- System Architecture
- Features
- Gesture Mapping
- Dataset Collection
- Model Training
---

# Project Overview

This project demonstrates a complete edge AI pipeline:

Camera Input → MediaPipe Landmark Extraction → Feature Normalization → ML Classification → Media Player Control

The model is trained offline on a laptop and deployed on Jetson Nano for real-time inference.

The system operates fully offline and does not require internet connectivity.

---

# System Architecture
Intel RealSense Camera
↓
MediaPipe Hand Landmark Detection (21 keypoints)
↓
Wrist-relative Normalization
↓
Feature Vector (63 values)
↓
ML Model (Random Forest / MLP)
↓
Gesture Prediction
↓
MPV IPC Socket Command
↓
Media Control


---

# Key Features

- Real-time gesture detection
- Edge deployment (Jetson Nano)
- Lightweight ML inference
- Gesture stabilization & debouncing
- FPS and latency overlay
- Hardware-accelerated media playback
- IPC-based MPV control (robust & efficient)

---

# Supported Gestures

| Gesture        | Action              |
|---------------|--------------------|
| Palm          | Play               |
| Fist          | Pause              |
| ThumbUp       | Volume +           |
| ThumbDown     | Volume −           |
| FastForward   | Seek +10 seconds   |
| FastBackward  | Seek −10 seconds   |

---

# Dataset Collection

Dataset is collected manually using MediaPipe hand landmarks.

Each sample contains:

- 21 hand landmarks
- (x, y, z) coordinates
- Wrist-relative normalization
- Scale normalization using hand size
- Gesture label
- User ID

Total Features per sample:
21 landmarks × 3 coordinates = 63 features

Dataset stored as: gesture_dataset_cleaned.csv


---

# Model Training

Training is performed offline on PC for computational efficiency.

### Models Used:

- Random Forest Classifier
- Multi-Layer Perceptron (MLP)

### Training Pipeline:

1. Load dataset
2. Remove unnecessary columns
3. Split into train/test sets
4. Apply scaling (for MLP)
5. Train model
6. Evaluate accuracy
7. Save model using joblib


<img width="1536" height="1024" alt="aRCHITECTURE DIAGTAM" src="https://github.com/user-attachments/assets/391a5b46-9d1c-4438-b71d-a2a536fdc77f" />
