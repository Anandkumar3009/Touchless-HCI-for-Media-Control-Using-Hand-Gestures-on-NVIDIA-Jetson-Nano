#  Real-Time Hand Gesture Controlled Media Player on Jetson Nano

An **Edge-AI based real-time gesture recognition system** enabling fully offline, touchless media control on the **NVIDIA Jetson Nano (ARM Cortex-A57 + Maxwell GPU)**.

The system performs:

- **GPU-accelerated hand landmark extraction (MediaPipe)**
- **Scale-invariant feature normalization**
- **Lightweight MLP-based gesture classification**
- **Non-blocking IPC communication with MPV**

---

#  Project Overview

This project implements a complete **low-latency Edge AI pipeline optimized for ARM-based SoCs**:

Camera Input
→ MediaPipe Landmark Extraction (GPU)
→ Scale-Invariant Feature Normalization
→ Lightweight MLP Classification
→ Temporal Stabilization (Debouncing)
→ Unix Socket IPC
→ MPV Media Control


- Model trained **offline on laptop**
- Deployed on **Jetson Nano for real-time inference**
- **Fully offline system (no internet required)**

---

#  Performance Metrics

| **Metric** | **Achieved** |
|------------|-------------|
| **Gesture Accuracy** | **98%** |
| **End-to-End Latency** | **~36 ms** |
| **FPS** | **25–28 FPS** |
| **Model Inference Time** | **5.86 ms** |
| **Power Consumption** | **~4.8W** |
| **Latency Reduction (RF → MLP)** | **~94%** |

>  The system exceeds the **<200 ms latency requirement by more than 5×**.

---

# System Architecture

Intel RealSense L515
↓
MediaPipe Hand Landmark Detection (21 keypoints) – GPU Accelerated
↓
Wrist-Relative & Scale Normalization
↓
Feature Vector (63 values)
↓
MLP Classifier (64 → 32 → 6)
↓
Temporal Filtering (n = 2 frames)
↓
MPV IPC Socket Command
↓
Media Control Execution


---

#  ARM SoC Optimization Highlights

This implementation is specifically optimized for **ARM Cortex-A57 architecture**:

- **Dense matrix operations (MLP) are SIMD-friendly (ARM NEON optimized)**
- **Random Forest removed due to branch misprediction overhead**
- **DVFS disabled for deterministic timing (MAXN mode)**
- CPU locked at **~1.43 GHz**
- GPU locked at **921 MHz**
- **Power-efficient edge deployment (~4.8W steady state)**
- **Zero swap usage**

---

# Key Features

- Real-time gesture detection
- Fully offline edge deployment
- Lightweight MLP inference
- GPU-accelerated landmark extraction
- Temporal stabilization & debouncing
- FPS and latency overlay
- Hardware-accelerated media playback
- IPC-based MPV control (robust & non-blocking)
- Near and far distance validation

---

#  Supported Gestures

| **Gesture** | **Action** |
|-------------|-----------|
| Palm | Play |
| Fist | Pause |
| ThumbUp | Volume + |
| ThumbDown | Volume − |
| FastForward | Seek +10 seconds |
| FastBackward | Seek −10 seconds |

---

#  Dataset Collection

Dataset collected manually using **MediaPipe hand landmarks**.

Each sample contains:

- 21 hand landmarks
- (x, y, z) coordinates
- Wrist-relative normalization
- Scale normalization
- Gesture label
- User ID

**Total Features per Sample:**

21 landmarks × 3 coordinates = 63 features


Dataset file:

gesture_dataset_cleaned.csv


---

#  Model Training

Training performed offline on PC for computational efficiency.

## Models Evaluated

- Random Forest
- Linear SVM
- **Multi-Layer Perceptron (Selected)**

---

##  Final Selected Model: Optimized MLP

### Architecture
- **Input:** 63 features
- **Hidden Layer 1:** 64 neurons (ReLU)
- **Hidden Layer 2:** 32 neurons (ReLU)
- **Output:** 6-class Softmax

### Training Pipeline

1. Load dataset  
2. Clean and preprocess data  
3. Train-test split  
4. Feature scaling (for MLP)  
5. Model training  
6. Accuracy evaluation  
7. Save model using joblib  

Final model file:

gesture_mlp_model.pkl


---

#  Installation (Jetson Nano)

## Environment

- JetPack 4.6.6
- CUDA 10.2
- Python 3.6
- MediaPipe 0.8.5 (CUDA build)

---
 Architecture Diagram
<img width="1536" height="1024" alt="aRCHITECTURE DIAGTAM" src="https://github.com/user-attachments/assets/391a5b46-9d1c-4438-b71d-a2a536fdc77f" />

## Install Dependencies

```bash
pip install opencv-python
pip install scikit-learn
pip install joblib
pip install mediapipe==0.8.5
Run the System
python jetson_gesture_mpv.py
📏 Real-Time Behavior
Distance	Range
Near	0.5–1m
Far	1.5–2.5m
Performance remains stable due to scale-invariant normalization.

 Why This Project Matters
This project demonstrates that:

High-accuracy gesture recognition can run on low-power ARM SoCs

Lightweight ML models outperform ensemble methods in embedded systems

Deterministic real-time execution is achievable under strict latency constraints

Edge AI pipelines can be production-ready without cloud dependency

 Potential Applications
Smart kiosks

Automotive infotainment systems

Touchless HCI interfaces

Embedded AI product deployment

Project Summary

A fully optimized ARM-aware Edge AI deployment demonstrating:

98% accuracy

36 ms latency

25–28 FPS real-time performance

4.8W steady-state power envelope

94% latency reduction vs Random Forest

A production-grade embedded AI pipeline built for real-world deployment.
