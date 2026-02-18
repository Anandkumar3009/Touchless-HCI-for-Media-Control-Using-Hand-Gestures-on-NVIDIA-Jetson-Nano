# Real-Time Hand Gesture Controlled Media Player (Jetson Nano)

A real-time touchless media control system using hand gesture recognition deployed on NVIDIA Jetson Nano.

The system detects hand landmarks using MediaPipe, classifies gestures using a trained ML model (Random Forest / MLP), and controls media playback via MPV player.

---

## Project Overview

This project implements an end-to-end edge AI pipeline:

Camera Input → MediaPipe → Feature Extraction → ML Model → MPV Media Control

The model is trained offline on a laptop and deployed on Jetson Nano for real-time inference.

---

## Features

- Real-time hand gesture recognition
- Random Forest / MLP classifier
- Edge deployment on Jetson Nano
- MPV media control via IPC socket
- Gesture stabilization & debouncing
- FPS and latency monitoring overlay
- Hardware acceleration enabled

---

## Supported Gestures

| Gesture        | Action              |
|---------------|--------------------|
| Palm          | Play               |
| Fist          | Pause              |
| ThumbUp       | Volume +           |
| ThumbDown     | Volume −           |
| FastForward   | Seek +10 seconds   |
| FastBackward  | Seek −10 seconds   |

---

## Technologies Used

- Python 3
- OpenCV
- MediaPipe
- Scikit-learn
- Joblib
- MPV Media Player
- Intel RealSense Camera
- NVIDIA Jetson Nano

---

## 📂 Project Structure

