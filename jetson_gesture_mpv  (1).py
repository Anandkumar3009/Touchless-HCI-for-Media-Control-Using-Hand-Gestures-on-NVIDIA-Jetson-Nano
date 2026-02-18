import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import joblib
import math
import subprocess
import time
import os
import socket
import json

# =========================
# MPV SETUP & IPC SOCKET
# =========================
mpv_socket = "/tmp/mpvsocket"
if os.path.exists(mpv_socket):
    os.remove(mpv_socket)

# Launch MPV with Hardware Acceleration and IPC Server
player = subprocess.Popen([
    "mpv",
    "--idle=yes",
    "--force-window=yes",
    "--keep-open=yes",
    "--loop-file=inf",
    "--hwdec=auto",                
    "--osd-level=3",               
    "--osd-duration=2000",         
    "--input-ipc-server=" + mpv_socket,
    "/home/nano/gesture_project/sample.mp4"
])

time.sleep(1) # Give MPV time to initialize the socket

# =========================
# LOAD MODEL & MEDIAPIPE
# =========================
# Ensure you have run your SVM training script first!
model = joblib.load("gesture_rf_model.pkl")
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# =========================
# REALSENSE CAMERA SETUP
# =========================
pipeline = rs.pipeline()
config = rs.config()
# Using 640x480 @ 30fps as it's the most stable for MediaPipe tracking
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# =========================
# HELPER FUNCTIONS
# =========================
def mpv_command(command_list):
    """Sends JSON commands to MPV via Unix Socket."""
    if not os.path.exists(mpv_socket): return
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(0.05)
        client.connect(mpv_socket)
        payload = json.dumps({"command": command_list}) + "\n"
        client.sendall(payload.encode())
        client.close()
    except:
        pass # Prevents script crash on socket saturation

# =========================
# MAIN EVALUATION LOOP
# =========================
last_action_time = 0
ACTION_DELAY = 0.5  # Time between repeat actions (like volume steps)
prev_time = 0

# Stabilization Variables
stable_fps = 0
stable_lat = 0
gesture_confirm_counter = 0
last_detected_gesture = "None"

print("System Active: Stabilized Metrics Overlay enabled.")

try:
    while True:
        loop_start = time.time()

        # 1. Get Frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Process Hand Landmarks
        results = hands.process(rgb)
        
        current_gesture = "None"
        confidence = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                middle_tip = hand_landmarks.landmark[12]
                hand_size = math.sqrt((middle_tip.x-wrist.x)**2 + (middle_tip.y-wrist.y)**2)
                
                if hand_size > 0:
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([
                            (lm.x - wrist.x)/hand_size, 
                            (lm.y - wrist.y)/hand_size, 
                            (lm.z - wrist.z)/hand_size
                        ])
                    
                    features = np.array(features).reshape(1, -1)
                    probs = model.predict_proba(features)[0]
                    gesture_index = np.argmax(probs)
                    current_gesture = model.classes_[gesture_index]
                    confidence = probs[gesture_index] * 100

        # 3. STABILIZATION & DEBOUNCING (Stops 'Broken Pipe')
        curr_t = time.time()
        
        # Require the same gesture for 2 consecutive frames to trigger action
        if current_gesture == last_detected_gesture and current_gesture != "None":
            gesture_confirm_counter += 1
        else:
            gesture_confirm_counter = 0
            last_detected_gesture = current_gesture

        # Only send command if gesture is stable and cooldown is over
        if gesture_confirm_counter >= 2 and (curr_t - last_action_time > ACTION_DELAY):
            if current_gesture == "Palm":
                mpv_command(["set_property", "pause", False])
            elif current_gesture == "Fist":
                mpv_command(["set_property", "pause", True])
            elif current_gesture == "FastForward":
                mpv_command(["seek", 10, "relative+exact"])
            elif current_gesture == "FastBackward":
                mpv_command(["seek", -10, "relative+exact"])
            elif current_gesture == "ThumbUp":
                mpv_command(["osd-msg", "add", "volume", 5])
            elif current_gesture == "ThumbDown":
                mpv_command(["osd-msg", "add", "volume", -5])
            
            last_action_time = curr_t

        # 4. METRICS & SMOOTH HUD
        raw_fps = 1 / (curr_t - prev_time) if prev_time != 0 else 0
        prev_time = curr_t
        raw_lat = (time.time() - loop_start) * 1000

        # Moving average filter (0.9 weight to old, 0.1 to new)
        stable_fps = (stable_fps * 0.9) + (raw_fps * 0.1)
        stable_lat = (stable_lat * 0.9) + (raw_lat * 0.1)

        # Draw Clean UI
        cv2.rectangle(frame, (0, 0), (230, 115), (0, 0, 0), -1) 
        cv2.putText(frame, f"FPS: {int(stable_fps)}", (10, 25), 1, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"LAT: {int(stable_lat)}ms", (10, 50), 1, 1.2, (0, 0, 255), 2)
        cv2.putText(frame, f"GEST: {current_gesture}", (10, 75), 1, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"CONF: {confidence:.0f}%", (10, 100), 1, 1.2, (255, 255, 0), 2)

        cv2.imshow("Gesture Control Hub", frame)

        if cv2.waitKey(1) & 0xFF == 27: # Press ESC to quit
            break

finally:
    pipeline.stop()
    player.terminate()
    cv2.destroyAllWindows()
    print("System Shutdown Cleanly.")
