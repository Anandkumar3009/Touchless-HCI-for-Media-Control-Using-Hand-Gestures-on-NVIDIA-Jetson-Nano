import cv2
import mediapipe as mp
import csv
import os
import math
import time

USER_ID = input("Enter User ID (ex: U1): ").strip()
CSV_FILE = "gesture_dataset_cleaned.csv"

gesture_map = {
    ord('1'): "Palm",
    ord('2'): "Fist",
    ord('3'): "ThumbUp",
    ord('4'): "ThumbDown",
    ord('5'): "FastForward",
    ord('6'): "FastBackward"
}

gesture_label = "None"
recording = False
sample_count = 0
last_capture_time = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

# -------- Create CSV --------
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]

        header += ["label", "user"]
        writer.writerow(header)

with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = None

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist = hand_landmarks.landmark[0]
                middle_tip = hand_landmarks.landmark[12]

                # -------- Calculate Hand Size --------
                hand_size = math.sqrt(
                    (middle_tip.x - wrist.x) ** 2 +
                    (middle_tip.y - wrist.y) ** 2 +
                    (middle_tip.z - wrist.z) ** 2
                )

                if hand_size == 0:
                    continue

                landmarks = []

                # -------- Normalized Landmarks --------
                for lm in hand_landmarks.landmark:
                    landmarks.append((lm.x - wrist.x) / hand_size)
                    landmarks.append((lm.y - wrist.y) / hand_size)
                    landmarks.append((lm.z - wrist.z) / hand_size)

        # -------- Recording Logic --------
        if recording and landmarks is not None and gesture_label != "None":

            # Prevent duplicate frames
            if time.time() - last_capture_time > 0.15:

                writer.writerow(landmarks + [gesture_label, USER_ID])
                sample_count += 1
                last_capture_time = time.time()

        # -------- Display --------
        status = "Recording..." if recording else "Idle"

        cv2.putText(frame, f"User: {USER_ID}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, status, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(frame, f"Samples: {sample_count}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame,
            "1-6 Select Gesture | R Record | Q Quit",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("Improved Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in gesture_map:
            gesture_label = gesture_map[key]
            print("Gesture set:", gesture_label)

        elif key == ord('r'):
            recording = not recording
            print("Recording:", recording)

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
