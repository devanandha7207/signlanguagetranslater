import cv2
import mediapipe as mp
import numpy as np
import json
import os
from tensorflow import keras
import streamlit as st
import time

# -------------------
# Page config
# -------------------
st.set_page_config(page_title="Sign Language to Text Converter", layout="centered")
st.title("🤟 Sign Language to Text Converter")

# -------------------
# Absolute Paths
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sign_model.keras")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "data", "label_map.json")

# -------------------
# Load Model + Labels
# -------------------
@st.cache_resource
def load_model_and_labels():
    model = keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    labels = {v: k for k, v in label_map.items()}
    return model, labels

model, LABELS = load_model_and_labels()

# -------------------
# Sidebar
# -------------------
st.sidebar.title("Controls")

camera_on = st.sidebar.toggle("📷 Camera ON / OFF", value=False)

if "word" not in st.session_state:
    st.session_state.word = ""

if st.sidebar.button("🧹 Clear word"):
    st.session_state.word = ""


# -------------------
# MediaPipe Setup
# -------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------
# Webcam
# -------------------
last_added_time = 0
last_added_label = None
ADD_DELAY = 1.2  # seconds 



frame_window = st.image([])
word_placeholder = st.markdown("### Word: ")

if camera_on:
    cap = cv2.VideoCapture(0)
    last_added_time = 0
    ADD_DELAY = 1.0

    while camera_on:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not available")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = []

            hands_sorted = sorted(
                result.multi_hand_landmarks,
                key=lambda h: sum(lm.x for lm in h.landmark) / 21
            )

            for hand in hands_sorted[:2]:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            while len(landmarks) < 126:
                landmarks.extend([0.0, 0.0, 0.0])

            if len(landmarks) == 126:
                X_input = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                prediction = model.predict(X_input, verbose=0)

                class_id = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                label = LABELS[class_id]

                cv2.putText(
                    frame,
                    f"{label} ({confidence * 100:.1f}%)",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 255, 0),
                    3
                )

                current_time = time.time()

                if confidence >= 0.90:
                    if (
                        label != last_added_label
                        and (current_time - last_added_time) > ADD_DELAY
                    ):
                        st.session_state.word += label
                        last_added_label = label
                        last_added_time = current_time


        frame_window.image(frame, channels="BGR")
        word_placeholder.markdown(f"### Word: **{st.session_state.word}**")

    cap.release()
else:
    st.info("📷 Camera is OFF")


hands.close()
cv2.destroyAllWindows()
