import cv2
import mediapipe as mp
import os
import numpy as np
import json
from tqdm import tqdm

# -------------------
# Absolute Paths (OS Safe)
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

OUTPUT_X = os.path.join(DATA_DIR, "X.npy")
OUTPUT_Y = os.path.join(DATA_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")

os.makedirs(DATA_DIR, exist_ok=True)

# -------------------
# MediaPipe Setup
# -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6
)

# -------------------
# Storage
# -------------------
X = []
y = []
label_map = {}
label_id = 0

print("\n[INFO] Starting two-hand landmark extraction...\n")

for label in sorted(os.listdir(DATASET_PATH)):
    class_dir = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(class_dir):
        continue

    if label not in label_map:
        label_map[label] = label_id
        label_id += 1

    for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {label}"):
        img_path = os.path.join(class_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = []

            # Sort hands left -> right for consistency
            hands_sorted = sorted(
                result.multi_hand_landmarks,
                key=lambda h: sum(lm.x for lm in h.landmark) / 21
            )

            for hand in hands_sorted[:2]:
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            # Pad with zeros if only one hand detected
            while len(landmarks) < 126:
                landmarks.extend([0.0, 0.0, 0.0])

            if len(landmarks) == 126:
                X.append(landmarks)
                y.append(label_map[label])

hands.close()

# -------------------
# Save
# -------------------
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f, indent=4)

print("\n[INFO] Extraction complete")
print("[INFO] Samples:", len(X))
print("[INFO] Classes:", len(label_map))
print("[INFO] Data saved in /data\n")
