import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------
# Absolute Paths
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------
# Load Data
# -------------------
print("\n[INFO] Loading dataset...")
X = np.load(X_PATH)
y = np.load(Y_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

num_classes = len(label_map)

print("[INFO] Samples:", len(X))
print("[INFO] Classes:", num_classes)

# -------------------
# Prepare Data
# -------------------
y_cat = keras.utils.to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

# -------------------
# Model
# -------------------
model = keras.Sequential([
    layers.Input(shape=(126,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------
# Callbacks
# -------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -------------------
# Train
# -------------------
print("\n[INFO] Training model...\n")
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

# -------------------
# Save
# -------------------
print("\n[INFO] Saving model to:", MODEL_PATH)
model.save(MODEL_PATH)

# -------------------
# Evaluate
# -------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("\n[RESULT] Test Accuracy:", round(acc * 100, 2), "%\n")
