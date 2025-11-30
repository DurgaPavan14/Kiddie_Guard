import os
from glob import glob
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    TimeDistributed,
    LSTM,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import cv2


# ------------------------- CONFIGURATION -------------------------

DATA_ROOT = "model"  # root directory containing class subfolders
CLASS_TO_INDEX = {
    "non_violent": 0,
    "violent": 1,
    # We keep an index for comedic for future extension; it will not be trained
    # unless you add a "comedic" folder with frames.
    "comedic": 2,
}

# Only use the classes that actually have data right now
TRAINED_CLASSES = ["non_violent", "violent"]

IMG_SIZE = (112, 112)  # smaller than inference size for speed
SEQUENCE_LENGTH = 16   # number of frames per sequence
BATCH_SIZE = 8
EPOCHS = 20
VAL_SPLIT = 0.2
RANDOM_SEED = 42
MODEL_OUTPUT_PATH = os.path.join(DATA_ROOT, "violent_non_violent_model.keras")


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ------------------------- DATA LOADING -------------------------


def load_frame_paths() -> List[Tuple[str, int]]:
    """Collect (frame_path, class_index) pairs from the dataset folders.

    Expected layout:
        model/violent/*.jpg|png
        model/non_violent/*.jpg|png
    """

    samples: List[Tuple[str, int]] = []
    for cls_name in TRAINED_CLASSES:
        cls_dir = os.path.join(DATA_ROOT, cls_name)
        pattern_jpg = os.path.join(cls_dir, "*.jpg")
        pattern_png = os.path.join(cls_dir, "*.png")
        paths = glob(pattern_jpg) + glob(pattern_png)
        label_idx = CLASS_TO_INDEX[cls_name]
        for p in paths:
            samples.append((p, label_idx))

    if not samples:
        raise RuntimeError(
            "No frame images found. Expected images under model/violent and model/non_violent."
        )

    # Shuffle
    np.random.shuffle(samples)
    return samples


def make_sequences(samples: List[Tuple[str, int]], sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Group consecutive frame paths into fixed-length sequences.

    This assumes frames in each class folder roughly belong to many short clips.
    For simplicity, we just take sequences of `sequence_length` contiguous frames
    from the same class folder.
    """

    sequences = []
    labels = []

    # Group by class
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append(path)

    for label, paths in by_class.items():
        # Sort for a stable temporal order
        paths = sorted(paths)
        # Create sliding windows of length sequence_length
        if len(paths) < sequence_length:
            continue
        for start in range(0, len(paths) - sequence_length + 1, sequence_length):
            seq_paths = paths[start : start + sequence_length]
            seq_frames = []
            for fp in seq_paths:
                img = cv2.imread(fp)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype("float32") / 255.0
                seq_frames.append(img)

            if len(seq_frames) == sequence_length:
                sequences.append(np.stack(seq_frames, axis=0))  # (T, H, W, C)
                labels.append(label)

    if not sequences:
        raise RuntimeError(
            "Not enough contiguous frames to form sequences. "
            "You may need at least SEQUENCE_LENGTH frames per class."
        )

    X = np.stack(sequences, axis=0)
    y = np.array(labels, dtype="int32")
    return X, y


# ------------------------- MODEL DEFINITION -------------------------


def build_cnn_lstm_model(
    sequence_length: int,
    img_size: Tuple[int, int],
    num_classes: int,
) -> Model:
    """Build a simple CNN+LSTM model for sequence classification."""

    h, w = img_size
    inputs = Input(shape=(sequence_length, h, w, 3), name="frames")

    # Per-frame CNN feature extractor
    td = TimeDistributed(
        Conv2D(32, (3, 3), activation="relu", padding="same"), name="td_conv1"
    )(inputs)
    td = TimeDistributed(MaxPooling2D((2, 2)), name="td_pool1")(td)

    td = TimeDistributed(
        Conv2D(64, (3, 3), activation="relu", padding="same"), name="td_conv2"
    )(td)
    td = TimeDistributed(MaxPooling2D((2, 2)), name="td_pool2")(td)

    td = TimeDistributed(Flatten(), name="td_flatten")(td)

    # Temporal modeling
    x = LSTM(128, return_sequences=False, name="lstm1")(td)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="kiddie_guard_cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------- TRAINING ENTRYPOINT -------------------------


def main():
    print("[Kiddie Guard] Loading frame paths...")
    samples = load_frame_paths()
    print(f"Total individual frames: {len(samples)}")

    print("[Kiddie Guard] Building sequences...")
    X, y = make_sequences(samples, SEQUENCE_LENGTH)
    print(f"Total sequences: {X.shape[0]} | sequence shape: {X.shape[1:]}")

    # Map labels to full 3-class one-hot, even though only 2 classes are present
    num_classes = len(CLASS_TO_INDEX)
    y_categorical = to_categorical(y, num_classes=num_classes)

    # Train/validation split
    num_samples = X.shape[0]
    val_size = int(num_samples * VAL_SPLIT)
    X_val = X[:val_size]
    y_val = y_categorical[:val_size]
    X_train = X[val_size:]
    y_train = y_categorical[val_size:]

    print(
        f"Train sequences: {X_train.shape[0]}, Val sequences: {X_val.shape[0]} (split {VAL_SPLIT:.0%})"
    )

    model = build_cnn_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        img_size=IMG_SIZE,
        num_classes=num_classes,
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_OUTPUT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("[Kiddie Guard] Training CNN+LSTM model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("[Kiddie Guard] Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc * 100:.2f}%")

    if not os.path.exists(MODEL_OUTPUT_PATH):
        # If ModelCheckpoint didn't save (e.g., training aborted early), save manually
        model.save(MODEL_OUTPUT_PATH)
        print(f"[Kiddie Guard] Model saved to {MODEL_OUTPUT_PATH}")
    else:
        print(f"[Kiddie Guard] Best model already saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
