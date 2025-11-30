from tensorflow.keras.models import load_model
import numpy as np
import cv2

IMG_SIZE = (224, 224)

# Class order we expose throughout the app.
CLASS_NAMES = ["non_violent", "violent", "comedic"]


class ViolenceDetectionModel:
    """Wrapper around the trained Keras model.

    NOTE: The original deployed model may be binary (violent vs non-violent)
    or multi-class. The Grad 699 version is expected to be a CNN+LSTM model
    producing three classes: non-violent, violent, comedic.

    This wrapper always returns a dictionary of probabilities over the
    classes defined in CLASS_NAMES so the rest of the system does not need
    to know the exact network architecture.

    This class also implements a *fallback* heuristic model when the
    serialized Keras model cannot be loaded (e.g., due to version
    incompatibilities). That fulfills the design document requirement of
    having a fallback mechanism if the model will not load.
    """

    def __init__(self, model_path: str):
        self.img_size = IMG_SIZE
        try:
            self.model = load_model(model_path)
            print(f"Loaded trained model from {model_path}")
        except Exception as e:
            # Fallback: simple heuristic model.
            print("WARNING: Could not load trained model. Using fallback "
                  "heuristic model instead.")
            print(f"Model load error: {e}")
            self.model = None

    def predict(self, frame):
        """Return (top_label, confidence, probs) for a single frame.

        - top_label: one of CLASS_NAMES
        - confidence: probability of that label (float in [0, 1])
        - probs: dict mapping class name -> probability
        """
        # Resize the frame to the required size
        frame_resized = cv2.resize(frame, self.img_size)
        frame_array = np.expand_dims(frame_resized, axis=0) / 255.0

        if self.model is None:
            # Heuristic fallback based on brightness.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_intensity = float(np.mean(gray)) / 255.0  # 0..1 (bright)

            # Simple 3-class distribution:
            # bright -> non-violent, medium -> comedic, dark -> violent.
            nv = mean_intensity
            c = max(0.0, 1.0 - abs(mean_intensity - 0.5) * 2.0)
            v = max(0.0, 1.0 - mean_intensity)
            raw = np.array([nv, v, c], dtype=float)
            if raw.sum() == 0:
                raw = np.array([1.0, 0.0, 0.0])
            probs_arr = raw / raw.sum()
        else:
            # Predict using the trained model.
            prediction = self.model.predict(frame_array, verbose=0)[0]
            prediction = np.array(prediction, dtype=float).ravel()

            if prediction.size == 1:
                # Binary output: interpret as P(violent)
                p_violent = float(prediction[0])
                p_non_violent = 1.0 - p_violent
                probs_arr = np.array([p_non_violent, p_violent, 0.0])
            elif prediction.size >= 3:
                # Assume first three entries correspond to
                # [non_violent, violent, comedic]
                probs_arr = prediction[:3]
                total = float(np.sum(probs_arr))
                if total <= 0:
                    probs_arr = np.array([1.0, 0.0, 0.0])
                else:
                    probs_arr = probs_arr / total
            else:
                # Fallback: try to derive a 3-class distribution from whatever
                # is available, defaulting to non-violent.
                p_violent = float(prediction[0])
                p_non_violent = 1.0 - p_violent
                probs_arr = np.array([p_non_violent, p_violent, 0.0])

        probs = {name: float(p) for name, p in zip(CLASS_NAMES, probs_arr)}
        top_index = int(np.argmax(probs_arr))
        top_label = CLASS_NAMES[top_index]
        confidence = float(probs_arr[top_index])

        print(
            f"Predicted: {top_label} | "
            f"NV={probs['non_violent']:.3f}, "
            f"V={probs['violent']:.3f}, "
            f"C={probs['comedic']:.3f}"
        )

        return top_label, confidence, probs


model = ViolenceDetectionModel('model/violent_non_violent_model.keras')
